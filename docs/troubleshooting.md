# Troubleshooting

Common failure modes, what they look like, and how to fix them. See
[operations.md](operations.md) for the maintenance CLI these fixes use.

## `hermes memory status` shows the plugin unavailable / falls back to built-in memory

Usually one of:

1. **The schema was never applied.** `hermes-pgvector migrate` was never
   run against this database. `initialize()` catches
   `MemoryStore.SchemaNotApplied` and logs `pgvector schema not applied`;
   the provider stays unhealthy (`_healthy = False`), and every hook
   becomes a fail-soft no-op. Run `hermes-pgvector migrate --admin-dsn
   "..."`.
2. **The discovery shim resolves to the wrong package, or nothing.** If you
   installed via `pip` and use Option 1 (the shim), re-run `hermes-pgvector
   install` — it verifies the shim actually imports in a fresh subprocess
   and warns if it can't, or if another package is shadowing this one
   (`import hermes_pgvector` resolving somewhere unexpected).
3. **The DSN is wrong, or the runtime role lacks a grant** — see the next
   section.

## "permission denied for table memory_entries" (or `conversations`)

The runtime role hasn't been granted DML. Migration `004` (and `002` for
the attribution tables) grants this automatically, resolved from the
`hermes_pgvector.runtime_role` session GUC (default `hermes`) — but if your
runtime role has a different name and you ran `migrate` without
`--runtime-role NAME`, the grant block skipped itself with a `NOTICE`
instead of granting the wrong role. Re-run:

```bash
hermes-pgvector migrate --admin-dsn "..." --runtime-role myagentrole
```

or grant manually as the admin role:

```sql
GRANT SELECT, INSERT, UPDATE, DELETE ON memory_entries, conversations TO myagentrole;
GRANT USAGE, SELECT ON SEQUENCE memory_entries_id_seq, conversations_id_seq TO myagentrole;
```

## "expected N dims (embed_dim), got M" / dimension mismatch

The embed endpoint's model returned a vector of a different length than
`embed_dim` (default `768`, matching migration `001`'s `vector(768)`
columns and the reference `nomic-embed-text` model). This is deliberately
**not** masked or coerced — a silently-truncated or zero-padded vector
would corrupt every future similarity comparison for that row. Either:

- point `embed_model` back at a model that returns `embed_dim`-length
  vectors, or
- follow the column-migration procedure to change `embed_dim` and the
  `vector(N)` columns together — see "Changing the embedding dimension" in
  [upgrading.md](upgrading.md#changing-the-embedding-dimension).

`hermes-pgvector backfill` (not `--dry-run`, and not `stats`, which never
contacts the endpoint) probes the endpoint's dimension *before* touching any
row and aborts the whole run on a mismatch -- `error: ...` on stderr, exit
`1` -- so this never partially corrupts a table.

## Embed endpoint down: rows land text-only, recall degrades

By design (invariant: fail-soft everywhere). A failed embed on the agent
thread (`prefetch`, `recall_memory`, `recall_conversation`) degrades to
full-text-only recall (if `hybrid_search: true`) or an empty result
(`hybrid_search: false`); a failed embed on the background writer stores
the row with `embedding = NULL` instead of dropping the write. The first
failure each session logs a WARNING (`pgvector embed failed (degrading to
text-only)`); later ones in the same session drop to DEBUG.

**Recovery:** once the endpoint is back, run `hermes-pgvector backfill` to
re-embed every `NULL`-embedding row. It is idempotent and safe to run on a
timer — see [operations.md](operations.md#backfill) for exit codes and
alerting. Rows with genuinely empty content are reported separately as
`unembeddable` and will never backfill (nothing to embed) — this is
expected, not a bug.

## Writer queue full: writes are being dropped

`write_queue_maxsize` (default `256`) bounds the async writer's queue. A
full queue drops the *newest* write and logs one WARNING per session
(`pgvector writer queue full (maxsize=N); dropping writes`) — not a retry
storm, by design (Honcho's failure mode this plugin exists to avoid). If
this fires regularly:

- the embed endpoint is likely too slow relative to write volume — check
  `embed_write_timeout` and the endpoint's actual p95 latency (see
  [scaling.md](scaling.md#embed-latency-vs-prefetch_budget));
- or raise `write_queue_maxsize` if the bursts are short and memory
  headroom allows buffering more in-flight writes.

Dropped writes are gone — there is no queue-overflow recovery path (the
built-in `memory` tool's own file already has the data; only the pgvector
*mirror* lost that write). Backfill cannot help here, because backfill
repairs missing *embeddings*, not missing *rows*.

## `migrate` fails with `'ascii' codec can't encode character`

Your database was created with `ENCODING 'SQL_ASCII'` and an older version
of this package sent migration file text through Python's `ascii` codec on
send. Fixed as of v0.6.0 (L10): migration files are read and executed as
raw UTF-8 bytes, and every non-ASCII character was removed from migration
comments. Upgrade the package and re-run `migrate` — this class of failure
should not recur on any current migration file, since none contain
non-ASCII characters going forward.

## `add()` fails with "there is no unique or exclusion constraint matching the ON CONFLICT specification"

A pre-0.6.0 version of this package is still running against a database
where migration `005` has already been applied. `005` drops
`memory_entries_unique` (the raw-text unique constraint), and the old
code's `ON CONFLICT (agent_identity, target, content) DO NOTHING` names
that exact constraint as its conflict target — once it's gone, every
`add()` on the old code raises this error.

**Fix:** upgrade the `hermes-memory-pgvector` package on this host to
`>= 0.6.0` (its `add()` uses a bare `ON CONFLICT DO NOTHING` with no named
target, satisfied by either the old or the new index). See
[upgrading.md](upgrading.md) for the full ordering — this is exactly the
failure mode that ordering exists to prevent: **upgrade every writing host
first, migrate second.**

## A `memory replace`/`remove` hit the wrong row

Should not happen as of v0.6.0 (H2): when the host supplies
`metadata["previous_content"]` (upstream's exact prior entry), `replace()`
/ `remove()` match it exactly instead of a caller-hint `old_text` substring
`LIKE`. If you're still seeing this, confirm the host (hermes-agent) is new
enough to send `previous_content` — the conformance suite
(`conformance/test_notify_memory_tool_write.py`) checks this contract
against the pinned upstream ref.
