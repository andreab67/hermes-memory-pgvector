# Operations

Day-to-day operator tasks: applying migrations, running the maintenance CLI,
and what to alert on. See [configuration.md](configuration.md) for every
config key and [troubleshooting.md](troubleshooting.md) for failure modes.

Every command below resolves connection + embed settings CLI flag > `--config
<config.yaml>` > built-in default. Destructive commands (`prune`, `cleanup`,
`remap`) default to dry-run; pass `--execute` to actually mutate. Every
mutating run is recorded in `memory_maintenance_log`.

## migrate

Applies every file in `hermes_pgvector/migrations/` in lexical order, as DB
admin (needs `CREATE EXTENSION vector` / `CREATE TABLE` / `GRANT`
privileges):

```bash
hermes-pgvector migrate --admin-dsn \
    "dbname=hermes_memory user=postgres host=/var/run/postgresql"
```

Idempotent — every migration uses `IF NOT EXISTS` / `IF EXISTS`, so
re-running on an already-migrated database is a no-op except for whatever is
genuinely new.

**Runtime role.** Migrations `002`, `004`, and `005` grant the runtime role
DML on the tables they create or touch. The role defaults to `hermes`; if
yours is different, pass it explicitly so the grants land on the right role
instead of silently no-op'ing with a `NOTICE`:

```bash
hermes-pgvector migrate --admin-dsn "..." --runtime-role myagentrole
```

Applying a single migration by hand with `psql` gets the same effect via
`PGOPTIONS`, which sets the same session GUC the CLI sets internally:

```bash
PGOPTIONS='-c hermes_pgvector.runtime_role=myagentrole' \
  psql -d hermes_memory -f hermes_pgvector/migrations/002_agent_attribution.sql
```

If the named role does not exist yet at migration time, the affected
`GRANT`s are skipped with a `NOTICE` (never a failed migration) — grant
manually afterward, or re-run `migrate` once the role exists.

**Upgrade ordering (0.6.0 and later).** Upgrade the `hermes-memory-pgvector`
package on every host that writes to this database *before* running
`migrate` — see [upgrading.md](upgrading.md) for why this order matters for
the `005` migration specifically.

## backfill

Re-embeds rows whose `embedding` is `NULL` — left behind by an embed-endpoint
outage, a dimension change, or a bulk import that ran before the endpoint was
reachable:

```bash
hermes-pgvector backfill --config $HERMES_HOME/config.yaml
hermes-pgvector backfill --dry-run          # count only, embeds nothing
```

Idempotent and resumable: it walks each table by id in one pass, so a row
that fails is retried on the *next* run, not re-fetched forever within the
same run. A dimension probe runs first and aborts the whole run (never
touching a row) if the endpoint's vectors don't match `embed_dim`.

**Exit codes** (script/timer-friendly):

| Code | Meaning |
|---|---|
| `0` | Done — `remaining: 0` on every table, no table aborted. |
| `2` | Embed endpoint unavailable, or a table's pass aborted after `max_consecutive_failures` (default 20) consecutive row failures. |
| `3` | Rows still remain NULL or failed this run, but no abort — usually just means "more than one run's worth of backlog"; re-run. |

**Alert on:** exit code `2` (endpoint down or flapping — page), and exit
code `3` persisting across more than one scheduled run (backlog not
shrinking — investigate before it does). A `note` field in the JSON output
(`"aborted-consecutive-failures"` or `"embed-unavailable"`) names which.
`unembeddable` counts rows with empty/whitespace content that will never
backfill (harmless, but worth occasionally cleaning up); `skipped_changed`
counts rows a concurrent `replace()`/`remove()` touched mid-backfill (also
harmless — picked up on retry).

Example systemd timer + alerting sketch:

```ini
# /etc/systemd/system/hermes-pgvector-backfill.service
[Service]
ExecStart=/path/to/venv/bin/hermes-pgvector backfill --config /opt/hermes/.hermes/config.yaml
# non-zero exit -> OnFailure=... your alerting unit
```

## prune

Deletes conversation turns older than N days. `memory_entries` are **never**
pruned by any code path — only `conversations`:

```bash
hermes-pgvector prune --days 90              # dry-run: reports what WOULD delete
hermes-pgvector prune --days 90 --execute    # actually deletes
```

Defaults `--days` to the configured `ttl_days` (default `0`, meaning "off" —
`prune` still requires an explicit operator run either way; nothing prunes
automatically in the background).

## cleanup

Deletes every row for one or more `agent_identity` values — for retiring a
theme or purging a mistakenly-created one:

```bash
hermes-pgvector cleanup --identities "old-theme,typo-theme"              # dry-run
hermes-pgvector cleanup --identities "old-theme,typo-theme" --execute
hermes-pgvector cleanup --identities whatsapp-dm --tables memory_entries --execute
```

Runs a PII content scan (10-11 digit number pattern) over the affected
tables first and prints the count — a phone number can live in row
*content* even after the identity itself has been bucketed away, so this is
your signal that a content-level scrub may still be needed. `--tables`
scopes both the scan and the delete to just the named table(s) (default:
both `memory_entries` and `conversations`).

The scan covers `content` only, not the `metadata` column: when the raw
session key differs from the canonical identity, rows record it as
`metadata->>'raw_identity'`, and for DM/group buckets that raw key can
contain a phone number or participant id. Deleting the rows removes it, but
the scan will not flag it, so check metadata separately if that matters.

## remap

Merges one `agent_identity` into another — the tool for folding a mixed-case
or renamed theme into its canonical form:

```bash
hermes-pgvector remap --old Marketing --new marketing              # dry-run
hermes-pgvector remap --old Marketing --new marketing --execute
hermes-pgvector remap --old Marketing --new marketing --execute --force   # >10 dupes
```

`memory_entries` duplicates (same target + content already present under the
new identity) are dropped, not errored, via `INSERT ... ON CONFLICT DO
NOTHING`; more than 10 dropped duplicates requires `--force` as a
data-loss guard. `conversations` rows are moved with a plain `UPDATE` (no
unique constraint there). Runs under an advisory lock shared with `cleanup`
so the two never interleave.

## stats

Read-only health + row-count summary:

```bash
hermes-pgvector stats --config $HERMES_HOME/config.yaml
```

Prints `health()` (liveness + `row_count_estimate` — an approximation from
`pg_class.reltuples`, not an exact count), exact scoped row counts for
`memory_entries` / `conversations`, per-table NULL-embedding counts (a
dry-run `backfill`), and — if migration `002` is applied — per-agent
attribution from `v_agent_memory`.

## What to alert on

- **Writer warnings** (`pgvector worker (...) failed: ...` at WARNING level)
  — the first DB write failure each session is escalated to WARNING; later
  ones in the same session drop to DEBUG so they don't flood logs. One
  WARNING is a Postgres restart or network blip; a warning on *every*
  session init is a real outage.
- **`backfill` exit code != 0** on the scheduled job — see the exit-code
  table above.
- **`remaining` not shrinking** across successive `backfill` runs — the
  embed endpoint is likely down or the dimension configuration drifted.
- **`hermes-pgvector migrate`'s `NOTICE` output** on a fresh install — a
  `runtime role "..." not found` notice means grants were skipped and the
  runtime user will hit `permission denied` on its first write.
