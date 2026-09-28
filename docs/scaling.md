# Scaling

Guidance for running this plugin under real fleet load: dozens of minions,
hundreds of writes per minute, multi-million-row tables. Everything here is
tuning within the existing design (no LLM in the hot path, no deriver) —
see [ROADMAP.md](../ROADMAP.md) M5 for what's still proposed rather than
shipped.

## HNSW index tuning

Migration `001` creates HNSW indexes on both `embedding` columns with
`m = 16, ef_construction = 64` — reasonable defaults for most deployments.
Increase either only once you have evidence recall quality is the
bottleneck, not before:

- **`m`** (default 16): number of bi-directional links per node. Higher
  improves recall accuracy at the cost of build time and index size.
  16-24 covers most workloads; go higher only past several million rows
  per theme.
- **`ef_construction`** (default 64): candidate list size during index
  build. Higher improves recall accuracy at the cost of build time only
  (no query-time cost). 64-128 is typical.

Changing either requires rebuilding the index (`DROP INDEX` +
`CREATE INDEX ... WITH (m = ..., ef_construction = ...)`, or
`CREATE INDEX CONCURRENTLY` on a large hot table to avoid blocking
writers — see [configuration.md](configuration.md) and
[upgrading.md](upgrading.md) for the equivalent steps when changing
`embed_dim`, which also requires a rebuild).

### Query-time recall: `hnsw.ef_search`

Set per-session or per-query via `SET hnsw.ef_search = N` (default 40 in
Postgres). Higher values improve recall accuracy at query time at the cost
of latency — worth raising if `recall_memory`/`recall_conversation` seem to
miss obviously-relevant rows on a large table. This plugin does not set it
itself; it is a session-level Postgres GUC an operator can set via the DSN
(keyword/value DSN: `options='-c hnsw.ef_search=100'` -- the quotes are
required, or the space splits the option; URI DSN:
`?options=-c%20hnsw.ef_search%3D100`) or a connection-pool-level default.

### Filtered search and `hnsw.iterative_scan`

Every search in this plugin filters by `agent_identity` (theme scoping) —
structurally a filtered ANN query. HNSW's default behaviour is to run the
similarity search first and post-filter by the `WHERE` clause, which means
a query scoped to a small-to-medium theme inside a very large overall table
can come back with fewer than `limit` rows even though more exist, because
the ANN search didn't happen to visit enough matching rows before its
internal budget ran out. This was reviewed and did not reproduce at 20k
rows (the planner chose an exact scan at that size), but it is a real
failure mode at larger scale and worth knowing about before it surprises
you.

**pgvector >= 0.8** ships `hnsw.iterative_scan` (`off` / `relaxed_order` /
`strict_order`), which re-runs the ANN search with a wider net when the
initial pass under-returns a filtered query — directly addresses this.
Enable per-session:

```sql
SET hnsw.iterative_scan = strict_order;
```

Version-gated: check `SELECT extversion FROM pg_extension WHERE extname =
'vector'` before setting it — on pgvector `< 0.8` the GUC does not exist
and setting it errors. This plugin does not set it automatically (keeping
runtime behaviour independent of the exact pgvector point release
installed); set it in your connection setup if your themes are large enough
for this to matter.

### Optional partial indexes per high-volume theme

Not shipped by this plugin (see ROADMAP M5) but straightforward to add
yourself if one `agent_identity` dominates table size and cross-theme
search on the rest becomes the slow query:

```sql
CREATE INDEX CONCURRENTLY ix_memory_entries_embedding_hnsw_excl_bigtheme
  ON memory_entries USING hnsw (embedding vector_cosine_ops)
  WHERE agent_identity <> 'the-big-theme'
  WITH (m = 16, ef_construction = 64);
```

Keep the full unscoped index too (queries scoped to `the-big-theme` itself
still need it); this is additive, not a replacement.

## Connection pool sizing

`psycopg_pool.ConnectionPool` is opened with `min_size=0, max_size=4`,
shared across the agent thread and the async-writer drain thread per
provider instance. `min_size=0` means an idle or abandoned pool holds
**zero** open connections — deliberate, so a session the gateway never
explicitly tears down cannot strand a Postgres backend (this was a real
connection-leak bug in v0.3.0, fixed in v0.3.1). `max_idle=30s` /
`max_lifetime=300s` keep connections short-lived under intermittent load
and bounded under sustained load.

Four connections per provider instance times N concurrently-initialized
minions is the number that matters for `max_connections` sizing on the
Postgres side — with dozens of minions, budget accordingly (or lower
`max_size` if your workload is read-light and write-light per minion; it
is not currently configurable and would need a code change to expose).

## Embed latency vs. `prefetch_budget`

`prefetch()`'s synchronous fallback path (used when `queue_prefetch()`
hasn't already cached a block for the session) enforces `prefetch_budget`
(default `5.0`s) as a hard wall-clock deadline across the whole embed +
search step. This must stay comfortably below hermes-agent's own
external-provider prefetch timeout (a fixed 8 seconds on the host side) —
if `prefetch()` doesn't return before that, the host logs a timeout warning
and skips this provider on later turns until the stuck call returns.

If your embed endpoint's p95 latency approaches `prefetch_budget`, prefer
`queue_prefetch()` (called by the host ahead of the turn that needs it) over
raising `prefetch_budget` toward the 8-second ceiling — a cached block
costs the agent thread nothing, while raising the synchronous budget just
moves the risk of hitting the host's own timeout.

`embed_timeout` (default `5.0`s, used by `prefetch`/recall tools/bulk
import) and `embed_write_timeout` (default `30.0`s, background writer only)
are deliberately asymmetric for the same reason: nothing is waiting on the
writer path, so it can afford to wait out a slow endpoint instead of
degrading a row to text-only.

## Estimated vs. exact counts

`health()` reports `row_count_estimate` (from `pg_class.reltuples`, an
approximation updated by autovacuum/`ANALYZE`) instead of an exact
`COUNT(*)` — avoiding a full-table scan on every session `initialize()`.
`count()` / `count_turns()` stay exact and index-backed for a *scoped*
query (a single `agent_identity`), which is cheap at any table size. Do not
expect `stats`' or `health()`'s global figure to be precise to the row on a
table that changed recently; use a scoped `count()` (or `SELECT COUNT(*)`
directly) when exactness matters more than speed.
