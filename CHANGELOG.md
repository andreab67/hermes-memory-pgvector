# Changelog

All notable changes to `hermes-memory-pgvector` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project does not yet follow strict semantic versioning (pre-1.0) — see
[ROADMAP.md](ROADMAP.md) and the "Compatibility policy" section of
[README.md](README.md) for what is considered the public, semver-covered
surface (`plugins.pgvector.*` config keys, tool names/params, CLI
commands/flags/exit codes, table/column names, the `pgvector` provider name,
and the package entry point — `MemoryStore` and every internal module are
not).

## [1.0.0rc1] - Unreleased

No behaviour changes relative to 0.6.0. Declares the compatibility policy
stable: the public surface listed in the "Compatibility policy" section of
[README.md](README.md) is now guaranteed stable across the 1.x series,
with breaking changes to that surface reserved for a future 2.0. `1.0.0`
will be this same release candidate re-tagged, unchanged, after a soak
period.

## [0.6.0] - Unreleased

Correctness release from a 1.0-readiness review
([`docs/code-review/1.0-readiness-review-2026-09-26.md`](docs/code-review/1.0-readiness-review-2026-09-26.md)):
four High findings, seven Medium, ten Low. Also adds CI, a typed config
schema, an upstream conformance suite, and this changelog — the four items
that were blocking a 1.0 release.

### Breaking / upgrade notes

Read all of these before upgrading. Full step-by-step procedure:
[`docs/upgrading.md`](docs/upgrading.md).

- **`embed_url` default changed** from a private LAN address to
  `http://localhost:11434`. If you relied on the old default, set
  `plugins.pgvector.embed_url` explicitly before upgrading.
- **`embed_timeout` default lowered from `10.0` to `5.0`.** Ambient recall
  (`prefetch()`) now has a hard wall-clock `prefetch_budget` (default `5.0`
  seconds) and can be pre-computed in the background via `queue_prefetch()`,
  so a slow embed endpoint degrades to full-text-only recall instead of
  blocking the agent turn.
- **Identities are lowercased.** `Marketing` and `marketing` now resolve to
  the same theme. The `scope` argument of `recall_memory` /
  `recall_conversation` is case-folded the same way, so `scope='Marketing'`
  reads the `marketing` theme. Find existing mixed-case themes before upgrading:

  ```sql
  SELECT DISTINCT agent_identity FROM memory_entries
   WHERE agent_identity <> lower(agent_identity);
  SELECT DISTINCT agent_identity FROM conversations
   WHERE agent_identity <> lower(agent_identity);
  ```

  Then fold each one with `hermes-pgvector remap --old <Mixed> --new <mixed> --execute`.
- **Migration ordering matters this time.** Upgrade the
  `hermes-memory-pgvector` package on **every host that writes to the
  database first**, then run `hermes-pgvector migrate` (which applies
  migration `005`). Migration `005` drops `memory_entries_unique`, the
  index a pre-0.6.0 plugin's `ON CONFLICT (agent_identity, target,
  content)` names as its conflict target — running `005` before every
  writer is upgraded makes every `add()` on the old code fail with
  `there is no unique or exclusion constraint matching the ON CONFLICT
  specification`. Pass `--runtime-role NAME` to `migrate` if your runtime
  role is not `hermes`.
- **`write_contexts` (new key, default `primary,cron`) gates writes by
  `agent_context`.** Contexts not in the list — `subagent`, `flush` — no
  longer write to Postgres (recall is unaffected). Delegated subagent
  output already reaches the parent via `on_delegation`, so this should be
  a no-op for most deployments; check your `agent_context` usage if not.
- **`backfill` exit codes changed**: `0` = done (`remaining: 0` on every
  table, no table aborted); `2` = embed endpoint unavailable or a table's
  pass aborted; `3` = rows still remaining or failed, with no abort. Update
  any timer/alerting that checked for a non-zero exit generically.
- **`conversations.parent_session_id` on a delegation row now means the
  delegating (parent) session**, matching migration `002`'s original
  column definition. It previously held the child session id — the
  reverse. The child session id is unchanged in
  `metadata.child_session_id` and in `memory_agent_edges`.
- **`health()` returns `row_count_estimate`, not `row_count`.** Internal
  API, but visible in `hermes-pgvector stats` output — the number is now an
  approximation (`pg_class.reltuples`), not an exact `COUNT(*)`.

### Added

- **`hermes-pgvector migrate --runtime-role NAME`** (H1) — resolves the
  runtime role migrations `002`/`004` grant DML to from a session GUC
  (`hermes_pgvector.runtime_role`) instead of hard-coding `hermes`. A `psql
  -f` run gets the same effect with `PGOPTIONS='-c
  hermes_pgvector.runtime_role=NAME'`. A role that does not exist yet gets
  a `NOTICE`, never a failed migration.
- **Migration `005_content_md5_unique.sql`** (M1) — unique index on
  `(agent_identity, target, md5(content))`, replacing the raw-text
  `UNIQUE(agent_identity, target, content)` btree that could not hold an
  entry over roughly 2.7 KB.
- **`replace()` / `remove()` gain an `exact_content` keyword** (H2) — an
  exact `content = %s` match (lowest id) instead of the `old_text`
  substring `LIKE`, used when the caller has the upstream tool's exact
  prior entry (`metadata["previous_content"]`).
- **`queue_prefetch(query, session_id=...)`** (H4) — a background,
  single-worker cache that pre-computes a ready-to-use ambient-recall block
  for a session; `prefetch()` consumes it on the next call instead of
  embedding + searching synchronously.
- **`prefetch_budget` config key** (default `5.0`s) (H4) — hard wall-clock
  budget for `prefetch()`'s synchronous fallback path, kept below the
  hermes-agent host's 8-second external-prefetch timeout.
- **`write_contexts` config key** (default `primary,cron`) (M4) — which
  `agent_context` values are allowed to write.
- **`shutdown_drain_timeout` config key** (default `10.0`s) (M3) — timeout
  for the writer-queue drain on `shutdown()`.
- **`AsyncWriter.draining`** read-only property (M3/L5), replacing the
  unused, never-set `_stop` event.
- **`hermes-pgvector --version`** — prints the installed distribution
  version via `importlib.metadata`.
- **Typed config schema** — every `get_config_schema()` entry now declares
  a real `type` (`text`/`integer`/`number`/`boolean`), a typed `default`,
  and `minimum`/`maximum` where meaningful; new entries for
  `identity_aliases`, `embed_write_backoff`, `write_contexts`,
  `prefetch_budget`, and `shutdown_drain_timeout` that existed in code but
  were previously undeclared.
- **`docs/configuration.md`** — full config reference, generated from
  `get_config_schema()` + `DEFAULTS` by `scripts/gen_config_doc.py`
  (`--check` verifies it is current; wired into CI).
- **CI** (`.github/workflows/ci.yml`) — lint, unit tests on Python
  3.11/3.12/3.13, live tests against Postgres 16/17/18 (with a
  no-`hermes`-role cluster for the H1 regression test), sdist/wheel build +
  `twine check` + wheel-contents assertion, the version-consistency check,
  and the upstream conformance suite below.
- **Upstream conformance suite** (`conformance/`, pinned to
  `conformance/HERMES_AGENT_REF`) — checks `PgvectorMemoryProvider` against
  hermes-agent's actual `MemoryProvider` ABC and `MemoryManager` call
  shapes: subclass/instantiation, per-method signature compatibility, the
  `hermes_agent.memory_providers` entry point, `previous_content` delivery
  through `notify_memory_tool_write`, and the external-prefetch timeout
  contract. Runs on every push/PR at the pinned ref, plus a non-blocking
  weekly drift check against hermes-agent `main`.
- **`docs/operations.md`**, **`docs/scaling.md`**,
  **`docs/troubleshooting.md`**, **`docs/upgrading.md`**,
  **`docs/release/RELEASING.md`**, **`SECURITY.md`** — the docs set.

### Changed

- **Backfill re-architected** (H3) — a single keyset-paginated pass per
  table (`id > last_id ORDER BY id LIMIT batch_size`) instead of re-fetching
  the same failing rows every batch; a consecutive-failure breaker
  (default 20) aborts a table's pass early on a flapping endpoint instead
  of looping forever, and the result gains `note` /
  `skipped_changed` keys.
- **Backfill's persisting `UPDATE` is now guarded** (M2) —
  `WHERE id = %s AND embedding IS NULL AND content = %s`; a row changed or
  deleted between the SELECT and the UPDATE (a concurrent `replace()` /
  `remove()`) is counted as `skipped_changed`, not `failed` or
  `succeeded`, and never gets stamped with a stale embedding.
- **`ON CONFLICT (agent_identity, target, content) DO NOTHING` → `ON
  CONFLICT DO NOTHING`** (M1, code) with no explicit conflict target, in
  `add()` and `remap_identity()` — satisfied by either the old or the new
  (`005`) unique index, so the same code works before and after migration
  `005`.
- **`bulk_upsert_md()` no longer aborts a whole file on one bad entry** —
  a per-entry insert error is now caught, logged once per file with a
  count, and the import continues with the next entry (M1, related).
- **`health()` no longer runs `COUNT(*)`** (M6) — returns
  `row_count_estimate` from `pg_class.reltuples` (an O(1) planner
  estimate) instead of an exact, full-scan count. `count()` /
  `count_turns()` are unchanged (scoped, index-backed).
- **`system_prompt_block()` is computed once per `initialize()`** (M6),
  matching the upstream contract that it is static per session, instead of
  re-querying the database on every call.
- **Pure-vector `search()` / `search_turns()` now filter `embedding IS NOT
  NULL`** (L6) — with `hybrid_search: false`, a NULL-embedding row
  (written during an embed-endpoint outage) no longer fills the result
  tail with `score: null`.
- **`scan_pii`'s default pattern is bounded** (L8) — `(^|[^0-9])[0-9]{10,11}([^0-9]|$)`
  no longer matches a 10-11 digit window inside a longer digit run.
- **`cleanup` passes `--tables` through to the PII scan** (L8) — scanning
  no longer always covers the whole table whitelist when the operator
  asked for one table.
- **Every CLI command closes its store in a `finally` block** (L4).
- **`on_session_end`'s turn-capture backstop no longer requires migration
  `002`** (L9) — it now runs whenever `sync_turns` is enabled and the
  provider is healthy; only the `register_agent` enqueue (a migration-002
  object) stays gated on it.
- **Delegation rows' `parent_session_id` meaning** (M5, finding — not
  roadmap milestone M5) — now the delegating session, matching migration
  `002`'s column definition; the child session id lives in
  `metadata.child_session_id` and `memory_agent_edges` only.
- **Identities are case-folded** (M7) — `normalize_identity()` lowercases
  after strip; alias keys and allow-list entries are compared lowercased.
  The raw value is still recorded as `raw_identity` metadata when it
  differs.
- **Stale text fixed** (L3) — the queue-full config description now says
  "newest writes are dropped" (it always dropped the newest, the text said
  "oldest"); the module docstring's tool list; `normalize_identity`'s
  reason list now documents `group-bucket`.

### Fixed

- **H1** — `migrate` no longer aborts on a runtime role other than
  `hermes`. Migrations `002` and `004`'s `GRANT` blocks now resolve the
  role from a session GUC and `RAISE NOTICE` (never error) when it does
  not exist, so `003`, `004`, and `005` still run.
- **H2** — `replace()` / `remove()` no longer rely solely on a caller-hint
  `old_text` substring `LIKE` (lowest id), which could match a stale row
  that merely *contains* the pattern rather than the row the built-in
  store actually edited. When upstream's `previous_content` is available
  (`memory_manager.notify_memory_tool_write`), it is matched exactly
  instead; a `remove` with no exact match logs and does nothing rather
  than falling back to `LIKE`.
- **H4** — `prefetch()` no longer risks a 20-second worst case (two
  sequential embed attempts at the old 10s timeout each) on the agent
  thread; `embed_timeout` dropped to `5.0`, `embed()`'s `auto` protocol now
  shares ONE deadline across both HTTP attempts instead of a fresh timeout
  per attempt, and `queue_prefetch()` lets a caller pre-compute the block
  off the agent thread entirely.
- **L1** — the real-looking Colorado Springs phone number in docstrings and
  tests is replaced with the fictional `555-01xx` range (e.g.
  `15550100123`). Git history was not rewritten; whether to purge the old
  number from history is left to the maintainer.
- **L2** — `embed_url`'s default is no longer a private LAN address.
- **L5** — the unused, never-set `AsyncWriter._stop` event is removed (see
  `draining` above).
- **L10** — `migrate` no longer fails against a `SQL_ASCII`-encoded
  database with `'ascii' codec can't encode character`: migration files are
  now read and executed as raw UTF-8 bytes. The comments in `002`/`004`
  (and the new `005`) are pure ASCII, enforced by a test; `001` and `003`
  are never edited once shipped, so their comments keep their original
  characters and rely on the byte-level execution.
- `sync_turn()` now accepts upstream's keyword-only `messages` /
  `turn_author` (and any later keyword via `**kwargs`), found by the
  conformance suite's signature-compatibility test. The host introspects
  the signature, so this was not a live crash, but the plugin could not
  receive them if upstream tightened the contract further.
- `initialize()` treats a non-string `agent_identity` /
  `gateway_session_key` / `agent_workspace` as absent instead of raising
  `AttributeError`, and `on_memory_write()` drops a non-dict `metadata`
  instead of raising — both found by fuzzing every hook for invariant #4
  (nothing may raise into the agent loop).
- `scripts/conformance.sh` strips whitespace when reading
  `conformance/HERMES_AGENT_REF`, so a CRLF checkout (`core.autocrlf=true`)
  no longer produces an invalid ref.

## [0.5.5] - 2026-09-22

- **Dependency floor raised**: `psycopg-pool>=3.3.3` (upstream patch
  release, 2026-09-22). psycopg-pool 3.3.3 fixes sync pool workers
  terminating after 24 hours with no task to run (upstream ticket #1419)
  — directly relevant here, since this package opens one long-lived
  shared `ConnectionPool` that can sit idle between agent turns.
  `psycopg[binary]>=3.3.6` is unchanged. No code changes.

## [0.5.4] - 2026-09-18

- **Dependency floor raised**: `psycopg[binary]>=3.3.6`,
  `psycopg-pool>=3.3.2` (upstream patch releases, 2026-09-18). psycopg
  3.3.6: Python 3.15 support; a cancelled query no longer waits forever on
  an unresponsive server (needs libpq 17+); cancels the running query on
  `SystemExit`; interval `Column.precision` now reports `None` instead of
  `65535`; fixes dumping nested list subclasses as arrays; discards
  prepared statements on `DEALLOCATE ALL`; better guards dumping a large
  int to binary numeric; faster async waits. psycopg-pool 3.3.2: propagates
  cancellation and other base exceptions raised during a connection check
  — relevant here since this package opens one shared `ConnectionPool`
  across the agent and async-writer threads. No code changes.

## [0.5.3] - 2026-09-17

Configurable embedding model (dimension, auth, protocol). Defaults are
unchanged: 768 dimensions, no `Authorization` header, `auto` protocol. No
schema changes and no new migrations.

- **The embedding dimension is configuration, not code.** `embed_dim`
  (default `768`) replaces the literal 768 in the response check, in the
  backfill dimension guard (`backfill_null_embeddings(expected_dim=...)`),
  and in the `stats` dry-run. The check was moved, not relaxed: a mismatch
  still fails fast with `expected N dims (embed_dim), got M`.
- **Bearer auth for hosted endpoints.** `embed_api_key_env` holds the
  *name* of an environment variable (for example `OPENROUTER_API_KEY`).
  When that variable is set and non-empty, the plugin sends
  `Authorization: Bearer <value>`. The value is read at call time and is
  never logged, stored in config, or included in exception messages.
- **Explicit protocol selection.** `embed_protocol: openai` uses only
  `/v1/embeddings`; `ollama` uses only `/api/embed`; `auto` keeps the old
  try-OpenAI-then-Ollama behavior. Unknown values fall back to `auto` with
  a warning.
- **One embed path.** Prefetch, both recall tools, the init-time bulk
  import, the writer drain, and `hermes-pgvector backfill` all resolve the
  endpoint through one helper. `backfill` gains `--embed-dim`,
  `--embed-api-key-env`, and `--embed-protocol`.
- **Fixed: embeds broke under hermes-agent's plugin loader.** The loader's
  `setattr(pkg, "embed", <the embed submodule>)` replaced the `embed`
  function the provider called, raising `TypeError: 'module' object is
  not callable` on every embed. Call sites now use a private alias the
  loader never touches.
- **Fixed: a read timeout escaped as a bare `TimeoutError`** instead of an
  `EmbeddingError`, bypassing every `except EmbeddingError` fail-soft path.

## [0.5.2] - 2026-09-07

Documentation only. `git diff v0.5.1..v0.5.2` touched `README.md`, one
test file, and the two version strings — no logic changed anywhere, so an
installed 0.5.2 behaves identically to 0.5.1. PyPI renders a project's
README frozen at upload time, so this release exists purely to get two
already-landed fixes onto the package page:

- The v0.5.1 release notes were out of order (sat before v0.5.0 instead of
  after it).
- A test asserted a `failed` count against a dry-run baseline hardcoded to
  `0`, making the comparison a no-op; asserted directly now.

## [0.5.1] - 2026-09-07

Patch release fixing a data-loss bug — upgrade promptly. No schema
changes, no migrations, no API changes.

- **`remove` deleted every mirrored entry for a theme, not one.** `_worker`
  passed `old_text=item.content`, but the built-in tool's remove op
  carries its target in `old_text` (via metadata) and leaves `content`
  empty — so `item.content` was always `""`, `store.remove` built
  `content LIKE '%%'`, and a single `memory remove` deleted the entire
  mirror for that `(agent_identity, target)`. `_worker` now reads
  `extra["old_text"]`, and `store.remove()` refuses an empty pattern
  outright. `remove()` also now deletes at most one row (lowest id),
  matching the built-in tool and this class's own `replace()`.
- **Nothing rejected empty content on write.** An `add`/`replace` carrying
  empty content created a row that could never be embedded (`embed()`
  raises on empty input); such writes are now ignored (`remove` is
  exempt).
- **`backfill_null_embeddings` retried an empty-content row forever** —
  permanently pinning `failed` above zero. Un-embeddable rows are now
  skipped and reported separately as `unembeddable`.

## [0.5.0] - 2026-09-07

**Breaking:** the import package renamed `pgvector` → `hermes_pgvector`.
The distribution (`hermes-memory-pgvector`), the CLI (`hermes-pgvector`),
and the hermes provider name (`pgvector`) are unchanged — only the Python
import moved, because the old top-level name is owned by
[pgvector-python](https://pypi.org/project/pgvector/) and sharing a venv
meant whichever installed last won. Any `python -m pgvector ...` command
(most commonly a scheduled backfill job) breaks and must become
`hermes-pgvector ...` / `python -m hermes_pgvector ...`. See
[`docs/upgrading.md`](docs/upgrading.md) for the full v0.5.0 procedure,
including `hermes-pgvector install --remove` / `--force`.

No schema changes and no new migrations, but not drop-in due to the import
rename above. `MemoryStore.search` / `hybrid_search` / `search_turns` /
`hybrid_search_turns` gain an `exclude_identities` parameter.

- **Read-side identity gate.** `whatsapp-dm` and `_bench` were write-side
  only; `scope='all'` (or naming a bucket directly) could surface DM
  content in any theme, and turn capture would then re-attribute it under
  the reading theme. `scope='all'` now excludes those sinks, and naming
  one explicitly is rejected. Bucketing happens at write time only —
  rows written before their key was bucketed keep their raw identity.
- **Embed timeouts are configurable, split by call path** — `embed_timeout`
  (default was `10.0` then; agent thread) and `embed_write_timeout`
  (default `30.0`; background writer). Previously every caller silently
  took a hardcoded, unconfigurable 10s.
- **Group / channel / thread keys are bucketed** into `external-group`,
  closing the same PII/cardinality gap the DM bucket exists for, reached
  through `chat_type` instead of a DM key.
- **`allowed_themes` accepts a string again** — it was declared a scalar
  string in the config schema but consumed as a list, so a string
  allow-list was iterated character by character and every theme silently
  routed to `default`.
- **Boolean toggles honour `false` again** — `embed_on_write`,
  `sync_turns`, `hybrid_search`, `bulk_sync_on_init` were read with plain
  truthiness against string values, and `bool("false")` is `True`.
- **Conversation turns are no longer written twice** by `sync_turn()` +
  `on_session_end()`; the latter is now a true backstop.
- **`memory` `replace` mirrors correctly** — it now updates the first match
  instead of one bulk `UPDATE` across every substring match (which raised
  `UniqueViolation` and updated zero rows whenever 2+ entries matched).
- **A dead database is no longer silent** — the first worker write failure
  now warns instead of logging at `debug` only.
- **`save_config` merges instead of replacing** `plugins.pgvector`, so
  hand-edited keys not declared by the schema survive a config-panel save.
- **Fail-soft hardening** — `sync_turn` is wrapped, config casts are
  guarded, recall tools coerce non-string args, and `install --remove`
  requires `--force` on a directory that is not a generated shim.

## [0.4.3] - 2026-08-31

- **Dependency floor raised**: `psycopg[binary]>=3.3.5` (upstream bugfix
  release: prepared-statement invalidation on `ALTER`/`DISCARD`, DataError
  fixes for malformed COPY/jsonb data, client-encoding aliases). No code
  changes.

## [0.4.2] - 2026-07-21

Pip-native install + review hardening.

- **`hermes-pgvector install`** generates the `$HERMES_HOME/plugins/pgvector/`
  discovery shim, making a plain `pip install hermes-memory-pgvector`
  deployable on any hermes-agent install — no more vendored copies or
  editable checkouts.
- **Migration `004`** grants the runtime role DML on `memory_entries` /
  `conversations` itself; the manual OWNER-transfer step is gone.
- Correctness fixes from a full-codebase review: `replace`/`remove` match
  `old_text` as a literal substring (LIKE metacharacters escaped); the
  async writer drains its queue on shutdown instead of abandoning queued
  writes; a wrong-dimension embed surfaces as `expected 768 dims, got N`
  instead of a masking 404; DM-key bucketing no longer sweeps ordinary
  `:signal:`-containing theme names into `whatsapp-dm`; bulk MEMORY.md
  import circuit-breaks after 3 consecutive embed failures; `remap`
  re-checks its duplicate-drop guard under the advisory lock; tool errors
  redact credential-looking fragments and preserve `score: null` for
  full-text-only hybrid hits (with `rrf_score` now included);
  `recall_memory(scope='session')` returns a helpful error instead of
  matching nothing.

## [0.4.1] - 2026-07-01

Hybrid recall: vector + full-text.

- `recall_memory` and `recall_conversation` fuse the HNSW vector ranking
  with a Postgres full-text ranking using Reciprocal Rank Fusion (RRF,
  `k=60`) — recovering exact-lexical hits the embedding smooths away and
  text-only rows with a NULL embedding (written while the embed endpoint
  was down).
- Migration `003` adds a GIN index over the existing `content` column on
  both tables (works without it, just slower).
- Toggle with `plugins.pgvector.hybrid_search` (default `true`); the
  ambient `prefetch()` path stays pure-vector. Fail-soft: a hybrid hiccup
  degrades to pure-vector, and a query that fails to embed degrades to
  full-text-only.

## [0.4.0] - 2026-06-13

Four capabilities, all storage-layer.

- **Identity governance.** `agent_identity` is normalized once at init:
  direct-message session keys collapse to a single `whatsapp-dm` bucket,
  benchmark traffic (`skill-bench*`) is isolated to `_bench`, and an
  optional `allowed_themes` allow-list routes unknown themes to `default`.
- **Agent attribution + delegation.** Migration `002` adds `memory_agents`
  (registry), `memory_agent_edges` (parent→child delegation provenance),
  and `conversations.parent_session_id`. `on_delegation` / `on_session_end`
  capture who delegated what to whom — provenance only, enqueue-only,
  fail-soft. Query via the `v_agent_memory` view.
- **Embedding backfill + writer resilience.** `hermes-pgvector backfill`
  re-embeds `NULL`-embedding rows left by an embed-endpoint outage
  (idempotent, dimension-guarded). The background writer gains a small
  bounded retry.
- **Conversation TTL + embed policy.** `hermes-pgvector prune --days N`
  trims old turns (operator-triggered only; `memory_entries` are never
  pruned). `conversation_embed_policy` tunes embedding cost.
- Maintenance CLI (`python -m pgvector` at the time; renamed to
  `hermes-pgvector` in v0.5.0): `migrate · stats · backfill · prune ·
  cleanup · remap` — destructive commands default to dry-run. A clean
  upgrade from v0.3.x: apply migration `002` to light up
  attribution/delegation; without it the new hooks no-op.

## [0.3.1] - 2026-05-29

Connection-leak hotfix. `initialize()` is called again on every new
session for a single registered provider instance; it previously
reassigned `self._store` / `self._writer` without closing the prior ones,
abandoning a `ConnectionPool` whose warm connection lingered in Postgres —
committed-but-idle — until the server's `idle_session_timeout`. Under a
burst of concurrent sessions these orphaned backends saturated the
database's connection slots.

- **`initialize()` teardown** — drains the prior `AsyncWriter` and closes
  the prior pool before re-initializing.
- **Self-draining pool** — `min_size=0` (an idle or abandoned pool holds
  zero connections) plus `max_idle=30s` / `max_lifetime=300s`.

## [0.3.0] - 2026-05-20

Identity propagation for stateless API minions via `X-Hermes-Session-Key`,
plus the M1/M2 baseline this release rests on:

- `initialize()` verifies schema, opens a `psycopg_pool.ConnectionPool`,
  and bulk-imports existing `MEMORY.md` + `USER.md` content.
- `on_memory_write(action, target, content, meta)` mirrors built-in
  `memory` writes into `memory_entries` (add / replace / remove).
- `sync_turn(user, assistant, session_id)` captures every substantive
  (>= 40 chars, not boilerplate) chat turn into `conversations`.
- `prefetch(query)` injects the top-K semantically similar
  `memory_entries` for the current theme, ambient.
- `recall_memory(query, scope, target, limit)` and
  `recall_conversation(query, scope, limit)` tools for explicit
  cross-theme / cross-session search.
- `psycopg_pool.ConnectionPool` (min=0, max=4) shared across the agent
  thread and the async-writer drain thread; `AsyncWriter` — bounded queue
  + daemon drain thread, so memory-write hooks return in microseconds.
- Single migration (`001_schema.sql`): `memory_entries` + `conversations`
  + HNSW indexes.
- Boilerplate filter for turn capture (length floor + acknowledgement
  regex) so the recall table stays high-signal.
- The plugin reads `kwargs.get('gateway_session_key')` (the
  `X-Hermes-Session-Key` header, forwarded by the gateway) in the
  `agent_identity` fallback chain, so per-minion themes actually take
  effect for API-routed traffic instead of collapsing to `default`.
