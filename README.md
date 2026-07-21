# hermes-memory-pgvector

**Postgres + pgvector memory provider for [hermes-agent](https://github.com/NousResearch/hermes-agent).** A shared memory substrate for a fleet of cooperating hermes-agent minions — built on Postgres and a single embedding endpoint you probably already run, with no LLM in the memory hot path.

```text
each minion → X-Hermes-Session-Key: <theme>
            → hermes-agent gateway
            → pgvector plugin
                ├── memory_entries  (mirrors built-in MEMORY.md / USER.md per theme)
                └── conversations   (every substantive turn, semantically searchable)
```

## Why it exists

Existing memory providers each solve a piece of the problem; the gap for **fleet deployments** is wide:

- **Built-in `memory` tool** persists to per-host `MEMORY.md` / `USER.md`. Two minions on the same host stomp on each other; minions on different hosts have no shared substrate.
- **Honcho** offers cross-session user modelling but requires a full external service, an LLM in the memory hot path for its deriver + dialectic loops, and its own ontology layered on top of the built-in tool. In high-concurrency fleet use it produces retry storms, embedding-endpoint queue backups, and gateway↔Honcho circular dependencies.
- **Holographic** is a fine in-process fact store but uses SQLite — a poor fit for many minions writing concurrently from many hosts.
- **Other providers** (Mem0, Hindsight, OpenViking, ByteRover, RetainDB, Supermemory) all either require a paid cloud, require LLM mediation for memory ops, or both.

What was missing: a **storage layer** that gives the built-in `memory` model durable, multi-tenant, semantically-searchable backing, with no LLM in the hot path, scoped cleanly per-minion so a marketing agent's notes don't pollute a trading agent's recall. That's what this plugin provides.

## Design philosophy

1. **Storage layer, not a memory model.** The agent keeps using `memory(action='add', target='memory'|'user', …)`. We mirror those writes via `on_memory_write`. No new ontology for the agent to learn.
2. **No LLM in the memory hot path.** Embeddings are vector math, not LLM calls. There is no deriver, no dialectic, no dream cycle — the failure modes that hurt Honcho cannot occur here by construction.
3. **Per-agent themes by default, cross-theme recall on explicit demand.** Every row carries `agent_identity` (resolved from `X-Hermes-Session-Key` header, profile name, workspace, or `'default'`). Recall is scoped to the current theme unless the agent asks for `scope='all'`.
4. **Fail-soft everywhere.** Embed endpoint down → degrade to text-only writes. Async writer queue full → drop with a one-time warning. DB down → log + skip. No exception escapes into the agent loop.
5. **Admin/runtime separation.** DDL (`CREATE EXTENSION vector`, `CREATE TABLE`, `CREATE INDEX`) runs once with superuser. The runtime user has DML only on the migrated schema. `ensure_schema()` at runtime is verify-only with a clear `SchemaNotApplied` error if the operator forgot the migration.

## Features (v0.3.0)

| Hook / surface | Behavior |
|---|---|
| `initialize()` | Verifies schema, opens `psycopg_pool.ConnectionPool`, bulk-imports existing `MEMORY.md` + `USER.md` content. |
| `on_memory_write(action, target, content, meta)` | Mirrors built-in `memory` writes into `memory_entries` (add / replace / remove). |
| `sync_turn(user, assistant, session_id)` | Captures every substantive (`>= 40` chars + not boilerplate) chat turn into `conversations`. |
| `prefetch(query)` | Top-K semantically similar `memory_entries` in current theme, injected ambient. |
| `recall_memory(query, scope, target, limit)` tool | Explicit cross-theme search of durable memory entries. |
| `recall_conversation(query, scope, limit)` tool | Explicit search over past chat turns. `scope ∈ {current, session, all, <theme>}`. |

Internals:

- **`psycopg_pool.ConnectionPool`** (min=0, max=4, lazy + thread-safe, `max_idle=30s` / `max_lifetime=300s`) shared across the agent thread and the async-writer drain thread. `min_size=0` keeps an idle — or abandoned — pool at **zero** open connections, so a session the gateway never explicitly shuts down cannot strand a Postgres backend (see *Fixed in v0.3.1* below).
- **`AsyncWriter`** — bounded queue + daemon drain thread. Memory write hooks return in microseconds. Worker embeds + writes in the background. Crash-resilient (auto-restart on next enqueue).
- **Single migration** (`pgvector/migrations/001_schema.sql`) — `memory_entries` + `conversations` + HNSW indexes. Same tuning operators typically use elsewhere.
- **Boilerplate filter** for turn capture — length floor + acknowledgement regex (`"ok"`, `"thanks"`, `"continue"`, …) so the recall table stays high-signal.

### Fixed in v0.3.1 — connection-leak hotfix

A single registered provider has `initialize()` called again for each new session. It previously
reassigned `self._store` / `self._writer` without closing the prior ones, **abandoning a
`ConnectionPool`** whose warm (`min_size=1`) connection lingered in Postgres — committed-but-idle —
until the server's `idle_session_timeout`. Under a burst of concurrent sessions (e.g. a swarm of
systemd-run minions firing on the same minute) these orphaned backends saturated the database's
connection slots. Fixed by:

1. **`initialize()` teardown** — drain the prior `AsyncWriter` + close the prior pool before
   re-initializing (the call is idempotent and skipped on first init).
2. **Self-draining pool** — `min_size=0` (an idle or abandoned pool holds *zero* connections) plus
   `max_idle=30s` / `max_lifetime=300s`, so connections are short-lived when idle and pooled only
   under active load.

## New in v0.4.0

Four capabilities, all storage-layer (still no LLM in the hot path):

- **Identity governance.** The resolved `agent_identity` is normalized once at init: direct-message session keys like `agent:main:whatsapp:dm:<phone>` collapse to a single `whatsapp-dm` bucket (no PII, no per-contact theme explosion), benchmark traffic (`skill-bench*`) is isolated to `_bench`, and an optional `allowed_themes` allow-list routes typo'd/unknown themes to `default`. The M3 resolution priority is preserved — normalization runs *after* the chain, never at read time. See [`pgvector/identity.py`](pgvector/identity.py).
- **Agent attribution + delegation (M4).** Migration `002` adds `memory_agents` (registry) and `memory_agent_edges` (parent→child delegation provenance) plus `conversations.parent_session_id`. The `on_delegation` / `on_session_end` hooks capture which agent delegated what to whom — strictly enqueue-only and fail-soft. **Provenance only** (who/when), never a fact-store ontology. Query it via the `v_agent_memory` view.
- **Embedding backfill + writer resilience.** Rows written text-only during an embed-endpoint outage are no longer permanently unsearchable: `python -m pgvector backfill` re-embeds `NULL`-embedding rows (idempotent, 768-dim-guarded). The background writer gains a small bounded retry; the hot path stays single-attempt.
- **Conversation TTL + embed policy.** `python -m pgvector prune --days N` trims old turns (operator-triggered only; `memory_entries` are never pruned). `conversation_embed_policy` (`all` default / `substantive_only` / `none`) tunes embedding cost.

Maintenance CLI (`python -m pgvector`, installed as `hermes-pgvector`): `migrate · stats · backfill · prune · cleanup · remap` — destructive commands default to dry-run. v0.4.0 is a clean upgrade from v0.3.x: apply migration `002` to light up attribution/delegation; without it the new hooks no-op and everything else runs unchanged.

## New in v0.4.1 — hybrid recall (vector + full-text)

`recall_memory` and `recall_conversation` now fuse the HNSW **vector** ranking with a Postgres **full-text** ranking using **Reciprocal Rank Fusion** (RRF, `k=60`). A row surfaces if *either* ranker likes it, which fixes the two blind spots of pure cosine similarity:

- **Exact-lexical hits** the embedding smooths away — a specific error code, hostname, flag name, or rare identifier the agent quotes verbatim.
- **Text-only rows with a `NULL` embedding** (written while the embed endpoint was down) — invisible to the vector index, but the full-text leg finds them. So hybrid recall doubles as best-effort recovery until the next `backfill`.

Still a storage-layer feature: **no LLM, no entity graph, no new tables or columns** — just a GIN index over the existing `content` column (migration `003`) and a fused query. It stays inside invariant #1 (a second index over the same text is not a parallel ontology). Fail-soft as ever: a hybrid hiccup degrades to the proven pure-vector path, and a query that *itself* fails to embed degrades to full-text-only instead of erroring. Toggle with `plugins.pgvector.hybrid_search` (default `true`); the ambient `prefetch()` path stays pure-vector. Works without migration `003` — the GIN index only makes the full-text leg faster.

## New in v0.4.2 — pip-native install + hardening

- **`hermes-pgvector install`** — makes a plain `pip install hermes-memory-pgvector` deployable on ANY hermes-agent install: generates the `$HERMES_HOME/plugins/pgvector/` discovery shim (see *Install · Option 1*). No more vendored copies or editable checkouts.
- **Migration `004`** — `hermes-pgvector migrate` now grants the runtime role DML on `memory_entries`/`conversations` itself; the manual OWNER-transfer step is gone (fresh installs previously hit `permission denied` if it was skipped).
- **Correctness fixes** from a full-codebase review: `replace`/`remove` now match `old_text` as a *literal* substring (LIKE `%`/`_`/`\` metacharacters no longer over- or under-match — parity with the built-in tool's `in` semantics); the async writer drains its queue on shutdown instead of silently abandoning up to 255 accepted writes when full; a wrong-dimension embed model now surfaces as `expected 768 dims, got N` instead of a masking 404; DM-key bucketing no longer sweeps ordinary `:signal:`-containing theme names into `whatsapp-dm`; bulk MEMORY.md import circuit-breaks after 3 consecutive embed failures (a hanging endpoint can no longer block session start for minutes); `remap` re-checks its duplicate-drop guard under the advisory lock; tool errors redact credential-looking fragments and preserve `score: null` for full-text-only hybrid hits (with `rrf_score` now included); `recall_memory(scope='session')` returns a helpful error instead of silently matching nothing.

## Multi-agent / per-minion themes

Each systemd-run minion sets one header on its OpenAI client; everything else flows automatically:

```python
client = AsyncOpenAI(
    base_url="http://127.0.0.1:8642/v1",
    api_key=API_KEY,
    default_headers={"X-Hermes-Session-Key": "marketing"},   # ← theme
)
```

The gateway plumbs `X-Hermes-Session-Key` through as `gateway_session_key=…` in `MemoryProvider.initialize` kwargs. The plugin reads it with **priority over the profile default**, so `agent_identity='default'` from unprofiled API traffic does not collapse every minion into one shared scope.

Convention: lowercase, dash-separated, stable. Active themes in this deployment:

- product/report themes: `marketing`, `sales`, `morning-report`, `morning-report-sr`, `sr-marketing`, `sr-cloud`
- per-worker minions: `agent-trading`, `agent-sre`, `agent-marketing`, `agent-gitlab`, `agent-cloud`, `agent-hermes`
- governed sinks (v0.4): `whatsapp-dm` (collapsed DM/session keys), `_bench` (benchmark traffic), `default` (last resort)

Set `plugins.pgvector.allowed_themes` to that product/worker list to enforce it — an unknown or typo'd header then falls back to `default` (with a one-time warning) instead of silently minting a new theme.

## Install

### Option 1: pip + discovery shim (recommended, v0.4.2+)

```bash
# 1. Install the package into the SAME environment hermes-agent runs in
pip install hermes-memory-pgvector

# 2. Create the discovery shim
hermes-pgvector install          # writes $HERMES_HOME/plugins/pgvector/

# 3. Apply ALL migrations (schema + attribution + FTS + runtime grants)
hermes-pgvector migrate --admin-dsn \
    "dbname=<your-memory-db> user=postgres host=/var/run/postgresql"

# 4. Activate + verify
hermes config set memory.provider pgvector
sudo systemctl restart hermes.service
hermes memory status             # expect: Provider: pgvector; Status: available
```

**Why the shim?** hermes-agent discovers memory providers by scanning plugin *directories* — `plugins/memory/<name>/` (bundled) and `$HERMES_HOME/plugins/<name>/` (user) — and never looks at installed packages. `pip install` alone is therefore invisible to it. `hermes-pgvector install` writes a two-line shim whose absolute import resolves to the pip-installed package, so upgrades are just `pip install -U hermes-memory-pgvector` + restart, and rollback is `pip install hermes-memory-pgvector==<prev>` + restart — the shim never changes. `--remove` deletes it; if the package is uninstalled the shim import fails cleanly and hermes falls back to built-in memory.

### Option 2: clone + run the installer script (from source)

```bash
git clone https://github.com/andreab67/hermes-memory-pgvector.git
cd hermes-memory-pgvector
./scripts/install.sh
```

That:

1. `pip install`s `psycopg[binary]`, `psycopg-pool`, `PyYAML` (with the upper-bound pins).
2. Copies `pgvector/` into `$HERMES_HOME/plugins/pgvector/` (defaults to `~/.hermes/plugins/pgvector/`).
3. Prints the admin migration + activation commands you run next.

### Option 3: manual

```bash
# Python deps
pip install 'psycopg[binary]>=3.3.4,<4' 'psycopg-pool>=3.3.1,<4' 'PyYAML>=6.0,<7'

# Plugin module
mkdir -p ~/.hermes/plugins
cp -r pgvector ~/.hermes/plugins/pgvector
```

### Then (admin once)

```bash
# Apply the schema migration (CREATE EXTENSION needs superuser)
sudo -u postgres psql -d <your-memory-db> \
     -f ~/.hermes/plugins/pgvector/migrations/001_schema.sql

# v0.4.0: apply the agent-attribution migration too (adds memory_agents /
# memory_agent_edges / conversations.parent_session_id and the GRANTs the
# runtime role needs — it self-grants, so no extra OWNER step for these).
sudo -u postgres psql -d <your-memory-db> \
     -f ~/.hermes/plugins/pgvector/migrations/002_agent_attribution.sql

# v0.4.1: apply the hybrid-search full-text indexes (GIN over content on both
# tables). Optional — hybrid recall works without it, just seq-scans the FTS
# leg. No new tables/columns/GRANTs; needs no OWNER step.
sudo -u postgres psql -d <your-memory-db> \
     -f ~/.hermes/plugins/pgvector/migrations/003_hybrid_search_fts.sql

# v0.4.2: grant the runtime role DML on the core tables (replaces the old
# manual "ALTER TABLE ... OWNER TO hermes" step; skips with a NOTICE if your
# runtime role isn't named 'hermes' — grant manually in that case).
sudo -u postgres psql -d <your-memory-db> \
     -f ~/.hermes/plugins/pgvector/migrations/004_runtime_grants.sql
# (or apply every migration in order:  hermes-pgvector migrate --admin-dsn "user=postgres host=/var/run/postgresql dbname=<your-memory-db>")

# Activate
hermes config set memory.provider pgvector
sudo systemctl restart hermes.service     # or however you run hermes
hermes memory status                       # expect: Provider: pgvector; Status: available
```

## Configuration

Lives in `$HERMES_HOME/config.yaml` under `plugins.pgvector` — every value optional, sensible defaults shown:

```yaml
plugins:
  pgvector:
    dsn: "dbname=hermes_memory user=hermes host=/var/run/postgresql"
    embed_url: "http://your-embed-endpoint:11434"
    embed_model: "nomic-embed-text"
    prefetch_limit: 5
    min_similarity: 0.30
    embed_on_write: true
    scope_default: "current"
    write_queue_maxsize: 256
    bulk_sync_on_init: true
    sync_turns: true
    turn_min_chars: 40
    # --- v0.4 identity governance + maintenance ---
    allowed_themes: []            # empty = governance off; a list enforces an allow-list
    bench_mode: "bucket"          # bucket -> _bench | reject -> default
    conversation_embed_policy: "all"   # all | substantive_only | none
    ttl_days: 0                   # 0 = off; only `pgvector prune` ever deletes (never automatic)
    embed_write_retries: 2        # writer-path only; hot path stays single-attempt
```

The embed endpoint can be any OpenAI-compatible `/v1/embeddings` or Ollama-native `/api/embed` URL that returns **768-dim vectors** (the schema is hard-coded to `vector(768)` to match `nomic-embed-text`). Use a different model only if it produces 768-dim output, or edit the migration before applying it.

## Schema

```sql
CREATE TABLE memory_entries (
  id              BIGSERIAL PRIMARY KEY,
  agent_identity  TEXT NOT NULL DEFAULT 'default',
  target          TEXT NOT NULL CHECK (target IN ('memory', 'user')),
  content         TEXT NOT NULL,
  embedding       vector(768),
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE (agent_identity, target, content)
);

CREATE TABLE conversations (
  id              BIGSERIAL PRIMARY KEY,
  session_id      TEXT NOT NULL,
  agent_identity  TEXT NOT NULL DEFAULT 'default',
  role            TEXT NOT NULL CHECK (role IN ('user','assistant','system','tool')),
  content         TEXT NOT NULL,
  ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
  embedding       vector(768),
  metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
);
```

Indexes: HNSW on each `embedding` column (m=16, ef_construction=64) plus per-agent + per-session btree timelines. Full DDL in [`pgvector/migrations/001_schema.sql`](pgvector/migrations/001_schema.sql).

## Tests

```bash
pip install -e ".[test]"

# Skip mode (no DB, no embed endpoint): everything skips gracefully
pytest tests/

# Live mode (against a throwaway Postgres + your embed endpoint)
export PG_TEST_DSN='dbname=hermes_test user=postgres host=/var/run/postgresql'
export PG_TEST_EMBED_URL='http://your-embed-endpoint:11434'
pytest tests/
```

DB tests skip when `PG_TEST_DSN` is unset; live embed tests skip when `PG_TEST_EMBED_URL` is unset.

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the full milestone table. Highlights:

- **M1 (v0.1, v0.1.1)** ✅ Shared storage with per-agent themes, async writer, connection pool, bulk import from `MEMORY.md`/`USER.md`
- **M2 (v0.2)** ✅ Conversation transcript table with `sync_turn` capture + `recall_conversation` tool
- **M3 (v0.3)** ✅ Identity propagation for stateless API minions via `X-Hermes-Session-Key`
- **M4 (v0.4)** ✅ Identity governance + `on_delegation()`/`on_session_end()` capture + agent attribution (`memory_agents`/`memory_agent_edges`), embedding backfill, conversation TTL, maintenance CLI
- **M5 (v0.5–v0.6)** ⏳ Decay scoring, partial HNSW indexes per-theme, Prometheus metrics, cross-provider bulk-import
- **M6 (v1.0)** ⏳ Stable config schema, full docs, CI coverage

The roadmap exists so the multi-agent positioning isn't a one-off claim — each milestone has to pass the test *"does this make N cooperating agents more capable?"* before it lands. The `What's not on the roadmap` section in `ROADMAP.md` lists what was deliberately rejected (LLM-mediated dialectic, fact-store ontologies, background derivers, in-plugin RBAC) so the boundaries are explicit.

## Rollback

```bash
hermes config set memory.provider none
sudo systemctl restart hermes.service

# Optional — drop the tables (data loss, irreversible)
sudo -u postgres psql -d <your-memory-db> -c "
DROP TABLE IF EXISTS conversations;
DROP TABLE IF EXISTS memory_entries;
"

# Optional — remove the plugin files
rm -rf ~/.hermes/plugins/pgvector
```

## Why a standalone plugin (not an upstream PR)?

Per the hermes-agent [`CONTRIBUTING.md`](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md):

> We are no longer accepting new memory providers into this repo. The set of built-in providers under `plugins/memory/` is closed. If you want to add a new memory backend, publish it as a standalone plugin repo that users install into `~/.hermes/plugins/` (or via a pip entry point).

The discovery system (`plugins/memory/__init__.py` in hermes-agent) scans `$HERMES_HOME/plugins/<name>/` for any directory whose `__init__.py` calls `register_memory_provider`. This plugin's `pgvector/__init__.py` does exactly that — no upstream change required.

## Contributing

Bug reports + PRs welcome. Open an issue describing the failure mode + your environment (hermes-agent version, Postgres version, embed endpoint), or a PR with a focused change + test.

## License

[BSD 3-Clause](LICENSE) © 2026 Green Yoga Inc
