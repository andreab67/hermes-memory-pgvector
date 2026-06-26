# Technical Details & Configuration

This document contains the deep technical details, schema definitions, and hook references for `hexus`.

## Plugin Internals (shared by both surfaces)

- **`psycopg_pool.ConnectionPool`** (min=0, max=4, lazy + thread-safe, `max_idle=30s` / `max_lifetime=300s`) shared across the agent thread and the async-writer drain thread. `min_size=0` keeps an idle — or abandoned — pool at **zero** open connections, so a session the gateway never explicitly shuts down cannot strand a Postgres backend.
- **`AsyncWriter`** — bounded queue + daemon drain thread. Memory write hooks return in microseconds. Worker embeds + writes in the background. Crash-resilient (auto-restart on next enqueue).
- **Single migration** (`hexus/migrations/001_schema.sql`) — `memory_entries` + `conversations` + HNSW indexes. Same tuning operators typically use elsewhere.
- **Boilerplate filter** for turn capture — length floor + acknowledgement regex (`"ok"`, `"thanks"`, `"continue"`, …) so the recall table stays high-signal.

## Schema

```sql
CREATE TABLE memory_entries (
  id              BIGSERIAL PRIMARY KEY,
  agent_identity  TEXT NOT NULL DEFAULT 'default',
  target          TEXT NOT NULL CHECK (target IN ('memory', 'user')),
  content         TEXT NOT NULL,
  embedding       vector(384),
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
  embedding       vector(384),
  metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
);
```

Indexes: HNSW on each `embedding` column (m=16, ef_construction=64) plus per-agent + per-session btree timelines. Full DDL in `hexus/migrations/001_schema.sql`.

## Hermes Plugin Surface

| Hook / surface | Behavior |
|---|---|
| `initialize()` | Verifies schema, opens pool, bulk-imports existing `MEMORY.md` + `USER.md` content. |
| `on_memory_write(action, target, content, meta)` | Mirrors built-in `memory` writes into `memory_entries` (add / replace / remove). |
| `sync_turn(user, assistant, session_id)` | Captures every substantive chat turn into `conversations`. |
| `prefetch(query)` | Top-K semantically similar `memory_entries` in current theme, injected ambient. |
| `recall_memory(query, scope, target, limit)` tool | Explicit cross-theme search of durable memory entries. |
| `recall_conversation(query, scope, limit)` tool | Explicit search over past chat turns. |
| `entity_graph(entity_type, entity_value, scope, limit)` tool | Find other entities that co-occur with a target entity. |
| `graph_walk(entity_type, entity_value, scope, max_depth, limit)` tool | Traverse the co-occurrence graph recursively up to N hops. |
| `common_topics(scope, min_strength, limit)` tool | Retrieve clusters/cliques of heavily co-occurring entities. |
| `confirm_memory(id)` tool | Confirm the relevance of a memory entry by incrementing its confirm count. |
| `reject_memory(id)` tool | Flag a memory entry as noise by incrementing its reject count. |
| `summarize_session(session_id, limit)` tool | Extractive summarization: returns turns closest to the session vector centroid. |

## MCP Server Surface

Nineteen tools exposed to any MCP client. All take an optional `agent_identity` argument so each connected client is isolated by default.

| Tool | Purpose |
|---|---|
| `memory_health` | Liveness + capability check (DB status, embedder model/dim, row counts). |
| `memory_retain` | Add one or many memory entries. |
| `memory_recall` | Semantic search over `memory_entries`. |
| `memory_search` | Browse entries (no embedding) — pagination, scoping. |
| `memory_forget` | Delete by id. **Dry-run by default**; pass `confirm=true` to actually delete. |
| `memory_recall_turns` | Semantic search over past chat turns. |
| `memory_append_turn` | Append one chat turn. |
| `memory_count` | Row counts for entries + turns, scoped. |
| `memory_entity_graph` | Find other entities that co-occur with a target entity. |
| `memory_graph_walk` | Traverse the co-occurrence graph recursively up to N hops from a start entity. |
| `memory_common_topics` | Retrieve clusters/cliques of heavily co-occurring entities. |
| `memory_confirm` | Confirm relevance. |
| `memory_reject` | Flag as noise. |
| `memory_summarize_session` | Compute centroid and return K closest turns. |
| `memory_cleanup` | Delete stale records based on TTL. |
| `memory_metrics` | Return operational metrics in Prometheus format. |
| `memory_retrieve` | Retrieve a specific memory entry by id. |
| `headroom_retrieve` | Retrieve a specific conversation turn by id. |
| `memory_consolidate` | Trigger memory consolidation for low-confidence or heavily co-occurring entries. |

## Configuration

Lives in `$HERMES_HOME/config.yaml` under `plugins.hexus` — every value optional, sensible defaults shown:

```yaml
plugins:
  hexus:
    dsn: "dbname=hermes_memory user=hermes host=/var/run/postgresql"
    embed_url: null
    embed_model: "sentence-transformers/all-MiniLM-L6-v2"
    prefetch_limit: 5
    min_similarity: 0.30
    embed_on_write: true
    scope_default: "current"
    write_queue_maxsize: 256
    bulk_sync_on_init: true
    sync_turns: true
    turn_min_chars: 40
    embed_eager_load: false
```

### MCP Server Environment Variables

- `HEXUS_DSN`: Postgres DSN connection string.
- `HEXUS_AGENT_IDENTITY`: Default agent identity (default: `default`).
- `HEXUS_CLEANUP_INTERVAL_HOURS`: Background cleanup schedule interval in hours (default: `24`).
- `HEXUS_CLEANUP_MEMORIES_TTL_DAYS`: Delete memory entries older than N days.
- `HEXUS_CLEANUP_CONVERSATIONS_TTL_DAYS`: Delete conversation turns older than N days.
- `HEXUS_CLEANUP_DELEGATIONS_TTL_DAYS`: Delete agent delegations older than N days.
- `HEXUS_DECAY_HALF_LIFE_DAYS`: Default decay half-life in days for memory search.
- `HEXUS_RECALL_BOOST_WEIGHT`: Default recall boost factor for search.
- `HEXUS_CONSOLIDATION_INTERVAL_HOURS`: Background consolidation schedule interval in hours (default: `12`).
- `HEXUS_SUMMARY_MODEL`: LLM model name used for memory reflection/consolidation.
- `LLM_API_BASE`: LLM API base endpoint.
- `LITELLM_MASTER_KEY` / `HEADROOM_INTERNAL_TOKEN`: Authentication token for the LLM endpoint.
- `HEXUS_VECTOR_PRECISION`: Vector quantization setting (`float32`, `float16`/`half`, `binary`).

## Admin Initialization

Once installed, run these commands to prepare your DB:

```bash
# Apply the schema migration (CREATE EXTENSION needs superuser)
sudo -u postgres psql -d <your-memory-db> \
     -f ~/.hermes/plugins/hexus/migrations/001_schema.sql

# Hand ownership of the new tables to the hermes runtime role
sudo -u postgres psql -d <your-memory-db> -c "
ALTER TABLE memory_entries OWNER TO hermes;
ALTER SEQUENCE memory_entries_id_seq OWNER TO hermes;
ALTER TABLE conversations OWNER TO hermes;
ALTER SEQUENCE conversations_id_seq OWNER TO hermes;
"

# Activate
hermes config set memory.provider hexus
sudo systemctl restart hermes.service
```
