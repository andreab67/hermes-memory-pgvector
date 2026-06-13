-- 001_schema.sql — hexus memory plugin schema.
--
-- Forked from andreab67/hermes-hexus (BSD-3-Clause).
--
-- TWO tables, both in the existing hermes_memory database:
--   memory_entries  → mirrors hermes-agent's built-in `memory` tool
--                     (MEMORY.md / USER.md from tools/memory_tool.py)
--   conversations   → every (user, assistant) chat turn, semantic-searchable
--                     for cross-session recall of "what did we talk about"
--
-- Both scoped per agent_identity (marketing / sales / trading / incident / …),
-- both with 384-dim embeddings, both with HNSW indexes tuned the same way as
-- hermes_memory.events.
--
-- Apply once:
--   sudo -u postgres psql -d hermes_memory -f 001_schema.sql
-- Idempotent (CREATE IF NOT EXISTS everywhere); safe to re-run.
--
-- v0.4.0 (hexus fork): dim was 768 (nomic-embed-text via HTTP).
-- Now 384 (sentence-transformers all-MiniLM-L6-v2 local). The schema is
-- rewritten in place; existing 768-dim installations need to re-embed all
-- rows before the dim will line up (see README §Migration from v0.3.x).

CREATE EXTENSION IF NOT EXISTS vector;


-- ---------------------------------------------------------------------------
-- memory_entries
-- One row per add/replace from hermes-agent's built-in `memory` tool,
-- mirrored via the on_memory_write hook. Hard-deleted on `remove`.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory_entries (
  id              BIGSERIAL PRIMARY KEY,

  -- Per-agent theme. Maps to hermes-agent's `agent_identity` profile
  -- (passed to MemoryProvider.initialize via kwargs).
  -- Examples: 'marketing', 'sales', 'trading', 'incident', 'default'.
  agent_identity  TEXT NOT NULL DEFAULT 'default',

  -- Mirrors hermes-agent's two built-in stores:
  --   'memory' → MEMORY.md (env facts, project conventions, tool quirks)
  --   'user'   → USER.md   (about the user: name, role, preferences)
  target          TEXT NOT NULL CHECK (target IN ('memory', 'user')),

  content         TEXT NOT NULL,
  embedding       vector(384),

  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- Provenance from on_memory_write kwargs: session_id, platform,
  -- write_origin, tool_name, parent_session_id, etc.
  metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,

  -- Built-in tool dedupes on (target, exact content match). We replicate
  -- that within a single agent's scope so add-of-existing is a no-op.
  CONSTRAINT memory_entries_unique
    UNIQUE (agent_identity, target, content)
);

-- Per-agent + target listing (the most common scan).
CREATE INDEX IF NOT EXISTS ix_memory_entries_agent_target
  ON memory_entries (agent_identity, target, updated_at DESC);

-- Semantic recall: SELECT … ORDER BY embedding <=> $1 LIMIT $2
-- Cosine distance via HNSW — same tuning as hermes_memory.events.
CREATE INDEX IF NOT EXISTS ix_memory_entries_embedding_hnsw
  ON memory_entries USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Full-text search index for hybrid search
CREATE INDEX IF NOT EXISTS ix_memory_entries_content_tsvector
  ON memory_entries USING gin (to_tsvector('english', content));


-- ---------------------------------------------------------------------------
-- conversations
-- One row per substantive chat turn, captured via sync_turn(). Short /
-- boilerplate turns ("ok", "thanks", < 40 chars) are filtered out at
-- the plugin layer before INSERT; what lands here is recall-worthy.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS conversations (
  id              BIGSERIAL PRIMARY KEY,
  session_id      TEXT NOT NULL,
  agent_identity  TEXT NOT NULL DEFAULT 'default',
  role            TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
  content         TEXT NOT NULL,
  ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
  embedding       vector(384),
  metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Per-session timeline (the "recent turns in this conversation" path).
CREATE INDEX IF NOT EXISTS ix_conversations_session_ts
  ON conversations (session_id, ts DESC);

-- Per-agent timeline (the "what's marketing been doing lately" path).
CREATE INDEX IF NOT EXISTS ix_conversations_agent_ts
  ON conversations (agent_identity, ts DESC);

-- Semantic recall across all turns. Same HNSW tuning as memory_entries.
CREATE INDEX IF NOT EXISTS ix_conversations_embedding_hnsw
  ON conversations USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Full-text search index for hybrid search
CREATE INDEX IF NOT EXISTS ix_conversations_content_tsvector
  ON conversations USING gin (to_tsvector('english', content));
