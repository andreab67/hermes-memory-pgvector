-- 002_agent_attribution.sql — v0.4.0 agent attribution + delegation provenance.
--
-- ADDITIVE ONLY. Never edit 001_schema.sql. Idempotent (IF NOT EXISTS everywhere);
-- safe to re-run. Apply once as DB admin (superuser/owner), AFTER 001:
--   sudo -u postgres psql -d hermes_memory -f 002_agent_attribution.sql
--
-- ===========================================================================
-- OWNERSHIP BOUNDARY  (hard architectural line — invariants #1 and #5)
-- ===========================================================================
-- This migration creates ONLY hermes-owned PROVENANCE tables. It MUST NOT
-- reference, FK, trigger, or view the postgres-owned events / entities /
-- relations tables (those are created by hermes-vps/scripts/memory/
-- memory-synthesize.py; relations FKs to events). Parent->child delegation
-- links use parent_session_id TEXT — a string reference to
-- conversations.session_id — NEVER an FK to events(id).
--
-- These tables are PROVENANCE ONLY ("which agent, which memory, who delegated
-- to whom, when"). Do NOT add confidence_score / derived_insight / entity-link
-- columns: that is a fact-store ontology and belongs in Holographic, not here.
--
-- ===========================================================================
-- LIVE-APPLY SAFETY  (no fleet downtime)
-- ===========================================================================
-- The only change to an existing table is:
--     ALTER TABLE conversations ADD COLUMN IF NOT EXISTS parent_session_id TEXT;
-- On PostgreSQL 18, adding a NULLable column with NO DEFAULT is a catalog-only
-- change (~10-20ms, SHARE UPDATE EXCLUSIVE). It does NOT rewrite the heap and
-- does NOT invalidate the HNSW index on the unchanged `embedding` column.
-- DO NOT REINDEX as part of this migration — a REINDEX takes ACCESS EXCLUSIVE
-- and would block the writing fleet. Take a pg_dump before applying.


-- ---------------------------------------------------------------------------
-- memory_agents — canonical registry of agent identities the plugin has seen.
-- One row per (normalized) agent_identity, upserted on each session init.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory_agents (
  id              BIGSERIAL PRIMARY KEY,
  agent_identity  TEXT NOT NULL UNIQUE,
  -- theme | worker | user | dm | bench | default  (see identity.classify_kind)
  kind            TEXT NOT NULL DEFAULT 'theme',
  first_seen      TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_seen       TIMESTAMPTZ NOT NULL DEFAULT now(),
  attrs           JSONB NOT NULL DEFAULT '{}'::jsonb
);


-- ---------------------------------------------------------------------------
-- memory_agent_edges — parent->child delegation / relationship provenance.
-- parent_session_id / child_session_id are STRING refs to conversations.session_id
-- (NOT foreign keys — and never to events(id), which is owned by another system).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory_agent_edges (
  id                BIGSERIAL PRIMARY KEY,
  parent_identity   TEXT,
  child_identity    TEXT,
  parent_session_id TEXT,
  child_session_id  TEXT,
  -- delegated | spawned | session
  kind              TEXT NOT NULL DEFAULT 'delegated',
  ts                TIMESTAMPTZ NOT NULL DEFAULT now(),
  attrs             JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_memory_agent_edges_parent
  ON memory_agent_edges (parent_identity, ts DESC);
CREATE INDEX IF NOT EXISTS ix_memory_agent_edges_child
  ON memory_agent_edges (child_identity, ts DESC);
CREATE INDEX IF NOT EXISTS ix_memory_agent_edges_psession
  ON memory_agent_edges (parent_session_id);


-- ---------------------------------------------------------------------------
-- conversations — add parent-session linkage for delegation traceback.
-- NULLable, no default => catalog-only change; HNSW index stays valid. No REINDEX.
-- ---------------------------------------------------------------------------
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS parent_session_id TEXT;


-- ---------------------------------------------------------------------------
-- memory_maintenance_log — audit trail for destructive maintenance ops
-- (cleanup deletes, identity remaps, TTL prunes, backfills). Hermes-owned.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory_maintenance_log (
  id               BIGSERIAL PRIMARY KEY,
  ts               TIMESTAMPTZ NOT NULL DEFAULT now(),
  operation        TEXT NOT NULL,   -- cleanup_delete | remap_identity | prune | backfill
  target_table     TEXT,
  identity_pattern TEXT,
  affected_count   INTEGER,
  dropped_dupes    INTEGER,
  dry_run          BOOLEAN NOT NULL DEFAULT true,
  details          JSONB NOT NULL DEFAULT '{}'::jsonb
);


-- ---------------------------------------------------------------------------
-- v_agent_memory — per-identity attribution across the TWO plugin-owned tables.
-- Reads ONLY memory_entries + conversations + memory_agents. Never touches the
-- postgres-owned events / entities / relations tables.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_agent_memory AS
WITH ids AS (
  SELECT agent_identity FROM memory_agents
  UNION SELECT agent_identity FROM memory_entries
  UNION SELECT agent_identity FROM conversations
)
SELECT
  i.agent_identity,
  a.kind,
  (SELECT count(*) FROM memory_entries m WHERE m.agent_identity = i.agent_identity) AS memory_rows,
  (SELECT count(*) FROM conversations c WHERE c.agent_identity = i.agent_identity)  AS conversation_rows,
  a.first_seen,
  a.last_seen
FROM ids i
LEFT JOIN memory_agents a ON a.agent_identity = i.agent_identity
ORDER BY i.agent_identity;


-- ---------------------------------------------------------------------------
-- GRANTs — runtime role 'hermes' gets DML on the NEW tables ONLY.
-- Scoped explicitly to these objects: never GRANT ... ON ALL TABLES/SEQUENCES,
-- which would leak privileges onto the postgres-owned events/entities/relations.
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA public TO hermes;
GRANT SELECT, INSERT, UPDATE, DELETE
  ON memory_agents, memory_agent_edges, memory_maintenance_log TO hermes;
GRANT USAGE, SELECT
  ON SEQUENCE memory_agents_id_seq, memory_agent_edges_id_seq, memory_maintenance_log_id_seq TO hermes;
GRANT SELECT ON v_agent_memory TO hermes;
