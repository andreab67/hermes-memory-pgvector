-- 003_hybrid_search_fts.sql — v0.4.1 full-text search indexes for hybrid recall.
--
-- ADDITIVE ONLY. Never edit 001_schema.sql / 002_agent_attribution.sql. Idempotent
-- (IF NOT EXISTS everywhere); safe to re-run. Apply once as DB admin, AFTER 001:
--   sudo -u postgres psql -d hermes_memory -f 003_hybrid_search_fts.sql
--
-- ===========================================================================
-- WHY
-- ===========================================================================
-- The HNSW vector index (001) gives semantic recall, but pure cosine similarity
-- misses exact-lexical matches — a query for a specific token (an error code, a
-- hostname, a flag name, a rare identifier) that the embedding smooths away.
-- These GIN indexes back a Postgres full-text ranking that the store fuses with
-- the vector ranking via Reciprocal Rank Fusion (see store.hybrid_search).
--
-- NO new tables, NO new columns, NO LLM, NO entity graph — this is purely an
-- index over the EXISTING `content` column. It stays inside invariant #1
-- (storage layer, not a memory model): FTS is a second index over the same
-- text, not a parallel ontology.
--
-- ===========================================================================
-- LIVE-APPLY SAFETY  (no fleet downtime)
-- ===========================================================================
-- A plain CREATE INDEX takes a SHARE lock that blocks writes for the duration
-- of the build. On the current table sizes (thousands of rows) that is
-- sub-second, so it is applied plain here — this file is run in one shot by
-- apply_migration_as_admin(), and CREATE INDEX CONCURRENTLY cannot run inside
-- the implicit multi-statement transaction that path uses (it would raise
-- "CREATE INDEX CONCURRENTLY cannot run inside a transaction block").
--
-- If you are applying this against a large, hot table and cannot take the brief
-- write lock, build the two indexes CONCURRENTLY by hand instead, one statement
-- per connection (outside any transaction), then re-run this file — the
-- IF NOT EXISTS guards make it a no-op for the already-built indexes:
--   CREATE INDEX CONCURRENTLY ix_memory_entries_content_fts
--     ON memory_entries USING gin (to_tsvector('english', content));
--   CREATE INDEX CONCURRENTLY ix_conversations_content_fts
--     ON conversations USING gin (to_tsvector('english', content));
--
-- No GRANT is needed: the runtime `hermes` role already holds SELECT on both
-- tables, and to_tsvector / websearch_to_tsquery are built-ins. No REINDEX.


-- Full-text index over durable memory entries. Matches the store's
-- to_tsvector('english', content) @@ websearch_to_tsquery('english', $1) probe.
CREATE INDEX IF NOT EXISTS ix_memory_entries_content_fts
  ON memory_entries USING gin (to_tsvector('english', content));

-- Full-text index over conversation turns (recall_conversation hybrid path).
CREATE INDEX IF NOT EXISTS ix_conversations_content_fts
  ON conversations USING gin (to_tsvector('english', content));
