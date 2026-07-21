-- 004_runtime_grants.sql — v0.4.2 runtime-role DML grants for the 001 tables.
--
-- ADDITIVE ONLY. Never edit 001/002/003. Idempotent (GRANT is); safe to re-run.
-- Apply once as DB admin, AFTER 001:
--   sudo -u postgres psql -d hermes_memory -f 004_runtime_grants.sql
--
-- ===========================================================================
-- WHY
-- ===========================================================================
-- 001 creates memory_entries + conversations with NO grants: the runtime role
-- only gained access through a MANUAL "ALTER TABLE ... OWNER TO hermes" step
-- documented in the README — a step `hermes-pgvector migrate` never ran. An
-- operator following the CLI path alone got "permission denied for table
-- memory_entries" on every plugin write (002 grants covered only the v0.4
-- provenance tables). This migration closes that gap so `migrate` alone
-- yields a fully-functional install.
--
-- GRANTs (not ownership transfer): the runtime role needs DML + sequence
-- usage, nothing more — ownership stays with the admin role that ran the
-- migration, consistent with the admin/runtime DDL split (invariant #5).
--
-- The role name 'hermes' matches this plugin's default DSN and migration
-- 002's existing GRANTs. Deployments using a different runtime role get a
-- NOTICE and must grant manually (same as they already had to for 002).

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hermes') THEN
    GRANT USAGE ON SCHEMA public TO hermes;
    GRANT SELECT, INSERT, UPDATE, DELETE ON memory_entries, conversations TO hermes;
    GRANT USAGE, SELECT ON SEQUENCE memory_entries_id_seq, conversations_id_seq TO hermes;
  ELSE
    RAISE NOTICE 'role "hermes" not found — grant DML on memory_entries/conversations (and USAGE on their id sequences) to your runtime role manually';
  END IF;
END
$$;
