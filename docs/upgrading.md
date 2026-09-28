# Upgrading

Version-to-version upgrade procedures. See [CHANGELOG.md](../CHANGELOG.md)
for the full detail behind each change referenced here.

## 0.5.x → 0.6.0

Read this whole section before upgrading — the ordering in step 2 matters.

1. **Set `embed_url` explicitly if you rely on the old default.** It
   changed from a private LAN address to `http://localhost:11434`. If your
   `config.yaml` already sets `embed_url`, this step is a no-op.

2. **Upgrade the package on every host that writes to this database
   FIRST, then run `migrate`.** Migration `005` drops
   `memory_entries_unique`, the constraint a pre-0.6.0 plugin's
   `ON CONFLICT (agent_identity, target, content)` names as its conflict
   target. Running `migrate` while any writer is still on an older package
   version breaks that writer's `add()` calls with "there is no unique or
   exclusion constraint matching the ON CONFLICT specification" (see
   [troubleshooting.md](troubleshooting.md)). Concretely:

   ```bash
   # On EVERY host running this plugin:
   pip install -U hermes-memory-pgvector   # 0.6.0 or newer
   sudo systemctl restart hermes.service   # or however you run hermes

   # THEN, once every writer is upgraded, on the admin/DB host:
   hermes-pgvector migrate --admin-dsn "dbname=<db> user=postgres host=..." \
       --runtime-role <name-if-not-hermes>
   ```

3. **Find and remap mixed-case themes.** Identities are now case-folded
   (`Marketing` and `marketing` are the same theme going forward). Existing
   rows are not retroactively rewritten by `migrate` — find any pre-existing
   mixed-case identities and fold them explicitly:

   ```sql
   SELECT DISTINCT agent_identity FROM memory_entries
    WHERE agent_identity <> lower(agent_identity);
   SELECT DISTINCT agent_identity FROM conversations
    WHERE agent_identity <> lower(agent_identity);
   ```

   For each one found:

   ```bash
   hermes-pgvector remap --old Marketing --new marketing --execute
   ```

4. **Check `write_contexts` if you rely on non-primary/cron writes.** The
   new default (`primary,cron`) skips writes from `subagent` and `flush`
   agent contexts. Delegated subagent output already reaches the parent
   theme via `on_delegation`, so this is a no-op for most deployments — but
   if you have a custom integration that writes directly under a
   `subagent`/`flush` context and expects it to land, add it to
   `write_contexts` (comma-separated) in `config.yaml`.

5. **Update backfill timers/alerting for the new exit codes.** `0` = done,
   `2` = endpoint unavailable / a table's pass aborted, `3` = rows still
   remain / failed with no abort. Previously `backfill` always exited `0`.
   See [operations.md](operations.md#backfill).

6. **If anything reads `conversations.parent_session_id` on delegation
   rows directly** (rather than through `on_delegation`/`recall_conversation`),
   note it now holds the *delegating* session's id, matching migration
   `002`'s original column definition — it previously held the *child*
   session id, the reverse. The child session id is unchanged in
   `metadata.child_session_id` and in `memory_agent_edges`.

7. **If anything reads `health()`'s row count programmatically**, the key
   is now `row_count_estimate` (an approximation), not `row_count` (an
   exact count). This is visible in `hermes-pgvector stats` output.

That's the whole procedure — there are no other schema or behaviour changes
that require operator action.

## 0.4.x → 0.5.0 (import rename, still relevant if you skipped it)

If you're upgrading directly from something older than 0.5.0, you also need
this step, folded into the sequence above (do it before step 2):

The import package renamed `pgvector` → `hermes_pgvector`. Any
`python -m pgvector ...` command (most commonly a scheduled backfill job)
breaks silently and must become `hermes-pgvector ...` /
`python -m hermes_pgvector ...`:

```bash
sudo grep -rl 'python -m pgvector' /etc/systemd/system/ /etc/cron.d/ 2>/dev/null
sudo systemctl daemon-reload   # after editing any matching unit
```

If you use the discovery shim (Option 1 install), an existing shim from
before 0.5.0 imports `from pgvector import ...`, which fails after
upgrading — the loader treats that as "plugin absent" and falls back to
built-in memory. Regenerate or remove it:

```bash
pip install -U hermes-memory-pgvector

# Preferred if your hermes-agent reads the hermes_agent.memory_providers
# entry point (declared since 0.5.0): drop the shim, let pip discovery
# take over.
hermes-pgvector install --remove

# Otherwise (an older host that only scans plugin directories):
hermes-pgvector install --force

sudo systemctl restart hermes.service
hermes memory status   # expect: Provider: pgvector; Status: available -- do not skip this check
```

`hermes-pgvector install` verifies in a clean subprocess that the shim it
just wrote can actually be imported, and warns if it cannot or if another
package shadows this one.

## Changing the embedding dimension

`embed_dim` has to agree with the `vector(N)` columns, so switching to a
model with a different output size is a migration, not a config edit.
Vectors from two different models are not comparable, so every row must be
re-embedded. Until config and columns agree, Postgres rejects each write
whose vector has the wrong length, and the row is lost outright (not stored
text-only) — stop the services first.

The shipped migration files are not meant to be edited for this. As the
table owner:

```sql
-- 1. With every hermes service that loads the provider stopped:
BEGIN;
DROP INDEX IF EXISTS ix_memory_entries_embedding_hnsw;
DROP INDEX IF EXISTS ix_conversations_embedding_hnsw;
ALTER TABLE memory_entries ALTER COLUMN embedding TYPE vector(1536) USING NULL::vector(1536);
ALTER TABLE conversations  ALTER COLUMN embedding TYPE vector(1536) USING NULL::vector(1536);
COMMIT;
```

2. Set `embed_model` and `embed_dim` (plus `embed_url`, `embed_api_key_env`,
   `embed_protocol` as needed), then start the services. New writes are
   embedded with the new model.
3. Re-embed the existing rows, which are all `NULL` now:
   `hermes-pgvector backfill --config $HERMES_HOME/config.yaml`. Repeat
   until every table reports `remaining: 0`. If the endpoint does not
   return `embed_dim`-length vectors, the run aborts on its first probe,
   before touching any row.
4. Rebuild the HNSW indexes with the shipped tuning (see
   [scaling.md](scaling.md#hnsw-index-tuning) for when to use different
   `m`/`ef_construction`). Building them after the backfill is faster than
   maintaining them during it:

```sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_memory_entries_embedding_hnsw
  ON memory_entries USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_conversations_embedding_hnsw
  ON conversations USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```

Until step 3 completes, rows without a vector are reachable only through
full-text recall (`hybrid_search: true`). pgvector's HNSW index supports
`vector` columns of up to 2,000 dimensions. This plugin maintains only
`memory_entries` and `conversations`; any other embedding columns in the
same database need the same change from whatever writes them.

## Older releases

Every release from `v0.3.0` onward has its own entry in
[CHANGELOG.md](../CHANGELOG.md) with the behaviour change and, where one was
needed, the upgrade note. None of `v0.3.1` through `v0.4.3` or `v0.5.1`
through `v0.5.5` required operator action beyond `pip install -U` +
restart + (for `v0.4.0`) applying migration `002`.
