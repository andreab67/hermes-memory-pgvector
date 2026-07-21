"""store.py — Postgres ops for the pgvector memory plugin.

Wraps psycopg3 + psycopg_pool. Mirrors hermes-agent's native built-in
memory model (`memory` tool's add/replace/remove on targets 'memory' /
'user') into a single Postgres table with embeddings.

Uses a small ConnectionPool because the plugin is touched from two
threads at runtime: the agent thread (for prefetch / recall_memory /
ensure_schema / health) and the async-writer drain thread (for the
mirrored INSERTs / UPDATEs / DELETEs). Pooling beats short-lived
connections under that two-thread pattern without adding much
complexity.

No SQLAlchemy, no LLM-mediated workers, no deriver loops.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .embed import to_pgvector_literal

logger = logging.getLogger(__name__)


def _escape_like(text: str) -> str:
    """Escape LIKE metacharacters so `text` matches as a literal substring.

    The built-in memory tool matches old_text with plain Python `in` — no
    wildcards. Postgres LIKE treats %/_ as wildcards and backslash as the
    default escape character, so an unescaped pattern both over-matches
    ('15% YoY' matches '15<anything>YoY' and can hit unrelated rows) and
    under-matches (backslashes in stored Windows paths are consumed as
    escapes, so the row that literally contains them never matches).
    Escaping all three restores exact-substring semantics (v0.4.2).
    """
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class MemoryStore:
    """Postgres-backed mirror of hermes-agent's built-in memory entries."""

    # Tables this plugin OWNS and is allowed to mutate in bulk-maintenance ops
    # (backfill / prune / cleanup / remap). The postgres-owned events / entities
    # / relations tables (created by hermes-vps memory-synthesize.py) are NOT in
    # this set and every maintenance method asserts membership before touching a
    # table — a hard guardrail against corrupting the synthesis layer (v0.4.0).
    CLEANUP_WHITELIST = ("memory_entries", "conversations")

    # Reciprocal Rank Fusion constant. The standard k=60 from the original RRF
    # paper (Cormack et al., 2009) — it dampens low-ranked items so a row only
    # floats to the top when BOTH rankers (vector + full-text) agree, while a
    # strong hit in either ranker alone still surfaces. Not tuning-sensitive.
    RRF_K = 60

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = 0,
        max_size: int = 4,
        timeout: float = 5.0,
        max_idle: float = 30.0,
        max_lifetime: float = 300.0,
    ):
        """Open a lazily-initialized, self-draining ConnectionPool.

        min_size=0 means an idle pool holds ZERO connections — critical so
        a pool that gets abandoned (a re-initialized provider, or a session
        the gateway never explicitly shuts down) cannot strand a warm
        backend in Postgres until the server's idle_session_timeout reaps
        it. Under load the pool still grows to max_size=4 so the agent
        thread and the async-writer drain thread can overlap.

        max_idle (30s) closes connections returned to the pool that then sit
        unused, shrinking back toward min_size. max_lifetime (300s) caps the
        absolute age of any pooled connection. Together these keep the
        connections "short-lived when idle, pooled under load" and bound the
        plugin's Postgres footprint to actual concurrent demand rather than
        to the number of sessions ever opened.
        """
        self._dsn = dsn
        self._lock = threading.Lock()
        self._pool: Optional[ConnectionPool] = None
        self._min_size = min_size
        self._max_size = max_size
        self._timeout = timeout
        self._max_idle = max_idle
        self._max_lifetime = max_lifetime

    # -- Pool lifecycle ------------------------------------------------------

    def _get_pool(self) -> ConnectionPool:
        """Return the live pool, constructing it on first call. Thread-safe."""
        if self._pool is not None:
            return self._pool
        with self._lock:
            if self._pool is None:
                self._pool = ConnectionPool(
                    conninfo=self._dsn,
                    min_size=self._min_size,
                    max_size=self._max_size,
                    timeout=self._timeout,
                    max_idle=self._max_idle,
                    max_lifetime=self._max_lifetime,
                    open=True,
                    name="pgvector-memory",
                )
        return self._pool

    def close(self) -> None:
        """Close the connection pool. Idempotent."""
        with self._lock:
            if self._pool is not None:
                try:
                    self._pool.close()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("pgvector pool close: %s", exc)
                finally:
                    self._pool = None

    # -- Schema --------------------------------------------------------------

    class SchemaNotApplied(RuntimeError):
        """Raised when memory_entries does not exist in the target DB."""

    def ensure_schema(self) -> None:
        """Verify the schema is in place. Does NOT run DDL.

        The migration (migrations/001_schema.sql) is admin-only — it
        runs `CREATE EXTENSION vector` which requires superuser, and
        creates the table + indexes which then end up owned by the
        admin role. The plugin's runtime user (hermes) only has
        SELECT/INSERT/UPDATE/DELETE on the existing schema, and that's
        the right separation: DDL at install time, DML at run time.

        Operators apply the migration once via:
            sudo -u postgres psql -d hermes_memory -f migrations/001_schema.sql
        """
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('memory_entries')")
                if cur.fetchone()[0] is None:
                    raise self.SchemaNotApplied(
                        "memory_entries table missing. Apply migrations as DB admin: "
                        "hermes-pgvector migrate --admin-dsn 'dbname=<db> user=postgres "
                        "host=/var/run/postgresql' (or psql -f the files in the "
                        "installed package's pgvector/migrations/ directory, in order)"
                    )

    def apply_migration_as_admin(self, *, admin_dsn: str, migration: str = "001_schema.sql") -> None:
        """One-shot admin path: run a single migration with privileged creds.

        Bypasses the runtime pool — opens a fresh autocommit connection
        with admin_dsn (typically `user=postgres host=/var/run/postgresql`)
        so CREATE EXTENSION + CREATE TABLE + CREATE INDEX + GRANT all succeed.
        Idempotent: every migration uses IF NOT EXISTS, so re-running on an
        already-migrated DB is a no-op. `migration` selects the file under
        migrations/ (default 001_schema.sql; v0.4.0 adds 002_agent_attribution.sql).
        """
        sql_path = Path(__file__).parent / "migrations" / migration
        sql = sql_path.read_text(encoding="utf-8")
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)

    def apply_all_migrations(self, *, admin_dsn: str) -> List[str]:
        """Apply every migrations/*.sql in lexical order. Returns the names run.

        Lexical order (001_, 002_, …) is the apply order. All migrations are
        additive + idempotent, so this is safe to run repeatedly.
        """
        mig_dir = Path(__file__).parent / "migrations"
        applied: List[str] = []
        for p in sorted(mig_dir.glob("*.sql")):
            self.apply_migration_as_admin(admin_dsn=admin_dsn, migration=p.name)
            applied.append(p.name)
        return applied

    def ensure_migration_002_applied(self) -> bool:
        """Verify-only check for the v0.4.0 (002) agent-attribution objects.

        Returns True iff memory_agents + memory_agent_edges exist AND
        conversations.parent_session_id exists. Deliberately SEPARATE from
        ensure_schema() (which stays 001-only): a v0.4.0 binary running on a
        v0.3 schema must NOT crash at startup — instead the provider disables
        the delegation/attribution hooks until an admin applies 002. Never
        raises (returns False on any error)."""
        try:
            with self._get_pool().connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT to_regclass('memory_agents'), to_regclass('memory_agent_edges')"
                    )
                    agents, edges = cur.fetchone()
                    if agents is None or edges is None:
                        return False
                    cur.execute(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_name = 'conversations' AND column_name = 'parent_session_id'"
                    )
                    return cur.fetchone() is not None
        except Exception as exc:  # noqa: BLE001
            logger.debug("ensure_migration_002_applied probe failed: %s", exc)
            return False

    # -- Built-in memory mirror (called by on_memory_write) ------------------

    def add(
        self,
        *,
        agent_identity: str,
        target: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Insert a memory entry. Returns row id, or None if duplicate (no-op).

        Matches the built-in tool's "reject exact duplicate" semantics via
        the (agent_identity, target, content) unique constraint + ON CONFLICT.
        """
        meta_json = json.dumps(metadata or {})
        vec_literal = to_pgvector_literal(embedding) if embedding is not None else None

        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_entries
                        (agent_identity, target, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s::vector, %s::jsonb)
                    ON CONFLICT (agent_identity, target, content) DO NOTHING
                    RETURNING id
                    """,
                    (agent_identity, target, content, vec_literal, meta_json),
                )
                row = cur.fetchone()
                conn.commit()
                return int(row[0]) if row else None

    def replace(
        self,
        *,
        agent_identity: str,
        target: str,
        old_text: str,
        new_content: str,
        new_embedding: Optional[List[float]] = None,
    ) -> int:
        """Update entries in (agent_identity, target) where content contains old_text.

        Matches built-in semantics — old_text is a substring match. Returns
        the number of rows updated (built-in updates the FIRST match; we
        update all matches in the same scope for safety).
        """
        vec_literal = (
            to_pgvector_literal(new_embedding) if new_embedding is not None else None
        )
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE memory_entries
                       SET content    = %s,
                           embedding  = %s::vector,
                           updated_at = now()
                     WHERE agent_identity = %s
                       AND target = %s
                       AND content LIKE %s
                    """,
                    (new_content, vec_literal, agent_identity, target,
                     f"%{_escape_like(old_text)}%"),
                )
                updated = cur.rowcount
                conn.commit()
                return int(updated)

    def remove(
        self,
        *,
        agent_identity: str,
        target: str,
        old_text: str,
    ) -> int:
        """Delete entries in (agent_identity, target) matching old_text substring.

        Returns the number of rows deleted.
        """
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM memory_entries
                     WHERE agent_identity = %s
                       AND target = %s
                       AND content LIKE %s
                    """,
                    (agent_identity, target, f"%{_escape_like(old_text)}%"),
                )
                deleted = cur.rowcount
                conn.commit()
                return int(deleted)

    # -- Reads ---------------------------------------------------------------

    def list_entries(
        self,
        *,
        agent_identity: str,
        target: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List entries in an agent's scope. If target is None, both stores."""
        params: List[Any] = [agent_identity]
        target_clause = ""
        if target:
            target_clause = "AND target = %s"
            params.append(target)
        params.append(limit)

        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT id, agent_identity, target, content, created_at, updated_at, metadata
                    FROM memory_entries
                    WHERE agent_identity = %s
                    {target_clause}
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    params,
                )
                return list(cur.fetchall())

    def search(
        self,
        *,
        query_embedding: List[float],
        agent_identity: Optional[str] = None,
        target: Optional[str] = None,
        limit: int = 5,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Semantic recall via cosine distance.

        agent_identity=None → search across ALL agents (cross-theme recall).
        target=None → search both 'memory' and 'user'.
        Returns rows with `score` = 1 - cosine_distance ∈ [0, 1].
        """
        vec_literal = to_pgvector_literal(query_embedding)
        clauses: List[str] = []
        params: List[Any] = []
        if agent_identity:
            clauses.append("agent_identity = %s")
            params.append(agent_identity)
        if target:
            clauses.append("target = %s")
            params.append(target)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT id, agent_identity, target, content, created_at,
                           updated_at, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM memory_entries
                    {where}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    [vec_literal, *params, vec_literal, limit],
                )
                rows = list(cur.fetchall())

        if min_similarity > 0:
            rows = [r for r in rows if (r.get("score") or 0) >= min_similarity]
        return rows

    def hybrid_search(
        self,
        *,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        agent_identity: Optional[str] = None,
        target: Optional[str] = None,
        limit: int = 5,
        pool: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid recall over memory_entries: vector + full-text, fused by RRF.

        Fuses the HNSW cosine ranking with a Postgres full-text ranking
        (websearch_to_tsquery / ts_rank_cd) via Reciprocal Rank Fusion, so a
        row surfaces if EITHER ranker likes it. This recovers exact-lexical
        hits — rare tokens, identifiers, flags, error codes — that pure cosine
        similarity smooths away, and, because the full-text leg has no
        embedding requirement, also recovers text-only rows whose embedding is
        still NULL (embed endpoint was down at write time) that the pure-vector
        search() cannot see.

        query_embedding=None → full-text only (used when the query itself failed
        to embed; recall degrades to lexical instead of erroring). query_text is
        always required — an empty query returns [].

        `pool` is the per-ranker candidate depth fused before the final cut
        (default max(limit*4, 40)). Returns rows shaped like search() plus an
        `rrf_score`; `score` is the cosine similarity, or None for a full-text-
        only row that has no embedding. There is no min_similarity: RRF ranks,
        it does not threshold — filter on `score` downstream if a floor matters.
        """
        if not query_text or not query_text.strip():
            return []
        if pool is None:
            pool = max(int(limit) * 4, 40)

        params: Dict[str, Any] = {
            "qtext": query_text, "k": self.RRF_K, "pool": pool, "limit": limit,
        }
        filters: List[str] = []
        if agent_identity:
            filters.append("agent_identity = %(agent)s")
            params["agent"] = agent_identity
        if target:
            filters.append("target = %(target)s")
            params["target"] = target
        extra = (" AND " + " AND ".join(filters)) if filters else ""

        sql = self._build_hybrid_sql(
            table="memory_entries",
            extra_filter=extra,
            has_vector=query_embedding is not None,
            columns="m.id, m.agent_identity, m.target, m.content, "
                    "m.created_at, m.updated_at, m.metadata",
        )
        if query_embedding is not None:
            params["qvec"] = to_pgvector_literal(query_embedding)

        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())

    @staticmethod
    def _build_hybrid_sql(
        *, table: str, extra_filter: str, has_vector: bool, columns: str
    ) -> str:
        """Assemble the RRF hybrid query for `table`.

        `table` is a trusted literal (never user input — only 'memory_entries'
        / 'conversations' from our own call sites). The vec CTE runs the ANN
        `ORDER BY embedding <=> q LIMIT pool` in a subquery and assigns ranks
        with ROW_NUMBER() OVER () *outside* that limit, so the HNSW index still
        serves the top-`pool` scan — a window ORDER BY on the distance would
        force an exact full sort and defeat the index. When has_vector is False
        the vec CTE is omitted and fusion runs on the full-text leg alone.
        """
        vec_cte = ""
        score_expr = "NULL::float8 AS score"
        if has_vector:
            # Rank AFTER the ANN limit: the inner `ORDER BY embedding <=> q
            # LIMIT pool` is a plain top-`pool` KNN the HNSW index serves, and
            # ROW_NUMBER() OVER () numbers those already-ordered rows without a
            # full sort. A window ORDER BY on the distance here would instead
            # force an exact sort over every candidate row and defeat the index.
            vec_cte = f"""
                vec AS (
                    SELECT id, ROW_NUMBER() OVER () AS rnk
                    FROM (
                        SELECT id
                        FROM {table}
                        WHERE embedding IS NOT NULL{extra_filter}
                        ORDER BY embedding <=> %(qvec)s::vector
                        LIMIT %(pool)s
                    ) v
                ),"""
            score_expr = (
                "CASE WHEN m.embedding IS NULL THEN NULL "
                "ELSE 1 - (m.embedding <=> %(qvec)s::vector) END AS score"
            )
        fts_cte = f"""
                fts AS (
                    SELECT id, ROW_NUMBER() OVER (
                               ORDER BY ts_rank_cd(
                                   to_tsvector('english', content), q) DESC) AS rnk
                    FROM {table}, websearch_to_tsquery('english', %(qtext)s) q
                    WHERE to_tsvector('english', content) @@ q{extra_filter}
                    ORDER BY ts_rank_cd(to_tsvector('english', content), q) DESC
                    LIMIT %(pool)s
                )"""
        if has_vector:
            fused = """
                fused AS (
                    SELECT COALESCE(vec.id, fts.id) AS id,
                           COALESCE(1.0 / (%(k)s + vec.rnk), 0)
                         + COALESCE(1.0 / (%(k)s + fts.rnk), 0) AS rrf
                    FROM vec FULL OUTER JOIN fts ON vec.id = fts.id
                )"""
        else:
            fused = """
                fused AS (
                    SELECT fts.id AS id, 1.0 / (%(k)s + fts.rnk) AS rrf
                    FROM fts
                )"""
        return f"""
            WITH{vec_cte}{fts_cte},
                {fused}
            SELECT {columns}, {score_expr}, f.rrf AS rrf_score
            FROM fused f
            JOIN {table} m ON m.id = f.id
            ORDER BY f.rrf DESC
            LIMIT %(limit)s
        """

    # -- Bulk import from MEMORY.md / USER.md (v0.1.1) ----------------------

    # Matches tools/memory_tool.py:ENTRY_DELIMITER. Keep in sync if upstream
    # ever changes it (currently stable; been "\n§\n" since the tool shipped).
    ENTRY_DELIMITER = "\n§\n"

    def bulk_upsert_md(
        self,
        *,
        agent_identity: str,
        target: str,
        file_path: "Path | str",
        embed_fn,
    ) -> Dict[str, int]:
        """Parse a MEMORY.md / USER.md file and upsert each entry.

        Idempotent + cheap on re-run: we SELECT the existing content set
        for (agent_identity, target) once, then only embed + INSERT new
        entries. So initial install embeds everything; subsequent inits
        with no MD changes do zero embed calls.

        embed_fn is a callable taking a string and returning a 768-dim
        list (or raising — we catch and store text-only). Wired by the
        caller so the plugin can pass its `embed()` with the configured
        base_url + model.

        Returns: {'parsed': N, 'inserted': M, 'skipped': K} where N=M+K.
        """
        from pathlib import Path as _Path
        p = _Path(file_path)
        if not p.exists():
            return {"parsed": 0, "inserted": 0, "skipped": 0}

        raw = p.read_text(encoding="utf-8", errors="replace")
        entries = [e.strip() for e in raw.split(self.ENTRY_DELIMITER) if e.strip()]
        if not entries:
            return {"parsed": 0, "inserted": 0, "skipped": 0}

        # Single bulk SELECT of existing content for this scope. Beats N+1
        # by a wide margin and keeps re-init nearly free.
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content FROM memory_entries WHERE agent_identity = %s AND target = %s",
                    (agent_identity, target),
                )
                existing = {row[0] for row in cur.fetchall()}

        inserted = 0
        skipped = 0
        # Circuit breaker (v0.4.2): this loop runs SYNCHRONOUSLY inside
        # initialize(), and a hanging (not refused) embed endpoint costs up to
        # ~20s per entry (two protocol attempts × 10s timeout). Without a
        # breaker, 50 new entries would block session start for ~17 minutes.
        # After 3 consecutive failures, stop embedding and insert the rest
        # text-only — the nightly backfill re-embeds them later.
        consecutive_embed_failures = 0
        for entry in entries:
            if entry in existing:
                skipped += 1
                continue
            vec = None
            if embed_fn and consecutive_embed_failures < 3:
                try:
                    vec = embed_fn(entry)
                    consecutive_embed_failures = 0
                except Exception:  # noqa: BLE001 — fail-soft on bulk embed
                    vec = None
                    consecutive_embed_failures += 1
                    if consecutive_embed_failures == 3:
                        logger.warning(
                            "bulk_upsert_md: 3 consecutive embed failures — "
                            "inserting remaining entries text-only "
                            "(the backfill sweep will re-embed them)"
                        )
            row_id = self.add(
                agent_identity=agent_identity,
                target=target,
                content=entry,
                embedding=vec,
                metadata={"source": "bulk_import", "file": str(p)},
            )
            if row_id is not None:
                inserted += 1
            else:
                # Lost a race with another writer that inserted the same row.
                skipped += 1
        return {"parsed": len(entries), "inserted": inserted, "skipped": skipped}

    # -- Conversation turns (v0.2) ------------------------------------------

    def append_turn(
        self,
        *,
        session_id: str,
        agent_identity: str,
        role: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_session_id: Optional[str] = None,
    ) -> int:
        """Insert one chat turn. Returns row id.

        No dedup (turns are inherently time-ordered events — same content
        twice is two distinct turns, even verbatim).

        `parent_session_id` is written to the v0.4.0 conversations column ONLY
        when provided (not None). The caller (provider) passes it only when
        migration 002 has been applied (self._delegation_enabled), so this stays
        compatible with a v0.3 schema where the column does not exist.
        """
        meta_json = json.dumps(metadata or {})
        vec_literal = to_pgvector_literal(embedding) if embedding is not None else None

        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                if parent_session_id is not None:
                    cur.execute(
                        """
                        INSERT INTO conversations
                            (session_id, agent_identity, role, content, embedding, metadata, parent_session_id)
                        VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb, %s)
                        RETURNING id
                        """,
                        (session_id, agent_identity, role, content, vec_literal, meta_json, parent_session_id),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO conversations
                            (session_id, agent_identity, role, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb)
                        RETURNING id
                        """,
                        (session_id, agent_identity, role, content, vec_literal, meta_json),
                    )
                row = cur.fetchone()
                conn.commit()
                return int(row[0])

    def search_turns(
        self,
        *,
        query_embedding: List[float],
        agent_identity: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 5,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Semantic recall over conversation turns. Same shape as `search()`."""
        vec_literal = to_pgvector_literal(query_embedding)
        clauses: List[str] = []
        params: List[Any] = []
        if agent_identity:
            clauses.append("agent_identity = %s")
            params.append(agent_identity)
        if session_id:
            clauses.append("session_id = %s")
            params.append(session_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT id, session_id, agent_identity, role, content, ts, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM conversations
                    {where}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    [vec_literal, *params, vec_literal, limit],
                )
                rows = list(cur.fetchall())

        if min_similarity > 0:
            rows = [r for r in rows if (r.get("score") or 0) >= min_similarity]
        return rows

    def hybrid_search_turns(
        self,
        *,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        agent_identity: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 5,
        pool: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid recall over conversation turns. RRF fusion, same shape as
        hybrid_search() but returns turn columns (session_id, role, ts)."""
        if not query_text or not query_text.strip():
            return []
        if pool is None:
            pool = max(int(limit) * 4, 40)

        params: Dict[str, Any] = {
            "qtext": query_text, "k": self.RRF_K, "pool": pool, "limit": limit,
        }
        filters: List[str] = []
        if agent_identity:
            filters.append("agent_identity = %(agent)s")
            params["agent"] = agent_identity
        if session_id:
            filters.append("session_id = %(session)s")
            params["session"] = session_id
        extra = (" AND " + " AND ".join(filters)) if filters else ""

        sql = self._build_hybrid_sql(
            table="conversations",
            extra_filter=extra,
            has_vector=query_embedding is not None,
            columns="m.id, m.session_id, m.agent_identity, m.role, "
                    "m.content, m.ts, m.metadata",
        )
        if query_embedding is not None:
            params["qvec"] = to_pgvector_literal(query_embedding)

        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())

    # -- Maintenance ---------------------------------------------------------

    def count_turns(
        self,
        *,
        agent_identity: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        clauses: List[str] = []
        params: List[Any] = []
        if agent_identity:
            clauses.append("agent_identity = %s")
            params.append(agent_identity)
        if session_id:
            clauses.append("session_id = %s")
            params.append(session_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM conversations {where}", params)
                return int(cur.fetchone()[0])

    def count(
        self,
        *,
        agent_identity: Optional[str] = None,
        target: Optional[str] = None,
    ) -> int:
        clauses: List[str] = []
        params: List[Any] = []
        if agent_identity:
            clauses.append("agent_identity = %s")
            params.append(agent_identity)
        if target:
            clauses.append("target = %s")
            params.append(target)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM memory_entries {where}", params)
                return int(cur.fetchone()[0])

    def health(self) -> Dict[str, Any]:
        """Liveness probe — pool reachable + table exists. Never raises."""
        try:
            with self._get_pool().connection(timeout=3.0) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT to_regclass('memory_entries') IS NOT NULL")
                    has_table = bool(cur.fetchone()[0])
                    if not has_table:
                        return {"ok": False, "error": "memory_entries table missing", "row_count": 0}
                    cur.execute("SELECT COUNT(*) FROM memory_entries")
                    return {"ok": True, "error": "", "row_count": int(cur.fetchone()[0])}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)[:200], "row_count": 0}

    # -- Agent attribution + delegation (v0.4.0; requires migration 002) ------

    def register_agent(
        self,
        *,
        agent_identity: str,
        kind: str = "theme",
        attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upsert an agent into memory_agents (provenance registry).

        ON CONFLICT bumps last_seen + merges attrs. DML only — no DDL.
        Caller must have verified migration 002 is applied.
        """
        attrs_json = json.dumps(attrs or {})
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_agents (agent_identity, kind, attrs)
                    VALUES (%s, %s, %s::jsonb)
                    ON CONFLICT (agent_identity) DO UPDATE
                       SET last_seen = now(),
                           kind      = EXCLUDED.kind,
                           attrs     = memory_agents.attrs || EXCLUDED.attrs
                    """,
                    (agent_identity, kind, attrs_json),
                )
                conn.commit()

    def record_delegation(
        self,
        *,
        parent_identity: Optional[str],
        child_identity: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        child_session_id: Optional[str] = None,
        kind: str = "delegated",
        attrs: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Insert a parent->child delegation edge. Returns edge id. DML only."""
        attrs_json = json.dumps(attrs or {})
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_agent_edges
                        (parent_identity, child_identity, parent_session_id,
                         child_session_id, kind, attrs)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING id
                    """,
                    (parent_identity, child_identity, parent_session_id,
                     child_session_id, kind, attrs_json),
                )
                row = cur.fetchone()
                conn.commit()
                return int(row[0]) if row else None

    def agent_attribution(self) -> List[Dict[str, Any]]:
        """Read the v_agent_memory view: per-identity memory + conversation counts."""
        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM v_agent_memory")
                return list(cur.fetchall())

    # -- Maintenance: backfill / prune / cleanup / remap (v0.4.0) -------------

    def _assert_whitelisted(self, tables) -> tuple:
        tables = tuple(tables or self.CLEANUP_WHITELIST)
        for t in tables:
            if t not in self.CLEANUP_WHITELIST:
                raise ValueError(
                    f"refusing to operate on non-whitelisted table {t!r}; "
                    f"only {self.CLEANUP_WHITELIST} are plugin-owned"
                )
        return tables

    def backfill_null_embeddings(
        self,
        *,
        embed_fn,
        tables=None,
        batch_size: int = 100,
        dry_run: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Re-embed rows with embedding IS NULL in plugin-owned tables.

        Idempotent + resumable: re-running only touches rows still NULL.
        Fail-soft: a row whose embed fails is left NULL and retried next run
        (no inline retry storm). A dimension guard probes embed_fn first and
        refuses to run on a model that returns != 768 dims (config drift must
        fail fast, never write a wrong-dim vector). If the embed endpoint is
        unreachable, the run aborts cleanly (nothing to backfill right now).

        Returns {table: {processed, succeeded, failed, remaining}}.
        """
        tables = self._assert_whitelisted(tables)
        result: Dict[str, Dict[str, Any]] = {}

        if not dry_run:
            try:
                probe = embed_fn("dimension probe")
            except Exception as exc:  # noqa: BLE001
                logger.warning("backfill aborted — embed endpoint unavailable: %s", str(exc)[:200])
                return {t: {"processed": 0, "succeeded": 0, "failed": 0,
                            "remaining": None, "note": "embed-unavailable"} for t in tables}
            if not isinstance(probe, list) or len(probe) != 768:
                got = len(probe) if isinstance(probe, list) else type(probe).__name__
                raise ValueError(
                    f"embed_fn returned {got} dims, expected 768 — refusing to "
                    "backfill (embedding-model drift would corrupt the vector column)"
                )

        for t in tables:
            with self._get_pool().connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT count(*) FROM {t} WHERE embedding IS NULL")
                    remaining = int(cur.fetchone()[0])
            if dry_run:
                result[t] = {"processed": 0, "succeeded": 0, "failed": 0, "remaining": remaining}
                continue

            processed = succeeded = failed = 0
            while True:
                with self._get_pool().connection() as conn:
                    with conn.cursor(row_factory=dict_row) as cur:
                        cur.execute(
                            f"SELECT id, content FROM {t} WHERE embedding IS NULL ORDER BY id LIMIT %s",
                            (batch_size,),
                        )
                        rows = list(cur.fetchall())
                if not rows:
                    break
                batch_progress = 0
                for r in rows:
                    processed += 1
                    # The whole embed → literal → UPDATE pipeline is fail-soft
                    # per row (v0.4.2): a vector that embeds "successfully" but
                    # is unstorable (NaN/Inf tokens rejected by vector_in, a
                    # transient DB error) must mark THIS row failed and move
                    # on, not abort the run for every remaining table.
                    try:
                        vec = embed_fn(r["content"])
                        vec_literal = to_pgvector_literal(vec)
                        with self._get_pool().connection() as conn:
                            with conn.cursor() as cur:
                                cur.execute(
                                    f"UPDATE {t} SET embedding = %s::vector WHERE id = %s",
                                    (vec_literal, r["id"]),
                                )
                                conn.commit()
                    except Exception:  # noqa: BLE001 — fail-soft, leave NULL
                        failed += 1
                        continue
                    succeeded += 1
                    batch_progress += 1
                # If an entire batch failed to embed (endpoint flapping), stop —
                # the same rows would just be re-fetched forever otherwise.
                if batch_progress == 0:
                    break

            with self._get_pool().connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT count(*) FROM {t} WHERE embedding IS NULL")
                    remaining = int(cur.fetchone()[0])
            result[t] = {"processed": processed, "succeeded": succeeded,
                         "failed": failed, "remaining": remaining}
        return result

    def prune_conversations(self, *, older_than_days: int, dry_run: bool = False) -> int:
        """Delete conversation turns older than N days. memory_entries are never pruned.

        older_than_days <= 0 is a no-op (TTL disabled). Returns rows
        deleted (or, in dry_run, rows that WOULD be deleted)."""
        if older_than_days is None or int(older_than_days) <= 0:
            return 0
        days = int(older_than_days)
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                if dry_run:
                    cur.execute(
                        "SELECT count(*) FROM conversations WHERE ts < now() - make_interval(days => %s)",
                        (days,),
                    )
                    return int(cur.fetchone()[0])
                cur.execute(
                    "DELETE FROM conversations WHERE ts < now() - make_interval(days => %s)",
                    (days,),
                )
                n = cur.rowcount
                conn.commit()
                return int(n)

    def scan_pii(self, *, pattern: str = r"\d{10,11}", tables=None) -> Dict[str, int]:
        """Count rows whose content matches a PII regex (default: 10-11 digit
        phone numbers). For pre-cleanup review — does NOT modify anything.

        Identity-based deletion only removes rows whose agent_identity is the DM
        key; a phone number embedded in another row's CONTENT survives. Run this
        to find those before declaring PII cleanup complete."""
        tables = self._assert_whitelisted(tables)
        out: Dict[str, int] = {}
        with self._get_pool().connection() as conn:
            for t in tables:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT count(*) FROM {t} WHERE content ~ %s", (pattern,))
                    out[t] = int(cur.fetchone()[0])
        return out

    def delete_by_identity(self, *, identities, tables=None, dry_run: bool = True) -> Dict[str, int]:
        """Delete all rows for the given agent_identities from plugin-owned tables.

        Whitelist-guarded (never events/entities/relations). dry_run=True (the
        default) only counts. Returns {table: rows_deleted_or_would_delete}."""
        identities = list(identities)
        tables = self._assert_whitelisted(tables)
        out: Dict[str, int] = {}
        with self._get_pool().connection() as conn:
            # Serialize destructive cleanup with remap_identity (same advisory lock)
            # so a concurrent remap can't interleave between our count and delete.
            # Read-only dry-run doesn't need it.
            if not dry_run:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_xact_lock(9999)")
            for t in tables:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT count(*) FROM {t} WHERE agent_identity = ANY(%s)",
                        (identities,),
                    )
                    cnt = int(cur.fetchone()[0])
                    if not dry_run and cnt:
                        cur.execute(
                            f"DELETE FROM {t} WHERE agent_identity = ANY(%s)",
                            (identities,),
                        )
                    out[t] = cnt
            if not dry_run:
                conn.commit()
        return out

    def remap_identity(
        self,
        *,
        old_identity: str,
        new_identity: str,
        dry_run: bool = True,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Merge old_identity into new_identity across plugin-owned tables.

        memory_entries has UNIQUE(agent_identity, target, content): a naive
        UPDATE would abort on collisions, so we INSERT ... ON CONFLICT DO NOTHING
        then DELETE the old rows (duplicates are dropped, not errored).
        conversations has no unique constraint (turns are events) → plain UPDATE.

        dry_run=True (default) reports {moved, dropped_duplicates} without
        changing anything. When duplicates would be dropped, force=True is
        required for >10 to guard against accidental data loss. Runs under
        advisory lock 9999 (shared with cleanup) so maintenance ops serialize."""
        self._assert_whitelisted(("memory_entries", "conversations"))  # consistency guard
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM memory_entries WHERE agent_identity = %s", (old_identity,))
                me_old = int(cur.fetchone()[0])
                cur.execute(
                    """
                    SELECT count(*) FROM memory_entries o
                     WHERE o.agent_identity = %s
                       AND EXISTS (SELECT 1 FROM memory_entries n
                                    WHERE n.agent_identity = %s
                                      AND n.target = o.target
                                      AND n.content = o.content)
                    """,
                    (old_identity, new_identity),
                )
                dupes = int(cur.fetchone()[0])
                cur.execute("SELECT count(*) FROM conversations WHERE agent_identity = %s", (old_identity,))
                conv_old = int(cur.fetchone()[0])

            if dry_run:
                return {
                    "dry_run": True,
                    "memory_entries": {"moved": me_old - dupes, "dropped_duplicates": dupes},
                    "conversations": {"moved": conv_old},
                }

            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(9999)")
                # Re-count duplicates INSIDE the advisory lock (v0.4.2): the
                # pre-count above ran unlocked (it also feeds the dry-run
                # report), so a concurrent writer inserting colliding rows
                # between count and mutate could push real duplicates past the
                # unlocked estimate — the >10 force-guard must be enforced
                # against the locked truth, mirroring delete_by_identity's
                # count-under-lock ordering.
                cur.execute(
                    """
                    SELECT count(*) FROM memory_entries o
                     WHERE o.agent_identity = %s
                       AND EXISTS (SELECT 1 FROM memory_entries n
                                    WHERE n.agent_identity = %s
                                      AND n.target = o.target
                                      AND n.content = o.content)
                    """,
                    (old_identity, new_identity),
                )
                dupes = int(cur.fetchone()[0])
                if dupes > 10 and not force:
                    raise RuntimeError(
                        f"remap {old_identity!r} -> {new_identity!r} would drop {dupes} "
                        "duplicate memory_entries rows; re-run with force=True to proceed"
                    )
                cur.execute(
                    """
                    INSERT INTO memory_entries
                        (agent_identity, target, content, embedding, created_at, updated_at, metadata)
                    SELECT %s, target, content, embedding, created_at, updated_at, metadata
                      FROM memory_entries WHERE agent_identity = %s
                    ON CONFLICT (agent_identity, target, content) DO NOTHING
                    """,
                    (new_identity, old_identity),
                )
                moved = cur.rowcount
                cur.execute("DELETE FROM memory_entries WHERE agent_identity = %s", (old_identity,))
                cur.execute(
                    "UPDATE conversations SET agent_identity = %s WHERE agent_identity = %s",
                    (new_identity, old_identity),
                )
                conv_moved = cur.rowcount
            conn.commit()
            return {
                "dry_run": False,
                "memory_entries": {"moved": int(moved), "dropped_duplicates": dupes},
                "conversations": {"moved": int(conv_moved)},
            }

    def log_maintenance(
        self,
        *,
        operation: str,
        target_table: Optional[str] = None,
        identity_pattern: Optional[str] = None,
        affected_count: Optional[int] = None,
        dropped_dupes: Optional[int] = None,
        dry_run: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Best-effort audit row in memory_maintenance_log. Never raises."""
        try:
            with self._get_pool().connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO memory_maintenance_log
                            (operation, target_table, identity_pattern, affected_count,
                             dropped_dupes, dry_run, details)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (operation, target_table, identity_pattern, affected_count,
                         dropped_dupes, dry_run, json.dumps(details or {})),
                    )
                    conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.debug("maintenance log write failed: %s", exc)
