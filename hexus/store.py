"""store.py — Postgres ops for the hexus memory plugin.
#
# Forked from andreab67/hermes-hexus (BSD-3-Clause).
#
# Wraps psycopg3 + psycopg_pool. Mirrors hermes-agent's native built-in
# memory model (`memory` tool's add/replace/remove on targets 'memory' /
# 'user') into a single Postgres table with embeddings.
#
# Uses a small ConnectionPool because the plugin is touched from two
# threads at runtime: the agent thread (for prefetch / recall_memory /
# ensure_schema / health) and the async-writer drain thread (for the
# mirrored INSERTs / UPDATEs / DELETEs). Pooling beats short-lived
# connections under that two-thread pattern without adding much
# complexity.
#
# No SQLAlchemy, no LLM-mediated workers, no deriver loops.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

try:
    from .embed import to_hexus_literal
except ImportError:
    from embed import to_hexus_literal

try:
    from .entity_extractor import EntityExtractor
except ImportError:
    from entity_extractor import EntityExtractor

import hashlib

try:
    from .ccr.cache import CCRCache
except ImportError:
    from ccr.cache import CCRCache

logger = logging.getLogger(__name__)

_cross_encoder_model: Any = None
_cross_encoder_lock = threading.Lock()

def get_cross_encoder() -> Any:
    global _cross_encoder_model
    if _cross_encoder_model is None:
        with _cross_encoder_lock:
            if _cross_encoder_model is None:
                from sentence_transformers import CrossEncoder
                _cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder_model


class MemoryStore:
    """Postgres-backed mirror of hermes-agent's built-in memory entries."""

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = 0,
        max_size: int = 4,
        timeout: float = 5.0,
        max_idle: float = 30.0,
        max_lifetime: float = 300.0,
        entity_extractor_enabled: bool = True,
        entity_extractor_patterns: Optional[Dict[str, str]] = None,
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
        env_enabled = os.environ.get("HEXUS_ENTITY_EXTRACTOR_ENABLED")
        if env_enabled is not None:
            entity_extractor_enabled = env_enabled.lower() not in ("0", "false", "no")

        env_patterns = os.environ.get("HEXUS_ENTITY_EXTRACTOR_PATTERNS")
        if env_patterns is not None:
            try:
                entity_extractor_patterns = json.loads(env_patterns)
            except Exception as exc:
                logger.warning("Failed to parse HEXUS_ENTITY_EXTRACTOR_PATTERNS: %s", exc)

        self._entity_extractor = EntityExtractor(
            patterns=entity_extractor_patterns,
            enabled=entity_extractor_enabled,
        )
        self._ccr_cache = CCRCache()

        # Resolve vector precision configuration
        precision = os.environ.get("HEXUS_VECTOR_PRECISION", "float32").lower()
        if precision in ("float16", "fp16", "half"):
            self._vector_precision = "float16"
        elif precision == "binary":
            self._vector_precision = "binary"
        else:
            self._vector_precision = "float32"
            
        self._actual_column_type = "vector(384)"  # Default fallback




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
                    name="hexus-memory",
                )
        return self._pool

    def close(self) -> None:
        """Close the connection pool. Idempotent."""
        with self._lock:
            if self._pool is not None:
                try:
                    self._pool.close()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("hexus pool close: %s", exc)
                finally:
                    self._pool = None

    # -- Schema --------------------------------------------------------------

    class SchemaNotApplied(RuntimeError):
        """Raised when memory_entries does not exist in the target DB."""

    def ensure_schema(self) -> None:
        """Verify the schema is in place. Does NOT run DDL."""
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('memory_entries')")
                if cur.fetchone()[0] is None:
                    raise self.SchemaNotApplied(
                        "memory_entries table missing. Apply the migration as DB admin: "
                        "psql -d <dbname> -f plugins/memory/hexus/migrations/001_schema.sql"
                    )
                cur.execute("SELECT to_regclass('delegations')")
                if cur.fetchone()[0] is None:
                    raise self.SchemaNotApplied(
                        "delegations table missing. Apply the migration as DB admin: "
                        "psql -d <dbname> -f plugins/memory/hexus/migrations/002_observability.sql"
                    )
        # Adapt schema types and indexes dynamically to match the configured precision
        self.adapt_vector_precision()

    def adapt_vector_precision(self) -> None:
        """Verify and adapt the database column types and indexes to match the configured precision."""
        precision = self._vector_precision
        
        # Determine target type
        target_type = "halfvec(384)" if precision == "float16" else "vector(384)"
        
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                # Check current type of memory_entries.embedding
                cur.execute("""
                    SELECT pg_catalog.format_type(atttypid, atttypmod)
                    FROM pg_catalog.pg_attribute
                    WHERE attrelid = 'memory_entries'::regclass
                      AND attname = 'embedding';
                """)
                row = cur.fetchone()
                if not row:
                    return  # Table doesn't exist yet, migrations will handle it.
                
                current_type = row[0]
                self._actual_column_type = current_type
                if current_type != target_type:
                    logger.info("Adapting database column types from %s to %s to match HEXUS_VECTOR_PRECISION=%s", current_type, target_type, precision)
                    try:
                        # 1. Drop existing indexes that depend on embedding column
                        cur.execute("DROP INDEX IF EXISTS ix_memory_entries_embedding_hnsw;")
                        cur.execute("DROP INDEX IF EXISTS ix_conversations_embedding_hnsw;")
                        cur.execute("DROP INDEX IF EXISTS ix_delegations_embedding_hnsw;")
                        
                        # 2. Alter column types
                        cur.execute(f"ALTER TABLE memory_entries ALTER COLUMN embedding TYPE {target_type};")
                        cur.execute(f"ALTER TABLE conversations ALTER COLUMN embedding TYPE {target_type};")
                        cur.execute(f"ALTER TABLE delegations ALTER COLUMN embedding TYPE {target_type};")
                        
                        conn.commit()
                        self._actual_column_type = target_type
                        logger.info("Successfully altered database columns to %s", target_type)
                    except Exception as exc:
                        conn.rollback()
                        logger.warning("Failed to alter database column types (insufficient permissions?): %s", exc)
                        
                # Ensure the correct indexes are in place based on precision
                try:
                    if precision == "float16":
                        # Drop binary indexes if they exist
                        cur.execute("DROP INDEX IF EXISTS ix_memory_entries_embedding_binary_hnsw;")
                        cur.execute("DROP INDEX IF EXISTS ix_conversations_embedding_binary_hnsw;")
                        cur.execute("DROP INDEX IF EXISTS ix_delegations_embedding_binary_hnsw;")

                        # Create HNSW index on halfvec_cosine_ops
                        cur.execute("""
                            CREATE INDEX IF NOT EXISTS ix_memory_entries_embedding_hnsw
                              ON memory_entries USING hnsw (embedding halfvec_cosine_ops)
                              WITH (m = 16, ef_construction = 64);
                        """)
                        cur.execute("""
                            CREATE INDEX IF NOT EXISTS ix_conversations_embedding_hnsw
                              ON conversations USING hnsw (embedding halfvec_cosine_ops)
                              WITH (m = 16, ef_construction = 64);
                        """)
                        cur.execute("""
                            CREATE INDEX IF NOT EXISTS ix_delegations_embedding_hnsw
                              ON delegations USING hnsw (embedding halfvec_cosine_ops)
                              WITH (m = 16, ef_construction = 64);
                        """)
                    elif precision == "binary":
                        # Drop standard HNSW index to save RAM (if they exist)
                        cur.execute("DROP INDEX IF EXISTS ix_memory_entries_embedding_hnsw;")
                        cur.execute("DROP INDEX IF EXISTS ix_conversations_embedding_hnsw;")
                        cur.execute("DROP INDEX IF EXISTS ix_delegations_embedding_hnsw;")
                        
                        # Create the binary HNSW expression indexes
                        cur.execute("""
                            CREATE INDEX IF NOT EXISTS ix_memory_entries_embedding_binary_hnsw
                              ON memory_entries USING hnsw ((binary_quantize(embedding)::bit(384)) bit_hamming_ops)
                              WITH (m = 16, ef_construction = 64);
                        """)
                        cur.execute("""
                            CREATE INDEX IF NOT EXISTS ix_conversations_embedding_binary_hnsw
                              ON conversations USING hnsw ((binary_quantize(embedding)::bit(384)) bit_hamming_ops)
                              WITH (m = 16, ef_construction = 64);
                        """)
                        cur.execute("""
                            CREATE INDEX IF NOT EXISTS ix_delegations_embedding_binary_hnsw
                              ON delegations USING hnsw ((binary_quantize(embedding)::bit(384)) bit_hamming_ops)
                              WITH (m = 16, ef_construction = 64);
                        """)
                    else:  # float32
                        # Drop binary indexes if they exist
                        cur.execute("DROP INDEX IF EXISTS ix_memory_entries_embedding_binary_hnsw;")
                        cur.execute("DROP INDEX IF EXISTS ix_conversations_embedding_binary_hnsw;")
                        cur.execute("DROP INDEX IF EXISTS ix_delegations_embedding_binary_hnsw;")

                        # Create HNSW index on vector_cosine_ops
                        cur.execute("""
                            CREATE INDEX IF NOT EXISTS ix_memory_entries_embedding_hnsw
                              ON memory_entries USING hnsw (embedding vector_cosine_ops)
                              WITH (m = 16, ef_construction = 64);
                        """)
                        cur.execute("""
                            CREATE INDEX IF NOT EXISTS ix_conversations_embedding_hnsw
                              ON conversations USING hnsw (embedding vector_cosine_ops)
                              WITH (m = 16, ef_construction = 64);
                        """)
                        cur.execute("""
                            CREATE INDEX IF NOT EXISTS ix_delegations_embedding_hnsw
                              ON delegations USING hnsw (embedding vector_cosine_ops)
                              WITH (m = 16, ef_construction = 64);
                        """)
                    conn.commit()
                except Exception as exc:
                    conn.rollback()
                    logger.warning("Failed to create/ensure quantization indexes: %s", exc)

    def apply_migration_as_admin(self, *, admin_dsn: str) -> None:
        """One-shot admin path: run the full migrations with privileged creds."""
        migrations_dir = Path(__file__).parent / "migrations"
        sql_files = sorted(migrations_dir.glob("*.sql"))
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                for sql_file in sql_files:
                    sql = sql_file.read_text(encoding="utf-8")
                    cur.execute(sql)

    # -- Built-in memory mirror (called by on_memory_write) ------------------

    def add(
        self,
        *,
        agent_identity: str,
        target: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        compressed: Optional[str] = None,
        content_hash: Optional[bytes] = None,
    ) -> Optional[int]:
        """Insert a memory entry. Returns row id, or None if duplicate (no-op)."""
        meta = dict(metadata or {})
        if "entities" not in meta:
            extracted = self._entity_extractor.extract_entities(content)
            if extracted:
                meta["entities"] = extracted

        # Hashing / deduplication logic
        if content_hash is None:
            hash_target = compressed if compressed is not None else content
            content_hash = hashlib.sha256(hash_target.encode("utf-8")).digest()

        vec_literal = to_hexus_literal(embedding) if embedding is not None else None

        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                # Deduplication check: does a row with this content_hash, target and agent_identity exist?
                cur.execute(
                    """
                    SELECT id, metadata FROM memory_entries
                    WHERE agent_identity = %s AND target = %s AND content_hash = %s
                    """,
                    (agent_identity, target, content_hash),
                )
                row = cur.fetchone()
                if row:
                    # Duplicate found! Merge metadata
                    existing_id = row["id"]
                    existing_meta = dict(row["metadata"] or {})
                    
                    # Merge logic: append provenance from new metadata
                    new_provenance = meta.get("provenance")
                    if new_provenance:
                        existing_provenances = existing_meta.get("provenance")
                        if existing_provenances is None:
                            existing_provenances = []
                        elif not isinstance(existing_provenances, list):
                            existing_provenances = [existing_provenances]
                        
                        if isinstance(new_provenance, list):
                            for p in new_provenance:
                                if p not in existing_provenances:
                                    existing_provenances.append(p)
                        else:
                            if new_provenance not in existing_provenances:
                                existing_provenances.append(new_provenance)
                        existing_meta["provenance"] = existing_provenances
                    
                    # Merge session_ids
                    existing_sids = existing_meta.setdefault("sessions", [])
                    if not isinstance(existing_sids, list):
                        existing_sids = [existing_sids]
                        existing_meta["sessions"] = existing_sids
                    
                    # Also collect existing session_id if present
                    orig_sid = existing_meta.get("session_id")
                    if orig_sid and orig_sid not in existing_sids:
                        existing_sids.append(orig_sid)
                        
                    new_sid = meta.get("session_id")
                    if new_sid and new_sid not in existing_sids:
                        existing_sids.append(new_sid)
                            
                    # Update metadata on the existing row
                    cur.execute(
                        """
                        UPDATE memory_entries
                        SET metadata = %s::jsonb, updated_at = now()
                        WHERE id = %s
                        """,
                        (json.dumps(existing_meta), existing_id),
                    )
                    conn.commit()
                    
                    if compressed:
                        self._ccr_cache.set(existing_id, content)
                    return None

                # Otherwise insert
                meta_json = json.dumps(meta)
                cur.execute(
                    """
                    INSERT INTO memory_entries
                        (agent_identity, target, content, embedding, metadata, compressed, content_hash)
                    VALUES (%s, %s, %s, %s::vector, %s::jsonb, %s, %s)
                    ON CONFLICT (agent_identity, target, content) DO NOTHING
                    RETURNING id
                    """,
                    (agent_identity, target, content, vec_literal, meta_json, compressed, content_hash),
                )
                res = cur.fetchone()
                conn.commit()
                
                row_id = None
                if res:
                    if isinstance(res, dict):
                        row_id = res.get("id")
                    else:
                        row_id = res[0]
                
                if row_id and compressed:
                    self._ccr_cache.set(row_id, content)
                return row_id


    def replace(
        self,
        *,
        agent_identity: str,
        target: str,
        old_text: str,
        new_content: str,
        new_embedding: Optional[List[float]] = None,
        compressed: Optional[str] = None,
        content_hash: Optional[bytes] = None,
    ) -> int:
        """Update entries in (agent_identity, target) where content contains old_text."""
        vec_literal = (
            to_hexus_literal(new_embedding) if new_embedding is not None else None
        )
        new_entities = self._entity_extractor.extract_entities(new_content)
        new_entities_json = json.dumps(new_entities)

        if content_hash is None:
            hash_target = compressed if compressed is not None else new_content
            content_hash = hashlib.sha256(hash_target.encode("utf-8")).digest()

        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                # Find matching rows to update cache
                cur.execute(
                    """
                    SELECT id FROM memory_entries
                     WHERE agent_identity = %s
                       AND target = %s
                       AND content LIKE %s
                    """,
                    (agent_identity, target, f"%{old_text}%"),
                )
                matching_ids = [r[0] for r in cur.fetchall()]

                cur.execute(
                    """
                    UPDATE memory_entries
                       SET content      = %s,
                           embedding    = %s::vector,
                           metadata     = jsonb_set(metadata, '{entities}', %s::jsonb, true),
                           compressed   = %s,
                           content_hash = %s,
                           updated_at   = now()
                      WHERE agent_identity = %s
                        AND target = %s
                        AND content LIKE %s
                    """,
                    (new_content, vec_literal, new_entities_json, compressed, content_hash, agent_identity, target, f"%{old_text}%"),
                )
                updated = cur.rowcount
                conn.commit()

                if updated > 0 and compressed:
                    for rid in matching_ids:
                        self._ccr_cache.set(rid, new_content)

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
                    (agent_identity, target, f"%{old_text}%"),
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
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List entries in an agent's scope. If target is None, both stores."""
        params: List[Any] = [agent_identity]
        target_clause = ""
        if target:
            target_clause = "AND target = %s"
            params.append(target)
        params.append(limit)
        params.append(offset)

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
                    OFFSET %s
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
        decay_half_life_days: float = 0.0,
        recall_boost_weight: float = 0.0,
        platform: Optional[str] = None,
        min_confidence: float = 0.0,
        rerank: bool = False,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic recall via cosine distance.

        agent_identity=None → search across ALL agents (cross-theme recall).
        target=None → search both 'memory' and 'user'.
        Returns rows with `score` = 1 - cosine_distance ∈ [0, 1].
        """
        vec_literal = to_hexus_literal(query_embedding)
        clauses: List[str] = []
        params: List[Any] = []
        if agent_identity:
            clauses.append("agent_identity = %s")
            params.append(agent_identity)
        if target:
            clauses.append("target = %s")
            params.append(target)
        if platform:
            clauses.append("metadata->>'platform' = %s")
            params.append(platform)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        cast_type = "halfvec" if "halfvec" in getattr(self, "_actual_column_type", "vector(384)") else "vector"

        if self._vector_precision == "binary":
            db_limit = max(limit * 10, 50)
            if rerank:
                db_limit = max(db_limit, 100)
            with self._get_pool().connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        f"""
                        SELECT id, agent_identity, target, content, compressed, created_at,
                               updated_at, metadata,
                               1 - (embedding <=> %s::{cast_type}) AS score
                        FROM memory_entries
                        {where}
                        ORDER BY (binary_quantize(embedding)::bit(384)) <~> (binary_quantize(%s::{cast_type})::bit(384))
                        LIMIT %s
                        """,
                        [vec_literal] + params + [vec_literal, db_limit],
                    )
                    rows = list(cur.fetchall())
        else:
            db_limit = max(limit, 50) if rerank else limit
            with self._get_pool().connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        f"""
                        SELECT id, agent_identity, target, content, compressed, created_at,
                               updated_at, metadata,
                               1 - (embedding <=> %s::{cast_type}) AS score
                        FROM memory_entries
                        {where}
                        ORDER BY embedding <=> %s::{cast_type}
                        LIMIT %s
                        """,
                        [vec_literal] + params + [vec_literal, db_limit],
                    )
                    rows = list(cur.fetchall())
                print(f"DEBUG_SEARCH_RAW: rows_len={len(rows)} data={[ {'id': r['id'], 'content': r['content'], 'score': r['score']} for r in rows ]}")

        # Apply boost & decay
        rows = self._apply_recall_boost(rows, recall_boost_weight)
        rows = self._apply_temporal_decay(rows, decay_half_life_days)
        rows = self._apply_min_confidence(rows, min_confidence)

        if rerank and query_text and rows:
            model = get_cross_encoder()
            pairs = [[query_text, r.get("compressed") or r.get("content")] for r in rows]
            rerank_scores = model.predict(pairs)
            for r, rerank_score in zip(rows, rerank_scores):
                r["rerank_score"] = float(rerank_score)
                r["score"] = r["rerank_score"]

        if self._vector_precision == "binary" or decay_half_life_days > 0.0 or recall_boost_weight > 0.0 or rerank:
            rows = sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)

        rows = rows[:limit]

        if min_similarity > 0:
            rows = [r for r in rows if (r.get("score") or 0) >= min_similarity]

        if rows:
            self.increment_recall_counts("memory_entries", [r["id"] for r in rows])
            for r in rows:
                if r.get("compressed") is not None:
                    self._ccr_cache.set(r["id"], r["content"])
                    r["content"] = r["compressed"]

        return rows


    def hybrid_search(
        self,
        *,
        query_embedding: List[float],
        query_text: str,
        agent_identity: Optional[str] = None,
        target: Optional[str] = None,
        limit: int = 5,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        min_similarity: float = 0.0,
        decay_half_life_days: float = 0.0,
        recall_boost_weight: float = 0.0,
        platform: Optional[str] = None,
        min_confidence: float = 0.0,
        rerank: bool = False,
    ) -> List[Dict[str, Any]]:
        """Blend semantic vector search and full-text search."""
        if not query_text or not query_text.strip():
            rows = self.search(
                query_embedding=query_embedding,
                agent_identity=agent_identity,
                target=target,
                limit=limit,
                min_similarity=min_similarity,
                decay_half_life_days=decay_half_life_days,
                recall_boost_weight=recall_boost_weight,
                platform=platform,
                min_confidence=min_confidence,
                rerank=rerank,
                query_text=None,
            )
            for r in rows:
                r["vector_score"] = r.get("vector_score", r.get("score"))
                r["text_score"] = 0.0
            return rows

        vec_literal = to_hexus_literal(query_embedding)
        
        clauses: List[str] = []
        params: List[Any] = []
        if agent_identity:
            clauses.append("agent_identity = %s")
            params.append(agent_identity)
        if target:
            clauses.append("target = %s")
            params.append(target)
        if platform:
            clauses.append("metadata->>'platform' = %s")
            params.append(platform)
            
        where = ("AND " + " AND ".join(clauses)) if clauses else ""

        db_limit = max(limit, 50) if rerank else limit

        sql = f"""
        WITH vector_search AS (
            SELECT id, agent_identity, target, content, compressed, created_at, updated_at, metadata,
                   1 - (embedding <=> %s::vector) AS vector_score
            FROM memory_entries
            WHERE 1=1 {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        ),
        text_search AS (
            SELECT id, agent_identity, target, content, compressed, created_at, updated_at, metadata,
                   ts_rank(to_tsvector('english', COALESCE(compressed, content)), websearch_to_tsquery('english', %s)) AS text_score
            FROM memory_entries
            WHERE to_tsvector('english', COALESCE(compressed, content)) @@ websearch_to_tsquery('english', %s)
              {where}
            ORDER BY text_score DESC
            LIMIT %s
        )
        SELECT COALESCE(v.id, t.id) AS id,
               COALESCE(v.agent_identity, t.agent_identity) AS agent_identity,
               COALESCE(v.target, t.target) AS target,
               COALESCE(v.content, t.content) AS content,
               COALESCE(v.compressed, t.compressed) AS compressed,
               COALESCE(v.created_at, t.created_at) AS created_at,
               COALESCE(v.updated_at, t.updated_at) AS updated_at,
               COALESCE(v.metadata, t.metadata) AS metadata,
               COALESCE(v.vector_score, 0.0) AS vector_score,
               COALESCE(t.text_score, 0.0) AS text_score,
               (%s * COALESCE(v.vector_score, 0.0)) + (%s * COALESCE(t.text_score, 0.0)) AS score
        FROM vector_search v
        FULL OUTER JOIN text_search t ON v.id = t.id
        ORDER BY score DESC
        LIMIT %s
        """

        v_params = [vec_literal]
        for p in params:
            v_params.append(p)
        v_params.extend([vec_literal, db_limit])
        
        t_params = [query_text, query_text] + params + [db_limit]
        
        all_params = v_params + t_params + [vector_weight, text_weight, db_limit]

        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, all_params)
                rows = list(cur.fetchall())

        # Blended Score Calculation:
        # Combined Score = 0.6 * S_vector + 0.3 * S_BM25 + 0.1 * S_recency
        now = datetime.now(timezone.utc)
        for r in rows:
            ts_val = r.get("updated_at") or r.get("ts") or r.get("created_at")
            if isinstance(ts_val, str):
                try:
                    from datetime import datetime as dt
                    ts_val = dt.fromisoformat(ts_val)
                except Exception:
                    pass
            if ts_val:
                if ts_val.tzinfo is None:
                    ts_val = ts_val.replace(tzinfo=timezone.utc)
                age_days = (now - ts_val).total_seconds() / 86400.0
                if decay_half_life_days > 0.0:
                    r["recency_score"] = math.exp(-math.log(2.0) * age_days / decay_half_life_days)
                else:
                    r["recency_score"] = 1.0
            else:
                r["recency_score"] = 1.0

        max_text_score = max([r.get("text_score", 0.0) for r in rows]) if rows else 0.0
        for r in rows:
            v_score = r.get("vector_score", 0.0)
            t_score = r.get("text_score", 0.0)
            norm_t_score = (t_score / max_text_score) if max_text_score > 0.0 else 0.0
            rec_score = r.get("recency_score", 1.0)
            r["score"] = 0.6 * v_score + 0.3 * norm_t_score + 0.1 * rec_score

        # Apply boost & min_confidence (decay is already part of score blending)
        rows = self._apply_recall_boost(rows, recall_boost_weight)
        rows = self._apply_min_confidence(rows, min_confidence)

        if rerank and rows:
            model = get_cross_encoder()
            pairs = [[query_text, r.get("compressed") or r.get("content")] for r in rows]
            rerank_scores = model.predict(pairs)
            for r, rerank_score in zip(rows, rerank_scores):
                r["rerank_score"] = float(rerank_score)
                r["score"] = r["rerank_score"]

        rows = sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)
        rows = rows[:limit]

        if min_similarity > 0:
            rows = [r for r in rows if (r.get("score") or 0) >= min_similarity]

        if rows:
            self.increment_recall_counts("memory_entries", [r["id"] for r in rows])
            for r in rows:
                if r.get("compressed") is not None:
                    self._ccr_cache.set(r["id"], r["content"])
                    r["content"] = r["compressed"]

        return rows

    def fetch_full(self, memory_id: int) -> Optional[str]:
        """Fetch the original full content of a memory entry, checking CCRCache first."""
        cached = self._ccr_cache.get(memory_id)
        if cached is not None:
            return cached

        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content FROM memory_entries WHERE id = %s",
                    (memory_id,),
                )
                row = cur.fetchone()
                if row:
                    content = row[0]
                    self._ccr_cache.set(memory_id, content)
                    return content
        return None

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
        for entry in entries:
            if entry in existing:
                skipped += 1
                continue
            vec = None
            try:
                vec = embed_fn(entry) if embed_fn else None
            except Exception:  # noqa: BLE001 — fail-soft on bulk embed
                vec = None
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
    ) -> int:
        """Insert one chat turn. Returns row id.

        No dedup (turns are inherently time-ordered events — same content
        twice is two distinct turns, even verbatim).
        """
        meta = dict(metadata or {})
        if "entities" not in meta:
            extracted = self._entity_extractor.extract_entities(content)
            if extracted:
                meta["entities"] = extracted
        meta_json = json.dumps(meta)
        vec_literal = to_hexus_literal(embedding) if embedding is not None else None


        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
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
        decay_half_life_days: float = 0.0,
        recall_boost_weight: float = 0.0,
        platform: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic recall over conversation turns. Same shape as `search()`."""
        vec_literal = to_hexus_literal(query_embedding)
        clauses: List[str] = []
        params: List[Any] = []
        if agent_identity:
            clauses.append("agent_identity = %s")
            params.append(agent_identity)
        if session_id:
            clauses.append("session_id = %s")
            params.append(session_id)
        if platform:
            clauses.append("metadata->>'platform' = %s")
            params.append(platform)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        cast_type = "halfvec" if "halfvec" in getattr(self, "_actual_column_type", "vector(384)") else "vector"

        if self._vector_precision == "binary":
            db_limit = max(limit * 10, 50)
            with self._get_pool().connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        f"""
                        SELECT id, session_id, agent_identity, role, content, ts, metadata,
                               1 - (embedding <=> %s::{cast_type}) AS score
                        FROM conversations
                        {where}
                        ORDER BY (binary_quantize(embedding)::bit(384)) <~> (binary_quantize(%s::{cast_type})::bit(384))
                        LIMIT %s
                        """,
                        [vec_literal] + params + [vec_literal, db_limit],
                    )
                    rows = list(cur.fetchall())
        else:
            with self._get_pool().connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        f"""
                        SELECT id, session_id, agent_identity, role, content, ts, metadata,
                               1 - (embedding <=> %s::{cast_type}) AS score
                        FROM conversations
                        {where}
                        ORDER BY embedding <=> %s::{cast_type}
                        LIMIT %s
                        """,
                        [vec_literal] + params + [vec_literal, limit],
                    )
                    rows = list(cur.fetchall())

        # Apply boost & decay
        rows = self._apply_recall_boost(rows, recall_boost_weight)
        rows = self._apply_temporal_decay(rows, decay_half_life_days)

        if self._vector_precision == "binary" or decay_half_life_days > 0.0 or recall_boost_weight > 0.0:
            rows = sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)

        if min_similarity > 0:
            rows = [r for r in rows if (r.get("score") or 0) >= min_similarity]

        if rows:
            self.increment_recall_counts("conversations", [r["id"] for r in rows])

        return rows

    def hybrid_search_turns(
        self,
        *,
        query_embedding: List[float],
        query_text: str,
        agent_identity: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 5,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        min_similarity: float = 0.0,
        decay_half_life_days: float = 0.0,
        recall_boost_weight: float = 0.0,
        platform: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Blend semantic vector search and full-text search over conversation turns."""
        if not query_text or not query_text.strip():
            rows = self.search_turns(
                query_embedding=query_embedding,
                agent_identity=agent_identity,
                session_id=session_id,
                limit=limit,
                min_similarity=min_similarity,
                decay_half_life_days=decay_half_life_days,
                recall_boost_weight=recall_boost_weight,
                platform=platform,
            )
            for r in rows:
                r["vector_score"] = r.get("vector_score", r.get("score"))
                r["text_score"] = 0.0
            return rows

        vec_literal = to_hexus_literal(query_embedding)
        clauses: List[str] = []
        params: List[Any] = []
        if agent_identity:
            clauses.append("agent_identity = %s")
            params.append(agent_identity)
        if session_id:
            clauses.append("session_id = %s")
            params.append(session_id)
        if platform:
            clauses.append("metadata->>'platform' = %s")
            params.append(platform)
            
        where = ("AND " + " AND ".join(clauses)) if clauses else ""

        sql = f"""
        WITH vector_search AS (
            SELECT id, session_id, agent_identity, role, content, ts, metadata,
                   1 - (embedding <=> %s::vector) AS vector_score
            FROM conversations
            WHERE 1=1 {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        ),
        text_search AS (
            SELECT id, session_id, agent_identity, role, content, ts, metadata,
                   ts_rank(to_tsvector('english', content), websearch_to_tsquery('english', %s)) AS text_score
            FROM conversations
            WHERE to_tsvector('english', content) @@ websearch_to_tsquery('english', %s)
              {where}
            ORDER BY text_score DESC
            LIMIT %s
        )
        SELECT COALESCE(v.id, t.id) AS id,
               COALESCE(v.session_id, t.session_id) AS session_id,
               COALESCE(v.agent_identity, t.agent_identity) AS agent_identity,
               COALESCE(v.role, t.role) AS role,
               COALESCE(v.content, t.content) AS content,
               COALESCE(v.ts, t.ts) AS ts,
               COALESCE(v.metadata, t.metadata) AS metadata,
               COALESCE(v.vector_score, 0.0) AS vector_score,
               COALESCE(t.text_score, 0.0) AS text_score,
               (%s * COALESCE(v.vector_score, 0.0)) + (%s * COALESCE(t.text_score, 0.0)) AS score
        FROM vector_search v
        FULL OUTER JOIN text_search t ON v.id = t.id
        ORDER BY score DESC
        LIMIT %s
        """

        v_params = [vec_literal]
        for p in params:
            v_params.append(p)
        v_params.extend([vec_literal, limit])
        
        t_params = [query_text, query_text] + params + [limit]
        
        all_params = v_params + t_params + [vector_weight, text_weight, limit]

        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, all_params)
                rows = list(cur.fetchall())

        # Apply boost & decay
        rows = self._apply_recall_boost(rows, recall_boost_weight)
        rows = self._apply_temporal_decay(rows, decay_half_life_days)

        if decay_half_life_days > 0.0 or recall_boost_weight > 0.0:
            rows = sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)

        if min_similarity > 0:
            rows = [r for r in rows if (r.get("score") or 0) >= min_similarity]

        if rows:
            self.increment_recall_counts("conversations", [r["id"] for r in rows])

        return rows

    def record_delegation(
        self,
        *,
        parent_session_id: str,
        child_session_id: str,
        agent_identity: str = "default",
        task: str,
        result: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a delegation entry. Returns row id."""
        vec_literal = to_hexus_literal(embedding) if embedding is not None else None
        meta_json = json.dumps(metadata or {})
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO delegations
                        (parent_session_id, child_session_id, agent_identity, task, result, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s::jsonb)
                    RETURNING id
                    """,
                    (parent_session_id, child_session_id, agent_identity, task, result, vec_literal, meta_json),
                )
                row = cur.fetchone()
                conn.commit()
                return int(row[0])

    def search_delegations(
        self,
        *,
        query_embedding: List[float],
        agent_identity: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        limit: int = 5,
        min_similarity: float = 0.0,
        decay_half_life_days: float = 0.0,
        recall_boost_weight: float = 0.0,
        platform: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic recall over delegations."""
        vec_literal = to_hexus_literal(query_embedding)
        clauses: List[str] = []
        params: List[Any] = []
        if agent_identity:
            clauses.append("agent_identity = %s")
            params.append(agent_identity)
        if parent_session_id:
            clauses.append("parent_session_id = %s")
            params.append(parent_session_id)
        if platform:
            clauses.append("metadata->>'platform' = %s")
            params.append(platform)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        cast_type = "halfvec" if "halfvec" in getattr(self, "_actual_column_type", "vector(384)") else "vector"

        if self._vector_precision == "binary":
            db_limit = max(limit * 10, 50)
            with self._get_pool().connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        f"""
                        SELECT id, parent_session_id, child_session_id, agent_identity, task, result, ts, metadata,
                               1 - (embedding <=> %s::{cast_type}) AS score
                        FROM delegations
                        {where}
                        ORDER BY (binary_quantize(embedding)::bit(384)) <~> (binary_quantize(%s::{cast_type})::bit(384))
                        LIMIT %s
                        """,
                        [vec_literal] + params + [vec_literal, db_limit],
                    )
                    rows = list(cur.fetchall())
        else:
            with self._get_pool().connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        f"""
                        SELECT id, parent_session_id, child_session_id, agent_identity, task, result, ts, metadata,
                               1 - (embedding <=> %s::{cast_type}) AS score
                        FROM delegations
                        {where}
                        ORDER BY embedding <=> %s::{cast_type}
                        LIMIT %s
                        """,
                        [vec_literal] + params + [vec_literal, limit],
                    )
                    rows = list(cur.fetchall())

        # Apply boost & decay
        rows = self._apply_recall_boost(rows, recall_boost_weight)
        rows = self._apply_temporal_decay(rows, decay_half_life_days)

        if self._vector_precision == "binary" or decay_half_life_days > 0.0 or recall_boost_weight > 0.0:
            rows = sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)

        if min_similarity > 0:
            rows = [r for r in rows if (r.get("score") or 0) >= min_similarity]

        if rows:
            self.increment_recall_counts("delegations", [r["id"] for r in rows])

        return rows

    def cleanup_stale_records(
        self,
        *,
        conversations_ttl_days: Optional[int] = None,
        memories_ttl_days: Optional[int] = None,
        delegations_ttl_days: Optional[int] = None,
    ) -> Dict[str, int]:
        """Delete records older than the specified TTL. Returns counts of deleted items."""
        deleted = {"conversations": 0, "memory_entries": 0, "delegations": 0}
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                if conversations_ttl_days is not None and conversations_ttl_days > 0:
                    limit_date = datetime.now(timezone.utc) - timedelta(days=conversations_ttl_days)
                    cur.execute("DELETE FROM conversations WHERE ts < %s", (limit_date,))
                    deleted["conversations"] = cur.rowcount
                if memories_ttl_days is not None and memories_ttl_days > 0:
                    limit_date = datetime.now(timezone.utc) - timedelta(days=memories_ttl_days)
                    cur.execute("DELETE FROM memory_entries WHERE updated_at < %s", (limit_date,))
                    deleted["memory_entries"] = cur.rowcount
                if delegations_ttl_days is not None and delegations_ttl_days > 0:
                    limit_date = datetime.now(timezone.utc) - timedelta(days=delegations_ttl_days)
                    cur.execute("DELETE FROM delegations WHERE ts < %s", (limit_date,))
                    deleted["delegations"] = cur.rowcount
                conn.commit()
        return deleted

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

    def get_metrics_data(self) -> Dict[str, Any]:
        """Fetch detailed metrics from the database for Prometheus output."""
        data = {
            "memory_entries": [],
            "memory_entries_compressed": [],
            "conversations": [],
            "delegations": [],
            "feedback": [],
            "conversation_recalls": [],
            "delegation_recalls": [],
            "memory_entities": [],
            "conversation_entities": [],
        }

        # Helper to execute query and return list of rows
        def query_safe(sql: str, params: Optional[list] = None) -> list:
            try:
                with self._get_pool().connection() as conn:
                    with conn.cursor(row_factory=dict_row) as cur:
                        cur.execute(sql, params or [])
                        return list(cur.fetchall())
            except Exception as exc:
                logger.warning("Metrics query failed (%s): %s", sql[:50], exc)
                return []

        # 1. Memory entries by agent and target
        data["memory_entries"] = query_safe(
            "SELECT agent_identity, target, COUNT(*) as count FROM memory_entries GROUP BY agent_identity, target"
        )

        # 2. Compressed memory entries
        data["memory_entries_compressed"] = query_safe(
            "SELECT agent_identity, COUNT(*) as count FROM memory_entries WHERE compressed IS NOT NULL GROUP BY agent_identity"
        )

        # 3. Conversations by agent and role
        data["conversations"] = query_safe(
            "SELECT agent_identity, role, COUNT(*) as count FROM conversations GROUP BY agent_identity, role"
        )

        # 4. Delegations by agent
        data["delegations"] = query_safe(
            "SELECT agent_identity, COUNT(*) as count FROM delegations GROUP BY agent_identity"
        )

        # 5. Feedback and recall counts on memory entries
        data["feedback"] = query_safe(
            """
            SELECT 
                agent_identity,
                SUM(COALESCE((metadata->>'recall_count')::int, 0)) AS total_recalls,
                SUM(COALESCE((metadata->>'confirm_count')::int, 0)) AS total_confirms,
                SUM(COALESCE((metadata->>'reject_count')::int, 0)) AS total_rejects
            FROM memory_entries
            GROUP BY agent_identity
            """
        )

        # 6. Conversation recalls
        data["conversation_recalls"] = query_safe(
            """
            SELECT 
                agent_identity,
                SUM(COALESCE((metadata->>'recall_count')::int, 0)) AS total_recalls
            FROM conversations
            GROUP BY agent_identity
            """
        )

        # 7. Delegation recalls
        data["delegation_recalls"] = query_safe(
            """
            SELECT 
                agent_identity,
                SUM(COALESCE((metadata->>'recall_count')::int, 0)) AS total_recalls
            FROM delegations
            GROUP BY agent_identity
            """
        )

        # 8. Memory entry entities
        data["memory_entities"] = query_safe(
            """
            SELECT
                agent_identity,
                COUNT(DISTINCT ((e->>'type') || ':' || (e->>'value'))) AS unique_entities,
                COUNT(*) AS total_entity_occurrences
            FROM memory_entries,
                 jsonb_array_elements(COALESCE(metadata->'entities', '[]'::jsonb)) AS e
            GROUP BY agent_identity
            """
        )

        # 9. Conversation entities
        data["conversation_entities"] = query_safe(
            """
            SELECT
                agent_identity,
                COUNT(DISTINCT ((e->>'type') || ':' || (e->>'value'))) AS unique_entities,
                COUNT(*) AS total_entity_occurrences
            FROM conversations,
                 jsonb_array_elements(COALESCE(metadata->'entities', '[]'::jsonb)) AS e
            GROUP BY agent_identity
            """
        )

        return data

    def _apply_recall_boost(self, rows: List[Dict[str, Any]], boost_weight: float) -> List[Dict[str, Any]]:
        if boost_weight <= 0.0:
            return rows
        for r in rows:
            meta = r.get("metadata") or {}
            try:
                recall_count = int(meta.get("recall_count", 0))
            except (ValueError, TypeError):
                recall_count = 0
            # log-based boost: score * (1 + boost_weight * log(1 + recall_count))
            r["score"] = r["score"] * (1.0 + boost_weight * math.log(1 + recall_count))
        return rows

    def _apply_temporal_decay(self, rows: List[Dict[str, Any]], half_life_days: float) -> List[Dict[str, Any]]:
        if half_life_days <= 0.0:
            return rows
        now = datetime.now(timezone.utc)
        for r in rows:
            ts_val = r.get("updated_at") or r.get("ts") or r.get("created_at")
            if isinstance(ts_val, str):
                try:
                    from datetime import datetime as dt
                    ts_val = dt.fromisoformat(ts_val)
                except Exception:
                    continue
            if not ts_val:
                continue
            
            # Ensure ts_val has timezone info (psycopg datetimes are timezone-aware, now is utc)
            if ts_val.tzinfo is None:
                ts_val = ts_val.replace(tzinfo=timezone.utc)
                
            age_days = (now - ts_val).total_seconds() / 86400.0
            # exponential decay: score * 2^(-age/half_life)
            decay = math.exp(-math.log(2.0) * age_days / half_life_days)
            r["score"] = r["score"] * decay
        return rows

    def _apply_min_confidence(self, rows: List[Dict[str, Any]], min_confidence: float) -> List[Dict[str, Any]]:
        if min_confidence <= 0.0:
            return rows
        filtered = []
        for r in rows:
            meta = r.get("metadata") or {}
            try:
                confirms = int(meta.get("confirm_count") or 0)
            except (ValueError, TypeError):
                confirms = 0
            try:
                rejects = int(meta.get("reject_count") or 0)
            except (ValueError, TypeError):
                rejects = 0
            total = confirms + rejects
            if total > 0:
                ratio = confirms / total
                if ratio >= min_confidence:
                    filtered.append(r)
            else:
                filtered.append(r)
        return filtered

    def confirm_entry(self, entry_id: int) -> bool:
        """Increment confirm_count in metadata JSONB for the given entry ID."""
        sql = """
        UPDATE memory_entries
        SET metadata = jsonb_set(
            metadata,
            '{confirm_count}',
            (COALESCE(metadata->>'confirm_count', '0')::int + 1)::text::jsonb
        )
        WHERE id = %s
        """
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (entry_id,))
                updated = cur.rowcount
                conn.commit()
                return updated > 0

    def reject_entry(self, entry_id: int) -> bool:
        """Increment reject_count in metadata JSONB for the given entry ID."""
        sql = """
        UPDATE memory_entries
        SET metadata = jsonb_set(
            metadata,
            '{reject_count}',
            (COALESCE(metadata->>'reject_count', '0')::int + 1)::text::jsonb
        )
        WHERE id = %s
        """
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (entry_id,))
                updated = cur.rowcount
                conn.commit()
                return updated > 0


    def increment_recall_counts(self, table: str, ids: List[int]) -> None:
        if not ids:
            return
        sql = f"""
        UPDATE {table}
        SET metadata = jsonb_set(
            metadata, 
            '{{recall_count}}', 
            (COALESCE(metadata->>'recall_count', '0')::int + 1)::text::jsonb
        )
        WHERE id = ANY(%s)
        """
        try:
            with self._get_pool().connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (ids,))
                    conn.commit()
        except Exception as exc:
            logger.warning("Failed to increment recall counts for %s: %s", table, exc)

    def entity_graph(
        self,
        *,
        entity_type: str,
        entity_value: str,
        agent_identity: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Find other entities that co-occur with the given entity."""
        query_entity_json = json.dumps([{"type": entity_type, "value": entity_value}])
        
        sql = """
        WITH source_entries AS (
            SELECT id, content, metadata, updated_at
            FROM memory_entries
            WHERE metadata->'entities' @> %s::jsonb
              AND (%s::text IS NULL OR agent_identity = %s)
        ),
        related_entities AS (
            SELECT
                e->>'type' AS ent_type,
                e->>'value' AS ent_value,
                COUNT(*) AS co_occurrences,
                (ARRAY_AGG(content ORDER BY updated_at DESC))[1] AS sample_content
            FROM source_entries,
                 jsonb_array_elements(metadata->'entities') AS e
            WHERE (e->>'type', e->>'value') != (%s, %s)
            GROUP BY e->>'type', e->>'value'
        )
        SELECT ent_type AS type, ent_value AS value, co_occurrences, sample_content
        FROM related_entities
        ORDER BY co_occurrences DESC
        LIMIT %s
        """
        params = [query_entity_json, agent_identity, agent_identity, entity_type, entity_value, limit]
        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                related = list(cur.fetchall())
                return {
                    "entity": {"type": entity_type, "value": entity_value},
                    "related": related,
                }

    def graph_walk(
        self,
        *,
        entity_type: str,
        entity_value: str,
        agent_identity: Optional[str] = None,
        max_depth: int = 2,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Perform recursive CTE path traversal from a start entity."""
        sql = """
        WITH RECURSIVE graph_walk AS (
            -- Anchor member: Hop 1 co-occurring entities
            SELECT 
                e->>'type' AS ent_type,
                e->>'value' AS ent_value,
                1 AS depth,
                ARRAY[jsonb_build_object('type', %s::text, 'value', %s::text)] AS path
            FROM memory_entries,
                 jsonb_array_elements(metadata->'entities') AS e
            WHERE metadata->'entities' @> jsonb_build_array(jsonb_build_object('type', %s::text, 'value', %s::text))
              AND (%s::text IS NULL OR agent_identity = %s)
              AND (e->>'type', e->>'value') != (%s, %s)

            UNION ALL

            -- Recursive member: Hop N
            SELECT 
                next_e->>'type' AS ent_type,
                next_e->>'value' AS ent_value,
                gw.depth + 1 AS depth,
                gw.path || jsonb_build_object('type', gw.ent_type, 'value', gw.ent_value) AS path
            FROM graph_walk gw
            JOIN memory_entries m
              ON m.metadata->'entities' @> jsonb_build_array(jsonb_build_object('type', gw.ent_type, 'value', gw.ent_value))
            CROSS JOIN LATERAL jsonb_array_elements(m.metadata->'entities') AS next_e
            WHERE gw.depth < %s
              -- Scoping
              AND (%s::text IS NULL OR m.agent_identity = %s)
              -- Avoid cycles
              AND NOT (jsonb_build_object('type', next_e->>'type', 'value', next_e->>'value') = ANY(gw.path))
              AND (next_e->>'type', next_e->>'value') != (%s, %s)
        )

        SELECT ent_type AS type, ent_value AS value, MIN(depth) AS min_depth, COUNT(*) AS occurrences
        FROM graph_walk
        GROUP BY ent_type, ent_value
        ORDER BY min_depth ASC, occurrences DESC
        LIMIT %s
        """
        params = [
            entity_type, entity_value,
            entity_type, entity_value,
            agent_identity, agent_identity,
            entity_type, entity_value,
            max_depth,
            agent_identity, agent_identity,
            entity_type, entity_value,
            limit
        ]
        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())

    def common_topics(
        self,
        *,
        agent_identity: Optional[str] = None,
        min_strength: int = 2,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find clusters of heavily co-occurring entities/topics."""
        sql = """
        SELECT 
            e1->>'type' AS type_a,
            e1->>'value' AS value_a,
            e2->>'type' AS type_b,
            e2->>'value' AS value_b,
            COUNT(*) AS strength
        FROM memory_entries,
             jsonb_array_elements(metadata->'entities') e1,
             jsonb_array_elements(metadata->'entities') e2
        WHERE e1->>'value' < e2->>'value' -- Avoid duplicate A-B/B-A and self-pairing
          AND (%s::text IS NULL OR agent_identity = %s)
        GROUP BY type_a, value_a, type_b, value_b
        HAVING COUNT(*) >= %s
        ORDER BY strength DESC
        LIMIT %s
        """
        params = [agent_identity, agent_identity, min_strength, limit]
        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())

    def summarize_session(
        self,
        *,
        session_id: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Compute the vector centroid of a session's turns and find the K closest turns."""
        count_sql = "SELECT COUNT(*) FROM conversations WHERE session_id = %s"
        sql = """
        WITH centroid AS (
            SELECT AVG(embedding) AS vec
            FROM conversations
            WHERE session_id = %s
        )
        SELECT id, role, content, ts, metadata,
               1 - (embedding <=> (SELECT vec FROM centroid)) AS centrality_score
        FROM conversations
        WHERE session_id = %s
          AND embedding IS NOT NULL
        ORDER BY embedding <=> (SELECT vec FROM centroid)
        LIMIT %s
        """
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(count_sql, (session_id,))
                total_turns = cur.fetchone()[0]

            if total_turns == 0:
                return {
                    "session_id": session_id,
                    "turn_count": 0,
                    "summary_turns": [],
                }

            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, (session_id, session_id, limit))
                rows = list(cur.fetchall())
                for r in rows:
                    if r.get("ts") and hasattr(r["ts"], "isoformat"):
                        r["ts"] = r["ts"].isoformat()
                return {
                    "session_id": session_id,
                    "turn_count": total_turns,
                    "summary_turns": rows,
                }


# TODO: Implement a background LLM reflection/consolidation loop (Lightweight LLM Re-Ranking / Reflection Loop)
# that runs when the agent is idle to group/summarize low-confidence or heavily co-occurring memory entries.

    def consolidate_low_confidence_memories(self, agent_identity: Optional[str] = None) -> Dict[str, Any]:
        """Query low-confidence (frequently rejected) memories and send them to the LLM for pruning/merging."""
        import json
        import urllib.request
        import urllib.error
        import os
        from hexus.store import dict_row

        api_base = os.environ.get("LLM_API_BASE") or "http://headroom:8787/v1"
        api_key = os.environ.get("HEADROOM_INTERNAL_TOKEN") or os.environ.get("LITELLM_MASTER_KEY")
        summary_model = os.environ.get("HEXUS_SUMMARY_MODEL")
        
        if not summary_model:
            logger.warning("HEXUS_SUMMARY_MODEL is not set. Skipping low-confidence memory consolidation.")
            return {"status": "skipped", "reason": "HEXUS_SUMMARY_MODEL not set"}

        # 1. Fetch candidates
        query = """
            SELECT id, agent_identity, target, content, metadata
            FROM memory_entries
            WHERE (metadata->>'reject_count')::int > 0
              AND (
                metadata->>'confirm_count' IS NULL 
                OR (metadata->>'confirm_count')::int = 0
                OR (metadata->>'confirm_count')::float / ((metadata->>'confirm_count')::float + (metadata->>'reject_count')::float) < 0.3
              )
        """
        params = []
        if agent_identity:
            query += " AND agent_identity = %s"
            params.append(agent_identity)
        query += " LIMIT 20"

        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) if hasattr(conn, "cursor") else conn.cursor() as cur:
                cur.execute(query, params)
                rows = list(cur.fetchall())

        if not rows:
            return {"status": "ok", "processed": 0, "deletions": 0, "replacements": 0}

        # 2. Call LLM
        memories_str = "\n".join([f"- [ID: {r['id']}] (Target: {r['target']}) \"{r['content']}\"" for r in rows])
        prompt = (
            "You are an AI memory manager. Analyze the following low-confidence/frequently rejected memory entries "
            "and decide what to do with them. Redundant, contradiction, obsolete, or incorrect facts should be deleted. "
            "Related facts should be merged. Valid facts should be kept.\n\n"
            f"Memories:\n{memories_str}\n\n"
            "Respond ONLY with a JSON object matching this schema. Do not include markdown code block formatting or explanations:\n"
            "{\n"
            "  \"deletions\": [id, id, ...],\n"
            "  \"replacements\": [\n"
            "    {\"ids\": [id, id, ...], \"content\": \"consolidated text\", \"target\": \"memory\"}\n"
            "  ]\n"
            "}"
        )

        payload = {
            "model": summary_model,
            "messages": [
                {"role": "system", "content": "You are a database cleanup manager. Output ONLY raw valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 800
        }

        url = f"{api_base.rstrip('/')}/chat/completions"
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                llm_response = resp_data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.error("Failed to query LLM for low-confidence memory consolidation: %s", exc)
            return {"status": "error", "reason": f"LLM query failed: {exc}"}

        # Parse JSON
        try:
            if llm_response.startswith("```"):
                lines = llm_response.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                llm_response = "\n".join(lines).strip()
            data = json.loads(llm_response)
        except Exception as exc:
            logger.error("Failed to parse LLM consolidation response JSON: %s. Response: %r", exc, llm_response)
            return {"status": "error", "reason": f"JSON parse failed: {exc}"}

        deletions = data.get("deletions", [])
        replacements = data.get("replacements", [])

        deleted_count = 0
        replaced_count = 0

        # Process deletions
        if deletions:
            valid_del_ids = [r["id"] for r in rows if r["id"] in deletions]
            if valid_del_ids:
                with self._get_pool().connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM memory_entries WHERE id = ANY(%s)", (valid_del_ids,))
                        deleted_count += cur.rowcount
                        conn.commit()

        # Process replacements
        for rep in replacements:
            rep_ids = rep.get("ids", [])
            rep_content = rep.get("content", "").strip()
            rep_target = rep.get("target", "memory")
            
            valid_rep_ids = [r["id"] for r in rows if r["id"] in rep_ids]
            if valid_rep_ids and rep_content:
                with self._get_pool().connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM memory_entries WHERE id = ANY(%s)", (valid_rep_ids,))
                        conn.commit()
                agent_id = next((r["agent_identity"] for r in rows if r["id"] in valid_rep_ids), agent_identity or "default")
                self.add(agent_identity=agent_id, target=rep_target, content=rep_content)
                replaced_count += len(valid_rep_ids)

        return {
            "status": "ok",
            "processed": len(rows),
            "deletions": deleted_count,
            "replacements": replaced_count,
        }

    def consolidate_cooccurring_memories(self, agent_identity: Optional[str] = None) -> Dict[str, Any]:
        """Query clusters of co-occurring entities and consolidate their memories using the LLM."""
        import json
        import urllib.request
        import urllib.error
        import os
        from hexus.store import dict_row

        api_base = os.environ.get("LLM_API_BASE") or "http://headroom:8787/v1"
        api_key = os.environ.get("HEADROOM_INTERNAL_TOKEN") or os.environ.get("LITELLM_MASTER_KEY")
        summary_model = os.environ.get("HEXUS_SUMMARY_MODEL")
        
        if not summary_model:
            logger.warning("HEXUS_SUMMARY_MODEL is not set. Skipping co-occurring memory consolidation.")
            return {"status": "skipped", "reason": "HEXUS_SUMMARY_MODEL not set"}

        topics = self.common_topics(agent_identity=agent_identity, min_strength=3, limit=5)
        if not topics:
            return {"status": "ok", "processed_topics": 0, "replacements": 0}

        processed_topics = 0
        replaced_count = 0
        processed_ids = set()

        for topic in topics:
            type_a, val_a = topic["type_a"], topic["value_a"]
            type_b, val_b = topic["type_b"], topic["value_b"]

            query = """
                SELECT id, agent_identity, target, content, metadata
                FROM memory_entries
                WHERE (%s::text IS NULL OR agent_identity = %s)
                  AND metadata @> %s::jsonb
                  AND metadata @> %s::jsonb
                LIMIT 10
            """
            meta_a = json.dumps({"entities": [{"type": type_a, "value": val_a}]})
            meta_b = json.dumps({"entities": [{"type": type_b, "value": val_b}]})
            params = [agent_identity, agent_identity, meta_a, meta_b]

            with self._get_pool().connection() as conn:
                with conn.cursor(row_factory=dict_row) if hasattr(conn, "cursor") else conn.cursor() as cur:
                    cur.execute(query, params)
                    rows = list(cur.fetchall())

            rows = [r for r in rows if r["id"] not in processed_ids]

            if len(rows) < 3:
                continue

            processed_topics += 1

            memories_str = "\n".join([f"- [ID: {r['id']}] (Target: {r['target']}) \"{r['content']}\"" for r in rows])
            prompt = (
                "You are an AI memory manager. The following memory entries relate to the same concepts "
                f"({val_a} and {val_b}) and contain redundant or overlapping information.\n\n"
                f"Memories:\n{memories_str}\n\n"
                "Consolidate and summarize these entries into a single, high-signal, comprehensive memory entry "
                "to clean up the database. Respond ONLY with a JSON object matching this schema. Do not include markdown code block formatting or explanations:\n"
                "{\n"
                "  \"ids_to_replace\": [id, id, ...],\n"
                "  \"consolidated_content\": \"consolidated text\",\n"
                "  \"target\": \"memory\"\n"
                "}"
            )

            payload = {
                "model": summary_model,
                "messages": [
                    {"role": "system", "content": "You are a database consolidation manager. Output ONLY raw valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 800
            }

            url = f"{api_base.rstrip('/')}/chat/completions"
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                if api_key:
                    req.add_header("Authorization", f"Bearer {api_key}")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    llm_response = resp_data["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                logger.error("Failed to query LLM for topic consolidation: %s", exc)
                continue

            try:
                if llm_response.startswith("```"):
                    lines = llm_response.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].startswith("```"):
                        lines = lines[:-1]
                    llm_response = "\n".join(lines).strip()
                data = json.loads(llm_response)
            except Exception as exc:
                logger.error("Failed to parse LLM topic consolidation JSON: %s. Response: %r", exc, llm_response)
                continue

            ids_to_replace = data.get("ids_to_replace", [])
            consolidated_content = data.get("consolidated_content", "").strip()
            rep_target = data.get("target", "memory")

            valid_ids = [r["id"] for r in rows if r["id"] in ids_to_replace]
            if len(valid_ids) >= 2 and consolidated_content:
                with self._get_pool().connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM memory_entries WHERE id = ANY(%s)", (valid_ids,))
                        conn.commit()
                agent_id = next((r["agent_identity"] for r in rows if r["id"] in valid_ids), agent_identity or "default")
                self.add(agent_identity=agent_id, target=rep_target, content=consolidated_content)
                replaced_count += len(valid_ids)
                processed_ids.update(valid_ids)

        return {
            "status": "ok",
            "processed_topics": processed_topics,
            "replacements": replaced_count,
        }




