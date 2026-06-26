"""mcp_server.tools — pure functions implementing the MCP tool surface.

Forked from andreab67/hermes-hexus (BSD-3-Clause).

These functions are transport-agnostic; `server.py` wraps each one as an
MCP tool. They're also the unit-test surface — every tool is testable
without spinning up an MCP transport, just by calling the function with
a `MemoryStore` and a dict of arguments.

Design contract:
  - Every function takes `store: MemoryStore` as the first positional
    arg (dependency injection — no global state).
  - Every function takes `args: dict` as the second arg (matches the MCP
    SDK's tool-call argument shape).
  - Every function returns a JSON-serializable `dict`. The wrapper in
    server.py is responsible for turning that into a `CallToolResult`.
  - Every function catches its own exceptions and returns them as
    `{"error": "..."}` rather than raising — MCP tool handlers should
    never crash the host process.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from hexus.embed import EmbeddingError, embed
from hexus.store import MemoryStore

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Default agent_identity resolution. Multi-agent deployments want each
# connected client to default to a different theme; this is the one
# place that the MCP server "leaks" process-level identity into the
# tool surface. The default can be overridden per-call.
# -----------------------------------------------------------------------




def default_agent_identity() -> str:
    """Return the default `agent_identity` to use when a tool call omits it.

    Resolution order:
      1. HEXUS_AGENT_IDENTITY env var (set per-client process
         in the docker compose `mcp` service, or in the host environment
         for a stdio-launched client)
      2. "default" — same as the upstream plugin's last-resort bucket.
    """
    return os.environ.get("HEXUS_AGENT_IDENTITY", "default")


# -----------------------------------------------------------------------
# Tool implementations
# -----------------------------------------------------------------------


def memory_health(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Liveness + capability check. Useful for MCP client setup probes.

    Returns the store's status, the embedder model name + dim, and a
    row count. Always succeeds if Postgres is reachable, even if the
    embedder isn't loaded yet (lazy load).
    """
    from hexus.embedder import DEFAULT_MODEL, DEFAULT_DIM

    try:
        store.ensure_schema()
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": f"schema check failed: {exc}",
            "embedder": {"model": DEFAULT_MODEL, "dim": DEFAULT_DIM},
        }

    n_entries = store.count(agent_identity=None, target=None) if hasattr(store, "count") else None
    return {
        "status": "ok",
        "schema_ok": True,
        "embedder": {
            "model": DEFAULT_MODEL,
            "dim": DEFAULT_DIM,
            "eager_loaded": os.environ.get("HEXUS_EMBED_EAGER_LOAD", "0")
            == "1",
        },
        "row_counts": {
            "memory_entries": n_entries,
        },
    }


def _coerce_agent_identity(args: Dict[str, Any]) -> str:
    """Read agent_identity from args, defaulting to env / 'default'."""
    a = args.get("agent_identity")
    if isinstance(a, str) and a.strip():
        return a.strip()
    return default_agent_identity()


def _coerce_target(args: Dict[str, Any]) -> Optional[str]:
    """target ∈ {'memory', 'user', None}. Anything else is rejected.
    
    Empty string is treated as None to match MCP clients that send
    default values as '' rather than omitting the field.
    """
    t = args.get("target")
    if t is None or (isinstance(t, str) and not t.strip()):
        return None
    if t not in ("memory", "user"):
        raise ValueError(f"target must be 'memory' or 'user' (got {t!r})")
    return t


def memory_retain(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Add one or many memory entries. Mirrors the plugin's `on_memory_write`.

    args:
      contents: list[str]  — text to store (one row per element)
      target:   'memory' | 'user' | None (default = 'memory')
      metadata: dict | list[dict] | None — per-item metadata
      agent_identity: str | None (default = env / 'default')
      doc_type: 'document' | 'note' | 'memory'  (default 'memory') — stored in metadata
      source_url: str | None  — stored in metadata['source_url']

    Returns: {"inserted": N, "duplicates": K, "errors": [...]}
    """
    contents = args.get("contents")
    if not isinstance(contents, list) or not contents:
        raise ValueError("contents must be a non-empty list of strings")
    for i, c in enumerate(contents):
        if not isinstance(c, str) or not c.strip():
            raise ValueError(f"contents[{i}] must be a non-empty string")

    target = _coerce_target(args) or "memory"
    agent = _coerce_agent_identity(args)
    doc_type = args.get("doc_type", "memory")
    source_url = args.get("source_url")
    metadata_in = args.get("metadata")

    # Normalize metadata to one entry per content
    if metadata_in is None:
        metas: List[Optional[Dict[str, Any]]] = [None] * len(contents)
    elif isinstance(metadata_in, list):
        if len(metadata_in) != len(contents):
            raise ValueError("metadata list length must match contents length")
        metas = [m if isinstance(m, dict) else None for m in metadata_in]
    elif isinstance(metadata_in, dict):
        metas = [dict(metadata_in) for _ in contents]
    else:
        raise ValueError("metadata must be a dict, a list of dicts, or None")

    # Stamp each item with doc_type + source_url
    stamped: List[Dict[str, Any]] = []
    for m in metas:
        out = dict(m) if m else {}
        if doc_type and "doc_type" not in out:
            out["doc_type"] = doc_type
        if source_url and "source_url" not in out:
            out["source_url"] = source_url
        stamped.append(out)

    # Embed once. We pass the full list so the local embedder can batch.
    try:
        vectors = _embed_batch(contents)
    except EmbeddingError as exc:
        return {"inserted": 0, "duplicates": 0, "errors": [str(exc)]}
    except Exception as exc:  # noqa: BLE001
        return {"inserted": 0, "duplicates": 0, "errors": [f"embed failed: {exc}"]}

    inserted = 0
    duplicates = 0
    errors: List[str] = []
    for content, vec, meta in zip(contents, vectors, stamped):
        try:
            row_id = store.add(
                agent_identity=agent,
                target=target,
                content=content,
                embedding=vec,
                metadata=meta or None,
            )
            if row_id is None:
                duplicates += 1
            else:
                inserted += 1
                webhook_url = os.environ.get("HEXUS_WEBHOOK_URL")
                if webhook_url:
                    from hexus.webhook.dispatcher import dispatch_webhook
                    dispatch_webhook(
                        url=webhook_url,
                        secret=os.environ.get("HEXUS_WEBHOOK_SECRET"),
                        event="memory_retain",
                        payload={
                            "agent_identity": agent,
                            "target": target,
                            "content": content,
                            "metadata": meta or {},
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"add failed for {content[:40]!r}: {exc}")
    return {"inserted": inserted, "duplicates": duplicates, "errors": errors}


def memory_recall(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Semantic search over memory_entries.

    args:
      query:   str
      top_k:   int (default 5, cap 100)
      agent_identity: str | None — if None, search ALL agents
      target:  'memory' | 'user' | None
      min_similarity: float ∈ [0, 1] (default 0)

    Returns: {"query": str, "count": N, "results": [{id, agent_identity, target, content, score, metadata, ...}]}
    """
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    top_k = int(args.get("top_k", 5))
    if top_k < 1:
        top_k = 1
    if top_k > 100:
        top_k = 100

    agent = args.get("agent_identity")
    if isinstance(agent, str) and agent.strip() == "":
        agent = None

    target = _coerce_target(args)
    min_similarity = float(args.get("min_similarity", 0.0))
    if min_similarity < 0:
        min_similarity = 0.0
    if min_similarity > 1:
        min_similarity = 1.0

    min_confidence = float(args.get("min_confidence", 0.0))
    min_confidence = max(0.0, min(min_confidence, 1.0))

    decay_val = args.get("decay_half_life_days")
    if decay_val is None:
        decay_val = os.environ.get("HEXUS_DECAY_HALF_LIFE_DAYS", 0.0)
    decay_half_life_days = max(0.0, float(decay_val))

    boost_val = args.get("recall_boost_weight")
    if boost_val is None:
        boost_val = os.environ.get("HEXUS_RECALL_BOOST_WEIGHT", 0.0)
    recall_boost_weight = max(0.0, float(boost_val))

    try:
        vec = embed(query)
    except EmbeddingError as exc:
        return {"query": query, "count": 0, "results": [], "error": str(exc)}

    rows = store.search(
        query_embedding=vec,
        agent_identity=agent,
        target=target,
        limit=top_k,
        min_similarity=min_similarity,
        min_confidence=min_confidence,
        decay_half_life_days=decay_half_life_days,
        recall_boost_weight=recall_boost_weight,
    )
    return {
        "query": query,
        "count": len(rows),
        "results": [_row_to_dict(r) for r in rows],
    }



def memory_search(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """List entries (no embedding) — browse / paginate / inspect.

    args:
      agent_identity: str | None (default = env)
      target:  'memory' | 'user' | None
      limit:   int (default 20, cap 200)
      offset:  int (default 0)

    Returns: {"count": N, "rows": [...], "limit": L, "offset": O}
    """
    agent = args.get("agent_identity")
    if agent is None or (isinstance(agent, str) and not agent.strip()):
        agent = _coerce_agent_identity(args)
    target = _coerce_target(args)
    limit = int(args.get("limit", 20))
    if limit < 1:
        limit = 1
    if limit > 200:
        limit = 200
    offset = int(args.get("offset", 0))
    if offset < 0:
        offset = 0

    rows = store.list_entries(
        agent_identity=agent, target=target, limit=limit, offset=offset
    )

    return {
        "count": len(rows),
        "limit": limit,
        "offset": offset,
        "rows": [_row_to_dict(r, include_embedding=False) for r in rows],
    }


def memory_forget(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a memory entry by id. Requires `confirm=true` to actually
    delete; without it the call is a dry-run that reports what would
    happen. This makes "drop everything matching a query" hard to do
    by accident across an MCP client that mis-issued a request."""
    entry_id = args.get("id")
    if not isinstance(entry_id, int) or entry_id <= 0:
        raise ValueError("id must be a positive integer")
    if not args.get("confirm", False):
        return {
            "deleted": 0,
            "dry_run": True,
            "would_delete_id": entry_id,
            "hint": "pass confirm=true to actually delete",
        }
    agent = args.get("agent_identity") or _coerce_agent_identity(args)
    with store._get_pool().connection() as conn:  # noqa: SLF001 — admin path
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM memory_entries WHERE id = %s AND agent_identity = %s RETURNING id, target, content",
                (entry_id, agent),
            )
            row = cur.fetchone()
            conn.commit()

    if row:
        webhook_url = os.environ.get("HEXUS_WEBHOOK_URL")
        if webhook_url:
            from hexus.webhook.dispatcher import dispatch_webhook
            dispatch_webhook(
                url=webhook_url,
                secret=os.environ.get("HEXUS_WEBHOOK_SECRET"),
                event="memory_forget",
                payload={
                    "agent_identity": agent,
                    "target": row[1],
                    "content": row[2],
                }
            )

    return {
        "deleted": 1 if row else 0,
        "dry_run": False,
        "id": entry_id,
        "agent_identity": agent,
    }


def memory_recall_turns(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Semantic search over conversation turns. Mirrors `recall_conversation`.

    args:
      query:   str
      top_k:   int (default 5, cap 100)
      agent_identity: str | None
      session_id: str | None
      min_similarity: float ∈ [0, 1] (default 0)

    Returns: {"query": str, "count": N, "results": [{id, session_id, agent_identity, role, content, score, ts, metadata}]}
    """
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    top_k = int(args.get("top_k", 5))
    if top_k < 1:
        top_k = 1
    if top_k > 100:
        top_k = 100
    agent = args.get("agent_identity")
    if isinstance(agent, str) and agent.strip() == "":
        agent = None
    session_id = args.get("session_id")
    if isinstance(session_id, str) and session_id.strip() == "":
        session_id = None
    min_similarity = float(args.get("min_similarity", 0.0))

    decay_val = args.get("decay_half_life_days")
    if decay_val is None:
        decay_val = os.environ.get("HEXUS_DECAY_HALF_LIFE_DAYS", 0.0)
    decay_half_life_days = max(0.0, float(decay_val))

    boost_val = args.get("recall_boost_weight")
    if boost_val is None:
        boost_val = os.environ.get("HEXUS_RECALL_BOOST_WEIGHT", 0.0)
    recall_boost_weight = max(0.0, float(boost_val))

    try:
        vec = embed(query)
    except EmbeddingError as exc:
        return {"query": query, "count": 0, "results": [], "error": str(exc)}

    rows = store.search_turns(
        query_embedding=vec,
        agent_identity=agent,
        session_id=session_id,
        limit=top_k,
        min_similarity=min_similarity,
        decay_half_life_days=decay_half_life_days,
        recall_boost_weight=recall_boost_weight,
    )
    return {
        "query": query,
        "count": len(rows),
        "results": [_row_to_dict(r) for r in rows],
    }


def memory_append_turn(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Append one chat turn. Mirrors the plugin's `sync_turn` capture.

    args:
      session_id: str
      role: 'user' | 'assistant' | 'system' | 'tool'
      content: str
      agent_identity: str | None (default = env)
      metadata: dict | None

    Returns: {"id": N}
    """
    session_id = args.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    role = args.get("role")
    if role not in ("user", "assistant", "system", "tool"):
        raise ValueError("role must be one of user/assistant/system/tool")
    content = args.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content must be a non-empty string")
    agent = _coerce_agent_identity(args)
    metadata = args.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict or None")

    try:
        vec = embed(content)
    except EmbeddingError as exc:
        return {"error": f"embed failed: {exc}"}

    row_id = store.append_turn(
        session_id=session_id,
        agent_identity=agent,
        role=role,
        content=content,
        embedding=vec,
        metadata=metadata,
    )
    return {"id": row_id, "session_id": session_id, "role": role}


def memory_count(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Return row counts for entries and turns, scoped as requested.

    args:
      agent_identity: str | None
      target: 'memory' | 'user' | None
      session_id: str | None  (turns only)

    Returns: {"memory_entries": N, "conversations": M, "agent_identity": ..., "target": ...}
    """
    agent = args.get("agent_identity")
    if agent is None or (isinstance(agent, str) and not agent.strip()):
        agent = _coerce_agent_identity(args)
    target = _coerce_target(args)
    session_id = args.get("session_id")
    if isinstance(session_id, str) and session_id.strip() == "":
        session_id = None

    return {
        "memory_entries": store.count(agent_identity=agent, target=target),
        "conversations": store.count_turns(
            agent_identity=agent, session_id=session_id
        ),
        "agent_identity": agent,
        "target": target,
        "session_id": session_id,
    }


# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------


def _embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts. For the local path this batches in one
    model.encode() call. The HTTP path embeds one at a time (limitation
    of the upstream embed() function) — fine for low-volume MCP traffic,
    not for bulk import. Callers that need bulk import should use the
    underlying `hexus.embedder.LocalBertEmbedder` directly."""
    return [embed(t) for t in texts]


def _row_to_dict(row: Any, *, include_embedding: bool = False) -> Dict[str, Any]:
    """Coerce a DB row (psycopg dict_row or plain tuple) into a JSON-safe dict."""
    if hasattr(row, "keys"):
        d = dict(row)
    else:
        d = dict(row._asdict()) if hasattr(row, "_asdict") else dict(row)
    # Strip the embedding vector from results by default — it's 384 floats
    # and bloats the JSON for no good reason on recall responses.
    if not include_embedding and "embedding" in d:
        d.pop("embedding", None)
    # Datetimes → ISO strings
    for k, v in list(d.items()):
        if hasattr(v, "isoformat"):
            d[k] = v.isoformat()
    return d


def memory_hybrid_search(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Blend semantic vector search and full-text search over memory_entries.

    args:
      query:   str
      top_k:   int (default 5, cap 100)
      vector_weight: float (default 0.7)
      text_weight: float (default 0.3)
      agent_identity: str | None — if None, search ALL agents
      target:  'memory' | 'user' | None
      min_similarity: float ∈ [0, 1] (default 0)

    Returns: {"query": str, "count": N, "results": [{id, agent_identity, target, content, score, vector_score, text_score, metadata, ...}]}
    """
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    top_k = int(args.get("top_k", 5))
    top_k = max(1, min(top_k, 100))

    vector_weight = float(args.get("vector_weight", 0.7))
    text_weight = float(args.get("text_weight", 0.3))

    agent = args.get("agent_identity")
    if isinstance(agent, str) and agent.strip() == "":
        agent = None

    target = _coerce_target(args)
    min_similarity = float(args.get("min_similarity", 0.0))
    min_similarity = max(0.0, min(min_similarity, 1.0))

    min_confidence = float(args.get("min_confidence", 0.0))
    min_confidence = max(0.0, min(min_confidence, 1.0))

    decay_val = args.get("decay_half_life_days")
    if decay_val is None:
        decay_val = os.environ.get("HEXUS_DECAY_HALF_LIFE_DAYS", 0.0)
    decay_half_life_days = max(0.0, float(decay_val))

    boost_val = args.get("recall_boost_weight")
    if boost_val is None:
        boost_val = os.environ.get("HEXUS_RECALL_BOOST_WEIGHT", 0.0)
    recall_boost_weight = max(0.0, float(boost_val))

    try:
        vec = embed(query)
    except EmbeddingError as exc:
        return {"query": query, "count": 0, "results": [], "error": str(exc)}

    rows = store.hybrid_search(
        query_embedding=vec,
        query_text=query,
        agent_identity=agent,
        target=target,
        limit=top_k,
        vector_weight=vector_weight,
        text_weight=text_weight,
        min_similarity=min_similarity,
        min_confidence=min_confidence,
        decay_half_life_days=decay_half_life_days,
        recall_boost_weight=recall_boost_weight,
    )
    return {
        "query": query,
        "count": len(rows),
        "results": [_row_to_dict(r) for r in rows],
    }



def memory_hybrid_recall_turns(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Blend semantic vector search and full-text search over conversation turns.

    args:
      query:   str
      top_k:   int (default 5, cap 100)
      vector_weight: float (default 0.7)
      text_weight: float (default 0.3)
      agent_identity: str | None
      session_id: str | None
      min_similarity: float ∈ [0, 1] (default 0)

    Returns: {"query": str, "count": N, "results": [{id, session_id, agent_identity, role, content, score, vector_score, text_score, ts, metadata}]}
    """
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    top_k = int(args.get("top_k", 5))
    top_k = max(1, min(top_k, 100))

    vector_weight = float(args.get("vector_weight", 0.7))
    text_weight = float(args.get("text_weight", 0.3))

    agent = args.get("agent_identity")
    if isinstance(agent, str) and agent.strip() == "":
        agent = None

    session_id = args.get("session_id")
    if isinstance(session_id, str) and session_id.strip() == "":
        session_id = None

    min_similarity = float(args.get("min_similarity", 0.0))
    min_similarity = max(0.0, min(min_similarity, 1.0))

    decay_val = args.get("decay_half_life_days")
    if decay_val is None:
        decay_val = os.environ.get("HEXUS_DECAY_HALF_LIFE_DAYS", 0.0)
    decay_half_life_days = max(0.0, float(decay_val))

    boost_val = args.get("recall_boost_weight")
    if boost_val is None:
        boost_val = os.environ.get("HEXUS_RECALL_BOOST_WEIGHT", 0.0)
    recall_boost_weight = max(0.0, float(boost_val))

    try:
        vec = embed(query)
    except EmbeddingError as exc:
        return {"query": query, "count": 0, "results": [], "error": str(exc)}

    rows = store.hybrid_search_turns(
        query_embedding=vec,
        query_text=query,
        agent_identity=agent,
        session_id=session_id,
        limit=top_k,
        vector_weight=vector_weight,
        text_weight=text_weight,
        min_similarity=min_similarity,
        decay_half_life_days=decay_half_life_days,
        recall_boost_weight=recall_boost_weight,
    )
    return {
        "query": query,
        "count": len(rows),
        "results": [_row_to_dict(r) for r in rows],
    }


def memory_record_delegation(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Record a subagent delegation.

    args:
      parent_session_id: str
      child_session_id:  str
      task:              str
      result:            str
      agent_identity:    str | None
      metadata:          dict | None
    """
    parent_session_id = args.get("parent_session_id")
    if not isinstance(parent_session_id, str) or not parent_session_id.strip():
        raise ValueError("parent_session_id must be a non-empty string")

    child_session_id = args.get("child_session_id")
    if not isinstance(child_session_id, str) or not child_session_id.strip():
        raise ValueError("child_session_id must be a non-empty string")

    task = args.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")

    result = args.get("result") or ""

    agent = args.get("agent_identity")
    if isinstance(agent, str) and agent.strip() == "":
        agent = "default"
    elif not agent:
        agent = "default"

    metadata = args.get("metadata") or {}

    combined_text = f"Task: {task}\nResult: {result}"
    try:
        vec = embed(combined_text)
    except EmbeddingError:
        vec = None

    row_id = store.record_delegation(
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
        agent_identity=agent,
        task=task,
        result=result,
        embedding=vec,
        metadata=metadata,
    )
    return {
        "id": row_id,
        "parent_session_id": parent_session_id,
        "child_session_id": child_session_id,
        "agent_identity": agent,
    }


def memory_recall_delegations(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Recall subagent delegations.

    args:
      query:                str
      top_k:                int (default 5, cap 100)
      agent_identity:       str | None
      parent_session_id:    str | None
      min_similarity:       float (default 0.0)
      decay_half_life_days: float (default 0.0)
      recall_boost_weight:  float (default 0.0)
    """
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    top_k = int(args.get("top_k", 5))
    top_k = max(1, min(top_k, 100))

    agent = args.get("agent_identity")
    if isinstance(agent, str) and agent.strip() == "":
        agent = None

    parent_session_id = args.get("parent_session_id")
    if isinstance(parent_session_id, str) and parent_session_id.strip() == "":
        parent_session_id = None

    min_similarity = float(args.get("min_similarity", 0.0))
    min_similarity = max(0.0, min(min_similarity, 1.0))

    decay_half_life_days = float(args.get("decay_half_life_days", 0.0))
    decay_half_life_days = max(0.0, decay_half_life_days)

    recall_boost_weight = float(args.get("recall_boost_weight", 0.0))
    recall_boost_weight = max(0.0, recall_boost_weight)

    try:
        vec = embed(query)
    except EmbeddingError as exc:
        return {"query": query, "count": 0, "results": [], "error": str(exc)}

    rows = store.search_delegations(
        query_embedding=vec,
        agent_identity=agent,
        parent_session_id=parent_session_id,
        limit=top_k,
        min_similarity=min_similarity,
        decay_half_life_days=decay_half_life_days,
        recall_boost_weight=recall_boost_weight,
    )
    return {
        "query": query,
        "count": len(rows),
        "results": [_row_to_dict(r) for r in rows],
    }


def memory_entity_graph(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Find other entities that co-occur with a target entity."""
    entity_type = args.get("entity_type")
    if not isinstance(entity_type, str) or not entity_type.strip():
        raise ValueError("entity_type must be a non-empty string")
    
    entity_value = args.get("entity_value")
    if not isinstance(entity_value, str) or not entity_value.strip():
        raise ValueError("entity_value must be a non-empty string")

    agent = args.get("agent_identity")
    if isinstance(agent, str) and agent.strip() == "":
        agent = None

    limit = int(args.get("limit", 5))
    limit = max(1, min(limit, 100))

    return store.entity_graph(
        entity_type=entity_type,
        entity_value=entity_value,
        agent_identity=agent,
        limit=limit,
    )


def memory_graph_walk(store: MemoryStore, args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Traverse the co-occurrence graph up to N hops away from a start entity."""
    entity_type = args.get("entity_type")
    if not isinstance(entity_type, str) or not entity_type.strip():
        raise ValueError("entity_type must be a non-empty string")
    
    entity_value = args.get("entity_value")
    if not isinstance(entity_value, str) or not entity_value.strip():
        raise ValueError("entity_value must be a non-empty string")

    agent = args.get("agent_identity")
    if isinstance(agent, str) and agent.strip() == "":
        agent = None

    max_depth = int(args.get("max_depth", 2))
    max_depth = max(1, min(max_depth, 5))

    limit = int(args.get("limit", 5))
    limit = max(1, min(limit, 100))

    return store.graph_walk(
        entity_type=entity_type,
        entity_value=entity_value,
        agent_identity=agent,
        max_depth=max_depth,
        limit=limit,
    )


def memory_common_topics(store: MemoryStore, args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Retrieve clusters/cliques of heavily co-occurring entities."""
    agent = args.get("agent_identity")
    if isinstance(agent, str) and agent.strip() == "":
        agent = None

    min_strength = int(args.get("min_strength", 2))
    min_strength = max(1, min_strength)

    limit = int(args.get("limit", 10))
    limit = max(1, min(limit, 100))

    return store.common_topics(
        agent_identity=agent,
        min_strength=min_strength,
        limit=limit,
    )


def memory_confirm(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Increment confirm_count in metadata JSONB for the given entry ID."""
    entry_id = args.get("id")
    if entry_id is None:
        raise ValueError("id is required")
    try:
        entry_id = int(entry_id)
    except (TypeError, ValueError):
        raise ValueError("id must be an integer")

    success = store.confirm_entry(entry_id)
    return {"id": entry_id, "success": success}


def memory_reject(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Increment reject_count in metadata JSONB for the given entry ID."""
    entry_id = args.get("id")
    if entry_id is None:
        raise ValueError("id is required")
    try:
        entry_id = int(entry_id)
    except (TypeError, ValueError):
        raise ValueError("id must be an integer")

    success = store.reject_entry(entry_id)
    return {"id": entry_id, "success": success}


def memory_summarize_session(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the vector centroid of a session's turns and find the K closest turns."""
    session_id = args.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")

    limit = int(args.get("limit", 5))
    limit = max(1, min(limit, 100))

    return store.summarize_session(
        session_id=session_id,
        limit=limit,
    )


def memory_retrieve(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve the original full content of a memory entry by its integer ID.

    args:
      id: the integer row ID of the memory entry to retrieve.
    """
    entry_id = args.get("id")
    if entry_id is None:
        raise ValueError("id is required")
    try:
        entry_id = int(entry_id)
    except (TypeError, ValueError):
        raise ValueError("id must be an integer")

    content = store.fetch_full(entry_id)
    if content is None:
        return {"id": entry_id, "found": False, "content": None}
    return {"id": entry_id, "found": True, "content": content}


def memory_stats(store: MemoryStore, args: Dict[str, Any]) -> Dict[str, Any]:
    """Return metrics from Hexus database and background async queue stats."""
    db_stats = {
        "memory_entries_count": store.count(agent_identity=None, target=None),
        "conversations_count": store.count_turns(agent_identity=None, session_id=None),
    }

    # Try to get queue stats from any active AsyncWriter instances
    queue_stats = {}
    import gc
    from hexus.writer import AsyncWriter
    for obj in gc.get_objects():
        if isinstance(obj, AsyncWriter):
            queue_stats = obj.stats()
            break

    cleanup_stats = getattr(store, "_cleanup_metrics", None) or {
        "total_runs": 0,
        "last_run_timestamp": 0.0,
        "deleted_conversations": 0,
        "deleted_memories": 0,
        "deleted_delegations": 0,
    }

    consolidation_stats = getattr(store, "_consolidation_metrics", None) or {
        "total_runs": 0,
        "last_run_timestamp": 0.0,
        "low_confidence_processed": 0,
        "low_confidence_deletions": 0,
        "low_confidence_replacements": 0,
        "cooccurring_processed_topics": 0,
        "cooccurring_replacements": 0,
    }

    return {
        "status": "ok",
        "database": db_stats,
        "async_writer": queue_stats or {
            "queue_size": 0,
            "queue_max": 256,
            "dropped_total": 0,
            "thread_alive": False,
            "p50_latency_sec": float("nan"),
            "p95_latency_sec": float("nan"),
        },
        "background_cleanup": cleanup_stats,
        "background_consolidation": consolidation_stats,
    }




