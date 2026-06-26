"""mcp_server.server — FastMCP wiring for hexus.

Forked from andreab67/hermes-hexus (BSD-3-Clause).

Each `@server.tool()` registers one of the pure functions in `tools.py`
as an MCP tool. The MCP transport (stdio or streamable-http) is selected
at run() time by `cli.py`.

Multi-agent: the `agent_identity` parameter on every write/read tool
keeps each connected client's data isolated. The server process can be
the same for N agents — agent isolation lives in the DB, not the
process. One model load (the LocalBertEmbedder singleton) is shared
across all of them.

The server has no opinion on transport, so the same FastMCP instance
works with:
  - `mcp.run(transport='stdio')`             for Claude Desktop / Cursor
  - `mcp.run(transport='streamable-http')`  for fleet use
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, List

from hexus.store import MemoryStore


import os
from . import tools

logger = logging.getLogger(__name__)


def _generate_metrics(store: MemoryStore) -> str:
    """Generate Prometheus metrics from DB and AsyncWriter status."""
    import math
    import gc
    from hexus.writer import AsyncWriter

    # 1. Base liveness and totals
    health = store.health()
    m_entries = store.count(agent_identity=None, target=None)
    m_turns = store.count_turns(agent_identity=None)

    lines = [
        "# HELP hexus_db_reachable Liveness check for the Postgres database (1=ok, 0=error)",
        "# TYPE hexus_db_reachable gauge",
        f"hexus_db_reachable {1 if health.get('ok') else 0}",
        "# HELP hexus_memory_entries_total Total number of stored memory entries",
        "# TYPE hexus_memory_entries_total counter",
        f"hexus_memory_entries_total {m_entries}",
        "# HELP hexus_conversation_turns_total Total number of stored conversation turns",
        "# TYPE hexus_conversation_turns_total counter",
        f"hexus_conversation_turns_total {m_turns}",
    ]

    # 2. Detailed Database Metrics from store.get_metrics_data()
    try:
        db_data = store.get_metrics_data()
    except Exception as exc:
        lines.append(f"# ERROR: Failed to query detailed metrics: {exc}")
        db_data = {}

    def clean_lbl(val: Any) -> str:
        if val is None:
            return "unknown"
        return str(val).replace('\\', '\\\\').replace('"', '\\"')

    if db_data:
        # A. Memory entries count by agent and target
        lines.append("# HELP hexus_memory_entries_count Number of memory entries by agent and target")
        lines.append("# TYPE hexus_memory_entries_count gauge")
        for row in db_data.get("memory_entries", []):
            agent = clean_lbl(row.get("agent_identity"))
            target = clean_lbl(row.get("target"))
            cnt = row.get("count", 0)
            lines.append(f'hexus_memory_entries_count{{agent_identity="{agent}",target="{target}"}} {cnt}')

        # B. Compressed memory entries count
        lines.append("# HELP hexus_memory_entries_compressed_count Number of compressed memory entries by agent")
        lines.append("# TYPE hexus_memory_entries_compressed_count gauge")
        for row in db_data.get("memory_entries_compressed", []):
            agent = clean_lbl(row.get("agent_identity"))
            cnt = row.get("count", 0)
            lines.append(f'hexus_memory_entries_compressed_count{{agent_identity="{agent}"}} {cnt}')

        # C. Conversation turns count by agent and role
        lines.append("# HELP hexus_conversation_turns_count Number of conversation turns by agent and role")
        lines.append("# TYPE hexus_conversation_turns_count gauge")
        for row in db_data.get("conversations", []):
            agent = clean_lbl(row.get("agent_identity"))
            role = clean_lbl(row.get("role"))
            cnt = row.get("count", 0)
            lines.append(f'hexus_conversation_turns_count{{agent_identity="{agent}",role="{role}"}} {cnt}')

        # D. Delegations count
        lines.append("# HELP hexus_delegations_count Number of agent-of-agents delegations by agent")
        lines.append("# TYPE hexus_delegations_count gauge")
        for row in db_data.get("delegations", []):
            agent = clean_lbl(row.get("agent_identity"))
            cnt = row.get("count", 0)
            lines.append(f'hexus_delegations_count{{agent_identity="{agent}"}} {cnt}')

        # E. Feedback metrics for memory entries
        lines.append("# HELP hexus_memory_recalls_total Total recall events on memory entries by agent")
        lines.append("# TYPE hexus_memory_recalls_total counter")
        for row in db_data.get("feedback", []):
            agent = clean_lbl(row.get("agent_identity"))
            recalls = row.get("total_recalls") or 0
            lines.append(f'hexus_memory_recalls_total{{agent_identity="{agent}"}} {recalls}')

        lines.append("# HELP hexus_memory_confirms_total Total memory confirm signals by agent")
        lines.append("# TYPE hexus_memory_confirms_total counter")
        for row in db_data.get("feedback", []):
            agent = clean_lbl(row.get("agent_identity"))
            confirms = row.get("total_confirms") or 0
            lines.append(f'hexus_memory_confirms_total{{agent_identity="{agent}"}} {confirms}')

        lines.append("# HELP hexus_memory_rejects_total Total memory reject signals by agent")
        lines.append("# TYPE hexus_memory_rejects_total counter")
        for row in db_data.get("feedback", []):
            agent = clean_lbl(row.get("agent_identity"))
            rejects = row.get("total_rejects") or 0
            lines.append(f'hexus_memory_rejects_total{{agent_identity="{agent}"}} {rejects}')

        # F. Conversation recalls
        lines.append("# HELP hexus_conversation_recalls_total Total recall events on conversation turns by agent")
        lines.append("# TYPE hexus_conversation_recalls_total counter")
        for row in db_data.get("conversation_recalls", []):
            agent = clean_lbl(row.get("agent_identity"))
            recalls = row.get("total_recalls") or 0
            lines.append(f'hexus_conversation_recalls_total{{agent_identity="{agent}"}} {recalls}')

        # G. Delegation recalls
        lines.append("# HELP hexus_delegation_recalls_total Total recall events on delegations by agent")
        lines.append("# TYPE hexus_delegation_recalls_total counter")
        for row in db_data.get("delegation_recalls", []):
            agent = clean_lbl(row.get("agent_identity"))
            recalls = row.get("total_recalls") or 0
            lines.append(f'hexus_delegation_recalls_total{{agent_identity="{agent}"}} {recalls}')

        # H. Memory entities
        lines.append("# HELP hexus_memory_entities_unique Number of unique entities in memory entries by agent")
        lines.append("# TYPE hexus_memory_entities_unique gauge")
        for row in db_data.get("memory_entities", []):
            agent = clean_lbl(row.get("agent_identity"))
            uniq = row.get("unique_entities") or 0
            lines.append(f'hexus_memory_entities_unique{{agent_identity="{agent}"}} {uniq}')

        lines.append("# HELP hexus_memory_entities_total Total entity occurrences in memory entries by agent")
        lines.append("# TYPE hexus_memory_entities_total counter")
        for row in db_data.get("memory_entities", []):
            agent = clean_lbl(row.get("agent_identity"))
            tot = row.get("total_entity_occurrences") or 0
            lines.append(f'hexus_memory_entities_total{{agent_identity="{agent}"}} {tot}')

        # I. Conversation entities
        lines.append("# HELP hexus_conversation_entities_unique Number of unique entities in conversations by agent")
        lines.append("# TYPE hexus_conversation_entities_unique gauge")
        for row in db_data.get("conversation_entities", []):
            agent = clean_lbl(row.get("agent_identity"))
            uniq = row.get("unique_entities") or 0
            lines.append(f'hexus_conversation_entities_unique{{agent_identity="{agent}"}} {uniq}')

        lines.append("# HELP hexus_conversation_entities_total Total entity occurrences in conversations by agent")
        lines.append("# TYPE hexus_conversation_entities_total counter")
        for row in db_data.get("conversation_entities", []):
            agent = clean_lbl(row.get("agent_identity"))
            tot = row.get("total_entity_occurrences") or 0
            lines.append(f'hexus_conversation_entities_total{{agent_identity="{agent}"}} {tot}')

    # 3. Async Writer Stats
    queue_stats = {}
    try:
        for obj in gc.get_objects():
            if isinstance(obj, AsyncWriter):
                queue_stats = obj.stats()
                break
    except Exception as exc:
        lines.append(f"# ERROR: Failed to extract writer queue stats: {exc}")

    if queue_stats:
        lines.extend([
            "# HELP hexus_writer_queue_size Current size of the background write queue",
            "# TYPE hexus_writer_queue_size gauge",
            f"hexus_writer_queue_size {queue_stats.get('queue_size', 0)}",
            "# HELP hexus_writer_queue_max Maximum capacity of the background write queue",
            "# TYPE hexus_writer_queue_max gauge",
            f"hexus_writer_queue_max {queue_stats.get('queue_max', 256)}",
            "# HELP hexus_writer_dropped_total Total write tasks dropped due to full queue",
            "# TYPE hexus_writer_dropped_total counter",
            f"hexus_writer_dropped_total {queue_stats.get('dropped_total', 0)}",
            "# HELP hexus_writer_thread_alive Whether the background drain thread is running (1=alive, 0=dead)",
            "# TYPE hexus_writer_thread_alive gauge",
            f"hexus_writer_thread_alive {1 if queue_stats.get('thread_alive') else 0}",
        ])
        p50 = queue_stats.get("p50_latency_sec")
        p95 = queue_stats.get("p95_latency_sec")

        lines.append("# HELP hexus_writer_latency_seconds Estimated background write latency quantiles")
        lines.append("# TYPE hexus_writer_latency_seconds gauge")
        if p50 is not None and not math.isnan(p50):
            lines.append(f'hexus_writer_latency_seconds{{quantile="0.5"}} {p50}')
        if p95 is not None and not math.isnan(p95):
            lines.append(f'hexus_writer_latency_seconds{{quantile="0.95"}} {p95}')

    # 4. Background Cleanup Stats
    cleanup_interval = int(os.environ.get("HEXUS_CLEANUP_INTERVAL_HOURS", 24))
    memories_ttl = os.environ.get("HEXUS_CLEANUP_MEMORIES_TTL_DAYS")
    memories_ttl = int(memories_ttl) if memories_ttl else None
    conversations_ttl = os.environ.get("HEXUS_CLEANUP_CONVERSATIONS_TTL_DAYS")
    conversations_ttl = int(conversations_ttl) if conversations_ttl else None
    delegations_ttl = os.environ.get("HEXUS_CLEANUP_DELEGATIONS_TTL_DAYS")
    delegations_ttl = int(delegations_ttl) if delegations_ttl else None

    # Check if thread is alive
    import threading
    thread_alive = any(t.name == "hexus-cleanup-thread" for t in threading.enumerate())

    cleanup_metrics = getattr(store, "_cleanup_metrics", None) or {
        "total_runs": 0,
        "last_run_timestamp": 0.0,
        "deleted_conversations": 0,
        "deleted_memories": 0,
        "deleted_delegations": 0,
    }

    lines.extend([
        "# HELP hexus_cleanup_thread_alive Whether the background cleanup thread is running (1=alive, 0=dead)",
        "# TYPE hexus_cleanup_thread_alive gauge",
        f"hexus_cleanup_thread_alive {1 if thread_alive else 0}",
        "# HELP hexus_cleanup_interval_hours Configured interval for background cleanup in hours",
        "# TYPE hexus_cleanup_interval_hours gauge",
        f"hexus_cleanup_interval_hours {cleanup_interval}",
        "# HELP hexus_cleanup_runs_total Total number of completed background cleanup runs",
        "# TYPE hexus_cleanup_runs_total counter",
        f"hexus_cleanup_runs_total {cleanup_metrics['total_runs']}",
        "# HELP hexus_cleanup_last_run_timestamp_seconds Epoch timestamp of the last cleanup run",
        "# TYPE hexus_cleanup_last_run_timestamp_seconds gauge",
        f"hexus_cleanup_last_run_timestamp_seconds {cleanup_metrics['last_run_timestamp']}",
        "# HELP hexus_cleanup_deleted_records_total Total number of stale records deleted by the cleanup thread",
        "# TYPE hexus_cleanup_deleted_records_total counter",
        f"hexus_cleanup_deleted_records_total{{table=\"conversations\"}} {cleanup_metrics['deleted_conversations']}",
        f"hexus_cleanup_deleted_records_total{{table=\"memory_entries\"}} {cleanup_metrics['deleted_memories']}",
        f"hexus_cleanup_deleted_records_total{{table=\"delegations\"}} {cleanup_metrics['deleted_delegations']}",
    ])

    # 5. Background Consolidation Stats
    consolidation_interval = int(os.environ.get("HEXUS_CONSOLIDATION_INTERVAL_HOURS", 12))
    consolidation_thread_alive = any(t.name == "hexus-consolidation-thread" for t in threading.enumerate())
    
    consolidation_metrics = getattr(store, "_consolidation_metrics", None) or {
        "total_runs": 0,
        "last_run_timestamp": 0.0,
        "low_confidence_processed": 0,
        "low_confidence_deletions": 0,
        "low_confidence_replacements": 0,
        "cooccurring_processed_topics": 0,
        "cooccurring_replacements": 0,
    }

    lines.extend([
        "# HELP hexus_consolidation_thread_alive Whether the background consolidation thread is running (1=alive, 0=dead)",
        "# TYPE hexus_consolidation_thread_alive gauge",
        f"hexus_consolidation_thread_alive {1 if consolidation_thread_alive else 0}",
        "# HELP hexus_consolidation_interval_hours Configured interval for background consolidation in hours",
        "# TYPE hexus_consolidation_interval_hours gauge",
        f"hexus_consolidation_interval_hours {consolidation_interval}",
        "# HELP hexus_consolidation_runs_total Total number of completed background consolidation runs",
        "# TYPE hexus_consolidation_runs_total counter",
        f"hexus_consolidation_runs_total {consolidation_metrics['total_runs']}",
        "# HELP hexus_consolidation_last_run_timestamp_seconds Epoch timestamp of the last consolidation run",
        "# TYPE hexus_consolidation_last_run_timestamp_seconds gauge",
        f"hexus_consolidation_last_run_timestamp_seconds {consolidation_metrics['last_run_timestamp']}",
        "# HELP hexus_consolidation_low_confidence_processed_total Total low-confidence memories processed",
        "# TYPE hexus_consolidation_low_confidence_processed_total counter",
        f"hexus_consolidation_low_confidence_processed_total {consolidation_metrics['low_confidence_processed']}",
        "# HELP hexus_consolidation_low_confidence_deletions_total Total low-confidence memories deleted",
        "# TYPE hexus_consolidation_low_confidence_deletions_total counter",
        f"hexus_consolidation_low_confidence_deletions_total {consolidation_metrics['low_confidence_deletions']}",
        "# HELP hexus_consolidation_low_confidence_replacements_total Total low-confidence memories replaced",
        "# TYPE hexus_consolidation_low_confidence_replacements_total counter",
        f"hexus_consolidation_low_confidence_replacements_total {consolidation_metrics['low_confidence_replacements']}",
        "# HELP hexus_consolidation_cooccurring_processed_topics_total Total cooccurring topics processed",
        "# TYPE hexus_consolidation_cooccurring_processed_topics_total counter",
        f"hexus_consolidation_cooccurring_processed_topics_total {consolidation_metrics['cooccurring_processed_topics']}",
        "# HELP hexus_consolidation_cooccurring_replacements_total Total cooccurring memory entries replaced",
        "# TYPE hexus_consolidation_cooccurring_replacements_total counter",
        f"hexus_consolidation_cooccurring_replacements_total {consolidation_metrics['cooccurring_replacements']}",
    ])

    return "\n".join(lines)


def _build_server(
    store: MemoryStore,
    *,
    name: str = "hexus",
    instructions: Optional[str] = None,
):
    """Build and return a configured `mcp.server.fastmcp.FastMCP` instance.

    The server is wired to the supplied `MemoryStore` (closed-over into
    each tool handler). Reuses the `LocalBertEmbedder` singleton if any
    of the tools trigger an embed — the first embed() call loads the
    model into the process; subsequent calls reuse it.
    """
    # Imported lazily so `pip install hexus` (no [mcp] extra)
    # doesn't pull mcp as a transitive runtime dep.
    import os
    from mcp.server.fastmcp import FastMCP

    if instructions is None:
        instructions = (
            "hexus exposes a Postgres + hexus shared knowledge "
            "base as MCP tools. All tools take an optional `agent_identity` "
            "argument that scopes writes/reads — every connected agent is "
            "isolated by default, and passes can use `agent_identity=None` "
            "(or omit it) on `memory_recall` / `memory_search` to query "
            "across all agents. Embeddings are produced locally by "
            "sentence-transformers MiniLM-L6-v2 (384-dim, no network)."
        )

    mcp = FastMCP(name=name, instructions=instructions, host="0.0.0.0", port=8000)

    # Attach cleanup metrics store to the MemoryStore instance
    store._cleanup_metrics = {
        "total_runs": 0,
        "last_run_timestamp": 0.0,
        "deleted_conversations": 0,
        "deleted_memories": 0,
        "deleted_delegations": 0,
    }

    # Attach consolidation metrics store to the MemoryStore instance
    store._consolidation_metrics = {
        "total_runs": 0,
        "last_run_timestamp": 0.0,
        "low_confidence_processed": 0,
        "low_confidence_deletions": 0,
        "low_confidence_replacements": 0,
        "cooccurring_processed_topics": 0,
        "cooccurring_replacements": 0,
    }

    # -- Scheduled Background Cleanup --------------------------------------
    cleanup_interval = int(os.environ.get("HEXUS_CLEANUP_INTERVAL_HOURS", 24))
    memories_ttl = os.environ.get("HEXUS_CLEANUP_MEMORIES_TTL_DAYS")
    memories_ttl = int(memories_ttl) if memories_ttl else None
    conversations_ttl = os.environ.get("HEXUS_CLEANUP_CONVERSATIONS_TTL_DAYS")
    conversations_ttl = int(conversations_ttl) if conversations_ttl else None
    delegations_ttl = os.environ.get("HEXUS_CLEANUP_DELEGATIONS_TTL_DAYS")
    delegations_ttl = int(delegations_ttl) if delegations_ttl else None

    if cleanup_interval > 0 and any(ttl is not None for ttl in (memories_ttl, conversations_ttl, delegations_ttl)):
        import threading
        import time

        def background_cleanup_loop():
            logger.info(
                "Background cleanup daemon thread started. Interval: %d hours. memories_ttl=%s, conversations_ttl=%s, delegations_ttl=%s",
                cleanup_interval, memories_ttl, conversations_ttl, delegations_ttl
            )
            while True:
                time.sleep(cleanup_interval * 3600)
                try:
                    logger.info("Running scheduled background database cleanup...")
                    deleted = store.cleanup_stale_records(
                        memories_ttl_days=memories_ttl,
                        conversations_ttl_days=conversations_ttl,
                        delegations_ttl_days=delegations_ttl,
                    )
                    store._cleanup_metrics["total_runs"] += 1
                    store._cleanup_metrics["last_run_timestamp"] = time.time()
                    store._cleanup_metrics["deleted_conversations"] += deleted.get("conversations", 0)
                    store._cleanup_metrics["deleted_memories"] += deleted.get("memory_entries", 0)
                    store._cleanup_metrics["deleted_delegations"] += deleted.get("delegations", 0)
                    logger.info("Scheduled background cleanup finished: %s", deleted)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error during scheduled background cleanup: %s", exc)

        cleanup_thread = threading.Thread(
            target=background_cleanup_loop,
            daemon=True,
            name="hexus-cleanup-thread",
        )
        cleanup_thread.start()

    # -- Scheduled Background Consolidation --------------------------------
    consolidation_interval = int(os.environ.get("HEXUS_CONSOLIDATION_INTERVAL_HOURS", 12))
    
    if consolidation_interval > 0:
        import threading
        import time
        import gc
        from hexus.writer import AsyncWriter

        def background_consolidation_loop():
            logger.info(
                "Background consolidation daemon thread started. Interval: %d hours.",
                consolidation_interval
            )
            while True:
                time.sleep(consolidation_interval * 3600)
                
                max_retries = 3
                retry_delay = 60
                
                for attempt in range(1, max_retries + 1):
                    # Check if writer queue is empty
                    writer_empty = True
                    for obj in gc.get_objects():
                        if isinstance(obj, AsyncWriter):
                            if obj._queue.qsize() > 0:
                                writer_empty = False
                                break
                    if not writer_empty:
                        logger.info("Async writer queue is not empty. Deferring consolidation.")
                        time.sleep(retry_delay)
                        continue

                    if not os.environ.get("HEXUS_SUMMARY_MODEL"):
                        logger.warning("HEXUS_SUMMARY_MODEL is not set. Skipping scheduled consolidation.")
                        break

                    try:
                        logger.info("Running scheduled background database consolidation (attempt %d/%d)...", attempt, max_retries)
                        res_low = store.consolidate_low_confidence_memories()
                        res_co = store.consolidate_cooccurring_memories()

                        if res_low.get("status") == "error":
                            raise RuntimeError(f"Low confidence consolidation failed: {res_low.get('reason')}")

                        # Update metrics
                        store._consolidation_metrics["total_runs"] += 1
                        store._consolidation_metrics["last_run_timestamp"] = time.time()
                        
                        store._consolidation_metrics["low_confidence_processed"] += res_low.get("processed", 0)
                        store._consolidation_metrics["low_confidence_deletions"] += res_low.get("deletions", 0)
                        store._consolidation_metrics["low_confidence_replacements"] += res_low.get("replacements", 0)
                        
                        store._consolidation_metrics["cooccurring_processed_topics"] += res_co.get("processed_topics", 0)
                        store._consolidation_metrics["cooccurring_replacements"] += res_co.get("replacements", 0)

                        logger.info("Scheduled background consolidation finished: low_confidence=%s, cooccurring=%s", res_low, res_co)
                        break  # Success
                    except Exception as exc:
                        logger.error("Error during scheduled background consolidation (attempt %d/%d): %s", attempt, max_retries, exc)
                        if attempt < max_retries:
                            logger.info("Retrying consolidation in %d seconds...", retry_delay)
                            time.sleep(retry_delay)

        consolidation_thread = threading.Thread(
            target=background_consolidation_loop,
            daemon=True,
            name="hexus-consolidation-thread",
        )
        consolidation_thread.start()

    # -- tools -------------------------------------------------------------

    @mcp.tool()
    def memory_health() -> Dict[str, Any]:
        """Liveness + capability check. Returns DB status, embedder model/dim, row counts."""
        return tools.memory_health(store, {})

    @mcp.tool()
    def memory_retain(
        contents: list[str],
        target: str = "memory",
        agent_identity: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        doc_type: str = "memory",
        source_url: str = "",
    ) -> Dict[str, Any]:
        """Add one or many memory entries. Each content becomes one row.

        Args:
          contents: list of non-empty strings, one per row.
          target: 'memory' (default — the agent's MEMORY.md mirror) or
                  'user' (the agent's USER.md mirror). Omit to default
                  to 'memory'.
          agent_identity: which agent's scope to write into. Defaults to
                          the env var HEXUS_AGENT_IDENTITY, then
                          'default'. Pick a stable lowercase-dashed name
                          per agent (e.g. 'marketing', 'sales',
                          'intraday-trading').
          metadata: optional dict applied to every row, or a list of
                    dicts (one per content) for per-item metadata. Each
                    dict is stored as JSONB alongside the content.
          doc_type: optional tag stored in metadata (default 'memory').
          source_url: optional URL stored in metadata['source_url'].

        Returns: {"inserted": N, "duplicates": K, "errors": [...]}
        """
        return tools.memory_retain(
            store,
            {
                "contents": contents,
                "target": target,
                "agent_identity": agent_identity,
                "metadata": metadata,
                "doc_type": doc_type,
                "source_url": source_url,
            },
        )

    @mcp.tool()
    def memory_recall(
        query: str,
        top_k: int = 5,
        agent_identity: str = "",
        target: str = "",
        min_similarity: float = 0.0,
        min_confidence: float = 0.0,
        decay_half_life_days: Optional[float] = None,
        recall_boost_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Semantic search over memory entries.

        Args:
          query: the natural-language search query.
          top_k: 1..100, default 5.
          agent_identity: scope to one agent, or empty / None to search
                          across every agent in the store.
          target: 'memory' | 'user' | '' (both).
          min_similarity: 0..1, default 0. Filter out lower-scored hits.
          min_confidence: 0..1, default 0. Filter out entries with lower confidence ratio.
          decay_half_life_days: optional decay half-life in days (0.0 to disable).
          recall_boost_weight: optional recall boost weight parameter.

        Returns: {"query", "count", "results": [{id, agent_identity, target,
                                                  content, score, metadata, ...}]}
        """
        return tools.memory_recall(
            store,
            {
                "query": query,
                "top_k": top_k,
                "agent_identity": agent_identity,
                "target": target,
                "min_similarity": min_similarity,
                "min_confidence": min_confidence,
                "decay_half_life_days": decay_half_life_days,
                "recall_boost_weight": recall_boost_weight,
            },
        )


    @mcp.tool()
    def memory_hybrid_search(
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        agent_identity: str = "",
        target: str = "",
        min_similarity: float = 0.0,
        min_confidence: float = 0.0,
        decay_half_life_days: Optional[float] = None,
        recall_boost_weight: Optional[float] = None,
     ) -> Dict[str, Any]:
         """Hybrid search blending semantic vector search and full-text search over memory entries.

         Args:
           query: the natural-language search query.
           top_k: 1..100, default 5.
           vector_weight: weight for semantic similarity (0..1, default 0.7).
           text_weight: weight for full-text search rank (0..1, default 0.3).
           agent_identity: scope to one agent, or empty / None to search all.
           target: 'memory' | 'user' | '' (both).
           min_similarity: 0..1, default 0. Filter out lower-scored hits.
           min_confidence: 0..1, default 0. Filter out entries with lower confidence ratio.
           decay_half_life_days: optional decay half-life in days (0.0 to disable).
           recall_boost_weight: optional recall boost weight parameter.

         Returns: {"query", "count", "results": [{id, agent_identity, target,
                                                   content, score, vector_score,
                                                   text_score, metadata, ...}]}
         """
         return tools.memory_hybrid_search(
             store,
             {
                 "query": query,
                 "top_k": top_k,
                 "vector_weight": vector_weight,
                 "text_weight": text_weight,
                 "agent_identity": agent_identity,
                 "target": target,
                 "min_similarity": min_similarity,
                 "min_confidence": min_confidence,
                 "decay_half_life_days": decay_half_life_days,
                 "recall_boost_weight": recall_boost_weight,
             },
         )


    @mcp.tool()
    def memory_search(
        agent_identity: str = "",
        target: str = "",
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Browse memory entries without semantic search (list / paginate).

        Returns: {"count", "limit", "offset", "rows": [...]}
        """
        return tools.memory_search(
            store,
            {
                "agent_identity": agent_identity,
                "target": target,
                "limit": limit,
                "offset": offset,
            },
        )

    @mcp.tool()
    def memory_forget(
        id: int,
        confirm: bool = False,
        agent_identity: str = "",
    ) -> Dict[str, Any]:
        """Delete a memory entry by id. Pass confirm=true to actually delete.

        Dry-run by default (returns what would happen). Restricted to the
        caller's agent_identity scope — you can only delete rows you
        own.
        """
        return tools.memory_forget(
            store,
            {
                "id": id,
                "confirm": confirm,
                "agent_identity": agent_identity,
            },
        )

    @mcp.tool()
    def memory_recall_turns(
        query: str,
        top_k: int = 5,
        agent_identity: str = "",
        session_id: str = "",
        min_similarity: float = 0.0,
        decay_half_life_days: Optional[float] = None,
        recall_boost_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Semantic search over past chat turns (every user/assistant exchange).

        Args:
          query: natural-language search.
          top_k: 1..100, default 5.
          agent_identity: scope to one agent, or '' / None to search all.
          session_id: optional — restrict to one session id.
          min_similarity: 0..1, default 0.
          decay_half_life_days: optional decay half-life in days (0.0 to disable).
          recall_boost_weight: optional recall boost weight parameter.

        Returns: {"query", "count", "results": [{id, session_id,
                                                  agent_identity, role,
                                                  content, score, ts, ...}]}
        """
        return tools.memory_recall_turns(
            store,
            {
                "query": query,
                "top_k": top_k,
                "agent_identity": agent_identity,
                "session_id": session_id,
                "min_similarity": min_similarity,
                "decay_half_life_days": decay_half_life_days,
                "recall_boost_weight": recall_boost_weight,
            },
        )

    @mcp.tool()
    def memory_hybrid_recall_turns(
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        agent_identity: str = "",
        session_id: str = "",
        min_similarity: float = 0.0,
        decay_half_life_days: Optional[float] = None,
        recall_boost_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Hybrid search blending semantic vector search and full-text search over conversation turns.

        Args:
          query: natural-language search.
          top_k: 1..100, default 5.
          vector_weight: weight for semantic similarity (0..1, default 0.7).
          text_weight: weight for full-text search rank (0..1, default 0.3).
          agent_identity: scope to one agent, or '' / None to search all.
          session_id: optional — restrict to one session id.
          min_similarity: 0..1, default 0.
          decay_half_life_days: optional decay half-life in days (0.0 to disable).
          recall_boost_weight: optional recall boost weight parameter.

        Returns: {"query", "count", "results": [{id, session_id,
                                                  agent_identity, role,
                                                  content, score, vector_score,
                                                  text_score, ts, ...}]}
        """
        return tools.memory_hybrid_recall_turns(
            store,
            {
                "query": query,
                "top_k": top_k,
                "vector_weight": vector_weight,
                "text_weight": text_weight,
                "agent_identity": agent_identity,
                "session_id": session_id,
                "min_similarity": min_similarity,
                "decay_half_life_days": decay_half_life_days,
                "recall_boost_weight": recall_boost_weight,
            },
        )

    @mcp.tool()
    def memory_append_turn(
        session_id: str,
        role: str,
        content: str,
        agent_identity: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append one chat turn. Use this to capture a (user, assistant)
        exchange into the conversation log for later semantic recall.

        Args:
          session_id: a stable per-conversation id (e.g. UUID).
          role: 'user' | 'assistant' | 'system' | 'tool'.
          content: the turn text.
          agent_identity: which agent's log to append to.
          metadata: optional dict, stored as JSONB.

        Returns: {"id", "session_id", "role"}
        """
        return tools.memory_append_turn(
            store,
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "agent_identity": agent_identity,
                "metadata": metadata,
            },
        )

    @mcp.tool()
    def memory_record_delegation(
        parent_session_id: str,
        child_session_id: str,
        task: str,
        result: str,
        agent_identity: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a subagent delegation.

        Args:
          parent_session_id: parent session ID delegating the task.
          child_session_id: child session ID executing the task.
          task: task description prompt.
          result: task output response.
          agent_identity: agent identity/theme.
          metadata: optional dict metadata.

        Returns: {"id", "parent_session_id", "child_session_id", "agent_identity"}
        """
        return tools.memory_record_delegation(
            store,
            {
                "parent_session_id": parent_session_id,
                "child_session_id": child_session_id,
                "task": task,
                "result": result,
                "agent_identity": agent_identity,
                "metadata": metadata,
            },
        )

    @mcp.tool()
    def memory_recall_delegations(
        query: str,
        top_k: int = 5,
        agent_identity: str = "",
        parent_session_id: str = "",
        min_similarity: float = 0.0,
        decay_half_life_days: float = 0.0,
        recall_boost_weight: float = 0.0,
    ) -> Dict[str, Any]:
        """Recall subagent delegations by semantic similarity query.

        Args:
          query: natural-language search query.
          top_k: max results to return.
          agent_identity: scope search to specific agent theme.
          parent_session_id: scope search to specific parent session.
          min_similarity: similarity score threshold.
          decay_half_life_days: temporal decay half life parameter.
          recall_boost_weight: recall boost parameter.

        Returns: {"query", "count", "results": [{id, parent_session_id, child_session_id, agent_identity, task, result, score, ts, metadata}]}
        """
        return tools.memory_recall_delegations(
            store,
            {
                "query": query,
                "top_k": top_k,
                "agent_identity": agent_identity,
                "parent_session_id": parent_session_id,
                "min_similarity": min_similarity,
                "decay_half_life_days": decay_half_life_days,
                "recall_boost_weight": recall_boost_weight,
            },
        )

    @mcp.tool()
    def memory_count(
        agent_identity: str = "",
        target: str = "",
        session_id: str = "",
    ) -> Dict[str, Any]:
        """Return row counts for memory_entries and conversations, scoped as requested.

        Args:
          agent_identity: default = env / 'default'.
          target: 'memory' | 'user' | '' (both).
          session_id: optional session id, restricts the conversation count.

        Returns: {"memory_entries": N, "conversations": M, ...}
        """
        return tools.memory_count(
            store,
            {
                "agent_identity": agent_identity,
                "target": target,
                "session_id": session_id,
            },
        )

    @mcp.tool()
    def memory_cleanup(
        conversations_ttl_days: Optional[int] = None,
        memories_ttl_days: Optional[int] = None,
        delegations_ttl_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Delete stale records from conversations, memory_entries, and delegations based on TTL.

        Args:
          conversations_ttl_days: delete conversations older than this many days (default None/disabled).
          memories_ttl_days: delete memory entries older than this many days (default None/disabled).
          delegations_ttl_days: delete delegations older than this many days (default None/disabled).

        Returns: {"status": "ok", "deleted": {"conversations": N, "memory_entries": M, "delegations": K}}
        """
        deleted = store.cleanup_stale_records(
            conversations_ttl_days=conversations_ttl_days,
            memories_ttl_days=memories_ttl_days,
            delegations_ttl_days=delegations_ttl_days,
        )
        return {"status": "ok", "deleted": deleted}

    @mcp.tool()
    def memory_consolidate(
        agent_identity: str = "",
    ) -> Dict[str, Any]:
        """Trigger memory consolidation for low-confidence or heavily co-occurring entries.

        Args:
          agent_identity: optional target agent identity (stable lowercase-dashed name) to filter consolidation.
        """
        res_agent = agent_identity.strip() if agent_identity else None
        if not res_agent:
            res_agent = os.environ.get("HEXUS_AGENT_IDENTITY", "default")
            
        res_low = store.consolidate_low_confidence_memories(agent_identity=res_agent)
        res_co = store.consolidate_cooccurring_memories(agent_identity=res_agent)
        
        # Track manual run metrics
        store._consolidation_metrics["total_runs"] += 1
        import time
        store._consolidation_metrics["last_run_timestamp"] = time.time()
        
        if res_low.get("status") == "ok":
            store._consolidation_metrics["low_confidence_processed"] += res_low.get("processed", 0)
            store._consolidation_metrics["low_confidence_deletions"] += res_low.get("deletions", 0)
            store._consolidation_metrics["low_confidence_replacements"] += res_low.get("replacements", 0)
        if res_co.get("status") == "ok":
            store._consolidation_metrics["cooccurring_processed_topics"] += res_co.get("processed_topics", 0)
            store._consolidation_metrics["cooccurring_replacements"] += res_co.get("replacements", 0)
            
        return {
            "status": "ok",
            "low_confidence": res_low,
            "cooccurring": res_co,
        }

    @mcp.tool()
    def memory_metrics() -> str:
        """Return operational metrics in a Prometheus-compatible format."""
        return _generate_metrics(store)

    @mcp.tool()
    def memory_entity_graph(
        entity_type: str,
        entity_value: str,
        agent_identity: str = "",
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Find other entities that co-occur with a target entity.

        Args:
          entity_type: the type of entity (e.g. 'docker_image', 'url').
          entity_value: the specific value (e.g. 'postgres', 'google.com').
          agent_identity: scope search to specific agent theme.
          limit: 1..100, default 5.
        """
        return tools.memory_entity_graph(
            store,
            {
                "entity_type": entity_type,
                "entity_value": entity_value,
                "agent_identity": agent_identity,
                "limit": limit,
            },
        )

    @mcp.tool()
    def memory_graph_walk(
        entity_type: str,
        entity_value: str,
        agent_identity: str = "",
        max_depth: int = 2,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Traverse the co-occurrence graph up to N hops away from a start entity.

        Args:
          entity_type: type of starting entity.
          entity_value: value of starting entity.
          agent_identity: scope search to specific agent theme.
          max_depth: 1..5, default 2. Max depth of graph walk.
          limit: 1..100, default 5.
        """
        return tools.memory_graph_walk(
            store,
            {
                "entity_type": entity_type,
                "entity_value": entity_value,
                "agent_identity": agent_identity,
                "max_depth": max_depth,
                "limit": limit,
            },
        )

    @mcp.tool()
    def memory_common_topics(
        agent_identity: str = "",
        min_strength: int = 2,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve clusters/cliques of heavily co-occurring entities.

        Args:
          agent_identity: scope search to specific agent theme.
          min_strength: minimum count of co-occurrences.
          limit: 1..100, default 10.
        """
        return tools.memory_common_topics(
            store,
            {
                "agent_identity": agent_identity,
                "min_strength": min_strength,
                "limit": limit,
            },
        )

    @mcp.tool()
    def memory_confirm(id: int) -> Dict[str, Any]:
        """Increment confirm_count in metadata JSONB for the given entry ID.

        Args:
          id: the integer row ID of the memory entry to confirm.
        """
        return tools.memory_confirm(
            store,
            {
                "id": id,
            },
        )

    @mcp.tool()
    def memory_reject(id: int) -> Dict[str, Any]:
        """Increment reject_count in metadata JSONB for the given entry ID.

        Args:
          id: the integer row ID of the memory entry to reject.
        """
        return tools.memory_reject(
            store,
            {
                "id": id,
            },
        )

    @mcp.tool()
    def memory_summarize_session(
        session_id: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Compute the vector centroid of a session's turns and find the K closest turns.

        Args:
          session_id: the session identifier to summarize.
          limit: 1..100, default 5.
        """
        return tools.memory_summarize_session(
            store,
            {
                "session_id": session_id,
                "limit": limit,
            },
        )

    @mcp.tool()
    def memory_retrieve(id: int) -> Dict[str, Any]:
        """Retrieve the original full content of a memory entry by its integer ID.

        Args:
          id: the integer row ID of the memory entry to retrieve.
        """
        return tools.memory_retrieve(
            store,
            {
                "id": id,
            },
        )

    @mcp.tool()
    def headroom_retrieve(id: int) -> Dict[str, Any]:
        """Retrieve the original full content of a memory entry by its integer ID.

        Args:
          id: the integer row ID of the memory entry to retrieve.
        """
        return tools.memory_retrieve(
            store,
            {
                "id": id,
            },
        )

    @mcp.tool()
    def memory_stats() -> Dict[str, Any]:
        """Return metrics from Hexus database and background async queue stats."""
        return tools.memory_stats(store, {})

    # -- patch ASGI app for Prometheus /metrics endpoint -------------------
    _orig_get_asgi_app = mcp.streamable_http_app
    def get_asgi_app_with_metrics(*args, **kwargs):
        app = _orig_get_asgi_app(*args, **kwargs)
        from starlette.responses import Response
        async def metrics(request):
            return Response(_generate_metrics(store), media_type="text/plain")
        app.add_route("/metrics", metrics)
        return app
    mcp.streamable_http_app = get_asgi_app_with_metrics

    return mcp





def build_server(
    dsn: str,
    *,
    name: str = "hexus",
    instructions: Optional[str] = None,
) -> Any:
    """Build (but don't run) an MCP server for the given DSN.

    The MemoryStore is constructed lazily on first use; closing the
    server is the caller's job (or the process exit, which is fine for
    the typical stdio / one-shot http deployments).
    """
    store = MemoryStore(dsn)
    return _build_server(store, name=name, instructions=instructions)
