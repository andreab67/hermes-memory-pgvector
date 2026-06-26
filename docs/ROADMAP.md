# Hexus Roadmap & Upcoming Features

This document outlines the past milestones that brought Hexus to where it is today, as well as the exciting features planned for the future. 

Our driving constraint throughout is **deliver multi-agent memory on the resources you already have** — your existing Postgres, a single in-process embedder, no LLM costs in the memory hot path, and no third-party services.

## Completed Milestones ✅

- **M1 (v0.1, v0.1.1):** Shared storage with per-agent themes, async writer, connection pool, and bulk import from `MEMORY.md`/`USER.md`.
- **M2 (v0.2):** Conversation transcript table with `sync_turn` capture and the `recall_conversation` tool.
- **M3 (v0.3):** Identity propagation for stateless API minions via `X-Hermes-Session-Key`.
- **M3.5 (v0.4 fork):** Swapped to a **Local BERT** embedder (MiniLM-L6-v2, 384-dim) and introduced the **MCP server** (`hexus-mcp serve`) for non-hermes clients.
- **Phase 5 Features:** 
  - **Hybrid Search:** Combined BM25 + vector search for RetainDB-level precision.
  - **Temporal Decay Scoring & TTL:** Memory forgetfulness, making older memories naturally decay unless reinforced.
- **Phase 6 Features:**
  - **Entity Tagging & Co-occurrence Graph:** Regex-based entity extraction forming a lightweight knowledge graph without an LLM.
  - **Confidence/Recall Counters:** Trust scoring to surface highly relevant, often-recalled entries.
  - **Conversation Summaries:** Extractive summarization mathematically capturing the centroid of a conversation.
- **Phase 7 Features:**
  - **Cross-Encoder Reranker:** Optional local reranking for precision.
  - **Event Webhooks:** Custom webhook triggers for memory events.

## What's Next? 🚀

### M5 — Production hardening at scale
**Goal:** Survive a fleet of dozens of minions, hundreds of writes per minute, multi-million-row tables.
- Refined TTL / decay policies.
- Optional partial HNSW indexes per high-volume `agent_identity`.
- Bulk-import CLI for migrating from Holographic, Honcho, Mem0, or Hindsight installations.
- Per-platform metadata facets (CLI vs cron vs telegram vs API) for richer recall filtering.

### M6 — Public Release (v1.0)
**Goal:** Documentation, contract guarantees, and rock-solid stable releases.
- Deep Prometheus-friendly metrics instrumentation.
- Official PyPI publishing (`pip install hexus`).
- GitHub Actions CI for GHCR image pushes and conformance testing.
- Stable configuration schema with semver guarantees.
- Comprehensive `hermes-agent` docs integration.

## What's *Not* on the Roadmap
- **LLM-mediated dialectic recall:** Synchronous LLM calls in the memory hot path kill performance. We embed text; we don't reason about it. The agent reasons.
- **Multi-tenant authentication at the plugin layer:** Postgres roles and `agent_identity` scoping are sufficient.
- **A massive `fact_store` ontology:** We don't duplicate existing agent memory models, we provide the best possible storage backbone for them.
