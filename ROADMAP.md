# pgvector — Roadmap

A versioned plan for evolving the plugin from "Postgres storage for a single agent" to "shared memory substrate for a fleet of cooperating hermes-agent minions." The driving constraint throughout is **deliver multi-agent memory on the resources you already have** — your existing Postgres, a single embedding endpoint, no LLM costs in the memory hot path, no third-party services.

The plugin is built around a clear separation of concerns:

- The **agent's mental model** stays unchanged — keep using the built-in `memory(action='add', target='memory'|'user', …)` tool.
- The **storage backbone** moves from per-host markdown files to a centralized Postgres table that every minion can read from and write to.
- The **scoping mechanism** (`agent_identity`) keeps each minion's working memory clean while still allowing explicit cross-theme recall when an agent needs the bigger picture.

---

## Milestones

### M1 — Shared storage with per-agent themes (v0.1 → v0.1.1) ✅ DONE

**Goal:** every minion's `memory(action='add', …)` write lands in a single Postgres table with semantic search on top, scoped so marketing's notes don't pollute trading's recall.

| Capability | Version |
|---|---|
| Mirror built-in `memory.add` / `replace` / `remove` via `on_memory_write` hook | v0.1.0 |
| Per-`agent_identity` row scoping (marketing / sales / trading / incident / `default`) | v0.1.0 |
| 768-dim embeddings via OpenAI-compatible or Ollama-native endpoint | v0.1.0 |
| HNSW index for sub-millisecond recall up to ~1M rows | v0.1.0 |
| Async writer (bounded queue + daemon drain) — agent loop never blocks on slow embeds | v0.1.0 |
| `psycopg_pool.ConnectionPool` — small pool shared across agent + writer threads | v0.1.0 |
| Admin/runtime DDL split (`apply_migration_as_admin()` + verify-only `ensure_schema()`) | v0.1.0 |
| `recall_memory(query, scope, target, limit)` cross-theme search tool | v0.1.0 |
| Bulk-import existing `MEMORY.md` + `USER.md` from disk on init (idempotent + cheap on re-init) | v0.1.1 |

**Why this matters for multi-agent deployments:** before this milestone, every hermes-agent process kept its own markdown files. Two agents on the same host stomped on each other; agents on different hosts had no shared substrate at all. M1 gives every minion a single source of truth, scoped per-theme so isolation is the default and cross-theme recall is an explicit opt-in.

---

### M2 — Conversation-history substrate (v0.2) ✅ DONE

**Goal:** every substantive chat turn across every minion becomes semantically searchable. Filter the boilerplate (`"ok"`, `"thanks"`, sub-40-char turns) so recall stays high-signal.

| Capability | Version |
|---|---|
| `conversations` table: id, session_id, agent_identity, role, content, ts, embedding, metadata | v0.2.0 |
| `sync_turn(user, assistant, session_id)` hook captures every turn pair | v0.2.0 |
| Boilerplate filter (length floor + acknowledgement regex) | v0.2.0 |
| Per-session + per-agent timeline indexes | v0.2.0 |
| HNSW index on conversation embeddings — same tuning as `memory_entries` | v0.2.0 |
| `recall_conversation(query, scope, limit)` tool with `scope ∈ {current, session, all, <theme>}` | v0.2.0 |

**Why this matters for multi-agent deployments:** in a fleet, *the chat is the memory*. A finance agent can pull "what did marketing say about last quarter's CAC" without anybody having to copy-paste between systems. The agent fetches it as a tool call, exactly when it needs it.

---

### M3 — Identity propagation for stateless API minions (v0.3) ✅ DONE

**Problem solved:** before v0.3, every systemd-run minion (`marketing-daily.py`, `sales-daily.py`, intraday workers, morning-report enrich) hit the gateway via `POST /v1/chat/completions` and the gateway forwarded a `gateway_session_key` kwarg (from the built-in `X-Hermes-Session-Key` header), but the pgvector plugin didn't consume it — every API-routed write collapsed to `agent_identity='default'`. The per-theme isolation promised in M1 was theoretical for fleet-style use.

| Capability | Version |
|---|---|
| Plugin reads `kwargs.get('gateway_session_key')` in the `agent_identity` fallback chain | v0.3.0 ✅ |
| Each minion sets `default_headers={'X-Hermes-Session-Key': '<theme>'}` on its OpenAI client | v0.3.0 ✅ |
| `intraday_loop._get_client_for(agent_name)` per-agent client cache so `intraday-trading`, `intraday-sre`, `intraday-marketing`, etc. are each scoped | v0.3.0 ✅ |
| Optional: per-theme allow-list in plugin config so a typo'd header can't silently create a new theme | v0.3.1 ⏳ |

**Theme naming convention:** lowercase, dash-separated, stable. Established themes:

- `marketing` — `scripts/marketing-daily.py`, `scripts/marketing/content-factory.py`
- `sales` — `scripts/sales-daily.py`
- `morning-report` — `scripts/morning-report/enrich.py`
- `intraday-{agent_name}` — every `scripts/agents/agent-*-worker.py` via `intraday_loop.run(agent_name=…)`. Concretely: `intraday-trading`, `intraday-sre`, `intraday-marketing`, `intraday-gitlab`, `intraday-cloud`, `intraday-hermes`.
- `incident` — reserved for incident-responder (Phase 1 observe-only today, will activate when its Phase 2 ships LLM hypothesis-forming)
- `default` — last-resort bucket for interactive `hermes chat` without `--profile` or any caller that doesn't set the header

**Why this matters for multi-agent deployments:** v0.3 is the smallest change that makes the M1 design true in practice. Without it, a marketing-daily run could surface a trading-agent's notes in its recall, defeating the isolation. With it, each minion has its own memory pool by default while still being able to ask cross-theme questions through `recall_memory(scope='all')`.

---

### M4 — Identity governance + agent-of-agents observability (v0.4.0) ✅ DONE

Shipped in v0.4.0 alongside identity governance (DM/PII bucketing, bench isolation, allow-list), an embedding backfill sweep + writer retry, conversation TTL/embed-policy, and the `python -m pgvector` maintenance CLI. `on_delegation` + `on_session_end` capture parent→child delegations into `memory_agents` / `memory_agent_edges` (provenance only). A dedicated `recall_delegation` tool was **deferred** — delegations are stored as conversation turns and surface through the existing `recall_conversation` tool, so a separate tool wasn't needed to ship the capability.

**Goal:** when one minion delegates to another (subagent pattern), capture the task/result pair so the parent can recall "what did I ask my research subagent last week and what did it find."

| Capability | Version |
|---|---|
| `on_delegation(task, result, child_session_id)` hook → row in `conversations` (or a separate `delegations` table) | v0.4.0 |
| `on_session_end(messages)` hook → optional session summary row, agent-decided | v0.4.0 |
| `recall_delegation(query, scope, limit)` tool — surfaces past delegation transcripts | v0.4.0 |
| Parent ↔ child session linkage in metadata for traceback | v0.4.0 |

**Why this matters for multi-agent deployments:** orchestrator patterns (one supervisor minion fanning out to N specialists) become much more reviewable when delegations are first-class durable records. Without this, the only place the supervisor remembers a delegation is the immediate conversation context, which compresses away.

---

### M5 — Production hardening at scale (v0.5 → v0.6) ⏳ PROPOSED

**Goal:** survive a fleet of dozens of minions, hundreds of writes per minute, multi-million-row tables.

| Capability | Version |
|---|---|
| TTL / decay policy on `memory_entries.updated_at` and `conversations.ts` so stale entries surface less | v0.5.0 |
| Optional partial HNSW indexes per high-volume `agent_identity` when cross-theme search becomes the slow query | v0.5.0 |
| Periodic re-sync of `MEMORY.md` / `USER.md` (not just on init) for callers that edit the markdown directly | v0.5.0 |
| Bulk-import CLI for migrating from Holographic / Honcho / Mem0 / Hindsight installations | v0.5.0 |
| Metrics: queue depth, drop count, embed latency p50/p95, recall hit rate (Prometheus-friendly) | v0.6.0 |
| Per-platform metadata facets (CLI vs cron vs telegram vs API) for richer recall filtering | v0.6.0 |

**Why this matters for multi-agent deployments:** a memory store that's fast for one user often falls over under fleet load. M5 is the slow + boring work that turns "works on my hermes" into "works for ten agents writing concurrently."

---

### M6 — Public release (v1.0) ⏳ PROPOSED

**Goal:** documentation, contract guarantees, and a CHANGELOG good enough that someone landing on this plugin from the hermes-agent docs can deploy it cleanly without reading the source.

| Capability | Version |
|---|---|
| Stable config schema (semver guarantees on `plugins.pgvector.*` keys) | v1.0.0 |
| Full hermes-agent docs page with config reference, scaling guide, troubleshooting | v1.0.0 |
| Coverage: store tests in upstream CI against a Postgres service container | v1.0.0 |
| Conformance test: validate `MemoryProvider` ABC contract against current upstream | v1.0.0 |

---

## What's *not* on the roadmap (and why)

These were considered and rejected. Keeping the list visible so it's clear the omissions are deliberate, not gaps.

- **A `fact_store` ontology with trust scoring, entity resolution, and HRR algebra.** Holographic does that well. Layering it in pgvector would duplicate Holographic's surface and force agents to learn a second memory model when the built-in `memory` tool already serves the same need.
- **LLM-mediated dialectic recall** (à la Honcho). The synchronous LLM call in the memory hot path is exactly the failure mode that motivated this plugin. We embed text; we don't reason about it. The agent reasons.
- **Background deriver / fact-extraction pipelines.** Same reason: any background LLM loop becomes a retry-storm liability. Fact extraction stays explicit (the agent decides to call `memory.add`) instead of implicit (a daemon thread scrapes turns).
- **Multi-tenant authentication / RBAC at the plugin layer.** Postgres roles + `agent_identity` scoping are sufficient. Anything fancier belongs in a separate access layer, not in a memory provider.

---

## Operating principle

Every milestone has to answer: **does this make N cooperating agents more capable, or does it just add features?** If it doesn't pass that test, it goes in the "Not on the roadmap" pile.
