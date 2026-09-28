# pgvector — Roadmap

A versioned plan for evolving the plugin from "Postgres storage for a single agent" to "shared memory substrate for a fleet of cooperating hermes-agent minions." The driving constraint throughout is **deliver multi-agent memory on the resources you already have** — your existing Postgres, a single embedding endpoint, no LLM costs in the memory hot path, no third-party services.

The plugin is built around a clear separation of concerns:

- The **agent's mental model** stays unchanged — keep using the built-in `memory(action='add', target='memory'|'user', …)` tool.
- The **storage backbone** moves from per-host markdown files to a centralized Postgres table that every minion can read from and write to.
- The **scoping mechanism** (`agent_identity`) keeps each minion's working memory clean while still allowing explicit cross-theme recall when an agent needs the bigger picture.

See [CHANGELOG.md](CHANGELOG.md) for the version-by-version detail behind every milestone below.

---

## Milestones

### M1 — Shared storage with per-agent themes (v0.1 → v0.1.1) — DONE

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

### M2 — Conversation-history substrate (v0.2) — DONE

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

### M3 — Identity propagation for stateless API minions (v0.3) — DONE

**Problem solved:** before v0.3, every systemd-run minion hit the gateway via `POST /v1/chat/completions` and the gateway forwarded a `gateway_session_key` kwarg (from the built-in `X-Hermes-Session-Key` header), but the pgvector plugin didn't consume it — every API-routed write collapsed to `agent_identity='default'`. The per-theme isolation promised in M1 was theoretical for fleet-style use.

| Capability | Version |
|---|---|
| Plugin reads `kwargs.get('gateway_session_key')` in the `agent_identity` fallback chain | v0.3.0 |
| Each minion sets `default_headers={'X-Hermes-Session-Key': '<theme>'}` on its OpenAI client | v0.3.0 |
| Per-theme allow-list in plugin config (`allowed_themes`) so a typo'd header can't silently create a new theme | v0.4.0 |

**Theme naming convention:** lowercase, dash-separated, stable (case-folded automatically since v0.6.0 — see M7 below). Established themes:

- product/report themes: `marketing`, `sales`, `morning-report`
- per-worker minions: `agent-trading`, `agent-sre`, `agent-marketing`, `agent-gitlab`, `agent-cloud`, `agent-hermes`
- `incident` — reserved for incident-responder use
- governed sinks (v0.4): `whatsapp-dm` (collapsed DM/session keys), `external-group` (collapsed group/channel/thread keys, v0.5.0), `_bench` (benchmark traffic), `default` (last resort)

**Why this matters for multi-agent deployments:** v0.3 is the smallest change that makes the M1 design true in practice. Without it, a marketing-daily run could surface a trading-agent's notes in its recall, defeating the isolation. With it, each minion has its own memory pool by default while still being able to ask cross-theme questions through `recall_memory(scope='all')`.

---

### M4 — Identity governance + agent-of-agents observability (v0.4.0) — DONE

Shipped in v0.4.0 alongside identity governance (DM/PII bucketing, bench isolation, allow-list — the M3 allow-list row above shipped here, not in a separate v0.3.1), an embedding backfill sweep + writer retry, conversation TTL/embed-policy, and the `hermes-pgvector` maintenance CLI (`python -m pgvector` at the time; the CLI/import were renamed in v0.5.0 — see M4.3). `on_delegation` + `on_session_end` capture parent→child delegations into `memory_agents` / `memory_agent_edges` (provenance only).

**Goal:** when one minion delegates to another (subagent pattern), capture the task/result pair so the parent can recall "what did I ask my research subagent last week and what did it find."

| Capability | Version |
|---|---|
| `on_delegation(task, result, child_session_id)` hook → row in `conversations` | v0.4.0 |
| `on_session_end(messages)` hook → turn-capture backstop | v0.4.0 |
| Parent ↔ child session linkage (`memory_agent_edges`, `conversations.parent_session_id`) for traceback | v0.4.0 |
| `recall_delegation` tool — **deferred, not shipped.** Delegations are stored as ordinary conversation turns (`[delegation] task: … result: …`) and surface through the existing `recall_conversation` tool, so a dedicated tool was not needed to ship the capability. Not on the roadmap going forward. | — |

**Why this matters for multi-agent deployments:** orchestrator patterns (one supervisor minion fanning out to N specialists) become much more reviewable when delegations are first-class durable records. Without this, the only place the supervisor remembers a delegation is the immediate conversation context, which compresses away.

---

### M4.1 — Hybrid recall: vector + full-text (v0.4.1) — DONE

`recall_memory` / `recall_conversation` fuse the HNSW cosine ranking with a Postgres full-text ranking via **Reciprocal Rank Fusion** (`k=60`). Recovers exact-lexical hits that pure cosine smooths away (error codes, hostnames, flags, rare identifiers) and text-only rows with a `NULL` embedding that the vector index can't see. Migration `003` adds a GIN index over the existing `content` column — **no new tables, columns, or LLM** (stays inside invariant #1: a second index over the same text, not a parallel ontology). Fail-soft: degrades to pure vector on a hybrid error, and to full-text-only when the query itself fails to embed. Toggle via `plugins.pgvector.hybrid_search` (default on); ambient `prefetch()` stays pure-vector.

| Capability | Version |
|---|---|
| GIN `to_tsvector('english', content)` indexes on `memory_entries` + `conversations` (migration `003`) | v0.4.1 |
| `MemoryStore.hybrid_search()` / `hybrid_search_turns()` — RRF fusion of vector + full-text rankers | v0.4.1 |
| Full-text-only degradation when the query fails to embed (recall instead of error) | v0.4.1 |
| `hybrid_search` config toggle | v0.4.1 |

---

### M4.2 — Pip-native install + review hardening (v0.4.2) — DONE

`pip install hermes-memory-pgvector` + `hermes-pgvector install` is now a complete deployment on any hermes-agent host: the `install` subcommand generates a `$HERMES_HOME/plugins/pgvector/` discovery shim (hermes-agent scans plugin directories only — never site-packages — so pip alone was invisible to it), and migration `004` grants the runtime role DML on the core tables so `migrate` alone yields a working install. Plus a full-codebase review's fixes: LIKE-literal `replace`/`remove` matching, drain-on-shutdown writer, unmasked dimension-mismatch errors, tightened DM-key regex, bulk-import circuit breaker, remap guard under lock, credential-redacted tool errors, preserved `score: null` + `rrf_score` for hybrid hits.

| Capability | Version |
|---|---|
| `hermes-pgvector install [--remove] [--force]` discovery-shim generator | v0.4.2 |
| Migration `004` — runtime-role grants (no manual OWNER step) | v0.4.2 |
| Review hardening (writer drain, LIKE escaping, dim-error surfacing, DM regex, circuit breaker) | v0.4.2 |

---

### M4.3 — Import rename + read-side identity gate (v0.5.0) — DONE

Two findings from an earlier full-codebase review that could not be fixed without a breaking change or a public-behaviour change, so they were split out of the v0.4.x line.

The import package moved `pgvector` → `hermes_pgvector` (the CLI is unaffected: it was already `hermes-pgvector`, replacing the earlier `python -m pgvector` form). The old top-level import name is owned by [pgvector-python](https://pypi.org/project/pgvector/); sharing a venv meant whichever installed last won, and the discovery shim's import could resolve to the wrong module — the loader then falls back to built-in memory on a single log line, taking the fleet's shared memory offline silently. The distribution name and the `pgvector` provider name are unchanged; only the import moved.

The PII/bench buckets also gained the read-side half they never had. `whatsapp-dm` and `_bench` isolation was write-side only, so `scope='all'` (or naming a bucket directly) surfaced DM bodies in any theme — and turn capture then rewrote them under the reading theme. Cross-theme recall stays opt-in and broad; it just no longer reaches the sinks.

| Capability | Version |
|---|---|
| Import package `pgvector` → `hermes_pgvector`; shim import verified in a clean subprocess at install time | v0.5.0 |
| `hermes_agent.memory_providers` entry point declared — `pip install` alone is sufficient on a host that reads it, no shim to go stale | v0.5.0 |
| Read-side exclusion of `whatsapp-dm` / `_bench` from `scope='all'` + explicit-scope rejection | v0.5.0 |
| Group/channel/thread keys bucketed into `external-group` (same PII/cardinality fix as the DM bucket, different `chat_type`) | v0.5.0 |
| Config type-contract fixes (`allowed_themes` string/list, boolean toggles declared as strings) | v0.5.0 |
| Turn double-write guard; `replace()` single-row UPDATE; fail-soft hardening | v0.5.0 |

---

### M4.4 — Remove-path data-loss fix + backfill signal repair (v0.5.1) — DONE

Two defects found by reviewing the v0.5.1 candidate, one of them data-loss. `_worker` passed `old_text=item.content` for a remove, but the built-in tool's remove op carries its target in `old_text` and leaves `content` empty — so the pattern was always `LIKE '%%'`, which matches every row. A single `memory remove` deleted the entire mirror for that `(agent_identity, target)`. Fixed at two layers: the worker reads `extra["old_text"]`, and `store.remove()` refuses an empty pattern so the destructive delete is unreachable by omission. No production loss occurred — every theme's history is continuous.

Separately, an empty-content row (created because nothing rejected empty `add`/`replace`) was retried by every nightly backfill forever, pinning `failed` above zero and making `remaining == 0` unreachable — destroying the signal operators watch. Un-embeddable rows are now skipped *and* reported as `unembeddable`.

| Capability | Version |
|---|---|
| `remove` targets `extra["old_text"]`; `store.remove()` rejects an empty pattern | v0.5.1 |
| Empty `add`/`replace` no longer mirrored; `remove` exempt (target is in metadata) | v0.5.1 |
| Backfill skips + reports un-embeddable rows so `remaining` can reach zero | v0.5.1 |

---

### M4.5 — Configurable embedding model + plugin-loader fix (v0.5.3) — DONE

The reference deployment moved its vector columns to `vector(1536)` and re-embedded with an OpenRouter-hosted OpenAI model, and the plugin could not follow through configuration: the 768-dim check was a literal in the response parser and in the backfill guard, and no `Authorization` header was ever sent. The dimension is now configuration. A mismatch still fails fast, because the check moved to config rather than being relaxed.

The same release fixes embeds under hermes-agent's directory loader, which binds each sibling module back onto the package after running it (that replaced the `embed` function with the `embed` submodule, so every embed raised `TypeError`), and fixes a read timeout escaping as a bare `TimeoutError` instead of an `EmbeddingError`.

| Capability | Version |
|---|---|
| `embed_dim` (default 768) drives the response check, backfill `expected_dim` guard and `stats` dry-run | v0.5.3 |
| `embed_api_key_env`: bearer token read from a named env var at call time, never logged | v0.5.3 |
| `embed_protocol`: `auto` / `openai` (no fallback, errors surface) / `ollama` | v0.5.3 |
| One config-driven embed helper for prefetch, recall tools, bulk import, writer drain and CLI | v0.5.3 |
| Loader clobber of `embed` fixed natively (private alias); read timeouts become `EmbeddingError` | v0.5.3 |

---

### M4.6 — psycopg dependency floors (v0.5.4, v0.5.5) — DONE

Two dependency-only releases, no code changes in either. This package opens a single long-lived `ConnectionPool` shared by the agent thread and the async-writer drain thread, so upstream pool fixes land directly on its hot path.

| Capability | Version |
|---|---|
| Floors raised to `psycopg[binary]>=3.3.6`, `psycopg-pool>=3.3.2` | v0.5.4 |
| Floor raised to `psycopg-pool>=3.3.3` (24h idle sync-worker fix, upstream #1419) | v0.5.5 |

---

### M6 — Public release (v0.6.0 → v1.0) — v0.6.0 DONE, "released" pending

**Goal:** documentation, contract guarantees, and a CHANGELOG good enough that someone landing on this plugin from the hermes-agent docs can deploy it cleanly without reading the source.

| Capability | Status |
|---|---|
| Stable, typed config schema (`get_config_schema()`: real `type`/`default`/`minimum`/`maximum` per key) | DONE (v0.6.0) |
| CI: unit + live-DB matrix (Python 3.11–3.13 × Postgres 16/17/18), build/wheel checks, version consistency | DONE (v0.6.0) |
| Conformance test: validate `MemoryProvider` ABC + `MemoryManager` call-shape contract against upstream (pinned ref in `conformance/HERMES_AGENT_REF`) | DONE (v0.6.0) |
| CHANGELOG.md (Keep a Changelog format) | DONE (v0.6.0) |
| Full docs set: config reference, operations, scaling, troubleshooting, upgrading, release checklist, security policy | DONE (v0.6.0) |
| **Released as 1.0.0** | NOT YET — 1.0.0rc1 (version/metadata only, no behaviour change) is merged; after the soak period, promoting it to 1.0.0 is the next step (see [docs/release/RELEASING.md](docs/release/RELEASING.md)) |

A prior 1.0-readiness review found four High-severity defects and thirteen Medium/Low ones; every one of them (H1–H4, M1–M7 except roadmap-M5, L1–L6, L8–L10) is fixed in v0.6.0 — see [CHANGELOG.md](CHANGELOG.md#060---2026-09-26). L7 (theme-filtered ANN recall at very large table sizes) did not reproduce at the tested scale and is documented as a scaling consideration instead of a fix — see [docs/scaling.md](docs/scaling.md).

---

### M7 — Identity case-folding (v0.6.0) — DONE

Identities are now case-folded: `Marketing` and `marketing` resolve to the same theme, closing a gap where the allow-list check and the raw priority chain were both case-sensitive while the README's naming convention was already "lowercase, dash-separated." Existing mixed-case themes need one `hermes-pgvector remap --execute` per theme when upgrading — see [docs/upgrading.md](docs/upgrading.md).

| Capability | Version |
|---|---|
| `normalize_identity()` lowercases after strip; alias keys and allow-list entries compared lowercased | v0.6.0 |
| Raw (pre-normalization) value still recorded as `raw_identity` metadata when it differs | v0.6.0 |

---

### M5 — Production hardening at scale (v1.1 → v1.2) — PROPOSED

> Retargeted from v0.6–v0.7 (the earlier plan for this milestone) to after 1.0: a 1.0-readiness review concluded a contract release should not carry new features, and none of these items were in progress. TTL pruning (the one M5 item that had actually shipped) is called out separately below rather than re-listed here.

**Goal:** survive a fleet of dozens of minions, hundreds of writes per minute, multi-million-row tables.

| Capability | Version |
|---|---|
| Recency decay in ranking — bias `recall_memory` / `recall_conversation` toward more recently updated rows (TTL pruning on age already shipped in v0.4.0 — see below; this is decay *within* the ranking function, not deletion) | v1.1 (proposed) |
| Optional partial HNSW indexes per high-volume `agent_identity` when cross-theme search becomes the slow query | v1.1 (proposed) |
| Periodic re-sync of `MEMORY.md` / `USER.md` (not just on init) for callers that edit the markdown directly | v1.1 (proposed) |
| Bulk-import CLI for migrating from Holographic / Honcho / Mem0 / Hindsight installations | v1.2 (proposed) |
| Metrics: queue depth, drop count, embed latency p50/p95, recall hit rate (Prometheus-friendly) | v1.2 (proposed) |
| Per-platform metadata facets (CLI vs cron vs telegram vs API) for richer recall filtering | v1.2 (proposed) |

**TTL prune (v0.4.0, done):** `hermes-pgvector prune --days N` deletes conversation turns older than N days, operator-triggered only (never automatic); `memory_entries` are never pruned by any code path. This is age-based deletion, already shipped — it is a different mechanism from the recency-decay-in-ranking row above, which keeps every row but ranks stale ones lower.

**Why this matters for multi-agent deployments:** a memory store that's fast for one user often falls over under fleet load. M5 is the slow + boring work that turns "works on my hermes" into "works for ten agents writing concurrently."

---

## What's *not* on the roadmap (and why)

These were considered and rejected. Keeping the list visible so it's clear the omissions are deliberate, not gaps.

- **A `fact_store` ontology with trust scoring, entity resolution, and HRR algebra.** Holographic does that well. Layering it in pgvector would duplicate Holographic's surface and force agents to learn a second memory model when the built-in `memory` tool already serves the same need.
- **LLM-mediated dialectic recall** (à la Honcho). The synchronous LLM call in the memory hot path is exactly the failure mode that motivated this plugin. We embed text; we don't reason about it. The agent reasons.
- **Background deriver / fact-extraction pipelines.** Same reason: any background LLM loop becomes a retry-storm liability. Fact extraction stays explicit (the agent decides to call `memory.add`) instead of implicit (a daemon thread scrapes turns).
- **Multi-tenant authentication / RBAC at the plugin layer.** Postgres roles + `agent_identity` scoping are sufficient. Anything fancier belongs in a separate access layer, not in a memory provider.
- **A dedicated `recall_delegation` tool** (considered for M4, dropped — see M4 above). Delegations already surface through `recall_conversation`; a second search tool over the same rows would be a distinction without a difference.

---

## Operating principle

Every milestone has to answer: **does this make N cooperating agents more capable, or does it just add features?** If it doesn't pass that test, it goes in the "Not on the roadmap" pile.
