# memory-pgvector

**Full Project Plan: Fork + Local BERT + MCP Shared Knowledge Base Adapter**
**For Hermes Agent + Any MCP Client (Claude, Cursor, etc.)**

---

> **FORK NOTICE**
> This is **Toby's fork** of `andreab67/hermes-memory-pgvector` v0.3.1, kept under the same repo name (`memory-pgvector`) by design — the upstream's package name, plugin path, schema, and Hermes integration points are deliberately preserved for drop-in compatibility. The work described in this plan adds local BERT embeddings and an MCP server without breaking the upstream contract.
>
| Upstream: | `https://github.com/andreab67/hermes-memory-pgvector` (BSD-3-Clause © 2026 Andrea Borghi) |
| This fork: | `git@github.com:codenamekt/memory-pgvector.git` |
| Working copy: | `/opt/data/workspace/memory-pgvector/` |
| Date: | June 2026 |
| Target Hardware: | Intel NUC6i7KYK (i7, 16 GB RAM, CPU-only) |
| Latest version: | **v0.4.0** (Phases 0–3 done) |

All Python files in this fork carry a `# Forked from andreab67/hermes-memory-pgvector (BSD-3-Clause)` header per upstream license requirements.

---

## Goal

A drop-in Hermes memory plugin AND a reusable MCP server that turn Postgres + pgvector into a fully local, shared knowledge base for documents + session data across agents — both hermes-agent minions and any MCP client (Claude Desktop, Cursor, custom agents). The whole thing runs offline on a NUC, no LLM in the hot path, no third-party services.

---

## 1. Why This Project Makes Sense (Recap)

- You want a Hermes-native memory provider that is fully offline and lightweight.
- You want the same vector store exposed as a standard MCP server so other agents (Claude, Cursor, etc.) can read/write the shared KB.
- The existing repo already solves 90% of the hard parts (async writer, connection pooling, multi-tenant scoping, Hermes hooks, schema, tests, migrations).
- **Decision:** Fork `https://github.com/andreab67/hermes-memory-pgvector` (not just inspiration). It is the exact building block we need.

## 2. License Confirmation

- **License:** BSD 3-Clause ("New BSD")
- **Copyright:** © 2026 Andrea Borghi
- **Commercial use:** Fully allowed (including closed-source derivatives, SaaS, internal tools, selling products).
- **Obligations:** Keep the original copyright notice + full BSD license text in any distributed copies. Add a clear "Forked from andreab67/hermes-memory-pgvector" note in README.
- **Per-file attribution:** Add a one-line header to the docstring of every Python file we touch or copy: `# Forked from andreab67/hermes-memory-pgvector (BSD-3-Clause)`.

## 3. Hardware & Embedding Model Choice

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Why perfect for the NUC:**
  - ~23M parameters, ~90MB on disk, <500MB RAM resident.
  - Excellent CPU performance (ONNX runtime optional for +20-30% speed, but skip initially).
  - Batch embedding ~10-20 sentences/sec on the i7.
  - Proven quality for semantic search; no GPU required.
- **Pre-warm strategy:** the multi-stage Dockerfile downloads the model into `$HF_HOME` during the deps build stage and copies the cache into the runtime stage at `/home/hermes/.cache/huggingface`. Runtime runs with `HF_HUB_OFFLINE=1` so the embedder never touches the network.
- **Dependencies:**
  - `psycopg[binary]>=3.3.4,<4` (Postgres driver)
  - `psycopg-pool>=3.3.1,<4` (connection pool)
  - `PyYAML>=6.0,<7` (plugin config)
  - `sentence-transformers>=2.7.0,<4` (local BERT)
  - `numpy>=1.24,<3` (vector math)
  - `mcp[cli]>=1.0,<2` (MCP server SDK — `[mcp]` optional extra)

## 4. High-Level Architecture (Shared Core)

```
memory-pgvector/                  (the fork, repo root)
├── pgvector/                       (Hermes plugin package, upstream compat)
│   ├── __init__.py                 ← PgvectorMemoryProvider (Hermes hooks)
│   ├── store.py                    ← MemoryStore (Postgres ops, SHARED with MCP)
│   ├── writer.py                   ← AsyncWriter (daemon drain thread, SHARED)
│   ├── embed.py                    ← module-level embed() dispatch (local OR HTTP)
│   ├── embedder.py                 ← LocalBertEmbedder (MiniLM-L6-v2)
│   ├── plugin.yaml
│   └── migrations/001_schema.sql   ← vector(384)
├── mcp_server/                     ← NEW: modelcontextprotocol server
│   ├── __init__.py
│   ├── tools.py                    ← pure functions: memory_retain/recall/search/...
│   ├── server.py                   ← FastMCP wiring (8 tools)
│   └── cli.py                      ← `memory-pgvector-mcp serve|doctor` console script
├── tests/                          ← 50+ tests: smoke + embedder + migration + mcp_server
├── docker/                         ← multi-stage, CPU-only torch, HF cache pre-populated
│   ├── Dockerfile
│   ├── compose.yml                 ← profiles: test, dev, mcp
│   └── entrypoint.sh               ← dispatches on $CMD_PROFILE: test | mcp | shell
├── pyproject.toml                  ← name, deps, [mcp] extra, console_script
├── PLAN.md                         ← this file
├── ROADMAP.md                      ← M1..M6 milestone table
└── README.md                       ← quickstart + MCP examples
```

One `MemoryStore` + one `LocalBertEmbedder` instance powers both the Hermes plugin AND the MCP server. Cold start hits exactly once per process. The two surfaces share the same `agent_identity` scoping model — a hermes minion and a Claude Desktop session can co-exist on the same DB without colliding.

---

## 5. Docker-First Development Environment (NEW)

**Principle:** every test, every smoke check, every MCP server invocation runs through docker. The host never installs Python deps directly. Sibling containers are the dev/test loop. This is the same pattern as the `docker-development` skill.

### 5.1 Image design (multi-stage Dockerfile)

```
Stage 1 — `deps` (builder)
  Base: python:3.11-slim
  • pip install build tooling
  • pip install all runtime + dev deps into a venv
  • pre-download the MiniLM-L6-v2 model into /opt/hf-cache

Stage 2 — `runtime` (production)
  Base: python:3.11-slim
  • Copy the venv from `deps`
  • Copy the preloaded HF cache from `deps`
  • Copy the installed plugin package
  • Non-root user, HF_HUB_OFFLINE=1, TINI as PID 1
  • Entrypoint dispatches on $CMD_PROFILE: test | mcp | shell

Stage 3 — `dev` (local iteration, optional)
  Base: runtime
  • Mount-bind the working tree at /app for live reload
  • pip install -e /app on container start
```

Build targets:
- `docker build --target runtime -t memory-pgvector:latest .`
- `docker build --target dev     -t memory-pgvector:dev .`

### 5.2 docker compose profiles

A single `docker/compose.yml` with three profiles. Operators pick one per command:

| Profile | Services                  | Use case                                   |
|---------|---------------------------|--------------------------------------------|
| `test`  | `pg` (postgres+pgvector), `test` (one-shot) | CI + local test runs                  |
| `dev`   | `pg`                      | Local dev: host runs pytest, container runs DB |
| `mcp`   | `pg`, `mcp`               | Run the MCP server for Claude/Cursor to connect to |

### 5.3 Canonical docker commands

```bash
# Run the full test suite (CI mode, one-shot)
docker compose -f docker/compose.yml --profile test up --abort-on-container-exit --exit-code-from test

# Local dev: bring up Postgres+pgvector, run pytest from host
# `--service-ports` is required to publish 5432 to the host (the compose uses
# `expose:`, not `ports:`, to avoid a host-port conflict with homelab-db).
docker compose -f docker/compose.yml --profile dev up --service-ports -d pg
PG_TEST_DSN="dbname=hermes_test user=postgres host=localhost" \
  pytest tests/

# Start the MCP server
docker compose -f docker/compose.yml --profile mcp up -d
docker compose -f docker/compose.yml logs -f mcp    # tail MCP server logs

# Build the image (cold)
docker build -f docker/Dockerfile -t memory-pgvector:latest .

# Build with the model pre-downloaded (slower build, faster runtime)
docker build -f docker/Dockerfile --target runtime --build-arg PRELOAD_MODEL=1 -t memory-pgvector:latest .
```

### 5.4 Image hygiene

- Non-root user (`hermes`, uid 1000).
- `HF_HUB_OFFLINE=1` in the runtime image to prevent accidental downloads.
- `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`.
- Healthcheck on the `mcp` service.
- Named volume for `pg` data so test runs are isolated but persistent across runs.
- `.dockerignore` excludes `.git`, `.venv`, `__pycache__`, `.pytest_cache`, `*.egg-info`.

### 5.5 CI integration

GitHub Actions workflow (`.github/workflows/test.yml`):
- Service: `pgvector/pgvector:pg16`
- Build: `docker build --target runtime -t memory-pgvector:test .`
- Run: `docker run --rm --network container:<pg> -e PG_TEST_DSN=... memory-pgvector:test pytest`
- No host Python needed in CI; the image is the test environment.

### 5.6 Implementation notes (from Phase 0 build-out)

Five adjustments made during the initial implementation, kept here so future readers know why the plan and the code diverge on these points:

1. **`expose:` not `ports:` for the `pg` service.** The host's port 5432 is already used by `homelab-db` in the homelab compose. Using `expose: ["5432"]` makes the port reachable to sibling containers via the internal docker network (as `pg:5432`) without claiming a host port. Dev profile users opt into publishing with `--service-ports`.
2. **Migration applied by the test entrypoint, not via `/docker-entrypoint-initdb.d/`.** The pgvector base image ships with a pre-existing `/docker-entrypoint-initdb.d/001_schema.sql/` *directory* (not a file) that conflicts with our file bind-mount — `psql` inside the container reports "Is a directory" when the entrypoint sources the file. Mounting the whole `migrations/` directory has the same symptom. Fix: drop the volume mount, rely on the Dockerfile `COPY` of the package (which includes the migrations dir) into the test image, and have the entrypoint apply the SQL via `psql -f /app/pgvector/migrations/001_schema.sql` with a `to_regclass('memory_entries')` guard so re-runs are no-ops.
3. **Entrypoint shebang is `#!/bin/bash`, not `#!/bin/sh`.** The TCP connectivity check uses `/dev/tcp/host/port` which is a bash built-in; `sh`/`dash` silently treat it as a regular file path and the check never actually opens a connection. `python:3.11-slim-bookworm` ships with bash at `/bin/bash`, so no extra package is needed.
4. **HF cache path is `/root/.cache/huggingface/hub/`, not `/root/.cache/huggingface/`.** When `HF_HOME=/root/.cache/huggingface` is set, `huggingface_hub` writes model files under `$HF_HOME/hub/`. The build-time `ls -la $HF_HOME` only shows the top-level dir, which appears empty (4K) even after a successful download. We added an explicit `ls -la $HF_HOME/hub` to the build's verification step so a silent download failure is impossible to miss.
5. **`ARG`s declared at the file root are NOT visible in `RUN` instructions inside a stage.** The deps stage's `python -c "SentenceTransformer('${EMBED_MODEL}')"` call was getting an empty model name (silent success, no download) when only a top-level `ARG EMBED_MODEL` was declared. Fix: re-declare `ARG EMBED_MODEL` inside the deps stage. Hardcoded `HF_HOME` to the same absolute path in both stages to remove the second moving part.

---

## 6. Detailed Phased Implementation Plan

### Phase 0 — Setup & Fork ✅ DONE

- [x] Fork the repo on GitHub → `codenamekt/memory-pgvector`
- [x] Update `pyproject.toml` (name=`memory-pgvector`, version=0.4.0, description, dependencies, fork URLs)
- [x] Update `README.md` (BERT default, MCP instructions, NUC notes, fork notice, 384-dim schema)
- [x] `LICENSE` stays unchanged (it's the original BSD-3-Clause, not our copyright)
- [x] Add fork attribution header to every Python file's docstring (`pgvector/__init__.py`, `embed.py`, `store.py`, `writer.py`, `embedder.py`, plus the new `mcp_server/*.py`)
- [x] Clone, install in editable mode on NUC, run existing tests
- [x] Docker scaffolding (`docker/Dockerfile`, `docker/compose.yml`, `docker/entrypoint.sh`, `.dockerignore`)
- [x] Verify: `docker compose --profile test up` passes (15/16 baseline + 19 embedder + 10 migration + 1 real-BERT + 8 mcp_server = 50+ tests, 1 skipped by design)

### Phase 1 — Local BERT Embedder Swap ✅ DONE

- [x] Add `sentence-transformers` to `pyproject.toml` dependencies
- [x] Create `pgvector/embedder.py` with `LocalBertEmbedder` class (lazy model load, thread-safe singleton, batch embed)
- [x] `pgvector/embed.py:embed()` dispatches: `base_url=None` → `LocalBertEmbedder`, else HTTP path (OpenAI-compat → Ollama-native fallback). The HTTP path is preserved as the opt-in fallback.
- [x] All 8 fake-vector test sites: 768 → 384
- [x] `pgvector/migrations/001_schema.sql`: `vector(768)` → `vector(384)`, idempotent (`CREATE TABLE IF NOT EXISTS` + HNSW index guarded by `to_regclass` check)
- [x] `pgvector/__init__.py`: `DEFAULTS['embed_url']=None`, `DEFAULTS['embed_model']='sentence-transformers/all-MiniLM-L6-v2'`, added `expected_dim=384` and `embed_eager_load` knobs
- [x] `tests/test_embedder.py` (NEW, 19 tests): constants, lazy import, dim properties, empty/whitespace handling, singleton (now keyed on full args + thread-safe), real-model load + embed + batch + semantic similarity, dispatch logic, HTTP 404 → EmbeddingError
- [x] `tests/test_migration.py` (NEW, 10 tests): file exists, idempotency, dim=384 on both tables, HNSW indexes, unique constraint, CHECK constraint, 384-dim insert works, 768-dim insert rejected
- [x] `tests/test_smoke.py`: renamed HTTP-live test to `test_embed_live_http_returns_384_dims`, added `test_real_bert_end_to_end_round_trip` (real BERT embed → store → HNSW query)

### Phase 2 — Hermes Plugin Polish ✅ DONE

- [x] All existing hooks preserved: `initialize`, `on_memory_write`, `sync_turn`, `prefetch`, `recall_memory`, `recall_conversation`, `shutdown`, `system_prompt_block`, `on_session_switch`.
- [x] `plugin.yaml` v0.4.0 (was 0.3.1); hooks list kept as a discovery hint (only `on_session_end` declared, the rest are class methods auto-discovered by the hermes-agent registry)
- [x] `DEFAULTS` updated for the local-BERT-first world; HTTP path remains opt-in via `embed_url`
- [x] `hermes memory setup` works unchanged (same tool surfaces, new default model, no migration needed by hermes itself)
- [x] Lazy model load: `embed_eager_load=False` default; `True` makes plugin init pre-load with a log line

### Phase 3 — MCP Adapter / Shared KB Server ✅ DONE

- [x] Add dependency: `mcp[cli]>=1.0,<2` (the official Anthropic-maintained Python SDK, https://github.com/modelcontextprotocol/python-sdk — note: the package name is `mcp` on PyPI, NOT `modelcontextprotocol` as the original plan said; the docs are clear but easy to get wrong).
- [x] `mcp_server/` package:
  - `tools.py` — pure functions: `memory_health`, `memory_retain`, `memory_recall`, `memory_search`, `memory_forget`, `memory_recall_turns`, `memory_append_turn`, `memory_count`. Each takes a `MemoryStore` + dict and returns a JSON-serializable dict.
  - `server.py` — FastMCP wiring: each pure function wrapped as a `@mcp.tool()` with explicit input types and docstring.
  - `cli.py` — `memory-pgvector-mcp serve|doctor` console script. `serve` blocks; `doctor` is one-shot health for ops/CI.
- [x] 8 MCP tools: `memory_health`, `memory_retain`, `memory_recall`, `memory_search`, `memory_forget`, `memory_recall_turns`, `memory_append_turn`, `memory_count`
- [x] Transports: stdio (Claude Desktop, Cursor) and streamable-http (fleet use). Selected by `--transport stdio|http` or `MEMORY_PGVECTOR_TRANSPORT` env var.
- [x] Multi-agent: `agent_identity` parameter on every tool call; `MEMORY_PGVECTOR_AGENT_IDENTITY` env var as the per-process default. Two agents pointing at the same MCP server see isolated views by default; `agent_identity=""` on read tools queries across all agents.
- [x] `tests/test_mcp_server.py` (NEW, ~25 tests): every pure function (validation, dedupe, multi-agent isolation, cross-agent recall, dry-run on forget, etc.) + FastMCP wiring checks (every tool registered with the right name + non-empty inputSchema).
- [x] `docker/compose.yml`: `mcp` profile (pg + long-lived server, `doctor` as the healthcheck)
- [x] `docker/entrypoint.sh`: `mcp` profile now actually starts the server (was previously a bash drop-in)

### Phase 4 — Production Hardening & Benchmarks

- [ ] PyPI publish: `pip install memory-pgvector` and `pip install memory-pgvector[mcp]`. (Blocked on: no PyPI account yet for `codenamekt`.)
- [x] GitHub Actions CI: workflow in place at `.github/workflows/ci.yml` (build image, run tests, push to GHCR on tag). Will run on first push.
- [x] Benchmark on the NUC (real MiniLM-L6-v2, 100 docs):
  - Cold start (first embed): 4.6s
  - Throughput: batch 1: 153 ops/sec, batch 8: 669 ops/sec, batch 32: 971 ops/sec, batch 128: 1463 ops/sec
  - Recall latency (top-10 HNSW): 7.9ms
  - Store/search round-trip: PASSED
- [x] Test the full streamable-http MCP transport: confirmed working in hermes-agent; crossed off per user.

**Total realistic effort:** Phases 0–3 done in ~1 week. Phase 4 is a half-week of follow-up work, most of which is publish + benchmark, not code.

---

## 7. Key Technical Notes & Gotchas

- **Async non-blocking writes** → existing `AsyncWriter` (`pgvector/writer.py:51`).
- **Connection pooling & leak fixes** → v0.3.1, `psycopg_pool.ConnectionPool` with `min=0`, `max=4`, `max_idle=30s`, `max_lifetime=300s` (`pgvector/store.py:37`).
- **Multi-tenant / per-minion scoping** → `agent_identity` priority chain: header > profile > workspace > 'default' (`pgvector/__init__.py:228`).
- **No LLM in hot path** → the BERT swap preserves this. Local CPU embeddings are ~10-20 sentences/sec on the i7.
- **HNSW index + hybrid search** → already in schema (m=16, ef_construction=64).
- **768→384 dim swap is invasive** — see Phase 1c, 5+ files.
- **`plugin.yaml` hooks list is a hint** — only `on_session_end` is declared, but the `PgvectorMemoryProvider` class methods are auto-discovered by hermes-agent's provider registry. (Verified against `hermes-agent/plugins/memory/__init__.py` — discovery is via `dir(provider)`.)
- **HF cache layout** — `HF_HOME=/root/.cache/huggingface` puts files at `/root/.cache/huggingface/hub/`, not at the parent. The build's `ls` step now also lists the `hub/` subdir to surface silent download failures.
- **Dockerfile ARG scope** — `ARG`s declared before the first `FROM` are NOT visible in `RUN` instructions inside a stage; you must re-declare. The plan originally assumed a global ARG was enough — it isn't.
- **MCP package name is `mcp`, not `modelcontextprotocol`** — the original plan said the latter; the PyPI package is `mcp[cli]` (which pulls FastMCP + uvicorn + starlette for the streamable-http transport).

## 8. Optional Future Extensions (Nice-to-Haves)

- Lightweight reflect step (tiny local LLM reranker, optional).
- Graph-like metadata relations (via JSONB queries).
- Bulk import from files/folders (CLI tool).
- Veracity-style proof counts (simple metadata counter).
- Vector quantization (int8 binary) for >10M row scale.
- LSP-style "memory as a tool surface" for editor integration (VS Code / Neovim extensions).
- Web UI for browsing the shared knowledge base (a thin FastAPI + htmx app on top of the MCP tools).

## 9. Open Questions to Resolve Before Phase 5

- [x] Should the existing 768-dim HTTP path remain as a fallback, or do we hard-cut to local BERT? — **Done**: dual-path. `base_url=None` → local; else HTTP. Operators migrate by flipping a key.
- [x] Renaming the package: keep `pgvector/` directory or rename to `pgbert/`? — **Done**: kept `pgvector/` for backward compat. The wheel name is `memory-pgvector`; the import name is `pgvector`.
- [ ] PyPI publish: yes/no? If yes, `memory-pgvector` as the package name (different from import name `pgvector` for clarity). — **Pending**: phase 4.
- [ ] Should the docker image be published to Docker Hub / GHCR? — **Pending**: phase 4, in the same publish pass as PyPI.
- [ ] Should we keep `model_count` or rename it to `memory_stats` for clarity? — **Pending**: probably rename in v0.5 for consistency with the new M4 `recall_delegation` tool.

## 10. Decision Log

- **2026-06-10:** Plan drafted; review against current fork confirmed 90% accuracy. Identified three under-specified risks: 768→384 invasiveness, plugin.yaml hooks list, model pre-warm strategy.
- **2026-06-10:** Project name normalized to `memory-pgvector` everywhere (title, docker image, CLI binary, PyPI package, repo tree label) — no codename, no qualifier.
- **2026-06-10:** Phase 0 docker scaffolding implemented and verified (15/16 tests pass in sibling container, 1 skipped by design). Three adjustments vs the plan-as-written, all documented in §5.6: `expose:` not `ports:`, entrypoint-applied migration, bash shebang.
- **2026-06-11:** Phase 1 + 2 + 3 landed in commit `1951d48` (v0.4.0): local BERT swap (`LocalBertEmbedder`, `embedder.py`, `EXPECTED_DIM=384`, schema 768→384, `embed_url=None` default), 19 embedder tests + 10 migration tests + 1 real-BERT e2e test added. Two new bugs caught mid-build and fixed: HF cache path is `$HF_HOME/hub/` not `$HF_HOME/` (silent 4K-dir failure); top-level `ARG` doesn't expand in stage `RUN` (silent empty-model-name failure). 45 tests green in 6.51s.
- **2026-06-11:** Phase 3 + docs landed in commit `feature/mcp-server`: `mcp_server/` package (`tools.py`, `server.py`, `cli.py`), 8 MCP tools exposed via FastMCP, `memory-pgvector-mcp serve|doctor` console script, `mcp` docker compose profile with `doctor` as the healthcheck. 50+ tests total, multi-agent isolation tested. README rewritten with MCP section, ROADMAP.md updated with M3.5 (BERT + MCP) milestone, PLAN.md updated to mark Phases 0-3 done.
- **2026-06-13:** Phase 4 progress (commit `f51feb4`): benchmark complete on real MiniLM-L6-v2 (cold start 4.6s, throughput 153-1463 ops/sec, recall latency 7.9ms over 100 docs); CI workflow created at `.github/workflows/ci.yml` (build, test, GHCR push on tag); MCP transport test confirmed working in hermes-agent (crossed off); PyPI publish still pending (no PyPI account yet for `codenamekt`).
