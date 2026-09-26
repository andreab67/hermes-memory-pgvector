# PLAN-1.0 — autonomous execution plan for hermes-memory-pgvector 1.0

This file is an executable brief for a Claude Code session running **Opus 5.5 as the orchestrator**,
delegating implementation to **Sonnet subagents** wherever possible. It runs end to end without a
human. It prepares two pull requests and stops. **It never publishes anything**: the maintainer
releases to PyPI by hand (see `docs/release/RELEASING.md`, created by WP-E).

Source of the work: [`docs/code-review/1.0-readiness-review-2026-09-26.md`](../code-review/1.0-readiness-review-2026-09-26.md)
(findings H1–H4, M1–M7, L1–L10, roadmap status, adopted decisions). Read it before anything else.

---

## 0. How to start it (for the human)

Open Claude Code at the repo root with the Opus model and a permission mode that lets it run
unattended (it needs `docker`, `git`, `python`, `pip`, and optionally `gh`). Then send:

> Read `docs/release/PLAN-1.0.md` and execute it end to end, autonomously. Do not ask me questions.

Prerequisites on the machine: Docker daemon running; push access to `origin`; network access to
PyPI, GitHub and Docker Hub. `gh` authenticated is optional (without it the run pushes branches and
writes the PR bodies to files instead of opening PRs).

What you get back:

- Branch `release/0.6.0` → PR **A** into `main`: every fix, the test harness, CI, conformance tests,
  docs, CHANGELOG, version `0.6.0`.
- Branch `release/1.0.0rc1` (stacked on A) → PR **B** into `release/0.6.0`: version `1.0.0rc1`,
  classifier and docs only — **no behaviour change** relative to 0.6.0.
- A handoff report in PR A's body (and in the final chat message): what changed, gate results,
  anything blocked, and the manual steps left for you.

---

## 1. Rules for the orchestrator

### 1.1 Autonomy

- Never ask the human anything. Every open decision is pre-decided in §3. If something new comes
  up, choose the option that is (a) reversible, (b) smallest, (c) consistent with the invariants in
  §2.2 — then record it in the run log under `DECISIONS`.
- Keep a run log at `.planning/1.0-run-log.md` (gitignored). Append: wave start/end, each agent's
  report summary, gate results, decisions, blocked items. The handoff (§8) is built from it.

### 1.2 Hard limits — never do these

- Push to `main`, merge any PR, create or push tags, upload to PyPI or TestPyPI, force-push,
  rewrite history, delete branches this run did not create.
- Touch any production system or any database other than the local Docker containers of §4.
- Edit `001_schema.sql` or `003_hybrid_search_fts.sql`. `002`/`004` may change **only** their GRANT
  blocks and non-ASCII comment characters (H1, L10). Schema changes go in new migrations (`005_…`).
- Add runtime dependencies. Dev/test-only dependencies go under `[project.optional-dependencies]`.
- Make a failing test pass by deleting it, skipping it, `xfail`-ing it or weakening its assertion,
  unless the test pins behaviour this plan deliberately changes — then update it and record the
  reason in the run log and the commit message.

### 1.3 Delegation model

| Role | Model | How | Does |
| --- | --- | --- | --- |
| Orchestrator | Opus (this session) | — | Loads context, runs baselines and gates, dispatches agents, reviews every diff, merges worktree branches, resolves conflicts, writes the handoff. Writes code only to resolve merge conflicts or a fix of ≤ 20 lines after a failed verification. |
| Implementer | Sonnet | `Agent` tool, `model: "sonnet"`, `isolation: "worktree"` for parallel work | One work package (WP) each, strictly inside its owned files. |
| Verifier | Sonnet, fresh context | `Agent` tool, `model: "sonnet"`, never the implementer's session | Reads the merged diff and the findings, runs the tests, tries to break each fix, reports PASS/FAIL per finding with evidence. |

Launch independent agents of the same wave **in one message** so they run concurrently.

**Implementer brief template** (fill every field; agents start with no context):

```
You are implementing work package <WP-ID> for hermes-memory-pgvector (Python 3.11+, psycopg 3,
pgvector). Repo root: <worktree path>. Work on the branch you are on; commit when done.

Read first: docs/code-review/1.0-readiness-review-2026-09-26.md (finding <IDs>), docs/release/PLAN-1.0.md
§2.2 (invariants) and §5 <WP-ID>, then the files you own.

Files you own (edit ONLY these; create new test files named tests/test_v060_<topic>.py):
<list>

Task and acceptance criteria: <copy the WP section verbatim>

Testing: the Docker Postgres from scripts/test-env.sh is running. Create your own database with
`scripts/test-env.sh db <name>` and export the DSNs it prints (never share a database with another
agent). Run the full suite with PG_TEST_DSN and PG_TEST_EMBED_URL set: `python -m pytest -q`.
All tests must pass. Run the repro scripts named in your WP; they must exit 0.

Constraints: no new runtime dependencies; fail-soft on every agent-thread path (no exception may
escape a provider hook); keep existing public names working unless the WP says otherwise.

Report back (≤ 40 lines): files changed; tests added; the last 5 lines of pytest output; repro
exit codes; any deviation from the WP and why; anything you could not do.
```

### 1.4 Retry and blocked policy

- A WP fails verification → send the verifier's report to a **new** Sonnet implementer with the
  same brief plus "Fix these verifier findings: …". At most 3 attempts per WP.
- After 3 failed attempts: revert that WP's commits on the release branch, mark it `BLOCKED` in the
  run log with the last error, and continue with every WP that does not depend on it. A blocked
  H-severity WP makes PR A a draft (`gh pr create --draft`).
- Environment failures (Docker pull, network): retry twice with 30 s backoff, then record and
  continue with what can run.

### 1.5 Git conventions

- Branch `release/0.6.0` from `origin/main`. Worktree branches `wp/<id>`; merge them into the
  release branch with `git merge --no-ff` after the orchestrator's review.
- Commit messages follow the repo's history: `fix:`, `feat:`, `test:`, `ci:`, `docs:`, `chore:`,
  subject ≤ 72 chars, body says which finding IDs it closes. Use the repo's configured git identity.

---

## 2. Context to load before Wave 0

### 2.1 Read

1. `docs/code-review/1.0-readiness-review-2026-09-26.md` (the findings; this plan's IDs refer to it).
2. `docs/code-review/full-codebase-review-2026-09-07.md` §5, §6 and §11 (earlier fixes and regressions — do not reintroduce them).
3. `ROADMAP.md`, `README.md`, `pyproject.toml`, `hermes_pgvector/plugin.yaml`.
4. All of `hermes_pgvector/` and `tests/`.
5. If a local `CLAUDE.md` exists (it is gitignored), read it; its invariants override §2.2.
6. Upstream contract: clone `https://github.com/NousResearch/hermes-agent` at ref
   `d0288be5` (record the resolved SHA) into `.cache/hermes-agent` (WP-0 adds `.cache/` and
   `.venv-conformance/` to `.gitignore`). Read `agent/memory_provider.py`,
   `agent/memory_manager.py` (`notify_memory_tool_write`, `_prefetch_provider`, `prefetch_all`,
   `queue_prefetch_all`), `agent/agent_init.py` (`agent_context`), `plugins/memory/__init__.py`.

### 2.2 Invariants (inferred from the code; a local CLAUDE.md wins)

1. Storage layer, not a memory model: no fact ontology, entity graph or LLM calls anywhere.
2. No retry storms on the agent thread: hot-path embeds are single-attempt and time-bounded; retries only on the background writer.
3. Per-theme scoping is the default; cross-theme recall is explicit; the PII/bench sinks are never readable from other themes.
4. Fail-soft everywhere: nothing raises into the agent loop.
5. Admin/runtime split: DDL only in migrations run as admin; the runtime role gets DML only; migrations are additive and idempotent.
6. Rows keep the identity they were written with; no retroactive rewrites except explicit operator `remap`/`cleanup`.
7. Misconfiguration surfaces as-is (e.g. dimension mismatch is never masked by a fallback error).
8. Never touch the postgres-owned `events`/`entities`/`relations` tables (maintenance whitelist).

### 2.3 Baseline (orchestrator, before Wave 0)

Record in the run log: `git rev-parse origin/main`, Python version, `docker version`, and the
current suite result without a database (`python -m pytest -q`; expect 170 passed, 48 skipped).

---

## 3. Decisions already made (do not re-open)

| Topic | Decision |
| --- | --- |
| Release shape | 0.6.0 carries every behaviour change; 1.0.0rc1 = 0.6.0 + version/metadata/docs; 1.0.0 = soaked rc re-tagged (human). CI and the conformance tests land in 0.6.0 (earlier than the review suggested): they are non-behavioural and protect every later wave. |
| M5 (roadmap) | Off the 1.0 path. Not implemented in this run. |
| `agent_context` (M4) | New key `write_contexts`, default `"primary,cron"`. Contexts not listed skip all writes (recall still works). Missing `agent_context` kwarg = `primary`. |
| Identity case (M7) | Lowercase after strip in `normalize_identity`; alias keys and allow-list entries compared lowercased. Raw value still recorded as `raw_identity` metadata. |
| Uniqueness (M1) | Migration `005`: unique index on `(agent_identity, target, md5(content))`, then drop `memory_entries_unique`. Code uses `ON CONFLICT DO NOTHING` with **no conflict target**, so it works before and after 005. |
| Delegation rows (M5) | `conversations.parent_session_id` = the delegating session of *this* row (`self._parent_session_id` or NULL); the child session id stays in `metadata.child_session_id` and in `memory_agent_edges`. |
| Runtime role (H1) | Read from `current_setting('hermes_pgvector.runtime_role', true)`, default `hermes`; missing role → `RAISE NOTICE`, never an error. CLI: `migrate --runtime-role NAME`. psql users: `PGOPTIONS='-c hermes_pgvector.runtime_role=NAME'`. |
| Default `embed_url` (L2) | `http://localhost:11434`. Called out as an upgrade note (set it explicitly before upgrading). |
| `embed_timeout` default (H4) | `5.0`; plus new `prefetch_budget` default `5.0` s (must stay below the host's 8 s). |
| Backfill exit codes (L4) | `0` done (`remaining == 0` for all tables), `2` endpoint unavailable / aborted, `3` rows still remaining or failed. Other commands unchanged. |
| Public API (semver) | `plugins.pgvector.*` keys, tool names/params, CLI commands/flags/exit codes, table/column names, provider name `pgvector`, entry point. `MemoryStore` and all modules are internal. |
| Support matrix | Python 3.11, 3.12, 3.13; PostgreSQL 16 and 17 (plus 18 if `pgvector/pgvector:pg18` pulls); pgvector ≥ 0.5.0 required, CI on the image's bundled version; hermes-agent: the pinned conformance ref. |
| Test-number scrub (L1) | Replace every real-looking number with the fictional 555-01xx range (e.g. `15550100123`). History rewrite is **not** done here — listed as a manual step. |

---

## 4. Test harness contract (built in WP-0, used by everything after)

- `scripts/test-env.sh up [pg16|pg17|pg18]` — starts `pgvector/pgvector:<tag>` as container
  `hpg-test-<tag>` on a free port (default 55432 for pg17, 55416 for pg16, 55418 for pg18), with
  `POSTGRES_PASSWORD=postgres`, `tmpfs` data dir; waits for readiness; creates role `hermes`
  (`LOGIN PASSWORD 'hermes'`); starts the fake embed server (below) on 127.0.0.1:11999 if not
  running.
- `scripts/test-env.sh db <name> [--runtime-role ROLE]` — (re)creates database `<name>`, runs
  `python -m hermes_pgvector migrate --admin-dsn …` (with `--runtime-role` once WP-B lands), prints
  `export PG_TEST_DSN=…`, `export PG_TEST_ADMIN_DSN=…`, `export PG_TEST_EMBED_URL=http://127.0.0.1:11999`.
- `scripts/test-env.sh down` — removes the containers and stops the embed server.
- `tests/fake_embed_server.py` — stdlib only. Serves `POST /v1/embeddings` and `POST /api/embed`
  with deterministic unit vectors derived from SHA-256 of the input (same text → same vector;
  similar prefixes need not be similar). Env/CLI: `--dim` (default 768), `--delay SECONDS` (per
  request, for timeout tests), `--fail-substring S` (HTTP 400 when the input contains S),
  `--require-bearer TOKEN` (401 without it). Can also be started in-process by tests via a fixture
  in `tests/conftest.py` (random port), so unit tests do not depend on the script.
- Everything works on a machine with only Docker and one Python ≥ 3.11. The Python matrix in gates
  runs inside `python:3.1x-slim` containers with `--network host` and the repo bind-mounted.

---

## 5. Work packages

### Wave 0 — harness and CI (1 Sonnet implementer, serial, no worktree)

**WP-0 Harness + CI.** Owns: `scripts/test-env.sh`, `tests/fake_embed_server.py`, `tests/conftest.py`
(new), `.github/workflows/ci.yml`, `scripts/check_versions.py`, `.gitignore`.

- Build §4 exactly.
- `.github/workflows/ci.yml`, on `push` and `pull_request`:
  - `lint`: `ruff check --select E9,F63,F7,F82 .` (syntax errors and undefined names only — do not
    introduce style churn).
  - `unit`: Python 3.11/3.12/3.13, `pip install -e ".[test]"`, `pytest -q` (no DB).
  - `live`: matrix pg16/pg17 using a `services:` container `pgvector/pgvector:pg1x`; steps create
    role `hermes`, run `migrate`, start the fake embed server, run `pytest -q` with the DSNs, then
    run every `docs/code-review/repro-2026-09-26/repro_*.py` that applies (H1 needs a cluster
    without role `hermes` — use a second service container with no role setup). **Until the fixes
    land these repro steps are expected to fail — add them with `continue-on-error: true` in WP-0 and
    flip to required in Gate G1.**
  - `build`: `python -m build`, `twine check dist/*`, assert the wheel contains
    `hermes_pgvector/migrations/*.sql` and `hermes_pgvector/plugin.yaml`.
  - `versions`: `python scripts/check_versions.py` — the version in `pyproject.toml` equals
    `plugin.yaml`'s; the dependency pins in `pyproject.toml` match `plugin.yaml`, the README's
    "Option 3: manual" block and `scripts/install.sh`.
- Add `build`, `twine`, `ruff` to a new `dev` extra (not `test`).
- `.gitignore`: add `.cache/` and `.venv-conformance/`.
- Acceptance: `scripts/test-env.sh up pg17 && scripts/test-env.sh db hermes_test` then the full
  suite passes with **0 skipped DB/embed tests** (218 passed on current code: 217 + the embed test).
  `python scripts/check_versions.py` exits 0. `actionlint` if available, else YAML parses.

Orchestrator after WP-0: merge, bring the harness up for pg17, run the three repro scripts and
record that each exits **1** (bug present) — this is the baseline the fixes must flip.

### Wave 1 — three parallel implementers (worktrees), disjoint files

**WP-A Store correctness.** Owns: `hermes_pgvector/store.py`, new `tests/test_v060_store.py`.
Findings: H3, M2, M6 (store part), L6, L8 (store part), M1 (code part), H2 (store part).

- H3: backfill uses keyset pagination (`AND id > %(last_id)s ORDER BY id LIMIT %(batch)s`), a single
  pass per table, each row counted once. Add a consecutive-failure breaker (constructor/param
  `max_consecutive_failures=20`): when tripped, stop that table and set `note: "aborted-consecutive-failures"`.
  Result dict keeps its keys and may add `note` and `skipped_changed`.
- M2: backfill UPDATE becomes `WHERE id = %s AND embedding IS NULL AND content = %s`; rowcount 0 →
  counted as `skipped_changed`, not failed.
- M1 (code): every `ON CONFLICT (agent_identity, target, content) DO NOTHING` → `ON CONFLICT DO
  NOTHING` (add, remap). Bulk import (`bulk_upsert_md`) catches a per-entry insert error, logs one
  warning per file with the count, and continues with the next entry.
- H2 (store): `replace()` and `remove()` gain an optional keyword `exact_content: str | None`. When
  given, match `content = %s` (lowest id), never `LIKE`. Without it, behaviour is unchanged
  (escaped substring `LIKE` — the fallback for older hosts).
- M6 (store): `health()` no longer counts rows; it returns `row_count_estimate` from
  `pg_class.reltuples` (−1/0 → `None`). Add `estimate_count(table)` helper. Keep `count()` for the
  scoped count (index-backed).
- L6: pure-vector `search()` and `search_turns()` add `embedding IS NOT NULL`.
- L8 (store): `scan_pii()` default pattern `(^|[^0-9])[0-9]{10,11}([^0-9]|$)`; honours `tables`.
- Acceptance: new tests for each item; `repro_h3_backfill.py` exits 0; the full suite passes (M1
  repro only passes after WP-B's migration — do not expect it here).

**WP-B Migrations + CLI.** Owns: `hermes_pgvector/migrations/002_agent_attribution.sql` (GRANT
block and comment characters only), `004_runtime_grants.sql` (same), new `005_content_md5_unique.sql`,
`hermes_pgvector/__main__.py`, new `tests/test_v060_migrations_cli.py`.
Findings: H1, L10, M1 (schema), L4, L8 (CLI part), plus `--version`.

- H1: 002's GRANTs move into a `DO $$ … $$` block that reads the role from
  `current_setting('hermes_pgvector.runtime_role', true)` (NULL/empty → `hermes`), uses
  `format('%I', role)` with `EXECUTE`, and `RAISE NOTICE`s (not errors) when the role is missing.
  004 gets the same role resolution. `apply_migration_as_admin()` / `apply_all_migrations()` (in
  `store.py` — coordinate: WP-B **may** edit only these two methods in `store.py`, and WP-A must not
  touch them) gain `runtime_role: str | None`, applied with `SELECT set_config('hermes_pgvector.runtime_role', %s, false)`
  on the admin connection before executing the file. CLI: `migrate --runtime-role NAME`.
- L10: replace every non-ASCII character in all migration files' comments with ASCII (`--`, `->`),
  and read migration files as UTF-8 but assert ASCII in a test so it cannot regress.
- M1 (schema): `005_content_md5_unique.sql` — `CREATE UNIQUE INDEX IF NOT EXISTS
  memory_entries_unique_md5 ON memory_entries (agent_identity, target, md5(content));` then
  `ALTER TABLE memory_entries DROP CONSTRAINT IF EXISTS memory_entries_unique;`. Header comment in
  the same style as 002/003 (why, live-apply safety, the `CREATE UNIQUE INDEX CONCURRENTLY` manual
  alternative for large tables). ASCII only.
- L4: `backfill` exit codes per §3; every command closes its store in `finally`.
- L8 (CLI): `cleanup` passes `--tables` to the PII scan.
- `hermes-pgvector --version` prints the installed distribution version
  (`importlib.metadata.version("hermes-memory-pgvector")`).
- Acceptance: `repro_h1_migrate_role.py` exits 0 against a fresh pg17 container with no `hermes`
  role; `repro_m1_unique_limit.py` exits 0 after `migrate`; migrating an already-migrated 0.5.5
  database is a no-op except 005; the full suite passes.

**WP-C Identity + writer + scrub.** Owns: `hermes_pgvector/identity.py`, `hermes_pgvector/writer.py`,
`tests/test_identity.py`, `tests/test_store_v04.py` (number scrub only), new
`tests/test_v060_identity_writer.py`. Findings: M7, L1, L3 (identity part), L5, M3 (writer part).

- M7: lowercase per §3; update the docstring's rule list (add `group-bucket`, L3). Existing tests
  that assert case preservation are updated deliberately (record which).
- L1: scrub every real-looking phone number in `identity.py`, `__main__.py` docstring (coordinate:
  WP-C may edit **only the module docstring** of `__main__.py`; WP-B owns the rest — if the merge
  conflicts, the orchestrator resolves), and the two test files. `git grep -nE '719[0-9]{7}|1719'`
  must return nothing afterwards.
- L5: remove the unused `_stop` event (or wire it into a documented hard-stop); keep `stats()`.
- M3 (writer): `AsyncWriter` exposes `draining` (read-only property). `shutdown(timeout)` keeps
  its signature.
- Acceptance: full suite passes; `git grep` check above is clean.

**Wave 1 verification.** Orchestrator merges `wp/A`, `wp/B`, `wp/C` into `release/0.6.0` (resolve
the two allowed overlaps), reruns the full suite and all repro scripts, then dispatches one fresh
Sonnet verifier with findings H1, H3, M1, M2, M6(store), M7, L1, L4–L6, L8, L10. Apply §1.4 on FAIL.

### Wave 2 — provider (`__init__.py`), two serial Sonnet implementers

**WP-D1 Write paths.** Owns: `hermes_pgvector/__init__.py`, new `tests/test_v060_provider_writes.py`.
Findings: H2 (provider), M3 (provider), M4, M5, L9.

- H2: `on_memory_write` puts `metadata["previous_content"]` (when a non-empty string) into
  `extra["previous_content"]` and removes it from the metadata that is persisted. `_worker`:
  replace/remove pass `exact_content=previous_content` when present. If exact is present and matches
  nothing: `replace` degrades to `add` (as today); `remove` logs at debug and does nothing — it must
  **not** fall back to `LIKE`. Without `previous_content` the old path is unchanged.
- M3: set a provider flag before `self._writer.shutdown()`; while it is set `_maybe_embed` returns
  `None` without calling the endpoint, so the drain is DB-only and fast (rows land text-only;
  backfill heals them). New config key `shutdown_drain_timeout` (default `10.0`).
- M4: `write_contexts` per §3; read `kwargs.get("agent_context")` in `initialize`; when excluded,
  `on_memory_write`, `sync_turn`, `on_session_end` and `on_delegation` return immediately; log once
  at info.
- M5: delegation turn rows per §3.
- L9: the `on_session_end` turn backstop runs whenever `sync_turns` is on; only the
  `register_agent` enqueue stays gated on migration 002.
- Acceptance: a test drives the real upstream `MemoryManager.notify_memory_tool_write` shape (build
  the metadata dict exactly as upstream does, with `replaced_entry`/`removed_entry`) against a
  seeded mirror containing a stale substring-matching row and proves the correct row changes; full
  suite passes.

**WP-D2 Read paths + config contract.** Owns: `hermes_pgvector/__init__.py`, new
`tests/test_v060_provider_reads.py`, `tests/test_embed_timeouts.py`, `tests/test_config_coercion.py`,
`hermes_pgvector/embed.py`. Findings: H4, M6 (provider), L2, L3 (provider/config), config schema.

- H4: implement `queue_prefetch(query, *, session_id="")` — one background daemon worker (at most
  one in flight; a newer request replaces a queued one) that embeds and searches and caches the
  formatted block keyed by `session_id`. `prefetch()` returns and clears the cached block for that
  session when present; otherwise it runs the synchronous path under a hard wall-clock
  `prefetch_budget` (default 5.0 s): `embed()` must enforce one overall deadline across the `auto`
  protocol's two attempts (pass the remaining budget as each attempt's timeout). Follow upstream's
  semantics from `agent/memory_provider.py` at the pinned ref. `embed_timeout` default → 5.0.
- M6: compute `system_prompt_block()` once per `initialize()` (upstream: "STATIC system-prompt
  text"), using the scoped `count()` and `estimate_count()` for the global number ("~N total").
- L2: `DEFAULTS["embed_url"] = "http://localhost:11434"` (module docstring example too).
- L3: fix the queue-full description ("newest writes are dropped"), the module docstring's tool
  list, and every `v0.x:` prefix that is now just noise in schema descriptions (keep "since" info
  in `docs/configuration.md` instead).
- Config schema: every entry gets `type` (`text|integer|number|boolean`), real-typed `default`,
  `minimum`/`maximum` where meaningful, and declare `identity_aliases`, `embed_write_backoff`,
  `write_contexts`, `prefetch_budget`, `shutdown_drain_timeout`. Keep all `_as_bool`/`_as_int`
  coercion (string values must keep working — test_config_coercion stays green).
- Optional, only if upstream's use of `identity_signature()` is clear from the pinned source:
  implement it returning the governance config (`allowed_themes`, `identity_aliases`, `bench_mode`,
  `write_contexts`) under `pgvector.`-prefixed keys. Otherwise skip and log.
- Acceptance: with the fake embed server at `--delay 20`, `prefetch()` returns within
  `prefetch_budget + 0.5` s and returns `""`; after `queue_prefetch()` completes, the next
  `prefetch()` returns the cached block in < 50 ms; full suite passes. (Driving upstream's real
  `_prefetch_provider` is covered later by WP-F test 5.)

**Wave 2 verification.** Fresh Sonnet verifier for H2, H4, M3, M4, M5, M6(provider), L2, L3, L9.
The verifier must also re-check that no hook can raise (invariant 4) by fuzzing each hook with
`None`/wrong types/an unreachable DSN.

### Wave 3 — two parallel implementers (worktrees)

**WP-E Docs, CHANGELOG, roadmap.** Owns: `README.md`, `ROADMAP.md`, `CHANGELOG.md` (new),
`docs/configuration.md`, `docs/operations.md`, `docs/scaling.md`, `docs/troubleshooting.md`,
`docs/upgrading.md`, `docs/release/RELEASING.md`, `SECURITY.md` (all new), `scripts/install.sh`
(text only).

- CHANGELOG.md in Keep a Changelog format: move every "New in v…" README section into it (newest
  first, content preserved), then the `0.6.0` entry with a **Breaking / upgrade notes** block:
  default `embed_url` changed (set it explicitly first); `embed_timeout` default 10 → 5; identities
  lowercased (find mixed-case themes with
  `SELECT DISTINCT agent_identity FROM memory_entries WHERE agent_identity <> lower(agent_identity)`
  and `remap` them); run `migrate` (applies 005; pass `--runtime-role` if your role is not `hermes`);
  `write_contexts` default skips subagent writes; `backfill` exit codes changed (update timers);
  `parent_session_id` meaning on delegation rows.
- README: slim to what/why, install, quick config, multi-agent themes, compatibility policy (§3
  "Public API"), support matrix, links to the docs set and CHANGELOG. Fix every drift item listed in
  the review.
- ROADMAP: fix every drift item in the review; mark M6 items with their status after this run;
  retarget M5 to 1.1/1.2; add the CHANGELOG row.
- `docs/configuration.md`: every key — type, default, since-version, effect — generated from
  `get_config_schema()` plus `DEFAULTS` (write a small script `scripts/gen_config_doc.py` so CI can
  check it is current; add that check to the `versions` job).
- `docs/operations.md`: migrate (incl. runtime role and PGOPTIONS form), backfill timer with exit
  codes and alerting, prune, cleanup, remap, stats, what to alert on (writer warnings, backfill
  exit ≠ 0, `remaining`).
- `docs/scaling.md`: HNSW `m`/`ef_construction`, `hnsw.ef_search`, filtered-search behaviour and
  `hnsw.iterative_scan` (pgvector ≥ 0.8, version-gated), partial indexes, pool sizing, embed latency
  vs `prefetch_budget`, estimated counts.
- `docs/release/RELEASING.md`: the manual PyPI checklist — version bump in both files, CHANGELOG
  date, `python -m build`, `twine check`, wheel content check, clean-venv smoke, TestPyPI
  rehearsal, signed tag, `twine upload`, post-release verification on one host, GitHub release.
- Acceptance: `python scripts/gen_config_doc.py --check` exits 0; every relative link resolves
  (write a tiny link checker or use `lychee --offline` if present); README ≤ 250 lines.

**WP-F Upstream conformance tests.** Owns: `conformance/` (new, at the repo root so the normal
`pytest` run in `tests/` never collects it), `scripts/conformance.sh` (new), the `conformance` job in
`.github/workflows/ci.yml`, `conformance/HERMES_AGENT_REF`.

- `scripts/conformance.sh [REF]`: clones hermes-agent at `REF` (default from `HERMES_AGENT_REF`)
  into `.cache/hermes-agent`, creates `.venv-conformance`, `pip install -e` hermes-agent (fall back
  to `PYTHONPATH` import of its source tree if its install fails — record which), installs this
  package, runs `pytest conformance -q`.
- Tests (skip cleanly when hermes-agent is not importable):
  1. `PgvectorMemoryProvider` subclasses upstream `MemoryProvider` and instantiates (no abstract
     methods left).
  2. For every method the plugin overrides, every parameter of the upstream signature is accepted
     (explicitly or via `**kwargs`) — `inspect.signature(...).bind(...)` with upstream's shape.
  3. The `hermes_agent.memory_providers` entry point named `pgvector` loads through upstream's
     `plugins/memory` loader and exposes `register`.
  4. `MemoryManager.notify_memory_tool_write` with a single-op and a batched `operations` result
     delivers `previous_content` to the provider, and the provider's mirror edits the exact row
     (DB-backed; skip without `PG_TEST_DSN`).
  5. `MemoryManager` prefetch with the fake embed server at `--delay 20` returns without the
     external-prefetch timeout warning.
- CI: `conformance` job on pull requests at the pinned ref, plus a weekly `schedule` against
  upstream `main` that is allowed to fail (reports drift without blocking).
- Acceptance: `scripts/conformance.sh` passes locally against the pinned ref.

**Wave 3 verification.** Orchestrator reviews both diffs, merges, runs the full suite, the
conformance script, `gen_config_doc.py --check`, the link check.

---

## 6. Gate G1 — release/0.6.0 is done

All of the following, recorded in the run log with outputs:

1. Version: `pyproject.toml` and `plugin.yaml` say `0.6.0`; CHANGELOG `0.6.0` entry dated `Unreleased`
   (the human dates it at release). `scripts/check_versions.py` exits 0.
2. Matrix: full suite passes on pg16 and pg17 (and pg18 if the image pulls) × Python 3.11, 3.12,
   3.13 (containers, §4) — with **no skipped DB/embed tests**.
3. Every `repro_*.py` exits 0; flip their CI steps to required.
4. `scripts/conformance.sh` passes.
5. Build: `python -m build`, `twine check dist/*`, wheel contains migrations `001`–`005` and
   `plugin.yaml`; install the wheel in a clean venv and run `hermes-pgvector --version` and
   `hermes-pgvector migrate` against a scratch database.
6. Upgrade rehearsal on pg17: fresh database; `pip install hermes-memory-pgvector==0.5.5` in a
   venv; `migrate`; seed through the 0.5.5 provider class (memory adds including a mixed-case
   identity, turns, a delegation, NULL-embedding rows); then install the branch, `migrate
   --runtime-role hermes`, and verify: row counts unchanged, `memory_entries_unique` gone and
   `memory_entries_unique_md5` present, duplicate adds still no-ops, `backfill` exits 0 and reaches
   `remaining: 0`, `remap` of the mixed-case theme works. Script it as
   `scripts/upgrade_rehearsal.sh` and commit it.
7. Final review: the orchestrator reads the whole `git diff origin/main...release/0.6.0` once,
   then a fresh Sonnet "red team" verifier gets the full finding list and the diff with the
   instruction to find any regression of the 2026-09-07 fixes (§5–§6 of that review) or any
   invariant break. Fix confirmed issues (§1.4).
8. Push `release/0.6.0`; open PR A into `main` (`gh pr create`, draft if anything H-level is
   BLOCKED) with the handoff (§8) as the body. If `gh` is authenticated, watch CI
   (`gh pr checks --watch`, 30 min cap) and fix failures, at most 3 rounds.

---

## 7. Gate G2 — release/1.0.0rc1

**WP-G (Sonnet, no worktree), on a new branch `release/1.0.0rc1` from `release/0.6.0`.** Owns:
`pyproject.toml`, `hermes_pgvector/plugin.yaml`, `CHANGELOG.md`, `README.md`, `ROADMAP.md`.

- Version `1.0.0rc1` in both files; classifier `Development Status :: 5 - Production/Stable`.
- CHANGELOG `1.0.0rc1` entry: "No behaviour changes relative to 0.6.0. Declares the compatibility
  policy stable."
- ROADMAP: M6 rows done except "released"; README compatibility section says 1.x guarantees.

Gate: `git diff --name-only release/0.6.0...release/1.0.0rc1` lists only those five files;
`scripts/check_versions.py` and the unit suite pass; build + `twine check` pass. Push; open PR B
with base `release/0.6.0` and a body stating it must merge only after 0.6.0 has been released and
soaked.

---

## 8. Handoff (PR A body and final chat message)

```
## hermes-memory-pgvector 1.0 prep — run report
Base: origin/main <sha>   Branches: release/0.6.0 (PR A), release/1.0.0rc1 (PR B)
Upstream conformance ref: <sha>

### Findings
| ID | Status (fixed / blocked / partial) | Commit | Test(s) |
...one row per H1–H4, M1–M7, L1–L10...

### Gates
G1.1–G1.8 and G2 with pass/fail and the key numbers (tests passed per matrix cell, repro exits).

### Decisions taken during the run
...from the run log...

### Manual steps for the maintainer
1. Review and merge PR A.
2. Before deploying 0.6.0 to the fleet: set `embed_url` explicitly in every host's config; find
   mixed-case themes and plan `remap`; update the backfill timer for the new exit codes; update the
   hermes-vps deploy runbook (`docs/memory/PGVECTOR-PLUGIN-DEPLOY.md` in that repo).
3. Release 0.6.0 to PyPI following docs/release/RELEASING.md; deploy; run `hermes-pgvector migrate`
   (with `--runtime-role` if not `hermes`); verify `hermes memory status` and `hermes-pgvector stats`.
4. Soak (duration is yours). Watch for writer warnings and backfill exit codes.
5. Merge PR B, release 1.0.0rc1, soak, then bump to 1.0.0 with no other change and release.
6. Decide whether to purge the scrubbed phone number from git history (not done by this run).
```

---

## 9. Out of scope for this run

M5 features (metrics, decay, partial indexes, MD re-sync, cross-provider import), `recall_status()`,
anything in the hermes-vps repo, publishing, and history rewrites.
