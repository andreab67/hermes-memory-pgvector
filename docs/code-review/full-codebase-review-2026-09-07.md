# Full-codebase review — hermes-memory-pgvector

**Date:** 2026-09-07
**Reviewer:** Claude Code (Opus 5 controller; Sonnet partition reviewers + implementers; Opus cross-cutting and final challenge)

---

## 1. Executive summary

A complete review-and-repair pass over all 24 tracked files (~4,600 lines) of
`hermes-memory-pgvector` at v0.4.3. Five review partitions (four per-file, one
cross-cutting) produced 24 candidate findings. After controller verification and
deduplication, **22 were confirmed actionable and 20 were fixed**; 2 were
deliberately deferred as product decisions rather than defects, and 8 further
candidates were rejected with recorded reasons.

The highest-value findings came from the cross-cutting pass, not the per-file
passes: three self-inconsistent config contracts entirely inside this repo, each
of which silently inverted a governance or safety control while appearing
correctly configured.

The single most consequential defect: **`allowed_themes` declared as a string in
the config schema but consumed as an iterable of names**, so a string allow-list
was iterated character-by-character and routed *every* theme in the fleet to
`default` — governance appearing configured while doing the exact opposite of
its purpose.

Note that this repository has **no CI pipeline configured**, so the exact-SHA
pipeline gate cannot pass; it resolves to `NO PIPELINE CONFIGURED`, reported
honestly rather than as a pass. Evidence in section 10.

---

## 2. Repository and Git state

| Item | Value |
|---|---|
| Repository | `hermes-memory-pgvector` (github.com/andreab67/hermes-memory-pgvector) |
| Target branch | `main` |
| Base SHA | `85eba5e26f0a8d422959639e2e50386f72a9c747` |
| Review branch | `code-review/full-codebase-review-20260907-0425` |
| Initial state | clean working tree, `main` in sync with `origin/main` (0 ahead / 0 behind) |
| Isolation method | Review branch cut from the clean target. No worktree needed (tree was clean); no pre-existing work to preserve. |
| History rewriting | None. No force-push, no reset, no history edits. |

---

## 3. Scope reviewed

Python 3.11+ single-package distribution: a Postgres + pgvector memory-provider
plugin for hermes-agent. One package (`pgvector/`), one operator CLI
(`hermes-pgvector`), four SQL migrations, one shell installer, five (now nine)
test modules.

### Coverage manifest

24 tracked files: **22 reviewed**, **2 documented exclusions**, **0 gaps**.

| Partition | Owner | Files |
|---|---|---|
| A | Sonnet reviewer | `pgvector/store.py` (1172 lines) |
| B | Sonnet reviewer | `pgvector/__init__.py` (1173 lines) |
| C | Sonnet reviewer | `writer.py`, `embed.py`, `identity.py`, `__main__.py`, `scripts/install.sh` |
| D | Sonnet reviewer | 5 test modules, 4 migrations, `pyproject.toml`, `plugin.yaml`, `.gitignore`, `README.md`, `ROADMAP.md` |
| E | Opus | Cross-cutting: invariants, contract drift, data integrity, migration/code coherence, identity end-to-end, config coherence |
| Final | Opus | Independent adversarial challenge of every fix + fresh full sweep |

**Exclusions (documented):** `LICENSE` (verbatim BSD-3-Clause text),
`.varmem/reviews.json` and `.varmem/.gitignore` (machine-generated adjudication
registry — consumed as review *input*, not reviewed as logic).

---

## 4. Baseline validation

| Check | Result |
|---|---|
| `python -m pytest tests/ -q` | 24 passed, 34 skipped |
| Lint / format / type tooling | **None configured** — no ruff/black/mypy/flake8 in `pyproject.toml`; no `.ruff.toml`, `setup.cfg`, `tox.ini`, `mypy.ini`, or `.pre-commit-config.yaml` |
| CI | **None configured** (see section 10) |

The 34 skips are live-DB tests gated on `PG_TEST_DSN` / `PG_TEST_EMBED_URL`.
That is the documented intended design (CLAUDE.md: "skip mode is fine without
DB") and was not treated as a defect. No check was weakened at any point.

---

## 5. Findings and disposition

**Totals:** 24 candidates → 22 confirmed actionable (after dedupe) → **20 fixed**,
**2 deferred**, 8 rejected.

By severity (confirmed): critical 0 · high 6 · medium 12 · low 4.

### 5.1 Fixed

| ID | Sev | Location | Defect |
|---|---|---|---|
| XCUT-1 | high | `__init__.py:309` to `identity.py:116` | `allowed_themes` declared as a scalar string in the config schema, documented as a list in the README, consumed as an iterable of names. A string was iterated CHARACTER BY CHARACTER, so every real theme failed the membership test and the whole fleet collapsed into `default`. Proven by execution. |
| XCUT-2 | high | `__init__.py`, 4 read sites | The config schema declares `embed_on_write`, `sync_turns`, `hybrid_search`, `bulk_sync_on_init` as the STRINGS "true"/"false"; the code read them with plain truthiness. `bool("false")` is `True`, so setting any toggle false via the schema path did nothing. Proven by execution. |
| XCUT-3a | high | `__init__.py` worker | `_healthy` is set once at init and never re-probed; a post-init Postgres failure logged at DEBUG only. Every durable write for the rest of the session lost with zero operator signal. Now warns once, then debug. |
| XCUT-3b | high | `__init__.py` `system_prompt_block` | A failed count fell through to the `count_all == 0` branch and told the model "Active. Empty store." — an affirmative false statement about a store that is merely unreachable. Now returns empty string. |
| STORE-1 | high | `store.py` `replace()` | Bulk UPDATE set every LIKE-match to the same content, colliding with the non-deferrable `UNIQUE(agent_identity,target,content)` constraint, raising UniqueViolation and updating 0 rows. Now single-row, matching built-in first-match semantics. |
| AUX-1 / CORE-3 | high | `README.md:158`, `scripts/install.sh:34` | Both the documented and scripted install paths still pinned `psycopg>=3.3.4` while `pyproject.toml` requires `>=3.3.5` — re-installing the exact version v0.4.3 exists to move off. Found independently by two partitions; deduped. |
| XCUT-4 | medium | `__init__.py` `sync_turn` + `on_session_end` | Both hooks enqueue `action="turn"` over overlapping content and `conversations` has no unique constraint, so every turn was written twice. **Verified against the live host contract**, not inferred. Now fingerprint-guarded. |
| INIT-1 | medium | `__init__.py` `on_session_end` | Never checked the `sync_turns` toggle that `sync_turn` honors, and hardcoded `min_chars=40` instead of reading `turn_min_chars`. An operator who disabled turn capture still got turns persisted. |
| INIT-2 | medium | `__init__.py` `sync_turn` | No try/except anywhere; an unguarded `int()` cast on config could raise into the agent turn path (INV-4). Body now fail-soft. |
| INIT-3 | medium | `__init__.py`, 5 sites | `.strip()` called on unvalidated tool args; a non-string `scope`/`target`/`query` raised AttributeError out of a `recall_*` hook. The adjacent `limit` parse was already guarded, confirming the asymmetry was an oversight. |
| INIT-4 | medium | `__init__.py` `initialize` | Unguarded `int(write_queue_maxsize)` aborted bootstrap on a non-numeric config value. |
| INIT-5 | medium | `__init__.py` `on_session_switch` | Reset only `_session_id`, leaving `_parent_session_id` stale, so session B's writes were tagged with session A's delegation parent (provenance corruption); `_embed_warned` likewise stale. |
| CORE-1 | medium | `embed.py:124` | Extractor guard omitted AttributeError; the Ollama Path B extractor uses `d.get()`, so a non-dict JSON payload escaped `embed()` uncaught. |
| CORE-2 | medium | `__main__.py` `cmd_install --remove` | The marker check short-circuited when `__init__.py` was absent, so a directory that was demonstrably not a generated shim was `rmtree`'d unconditionally without `--force`. Now fails closed. **Proven behaviorally.** |
| XCUT-5 | medium | `__init__.py` `save_config` | Replaced the whole `plugins.pgvector` subtree with schema-declared keys only, silently deleting live runtime-read keys (`identity_aliases`, `embed_write_backoff`). Now merges. |
| AUX-2b | medium | `plugin.yaml` | `pip_dependencies` omitted PyYAML, a real runtime dependency. **Failure mode corrected during verification:** the reviewer claimed ModuleNotFoundError; in fact all three `import yaml` sites sit in try/except that swallows it, so the operator's entire config block is *silently ignored* and the plugin runs on built-in defaults. |
| AUX-3 / XCUT-8 | medium | `tests/` | No coverage for AsyncWriter drain-on-shutdown, the bulk-import circuit breaker, the identity priority chain, or the writer action vocabulary. |
| XCUT-9 | low | `__init__.py:429` | Comment claimed the bulk sync "sees direct file edits"; it is insert-only and leaves stale rows alongside edited ones. Claim corrected. Orphan *reporting* deliberately not built (feature, not defect). |
| XCUT-10 | low | `plugin.yaml` | `hooks:` declared 2 of the 7 implemented hooks, omitting `on_memory_write` — the plugin's primary integration point. |
| AUX-2a | low | `plugin.yaml` | `pip_dependencies` carried no upper bounds, violating the cited AGENTS.md pinning rule. |
| STORE-2 | low | `store.py` `_get_pool` | `self._pool` read twice outside the lock; a concurrent `close()` between the reads returns None, then AttributeError at the call site. |
| AUX-4 | low | `README.md` | "New in" sections out of version order (0.4.3 spliced above 0.4.2). |

### 5.2 Deferred — product decisions, not defects

Both are real and evidenced. Neither was silently changed, because the repair
alters documented public behavior and that is the owner's call.

**XCUT-6 (medium) — read-side PII/bench exposure.** `scope='all'` applies no
identity filter, so `whatsapp-dm` and `_bench` rows surface in any theme's
recall; `scope` is also free-text, so a bucket can be named directly. The
PII/bench bucketing in `identity.py` is write-side only — there is no read-side
gate anywhere. *Why deferred:* INV-3 documents cross-theme recall as
intentionally opt-in; adding a read-side exclusion changes the semantics of a
public tool. **Recommend deciding this explicitly.**

**XCUT-7 (medium) — top-level import-name collision.** The distribution claims
the import name `pgvector`, which is owned by the widely-used pgvector-python
package. In a shared venv whichever installs last wins, the generated shim's
`from pgvector import PgvectorMemoryProvider` breaks, and the loader falls back
to built-in memory — fleet memory goes dark on a single log line. Confirmed
live: `/opt/hermes/hermes-agent/.venv/lib/python3.12/site-packages/pgvector/` is
this package. *Why deferred:* the durable fix is a rename to `hermes_pgvector`,
which is breaking and must be scheduled deliberately rather than slipped into a
review branch.

### 5.3 Rejected candidates (recorded)

- **ROADMAP version-agreement rule — inapplicable, not violated.** ROADMAP is
  milestone-structured (M1 through M5) and carries no single version string. The
  reviewer was explicitly instructed not to manufacture a violation, and did not.
- `store.py` missing commits / leaked transactions — ruled out by reading
  `psycopg_pool`'s actual source (`connection()` wraps in `with conn:`, so every
  exception path auto-rolls-back before the connection returns to the pool).
- SQL injection via table-name f-strings — ruled out; all gated by
  `_assert_whitelisted` against `CLEANUP_WHITELIST`.
- `search()` post-LIMIT `min_similarity` filter — ruled out; results are
  pre-sorted by descending similarity, so the property is monotonic.
- `remap_identity` `me_old`/`dupes`/`conv_old` — pre-adjudicated as NOT
  duplicates in `.varmem/reviews.json`; merging them would be a bug.
- `cmd_prune` `days > 0` audit-log skip — ruled out; `older_than_days <= 0` is
  an explicit no-op that executes no DELETE.
- Writer shutdown-then-re-enqueue race — ruled out; the item is delayed, not
  dropped.
- **`on_session_end` signature mismatch — controller hypothesis, refuted by
  evidence.** `server.py`'s `invoke_hook` targets the CLI plugin system; the
  memory provider is driven by `memory_manager.py:988`, which calls
  `provider.on_session_end(messages)` positionally. The signature is correct.

---

## 6. A regression introduced, caught, and fixed

Recorded because it materially affects confidence in the process.

The XCUT-4 dedup initially recorded a turn fingerprint *before* calling
`AsyncWriter.enqueue()`. But `enqueue()` returns `False` when its bounded queue
is full and the write is **dropped**. Fingerprinting unconditionally meant a
dropped turn would be marked as captured, and `on_session_end` would then skip
it — converting a recoverable drop into permanent data loss, and defeating the
very backstop the change was meant to preserve.

Caught by checking `writer.py`'s actual drop semantics rather than assuming
them. Fixed to fingerprint **only on accepted enqueue**, and pinned by two new
tests (`test_dropped_turn_is_not_fingerprinted`,
`test_accepted_turn_is_fingerprinted`).

---

## 7. Tests added

New: `tests/test_config_coercion.py`, `tests/test_async_writer.py`,
`tests/test_turn_dedup.py`, `tests/test_tool_args_hardening.py`. Plus a
docstring correction in `tests/test_smoke.py` — its claim that the provider "is
only exercised when the plugin runs inside hermes-agent" was stale; the provider
constructs standalone via the `MemoryProvider = object` ImportError fallback.

**24 passed → 53 passed** (+29). Skip count unchanged at 34. No test was
weakened, skipped, or deleted.

**Honest caveat, surfaced by the test author rather than hidden:** 4 of the
tool-arg tests exercise the outer fail-soft contract but pass against unfixed
code too, because `handle_tool_call` short-circuits on `_healthy` before
touching the args. The 3 deeper tests (healthy provider plus fake store) are the
ones that genuinely pin the defect. Both sets were kept and the distinction
documented. The controller independently confirmed this by reverting one
coercion, observing `AttributeError: 'int' object has no attribute 'strip'`,
then restoring.

---

## 8. Validation performed

| Command | Result |
|---|---|
| `python -m pytest tests/ -q` (baseline) | 24 passed, 34 skipped |
| `python -m pytest tests/ -q` (final) | **53 passed, 34 skipped** |
| `python -c "import pgvector, pgvector.store, pgvector.embed, pgvector.__main__"` | imports OK |
| `bash -n scripts/install.sh` | syntax OK |
| `yaml.safe_load(open('pgvector/plugin.yaml'))` | valid; reflects all three changes |
| `python -m pgvector install --help` | parses; `--force` present on subparser |
| Behavioral: `install --remove` on non-shim dir | refuses, exit 1, operator file survives |
| Behavioral: `install --remove --force` | proceeds, exit 0 |
| Behavioral: `_as_bool` / `_as_theme_list` / `normalize_identity` | string forms now handled correctly |
| Behavioral: revert-one-coercion | target test fails as predicted, then restored |
| Secret scan of full diff | no secret-shaped additions |

---

## 9. Convergence history

| Pass | Activity | Outcome |
|---|---|---|
| 1 | 4 Sonnet partition reviews + 1 Opus cross-cutting review | 24 candidates → 22 confirmed after controller verification and dedupe |
| 1 | Repair: 2 Sonnet implementers (disjoint file ownership) + controller on `__init__.py` | 20 fixed; 2 deferred with rationale |
| 1 | Test coverage: 1 Sonnet implementer | +29 tests |
| 2 | Controller self-check of its own change | 1 self-introduced regression found and fixed (section 6) |
| 2 | Opus independent adversarial challenge + fresh full sweep | recorded in the final response |

No two agents ever held write access to the same file. All review workers were
read-only; the controller assigned every edit and owned all Git state.

---

## 10. Pipeline

**`NO PIPELINE CONFIGURED`** — proven three independent ways:

1. `git ls-files` matches nothing for `.github`, `.gitlab`, `azure-pipelines`,
   `jenkins`, `circleci`, `travis`, `appveyor`, `woodpecker`, or `drone`.
2. No `.github/` directory and no `.gitlab-ci.yml` on disk.
3. `gh api repos/andreab67/hermes-memory-pgvector/actions/workflows` returns
   `total_count: 0` with an empty workflow list.

This is reported as *not configured*, **not** as passing. The exact-SHA pipeline
gate therefore cannot be satisfied by this repository as it stands, and local
validation (section 8) is the only available evidence.

Per the delivery protocol, a committed report cannot self-record the pipeline
result for its own final SHA without creating a further SHA; authoritative
final-SHA status belongs in the merge proposal and the final response.

---

## 11. Residual risks

1. **34 live-DB tests never ran.** Everything touching real SQL — including the
   `replace()` rewrite (STORE-1), the single most behaviorally significant code
   change — is covered by reading, not execution. **Running the suite once with
   `PG_TEST_DSN` set against a scratch database is the highest-value follow-up.**
2. **No CI**, so nothing prevents this class of drift from recurring. The stale
   psycopg pin appeared in two files and survived a release; a three-line
   consistency check comparing `pyproject.toml` against `plugin.yaml`,
   `README.md`, and `scripts/install.sh` would have caught it.
3. **XCUT-6 and XCUT-7 remain open by design** (section 5.2).
4. **`plugin.yaml`'s `hooks:` semantics are unconfirmed.** The list was expanded
   to all 7 implemented hooks, which is strictly more accurate, but whether
   hermes-agent treats the field as authoritative or informational was not
   determined. If authoritative, the previous 2-hook list may have been
   suppressing `on_memory_write` — worth confirming before the next release.
5. **`replace()` semantics changed** from "update all matches" to "update first
   match". This matches the built-in memory tool and the docstring's own
   admission, and the previous behavior raised UniqueViolation on any
   multi-match so it could not have been working as documented. It is still a
   behavior change and is called out explicitly.

---

## 12. Commits and rollback

All work is on `code-review/full-codebase-review-20260907-0425`, branched from
`85eba5e`. `main` was never modified and never merged into.

Rollback: delete the local and remote review branch. No history was rewritten,
so nothing on `main` needs reverting.

---

## 13. Recommendation

The confirmed defects are fixed, the suite more than doubled, and no
architectural invariant was violated. Two items are deferred to the owner by
design, and the exact-SHA pipeline gate cannot be satisfied because no pipeline
exists.

**Before merge:** run the suite once with `PG_TEST_DSN` set, to exercise the 34
live-DB tests — the `replace()` rewrite is the highest-impact change and is
currently unexecuted.

**After merge:** decide XCUT-6, schedule XCUT-7, and add a minimal CI workflow.
