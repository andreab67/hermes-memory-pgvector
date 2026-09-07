# Full-codebase review — hermes-memory-pgvector

**Date:** 2026-09-07
**Reviewer:** Claude Code (Opus 5 controller; Sonnet partition reviewers + implementers; Opus cross-cutting and final challenge)

---

## 1. Executive summary

A complete review-and-repair pass over all 24 tracked files (~4,600 lines) of
`hermes-memory-pgvector` at v0.4.3. Five review partitions (four per-file, one
cross-cutting) produced 24 candidate findings. After controller verification and
deduplication, **22 were confirmed actionable**; a second adversarial pass added
6 more. **All 28 are fixed** — including the two originally deferred as product
decisions, which the owner subsequently asked to be fixed — and 8 further
candidates were rejected with recorded reasons.

**Two regressions were introduced during repair and both were caught before
merge** — one by the controller self-checking its own change, one by the
independent challenge. Both are documented in section 6, not glossed over.

The release is **v0.5.0**, not a patch bump: it carries a breaking import rename
(`pgvector` → `hermes_pgvector`) alongside the behavior fixes. v0.4.3 is already
published to PyPI describing itself as "No code changes", so none of this could
ship under it.

The highest-value findings came from the cross-cutting pass, not the per-file
passes: three self-inconsistent config contracts entirely inside this repo, each
of which silently inverted a governance or safety control while appearing
correctly configured.

The single most consequential defect: **`allowed_themes` declared as a string in
the config schema but consumed as an iterable of names**, so a string allow-list
was iterated character-by-character and routed *every* theme in the fleet to
`default` — governance appearing configured while doing the exact opposite of
its purpose.

This repository defines **no CI workflow**, and the only check that runs on PR
commits is a third-party app that is currently disabled — so **no passing
pipeline evidence exists for the final SHA, and none is claimed**. Evidence, and
a correction to an earlier over-broad claim, are in section 10.

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
plugin for hermes-agent. One package (`hermes_pgvector/`, renamed from `pgvector/` in this branch), one operator CLI
(`hermes-pgvector`), four SQL migrations, one shell installer, five (now nine)
test modules.

### Coverage manifest

24 tracked files: **22 reviewed**, **2 documented exclusions**, **0 gaps**.

| Partition | Owner | Files |
|---|---|---|
| A | Sonnet reviewer | `hermes_pgvector/store.py` (1172 lines) |
| B | Sonnet reviewer | `hermes_pgvector/__init__.py` (1173 lines) |
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
| INIT-5 | medium | `__init__.py` `on_session_switch` | **PARTIALLY WITHDRAWN — see section 6.2.** The `_embed_warned` / `_db_warned` / `_turn_fingerprints` resets are correct and were kept. The `_parent_session_id` half was WRONG and has been reverted: the host's `on_session_switch(parent_session_id=...)` carries the previous session in the agent's own lineage, not a delegation parent. |
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

### 5.2 Originally deferred, then fixed on request

Both were held back from the first pass because the repair alters documented
public behavior, which is the owner's call rather than a reviewer's. The owner
then asked for both; both are implemented in this branch.

**XCUT-6 (medium) — read-side PII/bench exposure. FIXED.** `scope='all'` applied
no identity filter, so `whatsapp-dm` and `_bench` rows surfaced in any theme's
recall; `scope` is also free-text, so a bucket could be named directly. The
bucketing in `identity.py` was write-side only — it strips PII from the
*identity*, but message bodies still live in `content`, and with turn capture on,
a reply quoting leaked DM text was written back under the *reading* theme,
permanently re-attributing it.

*Fix:* `scope='all'` now passes an explicit `exclude_identities` down all six
recall paths — including the hybrid-failure fallback, which would otherwise have
let any transient error bypass the gate — and naming a sink explicitly is
rejected before the store is queried. The gate deliberately does not over-reach:
an agent whose own identity IS a bucket keeps full access to its own rows, and
ordinary cross-theme recall (`scope='sales'`, etc.) is unchanged. 11 tests;
disabling the gate fails 8 of them.

**XCUT-7 (medium) — top-level import-name collision. FIXED.** The distribution
claimed the import name `pgvector`, owned by the widely-used pgvector-python
package. In a shared venv whichever installed last won, the generated shim's
`from pgvector import PgvectorMemoryProvider` broke, and the loader fell back to
built-in memory — fleet memory dark on a single log line. Confirmed live:
`/opt/hermes/hermes-agent/.venv/lib/python3.12/site-packages/pgvector/` is this
package.

*Fix:* the import package is renamed `pgvector` → `hermes_pgvector`, which is why
this release is v0.5.0. Three namespaces were deliberately left alone because
they are not the colliding one: the distribution `hermes-memory-pgvector`, the
`hermes-pgvector` CLI, and the hermes provider name `pgvector`
(`memory.provider: pgvector`, discovery dir `$HERMES_HOME/plugins/pgvector/`) —
renaming that last one would break every existing operator config for no benefit.

The upgrade is **not** transparent, and the README says so prominently: a
pre-0.5.0 shim still reads `from pgvector import ...`, and the hermes-agent
loader treats the resulting ImportError as "plugin absent", falling back to
built-in memory. `hermes-pgvector install --force` must be re-run once after
upgrading. To make that failure mode louder in future, `install` now verifies
after writing the shim that `import hermes_pgvector` actually resolves to this
package, and warns if anything shadows it.

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
  evidence.** (Note: `CLAUDE.md` states the hermes-agent checkout is gone
  from this workstation. **That note is stale — the checkout exists at
  `~/hermes-agent`**, and the final review pass used it directly. Earlier
  passes had to read the deployed copy on the hermes VM instead. Worth
  correcting in `CLAUDE.md`.) `server.py`'s `invoke_hook` targets the CLI plugin system; the
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

### 6.2 A second regression — caught by the independent challenge, not by me

The INIT-5 fix added `self._parent_session_id = kwargs.get("parent_session_id")`
to `on_session_switch`. That was **wrong, and worse than the defect it claimed
to fix.**

The host's `on_session_switch(parent_session_id=...)` does not carry a
delegation parent. It carries **the previous session in the same agent's
lineage**: `hermes_cli/cli_session_mixin.py:585` passes `old_session_id` on
`/new`, `cli_commands_mixin.py:349` the original session on `/resume` and
`/branch`, `conversation_compression.py:1431` the boundary parent on
compression, and `cli_session_mixin.py:903` passes an empty string on `/undo`.

But the plugin uses `_parent_session_id` *exclusively* for delegation
provenance — it flows into `conversations.parent_session_id`, which
`migrations/002_agent_attribution.sql:74` defines as "parent-session linkage for
delegation traceback". So the fix would have stamped a **fabricated delegation
edge** on every turn and memory write after any `/new`, `/resume`, `/branch` or
context compression — which fires on any long session, so continuously in
production — and the `/undo` path would have **erased a genuine subagent's real
delegation parent** mid-session.

Worse, the defect it was meant to fix was unreachable on the primary path:
`agent/agent_init.py:_memory_provider_init_kwargs` never passes
`parent_session_id` to the provider at all, so the field was always `None` there
to begin with.

**Reverted.** The `_embed_warned` / `_db_warned` / `_turn_fingerprints` resets in
the same method are correct and were kept; clearing the fingerprints is provably
safe because the host runs `on_session_end` strictly before `on_session_switch`
(`memory_manager.py:597-620`). A regression test now pins the contract so the
line is not "helpfully" re-added.

**Why this matters for confidence in the whole pass:** this was the one change
in the set with no test, and it was authored by the controller rather than a
worker. It was caught only because an independent reviewer was given an explicit
mandate to refute the fixes, and had access to evidence the earlier passes did
not. It is recorded here rather than quietly corrected.

---

### 6.3 Three false claims in my own output — caught by the third challenge

The v0.5.0 work (read-side gate + import rename) went through a further
adversarial pass. It found no correctness, security, or concurrency defect in
the new code, but it did find that **three statements I had written were false**:

1. **"`install` warns loudly if something shadows this package."** The check I
   added was a tautology: `__main__.py` does `from . import DEFAULTS`, so the
   package is already in `sys.modules` and `import hermes_pgvector` could never
   resolve anywhere else. It could not fire under any circumstance, while the
   README and this report both advertised it as a safety net. Replaced with a
   real check that runs a clean subprocess — verified to fire against a decoy
   package, and to correctly report a from-clone environment where the
   distribution is not installed.

2. **"The host runs `on_session_end` strictly before `on_session_switch`."**
   True only for `commit_session_boundary_async`.
   `agent/conversation_compression.py` calls `on_session_switch` directly with no
   `on_session_end` at all — and compression fires on any long session. Because
   the fix cleared the fingerprint set on switch, the turn double-write it exists
   to prevent came straight back on every compression (reproduced: 4 enqueues
   instead of 2). Fingerprints now survive session switches and are bounded by
   `_FINGERPRINT_CAP` instead. The test that encoded the false guarantee was
   rewritten to assert the correct contract.

3. **"hermes-agent never looks at installed packages, so `pip install` alone is
   invisible to it."** False against the checked-out host:
   `plugins/memory/__init__.py` has supported the `hermes_agent.memory_providers`
   entry-point group since 2026-05-02, and hermes-agent's own `CONTRIBUTING.md`
   says "or via a pip entry point". This package now **declares that entry
   point**, so on a supporting host `pip install` is sufficient and no shim is
   needed. (A shim *directory* still takes precedence over the entry point, so a
   stale shim still wins — which is why the upgrade instructions now recommend
   removing it rather than regenerating it.)

It also caught two operator-facing breaks the rename introduced, both blocking:
shipped help text and the `SchemaNotApplied` message still told operators to run
`python -m pgvector`, and the from-clone install path still said `cp -r pgvector`
while `install.sh` advertised a console script it never installs.

**The `python -m pgvector` break was live.** The deployed unit
`/etc/systemd/system/hermes-pgvector-backfill.service` runs
`ExecStart=.../python -m pgvector backfill`; upgrading to 0.5.0 would have
stopped nightly re-embedding silently, leaving rows written during an embed
outage permanently unsearchable. The README upgrade block now leads with that,
including the `grep` to find affected units.

### 6.4 Round 4 — group-key PII, and a drop-safety bug the PR bot caught

**Group / channel / thread keys were not bucketed.** Flagged in round 3 as
pre-existing and deferred; the owner asked for it. The host builds session keys
as `<ns>:<platform>:<chat_type>[:<chat_id>][:<thread_id>][:<user>]`
(`gateway/session.py:build_session_key`), and with `group_sessions_per_user` —
the default — the trailing segment is the participant id, a phone number on
WhatsApp/SMS/Signal. Only `dm` was bucketed, so `group`, `channel` and `thread`
keys stored that phone number verbatim as an `agent_identity`: the exact failure
`identity.py` exists to prevent, reached through a different `chat_type`. Under a
configured allow-list they instead collapsed to `default`, where every theme
could read the external content.

Now bucketed to a single `external-group` sink, added to `_ALWAYS_ALLOWED` (so a
strict allow-list cannot route it to `default`) and to the read-side restricted
set. The pattern is anchored on `<platform>:<chat_type>:` rather than a bare
chat-type token, deliberately: a loose pattern would sweep ordinary namespaced
themes like `eng:channel:alerts` — the same trap the v0.4.2 note records for a
bare `:signal:` alternative. Tested against all six real key shapes plus six
benign look-alikes.

**`on_session_end` fingerprinted turns before the enqueue was accepted.** Found
by the PR review bot, and it was right. The drop-safety contract was applied to
`sync_turn` but not to `on_session_end` — and it matters *more* there, because
that hook replays an entire transcript at once through a 256-slot queue, so a
full queue is most likely exactly at that moment. A dropped turn was marked
captured, and since fingerprints now survive session switches (section 6.3), the
next replay skipped it permanently.

My earlier reasoning for leaving it — "on_session_end is the last chance, so it
does not matter" — was simply wrong: the host calls it *again* on session
rotation with the same message list. Fixed to fingerprint only on accept, with
two regression tests verified to fail against the reintroduced defect.

---

## 7. Tests added

New: `tests/test_config_coercion.py`, `tests/test_async_writer.py`,
`tests/test_turn_dedup.py`, `tests/test_tool_args_hardening.py`. Plus a
docstring correction in `tests/test_smoke.py` — its claim that the provider "is
only exercised when the plugin runs inside hermes-agent" was stale; the provider
constructs standalone via the `MemoryProvider = object` ImportError fallback.

**24 passed → 72 passed** (+48); skips 34 → 35 (one added live-mode test for
`replace()`). No test was weakened, skipped, or deleted.

Each new non-live test was empirically verified to FAIL against the reintroduced
bug. For the read-side gate, disabling `_restricted_identities()` fails 8 of its
11 tests — and the 3 that still pass are precisely the "must not over-reach"
assertions, which should hold either way.

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
| **`PG_TEST_DSN=... python -m pytest tests/ -q`** (live) | **126 passed** (see 8.1) |
| `python -m pytest tests/ -q` (final) | **74 passed, 35 skipped** |
| `python -c "import hermes_pgvector, hermes_pgvector.store, hermes_pgvector.embed"` | imports OK |
| `bash -n scripts/install.sh` | syntax OK |
| `yaml.safe_load(open('hermes_pgvector/plugin.yaml'))` | valid; reflects all three changes |
| `python -m hermes_pgvector install --help` | parses; `--force` present on subparser |
| Behavioral: `install --remove` on non-shim dir | refuses, exit 1, operator file survives |
| Behavioral: `install --remove --force` | proceeds, exit 0 |
| Behavioral: `_as_bool` / `_as_theme_list` / `normalize_identity` | string forms now handled correctly |
| Behavioral: revert-one-coercion | target test fails as predicted, then restored |
| Secret scan of full diff | no secret-shaped additions |

---

### 8.1 The live-DB run

The earlier passes could not execute anything touching real SQL. That gap is now
closed. A throwaway Postgres 16 + pgvector 0.8.6 container was provisioned, all
four migrations applied through the real `hermes-pgvector migrate` path, and the
full suite run against it.

**Result: 126 passed.** Everything previously verified only by reading is now
executed — including `replace()`'s single-row UPDATE (the fix for the
`UniqueViolation` that silently updated zero rows) and the new
`agent_identity <> ALL(%s)` exclusion filter.

Two things worth recording:

**A defect only execution could find.** `test_bulk_upsert_md_skips_existing`
failed: expected 3 parsed entries, got 1. Cause: the test called
`md.write_text(...)` with no `encoding=`, so on a cp1252 default locale
(Windows) the `§` entry delimiter was written as the single byte `0xA7`.
`bulk_upsert_md` reads UTF-8 with `errors="replace"`, turning it into U+FFFD, so
the delimiter never matched and the file parsed as one entry. A **test
portability bug, not a product bug** — and proof this suite had never been
executed on this platform. Fixed by pinning the encoding, with the reason
recorded in the test so it is not "cleaned up" later.

**Live coverage added for the exclusion filter.** The gate's *policy* was already
covered DB-free, but the SQL was not. `tests/test_exclude_identities_live.py`
now exercises `exclude_identities` against real Postgres on all four store
methods, in both the positional and named-parameter binding forms, plus the
full-text-only RRF leg, an empty list, an absent identity, and a literal
containing `%` and a quote (proving it is a bound parameter, not interpolation).
Verified non-vacuous: neutering the positional filter fails exactly the three
positional tests.

### 8.2 An operational finding from the live run

The one test still failing under `PG_TEST_EMBED_URL` is not a code defect. The
configured embed endpoint (`192.168.100.50:11434`, nomic-embed-text) responds in
**6.6–16.7 s** across repeated back-to-back calls, while `embed()`'s default
timeout is **10 s**. So a substantial share of production embeds time out, take
the fail-soft path, and land as text-only rows with NULL embeddings.

That is by design — but it means the plugin depends heavily on the nightly
backfill sweep to make those rows searchable again, **and that sweep is exactly
what the `python -m pgvector` break in section 6.3 would have silently killed**.
The two findings compound. Worth deciding whether to raise the default timeout,
independently of this branch.

---

## 9. Convergence history

| Pass | Activity | Outcome |
|---|---|---|
| 1 | 4 Sonnet partition reviews + 1 Opus cross-cutting review | 24 candidates → 22 confirmed after controller verification and dedupe |
| 1 | Repair: 2 Sonnet implementers (disjoint file ownership) + controller on `__init__.py` | 20 fixed; 2 deferred with rationale |
| 1 | Test coverage: 1 Sonnet implementer | +29 tests |
| 2 | Controller self-check of its own change | 1 self-introduced regression found and fixed (section 6) |
| 2 | Opus independent adversarial challenge + fresh full sweep | **NOT READY** — 1 HIGH regression in a controller-authored fix (section 6.2), plus 6 actionable non-blocking findings |
| 2 | Repair round 2: revert the regression, guard remaining casts, bump to 0.4.4, close test gaps | blocking finding resolved; +8 tests |
| 3 | Owner asked for the two deferred items; implemented read-side gate + import rename (v0.5.0) | +11 gate tests |
| 3 | Opus independent challenge of v0.5.0 | **NOT READY** — 2 blocking operator-instruction defects introduced by the rename, plus 9 non-blocking findings including three false claims in my own docs |
| 3 | Repair round 3: stale CLI invocations, from-clone path, real shim verification, fingerprint retention, entry point, doc corrections | blocking findings resolved; +2 tests |

No two agents ever held write access to the same file. All review workers were
read-only; the controller assigned every edit and owned all Git state.

---

## 10. Pipeline

**No repository-defined CI pipeline** — proven three independent ways:

1. `git ls-files` matches nothing for `.github`, `.gitlab`, `azure-pipelines`,
   `jenkins`, `circleci`, `travis`, `appveyor`, `woodpecker`, or `drone`.
2. No `.github/` directory and no `.gitlab-ci.yml` on disk.
3. `gh api repos/andreab67/hermes-memory-pgvector/actions/workflows` returns
   `total_count: 0` with an empty workflow list.

### Correction: a third-party check DOES run on PR commits

An earlier draft of this report stated flatly that no pipeline exists. That was
incomplete. No CI *workflow* is defined in the repository — the three proofs
above hold — but a third-party GitHub App check runs on pull-request commits,
and it was found only after pushing:

| Field | Value |
|---|---|
| Check | `Kilo Code Review` (app `kilo-code-bot`) |
| SHA | `2f9ad373564563c2737867b0f79883828a18c93a` (exact final pushed SHA) |
| Status | completed |
| Conclusion | **`action_required`** — not a pass |
| Reported title | "Code Reviewer disabled: model unavailable" |
| Reported summary | "Code Reviewer was disabled because the selected model is not available for cloud agent sessions. Choose an available model, then enable Code Reviewer again." |

The base `main` SHA `85eba5e` has **0** check runs, confirming this app fires
only on pull requests.

**This failure is not caused by anything in this branch.** It is an account-level
Kilo configuration state (a selected model unavailable for cloud agent sessions)
and cannot be fixed by a code change here. It was therefore not treated as a
review-caused CI failure to repair, and it was not retried as an infrastructure
transient, because it is a deterministic configuration state rather than a flake.

**The exact-SHA gate is consequently NOT satisfied**, and no claim is made that
it is. Local validation (section 8) is the only passing evidence available.

Per the delivery protocol, a committed report cannot self-record the pipeline
result for its own final SHA without creating a further SHA; authoritative
final-SHA status belongs in the merge proposal and the final response.

---

## 11. Residual risks

1. ~~34 live-DB tests never ran.~~ **RESOLVED — they were executed.** See
   section 8.1. All migrations applied and the full suite ran against a real
   Postgres 16 + pgvector 0.8.6 instance: **126 passed**. This surfaced one
   defect that no amount of reading would have (section 8.1).
2. **No CI**, so nothing prevents this class of drift from recurring. The stale
   psycopg pin appeared in two files and survived a release; a three-line
   consistency check comparing `pyproject.toml` against `plugin.yaml`,
   `README.md`, and `scripts/install.sh` would have caught it.
3. **The v0.5.0 upgrade needs a manual step.** `hermes-pgvector install --force`
   must be re-run after `pip install -U`, or the stale shim silently disables the
   plugin. The hermes-vps deploy runbook (`docs/memory/PGVECTOR-PLUGIN-DEPLOY.md`)
   currently describes deploy as `pip install -U` + migrations + restart, and
   **must be updated before this ships**.
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

**After merge:** update the hermes-vps deploy runbook for the `install --force`
step, and add a minimal CI workflow.

**Correct `CLAUDE.md`:** it states the hermes-agent checkout is gone from this
workstation. It exists at `~/hermes-agent`, and having it available materially
improved the final review pass — the blocking regression in section 6.2 was only
provable against that source.
