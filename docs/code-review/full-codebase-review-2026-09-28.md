# Full-codebase review -- hermes-memory-pgvector (pre-1.0.0)

Date: 2026-09-28. Terminal status: **READY FOR MERGE** (subject to the
exact-SHA CI result recorded on the merge proposal -- see "Validation").

| | |
|---|---|
| Repository | `andreab67/hermes-memory-pgvector` |
| Target branch / base | `main` @ `629a07a` (1.0.0rc1 merged, PR #11) |
| Review branch | `code-review/full-codebase-review-20260928-2136` |
| Reviewed range | `629a07a..HEAD` (17 commits including this report) |
| Upstream contract | `NousResearch/hermes-agent` @ `d0288be5` (`conformance/HERMES_AGENT_REF`) |

## Executive summary

A five-pass, whole-repository review ahead of the 1.0.0 release. It found
**1 High** (data loss), **10 Medium** and **54 Low** actionable issues. All
are fixed on the review branch with regression tests, except two the owner
accepted as documented behaviour (PROV-2 Medium, PROV-4 Low). The test suite
grew from 442 to 528 tests (0 skipped against live Postgres + pgvector), and
CI now fails on any skipped test in the live and conformance jobs.

The High: `hermes-pgvector remap --old X --new X --execute` deleted every
`memory_entries` row for `X` and exited 0.

The most consequential Medium class was identity bucketing: session keys for
Telegram forum topics, LINE rooms and webhook deliveries (all emitted by the
pinned upstream gateway) carried a participant or delivery id into a theme of
its own, readable from any theme via `scope='all'`. The fifth pass
enumerated every chat type upstream feeds into `build_session_key` and
confirmed every generated key shape is now bucketed.

## Method and isolation

- Initial state: branch `claude/codebase-review-sonnet-5-5-65b7fc`, clean,
  at `629a07a`. The review branch was created in place from `origin/main`
  (clean tree, so no extra worktree was needed) and its upstream unset so
  nothing could push to `main`. `main` was never modified.
- Orchestration: one controller (Opus) owned the coverage manifest, findings
  ledger, git state, commits and CI; all review, verification and
  implementation work was delegated to Sonnet subagents, per the maintainer's
  instruction to use Sonnet 5.5 subagents -- including the independent
  challenge role the review skill normally assigns to Opus. At most four
  workers ran concurrently with non-overlapping file scopes; review workers
  were read-only; the controller reviewed every diff before committing.
- Every candidate needed a concrete failure scenario. Most were reproduced
  live (throwaway databases on a pgvector/pg17 container, the repo's fake
  embed server, the real upstream hermes-agent source at the pinned ref).
  Test-gap findings were proven by mutation: a one-line production change
  the suite failed to catch. Contested findings went to an independent
  skeptic, and each fix's new test was shown to fail on the old code.
- Environment: WSL2 Ubuntu, Python 3.12, Postgres 17 + pgvector
  (`scripts/test-env.sh`); GitHub Actions for the full matrix.

## Coverage

`git ls-files`: 79 tracked files at base, 85 at HEAD (six new test files and
this report). Every human-maintained file was reviewed:

| Partition | Files at HEAD | Passes |
|---|---|---|
| Provider (`__init__.py`, `identity.py`, `plugin.yaml`) | 3 | 1-5 |
| Store / writer / embed / CLI + migrations 001-005 (`hermes_pgvector/` also has `__main__`) | 9 | 1-5 |
| Scripts, CI, packaging, conformance suite, v0.5.5 fixtures | 20 | 1-5 |
| Tests (`tests/`) | 36 | 1-5 (mutation-tested in pass 2) |
| Docs, README, CHANGELOG, ROADMAP, SECURITY, LICENSE, repro scripts | 17 | 1-5 |

Exclusions: `.varmem/reviews.json` (tool-local data; scanned for secrets
only), the historical review records under `docs/code-review/*.md` (read for
context), and generated `docs/configuration.md` (verified against its
generator with `gen_config_doc.py --check`).

## Baseline (at `629a07a`)

ruff (E9,F63,F7,F82) clean; `pytest` **442 passed, 0 skipped** (live pg17 +
fake embed server); `check_versions.py`, `gen_config_doc.py --check`, repro
H3/M1, `python -m build`, `twine check` all pass.

## Findings

IDs are stable across passes (P2*/P3*/P4*/P5* = pass found; CHAL =
independent challenger).

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| STORE-1 / OPS-1 | **High** | `remap` with identical `--old`/`--new` deleted every row of that theme (exit 0) | Fixed `8e7e988`, refined `b58e2fc` |
| PROV-1 | Medium | `identity_signature()` read the startup config, so governance edits never busted cached gateway agents | Fixed `61d760a` |
| PROV-2 | Medium | Default-profile CLI/cron sessions resolve to theme `hermes`, not `default` | **Accepted** (intentional since 0.3.0; changing it would hide existing rows) -- documented `02cd643`, `ded4f1c` |
| TESTDOC-2 | Medium | No working rc-to-final release step; release-notes command returned nothing | Fixed `06a3b88`, `0fcef79` |
| TESTDOC-7 | Medium | `remap` >10-duplicate `--force` guard untested | Fixed `8e7e988` |
| TESTS2-1 | Medium | Prefetch worker generation guard untested (cross-theme cache leak undetected) | Fixed `61d760a` |
| TESTS2-2 | Medium | `recall_conversation` exclusion on hybrid fallback / empty session scope untested | Fixed `61d760a` |
| TESTS2-3 | Medium | `memory_agent_edges` write never asserted | Fixed `61d760a` |
| P2TEST-1 | Medium | Conformance CI job had no zero-skip gate (an upstream import break would pass green) | Fixed `edb7dab` |
| P3PROV-1 | Medium | Telegram `forum` / `webhook` session keys not bucketed (participant id as theme, readable via `scope='all'`) | Fixed `bd1b5b2` |
| P4PROV-1 | Medium | LINE `room` session keys not bucketed (same class) | Fixed `c1bb9c0` |
| STORE-2 | Low | `replace()` with blank `old_text` overwrote an arbitrary row | Fixed `8e7e988` |
| OPS-2 | Low | Unreadable explicit `--config` silently fell back to the default DSN | Fixed `8e7e988` |
| OPS-3 | Low | `conformance.sh` reused a stale local branch | Fixed `06a3b88` |
| PROV-3 | Low | Backstop double-wrote multimodal user turns | Fixed `61d760a` |
| PROV-4 | Low | `metadata.raw_identity` may hold a raw DM key | **Accepted** (documented; metadata key is compat surface) -- doc `02cd643` |
| PROV-5 | Low | `on_delegation` child identity not normalised (always NULL upstream) | Fixed `61d760a` + doc |
| TESTS2-4..6 | Low | Retry bounds, parent-session gate, skill dedup, cache isolation untested | Fixed `61d760a` |
| TESTS2-7 | Low | Backfill blank filter disagreed with `str.strip()` (NBSP etc.) | Fixed `fa5090f`, `b58e2fc` |
| TESTS2-8 | Low | `_as_bool("")` disabled default-on toggles | Fixed `61d760a` |
| TESTS2-9 | Low | `prefetch_budget` max not enforced (host 8 s cap) | Fixed `61d760a` |
| TESTS2-10 | Low | H3 repro misreported environment errors as "bug present" | Fixed `02cd643` |
| TESTS2-11 | Low | Slow bogus-DSN tests, coarse-clock assumption, stale comments, PLAN banner | Fixed `61d760a`, `02cd643` |
| TESTDOC-1,3,4,6 | Low | DSN quoting, notes extraction, ROADMAP/upgrading drift, README path | Fixed `06a3b88` |
| TESTDOC-5 | Low | LICENSE vs README copyright holder | Fixed `0fcef79` (owner: Green Yoga Inc) |
| TESTDOC-8..12 | Low | Coverage and test-isolation gaps | Fixed `8e7e988` |
| TESTDOC-11 | Low | Live CI job did not enforce 0 skipped | Fixed `06a3b88` |
| P2PROV-1 | Low | Backstop stored `/skill` scaffolding | Fixed `65b987c` |
| P2STORE-1 | Low | SQL_ASCII non-UTF-8 rows mislabeled `skipped_changed` | Fixed `b58e2fc` |
| P2STORE-2 | Low | NaN/Inf vectors lost durable rows | Fixed `b58e2fc` |
| P2STORE-3A/B | Low | Over-broad remap guard (pass-1 regression); replace onto existing content kept stale row | Fixed `b58e2fc` |
| P2OPS-1 | Low | Blank predicate locale-dependent (`LC_CTYPE=C`) | Fixed `b58e2fc` |
| P2OPS-2 | Low | "No behaviour change / re-tag" claims now false | Fixed `0fcef79` |
| P2OPS-3,4,6,7,8 | Low | Rehearsal state dir, allow-list note, M1/H1 exit codes, backups discovered as providers, test env docs | Fixed `ded4f1c` |
| P2OPS-5 | Low | `prefetch_limit` / `min_similarity` bounds not enforced | Fixed `65b987c` |
| P2TEST-2..4 | Low | Stale wording, vacuous test cases, over-eager collect hook | Fixed `edb7dab` |
| P3REST-1..3 | Low | Missing report, backup-name mismatch, stale docstring | Fixed (this report), `4e17007` |
| CHAL-1 | Low | Oversized-int vector raised OverflowError, skipping text-only fallback | Fixed `4e17007` |
| P3PROV-2 | Low | Blank `embed_url` dropped durable writes | Fixed `bd1b5b2` |
| P4PROV-2 | Low | Embed-bug warning silenced by an earlier transient error | Fixed `c1bb9c0` |
| P4 (store) | Low | Finite values beyond float4 lost durable rows | Fixed `c1bb9c0` |
| P4REST-1,2 | Low | Dimension-probe / exit-code docs; migration 005 grant claim | Fixed `c1bb9c0` |
| P5PROV-1 | Low | Bare host DM key `agent:<ns>:<platform>:dm` not DM-bucketed | Fixed `cda0554` |
| P5REST-1 | Low | Signature tests read a real host config.yaml | Fixed `cda0554` |
| P5REST-2,3 | Low | Dead doc reference in `embed_dim` description; `stats` count wording | Fixed `cda0554` |
| P5REST-4 | Low | Draft-report counts | Fixed (this report) |

Also accepted by design: the `on_session_end` backstop persists intermediate
assistant narration that `sync_turn` never sees; on non-UTF8 databases,
blanks made of non-ASCII whitespace stay in backfill `remaining` (documented
in `store.py`).

## Convergence history

| Pass | Reviewers | New actionable findings | Outcome |
|---|---|---|---|
| 1 | 5 partitions + skeptic | 1 High, 7 Medium, 24 Low | fixed / 2 accepted |
| 2 | 4 partitions (fix verification, mutation testing) | 1 Medium, 16 Low (one a pass-1 regression) | fixed |
| 3 | 3 partitions + independent challenger (SHIP) | 1 Medium, 5 Low | fixed |
| 4 | 3 partitions incl. challenger of the pass-3 delta | 1 Medium, 4 Low | fixed |
| 5 | 3 partitions incl. full chat-type enumeration + wheel-level operator smoke; challenger verdict SHIP | 0 High/Medium, 5 Low | fixed, each verified by a targeted test |

The loop stopped at the five-pass limit. Pass 5 found no High or Medium
issue; its Low findings were fixed and verified with targeted tests (each
failing on the old code) and the full native suite, not by a sixth full
review pass.

## Validation

| Command | Result at `cda0554` |
|---|---|
| `ruff check --select E9,F63,F7,F82 .` | pass |
| `pytest -q` (live pg17 + fake embed, `PG_TEST_ADMIN_DSN` set) | 528 passed, 0 skipped |
| `pytest tests` with a fake host `hermes_constants` + config.yaml, no DB | 414 passed, 114 skipped (DB-gated), 0 failed |
| `scripts/check_versions.py` | version OK: 1.0.0 |
| `scripts/gen_config_doc.py --check` | current |
| `repro_h3_backfill.py`, `repro_m1_unique_limit.py` | pass |
| `scripts/conformance.sh` (upstream @ `d0288be5`) | 11 passed, 0 skipped (at `61d760a`) |
| `python -m build` + `twine check` | pass (`hermes_memory_pgvector-1.0.0`) |
| Wheel operator smoke (`--version`, `migrate` x2, `stats`, `backfill`, `remap`, `prune`, `cleanup`, `install`) | all expected exit codes (at `c1bb9c0`) |
| GitHub Actions (full matrix) | green at `61d760a`; the result for the final SHA is on the merge proposal -- a committed report cannot record the CI result of its own commit |

## Commits

`06a3b88` docs/CI fixes - `8e7e988` remap data loss, blank replace, --config -
`02cd643` identity/PII docs, H3 repro - `fa5090f` backfill blank filter -
`61d760a` live identity_signature, multimodal dedup, coercion - `b58e2fc`
NaN vectors, replace collisions, locale-free filter - `ded4f1c` hidden install
backups, docs - `65b987c` /skill backstop, prefetch clamps - `edb7dab`
conformance zero-skip gate - `0fcef79` release: v1.0.0 - `4e17007` embed
overflow, doc nits - `bd1b5b2` forum/webhook bucketing, blank embed_url -
`c1bb9c0` LINE room, visible embed bugs - `cda0554` bare DM key, test
isolation - plus this report.

## Rollback

Every fix is an ordinary commit; revert individual commits or the merge
commit. There are no schema changes and no new migrations relative to 0.6.0,
so a package downgrade to 0.6.0 needs no database action.

## Residual risks

- Content containing a NUL byte cannot be stored in Postgres `text`; such a
  write is logged and dropped (pre-existing).
- Theme-filtered HNSW recall at very large table sizes (L7 from the
  2026-09-26 review) remains a scaling consideration; see `docs/scaling.md`.
- The relay transport passes a remote `chat_type` through unchecked; an
  arbitrary value from a relay peer would not be bucketed (host-controlled).
- 1.0.0 ships fixes that did not soak as a release candidate (owner
  decision). Upgrade the live hermes host first and watch writer warnings
  and backfill exit codes.
