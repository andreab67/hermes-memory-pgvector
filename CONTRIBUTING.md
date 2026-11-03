# Contributing

Keep changes focused on making cooperating agents more capable. Explain the
failure or capability being addressed before adding code. Reuse an existing
path before introducing a helper, dependency, abstraction or configuration key.
Do not add speculative functionality, compatibility layers without supported
callers, copied implementations, or comments that merely repeat the code.

## Architecture rules

- Memory hooks and writer drains use embeddings and storage only; never add
  LLM calls, extraction pipelines or new agent write tools.
- Preserve identity priority and restricted-theme read gates. Cross-theme
  recall remains explicit. Never log credentials or raw private session keys.
- Hooks catch, log and degrade on failure. Catch specific exceptions inside
  ordinary helpers; broad catches belong at fail-soft boundaries and need a
  local `BLE001` suppression with a reason. Never catch `BaseException`.
- The agent thread only enqueues writes. Embedding and database I/O stay in
  the drain thread. Cover queue saturation, worker failure and shutdown.
- Runtime schema checks remain read-only. Admin DDL goes in a new numbered
  migration; never edit an already released migration.
- Bound new dependencies above and below. Keep runtime dependencies aligned
  across package metadata, plugin YAML, README and the install script.

The tracked `.claude/rules/python-*.md` files contain additional review
examples for input validation, bounded state, structure and meaningful tests.
Those files are generated copies; update their source before resynchronizing.

## Local checks

```bash
python -m pip install -e ".[test,dev]"
pre-commit install
pre-commit run --all-files
python -m pytest tests/ -q
```

Ruff uses `pyproject.toml` in hooks and CI. Do not override the selected rules
on the command line, disable a rule globally to hide a finding, or weaken
assertions to make a change pass. Register shared pytest fixture modules with
`pytest_plugins` instead of shadowing imported fixture names.

Run the complete database suite against a disposable database:

```bash
scripts/test-env.sh up pg17
eval "$(scripts/test-env.sh db hermes_test)"
python -m pytest tests/ --cov --cov-report=term-missing --cov-report=xml -q
scripts/test-env.sh down
```

CI runs Python 3.11–3.13, PostgreSQL 16–18, package checks and pinned upstream
conformance. Live jobs reject skipped tests and enforce the combined statement
and branch coverage floor in `pyproject.toml`. All runtime modules count. Do
not exclude difficult code, add `pragma: no cover`, mark regressions skipped,
or mock the implementation being tested to raise the percentage. Use real
HTTP/database tests for protocol and SQL behavior, boundary fakes for outages,
and assert observable outcomes. Report residual coverage gaps honestly.

Changes require a regression test that fails before a bug fix, green checks,
and review of the diff against the original requirement. See the
[release checklist](docs/release/RELEASING.md) for publishing with provenance.
