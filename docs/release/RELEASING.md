# Releasing

Manual PyPI release checklist for maintainers. There is no automated
publish step (deliberately — see
[`docs/release/PLAN-1.0.md`](PLAN-1.0.md): "It never publishes anything").

## 1. Version bump

- [ ] `pyproject.toml`: `[project].version`
- [ ] `hermes_pgvector/plugin.yaml`: `version:`
- [ ] Both must match exactly. Verify: `python scripts/check_versions.py`
      (also checks the dependency pins across `pyproject.toml`,
      `plugin.yaml`, README's "Option 3: manual" block, and
      `scripts/install.sh` are all in sync).

## 2. CHANGELOG

- [ ] `CHANGELOG.md`: change `## [X.Y.Z] - Unreleased` to
      `## [X.Y.Z] - YYYY-MM-DD` (today's date, in the release commit — not
      before).
- [ ] Confirm every finding/behaviour change fixed since the last release
      has an entry (Added/Changed/Fixed, plus a **Breaking / upgrade
      notes** block for anything that needs operator action).

## 3. Config docs

- [ ] `python scripts/gen_config_doc.py` (regenerate) then `git diff
      --stat docs/configuration.md` — should be empty if nothing about the
      config schema changed since the last release; if not empty, review
      and commit it.
- [ ] `python scripts/gen_config_doc.py --check` exits 0.

## 4. Build + verify the artifact

```bash
pip install -e ".[dev]"        # build, twine, ruff
rm -rf dist/ build/ *.egg-info
python -m build
twine check dist/*
```

- [ ] `twine check dist/*` passes (valid long-description / metadata).
- [ ] Wheel contains every migration and `plugin.yaml`:

```bash
python -c "
import glob, zipfile
wheel = glob.glob('dist/*.whl')[0]
names = zipfile.ZipFile(wheel).namelist()
assert any(n.startswith('hermes_pgvector/migrations/') and n.endswith('.sql') for n in names)
assert any(n.endswith('hermes_pgvector/plugin.yaml') for n in names)
print('wheel OK:', wheel, sorted(n for n in names if 'migrations/' in n))
"
```

- [ ] Confirm the migrations listed are `001` through the current highest
      number (nothing missing, nothing stale from an old build in `dist/`).

## 5. Clean-venv smoke test

```bash
python -m venv /tmp/hpg-release-smoke
/tmp/hpg-release-smoke/bin/pip install dist/*.whl
/tmp/hpg-release-smoke/bin/hermes-pgvector --version   # prints the version from step 1

# Against a scratch Postgres + pgvector (e.g. `scripts/test-env.sh up pg17 && scripts/test-env.sh db release_smoke`):
/tmp/hpg-release-smoke/bin/hermes-pgvector migrate --admin-dsn "$PG_TEST_ADMIN_DSN"
/tmp/hpg-release-smoke/bin/hermes-pgvector stats --dsn "$PG_TEST_DSN"
```

- [ ] `--version` prints the correct version (proves `importlib.metadata`
      resolves against this exact build, not a stale editable install).
- [ ] `migrate` applies cleanly against a fresh database.
- [ ] `stats` runs without error.

## 6. TestPyPI rehearsal (recommended before the first release of a minor,
      required before any release that touched packaging/`pyproject.toml`)

```bash
twine upload --repository testpypi dist/*
python -m venv /tmp/hpg-testpypi-smoke
/tmp/hpg-testpypi-smoke/bin/pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ hermes-memory-pgvector==X.Y.Z
/tmp/hpg-testpypi-smoke/bin/hermes-pgvector --version
```

- [ ] Installs cleanly from TestPyPI with real dependencies resolving from
      the real PyPI index (`--extra-index-url` above).

## 7. Tag + upload

```bash
git tag -s vX.Y.Z -m "vX.Y.Z"       # signed tag
git push origin vX.Y.Z
twine upload dist/*
```

- [ ] Tag is signed (`-s`) and pushed.
- [ ] `twine upload dist/*` (the real index, not TestPyPI) succeeds.

## 8. Post-release verification (one real host)

```bash
pip install -U hermes-memory-pgvector==X.Y.Z
hermes-pgvector --version
hermes memory status        # expect: Provider: pgvector; Status: available
```

- [ ] Version matches on the installed host.
- [ ] `hermes memory status` reports the provider available after a
      restart.
- [ ] For a release with schema changes: `hermes-pgvector migrate
      --admin-dsn "..." [--runtime-role NAME]`, then `hermes-pgvector
      stats` to confirm.

## 9. GitHub Release object

Tagging does **not** create a GitHub Release — that is a separate object
with its own release notes, and PyPI's project page links to it. Create it
manually (or `gh release create vX.Y.Z --notes-file <(sed -n '/## \[X.Y.Z\]/,/## \[/p' CHANGELOG.md)`
trimmed to just that section), pointing at the pushed tag.

- [ ] GitHub Release created, notes match the CHANGELOG entry.

## 10. Downstream bump (reminder only)

If you maintain a downstream deployment or announcement channel for this
plugin (a fleet's deploy runbook, an internal changelog page, a status
post), bump/announce it now. This step intentionally does not name any
specific private repo or site here — the release checklist for *this*
package ends at step 9; anything downstream is tracked wherever your fleet
tracks its own deploy state.
