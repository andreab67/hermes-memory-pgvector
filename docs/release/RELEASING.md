# Releasing

Release checklist for maintainers. Publishing is automated: pushing a
`vX.Y.Z` tag runs `.github/workflows/release.yml`, which verifies the tag
against `pyproject.toml`, runs the complete reusable CI suite (including
zero-skip live PostgreSQL tests, 99.5% branch-aware coverage, lint and upstream
conformance), builds, runs `twine check` and `scripts/check_dist.py`,
publishes to PyPI via Trusted Publishing (OIDC, no stored token) and creates
the GitHub Release with the artifacts attached (notes = the CHANGELOG
section). Steps 1-6 are still done by hand before tagging; steps 7 and 9
are what the workflow automates (the manual commands remain as a fallback).
`workflow_dispatch` runs the checks and build only, never publishing.
Every external action is pinned by commit and Dependabot proposes updates.

The build job generates GitHub build provenance for both distributions.
The separate publish job has `id-token: write`, no repository checkout or
stored PyPI password, and explicitly enables PEP 740 publish attestations.
The exact archives built and verified are passed between jobs as one artifact.

One-time setup (owner): add a Trusted Publisher at
<https://pypi.org/manage/project/hermes-memory-pgvector/settings/publishing/>
with Owner `andreab67`, Repository `hermes-memory-pgvector`, Workflow
`release.yml`, Environment `pypi`; and create a GitHub Environment named
`pypi` (optionally with required reviewers) in the repo settings.

The workflow filename and environment name must match the Trusted Publisher
exactly. An `invalid-publisher` failure means no configured publisher matched
those claims; fix the publisher, then rerun the failed job. Do not substitute a
token upload: that would lose the publish provenance this workflow guarantees.

PyPI attaches attestations at upload time. Existing files cannot gain
attestations retroactively and deleting them does not free their filenames.
Publish a new version instead; leave older versions available to consumers.
See [PyPI's upload policy](https://docs.pypi.org/publishing/troubleshooting/)
and [attestation documentation](https://docs.pypi.org/attestations/producing-attestations/).

## 1. Version bump

- [ ] `pyproject.toml`: `[project].version`
- [ ] `hermes_pgvector/plugin.yaml`: `version:`
- [ ] README.md and ROADMAP.md: `Current version` marker
- [ ] Both must match exactly. Verify: `python scripts/check_versions.py`
      (also checks README/ROADMAP version markers and dependency pins across `pyproject.toml`,
      `plugin.yaml`, README's "Option 3: manual" block, and
      `scripts/install.sh` are all in sync).

## 2. CHANGELOG

- [ ] `CHANGELOG.md`: change `## [X.Y.Z] - Unreleased` to
      `## [X.Y.Z] - YYYY-MM-DD` (today's date, in the release commit — not
      before).
- [ ] **Promoting a release candidate to final** (e.g. `1.0.0rc1` ->
      `1.0.0`): bump `pyproject.toml` and `plugin.yaml` to the final
      version (step 1), then add or rename the CHANGELOG heading to
      `## [1.0.0] - YYYY-MM-DD` and keep a note about the rc under it
      (e.g. "promoted from 1.0.0rc1, no behaviour change" -- only if that
      is literally true; list any post-rc fix like any other change).
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
python scripts/check_dist.py
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

## 6. TestPyPI rehearsal

Use TestPyPI before changing the build backend, package discovery, artifact
contents, or publishing configuration. For a patch that only changes the
version or dependency bounds, the build, metadata validation, clean-wheel
installation and fresh-database smoke tests in steps 4-5 are sufficient;
TestPyPI is optional. The tag-triggered release workflow still runs its
tests and artifact checks before publishing.

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
git push origin vX.Y.Z              # triggers release.yml -> PyPI + GitHub Release
```

- [ ] Tag is signed (`-s`) and pushed.
- [ ] The `Release` workflow run is green (build, publish-pypi,
      github-release).
- [ ] Both distributions show provenance on PyPI. Do not bypass the workflow
      with a token upload if publication fails.

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

Tagging alone does **not** create a GitHub Release — that is a separate
object with its own release notes, and PyPI's project page links to it. The
`github-release` job of `release.yml` creates it from the pushed tag. If you
have to do it by hand, create it (or `gh release create vX.Y.Z --notes-file <(awk -v v='X.Y.Z' 'BEGIN{h="## [" v "] "} index($0,h)==1{f=1;next} /^## \[/{f=0} f' CHANGELOG.md)`,
which prints only that version's section body), pointing at the pushed tag.

- [ ] GitHub Release created, notes match the CHANGELOG entry.
- [ ] Download the two published files and verify their build provenance:

```bash
gh release download vX.Y.Z --repo andreab67/hermes-memory-pgvector --dir /tmp/hpg-verify
gh attestation verify /tmp/hpg-verify/hermes_memory_pgvector-X.Y.Z.tar.gz --repo andreab67/hermes-memory-pgvector
gh attestation verify /tmp/hpg-verify/hermes_memory_pgvector-X.Y.Z-py3-none-any.whl --repo andreab67/hermes-memory-pgvector
```

- [ ] On PyPI, open each file's details and check the attestation names
      `andreab67/hermes-memory-pgvector`, `release.yml` and the released commit.
      The [Integrity API](https://docs.pypi.org/api/integrity/) exposes the
      same provenance at `/integrity/hermes-memory-pgvector/X.Y.Z/FILENAME/provenance`.

## 10. Downstream bump (reminder only)

If you maintain a downstream deployment or announcement channel for this
plugin (a fleet's deploy runbook, an internal changelog page, a status
post), bump/announce it now. This step intentionally does not name any
specific private repo or site here — the release checklist for *this*
package ends at step 9; anything downstream is tracked wherever your fleet
tracks its own deploy state.
