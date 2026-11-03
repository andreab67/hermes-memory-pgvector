"""Release archives must contain the current checkout's exact runtime bytes."""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

CHECKER = Path(__file__).resolve().parents[1] / "scripts" / "check_dist.py"
PACKAGE = "hermes_pgvector"
STEM = "hermes_memory_pgvector-1.2.3"
RUNTIME = {
    f"{PACKAGE}/__init__.py": b'"""current provider"""\n',
    f"{PACKAGE}/embed.py": b"EMBED_DIM = 768\n",
    f"{PACKAGE}/migrations/001_schema.sql": b"SELECT 1;\n",
    f"{PACKAGE}/migrations/002_attribution.sql": b"SELECT 2;\n",
    f"{PACKAGE}/plugin.yaml": b"version: 1.2.3\n",
}


def _write_archives(dist, wheel_files, sdist_files):
    wheel = dist / f"{STEM}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, contents in wheel_files.items():
            archive.writestr(name, contents)
        archive.writestr(f"{STEM}.dist-info/METADATA", b"Name: hermes-memory-pgvector\nVersion: 1.2.3\n")
    sdist = dist / f"{STEM}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name, contents in sdist_files.items():
            member = tarfile.TarInfo(f"{STEM}/{name}")
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))
    return wheel, sdist


@pytest.fixture
def release(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "hermes-memory-pgvector"\nversion = "1.2.3"\n', encoding="utf-8",
    )
    for name, contents in RUNTIME.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    dist = tmp_path / "dist"
    dist.mkdir()
    _write_archives(dist, RUNTIME, {**RUNTIME, "README.md": b"release readme\n"})
    return repo, dist


def _checker():
    spec = importlib.util.spec_from_file_location("release_integrity_checker", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_checker_cli_accepts_matching_archives(release):
    repo, dist = release
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--repo-root", str(repo), "--dist-dir", str(dist)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "match checkout" in result.stdout


def test_function_accepts_matching_archives_and_sdist_project_files(release):
    repo, dist = release
    assert _checker().check_dist(dist_dir=dist, repo_root=repo) is None


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("missing", [f"{PACKAGE}/embed.py", f"{PACKAGE}/migrations/002_attribution.sql", f"{PACKAGE}/plugin.yaml"])
def test_rejects_missing_runtime_file_in_either_archive(release, kind, missing):
    repo, dist = release
    incomplete = {name: contents for name, contents in RUNTIME.items() if name != missing}
    _write_archives(dist, incomplete if kind == "wheel" else RUNTIME,
                    incomplete if kind == "sdist" else RUNTIME)
    with pytest.raises(ValueError, match="missing.*" + missing):
        _checker().check_dist(dist_dir=dist, repo_root=repo)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_rejects_content_tamper_in_either_archive(release, kind):
    repo, dist = release
    tampered = {**RUNTIME, f"{PACKAGE}/embed.py": b"stale implementation\n"}
    _write_archives(dist, tampered if kind == "wheel" else RUNTIME,
                    tampered if kind == "sdist" else RUNTIME)
    with pytest.raises(ValueError, match="differs.*hermes_pgvector/embed.py"):
        _checker().check_dist(dist_dir=dist, repo_root=repo)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("extra", ["pgvector/__init__.py", f"{PACKAGE}/old_module.py", f"{PACKAGE}/migrations/999_stale.sql", f"{PACKAGE}/__pycache__/embed.pyc"])
def test_rejects_legacy_and_unexpected_package_files(release, kind, extra):
    repo, dist = release
    contaminated = {**RUNTIME, extra: b"unexpected runtime bytes"}
    _write_archives(dist, contaminated if kind == "wheel" else RUNTIME,
                    contaminated if kind == "sdist" else RUNTIME)
    with pytest.raises(ValueError, match="legacy|unexpected"):
        _checker().check_dist(dist_dir=dist, repo_root=repo)


@pytest.mark.parametrize("suffix", ["-py3-none-any.whl", ".tar.gz"])
def test_rejects_archive_version_mismatch(release, suffix):
    repo, dist = release
    (dist / f"{STEM}{suffix}").rename(dist / f"hermes_memory_pgvector-0.0.1{suffix}")
    with pytest.raises(ValueError, match="expected.*1.2.3"):
        _checker().check_dist(dist_dir=dist, repo_root=repo)


@pytest.mark.parametrize("kind", ["missing", "extra"])
def test_requires_exactly_one_wheel_and_one_sdist(release, kind):
    repo, dist = release
    if kind == "missing":
        (dist / f"{STEM}.tar.gz").unlink()
    else:
        (dist / "old-0.0.1-py3-none-any.whl").write_bytes(b"stale build")
    with pytest.raises(ValueError, match="exactly one wheel and one sdist"):
        _checker().check_dist(dist_dir=dist, repo_root=repo)


def test_cli_reports_invalid_archive_with_exit_one(release):
    repo, dist = release
    (dist / f"{STEM}-py3-none-any.whl").write_bytes(b"not a zip")
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--repo-root", str(repo), "--dist-dir", str(dist)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert "error:" in result.stderr and "Traceback" not in result.stderr
