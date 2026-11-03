#!/usr/bin/env python3
"""Verify release filenames and runtime bytes without extracting archives.

Run after a clean build: python scripts/check_dist.py
Only the standard library is required, including Python 3.11's tomllib.
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

PACKAGE = "hermes_pgvector"


def _compare(artifact: Path, contents: dict[str, bytes], expected: dict[str, bytes]) -> None:
    legacy = sorted(name for name in contents if name == "pgvector.py" or name.startswith("pgvector/"))
    if legacy:
        raise ValueError(f"{artifact.name}: leaked legacy pgvector package: {', '.join(legacy)}")
    runtime = {name: data for name, data in contents.items() if name.startswith(f"{PACKAGE}/")}
    missing = sorted(expected.keys() - runtime.keys())
    extra = sorted(runtime.keys() - expected.keys())
    if missing:
        raise ValueError(f"{artifact.name}: missing runtime files: {', '.join(missing)}")
    if extra:
        raise ValueError(f"{artifact.name}: unexpected runtime files: {', '.join(extra)}")
    for name, data in expected.items():
        if runtime[name] != data:
            raise ValueError(f"{artifact.name}: content differs from checkout: {name}")


def check_dist(dist_dir: Path | str | None = None, repo_root: Path | str | None = None) -> None:
    """Raise ValueError when release archives do not match this checkout."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    dist = Path(dist_dir) if dist_dir is not None else root / "dist"
    with (root / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)["project"]
    distribution = re.sub(r"[-_.]+", "_", project["name"])
    stem = f"{distribution}-{project['version']}"
    wheels, sdists = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(f"{dist}: expected exactly one wheel and one sdist; found {len(wheels)} wheel(s), {len(sdists)} sdist(s)")
    wheel, sdist = wheels[0], sdists[0]
    for path, wanted in ((wheel, f"{stem}-py3-none-any.whl"), (sdist, f"{stem}.tar.gz")):
        if path.name != wanted:
            raise ValueError(f"unexpected artifact {path.name}; expected {wanted}")

    package = root / PACKAGE
    sources = sorted(package.rglob("*.py")) + sorted((package / "migrations").glob("*.sql"))
    sources.append(package / "plugin.yaml")
    expected = {path.relative_to(root).as_posix(): path.read_bytes() for path in sources}
    if f"{PACKAGE}/__init__.py" not in expected:
        raise ValueError(f"{package}: checkout has no runtime package")

    with zipfile.ZipFile(wheel) as archive:
        contents = {}
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = member.filename
            if name in contents:
                raise ValueError(f"{wheel.name}: duplicate archive member: {name}")
            if not name.startswith((f"{PACKAGE}/", "pgvector/", f"{stem}.dist-info/")) and name != "pgvector.py":
                raise ValueError(f"{wheel.name}: unexpected runtime file: {name}")
            contents[name] = archive.read(member)
        _compare(wheel, contents, expected)

    with tarfile.open(sdist, "r:gz") as archive:
        contents = {}
        for member in archive:
            if member.isdir():
                continue
            prefix = f"{stem}/"
            if not member.name.startswith(prefix):
                raise ValueError(f"{sdist.name}: unexpected archive root: {member.name}")
            name = member.name[len(prefix):]
            if name in contents:
                raise ValueError(f"{sdist.name}: duplicate archive member: {name}")
            if not member.isfile():
                raise ValueError(f"{sdist.name}: unexpected non-file archive member: {name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"{sdist.name}: unreadable archive member: {name}")
            with source:
                contents[name] = source.read()
        _compare(sdist, contents, expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)
    try:
        check_dist(dist_dir=args.dist_dir, repo_root=args.repo_root)
    except (OSError, ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("release artifacts match checkout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
