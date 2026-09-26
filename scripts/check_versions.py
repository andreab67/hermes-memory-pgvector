#!/usr/bin/env python3
"""check_versions.py — the four places a version/dependency pin can drift.

Exits 0 iff:

1. `pyproject.toml`'s `[project].version` equals `hermes_pgvector/plugin.yaml`'s
   `version:`.
2. The runtime dependency pins in `pyproject.toml`'s `[project].dependencies`
   match, as a SET (order-independent), the pins in:
     - `plugin.yaml`'s `pip_dependencies:`
     - README.md's "Option 3: manual" install block
     - `scripts/install.sh`'s `pip install` block

On any mismatch, prints a clear diff (version strings, or the requirement
strings present on one side and missing on the other) and exits 1.

stdlib only (tomllib, re) -- this has to run before anything is installed, in
a bare checkout, so it does not import PyYAML even though the runtime
dependency list makes it available in an installed environment.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Dict, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
PLUGIN_YAML = REPO_ROOT / "hermes_pgvector" / "plugin.yaml"
README = REPO_ROOT / "README.md"
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

# name[extras] followed by a version specifier, e.g. psycopg[binary]>=3.3.6,<4
_REQ_RE = re.compile(r"^([A-Za-z0-9_.-]+(?:\[[^\]]*\])?)\s*(.*)$")


def normalize_requirement(req: str) -> Tuple[str, str]:
    """('psycopg[binary]>=3.3.6,<4') -> ('psycopg[binary]', '>=3.3.6,<4')

    Name is lowercased (PyYAML == pyyaml); the specifier is compared verbatim
    (with whitespace stripped) because the whole point of this check is that
    the BOUNDS must match exactly across every file.
    """
    req = req.strip()
    m = _REQ_RE.match(req)
    if not m:
        return (req.lower(), "")
    name, specifier = m.group(1), m.group(2)
    # Lowercase the base name but keep an extras bracket's contents as-is.
    if "[" in name:
        base, _, rest = name.partition("[")
        name = base.lower() + "[" + rest
    else:
        name = name.lower()
    return (name, specifier.replace(" ", ""))


def as_req_set(reqs) -> Set[Tuple[str, str]]:
    return {normalize_requirement(r) for r in reqs if r.strip()}


def load_pyproject() -> Tuple[str, list]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    return project["version"], list(project["dependencies"])


def load_plugin_yaml() -> Tuple[str, list]:
    text = PLUGIN_YAML.read_text(encoding="utf-8")
    version_match = re.search(r'^version:\s*"?([^"\n]+)"?\s*$', text, re.MULTILINE)
    if not version_match:
        raise ValueError(f"could not find 'version:' in {PLUGIN_YAML}")
    version = version_match.group(1).strip()

    deps_match = re.search(r"^pip_dependencies:\s*\n((?:[ \t]+-.*\n?)*)", text, re.MULTILINE)
    deps = []
    if deps_match:
        for line in deps_match.group(1).splitlines():
            item = line.strip()
            if item.startswith("- "):
                item = item[2:].strip()
            deps.append(item.strip("\"'"))
    return version, deps


def load_readme_option3_deps() -> list:
    text = README.read_text(encoding="utf-8")
    section_match = re.search(r"### Option 3: manual.*?```bash\n(.*?)```", text, re.DOTALL)
    if not section_match:
        raise ValueError(f"could not find the 'Option 3: manual' code block in {README}")
    block = section_match.group(1)
    pip_line_match = re.search(r"^pip install (.+)$", block, re.MULTILINE)
    if not pip_line_match:
        raise ValueError("could not find a 'pip install ...' line in the Option 3 block")
    return re.findall(r"'([^']+)'", pip_line_match.group(1))


def load_install_sh_deps() -> list:
    text = INSTALL_SH.read_text(encoding="utf-8")
    block_match = re.search(
        r'"\$PIP"\s+install\s*\\\n(.*?)(?=\n\S|\n#|\Z)', text, re.DOTALL,
    )
    if not block_match:
        raise ValueError(f"could not find the '\"$PIP\" install' block in {INSTALL_SH}")
    return re.findall(r"'([^']+)'", block_match.group(1))


def diff_sets(label: str, a: Set[Tuple[str, str]], b: Set[Tuple[str, str]], a_label: str, b_label: str) -> bool:
    """Prints a diff and returns True iff a == b."""
    if a == b:
        return True
    only_a = a - b
    only_b = b - a
    print(f"MISMATCH: {label}")
    if only_a:
        print(f"  only in {a_label}: {sorted(only_a)}")
    if only_b:
        print(f"  only in {b_label}: {sorted(only_b)}")
    return False


def main() -> int:
    ok = True

    try:
        pyproject_version, pyproject_deps = load_pyproject()
    except Exception as exc:  # noqa: BLE001
        print(f"error reading {PYPROJECT}: {exc}", file=sys.stderr)
        return 1

    try:
        plugin_version, plugin_deps = load_plugin_yaml()
    except Exception as exc:  # noqa: BLE001
        print(f"error reading {PLUGIN_YAML}: {exc}", file=sys.stderr)
        return 1

    if pyproject_version != plugin_version:
        print(
            f"MISMATCH: version -- pyproject.toml={pyproject_version!r} "
            f"plugin.yaml={plugin_version!r}"
        )
        ok = False
    else:
        print(f"version OK: {pyproject_version}")

    pyproject_set = as_req_set(pyproject_deps)

    sources: Dict[str, list] = {"plugin.yaml": plugin_deps}
    try:
        sources["README.md (Option 3: manual)"] = load_readme_option3_deps()
    except Exception as exc:  # noqa: BLE001
        print(f"error reading {README}: {exc}", file=sys.stderr)
        return 1
    try:
        sources["scripts/install.sh"] = load_install_sh_deps()
    except Exception as exc:  # noqa: BLE001
        print(f"error reading {INSTALL_SH}: {exc}", file=sys.stderr)
        return 1

    for label, deps in sources.items():
        other_set = as_req_set(deps)
        if diff_sets(f"dependencies (pyproject.toml vs {label})", pyproject_set, other_set,
                      "pyproject.toml", label):
            print(f"dependencies OK: pyproject.toml == {label}")
        else:
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
