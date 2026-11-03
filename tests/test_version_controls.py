"""Release checks reject documentation version drift before publication."""

from pathlib import Path
import re

import pytest

from scripts import check_versions


@pytest.fixture
def release_tree(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parent.parent
    for name, relative in {
        "PYPROJECT": "pyproject.toml",
        "PLUGIN_YAML": "hermes_pgvector/plugin.yaml",
        "README": "README.md",
        "INSTALL_SH": "scripts/install.sh",
        "ROADMAP": "ROADMAP.md",
    }.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = (root / relative).read_text(encoding="utf-8")
        if name == "PYPROJECT":
            content = re.sub(r'^version = ".*"', 'version = "1.0.2"', content, flags=re.M)
        elif name == "PLUGIN_YAML":
            content = re.sub(r'^version: .*', 'version: "1.0.2"', content, flags=re.M)
        elif name in ("README", "ROADMAP"):
            content = re.sub(r'^Current version: .*\n', "", content, flags=re.M)
            content += "\nCurrent version: **1.0.2**.\n"
        target.write_text(content, encoding="utf-8")
        monkeypatch.setattr(check_versions, name, target, raising=False)
    return tmp_path


def test_aligned_release_metadata_passes(release_tree):
    assert check_versions.main() == 0


@pytest.mark.parametrize("name", ["README", "ROADMAP"])
@pytest.mark.parametrize("replacement", ["Current version: **1.0.1**.", ""])
def test_release_rejects_stale_or_missing_documentation_version(release_tree, capsys, name, replacement):
    target = getattr(check_versions, name)
    target.write_text(target.read_text(encoding="utf-8").replace("Current version: **1.0.2**.", replacement), encoding="utf-8")
    assert check_versions.main() == 1
    assert target.name in capsys.readouterr().out
