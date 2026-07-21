"""Unit tests for the `hermes-pgvector install` discovery-shim generator.

Pure filesystem — no DB, no embed endpoint, runs everywhere. The shim is the
bridge that makes a pip-installed package visible to hermes-agent's
directory-scan provider discovery, so its exact contents are contract:
the loader greps the first 8 KiB of __init__.py for "MemoryProvider" /
"register_memory_provider", and the absolute import must resolve the real
package.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pgvector.__main__ import SHIM_MARKER, main  # noqa: E402


def test_install_creates_shim(tmp_path):
    assert main(["install", "--hermes-home", str(tmp_path)]) == 0
    init = tmp_path / "plugins" / "pgvector" / "__init__.py"
    text = init.read_text(encoding="utf-8")
    assert SHIM_MARKER in text
    # Discovery heuristic strings the hermes-agent loader scans for.
    assert "MemoryProvider" in text
    assert "register_memory_provider" in text
    # The actual bridge: absolute import of the installed package.
    assert "from pgvector import PgvectorMemoryProvider, register" in text
    # plugin.yaml copied beside the shim for the discovery description.
    assert (tmp_path / "plugins" / "pgvector" / "plugin.yaml").exists()


def test_install_is_idempotent(tmp_path):
    assert main(["install", "--hermes-home", str(tmp_path)]) == 0
    assert main(["install", "--hermes-home", str(tmp_path)]) == 0
    assert SHIM_MARKER in (
        tmp_path / "plugins" / "pgvector" / "__init__.py"
    ).read_text(encoding="utf-8")


def test_install_refuses_foreign_dir(tmp_path):
    d = tmp_path / "plugins" / "pgvector"
    d.mkdir(parents=True)
    (d / "__init__.py").write_text("# hand-written plugin, not a shim\n", encoding="utf-8")
    assert main(["install", "--hermes-home", str(tmp_path)]) == 1
    # untouched
    assert "hand-written" in (d / "__init__.py").read_text(encoding="utf-8")


def test_install_force_moves_foreign_dir_aside(tmp_path):
    d = tmp_path / "plugins" / "pgvector"
    d.mkdir(parents=True)
    (d / "__init__.py").write_text("# hand-written plugin, not a shim\n", encoding="utf-8")
    assert main(["install", "--hermes-home", str(tmp_path), "--force"]) == 0
    baks = list((tmp_path / "plugins").glob("pgvector.bak-*"))
    assert len(baks) == 1
    assert "hand-written" in (baks[0] / "__init__.py").read_text(encoding="utf-8")
    assert SHIM_MARKER in (d / "__init__.py").read_text(encoding="utf-8")


def test_remove_deletes_generated_shim(tmp_path):
    main(["install", "--hermes-home", str(tmp_path)])
    assert main(["install", "--hermes-home", str(tmp_path), "--remove"]) == 0
    assert not (tmp_path / "plugins" / "pgvector").exists()


def test_remove_refuses_foreign_dir(tmp_path):
    d = tmp_path / "plugins" / "pgvector"
    d.mkdir(parents=True)
    (d / "__init__.py").write_text("# hand-written plugin, not a shim\n", encoding="utf-8")
    assert main(["install", "--hermes-home", str(tmp_path), "--remove"]) == 1
    assert d.exists()
