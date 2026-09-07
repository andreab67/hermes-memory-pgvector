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

from hermes_pgvector.__main__ import SHIM_MARKER, main  # noqa: E402


def test_install_creates_shim(tmp_path):
    assert main(["install", "--hermes-home", str(tmp_path)]) == 0
    init = tmp_path / "plugins" / "pgvector" / "__init__.py"
    text = init.read_text(encoding="utf-8")
    assert SHIM_MARKER in text
    # Discovery heuristic strings the hermes-agent loader scans for.
    assert "MemoryProvider" in text
    assert "register_memory_provider" in text
    # The actual bridge: absolute import of the installed package.
    assert "from hermes_pgvector import PgvectorMemoryProvider, register" in text
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


# ---------------------------------------------------------------------------
# The actual fail-closed fix: test_remove_refuses_foreign_dir (above) only
# exercises a dir that HAS a non-shim __init__.py -- that branch already
# refused correctly pre-fix. The real gap was a dir with NO __init__.py at
# all: `is_marked_shim = init_py.exists() and SHIM_MARKER in ...` must
# short-circuit to False (fail closed) rather than treating "no __init__.py"
# as "nothing to check" and falling through to shutil.rmtree(). These two
# tests pin that: --remove without --force must refuse (and leave the
# directory + its contents untouched); --remove --force must still delete it.
# ---------------------------------------------------------------------------

def test_remove_refuses_dir_with_no_init_py(tmp_path):
    d = tmp_path / "plugins" / "pgvector"
    d.mkdir(parents=True)
    other = d / "notes.txt"
    other.write_text("some unrelated file; no __init__.py present at all\n", encoding="utf-8")
    assert main(["install", "--hermes-home", str(tmp_path), "--remove"]) == 1
    # Fail closed: nothing should have been deleted.
    assert d.exists()
    assert other.exists()


def test_remove_force_deletes_dir_with_no_init_py(tmp_path):
    d = tmp_path / "plugins" / "pgvector"
    d.mkdir(parents=True)
    (d / "notes.txt").write_text("some unrelated file; no __init__.py present at all\n", encoding="utf-8")
    assert main(["install", "--hermes-home", str(tmp_path), "--remove", "--force"]) == 0
    assert not d.exists()
