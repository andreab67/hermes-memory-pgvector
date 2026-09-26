"""conformance/conftest.py — collection guard + shared fixtures for the
upstream conformance suite (docs/release/PLAN-1.0.md Sec 5 WP-F).

This directory is NOT under tests/ on purpose: it needs hermes-agent
installed/importable plus (for a couple of tests) a live Postgres + fake
embed endpoint, none of which the plain root `python -m pytest -q` sets
up. That run must collect exactly what it collected before this directory
existed.

Collection guard
-----------------
There is no `[tool.pytest.ini_options]` / `testpaths` anywhere in this repo
(no pytest.ini, no setup.cfg, no such section in pyproject.toml), so a bare
`pytest -q` invoked from the repo root walks every subdirectory, including
this one, by default. `pytest_ignore_collect` below keeps this directory out
of that walk while still collecting normally for `pytest conformance -q`
(what `scripts/conformance.sh` runs) or any other invocation that names
`conformance` explicitly. Verified empirically (not just reasoned about):
a `pytest_ignore_collect` hook defined in a subdirectory's own conftest.py
DOES gate that same subdirectory's collection when the parent walk reaches
it — pytest imports a directory's local conftest before deciding what to
collect *within* it, ignore-hook included.

Import order matters
---------------------
hermes_pgvector/__init__.py does::

    try:
        from agent.memory_provider import MemoryProvider
        ...
    except ImportError:
        MemoryProvider = object  # stub

That binding is resolved ONCE, the first time `hermes_pgvector` is
imported in this process, and then cached in `sys.modules` for the rest of
the run. If anything here imported `hermes_pgvector` before hermes-agent
was on `sys.path`, `PgvectorMemoryProvider` would be permanently bound to
the `object` stub instead of the real ABC — and every test in this module
would silently pass against the wrong base class instead of catching real
drift. So: every test module in this directory MUST import
`agent.memory_provider` (e.g. via `pytest.importorskip("agent.memory_provider")`)
BEFORE its own `import hermes_pgvector`. This conftest deliberately does
NOT import either at module scope, so it cannot get the order wrong for
whichever test module collects first.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List

import pytest

_CONFORMANCE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CONFORMANCE_DIR.parent


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    """True (ignore) unless `conformance` was explicitly named on the command line.

    `config.invocation_params.args` is the raw argv pytest was invoked with
    (flags and all). A bare `pytest -q` / `pytest` from the repo root has no
    non-flag argument at all -> ignore. `pytest conformance -q`,
    `pytest conformance/test_x.py::test_y`, or an absolute path under this
    directory all contain "conformance" in one of the non-flag args -> collect.
    """
    args = [str(a) for a in config.invocation_params.args]
    for arg in args:
        if arg.startswith("-"):
            continue
        if "conformance" in arg.replace("\\", "/"):
            return False
    return True


def pinned_ref() -> str:
    """The full SHA `scripts/conformance.sh` clones by default."""
    return (_CONFORMANCE_DIR / "HERMES_AGENT_REF").read_text(encoding="utf-8").strip()


@pytest.fixture(scope="session")
def hermes_agent_ref() -> str:
    return pinned_ref()


@pytest.fixture
def pg_test_dsn() -> str:
    """PG_TEST_DSN, or a clean skip -- same gate the tests/ suite uses."""
    import os

    dsn = os.environ.get("PG_TEST_DSN")
    if not dsn:
        pytest.skip("PG_TEST_DSN not set")
    return dsn


@pytest.fixture
def fake_embed_server() -> Callable[..., str]:
    """Factory fixture: call it to start tests/fake_embed_server.py's server
    in-process on a random free port. Reuses the exact same implementation
    the main tests/ suite uses (tests/conftest.py), imported by path since
    tests/ is a sibling directory and not an installed package.

        url = fake_embed_server()             # dim=768, no delay
        url = fake_embed_server(delay=20)      # timeout tests

    Every server started through this fixture in a given test is stopped
    when that test ends.
    """
    tests_dir = str(_REPO_ROOT / "tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from fake_embed_server import FakeEmbedServer  # noqa: E402

    started: List[FakeEmbedServer] = []

    def _make(**kwargs) -> str:
        srv = FakeEmbedServer(host="127.0.0.1", port=0, **kwargs)
        srv.start()
        started.append(srv)
        return srv.base_url

    yield _make

    for srv in started:
        srv.stop()
