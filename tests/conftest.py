"""Shared pytest fixtures for the hermes-memory-pgvector test suite.

`fake_embed_server` is a factory fixture: call it to start `fake_embed_server.py`'s
HTTP server in-process, on a random free port, in a daemon thread -- no
subprocess, no script invocation. It returns the server's base URL.

    def test_something(fake_embed_server):
        url = fake_embed_server()                       # dim=768, no delay
        url = fake_embed_server(dim=32)                  # short vectors
        url = fake_embed_server(delay=20)                # timeout tests
        url = fake_embed_server(fail_substring="denied") # 400 tests
        url = fake_embed_server(require_bearer="secret") # 401 tests

Every server started through the fixture in a given test is stopped when
that test ends -- tests never call .stop() themselves.

This file intentionally does NOT touch how the rest of the suite decides to
skip: the PG_TEST_DSN / PG_TEST_EMBED_URL gates in the existing test modules
are untouched, and no autouse fixture is added here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List

import pytest

# tests/ has no __init__.py, so pytest already puts this file's directory on
# sys.path -- but be explicit rather than relying on that, since the exact
# insertion point differs across pytest/rootdir configurations.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_embed_server import FakeEmbedServer  # noqa: E402


@pytest.fixture
def fake_embed_server() -> Callable[..., str]:
    started: List[FakeEmbedServer] = []

    def _make(**kwargs) -> str:
        srv = FakeEmbedServer(host="127.0.0.1", port=0, **kwargs)
        srv.start()
        started.append(srv)
        return srv.base_url

    yield _make

    for srv in started:
        srv.stop()
