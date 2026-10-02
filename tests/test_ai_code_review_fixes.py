"""Fixes from the ai-code-review checkpoint review (file review, 2026-09-29).

Every test is DB-free: stub stores/writers plus the in-process fake embed
server from conftest.py.

Two groups:

* PIN tests describe behaviour that must NOT change. They cover the memory
  write path (sync_turn, on_session_end, initialize's writer queue) and the
  recall tools' limit handling, and they pass both before and after the fixes.
* FIX tests fail on the code as reviewed and pass after the fix:
  - int() coercions that caught (TypeError, ValueError) but not OverflowError,
    so a float infinity (JSON "Infinity" in tool args, YAML ".inf" in config)
    raised into the agent loop (recall tools, sync_turn, initialize, _as_int);
  - save_config() silently saving nothing when config.yaml has an empty
    `plugins:` key;
  - _safe_err() leaving part of a quoted, whitespace-containing credential.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hermes_pgvector as pkg  # noqa: E402
from hermes_pgvector import (  # noqa: E402
    PgvectorMemoryProvider,
    _as_int,
    _embed_dim,
    _safe_err,
)

INF = float("inf")

# 20 chars: below the default 40-char floor. 50 chars: above it.
_SHORT_USER = "u" * 20
_LONG_ASSISTANT = "a" * 50
_DEFAULT_MIN_CHARS = 40
_DEFAULT_QUEUE_MAX = 256
_DEFAULT_LIMIT = 5


class _CapturingWriter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def enqueue(self, **kwargs: Any) -> bool:
        self.calls.append(kwargs)
        return True

    def turn_roles(self) -> list[str]:
        return [c["extra"]["role"] for c in self.calls if c.get("action") == "turn"]


class _RecordingStore:
    """Read-side store stub: records the kwargs each search method receives."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _record(self, name: str, kwargs: dict[str, Any]) -> list:
        self.calls.append({"method": name, **kwargs})
        return []

    def hybrid_search(self, **kw: Any) -> list:
        return self._record("hybrid_search", kw)

    def hybrid_search_turns(self, **kw: Any) -> list:
        return self._record("hybrid_search_turns", kw)

    def search(self, **kw: Any) -> list:
        return self._record("search", kw)

    def search_turns(self, **kw: Any) -> list:
        return self._record("search_turns", kw)


def _writing_provider(config: dict[str, Any], writer: _CapturingWriter) -> PgvectorMemoryProvider:
    p = PgvectorMemoryProvider(config)
    p._healthy = True
    p._writer = writer
    p._agent_identity = "default"
    p._raw_identity = "default"
    p._session_id = "sess-acr"
    return p


def _reading_provider(embed_url: str, **config: Any) -> tuple:
    store = _RecordingStore()
    p = PgvectorMemoryProvider({"embed_url": embed_url, **config})
    p._healthy = True
    p._store = store
    p._agent_identity = "default"
    p._session_id = "sess-acr"
    return p, store


class _InitStore:
    """MemoryStore stand-in that lets initialize() run without a database."""

    SchemaNotApplied = type("SchemaNotApplied", (RuntimeError,), {})

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def ensure_schema(self) -> None:
        return None

    def health(self) -> dict[str, Any]:
        return {"ok": True}

    def ensure_migration_002_applied(self) -> bool:
        return False

    def count(self, **kw: Any) -> int:
        return 0

    def estimate_count(self, table: str) -> None:
        return None

    def close(self) -> None:
        return None


def _initialized_provider(monkeypatch, tmp_path, **config: Any) -> PgvectorMemoryProvider:
    monkeypatch.setattr(pkg, "MemoryStore", _InitStore)
    p = PgvectorMemoryProvider({"bulk_sync_on_init": False, **config})
    p.initialize("sess-acr", agent_identity="acr-test", hermes_home=str(tmp_path))
    return p


# ---------------------------------------------------------------------------
# PIN: memory write path -- turn_min_chars gates what sync_turn / on_session_end
# hand to the writer.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "min_chars, expected_roles",
    [
        (None, ["assistant"]),      # unset -> default floor (40)
        (40, ["assistant"]),
        (10, ["user", "assistant"]),
        ("10", ["user", "assistant"]),   # save_config() persists strings
        ("not-a-number", ["assistant"]),  # malformed -> default floor
        (0, ["user", "assistant"]),
    ],
)
def test_sync_turn_turn_min_chars_gates_writes(min_chars, expected_roles):
    writer = _CapturingWriter()
    config = {} if min_chars is None else {"turn_min_chars": min_chars}
    p = _writing_provider(config, writer)
    p.sync_turn(_SHORT_USER, _LONG_ASSISTANT)
    assert writer.turn_roles() == expected_roles


@pytest.mark.parametrize(
    "min_chars, expected_roles",
    [
        (None, ["assistant"]),
        (10, ["user", "assistant"]),
        ("10", ["user", "assistant"]),
        ("not-a-number", ["assistant"]),
    ],
)
def test_on_session_end_turn_min_chars_gates_writes(min_chars, expected_roles):
    writer = _CapturingWriter()
    config = {} if min_chars is None else {"turn_min_chars": min_chars}
    p = _writing_provider(config, writer)
    p.on_session_end([
        {"role": "user", "content": _SHORT_USER},
        {"role": "assistant", "content": _LONG_ASSISTANT},
    ])
    assert writer.turn_roles() == expected_roles


# ---------------------------------------------------------------------------
# PIN: memory write path -- write_queue_maxsize sizes the writer queue.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "configured, expected",
    [
        (3, 3),
        ("3", 3),
        ("bogus", _DEFAULT_QUEUE_MAX),
        (None, _DEFAULT_QUEUE_MAX),
    ],
)
def test_initialize_write_queue_maxsize(monkeypatch, tmp_path, configured, expected):
    p = _initialized_provider(monkeypatch, tmp_path, write_queue_maxsize=configured)
    try:
        assert p._writer.stats()["queue_max"] == expected
    finally:
        p.shutdown()


# ---------------------------------------------------------------------------
# PIN: recall tools -- limit is parsed, clamped to 1..20, defaults to 5.
# ---------------------------------------------------------------------------

_TOOLS = ["recall_memory", "recall_conversation"]


@pytest.mark.parametrize("tool", _TOOLS)
@pytest.mark.parametrize(
    "raw_limit, expected",
    [(7, 7), ("7", 7), (99, 20), (0, 1), (-3, 1), ("abc", _DEFAULT_LIMIT), (None, _DEFAULT_LIMIT)],
)
def test_recall_tool_limit_parsing(fake_embed_server, tool, raw_limit, expected):
    p, store = _reading_provider(fake_embed_server())
    out = json.loads(p.handle_tool_call(tool, {"query": "hello", "limit": raw_limit}))
    assert out["count"] == 0 and "error" not in out
    assert store.calls[-1]["limit"] == expected


# ---------------------------------------------------------------------------
# FIX: OverflowError from int() on a float infinity must be handled like the
# other unparseable values (fall back to the default), never raised.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [INF, -INF])
def test_as_int_infinity_falls_back_to_default(bad):
    assert _as_int(bad, 5) == 5


def test_embed_dim_infinity_falls_back_to_default():
    assert _embed_dim({"embed_dim": INF}) == pkg.DEFAULTS["embed_dim"]


@pytest.mark.parametrize("tool", _TOOLS)
def test_recall_tool_infinite_limit_does_not_raise(fake_embed_server, tool):
    # json.loads accepts the bare token Infinity, so a model can emit it.
    args = json.loads('{"query": "hello", "limit": Infinity}')
    p, store = _reading_provider(fake_embed_server())
    out = json.loads(p.handle_tool_call(tool, args))
    assert out["count"] == 0 and "error" not in out
    assert store.calls[-1]["limit"] == _DEFAULT_LIMIT


def test_recall_memory_infinite_embed_dim_does_not_raise(fake_embed_server):
    # embed_dim: .inf in config.yaml reached _as_int() inside
    # _embed_with_config(), which handle_tool_call does not guard.
    p, store = _reading_provider(fake_embed_server(), embed_dim=INF)
    out = json.loads(p.handle_tool_call("recall_memory", {"query": "hello"}))
    assert out["count"] == 0 and "error" not in out
    assert store.calls[-1]["method"] == "hybrid_search"


def test_sync_turn_infinite_turn_min_chars_uses_default_floor():
    writer = _CapturingWriter()
    p = _writing_provider({"turn_min_chars": INF}, writer)
    p.sync_turn(_SHORT_USER, _LONG_ASSISTANT)  # must not raise into the agent loop
    assert writer.turn_roles() == ["assistant"]


def test_on_session_end_infinite_turn_min_chars_uses_default_floor():
    writer = _CapturingWriter()
    p = _writing_provider({"turn_min_chars": INF}, writer)
    p.on_session_end([
        {"role": "user", "content": _SHORT_USER},
        {"role": "assistant", "content": _LONG_ASSISTANT},
    ])
    assert writer.turn_roles() == ["assistant"]


def test_initialize_infinite_write_queue_maxsize_uses_default(monkeypatch, tmp_path):
    p = _initialized_provider(monkeypatch, tmp_path, write_queue_maxsize=INF)
    try:
        assert p._writer.stats()["queue_max"] == _DEFAULT_QUEUE_MAX
    finally:
        p.shutdown()


# ---------------------------------------------------------------------------
# FIX: save_config() with an empty `plugins:` key (YAML null).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "existing_yaml",
    [
        "model: keep-me\nplugins:\n",                  # plugins is null
        "model: keep-me\nplugins:\n  pgvector:\n",     # plugins.pgvector is null
        "model: keep-me\n",                            # plugins absent
    ],
)
def test_save_config_handles_null_and_missing_plugins(tmp_path, existing_yaml):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(existing_yaml, encoding="utf-8")
    PgvectorMemoryProvider().save_config({"dsn": "dbname=new"}, str(tmp_path))
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["plugins"]["pgvector"]["dsn"] == "dbname=new"
    assert saved["model"] == "keep-me"


# ---------------------------------------------------------------------------
# FIX: _safe_err() redaction of quoted credential values.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, leaked, kept",
    [
        ("dsn password=hunter2 host=x", "hunter2", "host=x"),            # unquoted (already worked)
        ("dsn password='ab cd' host=x", "cd", "host=x"),                 # libpq single quotes, has a space
        (r"dsn password='it\'s a secret' host=x", "secret", "host=x"),   # escaped quote inside
        ('dsn password="s3 cret" host=x', "cret", "host=x"),             # double-quoted variant
        ("dsn sslpassword='k e y' host=x", "y'", "host=x"),
        ("dsn password='ab cd", "cd", "dsn"),                            # closing quote lost to truncation
        ("dsn password = hunter2 host=x", "hunter2", "host=x"),          # libpq allows spaces around =
    ],
)
def test_safe_err_redacts_whole_credential_value(text, leaked, kept):
    out = _safe_err(Exception(text))
    assert leaked not in out
    assert "password=[redacted]" in out.lower()
    assert kept in out


def test_safe_err_leaves_plain_text_and_truncates():
    assert _safe_err(Exception("connection failed: timeout")) == "connection failed: timeout"
    assert len(_safe_err(Exception("x" * 1000))) == 300
