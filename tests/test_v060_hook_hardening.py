"""Invariant #4 regressions found by the Wave 2 hook fuzz (v0.6.0).

initialize() raised AttributeError from normalize_identity() when a host
passed a non-string identity kwarg, and on_memory_write() raised
TypeError/ValueError from dict(metadata) on a non-dict metadata. Both hooks
must degrade instead. No DB: MemoryStore is replaced with an unhealthy fake.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hermes_pgvector as pkg  # noqa: E402
from hermes_pgvector import PgvectorMemoryProvider  # noqa: E402


class _UnhealthyStore:
    def __init__(self, dsn):
        self.dsn = dsn

    def ensure_schema(self):
        return None

    def health(self):
        return {"ok": False, "error": "fake", "row_count_estimate": None}

    def close(self):
        return None


class _RecordingWriter:
    def __init__(self):
        self.items = []

    def enqueue(self, **kwargs):
        self.items.append(kwargs)
        return True


@pytest.mark.parametrize("bad", [42, ["marketing"], {"a": 1}, object(), 3.5])
@pytest.mark.parametrize("key", ["agent_identity", "gateway_session_key", "agent_workspace"])
def test_initialize_ignores_non_string_identity_kwargs(monkeypatch, key, bad):
    monkeypatch.setattr(pkg, "MemoryStore", _UnhealthyStore)
    p = PgvectorMemoryProvider(config={"bulk_sync_on_init": False})
    p.initialize("s1", **{key: bad})
    assert p._agent_identity == "default"
    p.shutdown()


def test_initialize_non_string_header_falls_through_to_valid_identity(monkeypatch):
    monkeypatch.setattr(pkg, "MemoryStore", _UnhealthyStore)
    p = PgvectorMemoryProvider(config={"bulk_sync_on_init": False})
    p.initialize("s1", gateway_session_key=["x"], agent_workspace={}, agent_identity="Marketing")
    assert p._agent_identity == "marketing"
    p.shutdown()


@pytest.mark.parametrize("bad", [42, "not-a-dict", [1, 2], object(), ("a", "b")])
def test_on_memory_write_tolerates_non_dict_metadata(bad):
    p = PgvectorMemoryProvider(config={})
    p._healthy = True
    writer = _RecordingWriter()
    p._writer = writer
    p.on_memory_write("add", "memory", "a durable note worth keeping", metadata=bad)
    assert len(writer.items) == 1
    assert isinstance(writer.items[0]["metadata"], dict)
