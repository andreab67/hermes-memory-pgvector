"""Invariant #4 regressions found by the Wave 2 hook fuzz (v0.6.0).

initialize() raised AttributeError from normalize_identity() when a host
passed a non-string identity kwarg, and on_memory_write() raised
TypeError/ValueError from dict(metadata) on a non-dict metadata. Both hooks
must degrade instead. No DB: MemoryStore is replaced with an unhealthy fake.
"""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# Red-team (G1.7): explicit `scope` names are case-folded like stored
# identities (M7), so a mixed-case theme name still finds the theme and a
# mixed-case sink name still hits the restricted-sink rejection.
# ---------------------------------------------------------------------------


class _RecordingStore:
    def __init__(self):
        self.calls = []

    def _record(self, **kwargs):
        self.calls.append(kwargs)
        return []

    def hybrid_search(self, **kwargs):
        return self._record(**kwargs)

    def hybrid_search_turns(self, **kwargs):
        return self._record(**kwargs)

    def search(self, **kwargs):
        return self._record(**kwargs)

    def search_turns(self, **kwargs):
        return self._record(**kwargs)


def _recall_provider(monkeypatch, identity="sales"):
    monkeypatch.setattr(pkg, "_embed_with_config", lambda *a, **k: [0.0] * 768)
    p = PgvectorMemoryProvider(config={})
    p._healthy = True
    p._agent_identity = identity
    p._store = _RecordingStore()
    return p


@pytest.mark.parametrize("tool", ["recall_memory", "recall_conversation"])
@pytest.mark.parametrize("scope", ["Marketing", "MARKETING", " marketing "])
def test_explicit_theme_scope_is_case_folded(monkeypatch, tool, scope):
    p = _recall_provider(monkeypatch)
    out = json.loads(p.handle_tool_call(tool, {"query": "campaign budget", "scope": scope}))
    assert "error" not in out
    assert p._store.calls[-1]["agent_identity"] == "marketing"


@pytest.mark.parametrize("tool", ["recall_memory", "recall_conversation"])
@pytest.mark.parametrize("scope", ["WhatsApp-DM", "WHATSAPP-DM", "External-Group", "_BENCH"])
def test_mixed_case_restricted_sink_is_rejected(monkeypatch, tool, scope):
    p = _recall_provider(monkeypatch)
    out = json.loads(p.handle_tool_call(tool, {"query": "anything", "scope": scope}))
    assert "restricted sink" in out.get("error", "")
    assert p._store.calls == []


@pytest.mark.parametrize("tool", ["recall_memory", "recall_conversation"])
def test_mixed_case_all_scope_still_excludes_sinks(monkeypatch, tool):
    p = _recall_provider(monkeypatch)
    p.handle_tool_call(tool, {"query": "anything", "scope": "ALL"})
    call = p._store.calls[-1]
    assert call["agent_identity"] is None
    assert set(call["exclude_identities"]) == {"whatsapp-dm", "external-group", "_bench"}
