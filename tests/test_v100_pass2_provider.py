"""1.0 review pass-2 fixes for hermes_pgvector/__init__.py (provider layer).

Covers: P2PROV-1 (on_session_end backstop must decide noise / dedup / storage
on the SAME normalized string the host hands sync_turn(): a /skill turn is
stripped to the user's instruction, a bare /skill is skipped, and assistant
list content is flattened with the host's helper) and P2OPS-5 (prefetch_limit
and min_similarity are clamped to the bounds the config schema advertises).

DB-free: stub writers/stores and a stub `agent.skill_commands` injected with
monkeypatch.setitem so it is removed after each test.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hermes_pgvector as pkg  # noqa: E402
from hermes_pgvector import (  # noqa: E402
    _MIN_SIMILARITY_MAX,
    _MIN_SIMILARITY_MIN,
    _PREFETCH_LIMIT_MAX,
    _PREFETCH_LIMIT_MIN,
    PgvectorMemoryProvider,
)

_LONG_A = "An assistant reply that is comfortably past the noise floor for capture."
_BODY = "SKILLBODY " * 300
_MARK = "\n\nUSER INSTRUCTION: "


class _Writer:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def enqueue(self, **kwargs) -> bool:
        self.calls.append(kwargs)
        return True

    def turns(self, role: str) -> List[str]:
        return [c["content"] for c in self.calls
                if c.get("action") == "turn" and c["extra"]["role"] == role]


def _provider(writer) -> PgvectorMemoryProvider:
    p = PgvectorMemoryProvider()
    p._healthy = True
    p._writer = writer
    p._agent_identity = "default"
    p._raw_identity = "default"
    p._session_id = "sess-p2"
    p._turn_fingerprints = set()
    return p


def _stub_skill_commands(monkeypatch):
    """Mimic upstream extract_user_instruction_from_skill_message: None for a
    bare invocation, the instruction for a skill turn, the text otherwise."""
    def _extract(content):
        if not content.startswith("[SKILL]"):
            return content
        _, sep, instr = content.partition(_MARK)
        return instr.strip() if sep and instr.strip() else None

    mod = types.ModuleType("agent.skill_commands")
    mod.extract_user_instruction_from_skill_message = _extract
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.skill_commands", mod)


def _skill_msg(instr: str) -> str:
    return "[SKILL] " + _BODY + (_MARK + instr if instr else "")


def test_skill_turn_with_noise_instruction_stores_no_skill_body(monkeypatch):
    _stub_skill_commands(monkeypatch)
    writer = _Writer()
    p = _provider(writer)
    # Host would call sync_turn("ok", ...) -> noise, dropped; then session end.
    p.sync_turn("ok", _LONG_A, session_id="sess-p2")
    p.on_session_end([
        {"role": "user", "content": _skill_msg("ok")},
        {"role": "assistant", "content": _LONG_A},
    ])
    assert not any("SKILLBODY" in t for t in writer.turns("user"))
    assert writer.turns("user") == []
    assert writer.turns("assistant") == [_LONG_A]


def test_bare_skill_invocation_stores_no_user_turn(monkeypatch):
    _stub_skill_commands(monkeypatch)
    writer = _Writer()
    p = _provider(writer)
    # Host skips sync_turn entirely for a bare /skill (strip returns None).
    p.on_session_end([
        {"role": "user", "content": _skill_msg("")},
        {"role": "assistant", "content": _LONG_A},
    ])
    assert writer.turns("user") == []


def test_long_skill_instruction_still_dedups_and_stores_only_the_instruction(monkeypatch):
    _stub_skill_commands(monkeypatch)
    instr = "please summarize the quarterly numbers for the whole team now"
    writer = _Writer()
    p = _provider(writer)
    p.sync_turn(instr, _LONG_A, session_id="sess-p2")
    assert writer.turns("user") == [instr]
    p.on_session_end([
        {"role": "user", "content": _skill_msg(instr)},
        {"role": "assistant", "content": _LONG_A},
    ])
    assert writer.turns("user") == [instr], "the /skill turn was written twice"
    assert writer.turns("assistant") == [_LONG_A]

    # A backstop-only replay (sync_turn never ran) stores the stripped text.
    writer2 = _Writer()
    _provider(writer2).on_session_end([{"role": "user", "content": _skill_msg(instr)}])
    assert writer2.turns("user") == [instr]


def test_assistant_list_content_already_captured_is_not_reenqueued():
    writer = _Writer()
    p = _provider(writer)
    part1 = "First paragraph of the assistant reply, long enough alone."
    part2 = "Second paragraph of the assistant reply, also long enough."
    # The host flattens with newline separators before sync_turn().
    p.sync_turn(_LONG_A, part1 + "\n" + part2, session_id="sess-p2")
    assert len(writer.turns("assistant")) == 1
    p.on_session_end([
        {"role": "user", "content": _LONG_A},
        {"role": "assistant", "content": [
            {"type": "text", "text": part1},
            {"type": "text", "text": part2},
        ]},
    ])
    assert len(writer.turns("assistant")) == 1, "assistant list content hashed differently"
    assert len(writer.turns("user")) == 1


def test_strip_skill_scaffolding_falls_back_to_content_only_without_the_host(monkeypatch):
    monkeypatch.setitem(sys.modules, "agent.skill_commands", None)  # import raises
    assert PgvectorMemoryProvider._strip_skill_scaffolding("plain text") == "plain text"


def test_strip_skill_scaffolding_returns_empty_when_host_helper_returns_none(monkeypatch):
    _stub_skill_commands(monkeypatch)
    assert PgvectorMemoryProvider._strip_skill_scaffolding(_skill_msg("")) == ""


# ---------------------------------------------------------------------------
# P2OPS-5 -- prefetch_limit / min_similarity clamped to the schema bounds
# ---------------------------------------------------------------------------

class _RecordingStore:
    def __init__(self):
        self.kw: List[Dict[str, Any]] = []

    def search(self, **kw):
        self.kw.append(kw)
        return [{"content": "a recalled row", "target": "memory", "score": 0.9}]

    def close(self):
        return None


def _prefetch_kwargs(monkeypatch, config):
    monkeypatch.setattr(pkg, "_embed_text", lambda *a, **k: [0.0] * 8)
    store = _RecordingStore()
    p = PgvectorMemoryProvider(config=config)
    p._healthy = True
    p._store = store
    p._agent_identity = "default"
    p.prefetch("q")
    assert len(store.kw) == 1
    return store.kw[0]


def test_prefetch_limit_and_min_similarity_schema_share_the_clamp_bounds():
    schema = {e["key"]: e for e in PgvectorMemoryProvider().get_config_schema()}
    assert schema["prefetch_limit"]["minimum"] == _PREFETCH_LIMIT_MIN
    assert schema["prefetch_limit"]["maximum"] == _PREFETCH_LIMIT_MAX
    assert schema["min_similarity"]["minimum"] == _MIN_SIMILARITY_MIN
    assert schema["min_similarity"]["maximum"] == _MIN_SIMILARITY_MAX


@pytest.mark.parametrize("raw,expected", [
    (500, _PREFETCH_LIMIT_MAX),
    (-1, _PREFETCH_LIMIT_MIN),
    (0, _PREFETCH_LIMIT_MIN),
    ("500", _PREFETCH_LIMIT_MAX),
    (7, 7),
])
def test_prefetch_limit_is_clamped(monkeypatch, raw, expected):
    kw = _prefetch_kwargs(monkeypatch, {"prefetch_limit": raw})
    assert kw["limit"] == expected


@pytest.mark.parametrize("raw,expected", [
    (2.0, _MIN_SIMILARITY_MAX),
    (-1, _MIN_SIMILARITY_MIN),
    ("2.0", _MIN_SIMILARITY_MAX),
    (0.55, 0.55),
])
def test_min_similarity_is_clamped(monkeypatch, raw, expected):
    kw = _prefetch_kwargs(monkeypatch, {"min_similarity": raw})
    assert kw["min_similarity"] == expected
