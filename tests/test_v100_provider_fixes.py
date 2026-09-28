"""1.0 review fixes for hermes_pgvector/__init__.py (provider layer).

Covers: PROV-1 (identity_signature reads live config), PROV-3 (on_session_end
flattens multimodal user content like the host), PROV-5 (on_delegation
normalizes a supplied child identity), TESTS2-1 (prefetch generation guard
through the worker path), TESTS2-5, TESTS2-6, TESTS2-9 (prefetch_budget clamp).

DB-free: stub stores/writers, a fake hermes_constants module, and the
in-process fake embed server.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hermes_pgvector as pkg  # noqa: E402
from hermes_pgvector import (  # noqa: E402
    _PREFETCH_BUDGET_MAX,
    _PREFETCH_BUDGET_MIN,
    PgvectorMemoryProvider,
    _flatten_user_content,
)


class _CapturingWriter:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def enqueue(self, **kwargs) -> bool:
        self.calls.append(kwargs)
        return True

    def actions(self, action: str) -> List[Dict[str, Any]]:
        return [c for c in self.calls if c.get("action") == action]


def _writing_provider(writer, **attrs) -> PgvectorMemoryProvider:
    p = PgvectorMemoryProvider()
    p._healthy = True
    p._writer = writer
    p._agent_identity = "default"
    p._raw_identity = "default"
    p._session_id = "sess-v100"
    p._turn_fingerprints = set()
    for k, v in attrs.items():
        setattr(p, k, v)
    return p


_LONG_U = "A substantive user turn well past the noise threshold for capture."
_LONG_A = "A substantive assistant reply, also long enough to be captured here."


# ---------------------------------------------------------------------------
# PROV-1 -- identity_signature() follows config.yaml, memoised on stat
# ---------------------------------------------------------------------------

@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """A fake hermes_constants pointing get_hermes_home() at tmp_path."""
    mod = types.ModuleType("hermes_constants")
    mod.get_hermes_home = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "hermes_constants", mod)
    return tmp_path


def _write_cfg(home: Path, allowed: str) -> None:
    path = home / "config.yaml"
    path.write_text(f"plugins:\n  pgvector:\n    allowed_themes: {allowed}\n", encoding="utf-8")
    # Guarantee a distinct mtime even on a coarse-resolution filesystem.
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000 * _write_cfg.n))
    _write_cfg.n += 1


_write_cfg.n = 1


def test_identity_signature_changes_when_config_yaml_changes(hermes_home):
    _write_cfg(hermes_home, "marketing")
    # Constructed like register(): the config is frozen at this point.
    p = PgvectorMemoryProvider(config=pkg._load_plugin_config())
    before = p.identity_signature()
    assert before["pgvector.allowed_themes"] == ["marketing"]

    _write_cfg(hermes_home, "marketing,sales")
    after = p.identity_signature()

    assert after != before
    assert after["pgvector.allowed_themes"] == ["marketing", "sales"]
    assert p._config["allowed_themes"] == "marketing", "self._config must not be mutated"


def test_identity_signature_does_not_reparse_an_unchanged_file(hermes_home, monkeypatch):
    _write_cfg(hermes_home, "marketing")
    p = PgvectorMemoryProvider(config=pkg._load_plugin_config())

    real = pkg._load_plugin_config
    parses = []
    monkeypatch.setattr(pkg, "_load_plugin_config", lambda: parses.append(1) or real())

    first = p.identity_signature()
    second = p.identity_signature()
    third = p.identity_signature()

    assert first == second == third
    assert len(parses) == 1, "an unchanged file must be parsed once, then served from the memo"

    _write_cfg(hermes_home, "sales")
    p.identity_signature()
    assert len(parses) == 2


def test_identity_signature_falls_back_to_frozen_config_without_a_file(hermes_home):
    p = PgvectorMemoryProvider(config={"allowed_themes": "marketing"})
    assert not (hermes_home / "config.yaml").exists()
    assert p.identity_signature()["pgvector.allowed_themes"] == ["marketing"]


def test_identity_signature_never_raises_when_home_lookup_fails(monkeypatch):
    mod = types.ModuleType("hermes_constants")

    def _boom():
        raise RuntimeError("no home")

    mod.get_hermes_home = _boom
    monkeypatch.setitem(sys.modules, "hermes_constants", mod)
    p = PgvectorMemoryProvider(config={"allowed_themes": "sales"})
    assert p.identity_signature()["pgvector.allowed_themes"] == ["sales"]


# ---------------------------------------------------------------------------
# PROV-3 -- on_session_end flattens list content like the host
# ---------------------------------------------------------------------------

def test_flatten_user_content_matches_the_host_summarizer():
    img = {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
    assert _flatten_user_content("plain") == "plain"
    assert _flatten_user_content(None) == ""
    assert _flatten_user_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert _flatten_user_content([{"type": "text", "text": "cap"}, img]) == "[1 image] cap"
    assert _flatten_user_content([img, img, {"type": "input_text", "text": "cap"}]) == "[2 images] cap"
    assert _flatten_user_content([img]) == "[1 image]"


def test_on_session_end_does_not_rewrite_a_multimodal_turn_sync_turn_recorded():
    writer = _CapturingWriter()
    p = _writing_provider(writer)
    # The host hands sync_turn the flattened form.
    p.sync_turn(f"[1 image] {_LONG_U}", _LONG_A)
    assert len(writer.actions("turn")) == 2

    p.on_session_end([
        {"role": "user", "content": [
            {"type": "text", "text": _LONG_U},
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
        ]},
        {"role": "assistant", "content": _LONG_A},
    ])
    assert len(writer.actions("turn")) == 2, "the multimodal turn was written a second time"


def test_on_session_end_joins_multiple_text_parts_with_newlines():
    writer = _CapturingWriter()
    p = _writing_provider(writer)
    p.on_session_end([
        {"role": "user", "content": [
            {"type": "text", "text": _LONG_U},
            {"type": "text", "text": "second part of the same turn"},
        ]},
    ])
    turns = writer.actions("turn")
    assert len(turns) == 1
    assert turns[0]["content"] == f"{_LONG_U}\nsecond part of the same turn"


# ---------------------------------------------------------------------------
# PROV-5 -- on_delegation normalizes a supplied child identity
# ---------------------------------------------------------------------------

def test_on_delegation_normalizes_a_supplied_child_identity():
    writer = _CapturingWriter()
    p = _writing_provider(
        writer, _delegation_enabled=True, _writes_enabled=True,
        _config={**pkg.DEFAULTS, "allowed_themes": "marketing",
                 "identity_aliases": "helper=marketing"},
    )
    p.on_delegation("t", "a long enough result text", child_identity="Helper", child_session_id="c1")
    p.on_delegation("t", "a long enough result text", child_identity="rogue-theme", child_session_id="c2")
    p.on_delegation("t", "a long enough result text", child_identity="agent:main:whatsapp:dm:15550100123")

    ids = [e["extra"]["child_identity"] for e in writer.actions("edge")]
    assert ids == ["marketing", "default", "whatsapp-dm"]


def test_on_delegation_keeps_child_identity_none_when_absent():
    writer = _CapturingWriter()
    p = _writing_provider(writer, _delegation_enabled=True, _writes_enabled=True)
    p.on_delegation("t", "a long enough result text", child_session_id="c1")
    p.on_delegation("t", "a long enough result text", child_identity="", child_session_id="c2")
    assert [e["extra"]["child_identity"] for e in writer.actions("edge")] == [None, None]


# ---------------------------------------------------------------------------
# TESTS2-1 -- prefetch generation guard through the WORKER path
# ---------------------------------------------------------------------------

class _RowStore:
    def search(self, **kw):
        return [{"content": "a recalled row", "target": "memory", "score": 0.9}]

    def close(self):
        return None


def test_worker_started_before_a_new_session_cannot_fill_the_new_sessions_cache(monkeypatch):
    gate = threading.Event()
    entered = threading.Event()

    def _blocking_embed(text, *, base_url, model, timeout=None, **kw):
        entered.set()
        gate.wait(timeout=5)
        return [0.1] * 8

    monkeypatch.setattr(pkg, "_embed_text", _blocking_embed)
    p = PgvectorMemoryProvider()
    p._healthy = True
    p._store = _RowStore()
    p._agent_identity = "default"

    p.queue_prefetch("q", session_id="sess-old")
    assert entered.wait(timeout=2.0), "the worker never reached the embed"
    worker = p._prefetch_thread

    # A new session begins while the old worker is still in flight; shutdown()
    # is what initialize() runs on a reused instance and bumps the generation.
    p.shutdown()
    gate.set()
    worker.join(timeout=3.0)
    assert not worker.is_alive()

    assert dict(p._prefetch_cache) == {}, "a stale worker wrote into the next session's cache"


# ---------------------------------------------------------------------------
# TESTS2-5 -- sync_turn parent gating; on_session_end skill normalization
# ---------------------------------------------------------------------------

def test_sync_turn_writes_no_parent_session_id_when_delegation_is_disabled():
    writer = _CapturingWriter()
    p = _writing_provider(writer, _delegation_enabled=False, _parent_session_id="p")
    p.sync_turn(_LONG_U, _LONG_A)
    turns = writer.actions("turn")
    assert len(turns) == 2
    assert all(t["extra"]["parent_session_id"] is None for t in turns)


def test_on_session_end_uses_skill_normalization_to_dedupe_a_skill_turn(monkeypatch):
    """The host hands sync_turn the stripped /skill text but on_session_end the
    RAW transcript; the backstop must normalize before fingerprinting."""
    raw = "[SKILL scaffolding] " + _LONG_U
    monkeypatch.setattr(
        PgvectorMemoryProvider, "_strip_skill_scaffolding",
        staticmethod(lambda c: c[len("[SKILL scaffolding] "):] if c.startswith("[SKILL") else c),
    )
    writer = _CapturingWriter()
    p = _writing_provider(writer)
    p.sync_turn(_LONG_U, _LONG_A)
    assert len(writer.actions("turn")) == 2

    p.on_session_end([{"role": "user", "content": raw}, {"role": "assistant", "content": _LONG_A}])
    assert len(writer.actions("turn")) == 2, "the /skill turn was written twice"


# ---------------------------------------------------------------------------
# TESTS2-6 -- prefetch cache is per session
# ---------------------------------------------------------------------------

def test_prefetch_does_not_return_another_sessions_cached_block(monkeypatch):
    monkeypatch.setattr(pkg, "_embed_text", lambda *a, **k: (_ for _ in ()).throw(pkg.EmbeddingError("x")))
    p = PgvectorMemoryProvider()
    p._healthy = True
    p._store = _RowStore()
    p._agent_identity = "default"
    p._cache_prefetch("sess-A", "block for A", generation=p._prefetch_generation)

    assert p.prefetch("q", session_id="sess-B") == ""
    assert p._prefetch_cache.get("sess-A") == "block for A", "sess-A must stay cached"
    assert p.prefetch("q", session_id="sess-A") == "block for A"


# ---------------------------------------------------------------------------
# TESTS2-9 -- prefetch_budget is clamped to the schema bounds
# ---------------------------------------------------------------------------

def test_prefetch_budget_schema_and_clamp_share_the_same_bounds():
    (entry,) = [e for e in PgvectorMemoryProvider().get_config_schema() if e.get("key") == "prefetch_budget"]
    assert entry["minimum"] == _PREFETCH_BUDGET_MIN
    assert entry["maximum"] == _PREFETCH_BUDGET_MAX


def test_prefetch_clamps_an_oversized_budget_under_the_host_cap(monkeypatch):
    seen = []

    def _rec(text, *, base_url, model, timeout=None, **kw):
        seen.append(timeout)
        return [0.0] * 8

    monkeypatch.setattr(pkg, "_embed_text", _rec)
    p = PgvectorMemoryProvider(config={"prefetch_budget": 30})
    p._healthy = True
    p._store = _RowStore()
    p.prefetch("q")
    assert seen == [_PREFETCH_BUDGET_MAX]

    seen.clear()
    p._config["prefetch_budget"] = 0
    p.prefetch("q")
    assert seen == [_PREFETCH_BUDGET_MIN]


def test_prefetch_with_oversized_budget_returns_before_the_host_timeout(fake_embed_server, monkeypatch):
    # Shrink the clamp so the test stays fast, but exercise the real embed path.
    monkeypatch.setattr(pkg, "_PREFETCH_BUDGET_MAX", 0.5)
    url = fake_embed_server(delay=20, dim=8)
    p = PgvectorMemoryProvider(config={"embed_url": url, "embed_dim": 8, "prefetch_budget": 30})
    p._healthy = True
    p._store = _RowStore()
    p._agent_identity = "default"

    started = time.monotonic()
    assert p.prefetch("find something") == ""
    assert time.monotonic() - started < 0.5 + 0.5
