"""Read-side identity gate: the PII / bench sinks must not leak into recall.

No DB, no embed endpoint -- the provider is built via the standalone-import
fallback and handed a recording stub store, so these assert on the FILTERS the
provider passes down rather than on query results.

Why this exists. identity.py buckets direct-message traffic into `whatsapp-dm`
and benchmark traffic into `_bench`. That bucketing was WRITE-side only: it
strips PII from the *identity*, but the message bodies still land in `content`.
Nothing filtered on read, so any theme could pull DM content into its context
with scope='all' -- or, because `scope` is free text, by naming the bucket
outright. With turn capture on, the assistant reply quoting that content was
then written back under the *reading* theme, permanently re-attributing DM data
into a production theme.

The gate deliberately does NOT narrow ordinary cross-theme recall: scope='all'
still sweeps every real theme, and an agent whose own identity IS a bucket keeps
full access to its own rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_pgvector import PgvectorMemoryProvider  # noqa: E402
import hermes_pgvector as pgvector_pkg  # noqa: E402
from hermes_pgvector.identity import BENCH_BUCKET, DM_BUCKET  # noqa: E402


class _RecordingStore:
    """Captures the kwargs of whichever search path the provider chooses."""

    def __init__(self):
        self.kwargs = {}
        self.called = None

    def _rec(self, name, kw):
        self.called = name
        self.kwargs = kw
        return []

    def search(self, **kw):
        return self._rec("search", kw)

    def hybrid_search(self, **kw):
        return self._rec("hybrid_search", kw)

    def search_turns(self, **kw):
        return self._rec("search_turns", kw)

    def hybrid_search_turns(self, **kw):
        return self._rec("hybrid_search_turns", kw)


def _provider(identity: str, monkeypatch, *, hybrid: bool = True):
    # Never touch the network: the recall handlers embed the query first.
    monkeypatch.setattr(pgvector_pkg, "embed", lambda *a, **k: [0.0] * 768)
    p = PgvectorMemoryProvider()
    p._healthy = True
    p._store = _RecordingStore()
    p._agent_identity = identity
    p._raw_identity = identity
    p._session_id = "sess-gate"
    p._config["hybrid_search"] = hybrid
    return p


# --- scope='all' must exclude the sinks ------------------------------------

def test_recall_memory_scope_all_excludes_both_sinks(monkeypatch):
    p = _provider("marketing", monkeypatch)
    p.handle_tool_call("recall_memory", {"query": "q", "scope": "all"})
    assert p._store.kwargs.get("agent_identity") is None, "scope=all is still cross-theme"
    assert set(p._store.kwargs.get("exclude_identities") or []) == {DM_BUCKET, BENCH_BUCKET}


def test_recall_conversation_scope_all_excludes_both_sinks(monkeypatch):
    p = _provider("marketing", monkeypatch)
    p.handle_tool_call("recall_conversation", {"query": "q", "scope": "all"})
    assert set(p._store.kwargs.get("exclude_identities") or []) == {DM_BUCKET, BENCH_BUCKET}


def test_scope_all_excludes_sinks_on_the_non_hybrid_path(monkeypatch):
    p = _provider("marketing", monkeypatch, hybrid=False)
    p.handle_tool_call("recall_memory", {"query": "q", "scope": "all"})
    assert p._store.called == "search"
    assert set(p._store.kwargs.get("exclude_identities") or []) == {DM_BUCKET, BENCH_BUCKET}


def test_scope_all_excludes_sinks_on_the_hybrid_failure_fallback(monkeypatch):
    """The fallback is the dangerous one: if hybrid_search raises and the
    fallback search() were left ungated, any transient hybrid error would
    silently hand DM content to the caller."""
    p = _provider("marketing", monkeypatch)
    calls = {}

    def _boom(**kw):
        raise RuntimeError("hybrid exploded")

    def _fallback(**kw):
        calls.update(kw)
        return []

    p._store.hybrid_search = _boom
    p._store.search = _fallback
    p.handle_tool_call("recall_memory", {"query": "q", "scope": "all"})
    assert set(calls.get("exclude_identities") or []) == {DM_BUCKET, BENCH_BUCKET}


# --- naming a sink directly must be refused --------------------------------

def test_recall_memory_rejects_explicit_dm_bucket(monkeypatch):
    p = _provider("marketing", monkeypatch)
    out = json.loads(p.handle_tool_call("recall_memory", {"query": "q", "scope": DM_BUCKET}))
    assert "restricted sink" in (out.get("error") or "")
    assert p._store.called is None, "must refuse before querying the store"


def test_recall_memory_rejects_explicit_bench_bucket(monkeypatch):
    p = _provider("marketing", monkeypatch)
    out = json.loads(p.handle_tool_call("recall_memory", {"query": "q", "scope": BENCH_BUCKET}))
    assert "restricted sink" in (out.get("error") or "")


def test_recall_conversation_rejects_explicit_sink(monkeypatch):
    p = _provider("marketing", monkeypatch)
    out = json.loads(
        p.handle_tool_call("recall_conversation", {"query": "q", "scope": DM_BUCKET})
    )
    assert "restricted sink" in (out.get("error") or "")


# --- the gate must not over-reach ------------------------------------------

def test_dm_agent_keeps_access_to_its_own_rows(monkeypatch):
    """An agent operating AS the DM bucket must still recall its own history --
    the gate is about cross-theme leakage, not self-denial."""
    p = _provider(DM_BUCKET, monkeypatch)
    p.handle_tool_call("recall_memory", {"query": "q", "scope": "all"})
    excl = set(p._store.kwargs.get("exclude_identities") or [])
    assert DM_BUCKET not in excl
    assert excl == {BENCH_BUCKET}


def test_dm_agent_scope_current_is_unaffected(monkeypatch):
    p = _provider(DM_BUCKET, monkeypatch)
    p.handle_tool_call("recall_memory", {"query": "q", "scope": "current"})
    assert p._store.kwargs.get("agent_identity") == DM_BUCKET
    assert not p._store.kwargs.get("exclude_identities")


def test_ordinary_cross_theme_recall_is_not_narrowed(monkeypatch):
    """Naming an ordinary theme still works and carries no exclusion -- the
    gate must not turn into a general restriction on cross-theme recall."""
    p = _provider("marketing", monkeypatch)
    p.handle_tool_call("recall_memory", {"query": "q", "scope": "sales"})
    assert p._store.kwargs.get("agent_identity") == "sales"
    assert not p._store.kwargs.get("exclude_identities")


def test_scope_current_carries_no_exclusion(monkeypatch):
    p = _provider("marketing", monkeypatch)
    p.handle_tool_call("recall_memory", {"query": "q", "scope": "current"})
    assert p._store.kwargs.get("agent_identity") == "marketing"
    assert not p._store.kwargs.get("exclude_identities")
