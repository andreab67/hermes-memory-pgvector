"""Unit tests for tool-arg hardening in PgvectorMemoryProvider.handle_tool_call.

No DB, no embed endpoint, no hermes-agent runtime -- constructs the provider
directly via the standalone-import fallback (pgvector/__init__.py:41-63:
MemoryProvider = object when `agent.memory_provider` isn't importable, added
precisely so this class can be built outside hermes-agent).

GROUP D regression coverage: recall_memory / recall_conversation used to call
`.strip()` directly on unvalidated tool args (`scope`, `target`, `query`), so
a model emitting a non-string value (e.g. {"scope": 42}) raised an
AttributeError ('int' object has no attribute 'strip') out of the tool-call
hook -- a direct violation of invariant #4 (fail-soft everywhere, no
exception escapes into the agent loop). The fix wraps every such arg in
str(...) before .strip().
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_pgvector import PgvectorMemoryProvider  # noqa: E402
import hermes_pgvector as pgvector_pkg  # noqa: E402


# ---------------------------------------------------------------------------
# Outer contract: a bare, never-initialized provider (constructor default:
# _healthy=False, _store=None -- the same state a real deployment is in
# whenever Postgres is unreachable) must never raise out of handle_tool_call,
# no matter what shape of args a model hands it.
# ---------------------------------------------------------------------------

def test_recall_memory_non_string_scope_does_not_raise():
    provider = PgvectorMemoryProvider()
    result = provider.handle_tool_call("recall_memory", {"query": "x", "scope": 42})
    assert isinstance(result, str)
    json.loads(result)  # must be valid JSON, not a traceback string


def test_recall_conversation_non_string_scope_does_not_raise():
    provider = PgvectorMemoryProvider()
    result = provider.handle_tool_call("recall_conversation", {"query": "x", "scope": 42})
    assert isinstance(result, str)
    json.loads(result)


def test_recall_memory_non_string_target_does_not_raise():
    provider = PgvectorMemoryProvider()
    result = provider.handle_tool_call("recall_memory", {"query": "x", "target": 42})
    assert isinstance(result, str)
    json.loads(result)


def test_recall_memory_non_string_query_does_not_raise():
    provider = PgvectorMemoryProvider()
    result = provider.handle_tool_call("recall_memory", {"query": 42})
    assert isinstance(result, str)
    json.loads(result)


# ---------------------------------------------------------------------------
# Direct coverage of the fixed lines. A bare/unhealthy provider (above)
# short-circuits at the "not self._healthy or not self._store" guard before
# ever reaching the str(args.get(...)).strip() calls, so it only proves the
# outer contract, not the coercion itself. To exercise the actual patched
# lines without a DB, mark the provider healthy with a minimal in-memory
# store stub and monkeypatch the module-level `embed` symbol so no network
# call happens.
# ---------------------------------------------------------------------------

class _FakeStore:
    """Just enough surface for handle_tool_call's search calls to succeed."""

    def hybrid_search(self, **kwargs):
        return []

    def search(self, **kwargs):
        return []

    def hybrid_search_turns(self, **kwargs):
        return []

    def search_turns(self, **kwargs):
        return []


def _healthy_provider() -> PgvectorMemoryProvider:
    provider = PgvectorMemoryProvider()
    provider._healthy = True
    provider._store = _FakeStore()
    provider._agent_identity = "default"
    provider._session_id = "sess-test"
    return provider


def test_recall_memory_coerces_non_string_scope_before_use(monkeypatch):
    monkeypatch.setattr(pgvector_pkg, "embed", lambda *a, **k: [0.01] * 768)
    provider = _healthy_provider()

    # Pre-fix, this line called `(args.get("scope") or ...).strip()` directly
    # on the int 42 and raised AttributeError. Now it's str()-wrapped first.
    result = provider.handle_tool_call("recall_memory", {"query": "x", "scope": 42})
    payload = json.loads(result)
    assert payload["results"] == []
    assert payload["count"] == 0


def test_recall_memory_coerces_non_string_target_before_use(monkeypatch):
    monkeypatch.setattr(pgvector_pkg, "embed", lambda *a, **k: [0.01] * 768)
    provider = _healthy_provider()

    result = provider.handle_tool_call("recall_memory", {"query": "x", "target": 42})
    payload = json.loads(result)
    # target "42" isn't 'memory'/'user'/'both' -> a clean tool_error, not a crash.
    assert "error" in payload


def test_recall_conversation_coerces_non_string_scope_before_use(monkeypatch):
    monkeypatch.setattr(pgvector_pkg, "embed", lambda *a, **k: [0.01] * 768)
    provider = _healthy_provider()

    result = provider.handle_tool_call("recall_conversation", {"query": "x", "scope": 42})
    payload = json.loads(result)
    assert payload["results"] == []
    assert payload["count"] == 0
