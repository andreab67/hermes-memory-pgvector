"""Embed timeouts must be configurable and must reach embed() (v0.5.0).

No DB, no network -- embed() is monkeypatched to record the kwargs it receives.

Why this exists. Before v0.5.0 the `timeout` argument was never plumbed from
config at ALL: every caller silently took embed()'s hardcoded 10s default. On a
deployment whose embed endpoint answers in 6-17s (measured: the home k8s
nomic-embed-text) that meant a large share of background writes timed out, took
the fail-soft path, and landed as rows with a NULL embedding -- invisible to
recall until the nightly backfill sweep repaired them.

Retries could not rescue that, which is the subtle part: `embed_write_retries`
bounds re-attempts, but every attempt was capped BELOW the latency the endpoint
actually needs, so retrying just burned the budget and failed identically.

The two timeouts are deliberately different, and that asymmetry is the contract
these tests pin:

  * hot path (prefetch, recall_memory, recall_conversation, and the init-time
    bulk import) -- SHORT. A timeout there degrades recall to full-text-only,
    which is a good outcome; making the agent wait is a worse one.
  * writer drain -- LONG. Nothing waits on it, and giving up costs a
    permanently unsearchable row.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_pgvector import DEFAULTS, PgvectorMemoryProvider  # noqa: E402
import hermes_pgvector as pgvector_pkg  # noqa: E402


class _Recorder:
    """Stands in for embed(); records the timeout it was handed."""

    def __init__(self):
        self.timeouts = []

    def __call__(self, text, *, base_url, model, timeout=None, **kw):
        self.timeouts.append(timeout)
        return [0.0] * 768


def _provider(monkeypatch, **config):
    rec = _Recorder()
    monkeypatch.setattr(pgvector_pkg, "embed", rec)
    p = PgvectorMemoryProvider(config=config or None)
    p._healthy = True
    p._agent_identity = "marketing"
    p._raw_identity = "marketing"
    p._session_id = "sess-timeout"
    return p, rec


class _NullStore:
    def search(self, **kw): return []
    def hybrid_search(self, **kw): return []
    def search_turns(self, **kw): return []
    def hybrid_search_turns(self, **kw): return []


# --- the two defaults are distinct, and the write one is larger -------------

def test_write_timeout_default_is_more_generous_than_the_hot_path():
    assert DEFAULTS["embed_write_timeout"] > DEFAULTS["embed_timeout"], (
        "the background writer must tolerate a slower endpoint than the agent "
        "thread does -- that asymmetry is the whole point"
    )


def test_hot_path_timeout_default_is_the_documented_value():
    p = PgvectorMemoryProvider()
    assert p._embed_timeout() == DEFAULTS["embed_timeout"]


# --- config actually reaches embed() ---------------------------------------

def test_recall_memory_passes_the_configured_hot_timeout(monkeypatch):
    p, rec = _provider(monkeypatch, embed_timeout=3.5)
    p._store = _NullStore()
    p.handle_tool_call("recall_memory", {"query": "q"})
    assert rec.timeouts == [3.5], f"hot path did not plumb the timeout: {rec.timeouts}"


def test_recall_conversation_passes_the_configured_hot_timeout(monkeypatch):
    p, rec = _provider(monkeypatch, embed_timeout=4.5)
    p._store = _NullStore()
    p.handle_tool_call("recall_conversation", {"query": "q"})
    assert rec.timeouts == [4.5]


def test_writer_path_passes_the_configured_write_timeout(monkeypatch):
    p, rec = _provider(monkeypatch, embed_write_timeout=42.0)
    p._maybe_embed("some durable content worth embedding")
    assert rec.timeouts == [42.0], f"writer path did not plumb the timeout: {rec.timeouts}"


def test_writer_and_hot_path_do_not_share_a_timeout(monkeypatch):
    """The regression that motivated this: one hardcoded value for both."""
    p, rec = _provider(monkeypatch, embed_timeout=2.0, embed_write_timeout=60.0)
    p._store = _NullStore()
    p.handle_tool_call("recall_memory", {"query": "q"})
    p._maybe_embed("durable content")
    assert rec.timeouts == [2.0, 60.0], (
        f"hot and writer paths must use different timeouts, got {rec.timeouts}"
    )


def test_bulk_import_closure_uses_the_hot_timeout(monkeypatch):
    """The init-time bulk import blocks session start, so it takes the SHORT
    timeout despite being a write."""
    p, rec = _provider(monkeypatch, embed_timeout=5.0, embed_write_timeout=90.0)
    fn = p._make_embed_fn()
    assert fn is not None
    fn("an entry from MEMORY.md")
    assert rec.timeouts == [5.0], (
        f"bulk import must not inherit the writer's long timeout: {rec.timeouts}"
    )


# --- string / garbage config (invariant 9) ---------------------------------

def test_string_valued_timeouts_are_coerced(monkeypatch):
    """save_config() persists the schema's declared values as STRINGS."""
    p, rec = _provider(monkeypatch, embed_timeout="7.5", embed_write_timeout="25")
    p._store = _NullStore()
    p.handle_tool_call("recall_memory", {"query": "q"})
    p._maybe_embed("durable content")
    assert rec.timeouts == [7.5, 25.0]


def test_garbage_timeouts_fall_back_to_defaults(monkeypatch):
    p, rec = _provider(monkeypatch, embed_timeout="", embed_write_timeout="soon")
    p._store = _NullStore()
    p.handle_tool_call("recall_memory", {"query": "q"})
    p._maybe_embed("durable content")
    assert rec.timeouts == [DEFAULTS["embed_timeout"], DEFAULTS["embed_write_timeout"]]


def test_both_timeouts_are_exposed_in_the_config_schema():
    """Operators must be able to tune this without editing source: the whole
    defect was that the value existed only as a hardcoded literal."""
    keys = {e["key"] for e in PgvectorMemoryProvider().get_config_schema()}
    assert "embed_timeout" in keys
    assert "embed_write_timeout" in keys
