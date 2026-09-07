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


# ---------------------------------------------------------------------------
# The CLI recovery path, and the retry budget.
# ---------------------------------------------------------------------------

def test_backfill_cli_uses_the_write_timeout(monkeypatch):
    """`hermes-pgvector backfill` is the documented repair path for rows that
    landed with a NULL embedding. It used to pass NO timeout, so it inherited
    embed()'s hardcoded 10s -- the very condition that produced those rows on a
    slow endpoint, so the sweep kept failing at the one job it exists to do."""
    import hermes_pgvector.__main__ as cli

    rec = _Recorder()
    monkeypatch.setattr(cli, "embed", rec)

    class _Args:
        dsn = embed_url = embed_model = None
        config = None
    fn = cli._make_embed_fn(_Args(), {"embed_write_timeout": 45.0})
    fn("a row being re-embedded by the nightly sweep")
    assert rec.timeouts == [45.0], f"backfill did not plumb the timeout: {rec.timeouts}"


def test_backfill_cli_falls_back_to_the_default_timeout(monkeypatch):
    import hermes_pgvector.__main__ as cli

    rec = _Recorder()
    monkeypatch.setattr(cli, "embed", rec)

    class _Args:
        dsn = embed_url = embed_model = None
        config = None
    fn = cli._make_embed_fn(_Args(), {"embed_write_timeout": "not-a-number"})
    fn("text")
    assert rec.timeouts == [DEFAULTS["embed_write_timeout"]]


def test_retries_stop_at_the_total_budget():
    """Retries exist for TRANSIENT failures, which fail fast. They must not
    multiply a slow endpoint's per-attempt timeout into minutes of blocking:
    each attempt can cost 2x timeout (OpenAI-compat path, then the Ollama
    fallback), so 3 attempts x 2 paths x 30s would be 180s on the writer's
    drain thread, filling the bounded queue and dropping writes."""
    import time as _time
    # NB: hermes_pgvector/__init__.py does `from .embed import embed`, which
    # shadows the submodule attribute -- `hermes_pgvector.embed` is the
    # FUNCTION. import_module reaches the real module via sys.modules.
    from importlib import import_module
    embed_mod = import_module("hermes_pgvector.embed")

    attempts = {"n": 0}

    def _slow_fail(text, *, base_url, model, timeout):
        attempts["n"] += 1
        _time.sleep(0.05)
        raise embed_mod.EmbeddingError("endpoint is slow and failing")

    orig = embed_mod._embed_once
    embed_mod._embed_once = _slow_fail
    try:
        try:
            embed_mod.embed(
                "x", base_url="http://example.invalid", model="m",
                timeout=1.0, retries=10, backoff=0.0, max_total=0.12,
            )
        except embed_mod.EmbeddingError:
            pass
    finally:
        embed_mod._embed_once = orig

    assert attempts["n"] < 11, "the budget must cut retries short"
    assert attempts["n"] >= 2, "it must still retry at least once before the budget"


def test_no_budget_means_all_retries_still_run():
    """max_total=None preserves the old behaviour for callers that want it."""
    # NB: hermes_pgvector/__init__.py does `from .embed import embed`, which
    # shadows the submodule attribute -- `hermes_pgvector.embed` is the
    # FUNCTION. import_module reaches the real module via sys.modules.
    from importlib import import_module
    embed_mod = import_module("hermes_pgvector.embed")

    attempts = {"n": 0}

    def _fail(text, *, base_url, model, timeout):
        attempts["n"] += 1
        raise embed_mod.EmbeddingError("nope")

    orig = embed_mod._embed_once
    embed_mod._embed_once = _fail
    try:
        try:
            embed_mod.embed("x", base_url="http://example.invalid", model="m",
                            retries=3, backoff=0.0)
        except embed_mod.EmbeddingError:
            pass
    finally:
        embed_mod._embed_once = orig
    assert attempts["n"] == 4
