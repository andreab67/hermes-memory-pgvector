"""v0.6.0 WP-D2 coverage for hermes_pgvector/__init__.py's read paths + the
config contract.

Findings closed here (docs/code-review/1.0-readiness-review-2026-09-26.md,
docs/release/PLAN-1.0.md Sec 5 WP-D2):

H4  -- queue_prefetch(query, session_id=...) runs on ONE background daemon
       worker (at most one in flight; a newer request while one is running
       REPLACES the queued one rather than spawning a second thread) that
       embeds + searches and caches the formatted block keyed by
       session_id. prefetch() consumes (pops) that cache first; only when
       nothing is cached does it fall back to a SYNCHRONOUS embed + search
       bounded by the new `prefetch_budget` config key (default 5.0s, below
       the host's hard 8s external-prefetch timeout --
       agent.memory_manager._EXTERNAL_PREFETCH_TIMEOUT_S). embed_timeout's
       default also drops 10.0 -> 5.0 (see test_embed_timeouts.py for the
       embed()-level deadline-sharing fix this depends on). initialize()/
       shutdown() drop the cache (via a generation counter, so a worker
       still finishing a PRIOR session cannot write into the next one's
       cache) and neither prefetch() nor queue_prefetch() may ever raise.
M6  -- system_prompt_block() is computed ONCE (initialize() primes it as its
       last step; a lazy fallback covers a caller that skips initialize())
       and cached -- later calls never re-query the DB. The global "~N
       total" figure comes from estimate_count("memory_entries")
       (pg_class.reltuples), never an unscoped COUNT(*); a None estimate
       (unknown) is never rendered as "0 total", and a failed SCOPED count
       still returns "" rather than falsely claiming "Empty store".
L2  -- DEFAULTS["embed_url"] is now "http://localhost:11434", not a private
       LAN address.
L3  -- the module docstring lists both tools (recall_memory,
       recall_conversation); the write_queue_maxsize schema description says
       "newest" writes drop, not "oldest" (that is what
       AsyncWriter.enqueue's queue.Queue.put_nowait actually does); no
       `v0.x:` prefixes remain in schema descriptions.
Config schema -- every entry declares `type` (text|integer|number|boolean)
       and a REAL-typed `default` (not a string); every DEFAULTS key an
       operator may set has a schema entry, including the ones this WP adds
       (identity_aliases, embed_write_backoff, write_contexts,
       prefetch_budget, shutdown_drain_timeout).
identity_signature() (optional, implemented) -- governance config
       (allowed_themes, identity_aliases, bench_mode, write_contexts) under
       pgvector.-prefixed keys, JSON-serializable, cheap, read-only, works
       on an uninitialized instance.

Two kinds of tests:
  * DB-free, deterministic unit tests against stub stores/embed functions --
    these run everywhere and pin the exact caching/scheduling/formatting
    logic without any real network or DB race.
  * DB-gated acceptance tests (skip without PG_TEST_DSN) that exercise the
    real MemoryStore end to end.
"""

from __future__ import annotations

import json as _json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_pgvector import (  # noqa: E402
    DEFAULTS,
    EmbeddingError,
    PgvectorMemoryProvider,
)
import hermes_pgvector as pgvector_pkg  # noqa: E402


# ---------------------------------------------------------------------------
# Shared stubs (DB-free tests)
# ---------------------------------------------------------------------------

class _StubSearchStore:
    """Store stand-in for prefetch()/queue_prefetch(): records every
    search() call and returns a fixed row list."""

    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None):
        self.rows = rows if rows is not None else []
        self.search_calls: List[Dict[str, Any]] = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return list(self.rows)

    def close(self) -> None:
        """No-op -- lets shutdown() run against this stub without error."""


class _SequencedSearchStore:
    """Returns a DIFFERENT single-row result on each successive search()
    call, so a test can tell which of several queued requests actually
    produced the final cached value."""

    def __init__(self, contents: List[str]):
        self._contents = list(contents)
        self.search_calls = 0

    def search(self, **kwargs):
        idx = min(self.search_calls, len(self._contents) - 1)
        self.search_calls += 1
        return [{"content": self._contents[idx], "target": "memory", "score": 0.9}]

    def close(self) -> None:
        """No-op -- lets shutdown() run against this stub without error."""


class _CountingStore:
    """Spies on count()/estimate_count() so a test can assert
    system_prompt_block() queries the DB at most once per initialize()."""

    def __init__(self, scoped: int = 3, estimate: Optional[int] = 42):
        self.scoped = scoped
        self.estimate = estimate
        self.count_calls: List[Dict[str, Any]] = []
        self.estimate_calls: List[str] = []

    def count(self, **kwargs):
        self.count_calls.append(kwargs)
        return self.scoped

    def estimate_count(self, table):
        self.estimate_calls.append(table)
        return self.estimate


def _bare_healthy_provider(store, **attrs) -> PgvectorMemoryProvider:
    p = PgvectorMemoryProvider()
    p._healthy = True
    p._store = store
    p._agent_identity = "default"
    for k, v in attrs.items():
        setattr(p, k, v)
    return p


# ---------------------------------------------------------------------------
# L2 -- embed_url default
# ---------------------------------------------------------------------------

def test_embed_url_default_is_localhost_not_a_private_lan_address():
    assert DEFAULTS["embed_url"] == "http://localhost:11434"


# ---------------------------------------------------------------------------
# L3 -- module docstring + schema description text
# ---------------------------------------------------------------------------

def test_module_docstring_lists_both_tools():
    doc = pgvector_pkg.__doc__ or ""
    assert "recall_conversation" in doc
    assert "one explicit search tool" not in doc


def test_write_queue_maxsize_description_says_newest_not_oldest():
    schema = {e["key"]: e for e in PgvectorMemoryProvider().get_config_schema()}
    desc = schema["write_queue_maxsize"]["description"]
    assert "newest" in desc
    assert "oldest" not in desc


def test_schema_descriptions_have_no_stale_version_prefixes():
    version_prefix = re.compile(r"\bv0\.\d+(\.\d+)?:")
    for entry in PgvectorMemoryProvider().get_config_schema():
        desc = entry.get("description", "")
        assert not version_prefix.search(desc), (
            f"{entry['key']} description still has a version-prefix: {desc!r}"
        )


# ---------------------------------------------------------------------------
# Config schema contract -- type + real-typed default + completeness
# ---------------------------------------------------------------------------

_SCHEMA_TYPES = {
    "text": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


def test_every_schema_entry_declares_a_valid_type():
    for entry in PgvectorMemoryProvider().get_config_schema():
        assert "type" in entry, f"{entry['key']} has no 'type'"
        assert entry["type"] in _SCHEMA_TYPES, f"{entry['key']} has unknown type {entry['type']!r}"


def test_every_schema_entry_has_a_real_typed_default_not_a_string():
    for entry in PgvectorMemoryProvider().get_config_schema():
        key, t, default = entry["key"], entry["type"], entry.get("default")
        if t == "boolean":
            assert isinstance(default, bool), f"{key}: default {default!r} is not a real bool"
        elif t == "integer":
            assert isinstance(default, int) and not isinstance(default, bool), (
                f"{key}: default {default!r} is not a real int"
            )
        elif t == "number":
            assert isinstance(default, (int, float)) and not isinstance(default, bool), (
                f"{key}: default {default!r} is not a real number"
            )
        else:  # text
            assert isinstance(default, str), f"{key}: default {default!r} is not a real str"


def test_every_defaults_key_an_operator_may_set_has_a_schema_entry():
    schema_keys = {e["key"] for e in PgvectorMemoryProvider().get_config_schema()}
    missing = set(DEFAULTS.keys()) - schema_keys
    assert not missing, f"DEFAULTS keys with no config schema entry: {sorted(missing)}"


@pytest.mark.parametrize(
    "key", ["identity_aliases", "embed_write_backoff", "write_contexts",
            "prefetch_budget", "shutdown_drain_timeout"],
)
def test_previously_undeclared_keys_are_now_in_the_schema(key):
    schema_keys = {e["key"] for e in PgvectorMemoryProvider().get_config_schema()}
    assert key in schema_keys


def test_minimum_and_maximum_are_declared_where_meaningful():
    schema = {e["key"]: e for e in PgvectorMemoryProvider().get_config_schema()}
    assert schema["min_similarity"]["minimum"] == 0.0
    assert schema["min_similarity"]["maximum"] == 1.0
    assert schema["embed_dim"]["minimum"] == 1
    assert schema["prefetch_budget"]["minimum"] > 0
    # prefetch_budget must stay below the host's hard 8s external-prefetch
    # timeout -- the schema's own maximum should enforce that headroom.
    assert schema["prefetch_budget"]["maximum"] < 8.0


# ---------------------------------------------------------------------------
# identity_signature() (optional item, implemented)
# ---------------------------------------------------------------------------

def test_identity_signature_is_callable_on_an_uninitialized_instance():
    p = PgvectorMemoryProvider()  # initialize() never called
    sig = p.identity_signature()
    _json.dumps(sig)  # must be JSON-serializable
    assert set(sig.keys()) == {
        "pgvector.allowed_themes",
        "pgvector.identity_aliases",
        "pgvector.bench_mode",
        "pgvector.write_contexts",
    }


def test_identity_signature_reflects_coerced_config_values():
    p = PgvectorMemoryProvider(config={
        "allowed_themes": "marketing,sales",
        "identity_aliases": "mkt=marketing",
        "bench_mode": "reject",
        "write_contexts": "primary,cron,subagent",
    })
    sig = p.identity_signature()
    assert sig["pgvector.allowed_themes"] == ["marketing", "sales"]
    assert sig["pgvector.identity_aliases"] == {"mkt": "marketing"}
    assert sig["pgvector.bench_mode"] == "reject"
    assert sig["pgvector.write_contexts"] == ["primary", "cron", "subagent"]


def test_identity_signature_changes_when_governance_config_changes():
    a = PgvectorMemoryProvider(config={"allowed_themes": "marketing"}).identity_signature()
    b = PgvectorMemoryProvider(config={"allowed_themes": "sales"}).identity_signature()
    assert a != b


# ---------------------------------------------------------------------------
# M6 -- system_prompt_block(): computed once, estimate_count (not COUNT(*)),
# never a false "Empty store"/"0 total" claim.
# ---------------------------------------------------------------------------

def test_system_prompt_block_queries_the_db_at_most_once_regardless_of_call_count():
    store = _CountingStore(scoped=3, estimate=42)
    p = _bare_healthy_provider(store, _agent_identity="marketing")

    first = p.system_prompt_block()
    second = p.system_prompt_block()
    third = p.system_prompt_block()

    assert first == second == third
    assert len(store.count_calls) == 1
    assert len(store.estimate_calls) == 1


def test_system_prompt_block_uses_scoped_count_and_estimate_not_an_unscoped_count():
    store = _CountingStore(scoped=3, estimate=42)
    p = _bare_healthy_provider(store, _agent_identity="marketing")

    block = p.system_prompt_block()

    assert store.count_calls == [{"agent_identity": "marketing"}]
    assert store.estimate_calls == ["memory_entries"]
    assert "3 entries for 'marketing'" in block
    assert "42 total across all themes" in block


def test_system_prompt_block_never_renders_an_unknown_estimate_as_zero():
    store = _CountingStore(scoped=5, estimate=None)
    p = _bare_healthy_provider(store, _agent_identity="marketing")

    block = p.system_prompt_block()

    assert "0 total" not in block
    assert "Empty store" not in block
    assert "5 entries for 'marketing'" in block
    assert "unknown number of entries" in block


def test_system_prompt_block_reports_nothing_yet_not_a_false_empty_store_claim():
    store = _CountingStore(scoped=0, estimate=None)
    p = _bare_healthy_provider(store, _agent_identity="marketing")

    block = p.system_prompt_block()

    assert "Empty store" not in block
    assert "No entries yet for this theme" in block


def test_system_prompt_block_treats_a_missing_estimate_count_method_as_unknown():
    """A store without estimate_count() (e.g. a minimal test double) must
    degrade to 'unknown', not raise into the agent loop."""
    class _NoEstimateStore:
        def count(self, **kwargs):
            return 2

    p = _bare_healthy_provider(_NoEstimateStore(), _agent_identity="marketing")
    block = p.system_prompt_block()
    assert "2 entries for 'marketing'" in block
    assert "unknown number of entries" in block


def test_system_prompt_block_lazy_path_still_returns_empty_string_on_count_failure():
    """Same contract as tests/test_system_prompt_block.py, exercised through
    the new cached/lazy system_prompt_block(): a failed SCOPED count is not
    the same fact as an empty store."""
    class _RaisingCountStore:
        def count(self, **kwargs):
            raise RuntimeError("connection pool exhausted")

    p = _bare_healthy_provider(_RaisingCountStore())
    result = p.system_prompt_block()
    assert result == ""
    assert "Empty store" not in result


# ---------------------------------------------------------------------------
# H4 -- prefetch()/queue_prefetch(): cache consumption, budget, never raises.
# ---------------------------------------------------------------------------

def test_prefetch_returns_empty_string_when_unhealthy_or_store_missing():
    p = PgvectorMemoryProvider()
    assert p.prefetch("q") == ""
    p._healthy = True
    assert p.prefetch("q") == ""  # still no store


def test_prefetch_consumes_a_cached_block_exactly_once(monkeypatch):
    def _boom(text, *, base_url, model, timeout=None, **kw):
        raise EmbeddingError("no endpoint reachable in this test")

    monkeypatch.setattr(pgvector_pkg, "_embed_text", _boom)

    p = _bare_healthy_provider(_StubSearchStore())
    p._prefetch_cache["sess-1"] = "## Recall (pgvector, default)\n- cached block"

    first = p.prefetch("irrelevant query", session_id="sess-1")
    assert first == "## Recall (pgvector, default)\n- cached block"

    second = p.prefetch("irrelevant query", session_id="sess-1")
    assert second == "", "the cached block must be popped, not read repeatedly"


def test_prefetch_passes_prefetch_budget_as_the_embed_timeout(monkeypatch):
    seen = []

    def _rec(text, *, base_url, model, timeout=None, **kw):
        seen.append(timeout)
        return [0.0] * 768

    monkeypatch.setattr(pgvector_pkg, "_embed_text", _rec)
    p = _bare_healthy_provider(_StubSearchStore(), _config={**DEFAULTS, "prefetch_budget": 2.5})
    p.prefetch("q")
    assert seen == [2.5]


def test_prefetch_budget_default_is_5_seconds():
    assert DEFAULTS["prefetch_budget"] == 5.0


def test_prefetch_returns_within_its_budget_against_a_slow_endpoint(fake_embed_server):
    """H4 acceptance: with the fake embed server delayed well past the
    budget, prefetch() must return within budget + 0.5s, and return ""."""
    url = fake_embed_server(delay=20, dim=8)
    p = PgvectorMemoryProvider(config={
        "embed_url": url, "embed_dim": 8, "prefetch_budget": 0.4,
    })
    p._healthy = True
    p._agent_identity = "default"
    p._store = _StubSearchStore()

    started = time.monotonic()
    result = p.prefetch("find something")
    elapsed = time.monotonic() - started

    assert result == ""
    assert elapsed <= 0.4 + 0.5, f"prefetch exceeded prefetch_budget + 0.5s: {elapsed:.2f}s"


def test_prefetch_never_raises_when_the_embed_function_raises_something_unexpected(monkeypatch):
    """Invariant #4: prefetch() has an outer safety net beyond the narrow
    EmbeddingError catch, for a bug in the embed path that would otherwise
    escape into the agent loop."""
    def _boom(text, *, base_url, model, timeout=None, **kw):
        raise RuntimeError("not an EmbeddingError")

    monkeypatch.setattr(pgvector_pkg, "_embed_text", _boom)
    p = _bare_healthy_provider(_StubSearchStore())
    assert p.prefetch("q") == ""


def test_queue_prefetch_never_raises_when_store_is_missing():
    p = PgvectorMemoryProvider()
    p._healthy = True
    p.queue_prefetch("q", session_id="s1")  # must not raise


def test_queue_prefetch_is_a_noop_for_an_empty_query():
    p = _bare_healthy_provider(_StubSearchStore())
    p.queue_prefetch("", session_id="s1")
    assert p._prefetch_pending is None
    assert p._prefetch_thread is None


def test_queue_prefetch_worker_failure_is_swallowed_and_caches_an_empty_block(monkeypatch):
    def _boom(text, *, base_url, model, timeout=None, **kw):
        raise RuntimeError("boom -- not even an EmbeddingError")

    monkeypatch.setattr(pgvector_pkg, "_embed_text", _boom)
    p = _bare_healthy_provider(_StubSearchStore())

    p.queue_prefetch("q", session_id="s1")

    deadline = time.monotonic() + 2.0
    cached = None
    while time.monotonic() < deadline:
        with p._prefetch_lock:
            cached = p._prefetch_cache.get("s1")
        if cached is not None:
            break
        time.sleep(0.01)
    assert cached == "", "a worker failure must still resolve to a cached empty block, never a crash"


def test_queue_prefetch_worker_thread_is_a_daemon_thread(monkeypatch):
    monkeypatch.setattr(pgvector_pkg, "_embed_text", lambda *a, **kw: [0.0] * 8)
    p = _bare_healthy_provider(_StubSearchStore())
    p.queue_prefetch("q", session_id="s1")
    assert p._prefetch_thread is not None
    assert p._prefetch_thread.daemon is True
    p._prefetch_thread.join(timeout=2.0)


def test_queue_prefetch_runs_at_most_one_worker_and_the_newest_request_wins(monkeypatch):
    gate = threading.Event()
    seen_queries = []

    def _blocking_embed(text, *, base_url, model, timeout=None, **kw):
        seen_queries.append(text)
        gate.wait(timeout=5)
        return [0.1] * 8

    monkeypatch.setattr(pgvector_pkg, "_embed_text", _blocking_embed)
    store = _SequencedSearchStore(["row for A", "row for B"])
    p = _bare_healthy_provider(store)

    p.queue_prefetch("query A", session_id="s1")  # starts the worker; blocks on gate
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not seen_queries:
        time.sleep(0.01)
    assert seen_queries == ["query A"], "the worker must have started on A before B is queued"

    p.queue_prefetch("query B", session_id="s1")  # A is in flight -- must NOT spawn a 2nd thread
    time.sleep(0.05)
    assert seen_queries == ["query A"], "a second worker thread must not start while one is in flight"

    gate.set()  # release A; the worker should then pick up the newer B and finish

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and seen_queries != ["query A", "query B"]:
        time.sleep(0.01)
    assert seen_queries == ["query A", "query B"], f"expected A then the newer B, got {seen_queries}"

    deadline = time.monotonic() + 2.0
    cached = None
    while time.monotonic() < deadline:
        with p._prefetch_lock:
            cached = p._prefetch_cache.get("s1")
        if cached is not None:
            break
        time.sleep(0.01)
    assert cached is not None
    assert "row for B" in cached, "the final cached block must reflect the NEWER request, not the stale one"


def test_shutdown_drops_the_prefetch_cache_and_bumps_the_generation():
    p = PgvectorMemoryProvider()
    p._prefetch_cache["sess-1"] = "stale block"
    p._prefetch_pending = ("q", "sess-1", 0)
    generation_before = p._prefetch_generation

    p.shutdown()

    assert dict(p._prefetch_cache) == {}
    assert p._prefetch_pending is None
    assert p._prefetch_generation == generation_before + 1


def test_a_worker_from_a_stale_generation_cannot_write_into_the_next_sessions_cache():
    p = _bare_healthy_provider(_StubSearchStore())
    stale_generation = p._prefetch_generation
    p.shutdown()  # bumps the generation, as a real teardown/re-init would
    assert p._prefetch_generation != stale_generation

    p._cache_prefetch("sess-1", "stale block", generation=stale_generation)
    assert p._prefetch_cache.get("sess-1") is None


def test_prefetch_cache_is_bounded_to_the_most_recent_sessions():
    p = PgvectorMemoryProvider()
    cap = p._PREFETCH_CACHE_CAP
    for i in range(cap + 10):
        p._cache_prefetch(f"sess-{i}", f"block-{i}", generation=p._prefetch_generation)
    assert len(p._prefetch_cache) == cap
    assert f"sess-{cap + 9}" in p._prefetch_cache
    assert "sess-0" not in p._prefetch_cache


# ---------------------------------------------------------------------------
# DB-gated acceptance tests
# ---------------------------------------------------------------------------

FLAT = [0.1] * 768


@pytest.fixture
def dsn():
    d = os.environ.get("PG_TEST_DSN")
    if not d:
        pytest.skip("PG_TEST_DSN not set")
    return d


@pytest.fixture
def live_read_provider(dsn, monkeypatch):
    """A real, healthy provider (real MemoryStore) scoped to one random
    agent_identity. The embed function is monkeypatched to a FIXED vector so
    seeded rows and queries are guaranteed to match exactly (cosine
    similarity 1.0), independent of the real embed endpoint."""
    monkeypatch.setattr(pgvector_pkg, "_embed_text", lambda *a, **kw: list(FLAT))
    from hermes_pgvector.store import MemoryStore
    p = PgvectorMemoryProvider(config={"dsn": dsn})
    p._store = MemoryStore(dsn)
    p._store.ensure_schema()
    p._healthy = True
    p._agent_identity = "pytest-wpd2-" + os.urandom(4).hex()
    p._session_id = "sess-read"

    yield p

    agent = p._agent_identity
    p.shutdown()
    import psycopg
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_entries WHERE agent_identity = %s", (agent,))
            conn.commit()


def test_queue_prefetch_then_prefetch_returns_the_cached_block_fast(live_read_provider):
    """H4 acceptance: after queue_prefetch() completes, the NEXT prefetch()
    call for that session returns the cached block in < 50ms."""
    p = live_read_provider
    p._store.add(
        agent_identity=p._agent_identity, target="memory",
        content="a seeded note for recall", embedding=list(FLAT),
    )

    p.queue_prefetch("find the seeded note", session_id="sess-read")

    deadline = time.monotonic() + 5.0
    cached = None
    while time.monotonic() < deadline:
        with p._prefetch_lock:
            cached = p._prefetch_cache.get("sess-read")
        if cached is not None:
            break
        time.sleep(0.01)
    else:
        pytest.fail("queue_prefetch did not populate the cache in time")
    assert "a seeded note for recall" in cached

    started = time.monotonic()
    block = p.prefetch("irrelevant text -- the cache must win", session_id="sess-read")
    elapsed = time.monotonic() - started

    assert elapsed < 0.05, f"cached prefetch took {elapsed * 1000:.1f}ms, expected < 50ms"
    assert "a seeded note for recall" in block


def test_system_prompt_block_does_not_requery_the_db_after_a_real_initialize(dsn, tmp_path):
    agent = "pytest-wpd2-sysblock-" + os.urandom(4).hex()
    provider = PgvectorMemoryProvider(config={
        "dsn": dsn, "embed_on_write": False, "bulk_sync_on_init": False,
    })
    try:
        provider.initialize(session_id="s1", agent_identity=agent, hermes_home=str(tmp_path))
        assert provider._healthy is True
        cached_block = provider._system_prompt_block_cache
        assert cached_block is not None

        real_store = provider._store
        calls = {"count": 0, "estimate_count": 0}
        orig_count = real_store.count
        orig_estimate = real_store.estimate_count

        def _spy_count(**kw):
            calls["count"] += 1
            return orig_count(**kw)

        def _spy_estimate(table):
            calls["estimate_count"] += 1
            return orig_estimate(table)

        real_store.count = _spy_count
        real_store.estimate_count = _spy_estimate

        for _ in range(3):
            assert provider.system_prompt_block() == cached_block

        assert calls == {"count": 0, "estimate_count": 0}, (
            f"system_prompt_block() queried the DB after initialize(): {calls}"
        )
    finally:
        provider.shutdown()
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memory_entries WHERE agent_identity = %s", (agent,))
                cur.execute("DELETE FROM memory_agents WHERE agent_identity = %s", (agent,))
                conn.commit()
