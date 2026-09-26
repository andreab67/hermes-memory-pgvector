"""v0.6.0 WP-D1 coverage for hermes_pgvector/__init__.py's write paths.

Findings closed here (docs/code-review/1.0-readiness-review-2026-09-26.md,
docs/release/PLAN-1.0.md Sec 5 WP-D1):

H2  -- on_memory_write() extracts metadata["previous_content"] (the EXACT
       prior entry the built-in store edited -- see
       agent.memory_provider.MemoryProvider.on_memory_write's docstring and
       agent.memory_manager.MemoryManager.notify_memory_tool_write in the
       pinned hermes-agent contract) into extra["previous_content"] and pops
       it out of the metadata that gets persisted. _worker() then makes
       replace()/remove() match that exact content instead of old_text's
       substring LIKE, which can hit a stale lower-id row that merely
       contains the search fragment. Without previous_content, the old
       old_text/LIKE path (including remove()'s refusal of an empty
       pattern) is unchanged.
M3  -- shutdown() sets a provider-level `_draining` flag BEFORE draining the
       writer, so `_maybe_embed()` returns None without ever calling the
       embed endpoint -- the drain is DB-only and fast; rows land text-only
       and the nightly backfill sweep heals them. New `shutdown_drain_timeout`
       config key (default 10.0) is the timeout passed to
       AsyncWriter.shutdown(). The flag is cleared again once a fresh writer
       is built (initialize()), including on a reused provider instance
       where initialize() calls shutdown() internally first.
M4  -- new `write_contexts` config key (default "primary,cron"; comma string
       or YAML list; compared lowercased/stripped). initialize() reads
       kwargs.get("agent_context") (missing/empty -> "primary") and decides
       self._writes_enabled. When the context isn't listed, on_memory_write,
       sync_turn, on_session_end and on_delegation all no-op, and
       initialize() itself skips the MEMORY.md/USER.md bulk import and the
       register_agent enqueue. Recall (prefetch, recall_memory,
       recall_conversation, system_prompt_block) is never gated by this.
M5  -- on_delegation()'s captured turn now sets conversations.parent_session_id
       to THIS row's delegating session (self._parent_session_id or None,
       migration 002's definition), not the child's session id -- the child
       id still lands in metadata.child_session_id and in
       memory_agent_edges.
L9  -- on_session_end()'s turn-capture backstop now runs whenever sync_turns
       is on and the provider is healthy with a writer -- it no longer
       requires migration 002. Only the register_agent enqueue stays gated
       on self._delegation_enabled, and the backstop's own parent_session_id
       is gated the same way sync_turn() gates it (self._parent_session_id
       only when 002 is applied, else None), so it never tries to write a
       column that does not exist on a pre-002 schema.

Two kinds of tests:
  * DB-free, deterministic unit tests against stub writers/stores -- these
    run everywhere and pin the exact dispatch/gating logic without any
    threading or network race.
  * DB-gated acceptance tests (skip without PG_TEST_DSN) that exercise the
    real MemoryStore end to end, including (for H2) driving the exact
    metadata SHAPE agent.memory_manager.MemoryManager.notify_memory_tool_write
    builds -- replicated inline below since hermes-agent is not importable
    from this repo's tests.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_pgvector import (  # noqa: E402
    DEFAULTS,
    PgvectorMemoryProvider,
    _as_write_contexts,
)
from hermes_pgvector.writer import AsyncWriter, _PendingWrite  # noqa: E402


# ---------------------------------------------------------------------------
# Shared stubs (DB-free tests)
# ---------------------------------------------------------------------------

class _CapturingWriter:
    """Writer stand-in that accepts every write and records the full kwargs
    of each enqueue() call, so a test can inspect `extra`/`metadata` exactly
    as the provider built them."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def enqueue(self, **kwargs) -> bool:
        self.calls.append(kwargs)
        return True

    def actions(self, action: str) -> List[Dict[str, Any]]:
        return [c for c in self.calls if c.get("action") == action]


def _bare_provider(writer, **attrs) -> PgvectorMemoryProvider:
    p = PgvectorMemoryProvider()
    p._healthy = True
    p._writer = writer
    p._agent_identity = "default"
    p._raw_identity = "default"
    p._session_id = "sess-1"
    for k, v in attrs.items():
        setattr(p, k, v)
    return p


def _pending(action: str, *, extra=None, content="new content", metadata=None) -> _PendingWrite:
    return _PendingWrite(
        action=action,
        agent_identity="default",
        target="memory",
        content=content,
        extra=extra or {},
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# H2 -- on_memory_write(): previous_content extraction (DB-free)
# ---------------------------------------------------------------------------

def test_on_memory_write_moves_previous_content_from_metadata_to_extra():
    writer = _CapturingWriter()
    p = _bare_provider(writer)
    p.on_memory_write(
        "replace", "memory", "new content",
        metadata={"previous_content": "the exact old row", "session_id": "sess-1", "old_text": "old"},
    )
    assert len(writer.calls) == 1
    call = writer.calls[0]
    assert call["extra"]["previous_content"] == "the exact old row"
    assert "previous_content" not in call["metadata"], (
        "previous_content must never be persisted -- it is the full text of "
        "a row about to be overwritten/deleted, not provenance for the new row"
    )
    assert call["metadata"]["session_id"] == "sess-1"
    assert call["extra"]["old_text"] == "old"


def test_on_memory_write_blank_previous_content_is_dropped_not_forwarded():
    writer = _CapturingWriter()
    p = _bare_provider(writer)
    p.on_memory_write("remove", "memory", "", metadata={"previous_content": "   ", "old_text": "frag"})
    call = writer.calls[0]
    assert "previous_content" not in call["extra"]
    assert "previous_content" not in call["metadata"]
    assert call["extra"]["old_text"] == "frag"


def test_on_memory_write_non_string_previous_content_is_dropped_not_forwarded():
    writer = _CapturingWriter()
    p = _bare_provider(writer)
    p.on_memory_write("remove", "memory", "", metadata={"previous_content": 12345, "old_text": "frag"})
    call = writer.calls[0]
    assert "previous_content" not in call["extra"]
    assert "previous_content" not in call["metadata"]


def test_on_memory_write_missing_previous_content_old_path_unchanged():
    writer = _CapturingWriter()
    p = _bare_provider(writer)
    p.on_memory_write("replace", "memory", "new content", metadata={"old_text": "frag"})
    call = writer.calls[0]
    assert "previous_content" not in call["extra"]
    assert call["extra"]["old_text"] == "frag"


# ---------------------------------------------------------------------------
# H2 -- _worker() dispatch: exact_content vs old_text/LIKE (DB-free)
# ---------------------------------------------------------------------------

class _FakeStoreReplace:
    def __init__(self, replace_returns: int):
        self.replace_calls: List[Dict[str, Any]] = []
        self.add_calls: List[Dict[str, Any]] = []
        self._n = replace_returns

    def replace(self, **kwargs):
        self.replace_calls.append(kwargs)
        return self._n

    def add(self, **kwargs):
        self.add_calls.append(kwargs)
        return 1


def test_worker_replace_uses_exact_content_when_previous_content_present():
    store = _FakeStoreReplace(replace_returns=1)
    p = PgvectorMemoryProvider(config={"embed_on_write": False})
    p._store = store
    p._worker(_pending("replace", extra={"previous_content": "the exact old text", "old_text": "old"}))
    assert len(store.replace_calls) == 1
    call = store.replace_calls[0]
    assert call["exact_content"] == "the exact old text"
    assert "old_text" not in call
    assert store.add_calls == [], "must not degrade to add when the exact match succeeded"


def test_worker_replace_degrades_to_add_when_exact_content_matches_nothing():
    store = _FakeStoreReplace(replace_returns=0)
    p = PgvectorMemoryProvider(config={"embed_on_write": False})
    p._store = store
    p._worker(_pending("replace", extra={"previous_content": "text nowhere in the table"}))
    assert len(store.replace_calls) == 1
    assert len(store.add_calls) == 1


def test_worker_replace_without_previous_content_falls_back_to_old_text_like():
    store = _FakeStoreReplace(replace_returns=1)
    p = PgvectorMemoryProvider(config={"embed_on_write": False})
    p._store = store
    p._worker(_pending("replace", extra={"old_text": "old fragment"}))
    call = store.replace_calls[0]
    assert call["old_text"] == "old fragment"
    assert "exact_content" not in call


def test_worker_replace_without_old_text_or_previous_content_adds():
    store = _FakeStoreReplace(replace_returns=1)
    p = PgvectorMemoryProvider(config={"embed_on_write": False})
    p._store = store
    p._worker(_pending("replace", extra={}))
    assert store.replace_calls == []
    assert len(store.add_calls) == 1


class _FakeStoreRemove:
    def __init__(self, remove_returns: int):
        self.remove_calls: List[Dict[str, Any]] = []
        self._n = remove_returns

    def remove(self, **kwargs):
        self.remove_calls.append(kwargs)
        return self._n


def test_worker_remove_uses_exact_content_and_never_falls_back_to_like_on_zero_match():
    store = _FakeStoreRemove(remove_returns=0)
    p = PgvectorMemoryProvider(config={"embed_on_write": False})
    p._store = store
    p._worker(_pending("remove", content="", extra={"previous_content": "exact text", "old_text": "frag"}))
    assert len(store.remove_calls) == 1, "must not retry via old_text/LIKE after a zero-match exact lookup"
    call = store.remove_calls[0]
    assert call["exact_content"] == "exact text"
    assert "old_text" not in call


def test_worker_remove_without_previous_content_uses_old_text():
    store = _FakeStoreRemove(remove_returns=1)
    p = PgvectorMemoryProvider(config={"embed_on_write": False})
    p._store = store
    p._worker(_pending("remove", content="", extra={"old_text": "frag"}))
    call = store.remove_calls[0]
    assert call["old_text"] == "frag"
    assert "exact_content" not in call


def test_worker_remove_without_previous_content_or_old_text_refuses():
    store = _FakeStoreRemove(remove_returns=0)
    p = PgvectorMemoryProvider(config={"embed_on_write": False})
    p._store = store
    p._worker(_pending("remove", content="", extra={}))
    assert store.remove_calls == [], "an empty pattern must never reach remove() (would match every row)"


# ---------------------------------------------------------------------------
# H2 -- DB-gated acceptance: drive the real upstream metadata SHAPE.
#
# agent.memory_manager.MemoryManager.notify_memory_tool_write (pinned
# hermes-agent contract read for this WP) is replicated here byte-for-byte --
# it cannot be imported from this repo's tests. See the module docstring.
# ---------------------------------------------------------------------------

_MIRRORED_MEMORY_ACTIONS = {"add", "replace", "remove"}


def _memory_tool_result_succeeded(result: Any) -> bool:
    if isinstance(result, str):
        import json
        result = json.loads(result)
    return isinstance(result, dict) and result.get("success") is True and result.get("staged") is not True


def _notify_memory_tool_write_like_upstream(
    provider: PgvectorMemoryProvider,
    tool_result: Any,
    tool_args: Dict[str, Any],
    build_metadata=None,
) -> None:
    """Exact replica of MemoryManager.notify_memory_tool_write's metadata
    construction (single-op with replaced_entry/removed_entry; batched
    `operations` with replaced_entries/removed_entries keyed by 1-based
    index) -- calls provider.on_memory_write() exactly as upstream does."""
    import json as _json

    if not _memory_tool_result_succeeded(tool_result):
        return
    result = _json.loads(tool_result) if isinstance(tool_result, str) else tool_result
    target = str(tool_args.get("target") or "memory")
    operations = tool_args.get("operations")
    batched = isinstance(operations, list) and bool(operations)
    for index, op in enumerate(operations if batched else [tool_args], start=1):
        action = str(op.get("action") or "") if isinstance(op, dict) else ""
        if action not in _MIRRORED_MEMORY_ACTIONS:
            continue
        metadata = dict(build_metadata() if build_metadata else {})
        metadata.pop("previous_content", None)
        old_text = op.get("old_text")
        if old_text:
            metadata["old_text"] = str(old_text)
        field = {"replace": "replaced", "remove": "removed"}.get(action)
        if field:
            if batched:
                entries = result.get(f"{field}_entries", {})
                previous = entries.get(str(index), entries.get(index)) if isinstance(entries, dict) else None
            else:
                previous = result.get(f"{field}_entry")
            if isinstance(previous, str) and previous:
                metadata["previous_content"] = previous
        provider.on_memory_write(
            action, target, str(op.get("content") or op.get("new_text") or ""), metadata=metadata
        )


@pytest.fixture
def dsn():
    d = os.environ.get("PG_TEST_DSN")
    if not d:
        pytest.skip("PG_TEST_DSN not set")
    return d


@pytest.fixture
def live_provider(dsn):
    """A real, healthy provider (real MemoryStore + real AsyncWriter running
    its actual _worker) scoped to one random agent_identity. shutdown() at
    teardown drains + closes; rows are deleted for this identity only."""
    p = PgvectorMemoryProvider(config={"dsn": dsn, "embed_on_write": False})
    from hermes_pgvector.store import MemoryStore
    p._store = MemoryStore(dsn)
    p._store.ensure_schema()
    p._healthy = True
    p._delegation_enabled = p._store.ensure_migration_002_applied()
    p._agent_identity = "pytest-wpd1-" + os.urandom(4).hex()
    p._raw_identity = p._agent_identity
    p._session_id = "sess-live"
    p._writes_enabled = True
    p._writer = AsyncWriter(p._worker, maxsize=64)

    yield p

    agent = p._agent_identity
    p.shutdown()
    import psycopg
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_entries WHERE agent_identity = %s", (agent,))
            cur.execute("DELETE FROM conversations WHERE agent_identity = %s", (agent,))
            cur.execute("DELETE FROM memory_agents WHERE agent_identity = %s", (agent,))
            cur.execute(
                "DELETE FROM memory_agent_edges WHERE parent_identity = %s OR child_identity = %s",
                (agent, agent),
            )
            conn.commit()


def test_replace_via_upstream_shape_updates_the_real_row_not_a_stale_substring_match(live_provider):
    """A stale row (lower id) merely CONTAINS the search fragment; the real
    row (higher id) is what the built-in store actually edited. old_text's
    substring LIKE alone would hit the stale row first (ORDER BY id LIMIT 1)
    -- previous_content must steer replace() to the real one instead."""
    p = live_provider
    stale_id = p._store.add(
        agent_identity=p._agent_identity, target="memory",
        content="shared frag token A: irrelevant stale note",
    )
    real_id = p._store.add(
        agent_identity=p._agent_identity, target="memory",
        content="shared frag token A: the actual current entry",
    )
    assert stale_id < real_id

    tool_args = {
        "target": "memory", "action": "replace",
        "old_text": "frag token A", "content": "shared frag token A: UPDATED entry",
    }
    tool_result = {"success": True, "replaced_entry": "shared frag token A: the actual current entry"}
    _notify_memory_tool_write_like_upstream(p, tool_result, tool_args)
    p._writer.shutdown(timeout=5.0)

    rows = {r["id"]: r["content"] for r in p._store.list_entries(agent_identity=p._agent_identity, limit=10)}
    assert rows[stale_id] == "shared frag token A: irrelevant stale note", "the stale row must be untouched"
    assert rows[real_id] == "shared frag token A: UPDATED entry", "the real row must be the one updated"


def test_remove_via_upstream_shape_deletes_the_real_row_not_a_stale_substring_match(live_provider):
    p = live_provider
    stale_id = p._store.add(
        agent_identity=p._agent_identity, target="memory",
        content="shared frag token B: irrelevant stale note 2",
    )
    real_id = p._store.add(
        agent_identity=p._agent_identity, target="memory",
        content="shared frag token B: the actual entry to remove",
    )
    assert stale_id < real_id

    tool_args = {"target": "memory", "action": "remove", "old_text": "frag token B"}
    tool_result = {"success": True, "removed_entry": "shared frag token B: the actual entry to remove"}
    _notify_memory_tool_write_like_upstream(p, tool_result, tool_args)
    p._writer.shutdown(timeout=5.0)

    rows = {r["id"]: r["content"] for r in p._store.list_entries(agent_identity=p._agent_identity, limit=10)}
    assert stale_id in rows, "the stale row must survive"
    assert real_id not in rows, "the real row must be the one deleted"


def test_batched_operations_shape_uses_indexed_previous_content_for_replace_and_remove(live_provider):
    """The batched `operations` shape keys replaced_entries/removed_entries by
    1-based op index, not by content -- exercise both actions in one call,
    each with its own stale-substring trap."""
    p = live_provider
    stale_r = p._store.add(agent_identity=p._agent_identity, target="memory",
                            content="shared frag token A: irrelevant stale note")
    real_r = p._store.add(agent_identity=p._agent_identity, target="memory",
                           content="shared frag token A: the actual current entry")
    stale_d = p._store.add(agent_identity=p._agent_identity, target="memory",
                            content="shared frag token B: irrelevant stale note 2")
    real_d = p._store.add(agent_identity=p._agent_identity, target="memory",
                           content="shared frag token B: the actual entry to remove")

    tool_args = {
        "target": "memory",
        "operations": [
            {"action": "replace", "old_text": "frag token A", "content": "shared frag token A: UPDATED entry"},
            {"action": "remove", "old_text": "frag token B"},
        ],
    }
    tool_result = {
        "success": True,
        "replaced_entries": {"1": "shared frag token A: the actual current entry"},
        "removed_entries": {"2": "shared frag token B: the actual entry to remove"},
    }
    _notify_memory_tool_write_like_upstream(p, tool_result, tool_args)
    p._writer.shutdown(timeout=5.0)

    rows = {r["id"]: r["content"] for r in p._store.list_entries(agent_identity=p._agent_identity, limit=10)}
    assert rows[stale_r] == "shared frag token A: irrelevant stale note"
    assert rows[real_r] == "shared frag token A: UPDATED entry"
    assert stale_d in rows
    assert real_d not in rows


def test_previous_content_never_persisted_when_replace_degrades_to_add(live_provider):
    """When previous_content matches nothing, replace() degrades to add() --
    the row that lands must not carry previous_content in its metadata."""
    p = live_provider
    tool_args = {
        "target": "memory", "action": "replace",
        "old_text": "no such fragment anywhere", "content": "brand-new entry via degrade path",
    }
    tool_result = {"success": True, "replaced_entry": "text that matches no row at all"}

    def build_metadata():
        return {"session_id": "sess-live", "tool_name": "memory"}

    _notify_memory_tool_write_like_upstream(p, tool_result, tool_args, build_metadata=build_metadata)
    p._writer.shutdown(timeout=5.0)

    rows = p._store.list_entries(agent_identity=p._agent_identity, limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["content"] == "brand-new entry via degrade path"
    assert "previous_content" not in (row.get("metadata") or {}), (
        "previous_content must never reach the persisted row"
    )
    assert row["metadata"].get("session_id") == "sess-live"


# ---------------------------------------------------------------------------
# M3 -- shutdown() sets `_draining` before draining (DB-free, deterministic)
# ---------------------------------------------------------------------------

def test_shutdown_sets_draining_flag_before_the_writer_finishes_draining():
    """Adapted from tests/test_v060_identity_writer.py's AsyncWriter.draining
    coverage, but pinning the PROVIDER's own `_draining` flag (M3 requires a
    provider-level flag; AsyncWriter.draining alone is not enough because
    _maybe_embed() lives on the provider, not the writer)."""
    release = threading.Event()
    first_item_seen = threading.Event()
    seen: List[bool] = []

    p = PgvectorMemoryProvider()
    p._healthy = True

    def slow_worker(item):
        if not first_item_seen.is_set():
            first_item_seen.set()
            release.wait(timeout=5.0)
        seen.append(p._draining)

    p._writer = AsyncWriter(slow_worker, maxsize=8)
    assert p._draining is False

    assert p._writer.enqueue(action="turn", agent_identity="a", target="conversations", content="item-0")
    first_item_seen.wait(timeout=5.0)
    assert p._draining is False  # shutdown() has not run yet

    shutdown_thread = threading.Thread(target=p.shutdown)
    shutdown_thread.start()

    deadline = time.monotonic() + 2.0
    while not p._draining and time.monotonic() < deadline:
        time.sleep(0.01)
    assert p._draining is True  # set synchronously before/while draining

    release.set()
    shutdown_thread.join(timeout=5.0)
    assert not shutdown_thread.is_alive()
    assert seen == [True]
    assert p._writer is None  # shutdown() cleared it


def test_maybe_embed_returns_none_immediately_while_draining(fake_embed_server):
    """The endpoint would take 5s to answer if actually called; _draining
    must short-circuit BEFORE any network call, so this returns in well
    under that."""
    url = fake_embed_server(delay=5, dim=8)
    p = PgvectorMemoryProvider(config={"embed_url": url, "embed_dim": 8, "embed_write_timeout": 1.0})
    p._draining = True

    start = time.monotonic()
    vec = p._maybe_embed("some content that would otherwise get embedded")
    elapsed = time.monotonic() - start

    assert vec is None
    assert elapsed < 1.0, f"must short-circuit before calling the embed endpoint, took {elapsed:.2f}s"


def test_maybe_embed_still_embeds_normally_when_not_draining(fake_embed_server):
    url = fake_embed_server(dim=8)
    p = PgvectorMemoryProvider(config={"embed_url": url, "embed_dim": 8})
    p._draining = False
    vec = p._maybe_embed("hello world")
    assert isinstance(vec, list) and len(vec) == 8


def test_maybe_embed_draining_check_precedes_embed_on_write_check(fake_embed_server):
    """Sanity: draining short-circuits even when embed_on_write is true (the
    normal case) -- not an accidental interaction with that other gate."""
    url = fake_embed_server(delay=5, dim=8)
    p = PgvectorMemoryProvider(config={"embed_url": url, "embed_dim": 8, "embed_on_write": True})
    p._draining = True
    assert p._maybe_embed("content") is None


# ---------------------------------------------------------------------------
# M3 -- shutdown_drain_timeout config key (DB-free)
# ---------------------------------------------------------------------------

class _TimeoutSpyWriter:
    def __init__(self):
        self.timeouts: List[float] = []

    def shutdown(self, timeout: float = 5.0) -> None:
        self.timeouts.append(timeout)


def test_shutdown_passes_configured_drain_timeout_to_the_writer():
    p = PgvectorMemoryProvider(config={"shutdown_drain_timeout": 0.25})
    spy = _TimeoutSpyWriter()
    p._writer = spy
    p.shutdown()
    assert spy.timeouts == [0.25]


def test_shutdown_uses_the_default_drain_timeout_when_config_key_is_unset():
    p = PgvectorMemoryProvider()
    spy = _TimeoutSpyWriter()
    p._writer = spy
    p.shutdown()
    assert spy.timeouts == [DEFAULTS["shutdown_drain_timeout"]]


def test_shutdown_drain_timeout_accepts_a_string_config_value():
    """Invariant #9: config values arrive as strings from save_config()."""
    p = PgvectorMemoryProvider(config={"shutdown_drain_timeout": "3.5"})
    spy = _TimeoutSpyWriter()
    p._writer = spy
    p.shutdown()
    assert spy.timeouts == [3.5]


# ---------------------------------------------------------------------------
# M3 -- the flag is cleared again once a fresh writer is built (DB-free: a
# deliberately unreachable DSN keeps initialize() fast; the flag reset
# happens unconditionally right after the new AsyncWriter is constructed,
# regardless of DB health).
# ---------------------------------------------------------------------------

_BOGUS_DSN = "host=127.0.0.1 port=1 dbname=nonexistent connect_timeout=1"


def test_initialize_clears_draining_flag_for_the_new_writer():
    p = PgvectorMemoryProvider(config={"dsn": _BOGUS_DSN})
    p._draining = True  # simulate a leftover flag from some earlier state
    p.initialize(session_id="s1", agent_identity="tester", hermes_home="/nonexistent")
    assert p._draining is False
    p.shutdown()


def test_reinitialize_on_a_reused_provider_clears_draining_after_internal_shutdown():
    """initialize() calls self.shutdown() first when the instance already has
    a store/writer (the gateway reuses one provider across sessions) -- that
    internal shutdown() sets _draining=True to fast-drain the OLD writer.
    Once the NEW writer is built for this session, the flag must be cleared,
    or every write this new session enqueues would silently skip embedding."""
    p = PgvectorMemoryProvider(config={"dsn": _BOGUS_DSN})
    p.initialize(session_id="s1", agent_identity="tester", hermes_home="/nonexistent")
    assert p._draining is False

    p.initialize(session_id="s2", agent_identity="tester", hermes_home="/nonexistent")
    assert p._draining is False
    p.shutdown()


# ---------------------------------------------------------------------------
# M4 -- _as_write_contexts() coercion (DB-free)
# ---------------------------------------------------------------------------

def test_as_write_contexts_accepts_comma_string():
    assert _as_write_contexts("primary,cron") == ["primary", "cron"]


def test_as_write_contexts_trims_and_lowercases():
    assert _as_write_contexts(" Primary , CRON ") == ["primary", "cron"]


def test_as_write_contexts_accepts_a_yaml_list():
    assert _as_write_contexts(["primary", "Subagent"]) == ["primary", "subagent"]


def test_as_write_contexts_none_falls_back_to_default():
    assert _as_write_contexts(None) == ["primary", "cron"]


def test_as_write_contexts_empty_string_falls_back_to_default():
    assert _as_write_contexts("") == ["primary", "cron"]


def test_as_write_contexts_empty_list_falls_back_to_default():
    assert _as_write_contexts([]) == ["primary", "cron"]


def test_as_write_contexts_malformed_value_falls_back_to_default():
    assert _as_write_contexts(12345) == ["primary", "cron"]


# ---------------------------------------------------------------------------
# M4 -- initialize()'s agent_context resolution (DB-free: bogus DSN keeps it
# fast; the write_contexts decision happens before the DB is even touched).
# ---------------------------------------------------------------------------

def _init_and_get_gate(config=None, **kwargs):
    p = PgvectorMemoryProvider(config={"dsn": _BOGUS_DSN, **(config or {})})
    p.initialize(session_id="s1", agent_identity="tester", hermes_home="/nonexistent", **kwargs)
    gate = (p._agent_context, p._writes_enabled)
    p.shutdown()
    return gate


def test_missing_agent_context_kwarg_defaults_to_primary_and_enables_writes():
    assert _init_and_get_gate() == ("primary", True)


def test_empty_string_agent_context_defaults_to_primary():
    assert _init_and_get_gate(agent_context="") == ("primary", True)


def test_agent_context_primary_enables_writes():
    assert _init_and_get_gate(agent_context="primary") == ("primary", True)


def test_agent_context_cron_enables_writes():
    assert _init_and_get_gate(agent_context="cron") == ("cron", True)


def test_agent_context_subagent_disables_writes():
    assert _init_and_get_gate(agent_context="subagent") == ("subagent", False)


def test_agent_context_flush_disables_writes():
    assert _init_and_get_gate(agent_context="flush") == ("flush", False)


def test_agent_context_is_case_and_whitespace_insensitive():
    assert _init_and_get_gate(agent_context=" PRIMARY ") == ("primary", True)


def test_write_contexts_config_as_yaml_list_allows_subagent():
    gate = _init_and_get_gate(config={"write_contexts": ["primary", "subagent"]}, agent_context="subagent")
    assert gate == ("subagent", True)


def test_write_contexts_config_as_comma_string_narrowed_to_primary_only():
    gate = _init_and_get_gate(config={"write_contexts": "primary"}, agent_context="cron")
    assert gate == ("cron", False)


# ---------------------------------------------------------------------------
# M4 -- write hooks no-op entirely when the context is excluded (DB-free).
# ---------------------------------------------------------------------------

def test_on_memory_write_noop_when_writes_disabled():
    writer = _CapturingWriter()
    p = _bare_provider(writer, _writes_enabled=False)
    p.on_memory_write("add", "memory", "some content", metadata={})
    assert writer.calls == []


def test_sync_turn_noop_when_writes_disabled():
    writer = _CapturingWriter()
    p = _bare_provider(writer, _writes_enabled=False, _turn_fingerprints=set())
    p.sync_turn(
        "a substantive user turn well past the noise threshold",
        "a substantive assistant reply, also long enough",
    )
    assert writer.calls == []


def test_on_session_end_noop_when_writes_disabled():
    writer = _CapturingWriter()
    p = _bare_provider(
        writer, _writes_enabled=False, _delegation_enabled=True, _turn_fingerprints=set(),
    )
    p.on_session_end([
        {"role": "user", "content": "a substantive user turn well past the noise threshold"},
        {"role": "assistant", "content": "a substantive assistant reply, also long enough"},
    ])
    assert writer.calls == [], "register_agent must also be skipped, not just the turn capture"


def test_on_delegation_noop_when_writes_disabled():
    writer = _CapturingWriter()
    p = _bare_provider(writer, _writes_enabled=False, _delegation_enabled=True)
    p.on_delegation("do the thing", "did the thing, a long enough result", child_session_id="child-1")
    assert writer.calls == []


# ---------------------------------------------------------------------------
# M4 -- recall paths are NEVER gated by write_contexts (DB-free: fake store +
# in-process fake embed server, no Postgres needed).
# ---------------------------------------------------------------------------

class _FakeRecallStore:
    def __init__(self, rows):
        self._rows = rows

    def search(self, **kwargs):
        return list(self._rows)

    def hybrid_search(self, **kwargs):
        return list(self._rows)

    def count(self, **kwargs):
        return len(self._rows)


def test_prefetch_still_works_when_writes_are_disabled(fake_embed_server):
    url = fake_embed_server(dim=8)
    p = PgvectorMemoryProvider(config={"embed_url": url, "embed_dim": 8})
    p._healthy = True
    p._writes_enabled = False
    p._agent_identity = "default"
    p._store = _FakeRecallStore([{"content": "a recalled note", "target": "memory", "score": 0.9}])

    block = p.prefetch("query text")
    assert "a recalled note" in block


def test_recall_memory_tool_still_works_when_writes_are_disabled(fake_embed_server):
    import json as _json
    url = fake_embed_server(dim=8)
    p = PgvectorMemoryProvider(config={"embed_url": url, "embed_dim": 8, "hybrid_search": False})
    p._healthy = True
    p._writes_enabled = False
    p._agent_identity = "default"
    p._store = _FakeRecallStore([{"content": "a recalled note", "target": "memory", "score": 0.9, "id": 1}])

    out = _json.loads(p.handle_tool_call("recall_memory", {"query": "query text"}))
    assert out["count"] == 1
    assert out["results"][0]["content"] == "a recalled note"


def test_system_prompt_block_still_works_when_writes_are_disabled():
    p = PgvectorMemoryProvider()
    p._healthy = True
    p._writes_enabled = False
    p._agent_identity = "default"
    p._store = _FakeRecallStore([{"content": "x"}])

    block = p.system_prompt_block()
    assert "pgvector memory" in block


# ---------------------------------------------------------------------------
# M4 -- DB-gated: excluded contexts skip the bulk import + register_agent;
# allowed contexts run both.
# ---------------------------------------------------------------------------

@pytest.fixture
def memories_home(tmp_path):
    memories_dir = tmp_path / "memories"
    memories_dir.mkdir()
    (memories_dir / "MEMORY.md").write_text("a durable note from disk", encoding="utf-8")
    return tmp_path


def _cleanup_agent(dsn_value: str, agent: str) -> None:
    import psycopg
    with psycopg.connect(dsn_value) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_entries WHERE agent_identity = %s", (agent,))
            cur.execute("DELETE FROM conversations WHERE agent_identity = %s", (agent,))
            cur.execute("DELETE FROM memory_agents WHERE agent_identity = %s", (agent,))
            cur.execute(
                "DELETE FROM memory_agent_edges WHERE parent_identity = %s OR child_identity = %s",
                (agent, agent),
            )
            conn.commit()


@pytest.mark.parametrize("context", ["subagent", "flush"])
def test_excluded_context_skips_bulk_import_and_register_agent(dsn, memories_home, context):
    agent = "pytest-m4-" + os.urandom(4).hex()
    provider = PgvectorMemoryProvider(config={"dsn": dsn, "embed_on_write": False})
    try:
        provider.initialize(
            session_id="s1", agent_identity=agent, hermes_home=str(memories_home), agent_context=context,
        )
        assert provider._writes_enabled is False
        assert provider._store.count(agent_identity=agent) == 0, "bulk import must be skipped"

        provider.shutdown()
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM memory_agents WHERE agent_identity = %s", (agent,))
                assert cur.fetchone()[0] == 0, "register_agent must be skipped"
    finally:
        _cleanup_agent(dsn, agent)


@pytest.mark.parametrize("context", ["primary", "cron"])
def test_allowed_context_runs_bulk_import_and_register_agent(dsn, memories_home, context):
    agent = "pytest-m4-" + os.urandom(4).hex()
    provider = PgvectorMemoryProvider(config={"dsn": dsn, "embed_on_write": False})
    try:
        provider.initialize(
            session_id="s1", agent_identity=agent, hermes_home=str(memories_home), agent_context=context,
        )
        assert provider._writes_enabled is True
        assert provider._store.count(agent_identity=agent) == 1, "bulk import should have run"

        provider.shutdown()
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM memory_agents WHERE agent_identity = %s", (agent,))
                registered = cur.fetchone()[0]
        # register_agent additionally requires migration 002; on a fully
        # migrated test DB (the harness applies every migration) it must land.
        assert registered == 1
    finally:
        _cleanup_agent(dsn, agent)


# ---------------------------------------------------------------------------
# M5 -- on_delegation(): conversations.parent_session_id is the DELEGATING
# session, not the child's (DB-free enqueue-shape check).
# ---------------------------------------------------------------------------

def test_on_delegation_turn_parent_session_id_is_the_delegating_session_not_the_child():
    writer = _CapturingWriter()
    p = _bare_provider(
        writer, _delegation_enabled=True, _writes_enabled=True,
        _session_id="parent-sess-1", _parent_session_id="grandparent-sess-0",
    )
    p.on_delegation("do the thing", "did the thing, a long enough result", child_session_id="child-sess-99")

    turn_calls = writer.actions("turn")
    assert len(turn_calls) == 1
    extra = turn_calls[0]["extra"]
    assert extra["parent_session_id"] == "grandparent-sess-0", (
        "must be self._parent_session_id (the session that delegated to THIS "
        "agent), not the child's session id -- that was the pre-v0.6.0 bug"
    )
    metadata = turn_calls[0]["metadata"]
    assert metadata["child_session_id"] == "child-sess-99", "child id still belongs in metadata"

    edge_calls = writer.actions("edge")
    assert len(edge_calls) == 1
    assert edge_calls[0]["extra"]["parent_session_id"] == "parent-sess-1"  # this agent's own session, unaffected
    assert edge_calls[0]["extra"]["child_session_id"] == "child-sess-99"


def test_on_delegation_turn_parent_session_id_is_none_without_delegation_lineage():
    writer = _CapturingWriter()
    p = _bare_provider(
        writer, _delegation_enabled=True, _writes_enabled=True,
        _session_id="parent-sess-1", _parent_session_id=None,
    )
    p.on_delegation("task", "a long enough result text", child_session_id="child-sess-1")

    turn_calls = writer.actions("turn")
    assert turn_calls[0]["extra"]["parent_session_id"] is None


def test_on_delegation_persists_parent_session_id_as_the_delegating_session(live_provider):
    """End-to-end: the row that actually lands in `conversations` has
    parent_session_id set to the delegating session, and the child id only
    in metadata."""
    p = live_provider
    p._session_id = "parent-sess-live"
    p._parent_session_id = "grandparent-sess-live"

    p.on_delegation(
        "investigate the flaky test", "root-caused: a race in the drain thread",
        child_session_id="child-sess-live",
    )
    p._writer.shutdown(timeout=5.0)

    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(p._store._dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT parent_session_id, metadata FROM conversations WHERE agent_identity = %s",
                (p._agent_identity,),
            )
            row = cur.fetchone()
    assert row is not None
    assert row["parent_session_id"] == "grandparent-sess-live"
    assert row["metadata"].get("child_session_id") == "child-sess-live"


# ---------------------------------------------------------------------------
# L9 -- on_session_end() backstop no longer requires migration 002; only
# register_agent stays gated on it (DB-free, via _delegation_enabled=False).
# ---------------------------------------------------------------------------

_LONG_U = "a substantive user turn well past the noise threshold for capture"
_LONG_A = "a substantive assistant reply, also long enough to be captured here"
_MSGS = [{"role": "user", "content": _LONG_U}, {"role": "assistant", "content": _LONG_A}]


def test_on_session_end_backstop_runs_without_migration_002():
    writer = _CapturingWriter()
    p = _bare_provider(
        writer, _delegation_enabled=False, _writes_enabled=True, _turn_fingerprints=set(),
    )
    p.on_session_end(_MSGS)

    turn_calls = writer.actions("turn")
    assert len(turn_calls) == 2, "the backstop must capture turns even without migration 002"
    register_calls = writer.actions("register_agent")
    assert register_calls == [], "register_agent stays gated on migration 002"


def test_on_session_end_backstop_parent_session_id_is_none_without_migration_002():
    writer = _CapturingWriter()
    p = _bare_provider(
        writer, _delegation_enabled=False, _writes_enabled=True, _turn_fingerprints=set(),
        _parent_session_id="some-parent-session",
    )
    p.on_session_end(_MSGS)

    for call in writer.actions("turn"):
        assert call["extra"]["parent_session_id"] is None, (
            "writing self._parent_session_id unconditionally here would try "
            "to INSERT into a column that does not exist on a pre-002 schema"
        )


def test_on_session_end_backstop_parent_session_id_is_set_with_migration_002():
    writer = _CapturingWriter()
    p = _bare_provider(
        writer, _delegation_enabled=True, _writes_enabled=True, _turn_fingerprints=set(),
        _parent_session_id="some-parent-session",
    )
    p.on_session_end(_MSGS)

    for call in writer.actions("turn"):
        assert call["extra"]["parent_session_id"] == "some-parent-session"
    assert len(writer.actions("register_agent")) == 1


def test_on_session_end_backstop_disabled_by_sync_turns_off():
    writer = _CapturingWriter()
    p = _bare_provider(
        writer, _delegation_enabled=False, _writes_enabled=True, _turn_fingerprints=set(),
    )
    p._config = {**p._config, "sync_turns": False}
    p.on_session_end(_MSGS)
    assert writer.actions("turn") == []


def test_on_session_end_backstop_persists_turns_without_migration_002(live_provider):
    """End to end: with _delegation_enabled simulated False (pre-002), the
    backstop still inserts real rows into `conversations`, and they carry
    parent_session_id = NULL (the column may not even exist on a genuinely
    pre-002 schema; append_turn() only references it when not None)."""
    p = live_provider
    p._delegation_enabled = False
    p._parent_session_id = "would-be-parent"

    p.on_session_end(_MSGS)
    p._writer.shutdown(timeout=5.0)

    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(p._store._dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content, parent_session_id FROM conversations "
                "WHERE agent_identity = %s ORDER BY id",
                (p._agent_identity,),
            )
            persisted = list(cur.fetchall())
            cur.execute(
                "SELECT count(*) AS n FROM memory_agents WHERE agent_identity = %s",
                (p._agent_identity,),
            )
            registered = cur.fetchone()["n"]
    assert len(persisted) == 2
    for row in persisted:
        assert row["parent_session_id"] is None
    assert registered == 0, "register_agent must stay gated on migration 002"
