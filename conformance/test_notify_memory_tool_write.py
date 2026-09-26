"""WP-F conformance test 4 (docs/release/PLAN-1.0.md Sec 5 WP-F):

MemoryManager.notify_memory_tool_write, with a single-op and a batched
`operations` result, delivers `previous_content` to the provider, and the
provider's mirror edits the exact row. DB-backed; skips without PG_TEST_DSN.

Unlike tests/test_v060_provider_writes.py's DB-gated tests (which replicate
notify_memory_tool_write's metadata construction inline, because
hermes-agent is not importable from tests/), this drives the REAL
agent.memory_manager.MemoryManager end to end: add_provider() ->
notify_memory_tool_write(tool_result, tool_args) -> provider.on_memory_write
-> AsyncWriter -> MemoryStore. That is the actual conformance property WP-F
exists to protect: if upstream ever changes notify_memory_tool_write's
metadata shape (e.g. how replaced_entries/removed_entries are keyed, or
what "success" requires), this test fails even though
tests/test_v060_provider_writes.py's hand-written replica would not.

Each test seeds a STALE row (lower id) that merely CONTAINS the search
fragment as a substring, plus the REAL row (higher id) that the built-in
memory tool actually edited. old_text's substring LIKE alone would match
the stale row first (ORDER BY id LIMIT 1); only previous_content -- which
notify_memory_tool_write extracts from the tool result, not from the
caller's search argument -- steers replace()/remove() to the real one.
"""

from __future__ import annotations

import os

import pytest

agent_memory_provider = pytest.importorskip("agent.memory_provider")
agent_memory_manager = pytest.importorskip("agent.memory_manager")

import hermes_pgvector  # noqa: E402
from hermes_pgvector.store import MemoryStore  # noqa: E402

MemoryManager = agent_memory_manager.MemoryManager


@pytest.fixture
def dsn():
    d = os.environ.get("PG_TEST_DSN")
    if not d:
        pytest.skip("PG_TEST_DSN not set")
    return d


@pytest.fixture
def live_manager(dsn, tmp_path):
    """A real upstream MemoryManager with a real, healthy pgvector provider
    registered and initialized against PG_TEST_DSN, scoped to one random
    agent_identity so parallel test runs never collide. Rows for that
    identity are deleted at teardown; the writer is drained first so a
    slow-to-land write can't outlive its own test's cleanup.
    """
    identity = "pytest-wpf-" + os.urandom(4).hex()
    provider = hermes_pgvector.PgvectorMemoryProvider(
        config={"dsn": dsn, "embed_on_write": False, "bulk_sync_on_init": False}
    )
    manager = MemoryManager()
    manager.add_provider(provider)  # real upstream add_provider(): loads get_tool_schemas() etc.
    provider.initialize(
        session_id="conformance-write-path",
        hermes_home=str(tmp_path),
        agent_identity=identity,
    )
    assert provider._healthy, "provider failed to initialize against PG_TEST_DSN"
    assert provider._agent_identity == identity

    yield manager, provider

    # shutdown() itself calls writer.shutdown(timeout=...) again on the same
    # (already-drained, already-stopped) writer -- a safe no-op -- then
    # closes the store.
    provider.shutdown()
    import psycopg

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_entries WHERE agent_identity = %s", (identity,))
            conn.commit()


def _drain(provider) -> None:
    """Wait for the async writer to finish the writes notify_memory_tool_write
    just enqueued, without tearing the provider all the way down (the store
    stays open so the test can still query it through provider._store)."""
    provider._writer.shutdown(timeout=5.0)


def test_single_op_replace_uses_previous_content_not_a_stale_substring_match(live_manager):
    manager, provider = live_manager
    store: MemoryStore = provider._store
    identity = provider._agent_identity

    stale_id = store.add(agent_identity=identity, target="memory",
                          content="shared frag token A: irrelevant stale note")
    real_id = store.add(agent_identity=identity, target="memory",
                         content="shared frag token A: the actual current entry")
    assert stale_id < real_id

    tool_result = {"success": True, "replaced_entry": "shared frag token A: the actual current entry"}
    tool_args = {
        "target": "memory", "action": "replace",
        "old_text": "frag token A", "content": "shared frag token A: UPDATED entry",
    }
    manager.notify_memory_tool_write(tool_result, tool_args)
    _drain(provider)

    rows = {r["id"]: r["content"] for r in store.list_entries(agent_identity=identity, limit=10)}
    assert rows[stale_id] == "shared frag token A: irrelevant stale note", "the stale row must be untouched"
    assert rows[real_id] == "shared frag token A: UPDATED entry", "the real row must be the one updated"


def test_single_op_remove_uses_previous_content_not_a_stale_substring_match(live_manager):
    manager, provider = live_manager
    store: MemoryStore = provider._store
    identity = provider._agent_identity

    stale_id = store.add(agent_identity=identity, target="memory",
                          content="shared frag token B: irrelevant stale note 2")
    real_id = store.add(agent_identity=identity, target="memory",
                         content="shared frag token B: the actual entry to remove")
    assert stale_id < real_id

    tool_result = {"success": True, "removed_entry": "shared frag token B: the actual entry to remove"}
    tool_args = {"target": "memory", "action": "remove", "old_text": "frag token B"}
    manager.notify_memory_tool_write(tool_result, tool_args)
    _drain(provider)

    rows = {r["id"]: r["content"] for r in store.list_entries(agent_identity=identity, limit=10)}
    assert stale_id in rows, "the stale row must survive"
    assert real_id not in rows, "the real row must be the one deleted"


def test_batched_operations_use_index_keyed_previous_content(live_manager):
    """replaced_entries/removed_entries are keyed by 1-based op index, not by
    content -- exercise both actions in one batched notify_memory_tool_write
    call, each with its own stale-substring trap."""
    manager, provider = live_manager
    store: MemoryStore = provider._store
    identity = provider._agent_identity

    stale_r = store.add(agent_identity=identity, target="memory",
                         content="shared frag token A: irrelevant stale note")
    real_r = store.add(agent_identity=identity, target="memory",
                        content="shared frag token A: the actual current entry")
    stale_d = store.add(agent_identity=identity, target="memory",
                         content="shared frag token B: irrelevant stale note 2")
    real_d = store.add(agent_identity=identity, target="memory",
                        content="shared frag token B: the actual entry to remove")

    tool_result = {
        "success": True,
        "replaced_entries": {"1": "shared frag token A: the actual current entry"},
        "removed_entries": {"2": "shared frag token B: the actual entry to remove"},
    }
    tool_args = {
        "target": "memory",
        "operations": [
            {"action": "replace", "old_text": "frag token A", "content": "shared frag token A: UPDATED entry"},
            {"action": "remove", "old_text": "frag token B"},
        ],
    }
    manager.notify_memory_tool_write(tool_result, tool_args)
    _drain(provider)

    rows = {r["id"]: r["content"] for r in store.list_entries(agent_identity=identity, limit=10)}
    assert rows[stale_r] == "shared frag token A: irrelevant stale note"
    assert rows[real_r] == "shared frag token A: UPDATED entry"
    assert stale_d in rows, "the stale row must survive"
    assert real_d not in rows, "the real row must be the one deleted"


def test_unsuccessful_tool_result_is_never_mirrored(live_manager):
    """_memory_tool_result_succeeded() fails closed: no success=True, or a
    staged (not-yet-committed) write, must never reach on_memory_write."""
    manager, provider = live_manager
    store: MemoryStore = provider._store
    identity = provider._agent_identity

    row_id = store.add(agent_identity=identity, target="memory", content="untouched entry")

    for bad_result in ({"success": False}, {"success": True, "staged": True}, {}, "not json"):
        manager.notify_memory_tool_write(
            bad_result,
            {"target": "memory", "action": "replace", "old_text": "untouched", "content": "SHOULD NOT LAND"},
        )
    _drain(provider)

    rows = {r["id"]: r["content"] for r in store.list_entries(agent_identity=identity, limit=10)}
    assert rows[row_id] == "untouched entry"
    assert len(rows) == 1, "no new row should have been added either"
