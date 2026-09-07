"""Unit tests for pgvector/writer.py's AsyncWriter — pure in-process, no DB.

AsyncWriter decouples on_memory_write/sync_turn from the (potentially slow)
embed + INSERT path via a background drain thread; worker_fn here is a plain
recording stub, never a real DB call, so these run everywhere.

GROUP B regression coverage: shutdown() must actually DRAIN the queue before
the thread stops, not abandon whatever is still queued. Before the v0.4.2 fix
(see the docstring on AsyncWriter.shutdown), a full queue at shutdown time
downgraded to a hard stop that silently dropped up to maxsize-1 accepted
writes with no log line -- writes the caller believed were durably queued.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pgvector.writer import AsyncWriter  # noqa: E402


def test_shutdown_drains_all_enqueued_items():
    """Enqueue N items, shut down, assert the sink received every one."""
    received: list = []
    lock = threading.Lock()

    def sink(item):
        with lock:
            received.append(item)

    writer = AsyncWriter(sink, maxsize=256)
    n = 50
    for i in range(n):
        ok = writer.enqueue(
            action="turn",
            agent_identity="pytest-writer",
            target="conversations",
            content=f"item-{i}",
        )
        assert ok is True  # queue has room; nothing should be dropped on enqueue

    writer.shutdown(timeout=5.0)

    assert len(received) == n
    assert {item.content for item in received} == {f"item-{i}" for i in range(n)}


def test_shutdown_drains_a_queue_that_is_full_at_shutdown_time():
    """Regression for the exact v0.4.2 bug: shutdown() called while the queue
    is still full (worker_fn deliberately slow) must still drain every item
    rather than downgrading to a hard stop.

    Uses a threading.Event to synchronize with the worker instead of a bare
    sleep, so the test can't be flaky under slow CI.
    """
    maxsize = 8
    total = maxsize + 1  # one item in flight (dequeued, blocked in sink) + a full queue behind it
    received: list = []
    lock = threading.Lock()
    release = threading.Event()
    first_item_seen = threading.Event()

    def sink(item):
        # Block the FIRST item's processing until the test says go, so the
        # queue can fill up completely (and shutdown() can be called while
        # it's still full) before the drain thread makes any progress.
        if not first_item_seen.is_set():
            first_item_seen.set()
            release.wait(timeout=5.0)
        with lock:
            received.append(item)

    writer = AsyncWriter(sink, maxsize=maxsize)

    # First enqueue starts the drain thread, which immediately dequeues it and
    # blocks inside sink() on `release`, so nothing drains the queue while we
    # fill it up below.
    assert writer.enqueue(
        action="turn", agent_identity="a", target="conversations", content="item-0"
    )
    first_item_seen.wait(timeout=5.0)

    for i in range(1, total):
        assert writer.enqueue(
            action="turn", agent_identity="a", target="conversations", content=f"item-{i}"
        )
    # Queue is now completely full (item-0 is out being processed, the rest
    # of `total` fill it to maxsize) -- this is the exact "full queue at
    # shutdown time" state the v0.4.2 fix targets.
    assert writer.stats()["queue_size"] == maxsize

    # Let the worker (and the rest of the drain loop) proceed, then shut down.
    release.set()
    writer.shutdown(timeout=5.0)

    assert len(received) == total
    assert {item.content for item in received} == {f"item-{i}" for i in range(total)}


def test_shutdown_on_writer_with_no_items_is_a_noop():
    def sink(item):
        pass

    writer = AsyncWriter(sink, maxsize=8)
    # Never enqueued -> thread never started. Must not raise or hang.
    writer.shutdown(timeout=1.0)
    assert writer.stats()["thread_alive"] is False
