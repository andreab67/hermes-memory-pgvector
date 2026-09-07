"""writer.py — async write queue for the pgvector memory plugin.

Decouples `on_memory_write` from the (potentially slow) embed + INSERT
path. Built-in memory tool fires the hook → we enqueue → background
thread drains, embeds, writes. The hook returns in microseconds, never
blocks the agent loop on a slow embed endpoint.

Failure modes we deliberately swallow:
  • queue full              → drop write, log once, set a tripwire flag
  • embed endpoint slow/down → write goes through with embedding=NULL,
                              row is still recoverable, recall just
                              misses it until a future backfill
  • DB temporarily down     → drop write, log once
  • background thread crash → log + auto-restart on next enqueue

The whole point is to avoid the Honcho-style retry-storm meltdown.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# Pending-write payload — kept small so the queue stays bounded.
class _PendingWrite:
    __slots__ = ("action", "agent_identity", "target", "content", "extra", "metadata")

    def __init__(
        self,
        *,
        action: str,
        agent_identity: str,
        target: str,
        content: str,
        extra: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.action = action
        self.agent_identity = agent_identity
        self.target = target
        self.content = content
        self.extra = extra or {}
        self.metadata = metadata or {}


class AsyncWriter:
    """Bounded write queue with a background drain thread.

    Caller responsibilities:
      - Construct with `worker_fn`, a callable that takes a _PendingWrite
        and performs the actual embed + DB write. worker_fn MUST NOT raise
        — it should swallow + log its own failures.
      - Call `enqueue(...)` from the agent thread; returns instantly.
      - Call `shutdown(timeout=5)` on plugin teardown to drain gracefully.
    """

    def __init__(self, worker_fn: Callable[[_PendingWrite], None], *, maxsize: int = 256):
        self._worker_fn = worker_fn
        self._queue: "queue.Queue[Optional[_PendingWrite]]" = queue.Queue(maxsize=maxsize)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()      # hard stop: exit ASAP, abandon queue
        self._draining = threading.Event()  # graceful stop: exit once queue is empty
        self._dropped = 0
        self._dropped_warned = False
        self._lock = threading.Lock()

    # -- Public API ----------------------------------------------------------

    def enqueue(
        self,
        *,
        action: str,
        agent_identity: str,
        target: str,
        content: str,
        extra: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Enqueue a write. Returns True on accept, False on drop (queue full).

        Lazily starts the drain thread on first enqueue. Restarts the
        thread if it died (e.g., from an unforeseen exception in worker_fn).
        """
        self._ensure_thread()
        item = _PendingWrite(
            action=action,
            agent_identity=agent_identity,
            target=target,
            content=content,
            extra=extra,
            metadata=metadata,
        )
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            with self._lock:
                self._dropped += 1
                if not self._dropped_warned:
                    logger.warning(
                        "pgvector writer queue full (maxsize=%d); dropping writes",
                        self._queue.maxsize,
                    )
                    self._dropped_warned = True
            return False

    def shutdown(self, timeout: float = 5.0) -> None:
        """Drain the queue (up to `timeout` seconds), then stop the thread.

        The drain flag is set FIRST, so the thread exits only once the queue
        is empty — a full queue at shutdown time is drained rather than
        silently abandoned, and items racing in behind the wake-up sentinel
        are still processed (v0.4.2; previously a full queue downgraded to a
        hard stop that dropped up to maxsize-1 accepted writes with no log).
        """
        if not self._thread or not self._thread.is_alive():
            return
        self._draining.set()
        try:
            # Sentinel wakes the thread instantly if it's idle on an empty
            # queue; if the queue is full the 0.5s get-timeout wakes it anyway.
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("pgvector writer thread did not drain within %.1fs", timeout)
        leftover = self._queue.qsize()
        if leftover:
            logger.warning(
                "pgvector writer shutdown abandoned %d queued write(s)", leftover
            )

    def stats(self) -> Dict[str, Any]:
        return {
            "queue_size": self._queue.qsize(),
            "queue_max": self._queue.maxsize,
            "dropped_total": self._dropped,
            "thread_alive": bool(self._thread and self._thread.is_alive()),
        }

    # -- Internals -----------------------------------------------------------

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._draining.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="pgvector-writer",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._draining.is_set():
                    break  # queue fully drained — graceful shutdown complete
                continue
            if item is None:  # shutdown wake-up sentinel
                self._queue.task_done()
                if self._draining.is_set():
                    # Keep draining — real items can be queued BEHIND the
                    # sentinel (enqueue racing shutdown); exit on Empty above.
                    continue
                break
            try:
                self._worker_fn(item)
            except Exception as exc:  # noqa: BLE001
                # worker_fn is supposed to swallow its own errors. If
                # something slipped through, log and keep going — never
                # let the drain thread die over one bad write.
                logger.warning(
                    "pgvector writer worker raised %s on (%s/%s/%s): %s",
                    type(exc).__name__,
                    item.action,
                    item.agent_identity,
                    item.target,
                    str(exc)[:200],
                )
            finally:
                self._queue.task_done()
