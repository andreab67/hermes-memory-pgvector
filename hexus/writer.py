"""writer.py — async write queue for the hexus memory plugin.
#
# Forked from andreab67/hermes-hexus (BSD-3-Clause).
#
# Decouples `on_memory_write` from the (potentially slow) embed + INSERT
# path. Built-in memory tool fires the hook → we enqueue → background
# thread drains, embeds, writes. The hook returns in microseconds, never
# blocks the agent loop on a slow embed endpoint.

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
        self._stop = threading.Event()
        self._dropped = 0
        self._dropped_warned = False
        self._lock = threading.Lock()
        self._latencies: list[float] = []

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
                        "hexus writer queue full (maxsize=%d); dropping writes",
                        self._queue.maxsize,
                    )
                    self._dropped_warned = True
            return False

    def shutdown(self, timeout: float = 5.0) -> None:
        """Signal stop + wait for the queue to drain (up to `timeout` seconds)."""
        if not self._thread or not self._thread.is_alive():
            return
        # Sentinel wakes the thread even if the queue is empty.
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            self._stop.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("hexus writer thread did not drain within %.1fs", timeout)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            lats = list(self._latencies)
        p50 = float("nan")
        p95 = float("nan")
        if lats:
            sorted_lats = sorted(lats)
            p50 = sorted_lats[int(len(sorted_lats) * 0.5)]
            p95 = sorted_lats[int(len(sorted_lats) * 0.95)]
        return {
            "queue_size": self._queue.qsize(),
            "queue_max": self._queue.maxsize,
            "dropped_total": self._dropped,
            "thread_alive": bool(self._thread and self._thread.is_alive()),
            "p50_latency_sec": p50,
            "p95_latency_sec": p95,
        }

    # -- Internals -----------------------------------------------------------

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="hexus-writer",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        import time
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:  # shutdown sentinel
                self._queue.task_done()
                break
            try:
                t0 = time.perf_counter()
                self._worker_fn(item)
                duration = time.perf_counter() - t0
                with self._lock:
                    self._latencies.append(duration)
                    if len(self._latencies) > 1000:
                        self._latencies.pop(0)
            except Exception as exc:  # noqa: BLE001
                # worker_fn is supposed to swallow its own errors. If
                # something slipped through, log and keep going — never
                # let the drain thread die over one bad write.
                logger.warning(
                    "hexus writer worker raised %s on (%s/%s/%s): %s",
                    type(exc).__name__,
                    item.action,
                    item.agent_identity,
                    item.target,
                    str(exc)[:200],
                )
            finally:
                self._queue.task_done()
