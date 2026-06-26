"""Bounded worker thread pool for offloading blocking I/O.

The ZMQ consumer threads (FrameConsumer, ResultSubscriber) must remain
non-blocking to keep the camera→detection→notification pipeline flowing.
However, notification dispatch and video recording involve slow I/O
(HTTP calls, disk writes) that would block those threads for seconds.

This module provides a small ``WorkerPool`` that wraps a
``ThreadPoolExecutor`` with a bounded queue and a clean shutdown API
suitable for use by long-lived services.

Design constraints:
  - Submitting work from any thread must be O(1) and non-blocking.
  - Shutdown is bounded (no hang on stuck tasks).
  - Submit-after-shutdown is a logged no-op, not an exception, so a
    late callback from a dying ZMQ thread can't crash the gateway.
"""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")


class WorkerPool:
    """Bounded worker thread pool for background I/O.

    The pool is intended to be a long-lived singleton owned by the
    gateway. It spawns ``max_workers`` daemon threads and queues work
    in front of them. The queue is bounded so a runaway producer can't
    exhaust memory; tasks submitted after the queue is full are
    dropped (with a counter for diagnostics) rather than blocking the
    caller, because callers (ZMQ threads) must never block.
    """

    DEFAULT_MAX_WORKERS = 4
    DEFAULT_QUEUE_SIZE = 256

    def __init__(
        self,
        max_workers: int = DEFAULT_MAX_WORKERS,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        name_prefix: str = "larksnap-worker",
    ) -> None:
        self._logger = logging.getLogger("larksnap.worker_pool")
        self._max_workers = max(1, int(max_workers))
        self._queue_size = max(1, int(queue_size))
        self._name_prefix = name_prefix
        self._queue: queue.Queue[Callable[[], None] | None] = queue.Queue(
            maxsize=self._queue_size
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix=self._name_prefix,
        )
        self._workers: list[threading.Thread] = []
        self._shutdown = threading.Event()
        self._dropped_count = 0
        self._dropped_lock = threading.Lock()
        # Start a small set of long-lived worker threads that consume
        # the queue. We use the executor for the actual call() so we
        # get Future-based observability for free, but we drive the
        # queue ourselves to support a bounded queue with drop-on-full.
        for i in range(self._max_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"{self._name_prefix}-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    @property
    def dropped_count(self) -> int:
        """Number of tasks that were dropped because the queue was full."""
        with self._dropped_lock:
            return self._dropped_count

    def submit(self, fn: Callable[[], None]) -> bool:
        """Submit a task. Returns True if queued, False if dropped.

        A return value of False means either:
          - the pool is shut down, or
          - the queue is full (back-pressure: caller should drop or retry).

        Never blocks. Never raises after the pool is initialised.
        """
        if self._shutdown.is_set():
            return False
        try:
            self._queue.put_nowait(fn)
            return True
        except queue.Full:
            with self._dropped_lock:
                self._dropped_count += 1
            self._logger.warning(
                "WorkerPool queue full (size=%d); task dropped. "
                "Total dropped: %d",
                self._queue_size, self._dropped_count,
            )
            return False

    def submit_callable(
        self, fn: Callable[..., T], *args, **kwargs
    ) -> Future[T] | None:
        """Submit a task and return a Future, or None if pool is shut down.

        Unlike ``submit`` this uses the underlying executor directly,
        so it can block briefly while waiting for an executor worker.
        Intended for low-volume tasks where the caller wants the
        return value. ZMQ hot paths should use ``submit`` instead.
        """
        if self._shutdown.is_set():
            return None
        return self._executor.submit(fn, *args, **kwargs)

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop accepting work and join worker threads.

        Bounded: any worker still running a task is given ``timeout``
        seconds to finish, then the pool is abandoned. Daemon threads
        will be cleaned up at process exit.
        """
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        # Send sentinel values to wake workers out of get()
        for _ in self._workers:
            try:
                self._queue.put_nowait(None)
            except queue.Full:  # pragma: no cover
                pass
        for t in self._workers:
            t.join(timeout=timeout)
        # Workers are daemons, so abandoned ones die at process exit.
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._logger.info("WorkerPool shut down (dropped=%d)", self.dropped_count)

    def _worker_loop(self) -> None:
        """Pull tasks from the queue and run them.

        Tasks may raise; we log and continue so a single bad task
        doesn't take down the worker.
        """
        while not self._shutdown.is_set():
            try:
                fn = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if fn is None:  # sentinel
                break
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                self._logger.error("WorkerPool task raised: %s", e)
