"""Stress tests for thread communication under high concurrency.

These tests run the fixed concurrency primitives under heavy
concurrent load to shake out races that single-threaded or
lightly-threaded tests wouldn't catch. The goal is to detect
data corruption, deadlocks, or dropped exceptions, not to
benchmark (we leave that to a separate profile run).
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from larksnap.adapters.detector.interface import BBox, DetectionResult
from larksnap.gateway.event_bus import Event, EventBus, EventType
from larksnap.gateway.notification_service import (
    NotificationService,
    NotificationServiceConfig,
)
from larksnap.utils.worker_pool import WorkerPool


# ─── EventBus stress test ────────────────────────────────────────────


@pytest.mark.stress
def test_eventbus_under_concurrent_publish_subscribe() -> None:
    """Hammer the EventBus with many publishers, subscribers, and
    unsubscribers running in parallel. We assert that:
      - the bus never raises,
      - the per-handler call count never exceeds the number of
        publishes (no double-delivery),
      - the test completes within a reasonable time.
    """
    bus = EventBus()
    N_PUBLISHERS = 8
    N_PUBLISHES_PER_PUBLISHER = 500
    N_SUBSCRIBERS = 16
    RUNTIME_SECONDS = 2.0

    stop = threading.Event()
    counters = [0] * N_SUBSCRIBERS
    counter_locks = [threading.Lock() for _ in range(N_SUBSCRIBERS)]
    subscriber_handlers: list[Any] = []

    def make_handler(i: int):
        def handler(event: Event) -> None:
            with counter_locks[i]:
                counters[i] += 1
        return handler

    def subscriber(idx: int) -> None:
        handler = make_handler(idx)
        subscriber_handlers.append(handler)
        bus.subscribe(EventType.SYSTEM_STARTED, handler)
        # Run for RUNTIME_SECONDS, then unsubscribe.
        deadline = time.time() + RUNTIME_SECONDS
        while time.time() < deadline:
            time.sleep(0.01)
        bus.unsubscribe(EventType.SYSTEM_STARTED, handler)

    def publisher(idx: int) -> None:
        for i in range(N_PUBLISHES_PER_PUBLISHER):
            bus.publish(Event(
                type=EventType.SYSTEM_STARTED,
                data=(idx, i),
            ))

    subs = [threading.Thread(target=subscriber, args=(i,)) for i in range(N_SUBSCRIBERS)]
    pubs = [
        threading.Thread(target=publisher, args=(i,))
        for i in range(N_PUBLISHERS)
    ]
    start = time.time()
    for t in subs:
        t.start()
    # Tiny delay so subscribers install their handlers.
    time.sleep(0.05)
    for t in pubs:
        t.start()
    for t in pubs:
        t.join()
    stop.set()
    for t in subs:
        t.join()
    elapsed = time.time() - start
    # Sanity: no thread deadlocked, no exception escaped.
    assert elapsed < RUNTIME_SECONDS + 5
    # Every counter must be <= total publishes (no handler was
    # called more than the number of events the bus ever published).
    total_publishes = N_PUBLISHERS * N_PUBLISHES_PER_PUBLISHER
    for c in counters:
        assert 0 <= c <= total_publishes, f"counter out of range: {c}"


# ─── WorkerPool stress test ──────────────────────────────────────────


@pytest.mark.stress
def test_workerpool_under_high_throughput() -> None:
    """Submit many small tasks from many threads and confirm that
    every task either runs or is explicitly dropped (we count
    completions and ensure the total matches expectations)."""
    N_PRODUCERS = 16
    TASKS_PER_PRODUCER = 1000
    completed = [0]
    lock = threading.Lock()

    def task() -> None:
        with lock:
            completed[0] += 1

    pool = WorkerPool(max_workers=4, queue_size=64, name_prefix="stress")
    barrier = threading.Barrier(N_PRODUCERS)

    def producer() -> None:
        barrier.wait()
        for _ in range(TASKS_PER_PRODUCER):
            pool.submit(task)

    threads = [threading.Thread(target=producer) for _ in range(N_PRODUCERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Wait for the pool to drain.
    deadline = time.time() + 5.0
    while time.time() < deadline and completed[0] < N_PRODUCERS * TASKS_PER_PRODUCER - pool.dropped_count:
        time.sleep(0.05)
    pool.shutdown(timeout=5.0)
    # completed + dropped should equal the total submissions.
    total = N_PRODUCERS * TASKS_PER_PRODUCER
    assert completed[0] + pool.dropped_count == total, (
        f"completed={completed[0]} dropped={pool.dropped_count} total={total}"
    )


# ─── NotificationService stress test ─────────────────────────────────


@pytest.mark.stress
def test_notification_service_under_load() -> None:
    """Flood the service with results from multiple threads and
    confirm:
      - no exception escapes,
      - cooldown prevents duplicate dispatch under burst (good),
      - the service can drain the queue.
    """
    config = NotificationServiceConfig(notification_interval=10.0)
    call_count = [0]
    call_lock = threading.Lock()

    class _CountingNotifier:
        def send_message(self, message: Any) -> bool:
            with call_lock:
                call_count[0] += 1
            return True

    bus = EventBus()
    pool = WorkerPool(max_workers=4, queue_size=256)
    svc = NotificationService(
        config=config,
        notifier=_CountingNotifier(),  # type: ignore[arg-type]
        event_bus=bus,
        worker_pool=pool,
    )

    N_PRODUCERS = 8
    BATCHES_PER_PRODUCER = 50
    BATCH_SIZE = 5
    barrier = threading.Barrier(N_PRODUCERS)

    def producer() -> None:
        barrier.wait()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        for _ in range(BATCHES_PER_PRODUCER):
            results = [
                DetectionResult(
                    label="lbl", confidence=0.9, bbox=BBox(0, 0, 1, 1),
                )
                for _ in range(BATCH_SIZE)
            ]
            svc.handle_results(results, frame=frame)

    threads = [threading.Thread(target=producer) for _ in range(N_PRODUCERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Drain. The cooldown is 10s, so all batches from the same
    # label after the first are suppressed. With 8 producers each
    # firing BATCHES_PER_PRODUCER batches, at most 8 dispatches
    # (one per producer, the first batch of each). Allow a small
    # range in case of subtle timing.
    pool.shutdown(timeout=10.0)
    # call_count is incremented by the notifier (one call per
    # batch that survived cooldown). It must be at most
    # N_PRODUCERS, and at least 1 (someone succeeded).
    assert 1 <= call_count[0] <= N_PRODUCERS, (
        f"call_count={call_count[0]} outside [{1}, {N_PRODUCERS}]"
    )
