"""Notification service encapsulating cooldown, snapshot, and dispatch logic.

Owns its configuration (NotificationServiceConfig) and manages:
  - Per-label notification cooldown
  - Snapshot saving (only when notification is actually sent)
  - Message formatting and dispatch to NotifierAdapter

Concurrency:
  - ``handle_results`` is invoked from the ZMQ result-subscriber
    thread. It is fast and non-blocking: it only checks the
    notification-enabled flag, evaluates cooldowns, and queues a
    task on the ``WorkerPool`` for the slow I/O (disk write +
    HTTP POST).
  - All actual snapshot saving and ``notifier.send_message`` calls
    happen on the worker pool. This guarantees the ZMQ consumer
    thread never blocks on a slow Feishu API call, which would
    otherwise stall the camera→detection→notification pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from larksnap.adapters.detector.interface import DetectionResult
from larksnap.adapters.notifier.interface import NotificationMessage, NotifierAdapter
from larksnap.gateway.component_state import (
    ComponentKind,
    ComponentState,
    ComponentStatus,
)
from larksnap.gateway.event_bus import Event, EventBus, EventType
from larksnap.utils.worker_pool import WorkerPool


@dataclass
class NotificationServiceConfig:
    """Configuration owned by NotificationService."""

    notification_interval: float = 30.0
    snapshot_dir: str = "snapshots"
    message_template: str = (
        "[LarkSnap] 检测到 {label}，置信度: {confidence:.2%}，时间: {timestamp}"
    )


class NotificationService:
    """Encapsulates notification cooldown, snapshot saving, and dispatch.

    Usage:
        service = NotificationService(config, notifier, event_bus, worker_pool)
        service.handle_results(results, frame)
    """

    def __init__(
        self,
        config: NotificationServiceConfig,
        notifier: NotifierAdapter,
        event_bus: EventBus,
        worker_pool: WorkerPool | None = None,
    ) -> None:
        self._config = config
        self._notifier = notifier
        self._event_bus = event_bus
        # The service is fine with no worker pool; in tests it may
        # be omitted. When present, blocking I/O is offloaded to it
        # so the ZMQ thread stays responsive.
        self._worker_pool = worker_pool
        self._logger = logging.getLogger("larksnap.notification_service")
        # Cooldown table and enabled flag are written from the ZMQ
        # hot path (read/write of small primitives). A single lock
        # keeps them consistent without measurable overhead.
        self._cooldown_lock = threading.Lock()
        self._last_notification_time: dict[str, float] = {}
        # notification_enabled is also read by handlers, so guard it
        # with the same lock. We expose it via a property to keep
        # the public API stable.
        self._notification_enabled = True  # Enabled by default when pipeline starts

    @property
    def notification_enabled(self) -> bool:
        """Deprecated alias for :py:attr:`is_notification_enabled`."""
        return self.is_notification_enabled

    @property
    def is_notification_enabled(self) -> bool:
        with self._cooldown_lock:
            return self._notification_enabled

    def enable_notification(self) -> None:
        """Enable notification dispatch (triggered by /start command)."""
        with self._cooldown_lock:
            current = self._notification_enabled
            self._notification_enabled = True
        if not current:
            self._logger.info("Notification enabled")
            self._event_bus.publish(
                Event(type=EventType.NOTIFICATION_ENABLED, source="notification_service")
            )
            # Also publish the unified component state so the UI
            # status panel and menu checkbox can update in lock-step.
            self._publish_state(ComponentState.RUNNING)

    def disable_notification(self) -> None:
        """Disable notification dispatch (triggered by /stop command)."""
        with self._cooldown_lock:
            current = self._notification_enabled
            self._notification_enabled = False
        if current:
            self._logger.info("Notification disabled")
            self._event_bus.publish(
                Event(type=EventType.NOTIFICATION_DISABLED, source="notification_service")
            )
            self._publish_state(ComponentState.DISABLED)

    def _publish_state(self, state: ComponentState) -> None:
        """Publish a ``COMPONENT_STATE_CHANGED`` event with a status payload.

        The unified handler in the UI reads ``event.data`` to know which
        subsystem changed and what its new state is. Publishing without
        a payload would force every consumer to re-derive the state
        from the bus, defeating the single-source-of-truth design.
        """
        status = ComponentStatus(kind=ComponentKind.NOTIFIER, state=state)
        self._event_bus.publish(
            Event(
                type=EventType.COMPONENT_STATE_CHANGED,
                source="notification_service",
                data=status,
            )
        )

    @property
    def config(self) -> NotificationServiceConfig:
        return self._config

    def handle_results(
        self, results: list[DetectionResult], frame: np.ndarray | None = None
    ) -> None:
        """Process detection results: filter by cooldown, save snapshot, notify.

        This is the hot-path entry point. It runs on the ZMQ result
        subscriber thread, so it must be fast and non-blocking.
        Snapshot writes and notifier.send_message calls are dispatched
        to the worker pool (if available) so the ZMQ thread keeps
        polling the socket.
        """
        if not results:
            return

        with self._cooldown_lock:
            enabled = self._notification_enabled
            now = time.time()
            interval = self._config.notification_interval
            # Collect results that pass the cooldown check.
            to_notify: list[DetectionResult] = []
            for result in results:
                last_time = self._last_notification_time.get(result.label, 0)
                if now - last_time < interval:
                    self._logger.debug(
                        "Notification for '%s' suppressed (cooldown)", result.label
                    )
                    continue
                to_notify.append(result)
            # Reserve the cooldown under the same lock as the read
            # so a parallel handle_results() call doesn't double-fire.
            for r in to_notify:
                self._last_notification_time[r.label] = now

        if not enabled:
            self._logger.debug("Notification disabled, skipping dispatch")
            return

        if not to_notify:
            return

        # Offload snapshot save + notifier dispatch to the worker pool.
        # If no pool is configured (e.g. unit tests), fall back to
        # the inline path. The inline path is the same code, just
        # executed on this thread.
        if self._worker_pool is None:
            self._dispatch_batch(to_notify, frame, now)
        else:
            # Snapshot the frame once and hand the bytes to the
            # worker — sharing the same numpy view across threads
            # would be racy because the ZMQ subscriber overwrites
            # it on the next frame.
            frame_copy: np.ndarray | None = None
            if frame is not None:
                frame_copy = np.ascontiguousarray(frame)
            payload = (to_notify, frame_copy, now)
            queued = self._worker_pool.submit(
                lambda: self._dispatch_batch(*payload)
            )
            if not queued:
                # Queue full: the notifier is overwhelmed. Log and
                # drop this batch — better than blocking the ZMQ
                # thread on a slow HTTP retry. The next batch will
                # be retried.
                self._logger.warning(
                    "Worker pool full, dropping notification batch of %d result(s)",
                    len(to_notify),
                )

    def _dispatch_batch(
        self,
        to_notify: list[DetectionResult],
        frame: np.ndarray | None,
        now: float,
    ) -> None:
        """Save one snapshot and dispatch one notification per result.

        This method runs on a worker thread (or, in the test path,
        directly on the caller). It is allowed to block on I/O.
        """
        # Save snapshot once for this batch. Failure here is logged
        # but doesn't stop the notification from going out — the
        # snapshot is "best effort".
        snapshot_path: str | None = None
        if frame is not None:
            snapshot_path = self._save_snapshot(frame)

        for result in to_notify:
            self._dispatch_one(result, snapshot_path, now)

    def _dispatch_one(
        self, result: DetectionResult, snapshot_path: str | None, now: float
    ) -> None:
        """Format and send a single notification."""
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = NotificationMessage(
            title="LarkSnap Detection Alert",
            content=self._config.message_template.format(
                label=result.label,
                confidence=result.confidence,
                timestamp=timestamp_str,
                snapshot_path=snapshot_path or "",
            ),
            label=result.label,
            confidence=result.confidence,
            timestamp=timestamp_str,
            snapshot_path=snapshot_path,
        )

        self._logger.info(
            "DETECTED: %s (confidence: %.2f, time: %s, snapshot: %s)",
            result.label,
            result.confidence,
            timestamp_str,
            snapshot_path or "N/A",
        )

        try:
            success = self._notifier.send_message(message)
            if success:
                self._event_bus.publish(
                    Event(
                        type=EventType.NOTIFICATION_SENT,
                        data=result.label,
                        source="notification_service",
                    )
                )
        except Exception as e:
            self._logger.error("Failed to send notification: %s", e)
            self._event_bus.publish(
                Event(
                    type=EventType.ERROR_OCCURRED,
                    data=str(e),
                    source="notification_service",
                )
            )
        # Cooldown was already reserved in handle_results() so we
        # don't need to update it again here. The reservation prevents
        # duplicate dispatches even if the worker pool reorders tasks.

    def _save_snapshot(self, frame: np.ndarray) -> str | None:
        """Save a snapshot image to the configured directory."""
        try:
            snapshot_dir = Path(self._config.snapshot_dir)
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            filepath = snapshot_dir / filename
            cv2.imwrite(str(filepath), frame)
            self._logger.info("Snapshot saved: %s", filepath)
            return str(filepath)
        except Exception as e:
            self._logger.error("Failed to save snapshot: %s", e)
            return None

    def reset_cooldown(self) -> None:
        """Reset all cooldown timers."""
        with self._cooldown_lock:
            self._last_notification_time.clear()
