"""Notification service encapsulating cooldown, snapshot, and dispatch logic.

Owns its configuration (NotificationServiceConfig) and manages:
  - Per-label notification cooldown
  - Snapshot saving (only when notification is actually sent)
  - Message formatting and dispatch to NotifierAdapter
  - Defense-in-depth target-class filter (see :func:`handle_results`)

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

from larksnap.adapters.detector.interface import (
    DetectionResult,
    filter_results_by_classes,
)
from larksnap.adapters.notifier.interface import NotificationMessage, NotifierAdapter
from larksnap.gateway.component_state import (
    ComponentKind,
    ComponentState,
    ComponentStatus,
)
from larksnap.gateway.event_bus import Event, EventBus, EventType
from larksnap.utils.worker_pool import WorkerPool


class _SafeFormatMap(dict):
    """Mapping that returns ``"{key}"`` for unknown keys.

    Used by :py:meth:`NotificationService._dispatch_aggregated` so a
    user template with a typo (``{nonexistent}``) renders as the
    literal ``{nonexistent}`` in the message instead of crashing
    the whole notification path. Built-in keys are filled normally.
    """

    def __missing__(self, key: str) -> str:  # type: ignore[override]
        # Returning the literal placeholder, not an empty string,
        # makes the typo obvious to the user when they look at the
        # message — better than silently swallowing the field.
        return "{" + str(key) + "}"


@dataclass
class NotificationServiceConfig:
    """Configuration owned by NotificationService."""

    notification_interval: float = 30.0
    snapshot_dir: str = "snapshots"
    # Default template uses the multi-class placeholders so a
    # single notification can describe several detections in one
    # frame. ``{labels_summary}`` is ``"person (90%), car (78%)"``;
    # ``{labels_count}`` is the integer count. Users who want
    # the legacy single-result view can switch to ``{label}`` /
    # ``{confidence}`` in their config — those map to the
    # top-confidence detection.
    message_template: str = (
        "[LarkSnap] 检测到 {labels_summary}（{labels_count} 个目标），时间: {timestamp}"
    )
    # The classes the user has configured the detector to monitor.
    # Used as a defense-in-depth filter in :py:meth:`handle_results`
    # so a misbehaving / future detector adapter cannot leak
    # non-target classes into notifications. ``None`` disables the
    # filter at this layer (the detector adapter is then the sole
    # gatekeeper). An explicit empty list means "monitor nothing"
    # — the service will drop every result it receives.
    target_classes: list[str] | None = None


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

        Defense-in-depth target-class filter: even if a detector
        adapter forgot to apply the monitoring contract, the
        notification path will still drop results whose label is
        not in ``config.target_classes``. An explicit empty list
        means "monitor nothing" — every result is dropped here.
        ``None`` means the detector adapter is the sole gatekeeper
        and this layer forwards whatever arrives.
        """
        if not results:
            return

        # Target-class filter at the notification layer. The detector
        # adapters are expected to apply the same filter, but doing
        # it again here guarantees the contract even if a future
        # adapter forgets. Uses the same helper as the adapters so
        # semantics (case folding, empty-set handling) stay aligned.
        results = filter_results_by_classes(results, self._config.target_classes)
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
        """Save one snapshot and dispatch one aggregated notification.

        For multi-class monitoring (the user-configured normal case),
        a single frame can produce several target detections. We
        coalesce them into one Feishu message — one image upload,
        one text — so the user sees the full picture instead of N
        near-duplicate messages a few milliseconds apart. Single-
        result batches keep the same shape, so the message template
        remains a single-record template at the API level.

        This method runs on a worker thread (or, in the test path,
        directly on the caller). It is allowed to block on I/O.
        """
        # Save snapshot once for this batch. Failure here is logged
        # but doesn't stop the notification from going out — the
        # snapshot is "best effort".
        snapshot_path: str | None = None
        if frame is not None:
            snapshot_path = self._save_snapshot(frame)

        self._dispatch_aggregated(to_notify, snapshot_path, now)

    def _dispatch_aggregated(
        self,
        results: list[DetectionResult],
        snapshot_path: str | None,
        now: float,
    ) -> None:
        """Format and send one notification covering every result.

        The message template can reference either:

        * ``{label}`` / ``{confidence}`` — the top-confidence result
          (backwards-compatible with the single-class template).
        * ``{labels_summary}`` — e.g. ``person (90%), car (78%)``,
          sorted by confidence descending so the most important
          detection is first.
        * ``{labels_count}`` — number of distinct detections in
          the batch.
        * ``{labels}`` — comma-separated list of labels (no
          confidences), for templates that want a flat list.

        If the template references none of the multi-class keys,
        the existing single-result template keeps working — the
        one-record view is just rendered as the top-confidence
        detection, which matches the previous behaviour for
        single-class users.
        """
        if not results:
            return

        # Stable order: most-confident first, then a stable
        # tiebreaker on label so the rendered message is
        # reproducible across runs with identical input.
        ordered = sorted(
            results,
            key=lambda r: (-r.confidence, r.label),
        )
        top = ordered[0]

        labels_summary = ", ".join(
            f"{r.label} ({r.confidence:.0%})" for r in ordered
        )
        labels_csv = ", ".join(r.label for r in ordered)

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        format_kwargs: dict[str, Any] = {
            "label": top.label,
            "confidence": top.confidence,
            "timestamp": timestamp_str,
            "snapshot_path": snapshot_path or "",
            "labels_summary": labels_summary,
            "labels": labels_csv,
            "labels_count": len(ordered),
        }
        # Templates written before the multi-class keys existed
        # may use ``str.format`` with a strict spec. To stay
        # forward-compatible, we render the template with a
        # fallback mapping: known keys are filled, unknown
        # keys are left as the literal ``{key}`` in the
        # message (rather than crashing the whole notification).
        # This means a typo in a custom template degrades to a
        # slightly ugly message instead of an outage.
        try:
            content = self._config.message_template.format_map(
                _SafeFormatMap(format_kwargs)
            )
        except (KeyError, IndexError):
            # Defensive: ``str.format_map`` can still raise on
            # positional placeholders like ``{0}``. Fall back to
            # the bare legacy set, which only contains known keys.
            content = self._config.message_template.format(
                label=top.label,
                confidence=top.confidence,
                timestamp=timestamp_str,
                snapshot_path=snapshot_path or "",
            )

        message = NotificationMessage(
            title="LarkSnap Detection Alert",
            content=content,
            label=top.label,
            confidence=top.confidence,
            timestamp=timestamp_str,
            snapshot_path=snapshot_path,
        )

        self._logger.info(
            "DETECTED (%d): %s (time: %s, snapshot: %s)",
            len(ordered),
            labels_summary,
            timestamp_str,
            snapshot_path or "N/A",
        )

        try:
            success = self._notifier.send_message(message)
            if success:
                # Publish one event per label so downstream
                # consumers (stats panel, etc.) still get a
                # per-class signal. The aggregated message
                # is purely a presentation concern.
                for r in ordered:
                    self._event_bus.publish(
                        Event(
                            type=EventType.NOTIFICATION_SENT,
                            data=r.label,
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
