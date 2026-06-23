"""Notification service encapsulating cooldown, snapshot, and dispatch logic.

Owns its configuration (NotificationServiceConfig) and manages:
  - Per-label notification cooldown
  - Snapshot saving (only when notification is actually sent)
  - Message formatting and dispatch to NotifierAdapter
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from larksnap.adapters.detector.interface import DetectionResult
from larksnap.adapters.notifier.interface import NotificationMessage, NotifierAdapter
from larksnap.gateway.event_bus import Event, EventBus, EventType


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
        service = NotificationService(config, notifier, event_bus)
        service.handle_results(results, frame)
    """

    def __init__(
        self,
        config: NotificationServiceConfig,
        notifier: NotifierAdapter,
        event_bus: EventBus,
    ) -> None:
        self._config = config
        self._notifier = notifier
        self._event_bus = event_bus
        self._logger = logging.getLogger("larksnap.notification_service")
        self._last_notification_time: dict[str, float] = {}
        self._notification_enabled = True  # Enabled by default when pipeline starts

    @property
    def notification_enabled(self) -> bool:
        return self._notification_enabled

    def enable_notification(self) -> None:
        """Enable notification dispatch (triggered by /start command)."""
        if not self._notification_enabled:
            self._notification_enabled = True
            self._logger.info("Notification enabled")
            self._event_bus.publish(
                Event(type=EventType.NOTIFICATION_ENABLED, source="notification_service")
            )

    def disable_notification(self) -> None:
        """Disable notification dispatch (triggered by /stop command)."""
        if self._notification_enabled:
            self._notification_enabled = False
            self._logger.info("Notification disabled")
            self._event_bus.publish(
                Event(type=EventType.NOTIFICATION_DISABLED, source="notification_service")
            )

    @property
    def config(self) -> NotificationServiceConfig:
        return self._config

    def handle_results(
        self, results: list[DetectionResult], frame: np.ndarray | None = None
    ) -> None:
        """Process detection results: filter by cooldown, save snapshot, notify."""
        if not results:
            return

        # Skip notification dispatch if not enabled (waiting for /start)
        if not self._notification_enabled:
            self._logger.debug("Notification disabled, skipping dispatch")
            return

        now = time.time()
        interval = self._config.notification_interval

        # Collect results that pass the cooldown check
        to_notify: list[DetectionResult] = []
        for result in results:
            last_time = self._last_notification_time.get(result.label, 0)
            if now - last_time < interval:
                self._logger.debug(
                    "Notification for '%s' suppressed (cooldown)", result.label
                )
                continue
            to_notify.append(result)

        if not to_notify:
            return

        # Save snapshot once for this batch
        snapshot_path = self._save_snapshot(frame) if frame is not None else None

        for result in to_notify:
            self._dispatch(result, snapshot_path, now)

    def _dispatch(
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

        # Always update cooldown to prevent resource waste (snapshot + dispatch)
        # even if notification delivery failed (e.g. chat_id not configured)
        self._last_notification_time[result.label] = now

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
        self._last_notification_time.clear()
