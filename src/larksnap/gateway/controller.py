import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from larksnap.adapters.camera.interface import CameraAdapter
from larksnap.adapters.camera.opencv_adapter import OpenCVCameraAdapter
from larksnap.adapters.detector.interface import DetectionResult, DetectorAdapter
from larksnap.adapters.detector.mock_adapter import MockDetectorAdapter
from larksnap.adapters.detector.yolo_seg_adapter import YOLOSegDetectorAdapter
from larksnap.adapters.notifier.feishu_adapter import FeishuNotifierAdapter
from larksnap.adapters.notifier.interface import NotificationMessage, NotifierAdapter
from larksnap.config.models import AppConfig
from larksnap.gateway.event_bus import Event, EventBus, EventType
from larksnap.utils.exceptions import GatewayError


class GatewayController:
    """Gateway controller that orchestrates camera, detector, and notifier adapters."""

    def __init__(self, config: AppConfig, event_bus: EventBus | None = None) -> None:
        """Initialize the gateway with configuration and optional event bus."""
        self._config = config
        self._event_bus = event_bus or EventBus()
        self._logger = logging.getLogger("larksnap.gateway")
        self._running = False
        self._paused = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_notification_time: dict[str, float] = {}
        self._camera: CameraAdapter | None = None
        self._detector: DetectorAdapter | None = None
        self._notifier: NotifierAdapter | None = None

    @property
    def is_running(self) -> bool:
        """Check if the gateway is currently running."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Check if the gateway is currently paused."""
        return self._paused

    def _create_camera(self) -> CameraAdapter:
        return OpenCVCameraAdapter(self._config.camera)

    def _create_detector(self) -> DetectorAdapter:
        detector_type = self._config.detector.type
        if detector_type == "mock":
            return MockDetectorAdapter(self._config.detector)
        if detector_type == "yolo_seg":
            return YOLOSegDetectorAdapter(self._config.detector)
        raise GatewayError(f"Unknown detector type: {detector_type}")

    def _create_notifier(self) -> NotifierAdapter:
        notifier_type = self._config.notifier.type
        if notifier_type == "feishu":
            return FeishuNotifierAdapter(self._config.notifier)
        raise GatewayError(f"Unknown notifier type: {notifier_type}")

    def initialize(self) -> None:
        """Initialize all adapters (camera, detector, notifier)."""
        try:
            self._logger.info("Initializing gateway controller...")
            self._camera = self._create_camera()
            self._detector = self._create_detector()
            self._notifier = self._create_notifier()

            self._camera.initialize()
            self._detector.initialize()
            self._notifier.initialize()

            self._logger.info("Gateway controller initialized successfully")
        except Exception as e:
            self._release_all()
            raise GatewayError(f"Failed to initialize gateway: {e}") from e

    def start(self) -> None:
        """Start the gateway detection loop in a background thread."""
        if self._running:
            self._logger.warning("Gateway is already running")
            return

        self._running = True
        self._paused = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._event_bus.publish(Event(type=EventType.SYSTEM_STARTED, source="gateway"))
        self._logger.info("Gateway controller started")

    def stop(self) -> None:
        """Stop the gateway and release all adapter resources."""
        if not self._running:
            return

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

        self._release_all()
        self._event_bus.publish(Event(type=EventType.SYSTEM_STOPPED, source="gateway"))
        self._logger.info("Gateway controller stopped")

    def pause(self) -> None:
        """Pause the detection loop."""
        self._paused = True
        self._logger.info("Gateway controller paused")

    def resume(self) -> None:
        """Resume the detection loop."""
        self._paused = False
        self._logger.info("Gateway controller resumed")

    def _run_loop(self) -> None:
        while self._running:
            try:
                if self._paused:
                    time.sleep(0.5)
                    continue

                self._process_cycle()
                time.sleep(self._config.gateway.process_interval)
            except Exception as e:
                self._logger.error("Error in gateway loop: %s", e)
                self._event_bus.publish(
                    Event(type=EventType.ERROR_OCCURRED, data=str(e), source="gateway")
                )
                time.sleep(1.0)

    def _process_cycle(self) -> None:
        if self._camera is None or self._detector is None:
            return

        try:
            frame = self._camera.read_frame()
        except Exception as e:
            self._logger.error("Failed to read frame: %s", e)
            self._event_bus.publish(
                Event(type=EventType.ERROR_OCCURRED, data=str(e), source="camera")
            )
            return

        self._event_bus.publish(Event(type=EventType.FRAME_CAPTURED, source="camera"))

        try:
            results = self._detector.detect(frame)
        except Exception as e:
            self._logger.error("Detection failed: %s", e)
            self._event_bus.publish(
                Event(type=EventType.ERROR_OCCURRED, data=str(e), source="detector")
            )
            return

        filtered = self._filter_results(results)

        self._event_bus.publish(
            Event(type=EventType.DETECTION_COMPLETED, data=filtered, source="detector")
        )

        if filtered and self._notifier is not None:
            snapshot_path = self._save_snapshot(frame)
            self._notify_results(filtered, snapshot_path)

    def _filter_results(self, results: list[DetectionResult]) -> list[DetectionResult]:
        """Filter detection results by confidence threshold and target classes."""
        filtered = []
        threshold = self._config.detector.confidence_threshold
        target_classes = self._config.detector.target_classes

        for result in results:
            if result.confidence < threshold:
                continue
            if target_classes and result.label not in target_classes:
                continue
            filtered.append(result)

        return filtered

    def _save_snapshot(self, frame: np.ndarray) -> str | None:
        """Save a camera frame snapshot to the configured directory.

        Returns:
            The file path of the saved snapshot, or None on failure.
        """
        try:
            snapshot_dir = Path(self._config.gateway.snapshot_dir)
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

    def _notify_results(
        self, results: list[DetectionResult], snapshot_path: str | None = None
    ) -> None:
        """Send notifications for filtered detection results with cooldown."""
        now = time.time()
        cooldown = self._config.gateway.notification_cooldown

        for result in results:
            last_time = self._last_notification_time.get(result.label, 0)
            if now - last_time < cooldown:
                self._logger.debug(
                    "Notification for '%s' suppressed (cooldown)", result.label
                )
                continue

            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = NotificationMessage(
                title="LarkSnap Detection Alert",
                content=self._config.notifier.message_template.format(
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

            # Always log detection locally
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
                    self._last_notification_time[result.label] = now
                    self._event_bus.publish(
                        Event(
                            type=EventType.NOTIFICATION_SENT,
                            data=result.label,
                            source="notifier",
                        )
                    )
                else:
                    self._logger.info(
                        "Feishu notification skipped for '%s' "
                        "(see notifier logs for details)",
                        result.label,
                    )
            except Exception as e:
                self._logger.error("Failed to send notification: %s", e)
                self._event_bus.publish(
                    Event(
                        type=EventType.ERROR_OCCURRED,
                        data=str(e),
                        source="notifier",
                    )
                )

    def _release_all(self) -> None:
        adapters = [self._notifier, self._detector, self._camera]
        for adapter in adapters:
            if adapter is not None:
                try:
                    adapter.stop()
                    adapter.release()
                except Exception as e:
                    self._logger.error("Error releasing adapter: %s", e)

        self._notifier = None
        self._detector = None
        self._camera = None
