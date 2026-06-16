"""Gateway controller with ZMQ-based producer-consumer architecture.

Orchestrates camera (producer), detector (consumer), notifier, and recorder
through ZeroMQ frame queue, decoupling frame capture from model inference.
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import zmq

from larksnap.adapters.camera.interface import CameraAdapter
from larksnap.adapters.camera.opencv_adapter import OpenCVCameraAdapter
from larksnap.adapters.detector.interface import DetectionResult, DetectorAdapter
from larksnap.adapters.detector.mock_adapter import MockDetectorAdapter
from larksnap.adapters.detector.seg_adapter import SegDetectorAdapter
from larksnap.adapters.notifier.feishu_adapter import FeishuNotifierAdapter
from larksnap.adapters.notifier.feishu_ws_client import CommandHandler, FeishuWSClient
from larksnap.adapters.notifier.interface import NotificationMessage, NotifierAdapter
from larksnap.adapters.recorder.video_recorder import VideoRecorderAdapter
from larksnap.config.models import AppConfig
from larksnap.gateway.event_bus import Event, EventBus, EventType
from larksnap.gateway.frame_queue import (
    FrameConsumer,
    FramePacket,
    FrameProducer,
    FrameQueuePolicy,
    ResultPublisher,
    ResultSubscriber,
)
from larksnap.utils.exceptions import CameraError, GatewayError


class GatewayController:
    """Gateway controller with ZMQ producer-consumer architecture.

    Architecture:
        Camera → FrameProducer → ZMQ PUSH/PULL → FrameConsumer → Detector
        Detector → ResultPublisher → ZMQ PUB/SUB → ResultSubscriber → Notifier/UI
        FrameProducer → ZMQ PUSH/PULL → FrameConsumer → Recorder (optional)
    """

    FRAME_QUEUE_URL = "inproc://frame_queue"
    RESULT_QUEUE_URL = "inproc://detection_results"
    PREVIEW_QUEUE_URL = "inproc://preview_frames"

    def __init__(self, config: AppConfig, event_bus: EventBus | None = None) -> None:
        self._config = config
        self._event_bus = event_bus or EventBus()
        self._logger = logging.getLogger("larksnap.gateway")
        self._running = False
        self._paused = False
        self._lock = threading.Lock()
        self._last_notification_time: dict[str, float] = {}

        # Adapters
        self._camera: CameraAdapter | None = None
        self._detector: DetectorAdapter | None = None
        self._notifier: NotifierAdapter | None = None
        self._recorder: VideoRecorderAdapter | None = None
        self._ws_client: FeishuWSClient | None = None

        # ZMQ components
        self._zmq_context: zmq.Context | None = None
        self._producer: FrameProducer | None = None
        self._detector_consumer: FrameConsumer | None = None
        self._preview_consumer: FrameConsumer | None = None
        self._result_publisher: ResultPublisher | None = None
        self._result_subscriber: ResultSubscriber | None = None

        # State
        self._detection_count: int = 0
        self._camera_failed: bool = False
        self._latest_frame: np.ndarray | None = None
        self._latest_results: list[DetectionResult] = []
        self._frame_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_recording(self) -> bool:
        return self._recorder is not None and self._recorder.is_recording

    @property
    def detection_count(self) -> int:
        return self._detection_count

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def producer_fps(self) -> float:
        if self._producer is not None:
            return self._producer.fps
        return 0.0

    def get_latest_frame(self) -> np.ndarray | None:
        """Get the latest camera frame (thread-safe)."""
        with self._frame_lock:
            return self._latest_frame

    def get_latest_results(self) -> list[DetectionResult]:
        """Get the latest detection results (thread-safe)."""
        with self._frame_lock:
            return list(self._latest_results)

    def _create_camera(self) -> CameraAdapter:
        return OpenCVCameraAdapter(self._config.camera)

    def _create_detector(self) -> DetectorAdapter:
        detector_type = self._config.detector.type
        if detector_type == "mock":
            return MockDetectorAdapter(self._config.detector)
        if detector_type == "seg":
            return SegDetectorAdapter(self._config.detector)
        raise GatewayError(f"Unknown detector type: {detector_type}")

    def _create_notifier(self) -> NotifierAdapter:
        notifier_type = self._config.notifier.type
        if notifier_type == "feishu":
            return FeishuNotifierAdapter(self._config.notifier)
        raise GatewayError(f"Unknown notifier type: {notifier_type}")

    def _create_recorder(self) -> VideoRecorderAdapter:
        return VideoRecorderAdapter(
            output_dir=self._config.recorder.output_dir,
            fps=self._config.recorder.fps,
            codec=self._config.recorder.codec,
            frame_queue_url=self.FRAME_QUEUE_URL,
        )

    def _init_ws_client(self) -> None:
        try:
            self._ws_client = FeishuWSClient(
                config=self._config.notifier,
                on_command=self._handle_command,
            )
            self._ws_client.start()
        except Exception as e:
            self._logger.warning(
                "Feishu WS client failed to start (commands disabled): %s", e
            )
            self._ws_client = None

    def _handle_command(self, cmd: CommandHandler) -> None:
        self._logger.info("Processing command: /%s", cmd.name)

        if cmd.chat_id and isinstance(self._notifier, FeishuNotifierAdapter):
            self._notifier.set_chat_id(cmd.chat_id)

        if cmd.name == "start":
            if not self._running:
                self.start()
        elif cmd.name == "stop":
            if self._running:
                self.stop()
        elif cmd.name == "pause":
            if self._running and not self._paused:
                self.pause()
        elif cmd.name == "resume":
            if self._running and self._paused:
                self.resume()
        elif cmd.name == "status":
            status = "running" if self._running else "stopped"
            if self._running and self._paused:
                status = "paused"
            self._logger.info("Gateway status: %s", status)
        elif cmd.name == "help":
            self._logger.info(
                "Available commands: /start, /stop, /pause, /resume, /status, /help"
            )

    def initialize(self) -> None:
        """Initialize all adapters and ZMQ infrastructure."""
        try:
            self._logger.info("Initializing gateway controller...")

            # Initialize adapters
            self._camera = self._create_camera()
            self._detector = self._create_detector()
            self._notifier = self._create_notifier()
            self._recorder = self._create_recorder()

            try:
                self._camera.initialize()
            except CameraError as e:
                self._logger.error("Camera initialization failed: %s", e)
                self._event_bus.publish(Event(
                    type=EventType.CAMERA_FAILED,
                    data={"error": str(e), "device_index": self._config.camera.device_index},
                    source="gateway",
                ))
                self._camera_failed = True
                raise

            self._camera_failed = False
            self._detector.initialize()
            self._notifier.initialize()
            self._recorder.initialize()

            # Initialize ZMQ context
            self._zmq_context = zmq.Context.instance()

            # Initialize ZMQ components
            self._producer = FrameProducer(
                camera=self._camera,
                zmq_url=self.FRAME_QUEUE_URL,
                policy=FrameQueuePolicy.DROP_OLDEST,
                hwm=30,
            )

            self._detector_consumer = FrameConsumer(
                zmq_url=self.FRAME_QUEUE_URL,
                hwm=10,
            )

            self._preview_consumer = FrameConsumer(
                zmq_url=self.FRAME_QUEUE_URL,
                hwm=5,
            )

            self._result_publisher = ResultPublisher(
                zmq_url=self.RESULT_QUEUE_URL,
            )

            self._result_subscriber = ResultSubscriber(
                zmq_url=self.RESULT_QUEUE_URL,
                topic="",
            )

            # Start WebSocket command listener
            if self._config.notifier.app_id and self._config.notifier.app_secret:
                self._init_ws_client()

            self._logger.info("Gateway controller initialized successfully")
        except Exception as e:
            self._release_all()
            raise GatewayError(f"Failed to initialize gateway: {e}") from e

    def start(self) -> None:
        """Start the gateway with producer-consumer pipeline."""
        if self._running:
            self._logger.warning("Gateway is already running")
            return

        self._running = True
        self._paused = False

        # Start ZMQ result publisher
        self._result_publisher.start(context=self._zmq_context)

        # Start frame producer (camera → ZMQ)
        self._producer.start(context=self._zmq_context)

        # Start detector consumer (ZMQ → detector → result publisher)
        self._detector_consumer.start(
            on_frame=self._on_detection_frame,
            context=self._zmq_context,
        )

        # Start preview consumer (ZMQ → latest frame cache)
        self._preview_consumer.start(
            on_frame=self._on_preview_frame,
            context=self._zmq_context,
        )

        # Start result subscriber (results → notifier)
        self._result_subscriber.start(
            on_result=self._on_detection_result,
            context=self._zmq_context,
        )

        self._event_bus.publish(Event(type=EventType.SYSTEM_STARTED, source="gateway"))
        self._logger.info("Gateway controller started (producer-consumer mode)")

    def stop(self) -> None:
        """Stop the gateway and release all resources."""
        if not self._running:
            return

        self._running = False

        # Stop ZMQ components
        if self._producer is not None:
            self._producer.stop()
        if self._detector_consumer is not None:
            self._detector_consumer.stop()
        if self._preview_consumer is not None:
            self._preview_consumer.stop()
        if self._result_subscriber is not None:
            self._result_subscriber.stop()
        if self._result_publisher is not None:
            self._result_publisher.stop()

        if self._ws_client is not None:
            self._ws_client.stop()
            self._ws_client = None

        self._release_all()
        self._event_bus.publish(Event(type=EventType.SYSTEM_STOPPED, source="gateway"))
        self._logger.info("Gateway controller stopped")

    def pause(self) -> None:
        """Pause detection (producer keeps running for preview)."""
        self._paused = True
        self._logger.info("Gateway controller paused (detection paused, preview active)")

    def resume(self) -> None:
        """Resume detection."""
        self._paused = False
        self._logger.info("Gateway controller resumed")

    def start_recording(self) -> None:
        """Start video recording."""
        if self._recorder is not None and self._running:
            self._recorder.start_recording(context=self._zmq_context)
            self._event_bus.publish(
                Event(type=EventType.SYSTEM_STARTED, data="recording", source="recorder")
            )

    def stop_recording(self) -> None:
        """Stop video recording."""
        if self._recorder is not None:
            self._recorder.stop_recording()

    def _on_detection_frame(self, packet: FramePacket) -> None:
        """Handle frame from detector consumer: run detection and publish results."""
        if self._paused:
            return

        if self._detector is None:
            return

        try:
            results = self._detector.detect(packet.frame)
            self._detection_count += 1

            filtered = self._filter_results(results)

            # Update latest results
            with self._frame_lock:
                self._latest_results = filtered

            self._event_bus.publish(
                Event(type=EventType.DETECTION_COMPLETED, data=filtered, source="detector")
            )

            # Publish results via ZMQ for other consumers
            if self._result_publisher is not None:
                result_data = {
                    "frame_index": packet.frame_index,
                    "timestamp": packet.timestamp,
                    "results": [
                        {
                            "label": r.label,
                            "confidence": r.confidence,
                            "bbox": {
                                "x": r.bbox.x,
                                "y": r.bbox.y,
                                "width": r.bbox.width,
                                "height": r.bbox.height,
                            },
                        }
                        for r in filtered
                    ],
                }
                self._result_publisher.publish("detection", result_data)

        except Exception as e:
            self._logger.error("Detection failed: %s", e)
            self._event_bus.publish(
                Event(type=EventType.ERROR_OCCURRED, data=str(e), source="detector")
            )

    def _on_preview_frame(self, packet: FramePacket) -> None:
        """Cache the latest frame for UI preview."""
        with self._frame_lock:
            self._latest_frame = packet.frame.copy()

        self._event_bus.publish(Event(type=EventType.FRAME_CAPTURED, source="camera"))

    def _on_detection_result(self, topic: str, data: dict) -> None:
        """Handle detection results from ZMQ subscriber."""
        if topic != "detection":
            return

        results_data = data.get("results", [])
        if not results_data:
            return

        # Convert back to DetectionResult for notification
        detection_results = []
        for r in results_data:
            from larksnap.adapters.detector.interface import BBox

            bbox_data = r.get("bbox", {})
            detection_results.append(
                DetectionResult(
                    label=r["label"],
                    confidence=r["confidence"],
                    bbox=BBox(
                        x=bbox_data.get("x", 0),
                        y=bbox_data.get("y", 0),
                        width=bbox_data.get("width", 0),
                        height=bbox_data.get("height", 0),
                    ),
                )
            )

        # Save snapshot and notify (only for labels that pass notification interval)
        frame = self.get_latest_frame()
        self._notify_results(detection_results, frame)

    def _filter_results(self, results: list[DetectionResult]) -> list[DetectionResult]:
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
        self, results: list[DetectionResult], frame: np.ndarray | None = None
    ) -> None:
        if self._notifier is None:
            return

        now = time.time()
        interval = self._config.gateway.notification_interval

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
        adapters = [self._notifier, self._detector, self._camera, self._recorder]
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
        self._recorder = None
