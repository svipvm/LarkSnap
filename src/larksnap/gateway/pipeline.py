"""Pipeline module encapsulating ZMQ-based frame processing stages.

Manages the data flow: Camera → FrameProducer → ZMQ → FrameConsumer → callback.
Owns its configuration (PipelineConfig) and ZMQ infrastructure.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
import zmq

from larksnap.adapters.camera.interface import CameraAdapter
from larksnap.adapters.detector.interface import DetectionResult, DetectorAdapter
from larksnap.gateway.event_bus import Event, EventBus, EventType
from larksnap.gateway.frame_queue import (
    FrameConsumer,
    FramePacket,
    FrameProducer,
    FrameQueuePolicy,
    ResultPublisher,
    ResultSubscriber,
)


@dataclass
class PipelineConfig:
    """Configuration owned by the Pipeline."""

    frame_queue_hwm: int = 30
    frame_queue_policy: str = "drop_oldest"


class Pipeline:
    """ZMQ-based frame processing pipeline.

    Manages the full data flow:
        Camera → FrameProducer → ZMQ PUSH/PULL → DetectorConsumer → Detector
        Detector → ResultPublisher → ZMQ PUB/SUB → ResultSubscriber → callback

    Each stage owns its own ZMQ socket. The pipeline is started/stopped as a unit.
    """

    FRAME_QUEUE_URL = "inproc://frame_queue"
    RESULT_QUEUE_URL = "inproc://detection_results"

    def __init__(
        self,
        camera: CameraAdapter,
        detector: DetectorAdapter,
        config: PipelineConfig,
        event_bus: EventBus,
    ) -> None:
        self._camera = camera
        self._detector = detector
        self._config = config
        self._event_bus = event_bus
        self._logger = logging.getLogger("larksnap.pipeline")

        self._zmq_context: zmq.Context | None = None
        self._producer: FrameProducer | None = None
        self._detector_consumer: FrameConsumer | None = None
        self._preview_consumer: FrameConsumer | None = None
        self._result_publisher: ResultPublisher | None = None
        self._result_subscriber: ResultSubscriber | None = None

        self._running = False
        self._paused = False
        self._detection_count: int = 0
        self._latest_frame: np.ndarray | None = None
        self._latest_results: list[DetectionResult] = []
        self._frame_lock = threading.Lock()

        # Callbacks set by controller
        self._on_results_callback = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def detection_count(self) -> int:
        return self._detection_count

    @property
    def producer_fps(self) -> float:
        if self._producer is not None:
            return self._producer.fps
        return 0.0

    @property
    def config(self) -> PipelineConfig:
        return self._config

    def get_latest_frame(self) -> np.ndarray | None:
        """Get the latest camera frame (thread-safe)."""
        with self._frame_lock:
            return self._latest_frame

    def get_latest_results(self) -> list[DetectionResult]:
        """Get the latest detection results (thread-safe)."""
        with self._frame_lock:
            return list(self._latest_results)

    def set_on_results(self, callback) -> None:
        """Set callback for when detection results are ready for notification.

        callback(results: list[DetectionResult], frame: np.ndarray | None)
        """
        self._on_results_callback = callback

    def start(self) -> None:
        """Start the pipeline."""
        if self._running:
            return

        # Use a dedicated ZMQ context (not the global singleton)
        # so we can safely terminate it on stop without affecting other users.
        self._zmq_context = zmq.Context()
        policy = FrameQueuePolicy(self._config.frame_queue_policy)

        # Start ZMQ result publisher
        self._result_publisher = ResultPublisher(zmq_url=self.RESULT_QUEUE_URL)
        self._result_publisher.start(context=self._zmq_context)

        # Start frame producer (camera → ZMQ)
        self._producer = FrameProducer(
            camera=self._camera,
            zmq_url=self.FRAME_QUEUE_URL,
            policy=policy,
            hwm=self._config.frame_queue_hwm,
        )
        self._producer.set_on_fatal_error(self._on_producer_fatal_error)
        self._producer.start(context=self._zmq_context)

        # Start detector consumer (ZMQ → detector → result publisher)
        self._detector_consumer = FrameConsumer(
            zmq_url=self.FRAME_QUEUE_URL, hwm=10
        )
        self._detector_consumer.start(
            on_frame=self._on_detection_frame,
            context=self._zmq_context,
        )

        # Start preview consumer (ZMQ → latest frame cache)
        self._preview_consumer = FrameConsumer(
            zmq_url=self.FRAME_QUEUE_URL, hwm=5
        )
        self._preview_consumer.start(
            on_frame=self._on_preview_frame,
            context=self._zmq_context,
        )

        # Start result subscriber (results → notification callback)
        self._result_subscriber = ResultSubscriber(
            zmq_url=self.RESULT_QUEUE_URL, topic=""
        )
        self._result_subscriber.start(
            on_result=self._on_detection_result,
            context=self._zmq_context,
        )

        self._running = True
        self._detection_count = 0
        self._logger.info("Pipeline started")

    def stop(self) -> None:
        """Stop the pipeline and release ZMQ resources."""
        if not self._running:
            return

        self._running = False

        # Stop all components (closes their ZMQ sockets)
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

        self._producer = None
        self._detector_consumer = None
        self._preview_consumer = None
        self._result_subscriber = None
        self._result_publisher = None

        # Terminate ZMQ context to ensure all associated threads exit
        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None

        self._logger.info("Pipeline stopped")

    def pause(self) -> None:
        """Pause detection (producer keeps running for preview)."""
        self._paused = True
        self._logger.info("Pipeline paused (detection paused, preview active)")

    def resume(self) -> None:
        """Resume detection."""
        self._paused = False
        self._logger.info("Pipeline resumed")

    def _on_detection_frame(self, packet: FramePacket) -> None:
        """Handle frame from detector consumer: run detection and publish results."""
        if self._paused:
            return

        try:
            results = self._detector.detect(packet.frame)
            self._detection_count += 1

            # Update latest results
            with self._frame_lock:
                self._latest_results = results

            self._event_bus.publish(
                Event(type=EventType.DETECTION_COMPLETED, data=results, source="detector")
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
                        for r in results
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

    def _on_producer_fatal_error(self, error_message: str) -> None:
        """Handle fatal camera read errors from FrameProducer.

        Called when the producer stops itself after too many consecutive failures.
        Publishes an event and stops the pipeline.
        """
        self._logger.error("Frame producer fatal error: %s", error_message)
        self._event_bus.publish(Event(
            type=EventType.CAMERA_READ_FAILED,
            data={"error": error_message},
            source="pipeline",
        ))
        # Stop the pipeline since we can't produce frames
        self.stop()

    def _on_detection_result(self, topic: str, data: dict) -> None:
        """Handle detection results from ZMQ subscriber → notification callback."""
        if topic != "detection":
            return

        results_data = data.get("results", [])
        if not results_data:
            return

        # Convert back to DetectionResult
        from larksnap.adapters.detector.interface import BBox

        detection_results = []
        for r in results_data:
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

        # Invoke notification callback with results and current frame
        if self._on_results_callback is not None:
            frame = self.get_latest_frame()
            self._on_results_callback(detection_results, frame)
