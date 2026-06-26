"""Pipeline module encapsulating ZMQ-based frame processing stages.

Manages the data flow: Camera → FrameProducer → ZMQ → FrameConsumer → callback.
Owns its configuration (PipelineConfig) and ZMQ infrastructure.

Concurrency contract:
  - ``_running`` and ``_paused`` are guarded by ``_state_lock`` so
    the writer thread (start/stop/pause/resume) and reader threads
    (FrameConsumer, FrameProducer, ResultSubscriber) always see a
    consistent view.
  - ``_latest_frame`` and ``_latest_results`` are guarded by
    ``_frame_lock``; only the owning thread mutates them.
  - Fatal camera errors are delivered to the gateway via a thread-safe
    ``queue.Queue`` rather than a one-shot thread spawned by the
    producer. The gateway drains this queue on its own background
    thread, so producer→gateway communication is bounded and
    deterministic.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any

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

    FRAME_QUEUE_URL = "inproc://larksnap_frame_queue"
    RESULT_QUEUE_URL = "inproc://larksnap_detection_results"

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

        # Per-instance inproc URLs. Using fixed names is unsafe with
        # the singleton zmq context: an old bound endpoint can
        # survive across open/close cycles and the second ``bind``
        # will fail with ``Address in use``. The per-instance suffix
        # eliminates that class of bug.
        import uuid
        self._instance_id = uuid.uuid4().hex[:8]
        self._frame_queue_url = f"inproc://larksnap_frame_queue_{self._instance_id}"
        self._result_queue_url = f"inproc://larksnap_detection_results_{self._instance_id}"

        self._zmq_context: zmq.Context | None = None
        self._owns_context: bool = False  # True only if we created a non-singleton ctx
        self._producer: FrameProducer | None = None
        self._detector_consumer: FrameConsumer | None = None
        self._preview_consumer: FrameConsumer | None = None
        self._result_publisher: ResultPublisher | None = None
        self._result_subscriber: ResultSubscriber | None = None

        # All state mutations are serialised by this lock. The booleans
        # are read by both the writer thread (start/stop/pause/resume)
        # and the reader threads (consumer loops), so unsynchronised
        # access could let a consumer see stale "running" after stop()
        # returns and busy-loop briefly on a torn-down socket.
        self._state_lock = threading.RLock()
        self._running = False
        self._paused = False
        self._detection_count: int = 0
        self._latest_frame: np.ndarray | None = None
        self._latest_results: list[DetectionResult] = []
        self._frame_lock = threading.Lock()

        # Fatal-error delivery from FrameProducer → gateway. The producer
        # pushes messages here, the gateway drains it on its own thread.
        # Bounded so a runaway producer can't grow this queue unbounded.
        self._fatal_error_queue: queue.Queue[str] = queue.Queue(maxsize=16)
        self._fatal_error_drain_thread: threading.Thread | None = None
        self._fatal_error_drain_stop = threading.Event()

        # Callbacks set by controller
        self._on_results_callback = None

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._running

    @property
    def is_paused(self) -> bool:
        with self._state_lock:
            return self._paused

    @property
    def detection_count(self) -> int:
        with self._state_lock:
            return self._detection_count

    @property
    def producer_fps(self) -> float:
        if self._producer is not None:
            return self._producer.fps
        return 0.0

    @property
    def fatal_error_queue(self) -> queue.Queue[str]:
        """Queue where the producer posts fatal camera errors.

        The gateway should drain this on a dedicated thread and act on
        each message (typically by closing the pipeline). Sized to
        be small; messages are coalesced because once the producer has
        declared a fatal error, additional messages are redundant.
        """
        return self._fatal_error_queue

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
        with self._state_lock:
            if self._running:
                return
            self._running = True
            self._detection_count = 0
            self._paused = False

        # Use the global ZMQ context singleton (not a per-pipeline
        # one). This eliminates the entire class of "term() raced
        # with an in-flight poll()" crashes that plagued the
        # previous per-pipeline-context design. The singleton lives
        # for the lifetime of the process, which is fine for a
        # desktop app. We simply close our own sockets on stop().
        self._zmq_context = zmq.Context.instance()
        self._owns_context = False
        policy = FrameQueuePolicy(self._config.frame_queue_policy)

        # Reset the fatal-error drain: a new pipeline instance may
        # legitimately be the first place a new error appears.
        self._fatal_error_queue = queue.Queue(maxsize=16)
        self._fatal_error_drain_stop.clear()
        if self._fatal_error_drain_thread is None or not self._fatal_error_drain_thread.is_alive():
            self._fatal_error_drain_thread = threading.Thread(
                target=self._fatal_error_drain_loop,
                name="Pipeline-fatal-drain",
                daemon=True,
            )
            self._fatal_error_drain_thread.start()

        # Start ZMQ result publisher
        self._result_publisher = ResultPublisher(zmq_url=self._result_queue_url)
        self._result_publisher.start(context=self._zmq_context)

        # Start frame producer (camera → ZMQ)
        self._producer = FrameProducer(
            camera=self._camera,
            zmq_url=self._frame_queue_url,
            policy=policy,
            hwm=self._config.frame_queue_hwm,
        )
        self._producer.set_fatal_error_queue(self._fatal_error_queue)
        self._producer.start(context=self._zmq_context)

        # Start detector consumer (ZMQ → detector → result publisher)
        self._detector_consumer = FrameConsumer(
            zmq_url=self._frame_queue_url, hwm=10
        )
        self._detector_consumer.start(
            on_frame=self._on_detection_frame,
            context=self._zmq_context,
        )

        # Start preview consumer (ZMQ → latest frame cache)
        self._preview_consumer = FrameConsumer(
            zmq_url=self._frame_queue_url, hwm=5
        )
        self._preview_consumer.start(
            on_frame=self._on_preview_frame,
            context=self._zmq_context,
        )

        # Start result subscriber (results → notification callback)
        self._result_subscriber = ResultSubscriber(
            zmq_url=self._result_queue_url, topic=""
        )
        self._result_subscriber.start(
            on_result=self._on_detection_result,
            context=self._zmq_context,
        )

        self._logger.info("Pipeline started")

    def stop(self) -> None:
        """Stop the pipeline and release ZMQ resources.

        Every blocking operation is bounded so this method can never
        hang the calling thread indefinitely — important because it's
        invoked from the controller's background close worker, and
        any hang there would prevent the gateway from ever reaching
        IDLE again.

        The ZMQ context itself is never terminated here. We use the
        global ``zmq.Context.instance()`` singleton, which lives for
        the process lifetime. Calling ``Context.term()`` while IO
        threads are still inside ``poll()`` is a well-known crash
        trigger (access violation on Windows); the singleton pattern
        sidesteps that problem entirely.
        """
        with self._state_lock:
            if not self._running:
                return
            self._running = False

        # Stop the fatal-error drain thread before tearing down
        # components. The drain thread is daemon and exits when it
        # sees the stop event; we don't need to wait for it because
        # no caller depends on its completion.
        self._fatal_error_drain_stop.set()

        # Stop all components (closes their ZMQ sockets). Each
        # component.stop() has internal timeouts so a stuck IO thread
        # cannot block us here.
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

        # Drop our reference to the context. The singleton survives
        # in zmq.Context.instance() so we never call term/destroy.
        # If a non-singleton context is ever used (e.g. in tests),
        # fall back to the legacy bounded background-term path, but
        # only when all worker threads are confirmed exited.
        if self._zmq_context is not None:
            if self._owns_context:
                ctx = self._zmq_context
                self._zmq_context = None
                done = threading.Event()

                def _term() -> None:
                    try:
                        ctx.destroy(linger=0)
                    except Exception as e:
                        self._logger.error("ZMQ context.destroy() raised: %s", e)
                    finally:
                        done.set()

                t = threading.Thread(target=_term, name="ZMQ-destroy", daemon=True)
                t.start()
                if not done.wait(timeout=2.0):
                    self._logger.warning(
                        "ZMQ context.destroy() exceeded 2.0s budget; abandoning"
                    )
            else:
                # Singleton: just drop the reference.
                self._zmq_context = None

        self._logger.info("Pipeline stopped")

    def pause(self) -> None:
        """Pause detection (producer keeps running for preview)."""
        with self._state_lock:
            self._paused = True
        self._logger.info("Pipeline paused (detection paused, preview active)")

    def resume(self) -> None:
        """Resume detection."""
        with self._state_lock:
            self._paused = False
        self._logger.info("Pipeline resumed")

    def _on_detection_frame(self, packet: FramePacket) -> None:
        """Handle frame from detector consumer: run detection and publish results."""
        # Snapshot the paused flag under the lock so the writer
        # thread (pause/resume) can change it without us needing to
        # hold the lock during the expensive detection call.
        with self._state_lock:
            paused = self._paused
            running = self._running
        if not running or paused:
            return

        try:
            results = self._detector.detect(packet.frame)
            with self._state_lock:
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
            # FramePacket.frame is a fresh C-contiguous array (forced
            # in ``from_zmq_frames``), so swapping the reference is
            # safe. The previous frame is left to GC; we deliberately
            # don't synchronise with the UI thread's reads because
            # get_latest_frame() takes ``_frame_lock`` itself.
            self._latest_frame = packet.frame

        self._event_bus.publish(Event(type=EventType.FRAME_CAPTURED, source="camera"))

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

        # Invoke notification callback with results and current frame.
        # The callback (NotificationService) is responsible for offloading
        # any slow I/O to its own worker pool — the ZMQ thread must
        # not block on HTTP or disk writes.
        if self._on_results_callback is not None:
            frame = self.get_latest_frame()
            self._on_results_callback(detection_results, frame)

    def _fatal_error_drain_loop(self) -> None:
        """Drain fatal errors from the queue and forward them as events.

        Runs on a dedicated daemon thread so the ZMQ consumer/producer
        threads never need to spawn their own threads to invoke
        callbacks. Each error message is published as a CAMERA_READ_FAILED
        event exactly once, then the gateway's own subscriber can act
        on it (typically by calling pipeline.stop() and the
        controller's close path).
        """
        while not self._fatal_error_drain_stop.is_set():
            try:
                error_message = self._fatal_error_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._fatal_error_drain_stop.is_set():
                break
            self._logger.error("Frame producer fatal error: %s", error_message)
            try:
                self._event_bus.publish(Event(
                    type=EventType.CAMERA_READ_FAILED,
                    data={"error": error_message},
                    source="pipeline",
                ))
            except Exception as e:  # noqa: BLE001
                self._logger.error("Publish CAMERA_READ_FAILED failed: %s", e)
