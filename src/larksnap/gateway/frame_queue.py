"""ZeroMQ-based frame queue for producer-consumer video stream architecture.

Provides FrameProducer (camera → ZMQ PUSH) and FrameConsumer (ZMQ PULL → model)
with support for inproc/TCP transport, backpressure, and frame metadata.
"""

from __future__ import annotations

import json
import logging
import queue
import struct
import threading
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
import zmq

from larksnap.adapters.camera.interface import CameraAdapter
from larksnap.utils.exceptions import GatewayError


class FrameQueuePolicy(str, Enum):
    """Policy for handling backpressure when queue is full."""

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"


@dataclass
class FramePacket:
    """Serialized frame data with metadata for ZMQ transport."""

    frame: np.ndarray
    timestamp: float
    frame_index: int
    width: int
    height: int
    channels: int

    def to_zmq_frames(self) -> list[bytes]:
        """Serialize to ZMQ multi-frame message: [metadata, frame_data]."""
        metadata = json.dumps({
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": str(self.frame.dtype),
        }).encode("utf-8")
        frame_data = self.frame.tobytes()
        return [metadata, frame_data]

    @classmethod
    def from_zmq_frames(cls, metadata_bytes: bytes, frame_data: bytes) -> FramePacket:
        """Deserialize from ZMQ multi-frame message.

        The deserialized frame is forced to be C-contiguous so that
        downstream OpenCV calls won't trip the _step >= minstep assertion.
        """
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        h, w, c = int(metadata["height"]), int(metadata["width"]), int(metadata["channels"])
        dtype = np.dtype(metadata["dtype"])
        expected_bytes = h * w * c * dtype.itemsize
        if len(frame_data) < expected_bytes:
            raise ValueError(
                f"Frame data truncated: got {len(frame_data)} bytes, "
                f"expected {expected_bytes} for shape ({h},{w},{c}) {dtype}"
            )
        frame = np.frombuffer(frame_data, dtype=dtype).reshape(h, w, c)
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        return cls(
            frame=frame,
            timestamp=metadata["timestamp"],
            frame_index=metadata["frame_index"],
            width=w,
            height=h,
            channels=c,
        )


class FrameProducer:
    """Captures frames from camera and pushes them to ZMQ socket.

    Runs in a dedicated thread, decoupling camera capture from model inference.

    Implements exponential backoff retry on read failures and stops after
    max_consecutive_failures to avoid resource waste when camera is unavailable.
    """

    # Retry configuration
    INITIAL_BACKOFF_MS: int = 1000       # Initial backoff on read failure
    MAX_BACKOFF_MS: int = 5000          # Maximum backoff cap
    BACKOFF_MULTIPLIER: float = 1.5    # Exponential growth factor
    MAX_CONSECUTIVE_FAILURES: int = 5  # Stop producer after this many consecutive failures

    def __init__(
        self,
        camera: CameraAdapter,
        zmq_url: str = "inproc://frame_queue",
        policy: FrameQueuePolicy = FrameQueuePolicy.DROP_OLDEST,
        hwm: int = 30,
    ) -> None:
        self._camera = camera
        self._zmq_url = zmq_url
        self._policy = policy
        self._hwm = hwm
        self._logger = logging.getLogger("larksnap.frame_producer")
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_index = 0
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._fps: float = 0.0
        self._last_fps_time: float = 0.0
        self._fps_frame_count: int = 0
        self._consecutive_failures: int = 0
        self._current_backoff_ms: int = self.INITIAL_BACKOFF_MS
        self._fatal_error_queue: queue.Queue[str] | None = None
        self._last_friendly_error: str = ""

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def set_fatal_error_queue(self, error_queue: "queue.Queue[str]") -> None:
        """Set a queue to receive fatal camera error messages.

        Replacing the previous callback+thread-spawn design with a
        queue keeps the producer thread bounded and predictable: the
        worker thread can do a non-blocking ``put_nowait`` rather
        than spawning an orphan thread, and the receiver can drain
        the queue on its own dedicated thread (e.g. the pipeline's
        fatal-error drain).
        """
        self._fatal_error_queue = error_queue

    def start(self, context: zmq.Context | None = None) -> None:
        """Start the frame producer thread."""
        if self._running:
            return

        self._context = context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.set_hwm(self._hwm)
        self._socket.bind(self._zmq_url)

        self._running = True
        self._frame_index = 0
        self._fps_frame_count = 0
        self._last_fps_time = time.time()
        self._consecutive_failures = 0
        self._current_backoff_ms = self.INITIAL_BACKOFF_MS
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._logger.info("Frame producer started on %s", self._zmq_url)

    def stop(self) -> None:
        """Stop the frame producer.

        Bounded close: the worker thread is given at most 2.0s to
        exit before we move on. This guarantees the controller's
        close path never blocks indefinitely on a stuck camera read.
        """
        self._running = False
        # Close socket first to unblock any pending ZMQ operations
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception as e:  # noqa: BLE001
                self._logger.error("Producer socket close failed: %s", e)
            self._socket = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                self._logger.warning(
                    "Producer thread did not exit within 2.0s; abandoning"
                )
            self._thread = None
        self._logger.info("Frame producer stopped")

    def _run_loop(self) -> None:
        while self._running:
            # Fast exit if the socket was torn down by stop()
            if self._socket is None:
                break
            try:
                frame = self._camera.read_frame()

                # Re-check after read_frame(): stop() may have run while we
                # were blocked on the camera read, leaving the socket gone.
                if not self._running or self._socket is None:
                    break

                packet = FramePacket(
                    frame=frame,
                    timestamp=time.time(),
                    frame_index=self._frame_index,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    channels=frame.shape[2] if frame.ndim == 3 else 1,
                )

                if self._policy == FrameQueuePolicy.DROP_OLDEST:
                    try:
                        self._socket.send_multipart(packet.to_zmq_frames(), flags=zmq.NOBLOCK)
                    except zmq.Again:
                        # Queue full, drain oldest and retry
                        self._logger.debug("Queue full, dropping oldest frame")
                        try:
                            self._socket.send_multipart(packet.to_zmq_frames(), flags=zmq.NOBLOCK)
                        except zmq.Again:
                            pass
                elif self._policy == FrameQueuePolicy.DROP_NEWEST:
                    try:
                        self._socket.send_multipart(packet.to_zmq_frames(), flags=zmq.NOBLOCK)
                    except zmq.Again:
                        self._logger.debug("Queue full, dropping current frame")
                else:
                    self._socket.send_multipart(packet.to_zmq_frames())

                self._frame_index += 1
                self._fps_frame_count += 1
                self._update_fps()

                # Reset failure tracking on successful read
                if self._consecutive_failures > 0:
                    self._logger.info(
                        "Camera read recovered after %d consecutive failures",
                        self._consecutive_failures,
                    )
                self._consecutive_failures = 0
                self._current_backoff_ms = self.INITIAL_BACKOFF_MS

            except AttributeError as e:
                # 'NoneType' object has no attribute 'send_multipart' /
                # 'shape' — happens when stop() tore down the socket or
                # the camera between read_frame() and packet construction.
                # Treat as a clean shutdown, not a camera failure.
                if not self._running or self._socket is None:
                    break
                self._logger.debug("AttributeError during producer loop: %s", e)
                time.sleep(0.1)
            except Exception as e:
                # Translate raw OpenCV/MSMF errors to user-friendly messages
                try:
                    from larksnap.utils.camera_error_translator import (
                        translate_camera_error,
                    )
                    friendly_msg = translate_camera_error(str(e))
                except Exception:
                    friendly_msg = str(e)

                self._consecutive_failures += 1
                # Stash the friendly message for the fatal error callback
                self._last_friendly_error = friendly_msg

                if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    self._logger.error(
                        "Camera read failed %d consecutive times, stopping producer: %s",
                        self._consecutive_failures, friendly_msg,
                    )
                    self._running = False
                    # Hand the error to a queue instead of spawning
                    # a thread. The receiver (pipeline) drains the
                    # queue on its own thread, so the producer never
                    # has to manage thread lifecycle of an orphan.
                    # put_nowait is non-blocking and bounded; if the
                    # receiver is wedged, we drop the message and
                    # log it, which is the safe degradation.
                    if self._fatal_error_queue is not None:
                        try:
                            self._fatal_error_queue.put_nowait(friendly_msg)
                        except queue.Full:
                            self._logger.error(
                                "Fatal error queue is full; dropping message: %s",
                                friendly_msg,
                            )
                    break

                # Log with appropriate frequency
                log_interval = 1
                if self._consecutive_failures > 10:
                    log_interval = 10
                elif self._consecutive_failures > 5:
                    log_interval = 5

                if self._consecutive_failures % log_interval == 0:
                    self._logger.warning(
                        "Camera read failure %d/%d: %s (backoff: %.1fs)",
                        self._consecutive_failures,
                        self.MAX_CONSECUTIVE_FAILURES,
                        friendly_msg,
                        self._current_backoff_ms / 1000.0,
                    )

                # Exponential backoff
                time.sleep(self._current_backoff_ms / 1000.0)
                self._current_backoff_ms = min(
                    int(self._current_backoff_ms * self.BACKOFF_MULTIPLIER),
                    self.MAX_BACKOFF_MS,
                )

    def _update_fps(self) -> None:
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._last_fps_time = now


class FrameConsumer:
    """Pulls frames from ZMQ socket and dispatches to handler callbacks.

    Supports multiple consumers for parallel model inference.
    """

    def __init__(
        self,
        zmq_url: str = "inproc://frame_queue",
        hwm: int = 30,
    ) -> None:
        self._zmq_url = zmq_url
        self._hwm = hwm
        self._logger = logging.getLogger("larksnap.frame_consumer")
        self._running = False
        self._thread: threading.Thread | None = None
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._on_frame = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(
        self,
        on_frame,
        context: zmq.Context | None = None,
    ) -> None:
        """Start the frame consumer thread.

        Args:
            on_frame: Callback function receiving FramePacket.
            context: Optional shared ZMQ context.
        """
        if self._running:
            return

        self._on_frame = on_frame
        self._context = context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.PULL)
        self._socket.set_hwm(self._hwm)
        self._socket.connect(self._zmq_url)

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._logger.info("Frame consumer started on %s", self._zmq_url)

    def stop(self) -> None:
        """Stop the frame consumer (bounded).

        Like ``FrameProducer.stop``: socket close is best-effort and
        the worker thread is given 2.0s to exit.
        """
        self._running = False
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception as e:  # noqa: BLE001
                self._logger.error("Consumer socket close failed: %s", e)
            self._socket = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                self._logger.warning(
                    "Consumer thread did not exit within 2.0s; abandoning"
                )
            self._thread = None
        self._logger.info("Frame consumer stopped")

    def _run_loop(self) -> None:
        while self._running:
            try:
                if self._socket is None:
                    break

                if not self._socket.poll(timeout=1000):
                    continue

                # Re-check after poll: stop() may have torn down the socket
                if not self._running or self._socket is None:
                    break

                frames = self._socket.recv_multipart()
                if len(frames) != 2:
                    self._logger.warning("Invalid frame message, expected 2 parts")
                    continue

                packet = FramePacket.from_zmq_frames(frames[0], frames[1])
                if not self._running:
                    break
                if self._on_frame is not None:
                    self._on_frame(packet)

            except zmq.ZMQError as e:
                if not self._running or self._socket is None:
                    break
                self._logger.error("Frame consumer ZMQ error: %s", e)
            except AttributeError as e:
                # Socket torn down by stop() between iterations
                if not self._running or self._socket is None:
                    break
                self._logger.debug("AttributeError in consumer loop: %s", e)
            except Exception as e:
                if not self._running or self._socket is None:
                    break
                self._logger.error("Frame consumer error: %s", e)


class ResultPublisher:
    """Publishes detection results via ZMQ PUB socket.

    Used by model consumers to broadcast detection results to
    result subscribers (UI, notifier, recorder).
    """

    def __init__(
        self,
        zmq_url: str = "inproc://detection_results",
    ) -> None:
        self._zmq_url = zmq_url
        self._logger = logging.getLogger("larksnap.result_publisher")
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None

    def start(self, context: zmq.Context | None = None) -> None:
        self._context = context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(self._zmq_url)
        self._logger.info("Result publisher started on %s", self._zmq_url)

    def stop(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._logger.info("Result publisher stopped")

    def publish(self, topic: str, data: dict) -> None:
        """Publish a detection result message."""
        if self._socket is None:
            return
        try:
            topic_bytes = topic.encode("utf-8")
            data_bytes = json.dumps(data, default=str).encode("utf-8")
            self._socket.send_multipart([topic_bytes, data_bytes], flags=zmq.NOBLOCK)
        except zmq.Again:
            self._logger.debug("Result publish dropped (no subscribers or full)")
        except Exception as e:
            self._logger.error("Result publish error: %s", e)


class ResultSubscriber:
    """Subscribes to detection results via ZMQ SUB socket.

    Used by UI, notifier, and recorder to receive detection results.
    """

    def __init__(
        self,
        zmq_url: str = "inproc://detection_results",
        topic: str = "",
    ) -> None:
        self._zmq_url = zmq_url
        self._topic = topic
        self._logger = logging.getLogger("larksnap.result_subscriber")
        self._running = False
        self._thread: threading.Thread | None = None
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None

    def start(
        self,
        on_result,
        context: zmq.Context | None = None,
    ) -> None:
        if self._running:
            return

        self._on_result = on_result
        self._context = context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.set_hwm(30)
        self._socket.connect(self._zmq_url)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, self._topic)

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._logger.info("Result subscriber started on %s (topic=%s)", self._zmq_url, self._topic or "*")

    def stop(self) -> None:
        self._running = False
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception as e:  # noqa: BLE001
                self._logger.error("Result subscriber socket close failed: %s", e)
            self._socket = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                self._logger.warning(
                    "Result subscriber thread did not exit within 2.0s; abandoning"
                )
            self._thread = None
        self._logger.info("Result subscriber stopped")

    def _run_loop(self) -> None:
        while self._running:
            try:
                if self._socket is None:
                    break

                if not self._socket.poll(timeout=1000):
                    continue

                if not self._running or self._socket is None:
                    break

                frames = self._socket.recv_multipart()
                if len(frames) != 2:
                    continue

                if not self._running:
                    break
                topic = frames[0].decode("utf-8")
                data = json.loads(frames[1].decode("utf-8"))
                if self._on_result is not None:
                    self._on_result(topic, data)

            except zmq.ZMQError as e:
                if not self._running or self._socket is None:
                    break
                self._logger.error("Result subscriber ZMQ error: %s", e)
            except AttributeError:
                if not self._running or self._socket is None:
                    break
            except Exception as e:
                if not self._running or self._socket is None:
                    break
                self._logger.error("Result subscriber error: %s", e)
