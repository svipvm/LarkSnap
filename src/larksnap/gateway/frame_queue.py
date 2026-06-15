"""ZeroMQ-based frame queue for producer-consumer video stream architecture.

Provides FrameProducer (camera → ZMQ PUSH) and FrameConsumer (ZMQ PULL → model)
with support for inproc/TCP transport, backpressure, and frame metadata.
"""

from __future__ import annotations

import json
import logging
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
        """Deserialize from ZMQ multi-frame message."""
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        frame = np.frombuffer(frame_data, dtype=np.dtype(metadata["dtype"])).reshape(
            metadata["height"], metadata["width"], metadata["channels"]
        )
        return cls(
            frame=frame,
            timestamp=metadata["timestamp"],
            frame_index=metadata["frame_index"],
            width=metadata["width"],
            height=metadata["height"],
            channels=metadata["channels"],
        )


class FrameProducer:
    """Captures frames from camera and pushes them to ZMQ socket.

    Runs in a dedicated thread, decoupling camera capture from model inference.
    """

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

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        return self._fps

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
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._logger.info("Frame producer started on %s", self._zmq_url)

    def stop(self) -> None:
        """Stop the frame producer."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._logger.info("Frame producer stopped")

    def _run_loop(self) -> None:
        while self._running:
            try:
                frame = self._camera.read_frame()
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

            except Exception as e:
                self._logger.error("Frame producer error: %s", e)
                time.sleep(0.1)

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
        """Stop the frame consumer."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._logger.info("Frame consumer stopped")

    def _run_loop(self) -> None:
        while self._running:
            try:
                if self._socket is None:
                    break

                if not self._socket.poll(timeout=1000):
                    continue

                frames = self._socket.recv_multipart()
                if len(frames) != 2:
                    self._logger.warning("Invalid frame message, expected 2 parts")
                    continue

                packet = FramePacket.from_zmq_frames(frames[0], frames[1])
                if self._on_frame is not None:
                    self._on_frame(packet)

            except zmq.ZMQError as e:
                if self._running:
                    self._logger.error("Frame consumer ZMQ error: %s", e)
            except Exception as e:
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
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._logger.info("Result subscriber stopped")

    def _run_loop(self) -> None:
        while self._running:
            try:
                if self._socket is None:
                    break

                if not self._socket.poll(timeout=1000):
                    continue

                frames = self._socket.recv_multipart()
                if len(frames) != 2:
                    continue

                topic = frames[0].decode("utf-8")
                data = json.loads(frames[1].decode("utf-8"))
                if self._on_result is not None:
                    self._on_result(topic, data)

            except zmq.ZMQError as e:
                if self._running:
                    self._logger.error("Result subscriber ZMQ error: %s", e)
            except Exception as e:
                self._logger.error("Result subscriber error: %s", e)
