"""Video recorder adapter using OpenCV VideoWriter.

Consumes frames from ZMQ frame queue and writes them to video files.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from larksnap.adapters.base import BaseAdapter
from larksnap.gateway.frame_queue import FrameConsumer, FramePacket


class VideoRecorderAdapter(BaseAdapter):
    """Records video frames to file using OpenCV VideoWriter.

    Subscribes to the ZMQ frame queue and writes frames to a video file.
    Supports start/stop recording with configurable output directory and codec.
    """

    def __init__(
        self,
        output_dir: str = "recordings",
        fps: float = 30.0,
        codec: str = "mp4v",
        frame_queue_url: str = "inproc://frame_queue",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._fps = fps
        self._codec = codec
        self._frame_queue_url = frame_queue_url
        self._logger = logging.getLogger("larksnap.recorder")
        self._writer: cv2.VideoWriter | None = None
        self._consumer: FrameConsumer | None = None
        self._recording = False
        self._lock = threading.Lock()
        self._current_file: str | None = None
        self._frame_count: int = 0
        self._start_time: float = 0.0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def current_file(self) -> str | None:
        return self._current_file

    @property
    def duration(self) -> float:
        if not self._recording:
            return 0.0
        return time.time() - self._start_time

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def initialize(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info("Video recorder initialized, output dir: %s", self._output_dir)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.stop_recording()

    def release(self) -> None:
        self.stop_recording()

    def start_recording(self, context=None) -> None:
        """Start recording frames from the frame queue."""
        with self._lock:
            if self._recording:
                self._logger.warning("Already recording")
                return

            self._frame_count = 0
            self._start_time = time.time()
            self._consumer = FrameConsumer(
                zmq_url=self._frame_queue_url,
                hwm=60,
            )
            self._consumer.start(
                on_frame=self._on_frame,
                context=context,
            )
            self._recording = True
            self._logger.info("Video recording started")

    def stop_recording(self) -> None:
        """Stop recording and close the video file."""
        with self._lock:
            if not self._recording:
                return

            self._recording = False
            if self._consumer is not None:
                self._consumer.stop()
                self._consumer = None
            if self._writer is not None:
                self._writer.release()
                self._writer = None
                self._logger.info(
                    "Video recording stopped: %s (%d frames, %.1fs)",
                    self._current_file,
                    self._frame_count,
                    self.duration,
                )
            self._current_file = None

    def _on_frame(self, packet: FramePacket) -> None:
        """Handle a frame from the consumer."""
        if not self._recording:
            return

        with self._lock:
            if not self._recording:
                return

            if self._writer is None:
                fourcc = cv2.VideoWriter_fourcc(*self._codec)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}.mp4"
                self._current_file = str(self._output_dir / filename)
                self._writer = cv2.VideoWriter(
                    self._current_file,
                    fourcc,
                    self._fps,
                    (packet.width, packet.height),
                )
                if not self._writer.isOpened():
                    self._logger.error("Failed to open video writer: %s", self._current_file)
                    self._writer = None
                    self._recording = False
                    return
                self._logger.info("Recording to: %s", self._current_file)

            self._writer.write(packet.frame)
            self._frame_count += 1
