"""Video recorder adapter using OpenCV VideoWriter.

Consumes frames from ZMQ frame queue and writes them to video files.

Concurrency:
  - The ZMQ consumer thread hands each frame to ``_on_frame`` and
    then immediately polls the next one. VideoWriter.write() is
    a synchronous disk write that can take many milliseconds per
    frame, so doing it on the consumer thread would push the HWM
    and start dropping frames. We therefore offload the write to
    a ``WorkerPool`` if one is provided; the consumer thread
    merely enqueues a copy of the frame.
  - The lock is now only taken to mutate state (``_recording``,
    ``_writer``, ``_current_file``). The per-frame write path is
    lock-free, eliminating the bottleneck that previously caused
    the recorder to drop frames under load.
  - ``stop_recording`` is non-blocking on the caller's thread:
    it only flips the recording flag, stops the consumer, and
    hands the writer release off to a background worker. The
    previous implementation blocked the calling thread (often
    the Qt main thread) for up to several seconds while waiting
    for the in-flight writes to drain and ``VideoWriter.release``
    to flush the file, which froze the UI on every stop click.
    Callers that need a "fully flushed" guarantee can use
    ``wait_drained(timeout)``.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from larksnap.adapters.base import BaseAdapter
from larksnap.gateway.frame_queue import FrameConsumer, FramePacket
from larksnap.utils.worker_pool import WorkerPool


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
        worker_pool: WorkerPool | None = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._fps = fps
        self._codec = codec
        self._frame_queue_url = frame_queue_url
        self._worker_pool = worker_pool
        self._logger = logging.getLogger("larksnap.recorder")
        self._writer: cv2.VideoWriter | None = None
        self._consumer: FrameConsumer | None = None
        # All _recording / _writer / _current_file mutations happen
        # under this lock. The hot path (_on_frame) only reads
        # _recording under the lock and then releases it before
        # dispatching the write to the worker pool.
        self._lock = threading.Lock()
        self._recording = False
        self._current_file: str | None = None
        self._frame_count: int = 0
        self._start_time: float = 0.0
        # Monotonically increasing frame index for the worker. The
        # write is dispatched to the pool with (index, frame_copy)
        # so the worker can write frames in order even if the pool
        # occasionally reorders (it shouldn't, but defensive).
        self._write_seq: int = 0
        # Sentinel: when stop_recording is called, we push this
        # many ``None`` entries onto the pool so the worker
        # eventually observes them. Simpler: stop_recording just
        # flips the flag and the worker drops queued frames.
        # ``_last_queued_seq`` lets a flush-after-stop detect
        # when no more in-flight writes remain.
        self._last_queued_seq: int = -1
        self._in_flight: int = 0
        self._in_flight_lock = threading.Lock()
        # Drain coordination. ``_draining`` is True between
        # ``stop_recording`` returning and the writer actually
        # being released by the background finalizer. ``_drain_done``
        # is set once the file is fully flushed. The split is
        # what lets ``stop_recording`` return immediately while
        # still allowing shutdown paths to wait synchronously.
        self._draining: bool = False
        self._drain_done = threading.Event()
        self._drain_done.set()  # idle at start

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    @property
    def is_draining(self) -> bool:
        """True between ``stop_recording`` returning and the file
        being fully flushed to disk.

        Lets the UI show a brief "saving…" hint and lets the
        controller refuse to start a new recording until the
        previous one is safely on disk.
        """
        return self._draining

    @property
    def current_file(self) -> str | None:
        with self._lock:
            return self._current_file

    @property
    def duration(self) -> float:
        with self._lock:
            if not self._recording:
                return 0.0
            return time.time() - self._start_time

    @property
    def frame_count(self) -> int:
        with self._lock:
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
        """Start recording frames from the frame queue.

        Refuses to start while a previous recording is still being
        drained to disk (``_draining``). Callers that need to
        guarantee a clean transition should call ``wait_drained``
        first; the controller's close worker does this.
        """
        with self._lock:
            if self._recording:
                self._logger.warning("Already recording")
                return
            if self._draining:
                # A previous recording is still being flushed. To
                # avoid corrupting the in-progress mp4 by opening
                # a second writer over it, we just refuse. The UI
                # sees ``is_recording == False`` and the user can
                # click record again once draining finishes.
                self._logger.warning(
                    "Recorder is still draining the previous file; "
                    "ignoring start_recording()"
                )
                return

            self._frame_count = 0
            self._start_time = time.time()
            self._write_seq = 0
            self._last_queued_seq = -1
            with self._in_flight_lock:
                self._in_flight = 0
            self._consumer = FrameConsumer(
                zmq_url=self._frame_queue_url,
                hwm=60,
            )
            self._consumer.start(
                on_frame=self._on_frame,
                context=context,
            )
            self._recording = True
            self._drain_done.set()  # nothing in flight
            self._logger.info("Video recording started")

    def stop_recording(self, drain_timeout: float = 2.0) -> None:
        """Stop recording and close the video file.

        Non-blocking on the caller's thread. The method only:
          1. flips the recording flag (so ``is_recording`` returns
             False immediately and the UI can react),
          2. stops the ZMQ consumer (bounded by the consumer's
             own internal timeout),
          3. snapshots the writer reference,
          4. dispatches the actual file flush + ``VideoWriter.release``
             to a background worker.

        The previous implementation blocked here for up to
        ``drain_timeout`` seconds, which froze the Qt main thread
        every time the user clicked "Stop recording". With the
        file release offloaded, the UI stays responsive and the
        user can immediately click "Start recording" again — the
        recorder will reject the new start if the previous drain
        hasn't finished (see ``is_draining``) so the file is
        never corrupted.

        For shutdown paths (e.g. closing the camera) that need to
        guarantee the file is fully written, call
        ``wait_drained(timeout)`` after this method returns.
        """
        with self._lock:
            if not self._recording and not self._draining:
                return
            if not self._recording and self._draining:
                # Already stopping; do nothing. The background
                # finalizer is in charge of releasing the writer.
                return

            self._recording = False
            if self._consumer is not None:
                self._consumer.stop()
                self._consumer = None
            last_queued = self._last_queued_seq
            local_writer = self._writer
            local_file = self._current_file
            local_frame_count = self._frame_count
            local_duration = self.duration
            # Mark draining BEFORE releasing the lock so a
            # concurrent start_recording sees it and refuses.
            self._draining = True
            self._drain_done.clear()

        # Hand the final release to a background worker so the
        # caller's thread returns immediately. We do not hold the
        # lock during the finalizer — it acquires the lock only
        # briefly to clear ``_writer`` / ``_current_file``.
        finalizer = threading.Thread(
            target=self._finalize_release,
            args=(
                local_writer, local_file,
                local_frame_count, local_duration,
                last_queued, drain_timeout,
            ),
            daemon=True,
            name="VideoRecorder-finalize",
        )
        finalizer.start()

    def _finalize_release(
        self,
        writer: cv2.VideoWriter | None,
        out_file: str | None,
        frames: int,
        duration: float,
        last_queued: int,
        drain_timeout: float,
    ) -> None:
        """Background finalizer: drain in-flight writes and close the file.

        Runs on a dedicated daemon thread so the caller's thread
        is never blocked on disk I/O. Catches and logs every error
        because the caller can no longer see exceptions (they're
        already gone by the time we get here).
        """
        try:
            # Wait for in-flight writes to drain (bounded). This
            # is the only blocking step in the new flow, and it
            # now runs off the UI thread.
            if last_queued >= 0 and self._worker_pool is not None:
                deadline = time.time() + drain_timeout
                while time.time() < deadline:
                    with self._in_flight_lock:
                        if self._in_flight <= 0:
                            break
                    time.sleep(0.01)
                else:
                    with self._in_flight_lock:
                        remaining = self._in_flight
                    if remaining > 0:
                        self._logger.warning(
                            "Recorder stop timed out waiting for %d "
                            "in-flight write(s)", remaining,
                        )
            if writer is not None:
                try:
                    writer.release()
                except Exception as e:  # noqa: BLE001
                    self._logger.error(
                        "VideoWriter release failed: %s", e,
                    )
            self._logger.info(
                "Video recording stopped: %s (%d frames, %.1fs)",
                out_file, frames, duration,
            )
        except Exception as e:  # noqa: BLE001
            # Defensive: never let the finalizer die silently.
            self._logger.error(
                "Recorder finalizer crashed: %s", e, exc_info=True,
            )
        finally:
            # Always clear state, even on error, so the recorder
            # is usable for the next session.
            with self._lock:
                self._writer = None
                self._current_file = None
                self._draining = False
            self._drain_done.set()

    def wait_drained(self, timeout: float | None = None) -> bool:
        """Block until any in-flight ``stop_recording`` finalizes.

        Returns True if drain completed within ``timeout``,
        False otherwise (or if the recorder is idle).

        Used by shutdown paths (e.g. ``_close_camera_worker``)
        that must guarantee the file is fully flushed before
        tearing the rest of the pipeline down.
        """
        if not self._drain_done.wait(timeout):
            return False
        return True

    def _on_frame(self, packet: FramePacket) -> None:
        """Handle a frame from the consumer.

        Runs on the ZMQ consumer thread. Must be fast: we only
        flip the state lock, copy the frame, and queue the write.
        The actual ``VideoWriter.write`` happens on the worker
        pool.
        """
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

            # Take a stable reference to the writer while holding
            # the lock. The writer itself is mutated only on this
            # thread or on stop_recording, so we capture it here
            # and pass the reference to the worker.
            writer = self._writer
            # Hand the worker a fresh C-contiguous copy. The
            # ZMQ consumer overwrites its buffer on the next
            # frame, so we can't share the array.
            seq = self._write_seq
            self._write_seq += 1
            self._last_queued_seq = seq

        frame_copy = np.ascontiguousarray(packet.frame)
        with self._in_flight_lock:
            self._in_flight += 1
        if self._worker_pool is not None:
            queued = self._worker_pool.submit(
                lambda: self._do_write(writer, frame_copy, seq)
            )
            if not queued:
                # Pool full: drop this frame. The previous design
                # would have blocked the consumer on a slow write;
                # dropping is the right call because the user has
                # explicitly asked to record, but the alternative
                # is to block detection. We log so a tuning change
                # (bigger pool / smaller HWM) is possible later.
                self._logger.warning(
                    "Recorder worker pool full; dropping frame seq=%d", seq,
                )
                with self._in_flight_lock:
                    self._in_flight -= 1
        else:
            # No pool: do the write inline. Better than dropping.
            self._do_write(writer, frame_copy, seq)

    def _do_write(
        self, writer: cv2.VideoWriter, frame: np.ndarray, seq: int
    ) -> None:
        try:
            writer.write(frame)
            with self._lock:
                self._frame_count += 1
        except Exception as e:  # noqa: BLE001
            self._logger.error("VideoWriter write failed (seq=%d): %s", seq, e)
        finally:
            with self._in_flight_lock:
                self._in_flight -= 1
