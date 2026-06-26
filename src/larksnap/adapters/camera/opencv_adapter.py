import logging
import os
import time

# Suppress OpenCV's own noisy logs (MSMF/obsensor/DSHOW internals).
# Must be set before the first cv2 import — by the time we get here, cv2
# has been imported by some other module, but these vars still affect
# newly-created VideoCapture backends and reduce per-frame warnings.
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_INTEL_MFX", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

from larksnap.adapters.camera.interface import CameraAdapter
from larksnap.adapters.registry import camera_registry
from larksnap.config.models import CameraConfig
from larksnap.utils.exceptions import CameraError

# Force OpenCV's unified logger to ERROR level. The ``OPENCV_LOG_LEVEL``
# env var is *only* honoured at the first cv2 import; by the time this
# module runs cv2 has usually already been imported elsewhere (PySide6
# pull it in transitively), so the env var is a no-op. We must set the
# log level at runtime via the public API. Levels:
#   0 = SILENT, 1 = FATAL, 2 = ERROR, 3 = WARNING, 4 = INFO, 5 = DEBUG.
# We pick 2 (ERROR) so genuine errors still surface, but the spammy
# ``VIDEOIO(DSHOW): backend is generally available but can't be used
# to capture by index`` warning that fires on every DSHOW probe on
# Windows is suppressed.
try:
    cv2.setLogLevel(2)  # type: ignore[attr-defined]
except AttributeError:
    # Older OpenCV (<4.5) exposed the API under utils.logging.
    try:
        cv2.utils.logging.setLogLevel(2)  # type: ignore[attr-defined]
    except AttributeError:
        pass


def _silence_opencv_stderr() -> "object":
    """Return a context manager that silences OpenCV's C-level stderr.

    Some OpenCV backends (obsensor, MSMF, DSHOW) write warnings directly
    to the C-level stderr, bypassing both Python's ``sys.stderr`` and
    the unified ``cv2.setLogLevel`` machinery. We do a best-effort
    suppression: redirect Python's ``sys.stderr`` (catches the ones
    that go through the C runtime) and, on Windows, also swap the
    console stderr handle for a NUL handle (catches the ones that go
    through ``WriteConsole`` / OutputDebugString).
    """
    import contextlib
    import io
    import sys

    if os.name != "nt":
        # POSIX: ``contextlib.redirect_stderr`` plus an OS-level
        # dup2 of fd 2 to /dev/null covers the C runtime's stderr.
        return _PosixCSilence()

    return _WindowsStderrSilence()


class _PosixCSilence:
    """POSIX: redirect fd 2 to /dev/null for the duration of the block.

    Necessary because OpenCV's C code uses ``fprintf(stderr, ...)``
    which writes to the OS-level file descriptor 2, not Python's
    ``sys.stderr`` wrapper.
    """

    def __enter__(self) -> "_PosixCSilence":
        try:
            self._saved = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)
            self._ok = True
        except OSError:
            self._ok = False
        return self

    def __exit__(self, *exc: object) -> None:
        if getattr(self, "_ok", False):
            try:
                os.dup2(self._saved, 2)
                os.close(self._saved)
            except OSError:
                pass


class _WindowsStderrSilence:
    """Windows: redirect the stderr console handle to NUL.

    On Windows, OpenCV's DSHOW/MSMF backends use Win32 APIs
    (``WriteFile`` / console handles) which go to a different OS
    object than the C runtime's fd 2. To catch those, we use
    ``SetStdHandle(STD_ERROR_HANDLE, ...)`` to swap the handle
    for the NUL device for the duration of the block.
    """

    STD_ERROR_HANDLE = -12  # STD_ERROR_HANDLE
    GENERIC_WRITE = 0x40000000
    FILE_SHARE_WRITE = 0x00000002
    OPEN_ALWAYS = 4

    def __enter__(self) -> "_WindowsStderrSilence":
        self._ok = False
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            self._saved = kernel32.GetStdHandle(self.STD_ERROR_HANDLE)
            self._nul = kernel32.CreateFileA(
                b"NUL",
                self.GENERIC_WRITE,
                self.FILE_SHARE_WRITE,
                None,
                self.OPEN_ALWAYS,
                0,
                0,
            )
            kernel32.SetStdHandle(self.STD_ERROR_HANDLE, self._nul)
            # Also redirect the C runtime fd 2 in case some code
            # path still uses it.
            try:
                self._saved_fd = os.dup(2)
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, 2)
                os.close(devnull)
            except OSError:
                self._saved_fd = None
            self._ok = True
        except (OSError, AttributeError, OSError):
            self._ok = False
        return self

    def __exit__(self, *exc: object) -> None:
        if not getattr(self, "_ok", False):
            return
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetStdHandle(self.STD_ERROR_HANDLE, self._saved)
            if self._nul:
                kernel32.CloseHandle(self._nul)
            if getattr(self, "_saved_fd", None) is not None:
                os.dup2(self._saved_fd, 2)
                os.close(self._saved_fd)
        except (OSError, AttributeError):
            pass


def enumerate_cameras(
    max_index: int = 10,
    backend: int | None = None,
) -> list[int]:
    """Detect available camera device indices.

    Probes indices 0..max_index-1 using a specific Windows backend
    (DSHOW by default) to avoid triggering the obsensor / Orbbec SDK
    probe path that prints ``Camera index out of range`` for every
    non-Orbbec device on systems where obsensor is compiled in.

    Args:
        max_index: Highest device index to probe (exclusive).
        backend: OpenCV backend constant. Defaults to CAP_DSHOW on
            Windows, CAP_ANY elsewhere.

    Returns:
        List of device indices that successfully open.
    """
    if backend is None:
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY

    backend_name = {
        cv2.CAP_DSHOW: "DSHOW",
        cv2.CAP_MSMF: "MSMF",
        cv2.CAP_ANY: "ANY",
    }.get(backend, str(backend))

    available: list[int] = []
    for idx in range(max_index):
        cap = None
        try:
            with _silence_opencv_stderr():
                cap = cv2.VideoCapture(idx, backend)
            if cap is not None and cap.isOpened():
                # Probe a single frame so we don't list cameras that open
                # but can't actually deliver frames (e.g. phantom devices).
                with _silence_opencv_stderr():
                    ok, _ = cap.read()
                if ok:
                    available.append(idx)
        except Exception:
            pass
        finally:
            if cap is not None:
                cap.release()

    logging.getLogger("larksnap.camera").debug(
        "Camera enumeration complete: backend=%s, found=%s",
        backend_name, available,
    )
    return available


@camera_registry.register("opencv")
class OpenCVCameraAdapter(CameraAdapter):
    """OpenCV-based camera adapter for local webcam capture."""

    def __init__(self, config: CameraConfig) -> None:
        """Initialize the adapter with camera configuration."""
        self._config = config
        self._cap: cv2.VideoCapture | None = None
        self._logger = logging.getLogger("larksnap.camera")

    def open(self) -> None:
        """Open the camera with retry logic and backend fallback.

        Tries multiple OpenCV backends (MSMF → DSHOW → ANY) because some
        cameras work better with DSHOW than the default MSMF on Windows.
        Also detects the camera's actual resolution and FPS instead of
        blindly trusting the config, which prevents frame-metadata
        mismatches that lead to _step >= minstep assertions downstream.
        """
        from larksnap.utils.camera_error_translator import translate_camera_error

        max_retries = self._config.max_retries
        retry_interval = self._config.retry_interval
        last_raw_error = ""

        # Backend fallback order: MSMF (default Win10+) → DSHOW → ANY
        if os.name == "nt":
            backend_candidates = [
                cv2.CAP_MSMF,
                cv2.CAP_DSHOW,
                cv2.CAP_ANY,
            ]
        else:
            backend_candidates = [cv2.CAP_ANY]

        for attempt in range(1, max_retries + 1):
            for backend in backend_candidates:
                backend_name = {
                    cv2.CAP_MSMF: "MSMF",
                    cv2.CAP_DSHOW: "DSHOW",
                    cv2.CAP_ANY: "ANY",
                }.get(backend, str(backend))

                if self._cap is not None:
                    self._cap.release()
                    self._cap = None

                try:
                    # Suppress OpenCV C-level stderr (obsensor, MSMF etc.)
                    # that would otherwise print noisy "Camera index out of
                    # range" / HRESULT errors on Windows.
                    with _silence_opencv_stderr():
                        self._cap = cv2.VideoCapture(
                            self._config.device_index, backend,
                        )
                    if not self._cap.isOpened():
                        last_raw_error = f"backend {backend_name}: open failed"
                        continue

                    # Configure the camera. We set the requested resolution/FPS,
                    # then *re-read* the actual values because some cameras
                    # silently fall back to a different resolution.
                    with _silence_opencv_stderr():
                        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
                        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
                        self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)
                        # Force buffer flushing — drops any stale frames from
                        # previous runs that might leak into the first read.
                        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    # Drain a few stale frames to stabilise the pipeline
                    for _ in range(3):
                        with _silence_opencv_stderr():
                            self._cap.read()

                    # Probe-read one frame and validate its shape
                    with _silence_opencv_stderr():
                        ret, probe = self._cap.read()
                    if not ret or probe is None:
                        last_raw_error = f"backend {backend_name}: probe read failed"
                        continue
                    if probe.ndim < 2 or probe.size == 0:
                        last_raw_error = f"backend {backend_name}: invalid frame shape {probe.shape}"
                        continue
                    if not probe.flags.c_contiguous:
                        probe = np.ascontiguousarray(probe)

                    actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS)) or self._config.fps

                    self._logger.info(
                        "Camera opened: device=%d, backend=%s, requested=%dx%d@%d, actual=%dx%d@%d",
                        self._config.device_index,
                        backend_name,
                        self._config.width, self._config.height, self._config.fps,
                        actual_w, actual_h, actual_fps,
                    )
                    return

                except Exception as e:
                    last_raw_error = f"backend {backend_name}: {e}"
                    if self._cap is not None:
                        self._cap.release()
                        self._cap = None
                    continue

            # All backends failed for this attempt
            if self._cap is not None:
                self._cap.release()
                self._cap = None

            if attempt < max_retries:
                self._logger.warning(
                    "Camera open attempt %d/%d failed (%s), retrying in %.1fs...",
                    attempt, max_retries, last_raw_error, retry_interval,
                )
                time.sleep(retry_interval)
            else:
                translated = translate_camera_error(last_raw_error)
                self._logger.error(
                    "Camera open failed after %d attempts: %s", max_retries, translated,
                )
                raise CameraError(translated) from None

    def read_frame(self) -> np.ndarray:
        """Read a single frame from the opened camera.

        Validates the returned frame to guard against partial/corrupt
        frames that would later trigger OpenCV's _step >= minstep assertion
        when downstream code calls cv2.resize / cv2.cvtColor etc.
        """
        if not self.is_opened():
            raise CameraError("摄像头未打开")

        from larksnap.utils.camera_error_translator import translate_camera_error

        try:
            ret, frame = self._cap.read()
        except Exception as e:
            # Native cv2 errors (assertions, backend issues, etc.) bubble up
            # as Python exceptions — translate them to friendly text.
            raise CameraError(translate_camera_error(str(e))) from None

        if not ret or frame is None:
            raise CameraError("读取视频帧失败")

        # Validate frame shape
        if frame.ndim < 2 or frame.size == 0:
            raise CameraError("视频帧数据不完整或损坏")

        # Ensure C-contiguous memory so downstream OpenCV calls don't trip
        # the _step >= minstep assertion on sliced / non-contiguous arrays.
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)

        return frame

    def release(self) -> None:
        """Release the camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._logger.info("Camera released")

    def is_opened(self) -> bool:
        """Check if the camera connection is active."""
        return self._cap is not None and self._cap.isOpened()
