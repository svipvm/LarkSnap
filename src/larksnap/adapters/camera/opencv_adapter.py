import logging

import cv2
import numpy as np

from larksnap.adapters.camera.interface import CameraAdapter
from larksnap.config.models import CameraConfig
from larksnap.utils.exceptions import CameraError


class OpenCVCameraAdapter(CameraAdapter):
    """OpenCV-based camera adapter for local webcam capture."""

    def __init__(self, config: CameraConfig) -> None:
        """Initialize the adapter with camera configuration."""
        self._config = config
        self._cap: cv2.VideoCapture | None = None
        self._logger = logging.getLogger("larksnap.camera")

    def open(self) -> None:
        """Open the camera and apply configured settings."""
        try:
            self._cap = cv2.VideoCapture(self._config.device_index)
            if not self._cap.isOpened():
                raise CameraError(
                    f"Failed to open camera with device index "
                    f"{self._config.device_index}"
                )
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)
            self._logger.info(
                "Camera opened: device=%d, resolution=%dx%d, fps=%d",
                self._config.device_index,
                self._config.width,
                self._config.height,
                self._config.fps,
            )
        except CameraError:
            raise
        except Exception as e:
            raise CameraError(f"Failed to initialize camera: {e}") from e

    def read_frame(self) -> np.ndarray:
        """Read a single frame from the opened camera."""
        if not self.is_opened():
            raise CameraError("Camera is not opened")
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise CameraError("Failed to read frame from camera")
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
