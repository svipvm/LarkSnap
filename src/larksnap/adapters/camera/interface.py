from abc import abstractmethod

import numpy as np

from larksnap.adapters.base import BaseAdapter


class CameraAdapter(BaseAdapter):
    """Abstract camera adapter interface for video capture."""

    @abstractmethod
    def open(self) -> None:
        """Open the camera connection."""
        pass

    @abstractmethod
    def read_frame(self) -> np.ndarray:
        """Read a single frame from the camera."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release the camera resources."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if the camera is currently opened."""
        pass

    def initialize(self) -> None:
        self.open()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.release()
