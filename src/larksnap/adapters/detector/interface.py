from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from larksnap.adapters.base import BaseAdapter


@dataclass
class BBox:
    """Bounding box coordinates for a detection result."""

    x: float
    y: float
    width: float
    height: float


@dataclass
class DetectionResult:
    """Single object detection result."""

    label: str
    confidence: float
    bbox: BBox
    mask: np.ndarray | None = None


class DetectorAdapter(BaseAdapter):
    """Abstract detector adapter interface for object detection."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Run detection on a video frame."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the detection model."""
        pass

    def initialize(self) -> None:
        self.load_model()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.unload_model()

    def release(self) -> None:
        pass
