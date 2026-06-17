import logging
import random
import time

import numpy as np

from larksnap.adapters.detector.interface import BBox, DetectionResult, DetectorAdapter
from larksnap.adapters.registry import detector_registry
from larksnap.config.models import DetectorConfig
from larksnap.utils.exceptions import DetectorError


@detector_registry.register("mock")
class MockDetectorAdapter(DetectorAdapter):
    """Mock detector adapter that returns simulated detection results."""

    def __init__(self, config: DetectorConfig) -> None:
        """Initialize the mock detector with configuration."""
        self._config = config
        self._logger = logging.getLogger("larksnap.detector.mock")
        self._loaded = False

    def load_model(self) -> None:
        """Simulate loading the detection model."""
        self._logger.info("Loading mock detection model...")
        time.sleep(0.1)
        self._loaded = True
        self._logger.info("Mock detection model loaded")

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Return simulated detection results for a video frame."""
        if not self._loaded:
            raise DetectorError("Model not loaded")

        if self._config.mock.delay_seconds > 0:
            time.sleep(self._config.mock.delay_seconds)

        results: list[DetectionResult] = []
        h, w = frame.shape[:2]
        conf_min, conf_max = self._config.mock.confidence_range

        for label in self._config.mock.labels:
            if random.random() < 0.7:
                confidence = random.uniform(conf_min, conf_max)
                bbox_w = random.uniform(w * 0.1, w * 0.4)
                bbox_h = random.uniform(h * 0.1, h * 0.4)
                bbox_x = random.uniform(0, w - bbox_w)
                bbox_y = random.uniform(0, h - bbox_h)
                results.append(
                    DetectionResult(
                        label=label,
                        confidence=confidence,
                        bbox=BBox(x=bbox_x, y=bbox_y, width=bbox_w, height=bbox_h),
                    )
                )

        self._logger.debug("Mock detection returned %d results", len(results))
        return results

    def unload_model(self) -> None:
        """Simulate unloading the detection model."""
        self._loaded = False
        self._logger.info("Mock detection model unloaded")
