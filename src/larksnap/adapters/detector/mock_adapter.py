import logging
import random
import time

import numpy as np

from larksnap.adapters.detector.interface import (
    BBox,
    DetectionResult,
    DetectorAdapter,
    filter_results_by_classes,
)
from larksnap.adapters.registry import detector_registry
from larksnap.config.models import DetectorConfig
from larksnap.utils.exceptions import DetectorError


@detector_registry.register("mock")
class MockDetectorAdapter(DetectorAdapter):
    """Mock detector adapter that returns simulated detection results.

    Honours the monitoring contract: the random results it generates
    are filtered through :func:`filter_results_by_classes` so only
    labels present in ``target_classes`` ever leave the adapter. This
    keeps the mock adapter consistent with the real ``seg`` adapter
    in tests that exercise the gateway.
    """

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
        # Apply the monitoring contract — the mock and the real
        # ``seg`` adapter must agree on what counts as a detection
        # the user actually asked for.
        return filter_results_by_classes(results, self._config.target_classes)

    def unload_model(self) -> None:
        """Simulate unloading the detection model."""
        self._loaded = False
        self._logger.info("Mock detection model unloaded")
