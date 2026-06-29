"""Instance Segmentation ONNX Runtime detector adapter.

Wraps the SegORT inference engine to implement the DetectorAdapter interface,
providing object detection capabilities using instance segmentation models.

The adapter enforces the *monitoring contract* on top of the raw model
output: after inference, results are filtered against the configured
``target_classes`` so that the gateway only ever observes detections
the user actually asked to monitor. See
:func:`filter_results_by_classes` for the exact semantics.
"""

from __future__ import annotations

import logging

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


@detector_registry.register("seg")
class SegDetectorAdapter(DetectorAdapter):
    """Detector adapter using Instance Segmentation ONNX Runtime inference."""

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._logger = logging.getLogger("larksnap.detector.seg")
        self._predictor: _SegWrapper | None = None

    def load_model(self) -> None:
        try:
            self._predictor = _SegWrapper(self._config)
            self._logger.info(
                "Seg model loaded: %s", self._config.seg.model_path
            )
        except Exception as e:
            raise DetectorError(f"Failed to load seg model: {e}") from e

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        if self._predictor is None:
            raise DetectorError("Seg model not loaded")

        try:
            return self._predictor.predict(frame)
        except Exception as e:
            raise DetectorError(f"Seg detection failed: {e}") from e

    def unload_model(self) -> None:
        self._predictor = None
        self._logger.info("Seg model unloaded")


class _SegWrapper:
    """Thin wrapper around SegORT that converts SegResult to DetectionResult."""

    # COCO class id for "person"
    _PERSON_CLASS_ID = 0

    def __init__(self, config: DetectorConfig) -> None:
        from larksnap.adapters.detector._seg_ort import (
            COCO_NAMES,
            InferConfig,
            SegORT,
        )

        self._coco_names = COCO_NAMES
        # Cache the monitoring set so ``predict`` can apply the
        # contract without re-reading the config on every frame.
        # The list is small and stable for the lifetime of the
        # process, so caching it is safe and avoids the per-frame
        # set-construction in :func:`filter_results_by_classes`.
        self._target_classes: list[str] = list(config.target_classes)
        seg_cfg = config.seg

        providers: list[str] = []
        if seg_cfg.provider == "cuda":
            providers.extend(["CUDAExecutionProvider", "CPUExecutionProvider"])
        else:
            providers.append("CPUExecutionProvider")

        infer_config = InferConfig(
            model_path=seg_cfg.model_path,
            img_size=seg_cfg.img_size,
            conf_thres=config.confidence_threshold,
            iou_thres=seg_cfg.iou_thres,
            max_det=seg_cfg.max_det,
            providers=providers,
        )
        self._ort = SegORT(infer_config)

    def predict(self, frame: np.ndarray) -> list[DetectionResult]:
        seg_result = self._ort.predict(frame)
        results: list[DetectionResult] = []

        for i in range(seg_result.count):
            cls_id = int(seg_result.class_ids[i])
            label = (
                self._coco_names[cls_id]
                if cls_id < len(self._coco_names)
                else f"class_{cls_id}"
            )
            x1, y1, x2, y2 = seg_result.boxes[i]
            mask = seg_result.masks[i] if seg_result.masks is not None and i < len(seg_result.masks) else None
            results.append(
                DetectionResult(
                    label=label,
                    confidence=float(seg_result.scores[i]),
                    bbox=BBox(
                        x=float(x1),
                        y=float(y1),
                        width=float(x2 - x1),
                        height=float(y2 - y1),
                    ),
                    mask=mask,
                )
            )

        # Enforce the monitoring contract: the wrapper only ever
        # reports detections whose label is in ``target_classes``.
        # This makes the adapter's behaviour deterministic and
        # removes the need for the downstream notification path to
        # guess what the user wanted.
        return filter_results_by_classes(results, self._target_classes)

    def set_target_classes(self, target_classes: list[str]) -> None:
        """Hot-swap the monitoring set.

        Called by the controller when ``detector.target_classes``
        is updated via ``/config set``. The next ``predict()``
        call uses the new set; the onnx engine itself doesn't
        need a reload because the monitoring filter is a pure
        post-processing step on top of the raw model output.
        """
        # ``list(...)`` copies so the caller can mutate their
        # own list afterwards without surprising the adapter.
        self._target_classes = list(target_classes)
