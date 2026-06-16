"""Instance Segmentation ONNX Runtime detector adapter.

Wraps the SegORT inference engine to implement the DetectorAdapter interface,
providing object detection capabilities using instance segmentation models.
"""

from __future__ import annotations

import logging

import numpy as np

from larksnap.adapters.detector.interface import BBox, DetectionResult, DetectorAdapter
from larksnap.config.models import DetectorConfig
from larksnap.utils.exceptions import DetectorError


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

        return results
