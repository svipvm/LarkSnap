"""YOLO Segmentation ONNX Runtime detector adapter.

Wraps the YOLOSegORT inference engine to implement the DetectorAdapter interface,
providing person detection capabilities using YOLO segmentation models.
"""

from __future__ import annotations

import logging

import numpy as np

from larksnap.adapters.detector.interface import BBox, DetectionResult, DetectorAdapter
from larksnap.config.models import DetectorConfig
from larksnap.utils.exceptions import DetectorError


class YOLOSegDetectorAdapter(DetectorAdapter):
    """Detector adapter using YOLO Segmentation ONNX Runtime inference."""

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._logger = logging.getLogger("larksnap.detector.yolo_seg")
        self._predictor: _YOLOSegWrapper | None = None

    def load_model(self) -> None:
        try:
            self._predictor = _YOLOSegWrapper(self._config)
            self._logger.info(
                "YOLO Seg model loaded: %s", self._config.yolo_seg.model_path
            )
        except Exception as e:
            raise DetectorError(f"Failed to load YOLO Seg model: {e}") from e

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        if self._predictor is None:
            raise DetectorError("YOLO Seg model not loaded")

        try:
            return self._predictor.predict(frame)
        except Exception as e:
            raise DetectorError(f"YOLO Seg detection failed: {e}") from e

    def unload_model(self) -> None:
        self._predictor = None
        self._logger.info("YOLO Seg model unloaded")


class _YOLOSegWrapper:
    """Thin wrapper around YOLOSegORT that converts SegResult to DetectionResult."""

    # COCO class id for "person"
    _PERSON_CLASS_ID = 0

    def __init__(self, config: DetectorConfig) -> None:
        from larksnap.adapters.detector._yolo_seg_ort import (
            COCO_NAMES,
            InferConfig,
            YOLOSegORT,
        )

        self._coco_names = COCO_NAMES
        yolo_cfg = config.yolo_seg

        providers: list[str] = []
        if yolo_cfg.provider == "cuda":
            providers.extend(["CUDAExecutionProvider", "CPUExecutionProvider"])
        else:
            providers.append("CPUExecutionProvider")

        infer_config = InferConfig(
            model_path=yolo_cfg.model_path,
            img_size=yolo_cfg.img_size,
            conf_thres=config.confidence_threshold,
            iou_thres=yolo_cfg.iou_thres,
            max_det=yolo_cfg.max_det,
            providers=providers,
        )
        self._ort = YOLOSegORT(infer_config)

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
                )
            )

        return results
