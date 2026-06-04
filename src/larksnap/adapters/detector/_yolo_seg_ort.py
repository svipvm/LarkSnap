"""YOLO Segmentation ONNX Runtime inference module.

Based on onnxruntime, supports both end-to-end (built-in NMS) and standard
(manual NMS) ONNX export formats for YOLO instance segmentation models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# COCO 80 class names
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

PALETTE = np.random.RandomState(42).randint(
    50, 255, size=(80, 3), dtype=np.uint8
)


@dataclass
class SegResult:
    """Segmentation inference result for a single image."""

    boxes: np.ndarray  # (N, 4) xyxy format, original image coordinates
    scores: np.ndarray  # (N,)
    class_ids: np.ndarray  # (N,)
    masks: np.ndarray  # (N, H, W) binary mask

    @property
    def count(self) -> int:
        return len(self.scores)


@dataclass
class InferConfig:
    """Inference configuration."""

    model_path: str = "yolo26n-seg.onnx"
    img_size: int = 640
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    max_det: int = 300
    mask_threshold: float = 0.5
    providers: list[str] = field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )


class YOLOSegORT:
    """YOLO Segmentation ONNX Runtime predictor.

    Auto-detects model output format:
      - end-to-end: output0 (1, 300, 4+1+1+32) with built-in NMS
      - standard:   output0 (1, 4+nc+32, N) requiring manual NMS
    """

    def __init__(self, config: InferConfig | None = None):
        self.cfg = config or InferConfig()
        self._nc = len(COCO_NAMES)
        self._mask_dim = 32
        self._end2end = False
        self._init_session()

    @staticmethod
    def _add_cuda_dll_paths() -> None:
        """Add pip-installed NVIDIA CUDA/cuDNN DLL directories to search."""
        import os
        import sys

        venv_lib = Path(sys.prefix) / "Lib" / "site-packages"
        dll_dirs = [
            venv_lib / "nvidia" / "cudnn" / "bin",
            venv_lib / "nvidia" / "cublas" / "bin",
            venv_lib / "nvidia" / "cuda_nvrtc" / "bin",
        ]
        for d in dll_dirs:
            if d.is_dir():
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")

    def _init_session(self):
        self._add_cuda_dll_paths()

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3

        available = ort.get_available_providers()
        requested = self.cfg.providers
        providers = [p for p in requested if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        dropped = set(requested) - set(providers)
        if dropped:
            import logging
            logging.getLogger("larksnap.detector.yolo_seg").warning(
                "ONNX Runtime providers %s not available, using %s",
                dropped, providers,
            )

        self.session = ort.InferenceSession(
            self.cfg.model_path,
            sess_options=sess_opts,
            providers=providers,
        )
        input_info = self.session.get_inputs()[0]
        self._input_name = input_info.name
        self._input_shape = input_info.shape
        self._output_names = [o.name for o in self.session.get_outputs()]

        dummy = np.zeros(
            (1, 3, self.cfg.img_size, self.cfg.img_size), dtype=np.float32
        )
        outs = self.session.run(self._output_names, {self._input_name: dummy})
        if (
            len(outs[0].shape) == 3
            and outs[0].shape[2] == 4 + 1 + 1 + self._mask_dim
        ):
            self._end2end = True
        else:
            self._end2end = False

    def _preprocess(
        self, img_bgr: np.ndarray
    ) -> tuple[
        np.ndarray, tuple[int, int], float, tuple[float, float]
    ]:
        """Letterbox preprocessing."""
        h, w = img_bgr.shape[:2]
        target = self.cfg.img_size
        gain = min(target / h, target / w)
        new_w, new_h = round(w * gain), round(h * gain)

        dw = (target - new_w) / 2
        dh = (target - new_h) / 2

        img_resized = cv2.resize(
            img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        top, left = round(dh - 0.1), round(dw - 0.1)
        canvas[top : top + new_h, left : left + new_w] = img_resized

        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[np.newaxis, ...]
        return blob, (h, w), gain, (dw, dh)

    def _postprocess(
        self,
        output0: np.ndarray,
        output1: np.ndarray,
        orig_shape: tuple[int, int],
        gain: float,
        pad: tuple[float, float],
    ) -> SegResult:
        if self._end2end:
            return self._postprocess_e2e(
                output0, output1, orig_shape, gain, pad
            )
        return self._postprocess_standard(
            output0, output1, orig_shape, gain, pad
        )

    def _postprocess_e2e(
        self,
        output0: np.ndarray,
        output1: np.ndarray,
        orig_shape: tuple[int, int],
        gain: float,
        pad: tuple[float, float],
    ) -> SegResult:
        pred = output0[0]

        scores = pred[:, 4]
        keep = scores > self.cfg.conf_thres
        pred = pred[keep]
        scores = scores[keep]

        if len(scores) == 0:
            return self._empty_result(orig_shape)

        boxes_xyxy = pred[:, :4]
        class_ids = pred[:, 5].astype(np.int64)
        mask_coefs = pred[:, 6 : 6 + self._mask_dim]

        if len(scores) > self.cfg.max_det:
            topk = np.argsort(scores)[::-1][: self.cfg.max_det]
            boxes_xyxy = boxes_xyxy[topk]
            scores = scores[topk]
            class_ids = class_ids[topk]
            mask_coefs = mask_coefs[topk]

        protos = output1[0]
        masks = self._process_mask(
            protos, mask_coefs, boxes_xyxy, orig_shape
        )

        keep_mask = masks.max(axis=(1, 2)) > 0
        boxes_xyxy = boxes_xyxy[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask]
        masks = masks[keep_mask]

        boxes_xyxy = self._scale_boxes(boxes_xyxy, orig_shape, gain, pad)

        return SegResult(
            boxes=boxes_xyxy.astype(np.float32),
            scores=scores.astype(np.float32),
            class_ids=class_ids,
            masks=masks,
        )

    def _postprocess_standard(
        self,
        output0: np.ndarray,
        output1: np.ndarray,
        orig_shape: tuple[int, int],
        gain: float,
        pad: tuple[float, float],
    ) -> SegResult:
        pred = output0[0].transpose(1, 0)

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4 : 4 + self._nc]
        mask_coefs = pred[
            :, 4 + self._nc : 4 + self._nc + self._mask_dim
        ]

        max_scores = class_scores.max(axis=1)
        keep = max_scores > self.cfg.conf_thres
        boxes_xywh = boxes_xywh[keep]
        class_scores = class_scores[keep]
        mask_coefs = mask_coefs[keep]
        max_scores = max_scores[keep]

        if len(max_scores) == 0:
            return self._empty_result(orig_shape)

        boxes_xyxy = self._xywh2xyxy(boxes_xywh)
        class_ids = class_scores.argmax(axis=1)

        nms_keep = self._nms(
            boxes_xyxy, max_scores, class_ids, self.cfg.iou_thres
        )
        boxes_xyxy = boxes_xyxy[nms_keep]
        scores = max_scores[nms_keep]
        class_ids = class_ids[nms_keep]
        mask_coefs = mask_coefs[nms_keep]

        if len(scores) > self.cfg.max_det:
            topk = np.argsort(scores)[::-1][: self.cfg.max_det]
            boxes_xyxy = boxes_xyxy[topk]
            scores = scores[topk]
            class_ids = class_ids[topk]
            mask_coefs = mask_coefs[topk]

        protos = output1[0]
        masks = self._process_mask(
            protos, mask_coefs, boxes_xyxy, orig_shape
        )

        keep_mask = masks.max(axis=(1, 2)) > 0
        boxes_xyxy = boxes_xyxy[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask]
        masks = masks[keep_mask]

        boxes_xyxy = self._scale_boxes(boxes_xyxy, orig_shape, gain, pad)

        return SegResult(
            boxes=boxes_xyxy.astype(np.float32),
            scores=scores.astype(np.float32),
            class_ids=class_ids,
            masks=masks,
        )

    def _empty_result(self, orig_shape: tuple[int, int]) -> SegResult:
        return SegResult(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros(0, dtype=np.float32),
            class_ids=np.zeros(0, dtype=np.int64),
            masks=np.zeros((0, *orig_shape), dtype=np.uint8),
        )

    @staticmethod
    def _xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
        x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        return np.stack(
            [x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1
        )

    @staticmethod
    def _nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        iou_thres: float,
    ) -> list[int]:
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            remaining = order[1:]
            ious = YOLOSegORT._box_iou(
                boxes[i : i + 1], boxes[remaining]
            )[0]
            same_class = class_ids[remaining] == class_ids[i]
            suppress = (ious >= iou_thres) & same_class
            order = remaining[~suppress]
        return keep

    @staticmethod
    def _box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        inter_x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
        inter_y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
        inter_x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
        inter_y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])
        inter = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(
            inter_y2 - inter_y1, 0
        )
        return inter / (area1[:, None] + area2[None, :] - inter + 1e-7)

    def _process_mask(
        self,
        protos: np.ndarray,
        mask_coefs: np.ndarray,
        boxes_xyxy: np.ndarray,
        orig_shape: tuple[int, int],
    ) -> np.ndarray:
        c, mh, mw = protos.shape
        masks = (mask_coefs @ protos.reshape(c, -1)).reshape(-1, mh, mw)

        width_ratio = mw / self.cfg.img_size
        height_ratio = mh / self.cfg.img_size
        scaled_boxes = boxes_xyxy * np.array(
            [[width_ratio, height_ratio, width_ratio, height_ratio]]
        )
        masks = self._crop_mask(masks, scaled_boxes)

        if masks.shape[1:] != orig_shape:
            masks_resized = np.zeros(
                (len(masks), *orig_shape), dtype=np.float32
            )
            for i in range(len(masks)):
                masks_resized[i] = cv2.resize(
                    masks[i].astype(np.float32),
                    (orig_shape[1], orig_shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            masks = masks_resized

        masks = (masks > self.cfg.mask_threshold).astype(np.uint8)
        return masks

    @staticmethod
    def _crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        n, h, w = masks.shape
        for i, (x1, y1, x2, y2) in enumerate(boxes.round().astype(int)):
            x1, y1, x2, y2 = (
                max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
            )
            masks[i, :y1] = 0
            masks[i, y2:] = 0
            masks[i, :, :x1] = 0
            masks[i, :, x2:] = 0
        return masks

    @staticmethod
    def _scale_boxes(
        boxes: np.ndarray,
        orig_shape: tuple[int, int],
        gain: float,
        pad: tuple[float, float],
    ) -> np.ndarray:
        dw, dh = pad
        boxes[:, 0] -= dw
        boxes[:, 1] -= dh
        boxes[:, 2] -= dw
        boxes[:, 3] -= dh
        boxes[:, :4] /= gain
        h, w = orig_shape
        boxes[:, 0] = boxes[:, 0].clip(0, w)
        boxes[:, 1] = boxes[:, 1].clip(0, h)
        boxes[:, 2] = boxes[:, 2].clip(0, w)
        boxes[:, 3] = boxes[:, 3].clip(0, h)
        return boxes

    def predict(self, source: str | np.ndarray) -> SegResult:
        """Run inference on a single image.

        Args:
            source: Image file path or BGR np.ndarray.

        Returns:
            SegResult with segmentation results.
        """
        img = (
            cv2.imread(source)
            if isinstance(source, (str, Path))
            else source.copy()
        )
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {source}")

        blob, orig_shape, gain, (dw, dh) = self._preprocess(img)
        outputs = self.session.run(
            self._output_names, {self._input_name: blob}
        )
        return self._postprocess(
            outputs[0], outputs[1], orig_shape, gain, (dw, dh)
        )

    def draw(
        self,
        source: str | np.ndarray,
        result: SegResult,
        fps: float | None = None,
    ) -> np.ndarray:
        """Draw segmentation results on the image."""
        img = (
            cv2.imread(source)
            if isinstance(source, (str, Path))
            else source.copy()
        )
        overlay = img.copy()

        for i in range(result.count):
            x1, y1, x2, y2 = result.boxes[i].astype(int)
            cls_id = int(result.class_ids[i])
            score = result.scores[i]
            color = PALETTE[cls_id].tolist()

            mask = result.masks[i]
            colored_mask = np.zeros_like(img)
            colored_mask[mask > 0] = color
            cv2.addWeighted(colored_mask, 0.5, overlay, 1, 0, overlay)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{COCO_NAMES[cls_id]} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1
            )
            cv2.putText(
                img, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                1, cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, 0.5, img, 1 - 0.5, 0, img)

        if fps is not None:
            cv2.putText(
                img, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                cv2.LINE_AA,
            )

        return img
