"""YOLO-based fish detector and detection data types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Detection:
    """A single fish detection.

    Attributes:
        bbox: Bounding box as (x, y, w, h) in pixel coordinates.
        mask: Full-frame binary mask (uint8, 0/255) for this component only.
            ``None`` when the mask is equivalent to the bbox (e.g. YOLO
            detections without instance segmentation).
        area: Pixel area of the detected component.
        confidence: Detection confidence in [0, 1].
        angle: OBB rotation angle in radians, standard math convention,
            range ``[-pi, pi]``. ``None`` for non-OBB detectors (YOLO
            axis-aligned). Set by YOLO-OBB backend when available.
        obb_points: Oriented bounding box corner points, shape ``(4, 2)``,
            clockwise from top-left of the oriented box. ``None`` for
            non-OBB detectors. Set by YOLO-OBB backend when available.
    """

    bbox: tuple[int, int, int, int]
    mask: np.ndarray | None
    area: int
    confidence: float
    angle: float | None = None
    obb_points: np.ndarray | None = None


class YOLODetector:
    """Fish detector using YOLOv8 object detection.

    Wraps a trained ``ultralytics.YOLO`` model to produce :class:`Detection`
    objects. The ``ultralytics`` package is lazily imported inside ``__init__``
    to avoid import failures when it is not installed.

    Args:
        model_path: Path to trained YOLOv8 ``.pt`` weights file.
        conf_threshold: Minimum confidence score to keep a detection.
        iou_threshold: IoU threshold for non-max suppression. Lower values
            suppress more overlapping boxes (ultralytics default is 0.7).
        padding_fraction: Fraction of bbox dimension to add as symmetric
            padding (clamped to frame bounds).
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        padding_fraction: float = 0.15,
    ) -> None:
        from ultralytics import YOLO  # lazy import — ultralytics may not be installed

        self._model = YOLO(str(model_path))
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._padding_fraction = padding_fraction

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect fish in a frame, returning bounding boxes.

        Runs YOLOv8 inference on *frame* and converts each predicted bounding
        box to a :class:`Detection`.  The ``mask`` field is ``None`` since
        the detection is fully described by the bbox.

        Args:
            frame: BGR image as uint8 array of shape ``(H, W, 3)``.

        Returns:
            List of :class:`Detection` objects, one per detected fish.
        """
        h_frame, w_frame = frame.shape[:2]
        results = self._model.predict(
            frame, conf=self._conf, iou=self._iou, verbose=False
        )
        detections: list[Detection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                # Apply symmetric padding, clamped to frame bounds
                bw, bh = x2 - x1, y2 - y1
                pad_x = bw * self._padding_fraction
                pad_y = bh * self._padding_fraction
                rx1 = max(0, int(x1 - pad_x))
                ry1 = max(0, int(y1 - pad_y))
                rx2 = min(w_frame, int(x2 + pad_x))
                ry2 = min(h_frame, int(y2 + pad_y))
                bbox = (rx1, ry1, rx2 - rx1, ry2 - ry1)
                area = (rx2 - rx1) * (ry2 - ry1)
                detections.append(
                    Detection(bbox=bbox, mask=None, area=area, confidence=conf)
                )
        return detections


def make_detector(kind: str, **kwargs: Any) -> YOLODetector:
    """Create a fish detector by name.

    Note: ``"yolo_obb"`` is handled by
    ``core.detection.backends``, not this factory.

    Args:
        kind: Detector type — ``"yolo"``.
        **kwargs: Forwarded to the detector constructor. For ``"yolo"``,
            ``model_path`` is required.

    Returns:
        Configured detector instance.

    Raises:
        ValueError: If *kind* is not recognized.
    """
    if kind == "yolo":
        return YOLODetector(**kwargs)
    raise ValueError(f"Unknown detector_kind: {kind!r}. Available: 'yolo', 'yolo_obb'")
