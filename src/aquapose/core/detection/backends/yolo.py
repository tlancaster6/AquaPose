"""YOLO detection backend for the Detection stage.

Provides :class:`YOLODetector` (wraps a trained YOLO model),
:func:`make_detector` (factory), and :class:`YOLOBackend` (pipeline
backend with fail-fast eager loading). Importing from this module avoids
any dependency on the legacy ``segmentation.detector`` module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aquapose.core.types.detection import Detection

__all__ = ["YOLOBackend", "YOLODetector", "make_detector"]


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


class YOLOBackend:
    """YOLO detection backend that wraps YOLODetector with eager loading.

    The model is loaded at construction time. If the weights file does not
    exist a :class:`FileNotFoundError` is raised immediately rather than
    at run time.

    Args:
        weights_path: Path to trained YOLOv8 ``.pt`` weights file.
        conf_threshold: Minimum confidence score to keep a detection.
        iou_threshold: IoU threshold for non-max suppression.
        padding_fraction: Fraction of bbox dimension to add as symmetric padding.
        device: Device string (e.g. ``"cuda"``, ``"cpu"``). Accepted for
            interface consistency with other backends but not used — YOLO
            auto-selects the device.

    Raises:
        FileNotFoundError: If *weights_path* does not point to an existing file.
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        padding_fraction: float = 0.15,
        device: str = "cuda",
    ) -> None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {weights_path}. "
                "Provide a valid path to a trained .pt weights file."
            )
        self._detector = YOLODetector(
            model_path=weights_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            padding_fraction=padding_fraction,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect fish in a single frame.

        Args:
            frame: BGR image as uint8 array of shape ``(H, W, 3)``.

        Returns:
            List of :class:`~aquapose.segmentation.detector.Detection` objects,
            one per detected fish.
        """
        return self._detector.detect(frame)
