"""YOLO detection backend for the Detection stage.

Thin wrapper around :class:`~aquapose.segmentation.detector.YOLODetector`
that loads the model eagerly at construction time, providing a fail-fast
interface when the weights file is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from aquapose.segmentation.detector import Detection, YOLODetector

if TYPE_CHECKING:
    pass

__all__ = ["YOLOBackend"]


class YOLOBackend:
    """YOLO detection backend that wraps YOLODetector with eager loading.

    The model is loaded at construction time. If the weights file does not
    exist a :class:`FileNotFoundError` is raised immediately rather than
    at run time.

    Args:
        model_path: Path to trained YOLOv8 ``.pt`` weights file.
        conf_threshold: Minimum confidence score to keep a detection.
        iou_threshold: IoU threshold for non-max suppression.
        padding_fraction: Fraction of bbox dimension to add as symmetric padding.

    Raises:
        FileNotFoundError: If *model_path* does not point to an existing file.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        padding_fraction: float = 0.15,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {model_path}. "
                "Provide a valid path to a trained .pt weights file."
            )
        self._detector = YOLODetector(
            model_path=model_path,
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
