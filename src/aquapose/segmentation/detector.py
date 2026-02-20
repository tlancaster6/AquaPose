"""MOG2 background-subtraction fish detector with morphological cleanup."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    """A single fish detection from background subtraction.

    Attributes:
        bbox: Bounding box as (x, y, w, h) in pixel coordinates.
        mask: Full-frame binary mask (uint8, 0/255) for this component only.
        area: Pixel area of the detected component.
        confidence: Detection confidence (always 1.0 for MOG2; placeholder
            for downstream compatibility).
    """

    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    area: int
    confidence: float


class MOG2Detector:
    """Fish detector using MOG2 background subtraction with morphological cleanup.

    Wraps ``cv2.BackgroundSubtractorMOG2`` with connected-component filtering
    to produce bounding boxes and per-component masks from video frames.
    Frames must be fed in temporal order per camera (caller responsibility).

    Args:
        history: Number of frames for background model history.
        var_threshold: Variance threshold for foreground classification.
        detect_shadows: Whether MOG2 should detect shadows (gray=127).
        min_area: Minimum connected-component area in pixels to keep.
        padding_fraction: Fraction of bbox dimension to add as padding.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: int = 12,
        detect_shadows: bool = True,
        min_area: int = 200,
        padding_fraction: float = 0.15,
    ) -> None:
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self._mog2.setShadowThreshold(200)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._min_area = min_area
        self._padding_fraction = padding_fraction

    def apply(self, frame: np.ndarray) -> None:
        """Feed a frame to the background model without returning detections.

        Args:
            frame: BGR image as uint8 array of shape (H, W, 3).
        """
        self._mog2.apply(frame)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect fish in a frame, returning bounding boxes and masks.

        Internally calls :meth:`apply` to update the background model,
        then extracts foreground blobs via morphological cleanup and
        connected-component analysis.

        Args:
            frame: BGR image as uint8 array of shape (H, W, 3).

        Returns:
            List of :class:`Detection` objects, one per fish-sized blob.
        """
        fg_mask = self._mog2.apply(frame)

        # Threshold at 255 to exclude shadow pixels (gray=127)
        _, binary = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)

        # Morphological close then open to clean up noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._kernel)

        # Connected components
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        h_frame, w_frame = frame.shape[:2]
        detections: list[Detection] = []

        for label_id in range(1, num_labels):  # skip background (label 0)
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < self._min_area:
                continue

            # Raw bbox from component stats
            cx = int(stats[label_id, cv2.CC_STAT_LEFT])
            cy = int(stats[label_id, cv2.CC_STAT_TOP])
            cw = int(stats[label_id, cv2.CC_STAT_WIDTH])
            ch = int(stats[label_id, cv2.CC_STAT_HEIGHT])

            # Pad bbox
            pad_x = int(cw * self._padding_fraction)
            pad_y = int(ch * self._padding_fraction)
            x1 = max(0, cx - pad_x)
            y1 = max(0, cy - pad_y)
            x2 = min(w_frame, cx + cw + pad_x)
            y2 = min(h_frame, cy + ch + pad_y)

            bbox = (x1, y1, x2 - x1, y2 - y1)

            # Full-frame mask for this component only
            component_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
            component_mask[labels == label_id] = 255

            detections.append(
                Detection(bbox=bbox, mask=component_mask, area=area, confidence=1.0)
            )

        return detections

    def warm_up(self, frames: Sequence[np.ndarray]) -> None:
        """Feed frames to build a stable background model before detection.

        Args:
            frames: Sequence of BGR images to feed through :meth:`apply`.
        """
        for frame in frames:
            self.apply(frame)
