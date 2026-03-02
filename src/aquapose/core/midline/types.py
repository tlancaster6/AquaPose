"""AnnotatedDetection type for the Midline stage (Stage 4).

Defines AnnotatedDetection — a Detection enriched with midline data produced
by the segmentation or pose-estimation backends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aquapose.core.types.crop import CropRegion
from aquapose.core.types.detection import Detection
from aquapose.core.types.midline import Midline2D

__all__ = ["AnnotatedDetection"]


@dataclass
class AnnotatedDetection:
    """A Detection enriched with segmentation mask and 2D midline data.

    Wraps the base Detection from Stage 1 and augments it with the binary mask
    and extracted midline produced by the segment-then-extract backend.

    The midline may be None when extraction failed (mask too small, boundary
    clipped, skeleton too short, etc.). Downstream stages must handle None.

    Attributes:
        detection: Original Detection from Stage 1 (bbox, confidence, area).
        mask: Binary crop-space mask (uint8, 0/255), shape (H_crop, W_crop).
            None if segmentation produced no result.
        crop_region: CropRegion that describes where in the full frame the mask
            was extracted from. None if segmentation produced no result.
        midline: Extracted 2D midline in full-frame pixel coordinates.
            None if midline extraction failed or was skipped.
        camera_id: Camera identifier string for this detection.
        frame_index: Frame index within the video.
    """

    detection: Detection
    mask: np.ndarray | None = None
    crop_region: CropRegion | None = None
    midline: Midline2D | None = None
    camera_id: str = ""
    frame_index: int = 0
