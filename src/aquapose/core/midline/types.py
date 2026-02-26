"""Stage-specific types for the Midline stage (Stage 2).

Re-exports Midline2D and CropRegion from their canonical modules so
downstream code can import from a single location within the core package.
Also defines AnnotatedDetection — a Detection enriched with midline data
produced by the segment-then-extract backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aquapose.reconstruction.midline import Midline2D
from aquapose.segmentation.crop import CropRegion
from aquapose.segmentation.detector import Detection

__all__ = ["AnnotatedDetection", "CropRegion", "Midline2D"]


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
