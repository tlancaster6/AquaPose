"""Crop-segment-paste utilities for ROI-based mask prediction.

Provides shared machinery to crop an image region around a bounding box,
run a segmentation model on the crop, and paste the resulting mask back
into the full frame. Used by SAMPseudoLabeler and Mask R-CNN inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CropRegion:
    """A padded crop region within a full-frame image.

    Attributes:
        x1: Left edge of the crop in the full frame (clipped to 0).
        y1: Top edge of the crop in the full frame (clipped to 0).
        x2: Right edge of the crop in the full frame (clipped to frame width).
        y2: Bottom edge of the crop in the full frame (clipped to frame height).
        frame_h: Height of the full frame.
        frame_w: Width of the full frame.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    frame_h: int
    frame_w: int

    @property
    def width(self) -> int:
        """Width of the crop region in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Height of the crop region in pixels."""
        return self.y2 - self.y1


def compute_crop_region(
    bbox: tuple[int, int, int, int],
    frame_h: int,
    frame_w: int,
    padding: float = 0.25,
) -> CropRegion:
    """Compute a padded crop region around a bounding box.

    Adds fractional padding relative to bbox dimensions, then clips
    to frame boundaries.

    Args:
        bbox: Bounding box as (x, y, w, h) in pixel coordinates.
        frame_h: Full frame height.
        frame_w: Full frame width.
        padding: Fractional padding relative to bbox size. 0.25 means
            25% of the bbox dimension is added on each side.

    Returns:
        CropRegion with clipped coordinates.
    """
    x, y, w, h = bbox
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_w, x + w + pad_x)
    y2 = min(frame_h, y + h + pad_y)

    return CropRegion(x1=x1, y1=y1, x2=x2, y2=y2, frame_h=frame_h, frame_w=frame_w)


def extract_crop(image: np.ndarray, region: CropRegion) -> np.ndarray:
    """Extract a crop from a full-frame image.

    Args:
        image: Full-frame image array of shape (H, W) or (H, W, C).
        region: CropRegion specifying the area to extract.

    Returns:
        Cropped image array.
    """
    return image[region.y1 : region.y2, region.x1 : region.x2].copy()


def paste_mask(
    crop_mask: np.ndarray,
    region: CropRegion,
) -> np.ndarray:
    """Paste a crop-sized mask back into a full-frame binary mask.

    Args:
        crop_mask: Binary mask of the crop region, shape (crop_h, crop_w).
            Values should be 0 or 255 (uint8).
        region: CropRegion that defines where to paste.

    Returns:
        Full-frame binary mask (uint8, 0/255) of shape (frame_h, frame_w).
    """
    full_mask = np.zeros((region.frame_h, region.frame_w), dtype=np.uint8)

    # Resize crop_mask if it doesn't match the region dimensions
    if crop_mask.shape[:2] != (region.height, region.width):
        crop_mask = cv2.resize(
            crop_mask,
            (region.width, region.height),
            interpolation=cv2.INTER_NEAREST,
        )

    full_mask[region.y1 : region.y2, region.x1 : region.x2] = crop_mask
    return full_mask
