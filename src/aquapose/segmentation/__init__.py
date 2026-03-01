"""Multi-view fish segmentation pipeline."""

from .crop import CropRegion, compute_crop_region, extract_crop, paste_mask
from .detector import Detection, YOLODetector, make_detector

__all__ = [
    "CropRegion",
    "Detection",
    "YOLODetector",
    "compute_crop_region",
    "extract_crop",
    "make_detector",
    "paste_mask",
]
