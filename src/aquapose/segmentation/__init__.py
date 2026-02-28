"""Multi-view fish segmentation pipeline."""

from .crop import CropRegion, compute_crop_region, extract_crop, paste_mask
from .detector import Detection, MOG2Detector, YOLODetector, make_detector
from .model import MaskRCNNSegmentor, SegmentationResult, UNetSegmentor
from .pseudo_labeler import (
    AnnotatedFrame,
    FrameAnnotation,
    SAMPseudoLabeler,
    filter_mask,
    to_coco_dataset,
)

__all__ = [
    "AnnotatedFrame",
    "CropRegion",
    "Detection",
    "FrameAnnotation",
    "MOG2Detector",
    "MaskRCNNSegmentor",
    "SAMPseudoLabeler",
    "SegmentationResult",
    "UNetSegmentor",
    "YOLODetector",
    "compute_crop_region",
    "extract_crop",
    "filter_mask",
    "make_detector",
    "paste_mask",
    "to_coco_dataset",
]
