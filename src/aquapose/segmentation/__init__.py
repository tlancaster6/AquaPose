"""Multi-view fish segmentation pipeline."""

from .crop import CropRegion, compute_crop_region, extract_crop, paste_mask
from .dataset import CropDataset
from .detector import Detection, MOG2Detector, YOLODetector, make_detector
from .model import MaskRCNNSegmentor, SegmentationResult
from .pseudo_labeler import (
    AnnotatedFrame,
    FrameAnnotation,
    SAMPseudoLabeler,
    filter_mask,
    to_coco_dataset,
)
from .training import evaluate, train

__all__ = [
    "AnnotatedFrame",
    "CropDataset",
    "CropRegion",
    "Detection",
    "FrameAnnotation",
    "MOG2Detector",
    "MaskRCNNSegmentor",
    "SAMPseudoLabeler",
    "SegmentationResult",
    "YOLODetector",
    "compute_crop_region",
    "evaluate",
    "extract_crop",
    "filter_mask",
    "make_detector",
    "paste_mask",
    "to_coco_dataset",
    "train",
]
