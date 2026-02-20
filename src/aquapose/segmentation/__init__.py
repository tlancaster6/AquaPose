"""Multi-view fish segmentation pipeline."""

from .dataset import CropDataset
from .detector import Detection, MOG2Detector, YOLODetector, make_detector
from .label_studio import (
    export_to_label_studio,
    import_from_label_studio,
    to_coco_dataset,
)
from .model import MaskRCNNSegmentor, SegmentationResult
from .pseudo_labeler import AnnotatedFrame, FrameAnnotation, SAMPseudoLabeler
from .training import evaluate, train

__all__ = [
    "AnnotatedFrame",
    "CropDataset",
    "Detection",
    "FrameAnnotation",
    "MOG2Detector",
    "MaskRCNNSegmentor",
    "SAMPseudoLabeler",
    "SegmentationResult",
    "YOLODetector",
    "evaluate",
    "export_to_label_studio",
    "import_from_label_studio",
    "make_detector",
    "to_coco_dataset",
    "train",
]
