"""Multi-view fish segmentation pipeline."""

from .detector import Detection, MOG2Detector
from .label_studio import (
    export_to_label_studio,
    import_from_label_studio,
    to_coco_dataset,
)
from .pseudo_labeler import AnnotatedFrame, FrameAnnotation, SAMPseudoLabeler

__all__ = [
    "AnnotatedFrame",
    "Detection",
    "FrameAnnotation",
    "MOG2Detector",
    "SAMPseudoLabeler",
    "export_to_label_studio",
    "import_from_label_studio",
    "to_coco_dataset",
]
