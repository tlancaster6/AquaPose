"""Midline stage (Stage 4) for the AquaPose v3.0 5-stage pipeline.

Provides the MidlineStage class that reads detection bounding boxes from
Stage 1, crops and runs the configured YOLO backend to produce 2D midlines.
Supports two backends: ``"segmentation"`` (YOLO-seg) and ``"pose_estimation"``
(YOLO-pose). Includes head-tail orientation resolution via cross-camera
geometry, velocity, and temporal prior. Populates
PipelineContext.annotated_detections.
"""

from aquapose.core.midline.backends.pose_estimation import PoseEstimationBackend
from aquapose.core.midline.backends.segmentation import SegmentationBackend
from aquapose.core.midline.orientation import (
    OrientationConfigLike,
    resolve_orientation,
)
from aquapose.core.midline.stage import MidlineStage
from aquapose.core.midline.types import AnnotatedDetection
from aquapose.core.types.midline import Midline2D

__all__ = [
    "AnnotatedDetection",
    "Midline2D",
    "MidlineStage",
    "OrientationConfigLike",
    "PoseEstimationBackend",
    "SegmentationBackend",
    "resolve_orientation",
]
