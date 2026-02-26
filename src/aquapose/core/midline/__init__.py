"""Midline stage (Stage 2) for the AquaPose 5-stage pipeline.

Provides the MidlineStage class that reads detection bounding boxes from
Stage 1, crops and segments each detection via U-Net, then extracts 15-point
2D midlines with half-widths via skeletonization + BFS pruning. Populates
PipelineContext.annotated_detections.
"""

from aquapose.core.midline.stage import MidlineStage
from aquapose.core.midline.types import AnnotatedDetection
from aquapose.reconstruction.midline import Midline2D

__all__ = ["AnnotatedDetection", "Midline2D", "MidlineStage"]
