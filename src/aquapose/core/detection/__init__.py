"""Detection stage (Stage 1) for the AquaPose 5-stage pipeline.

Provides the DetectionStage class that reads video frames across all
cameras, runs YOLO object detection, and populates PipelineContext.
"""

from aquapose.core.detection.stage import DetectionStage
from aquapose.core.detection.types import Detection

__all__ = ["Detection", "DetectionStage"]
