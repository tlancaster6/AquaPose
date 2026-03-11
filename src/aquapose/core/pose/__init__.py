"""Pose stage (Stage 2) for the AquaPose 5-stage pipeline.

Provides the PoseStage class that reads detection bounding boxes from
Stage 1, crops and runs YOLO-pose inference to produce raw anatomical
keypoints. Keypoints are written directly onto Detection objects.
Supports the ``"pose_estimation"`` backend (YOLO-pose).
"""

from aquapose.core.pose.backends.pose_estimation import PoseEstimationBackend
from aquapose.core.pose.stage import PoseStage

__all__ = [
    "PoseEstimationBackend",
    "PoseStage",
]
