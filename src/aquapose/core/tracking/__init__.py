"""2D Tracking stage (Stage 2) domain types for the AquaPose pipeline.

Exports Tracklet2D -- the per-camera temporal tracklet produced by Stage 2 --
and FishTrack/TrackState for downstream reconstruction compatibility.
Also exports TrackingStage and KeypointTracker (single-pass keypoint tracker
with ORU/OCR occlusion recovery).
"""

from aquapose.core.tracking.keypoint_sigmas import (
    DEFAULT_SIGMAS,
    compute_keypoint_sigmas,
)
from aquapose.core.tracking.keypoint_tracker import (
    KeypointTracker,
    _KalmanFilter,
    _SinglePassTracker,
    build_cost_matrix,
    compute_heading,
    compute_ocm_matrix,
    compute_oks_matrix,
    interpolate_gaps,
)
from aquapose.core.tracking.stage import TrackingStage
from aquapose.core.tracking.types import FishTrack, TrackHealth, Tracklet2D, TrackState

__all__ = [
    "DEFAULT_SIGMAS",
    "FishTrack",
    "KeypointTracker",
    "TrackHealth",
    "TrackState",
    "TrackingStage",
    "Tracklet2D",
    "_KalmanFilter",
    "_SinglePassTracker",
    "build_cost_matrix",
    "compute_heading",
    "compute_keypoint_sigmas",
    "compute_ocm_matrix",
    "compute_oks_matrix",
    "interpolate_gaps",
]
