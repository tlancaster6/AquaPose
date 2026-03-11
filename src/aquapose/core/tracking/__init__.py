"""2D Tracking stage (Stage 2) domain types for the AquaPose v2.1 pipeline.

Exports Tracklet2D — the per-camera temporal tracklet produced by Stage 2 — and
FishTrack/TrackState for downstream reconstruction compatibility.
Also exports TrackingStage — the OC-SORT-backed Stage 2 implementation and
the custom keypoint tracker components.
"""

from aquapose.core.tracking.keypoint_sigmas import (
    DEFAULT_SIGMAS,
    compute_keypoint_sigmas,
)
from aquapose.core.tracking.keypoint_tracker import (
    _KalmanFilter,
    _SinglePassTracker,
    build_cost_matrix,
    compute_heading,
    compute_ocm_matrix,
    compute_oks_matrix,
)
from aquapose.core.tracking.stage import TrackingStage
from aquapose.core.tracking.types import FishTrack, TrackHealth, Tracklet2D, TrackState

__all__ = [
    "DEFAULT_SIGMAS",
    "FishTrack",
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
]
