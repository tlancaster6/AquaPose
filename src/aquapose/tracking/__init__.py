"""Cross-view fish identity association and temporal tracking."""

from .associate import (
    AssociationResult,
    ClaimResult,
    FrameAssociations,
    claim_detections_for_tracks,
    discover_births,
    ransac_centroid_cluster,
)
from .tracker import FishTrack, FishTracker, TrackHealth, TrackState
from .writer import TrackingWriter, read_tracking_results

__all__ = [
    "AssociationResult",
    "ClaimResult",
    "FishTrack",
    "FishTracker",
    "FrameAssociations",
    "TrackHealth",
    "TrackState",
    "TrackingWriter",
    "claim_detections_for_tracks",
    "discover_births",
    "ransac_centroid_cluster",
    "read_tracking_results",
]
