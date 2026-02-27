"""2D Tracking stage (Stage 2) domain types for the AquaPose v2.1 pipeline.

Exports Tracklet2D — the per-camera temporal tracklet produced by Stage 2 — and
FishTrack/TrackState for downstream reconstruction compatibility.
"""

from aquapose.core.tracking.types import FishTrack, TrackHealth, Tracklet2D, TrackState

__all__ = ["FishTrack", "TrackHealth", "TrackState", "Tracklet2D"]
