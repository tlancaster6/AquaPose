"""Tracking stage (Stage 4) for the AquaPose 5-stage pipeline.

Provides the TrackingStage class that assigns persistent fish identities
across frames via temporal association. Populates PipelineContext.tracks.

Also exports FishTrack and TrackState for downstream stages that need to
interact with track objects (e.g., Stage 5 Reconstruction).
"""

from aquapose.core.tracking.stage import TrackingStage
from aquapose.core.tracking.types import FishTrack, TrackState

__all__ = ["FishTrack", "TrackState", "TrackingStage"]
