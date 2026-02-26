"""Stage-specific types for the Tracking stage (Stage 4).

Re-exports FishTrack and TrackState from the canonical tracker module
(aquapose.tracking.tracker). These are the persistent track types used
by downstream stages such as Reconstruction (Stage 5).
"""

from __future__ import annotations

from aquapose.tracking.tracker import FishTrack, TrackState

__all__ = ["FishTrack", "TrackState"]
