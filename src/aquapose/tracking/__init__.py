"""Tracking package for AquaPose v2.1.

Per-camera 2D tracking is performed by OcSortTracker, which wraps the boxmot
OC-SORT implementation and produces Tracklet2D objects. All boxmot internals are
fully isolated inside ocsort_wrapper.py — no other module imports from boxmot
directly.

TrackingStage (Stage 2) in aquapose.core.tracking delegates to OcSortTracker
per camera. FishTrack and TrackState have moved to aquapose.core.tracking.types
for reconstruction/visualization compatibility.
"""

from aquapose.tracking.ocsort_wrapper import OcSortTracker

__all__ = ["OcSortTracker"]
