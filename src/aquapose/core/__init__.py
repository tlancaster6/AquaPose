"""Core domain logic for 3D fish pose estimation.

Exports the active pipeline stage classes, core data contracts (PipelineContext,
ChunkHandoff, Stage Protocol), domain types, and cache utilities. Each stage
satisfies the Stage Protocol via structural typing (no inheritance required).

Stage ordering (v3.7):
1. DetectionStage      — raw fish detection per-camera
2. PoseStage           — raw keypoint extraction per detection (before tracking)
3. TrackingStage       — per-camera 2D tracklet generation via KeypointTracker
4. AssociationStage    — cross-camera identity clustering via Leiden algorithm
5. ReconstructionStage — 3D B-spline midline triangulation
"""

from aquapose.core.context import (
    ChunkHandoff,
    PipelineContext,
    Stage,
    StaleCacheError,
    context_fingerprint,
    load_chunk_cache,
    load_stage_cache,
)
from aquapose.core.detection import DetectionStage
from aquapose.core.inference import BatchState, predict_with_oom_retry
from aquapose.core.pose import PoseStage
from aquapose.core.reconstruction import ReconstructionStage
from aquapose.core.synthetic import SyntheticDataStage

__all__ = [
    "BatchState",
    "ChunkHandoff",
    "DetectionStage",
    "PipelineContext",
    "PoseStage",
    "ReconstructionStage",
    "Stage",
    "StaleCacheError",
    "SyntheticDataStage",
    "context_fingerprint",
    "load_chunk_cache",
    "load_stage_cache",
    "predict_with_oom_retry",
]
