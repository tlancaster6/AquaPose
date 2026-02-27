"""Core domain logic for 3D fish pose estimation.

Exports the active pipeline stage classes, core data contracts (PipelineContext,
CarryForward, Stage Protocol), and domain types. Each stage satisfies the Stage
Protocol via structural typing (no inheritance required).

Stage ordering (v2.1):
1. DetectionStage      — raw fish detection per-camera
2. TrackingStage       — stub; per-camera 2D tracklet generation (Phase 24)
3. AssociationStage    — stub; cross-camera identity clustering (Phase 25)
4. MidlineStage        — 2D midline extraction per detection
5. ReconstructionStage — 3D B-spline midline triangulation
"""

from aquapose.core.context import CarryForward, PipelineContext, Stage
from aquapose.core.detection import DetectionStage
from aquapose.core.midline import MidlineStage
from aquapose.core.reconstruction import ReconstructionStage
from aquapose.core.synthetic import SyntheticDataStage

__all__ = [
    "CarryForward",
    "DetectionStage",
    "MidlineStage",
    "PipelineContext",
    "ReconstructionStage",
    "Stage",
    "SyntheticDataStage",
]
