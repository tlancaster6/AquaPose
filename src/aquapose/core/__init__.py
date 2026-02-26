"""Core domain logic for 3D fish pose estimation.

Exports all 5 pipeline stage classes. Each stage satisfies the Stage Protocol
via structural typing (no inheritance required).

Stage ordering:
1. DetectionStage  — raw fish detection per-camera
2. MidlineStage    — 2D midline extraction per detection
3. AssociationStage — cross-camera fish identity grouping
4. TrackingStage   — temporal track assignment
5. ReconstructionStage — 3D B-spline midline triangulation
"""

from aquapose.core.association import AssociationStage
from aquapose.core.detection import DetectionStage
from aquapose.core.midline import MidlineStage
from aquapose.core.reconstruction import ReconstructionStage
from aquapose.core.tracking import TrackingStage

__all__ = [
    "AssociationStage",
    "DetectionStage",
    "MidlineStage",
    "ReconstructionStage",
    "TrackingStage",
]
