"""Reconstruction stage and types for the 5-stage AquaPose pipeline.

Stage 5: Triangulates tracked fish 2D midlines into 3D B-spline midlines.
Provides ReconstructionStage (satisfies Stage Protocol), the canonical
Midline3D output type, plane-fitting utilities, and temporal smoothing
for z-denoising.
"""

from aquapose.core.reconstruction.plane_fit import (
    fit_plane_weighted,
    project_onto_plane,
)
from aquapose.core.reconstruction.stage import ReconstructionStage
from aquapose.core.reconstruction.temporal_smoothing import (
    rotate_control_points_to_plane,
    smooth_plane_normals,
)
from aquapose.core.types.reconstruction import Midline3D

__all__ = [
    "Midline3D",
    "ReconstructionStage",
    "fit_plane_weighted",
    "project_onto_plane",
    "rotate_control_points_to_plane",
    "smooth_plane_normals",
]
