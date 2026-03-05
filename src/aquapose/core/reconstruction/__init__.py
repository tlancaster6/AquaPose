"""Reconstruction stage and types for the 5-stage AquaPose pipeline.

Stage 5: Triangulates tracked fish 2D midlines into 3D B-spline midlines.
Provides ReconstructionStage (satisfies Stage Protocol), the canonical
Midline3D output type, and temporal z-smoothing for z-denoising.
"""

from aquapose.core.reconstruction.stage import ReconstructionStage
from aquapose.core.reconstruction.temporal_smoothing import smooth_centroid_z
from aquapose.core.types.reconstruction import Midline3D

__all__ = [
    "Midline3D",
    "ReconstructionStage",
    "smooth_centroid_z",
]
