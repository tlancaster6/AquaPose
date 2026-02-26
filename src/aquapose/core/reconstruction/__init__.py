"""Reconstruction stage and types for the 5-stage AquaPose pipeline.

Stage 5: Triangulates tracked fish 2D midlines into 3D B-spline midlines.
Provides ReconstructionStage (satisfies Stage Protocol) and the canonical
Midline3D output type.
"""

from aquapose.core.reconstruction.stage import ReconstructionStage
from aquapose.reconstruction.triangulation import Midline3D

__all__ = ["Midline3D", "ReconstructionStage"]
