"""Stage-level pure-function evaluators for all five pipeline stages."""

from aquapose.evaluation.stages.association import (
    DEFAULT_GRID as ASSOCIATION_DEFAULT_GRID,
)
from aquapose.evaluation.stages.association import (
    AssociationMetrics,
    evaluate_association,
)
from aquapose.evaluation.stages.detection import DetectionMetrics, evaluate_detection
from aquapose.evaluation.stages.midline import MidlineMetrics, evaluate_midline
from aquapose.evaluation.stages.reconstruction import (
    DEFAULT_GRID as RECONSTRUCTION_DEFAULT_GRID,
)
from aquapose.evaluation.stages.reconstruction import (
    ReconstructionMetrics,
    ZDenoisingMetrics,
    compute_z_denoising_metrics,
    evaluate_reconstruction,
)
from aquapose.evaluation.stages.tracking import TrackingMetrics, evaluate_tracking

__all__ = [
    "ASSOCIATION_DEFAULT_GRID",
    "RECONSTRUCTION_DEFAULT_GRID",
    "AssociationMetrics",
    "DetectionMetrics",
    "MidlineMetrics",
    "ReconstructionMetrics",
    "TrackingMetrics",
    "ZDenoisingMetrics",
    "compute_z_denoising_metrics",
    "evaluate_association",
    "evaluate_detection",
    "evaluate_midline",
    "evaluate_reconstruction",
    "evaluate_tracking",
]
