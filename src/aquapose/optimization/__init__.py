"""Pose optimization and analysis-by-synthesis reconstruction."""

from .loss import compute_angular_diversity_weights, multi_objective_loss, soft_iou_loss
from .renderer import RefractiveCamera, RefractiveSilhouetteRenderer

__all__ = [
    "RefractiveCamera",
    "RefractiveSilhouetteRenderer",
    "compute_angular_diversity_weights",
    "multi_objective_loss",
    "soft_iou_loss",
]
