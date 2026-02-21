"""Pose optimization and analysis-by-synthesis reconstruction."""

from .loss import compute_angular_diversity_weights, multi_objective_loss, soft_iou_loss
from .optimizer import FishOptimizer, make_optimizable_state, warm_start_from_velocity
from .renderer import RefractiveCamera, RefractiveSilhouetteRenderer
from .validation import evaluate_holdout_iou, render_overlay, run_holdout_validation

__all__ = [
    "FishOptimizer",
    "RefractiveCamera",
    "RefractiveSilhouetteRenderer",
    "compute_angular_diversity_weights",
    "evaluate_holdout_iou",
    "make_optimizable_state",
    "multi_objective_loss",
    "render_overlay",
    "run_holdout_validation",
    "soft_iou_loss",
    "warm_start_from_velocity",
]
