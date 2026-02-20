"""Cold-start 3D initialization pipeline for fish pose estimation.

Provides PCA-based keypoint extraction from binary masks, multi-camera
refractive triangulation, and FishState estimation from triangulated keypoints.
"""

from .keypoints import extract_keypoints, extract_keypoints_batch
from .triangulator import (
    init_fish_state,
    init_fish_states_from_masks,
    triangulate_keypoint,
)

__all__ = [
    "extract_keypoints",
    "extract_keypoints_batch",
    "init_fish_state",
    "init_fish_states_from_masks",
    "triangulate_keypoint",
]
