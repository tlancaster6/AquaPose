"""Synthetic data generation for controlled testing of reconstruction pipelines.

Provides tools to generate known ground truth 3D fish midlines, project them
through refractive camera models to produce 2D MidlineSets, and wire synthetic
data into the diagnostic pipeline via the --synthetic flag.
"""

from aquapose.synthetic.fish import (
    FishConfig,
    generate_fish_3d,
    generate_fish_half_widths,
    generate_synthetic_midline_sets,
    make_ground_truth_midline3d,
    project_fish_to_midline2d,
)
from aquapose.synthetic.rig import build_fabricated_rig
from aquapose.synthetic.stubs import (
    generate_synthetic_detections,
    generate_synthetic_tracks,
)

__all__ = [
    "FishConfig",
    "build_fabricated_rig",
    "generate_fish_3d",
    "generate_fish_half_widths",
    "generate_synthetic_detections",
    "generate_synthetic_midline_sets",
    "generate_synthetic_tracks",
    "make_ground_truth_midline3d",
    "project_fish_to_midline2d",
]
