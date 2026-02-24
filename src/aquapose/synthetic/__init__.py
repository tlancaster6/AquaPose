"""Synthetic data generation for controlled testing of reconstruction pipelines.

Provides tools to generate known ground truth 3D fish midlines, project them
through refractive camera models to produce 2D MidlineSets, and wire synthetic
data into the diagnostic pipeline via the --synthetic flag.

Also provides multi-fish 3D trajectory generation and per-camera Detection
streams for evaluating the Cross-View Identity and 3D Tracking pipeline.
"""

from aquapose.synthetic.detection import (
    DetectionGenConfig,
    NoiseConfig,
    SyntheticDataset,
    SyntheticFrame,
    generate_detection_dataset,
)
from aquapose.synthetic.fish import (
    FishConfig,
    generate_fish_3d,
    generate_fish_half_widths,
    generate_synthetic_midline_sets,
    make_ground_truth_midline3d,
    project_fish_to_midline2d,
)
from aquapose.synthetic.rig import build_fabricated_rig
from aquapose.synthetic.scenarios import (
    crossing_paths,
    generate_scenario,
    startle_response,
    tight_schooling,
    track_fragmentation,
)
from aquapose.synthetic.stubs import (
    generate_synthetic_detections,
    generate_synthetic_tracks,
)
from aquapose.synthetic.trajectory import (
    MotionConfig,
    SchoolingConfig,
    TankConfig,
    TrajectoryConfig,
    TrajectoryResult,
    generate_trajectories,
)

__all__ = [
    "DetectionGenConfig",
    "FishConfig",
    "MotionConfig",
    "NoiseConfig",
    "SchoolingConfig",
    "SyntheticDataset",
    "SyntheticFrame",
    "TankConfig",
    "TrajectoryConfig",
    "TrajectoryResult",
    "build_fabricated_rig",
    "crossing_paths",
    "generate_detection_dataset",
    "generate_fish_3d",
    "generate_fish_half_widths",
    "generate_scenario",
    "generate_synthetic_detections",
    "generate_synthetic_midline_sets",
    "generate_synthetic_tracks",
    "generate_trajectories",
    "make_ground_truth_midline3d",
    "project_fish_to_midline2d",
    "startle_response",
    "tight_schooling",
    "track_fragmentation",
]
