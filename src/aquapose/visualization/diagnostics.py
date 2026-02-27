"""Backward-compatible re-exports from split visualization modules.

This module previously contained all 2200 LOC of diagnostic visualization.
It has been split into focused modules:
  - ``aquapose.visualization.midline_viz`` — detection, tracking, midline viz
  - ``aquapose.visualization.triangulation_viz`` — triangulation, synthetic, optimizer viz

All public names are re-exported here for backward compatibility.
"""

from aquapose.visualization.midline_viz import (  # noqa: F401
    TrackSnapshot,
    vis_claiming_overlay,
    vis_confidence_histogram,
    vis_detection_grid,
    vis_midline_extraction_montage,
    vis_skip_reason_pie,
)
from aquapose.visualization.triangulation_viz import (  # noqa: F401
    vis_arclength_histogram,
    vis_optimizer_progression,
    vis_per_camera_spline_overlays,
    vis_residual_heatmap,
    vis_synthetic_3d_comparison,
    vis_synthetic_camera_overlays,
    vis_synthetic_error_distribution,
    write_diagnostic_report,
    write_synthetic_report,
)
