"""Visualization tools for 2D overlay rendering, 3D midline animation, and diagnostics.

Provides utilities to reproject 3D fish midlines onto camera frames,
render 3D animations of midlines in tank coordinates, and generate
per-stage diagnostic visualizations for the reconstruction pipeline.
"""

from .midline_viz import (
    TrackSnapshot,
    vis_claiming_overlay,
    vis_confidence_histogram,
    vis_detection_grid,
    vis_midline_extraction_montage,
    vis_skip_reason_pie,
)
from .overlay import FISH_COLORS, draw_midline_overlay, render_overlay_video
from .plot3d import plot_3d_frame, render_3d_animation
from .triangulation_viz import (
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

__all__ = [
    "FISH_COLORS",
    "TrackSnapshot",
    "draw_midline_overlay",
    "plot_3d_frame",
    "render_3d_animation",
    "render_overlay_video",
    "vis_arclength_histogram",
    "vis_claiming_overlay",
    "vis_confidence_histogram",
    "vis_detection_grid",
    "vis_midline_extraction_montage",
    "vis_optimizer_progression",
    "vis_per_camera_spline_overlays",
    "vis_residual_heatmap",
    "vis_skip_reason_pie",
    "vis_synthetic_3d_comparison",
    "vis_synthetic_camera_overlays",
    "vis_synthetic_error_distribution",
    "write_diagnostic_report",
    "write_synthetic_report",
]
