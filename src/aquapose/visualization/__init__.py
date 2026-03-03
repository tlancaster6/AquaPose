"""Visualization tools for 2D overlay rendering, 3D midline animation, and diagnostics.

Provides utilities to reproject 3D fish midlines onto camera frames,
render 3D animations of midlines in tank coordinates, and generate
per-stage diagnostic visualizations for the reconstruction pipeline.
"""

from .frames import synthetic_frame_iter
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

__all__ = [
    "FISH_COLORS",
    "TrackSnapshot",
    "draw_midline_overlay",
    "plot_3d_frame",
    "render_3d_animation",
    "render_overlay_video",
    "synthetic_frame_iter",
    "vis_claiming_overlay",
    "vis_confidence_histogram",
    "vis_detection_grid",
    "vis_midline_extraction_montage",
    "vis_skip_reason_pie",
]
