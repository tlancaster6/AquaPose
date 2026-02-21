"""Visualization tools for 2D overlay rendering and 3D midline animation.

Provides utilities to reproject 3D fish midlines onto camera frames and
render 3D animations of midlines in tank coordinates.
"""

from .overlay import FISH_COLORS, draw_midline_overlay, render_overlay_video
from .plot3d import plot_3d_frame, render_3d_animation

__all__ = [
    "FISH_COLORS",
    "draw_midline_overlay",
    "plot_3d_frame",
    "render_3d_animation",
    "render_overlay_video",
]
