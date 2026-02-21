"""3D visualization of fish midlines in tank coordinates.

Renders B-spline midlines as 3D line plots and produces MP4 or GIF
animations using matplotlib FuncAnimation.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.interpolate
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

if TYPE_CHECKING:
    from aquapose.reconstruction.triangulation import Midline3D

# Import the shared BGR palette and convert to RGB (0-1 floats) for matplotlib
from aquapose.visualization.overlay import FISH_COLORS

logger = logging.getLogger(__name__)

# Convert BGR to RGB float tuples for matplotlib
_FISH_COLORS_RGB: list[tuple[float, float, float]] = [
    (b / 255.0, g / 255.0, r / 255.0) for (b, g, r) in FISH_COLORS
]

# Number of tail frames to show trajectory trails
_TRAIL_FRAMES: int = 10


def _get_rgb_color(fish_id: int) -> tuple[float, float, float]:
    """Return RGB float tuple for a given fish_id."""
    return _FISH_COLORS_RGB[fish_id % len(_FISH_COLORS_RGB)]


def plot_3d_frame(
    midlines: dict[int, Midline3D],
    ax: Axes3D | None = None,
    *,
    n_eval: int = 30,
) -> Figure:
    """Plot 3D midlines for all fish in a single frame.

    Evaluates each fish's B-spline and plots as a labelled 3D line. Axis
    labels are set in metres with equal aspect ratio.

    Args:
        midlines: Dict mapping fish_id to Midline3D.
        ax: Existing Axes3D to draw on. If None, a new figure is created.
        n_eval: Number of evaluation points per spline.

    Returns:
        The matplotlib Figure containing the 3D plot.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        raw_fig = ax.get_figure()
        if not isinstance(raw_fig, Figure):
            raise ValueError("ax has no figure or is a SubFigure")
        fig = raw_fig

    u_vals = np.linspace(0.0, 1.0, n_eval)

    all_pts: list[np.ndarray] = []
    for fish_id, midline in midlines.items():
        color = _get_rgb_color(fish_id)
        spline = scipy.interpolate.BSpline(
            midline.knots.astype(np.float64),
            midline.control_points.astype(np.float64),
            midline.degree,
        )
        pts = spline(u_vals)  # shape (n_eval, 3)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, label=f"Fish {fish_id}")
        all_pts.append(pts)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Fish Midlines")

    if midlines:
        ax.legend(loc="upper right", fontsize=8)

    # Equal aspect ratio approximation for 3D axes
    if all_pts:
        combined = np.vstack(all_pts)
        ranges = combined.max(axis=0) - combined.min(axis=0)
        max_range = float(ranges.max())
        if max_range > 0:
            centers = (combined.max(axis=0) + combined.min(axis=0)) / 2.0
            for set_lim, center in zip(
                [ax.set_xlim, ax.set_ylim, ax.set_zlim], centers, strict=True
            ):
                set_lim(center - max_range / 2.0, center + max_range / 2.0)

    return fig


def render_3d_animation(
    midlines_per_frame: list[dict[int, Midline3D]],
    output_path: Path,
    *,
    fps: int = 15,
    n_eval: int = 30,
) -> None:
    """Render an animated MP4 (or GIF fallback) of 3D fish midlines.

    Each frame draws all fish midlines plus centroid trail dots for the last
    ``_TRAIL_FRAMES`` frames. Fish colors are consistent across frames.

    Args:
        midlines_per_frame: Per-frame fish midlines. Each entry maps
            fish_id to Midline3D.
        output_path: Output file path. Extension determines format: ``.mp4``
            for FFMpeg, ``.gif`` for Pillow.
        fps: Frame rate for the animation.
        n_eval: Number of evaluation points per spline.
    """
    from matplotlib.animation import FFMpegWriter, PillowWriter

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not midlines_per_frame:
        logger.warning("render_3d_animation: no frames to render")
        return

    fig = plt.figure(figsize=(8, 6))
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore[assignment]

    u_vals = np.linspace(0.0, 1.0, n_eval)

    # Precompute centroids per frame per fish for trail rendering
    def _centroid(midline: Midline3D) -> np.ndarray:
        return midline.control_points.mean(axis=0)  # shape (3,)

    from matplotlib.artist import Artist

    def _update(frame_idx: int) -> list[Artist]:
        ax.cla()
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"Frame {frame_idx}")

        frame_midlines = midlines_per_frame[frame_idx]

        # Draw midlines
        for fish_id, midline in frame_midlines.items():
            color = _get_rgb_color(fish_id)
            spline = scipy.interpolate.BSpline(
                midline.knots.astype(np.float64),
                midline.control_points.astype(np.float64),
                midline.degree,
            )
            pts = spline(u_vals)  # (n_eval, 3)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color)

        # Draw centroid trails (last _TRAIL_FRAMES frames)
        trail_start = max(0, frame_idx - _TRAIL_FRAMES)
        for trail_frame_idx in range(trail_start, frame_idx):
            alpha = (trail_frame_idx - trail_start + 1) / _TRAIL_FRAMES
            trail_midlines = midlines_per_frame[trail_frame_idx]
            for fish_id, midline in trail_midlines.items():
                color = _get_rgb_color(fish_id)
                centroid = _centroid(midline)
                ax.scatter(
                    centroid[0],
                    centroid[1],
                    centroid[2],
                    color=color,
                    alpha=alpha * 0.7,
                    s=10,
                )

        return []

    n_frames = len(midlines_per_frame)
    anim = FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=1000 // fps,
        blit=False,
    )

    # Determine writer based on FFMpeg availability
    if FFMpegWriter.isAvailable():
        writer = FFMpegWriter(fps=fps)
        save_path = output_path.with_suffix(".mp4")
        logger.info("Saving 3D animation with FFMpegWriter to %s", save_path)
    else:
        warnings.warn(
            "FFMpeg not available — falling back to PillowWriter (GIF output). "
            "Install FFMpeg for MP4 output.",
            UserWarning,
            stacklevel=2,
        )
        writer = PillowWriter(fps=fps)  # type: ignore[assignment]
        save_path = output_path.with_suffix(".gif")
        logger.info("Saving 3D animation with PillowWriter (GIF) to %s", save_path)

    anim.save(str(save_path), writer=writer)
    plt.close(fig)
    logger.info("3D animation saved to %s", save_path)
