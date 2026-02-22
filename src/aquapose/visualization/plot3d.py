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
        lo = np.percentile(combined, 2, axis=0)
        hi = np.percentile(combined, 98, axis=0)
        ranges = hi - lo
        max_range = float(ranges.max())
        if max_range > 0:
            centers = (lo + hi) / 2.0
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
    n_eval: int = 15,
) -> None:
    """Render an animated MP4 (or GIF fallback) of 3D fish midlines with body widths.

    Each frame draws fish midlines as spline curves with scatter points sized
    by the interpolated half-width at each evaluation point, giving a
    sausage-like body representation. Centroid trails from previous frames
    provide motion context.

    Axis bounds are precomputed from all frames so the camera stays fixed.

    Args:
        midlines_per_frame: Per-frame fish midlines. Each entry maps
            fish_id to Midline3D.
        output_path: Output file path. Extension determines format: ``.mp4``
            for FFMpeg, ``.gif`` for Pillow.
        fps: Frame rate for the animation.
        n_eval: Number of evaluation points per spline (and body spheres).
    """
    from matplotlib.animation import FFMpegWriter, PillowWriter

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not midlines_per_frame:
        logger.warning("render_3d_animation: no frames to render")
        return

    fig = plt.figure(figsize=(10, 8))
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore[assignment]

    u_vals = np.linspace(0.0, 1.0, n_eval)

    # Precompute fixed axis bounds from all midline points across all frames
    all_pts: list[np.ndarray] = []
    for frame_midlines in midlines_per_frame:
        for midline in frame_midlines.values():
            spline = scipy.interpolate.BSpline(
                midline.knots.astype(np.float64),
                midline.control_points.astype(np.float64),
                midline.degree,
            )
            all_pts.append(spline(u_vals))

    if not all_pts:
        logger.warning("render_3d_animation: no midline points")
        plt.close(fig)
        return

    combined = np.vstack(all_pts)
    # Use 2nd/98th percentile bounds to ignore outlier reconstructions
    lo = np.percentile(combined, 2, axis=0)
    hi = np.percentile(combined, 98, axis=0)
    ranges = hi - lo
    max_range = float(ranges.max())
    pad = max(max_range * 0.1, 0.01)
    centers = (lo + hi) / 2.0
    half = max_range / 2.0 + pad

    # Precompute centroids for trail rendering
    def _centroid(midline: Midline3D) -> np.ndarray:
        return midline.control_points.mean(axis=0)

    # Scale factor: convert half-width in metres to scatter marker size (points^2).
    # matplotlib scatter `s` is area in points^2. We map half-width to a visible
    # diameter relative to the axis range. Tuned so a typical fish (~0.02m hw)
    # gives a reasonable dot size.
    hw_scale = (72.0 * 10.0 / max(max_range, 0.01)) ** 2

    from matplotlib.artist import Artist

    def _update(frame_idx: int) -> list[Artist]:
        ax.cla()
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"Frame {frame_idx}")

        # Fixed axis bounds
        ax.set_xlim(centers[0] - half, centers[0] + half)
        ax.set_ylim(centers[1] - half, centers[1] + half)
        ax.set_zlim(centers[2] - half, centers[2] + half)

        frame_midlines = midlines_per_frame[frame_idx]

        for fish_id, midline in frame_midlines.items():
            color = _get_rgb_color(fish_id)
            spline = scipy.interpolate.BSpline(
                midline.knots.astype(np.float64),
                midline.control_points.astype(np.float64),
                midline.degree,
            )
            pts = spline(u_vals)  # (n_eval, 3)

            # Spline line
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=1)

            # Interpolate half-widths to n_eval points
            hw = midline.half_widths
            if len(hw) == n_eval:
                hw_interp = hw.astype(np.float64)
            else:
                hw_u = np.linspace(0.0, 1.0, len(hw))
                hw_interp = np.interp(u_vals, hw_u, hw.astype(np.float64))

            # Scatter sized by half-width
            sizes = hw_interp**2 * hw_scale
            sizes = np.clip(sizes, 5, 500)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=sizes,
                c=[color],
                alpha=0.35,
                edgecolors=color,
                linewidths=0.5,
            )

        # Centroid trails (last _TRAIL_FRAMES frames)
        trail_start = max(0, frame_idx - _TRAIL_FRAMES)
        for trail_fi in range(trail_start, frame_idx):
            alpha = (trail_fi - trail_start + 1) / _TRAIL_FRAMES
            for fish_id, midline in midlines_per_frame[trail_fi].items():
                color = _get_rgb_color(fish_id)
                centroid = _centroid(midline)
                ax.scatter(
                    centroid[0],
                    centroid[1],
                    centroid[2],
                    color=color,
                    alpha=alpha * 0.5,
                    s=8,
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
