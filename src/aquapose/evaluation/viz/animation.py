"""3D midline animation visualization: interactive Plotly HTML across all chunks."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import plotly.colors
import plotly.graph_objects as go
import scipy.interpolate

from aquapose.evaluation.viz._loader import load_midlines_from_h5, resolve_h5_path

logger = logging.getLogger(__name__)

# Number of points to evaluate along each B-spline.
_N_EVAL_POINTS: int = 50
_LINE_WIDTH: float = 4.0


def _eval_spline(
    spline: object,
    n_eval: int = _N_EVAL_POINTS,
) -> tuple[list[float], list[float], list[float]]:
    """Evaluate a B-spline or extract raw keypoints and return (x, y, z) lists.

    Args:
        spline: Midline3D-like object with optional control_points, knots,
            degree, and points attributes.
        n_eval: Number of evaluation points along the spline (ignored for raw
            keypoints).

    Returns:
        Tuple of (x_list, y_list, z_list) for Plotly traces.
    """
    control_points = getattr(spline, "control_points", None)
    if control_points is not None:
        cp = np.asarray(control_points, dtype=np.float64)
        if not np.all(np.isnan(cp)):
            knots = getattr(spline, "knots", None)
            degree = getattr(spline, "degree", None)
            if knots is not None and degree is not None:
                try:
                    bspl = scipy.interpolate.BSpline(
                        np.asarray(knots, dtype=np.float64), cp, degree
                    )
                    t_vals = np.linspace(0.0, 1.0, n_eval)
                    pts = bspl(t_vals)  # (N, 3)
                    return pts[:, 0].tolist(), pts[:, 1].tolist(), pts[:, 2].tolist()
                except Exception:
                    pass

    # Fall back to raw keypoints.
    raw_pts = getattr(spline, "points", None)
    if raw_pts is not None:
        pts = np.asarray(raw_pts, dtype=np.float64)
        if pts.ndim == 2 and pts.shape[1] == 3 and not np.all(np.isnan(pts)):
            return pts[:, 0].tolist(), pts[:, 1].tolist(), pts[:, 2].tolist()
    return [], [], []


def _build_figure(
    midlines_3d: list[dict],
    fish_colors: dict[int, str],
    sorted_fish_ids: list[int],
) -> go.Figure:
    """Build a Plotly animated figure for all frames across all chunks.

    Args:
        midlines_3d: Merged per-frame dicts mapping fish_id to Spline3D objects.
        fish_colors: Mapping of fish_id to Plotly color string.
        sorted_fish_ids: Sorted list of all unique fish IDs.

    Returns:
        Plotly Figure with animation frames and scrubber controls.
    """
    # Build initial traces (frame 0).
    frame_0 = midlines_3d[0] if midlines_3d else {}
    if not isinstance(frame_0, dict):
        frame_0 = {}

    initial_traces: list[go.Scatter3d] = []
    for fid in sorted_fish_ids:
        x, y, z = _eval_spline(frame_0[fid]) if fid in frame_0 else ([], [], [])
        initial_traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                name=f"Fish {fid}",
                line=dict(color=fish_colors[fid], width=_LINE_WIDTH),
            )
        )

    # Build animation frames.
    frames: list[go.Frame] = []
    for frame_idx, frame_dict in enumerate(midlines_3d):
        if not isinstance(frame_dict, dict):
            frame_dict = {}
        frame_traces: list[go.Scatter3d] = []
        for fid in sorted_fish_ids:
            x, y, z = (
                _eval_spline(frame_dict[fid]) if fid in frame_dict else ([], [], [])
            )
            frame_traces.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    name=f"Fish {fid}",
                    line=dict(color=fish_colors[fid], width=_LINE_WIDTH),
                )
            )
        frames.append(go.Frame(data=frame_traces, name=str(frame_idx)))

    # Compute global bounding box for fixed axis ranges.
    all_coords: list[np.ndarray] = []
    for frame_dict in midlines_3d:
        if not isinstance(frame_dict, dict):
            continue
        for spline in frame_dict.values():
            x, y, z = _eval_spline(spline)
            if x:
                all_coords.append(np.column_stack([x, y, z]))

    if all_coords:
        stacked = np.vstack(all_coords)
        finite_mask = np.isfinite(stacked).all(axis=1)
        stacked = stacked[finite_mask]
    if all_coords and len(stacked) > 0:
        mins = stacked.min(axis=0)
        maxs = stacked.max(axis=0)
        raw_range = maxs - mins
        pad = max(0.05 * raw_range.max(), 1e-6)
        spans = raw_range + 2 * pad
        max_span = max(spans.max(), 1e-6)
        aspect = {
            "x": max(float(spans[0] / max_span), 0.01),
            "y": max(float(spans[1] / max_span), 0.01),
            "z": max(float(spans[2] / max_span), 0.01),
        }
        axis_range = {
            "x": [float(mins[0] - pad), float(maxs[0] + pad)],
            "y": [float(mins[1] - pad), float(maxs[1] + pad)],
            "z": [float(mins[2] - pad), float(maxs[2] + pad)],
        }
    else:
        aspect = {"x": 1.0, "y": 1.0, "z": 1.0}
        axis_range = {"x": [-1, 1], "y": [-1, 1], "z": [-1, 1]}

    fig = go.Figure(data=initial_traces, frames=frames)
    fig.update_layout(
        title="AquaPose 3D Midline Animation",
        scene=dict(
            aspectmode="manual",
            aspectratio=aspect,
            xaxis=dict(title="X", range=axis_range["x"]),
            yaxis=dict(title="Y", range=axis_range["y"]),
            zaxis=dict(title="Z", range=axis_range["z"]),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=100, redraw=True),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(prefix="Frame: ", visible=True, xanchor="center"),
                steps=[
                    dict(
                        args=[
                            [str(i)],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                            ),
                        ],
                        label=str(i),
                        method="animate",
                    )
                    for i in range(len(midlines_3d))
                ],
            )
        ],
    )
    return fig


def _export_mp4(
    fig: go.Figure,
    midlines_3d: list[dict],
    fish_colors: dict[int, str],
    sorted_fish_ids: list[int],
    output_path: Path,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Render each frame as a static image and stitch into an MP4 with ffmpeg.

    Args:
        fig: Base Plotly figure (used for layout/scene settings).
        midlines_3d: Per-frame dicts mapping fish_id to spline objects.
        fish_colors: Mapping of fish_id to Plotly color string.
        sorted_fish_ids: Sorted list of all unique fish IDs.
        output_path: Destination MP4 path.
        fps: Frames per second for the output video.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Path to the written MP4 file.

    Raises:
        RuntimeError: If ffmpeg is not found or encoding fails.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (e.g. 'sudo apt install ffmpeg')."
        )

    n_frames = len(midlines_3d)
    tmpdir = tempfile.mkdtemp(prefix="aquapose_mp4_")
    try:
        sys.stderr.write(f"Rendering {n_frames} frames to PNG...\n")
        sys.stderr.flush()

        for frame_idx, frame_dict in enumerate(midlines_3d):
            if not isinstance(frame_dict, dict):
                frame_dict = {}

            # Update trace data in-place for this frame.
            for trace_idx, fid in enumerate(sorted_fish_ids):
                x, y, z = (
                    _eval_spline(frame_dict[fid]) if fid in frame_dict else ([], [], [])
                )
                fig.data[trace_idx].x = x
                fig.data[trace_idx].y = y
                fig.data[trace_idx].z = z

            png_path = Path(tmpdir) / f"frame_{frame_idx:06d}.png"
            fig.write_image(str(png_path), width=width, height=height, scale=1)

            if (frame_idx + 1) % 100 == 0 or frame_idx == n_frames - 1:
                sys.stderr.write(f"  {frame_idx + 1}/{n_frames}\n")
                sys.stderr.flush()

        sys.stderr.write("Encoding MP4 with ffmpeg...\n")
        sys.stderr.flush()

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(Path(tmpdir) / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

        sys.stderr.write(f"MP4 written to {output_path}\n")
        sys.stderr.flush()
        return output_path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def generate_animation(
    run_dir: Path,
    output_dir: Path | None = None,
    stride: int = 1,
    mp4: bool = False,
    fps: int = 30,
    *,
    unstitched: bool = False,
) -> Path:
    """Generate a 3D midline animation (HTML and/or MP4) from midlines HDF5.

    Prefers ``midlines_stitched.h5`` (post-stitching IDs) unless
    ``unstitched=True``, then falls back to ``midlines.h5``.

    Fish colors are deterministic: palette[fish_id % palette_length].

    Args:
        run_dir: Path to the pipeline run directory.
        output_dir: Directory for output. Defaults to ``{run_dir}/viz/``.
        stride: Keep every Nth frame. Default 1 (all frames). Use higher
            values (e.g. 3 or 5) to reduce HTML size for long videos.
        mp4: If True, export an MP4 video instead of interactive HTML.
        fps: Frames per second for MP4 output. Default 30.
        unstitched: If True, always use ``midlines.h5`` even when
            ``midlines_stitched.h5`` exists.

    Returns:
        Path to the written output file (HTML or MP4).

    Raises:
        RuntimeError: If no midlines HDF5 file is found.
    """
    out_dir = output_dir or run_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = resolve_h5_path(run_dir, unstitched=unstitched)
    if h5_path is None:
        raise RuntimeError(f"No midlines HDF5 found in {run_dir}")

    sys.stderr.write(f"Loading midlines from {h5_path.name}...\n")
    sys.stderr.flush()
    all_midlines_3d = load_midlines_from_h5(h5_path)

    if not all_midlines_3d:
        raise RuntimeError("No midlines_3d data found")

    if stride > 1:
        all_midlines_3d = all_midlines_3d[::stride]
        sys.stderr.write(
            f"Subsampled to {len(all_midlines_3d)} frames (stride={stride})\n"
        )
        sys.stderr.flush()

    # Collect all unique fish IDs.
    all_fish_ids: set[int] = set()
    for frame_dict in all_midlines_3d:
        if isinstance(frame_dict, dict):
            all_fish_ids.update(frame_dict.keys())
    sorted_fish_ids = sorted(all_fish_ids)

    # Assign deterministic colors per fish ID.
    palette = plotly.colors.qualitative.Plotly
    fish_colors: dict[int, str] = {
        fid: palette[fid % len(palette)] for fid in sorted_fish_ids
    }

    sys.stderr.write("Generating 3D animation...\n")
    sys.stderr.flush()

    fig = _build_figure(all_midlines_3d, fish_colors, sorted_fish_ids)

    if mp4:
        output_path = out_dir / "animation_3d.mp4"
        return _export_mp4(
            fig,
            all_midlines_3d,
            fish_colors,
            sorted_fish_ids,
            output_path,
            fps=fps,
        )

    output_path = out_dir / "animation_3d.html"
    fig.write_html(str(output_path), include_plotlyjs=True)
    logger.info("3D animation written to %s", output_path)
    return output_path


__all__ = ["generate_animation"]
