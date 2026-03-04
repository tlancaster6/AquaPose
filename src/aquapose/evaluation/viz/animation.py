"""3D midline animation visualization: interactive Plotly HTML across all chunks."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import plotly.colors
import plotly.graph_objects as go
import scipy.interpolate

from aquapose.evaluation.viz._loader import load_all_chunk_caches

logger = logging.getLogger(__name__)

# Number of points to evaluate along each B-spline.
_N_EVAL_POINTS: int = 50
_LINE_WIDTH: float = 4.0


def _eval_spline(
    spline: object,
    n_eval: int = _N_EVAL_POINTS,
) -> tuple[list[float], list[float], list[float]]:
    """Evaluate a B-spline at n_eval points and return (x, y, z) lists.

    Args:
        spline: Spline3D object with control_points, knots, and degree attributes.
        n_eval: Number of evaluation points along the spline.

    Returns:
        Tuple of (x_list, y_list, z_list) for Plotly traces.
    """
    control_points = getattr(spline, "control_points", None)
    if control_points is None:
        return [], [], []

    cp = np.asarray(control_points, dtype=np.float64)
    try:
        bspl = scipy.interpolate.BSpline(
            np.asarray(spline.knots, dtype=np.float64), cp, spline.degree
        )
        t_vals = np.linspace(0.0, 1.0, n_eval)
        pts = bspl(t_vals)  # (N, 3)
        return pts[:, 0].tolist(), pts[:, 1].tolist(), pts[:, 2].tolist()
    except Exception:
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
        mins = stacked.min(axis=0)
        maxs = stacked.max(axis=0)
        pad = 0.05 * (maxs - mins).max()
        spans = (maxs - mins) + 2 * pad
        max_span = spans.max()
        aspect = {
            "x": float(spans[0] / max_span),
            "y": float(spans[1] / max_span),
            "z": float(spans[2] / max_span),
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


def generate_animation(
    run_dir: Path,
    output_dir: Path | None = None,
) -> Path:
    """Generate an interactive 3D midline animation HTML across all chunks.

    Loads all chunk caches, merges midlines_3d data across chunks into a single
    flat list, and builds a Plotly animation with a unified scrubber timeline.
    Chunk boundaries are invisible to the viewer.

    Fish colors are deterministic: palette[fish_id % palette_length].

    Args:
        run_dir: Path to the pipeline run directory.
        output_dir: Directory for output. Defaults to ``{run_dir}/viz/``.

    Returns:
        Path to the written ``animation_3d.html`` file.

    Raises:
        RuntimeError: If no chunk caches are found in run_dir.
    """
    contexts = load_all_chunk_caches(run_dir)
    if not contexts:
        raise RuntimeError(f"No chunk caches found in {run_dir}")

    out_dir = output_dir or run_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "animation_3d.html"

    # Merge midlines_3d from all chunks.
    all_midlines_3d: list[dict] = []
    for ctx in contexts:
        mid = getattr(ctx, "midlines_3d", None)
        if mid is not None and isinstance(mid, list):
            all_midlines_3d.extend(mid)

    if not all_midlines_3d:
        raise RuntimeError("No midlines_3d data found in any chunk cache")

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True)

    logger.info("3D animation written to %s", output_path)
    return output_path


__all__ = ["generate_animation"]
