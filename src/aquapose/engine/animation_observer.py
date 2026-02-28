"""3D midline animation observer using Plotly for interactive HTML visualization."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import plotly.colors
import plotly.graph_objects as go
import scipy.interpolate

from aquapose.engine.events import Event, PipelineComplete

logger = logging.getLogger(__name__)


class Animation3DObserver:
    """Generates an interactive HTML viewer with animated 3D fish midlines.

    On PipelineComplete, builds a Plotly figure with ``go.Scatter3d`` traces
    (one per fish) and ``go.Frame`` animation objects (one per pipeline frame).
    The self-contained HTML file includes play/pause and frame scrubber controls.

    Args:
        output_dir: Directory where ``animation_3d.html`` will be written.
        n_eval_points: Number of points to evaluate along each spline for smooth curves.
        line_width: Plotly Scatter3d line width.

    Example::

        observer = Animation3DObserver(output_dir="/tmp/output")
        pipeline = PosePipeline(stages=stages, config=config, observers=[observer])
        context = pipeline.run()
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        n_eval_points: int = 50,
        line_width: float = 4.0,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._n_eval_points = n_eval_points
        self._line_width = line_width

    def on_event(self, event: Event) -> None:
        """Receive a dispatched event and trigger animation generation.

        Args:
            event: The event instance from the pipeline event bus.
        """
        if not isinstance(event, PipelineComplete):
            return

        context = event.context
        if context is None:
            return

        midlines_3d = getattr(context, "midlines_3d", None)
        if midlines_3d is None or not isinstance(midlines_3d, list):
            return

        try:
            sys.stderr.write("Generating 3D animation...\n")
            sys.stderr.flush()
            fig = self._build_figure(midlines_3d)
            self._write_html(fig, self._output_dir / "animation_3d.html")
            logger.info(
                "Wrote 3D animation to %s", self._output_dir / "animation_3d.html"
            )
        except Exception:
            logger.warning("Animation generation failed", exc_info=True)

    def _build_figure(self, midlines_3d: list[dict]) -> go.Figure:
        """Build a Plotly figure with animated 3D midline traces.

        Args:
            midlines_3d: Per-frame dicts mapping fish_id to Spline3D objects.

        Returns:
            Plotly Figure with animation frames and controls.
        """
        # Collect all unique fish IDs.
        all_fish_ids: set[int] = set()
        for frame_dict in midlines_3d:
            if isinstance(frame_dict, dict):
                all_fish_ids.update(frame_dict.keys())
        sorted_fish_ids = sorted(all_fish_ids)

        # Assign colors per fish.
        palette = plotly.colors.qualitative.Plotly
        fish_colors: dict[int, str] = {}
        for i, fid in enumerate(sorted_fish_ids):
            fish_colors[fid] = palette[i % len(palette)]

        # Build initial traces (frame 0).
        initial_traces: list[go.Scatter3d] = []
        frame_0 = midlines_3d[0] if midlines_3d else {}
        if not isinstance(frame_0, dict):
            frame_0 = {}

        for fid in sorted_fish_ids:
            if fid in frame_0:
                x, y, z = self._eval_spline(frame_0[fid])
            else:
                x, y, z = [], [], []
            initial_traces.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    name=f"Fish {fid}",
                    line=dict(
                        color=fish_colors[fid],
                        width=self._line_width,
                    ),
                )
            )

        # Build animation frames.
        frames: list[go.Frame] = []
        for frame_idx, frame_dict in enumerate(midlines_3d):
            if not isinstance(frame_dict, dict):
                frame_dict = {}
            frame_traces: list[go.Scatter3d] = []
            for fid in sorted_fish_ids:
                if fid in frame_dict:
                    x, y, z = self._eval_spline(frame_dict[fid])
                else:
                    x, y, z = [], [], []
                frame_traces.append(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="lines",
                        name=f"Fish {fid}",
                        line=dict(
                            color=fish_colors[fid],
                            width=self._line_width,
                        ),
                    )
                )
            frames.append(go.Frame(data=frame_traces, name=str(frame_idx)))

        # Compute global bounding box for fixed axis ranges.
        all_coords: list[np.ndarray] = []
        for frame_dict in midlines_3d:
            if not isinstance(frame_dict, dict):
                continue
            for spline in frame_dict.values():
                x, y, z = self._eval_spline(spline)
                if x:
                    all_coords.append(np.column_stack([x, y, z]))

        if all_coords:
            stacked = np.vstack(all_coords)
            mins = stacked.min(axis=0)
            maxs = stacked.max(axis=0)
            pad = 0.05 * (maxs - mins).max()  # 5% padding
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

        # Assemble figure.
        fig = go.Figure(data=initial_traces, frames=frames)

        # Layout with animation controls.
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

    def _eval_spline(
        self, spline: object
    ) -> tuple[list[float], list[float], list[float]]:
        """Evaluate a B-spline at N points and return x, y, z lists.

        Args:
            spline: Object with ``control_points`` array (7, 3).

        Returns:
            Tuple of (x_list, y_list, z_list) for Plotly traces.
        """
        control_points = getattr(spline, "control_points", None)
        if control_points is None:
            return [], [], []

        cp = np.asarray(control_points, dtype=np.float64)
        knots = spline.knots
        degree = spline.degree
        try:
            bspl = scipy.interpolate.BSpline(
                np.asarray(knots, dtype=np.float64), cp, degree
            )
            t_vals = np.linspace(0.0, 1.0, self._n_eval_points)
            pts = bspl(t_vals)  # shape (N, 3)
            return pts[:, 0].tolist(), pts[:, 1].tolist(), pts[:, 2].tolist()
        except Exception:
            return [], [], []

    @staticmethod
    def _write_html(fig: go.Figure, path: Path) -> None:
        """Write the Plotly figure as a self-contained HTML file.

        Args:
            fig: Plotly figure to write.
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs=True)
