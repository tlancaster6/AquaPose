"""Unit tests for Animation3DObserver."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aquapose.engine.animation_observer import Animation3DObserver
from aquapose.engine.events import PipelineComplete
from aquapose.engine.observers import Observer


def _make_spline(seed: int = 0) -> SimpleNamespace:
    """Create a mock Spline3D with control_points, knots, and degree."""
    rng = np.random.RandomState(seed)
    return SimpleNamespace(
        control_points=rng.randn(7, 3).astype(np.float32),
        knots=np.array(
            [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float64,
        ),
        degree=3,
    )


def test_animation_observer_satisfies_protocol(tmp_path: Path) -> None:
    """Animation3DObserver satisfies the Observer protocol via isinstance check."""
    observer = Animation3DObserver(output_dir=tmp_path)
    assert isinstance(observer, Observer)


def test_build_figure_creates_traces(tmp_path: Path) -> None:
    """_build_figure creates traces per fish and frames per pipeline frame."""
    observer = Animation3DObserver(output_dir=tmp_path)

    midlines_3d = [
        {0: _make_spline(0), 1: _make_spline(1)},
        {0: _make_spline(2), 1: _make_spline(3)},
        {0: _make_spline(4), 1: _make_spline(5)},
    ]

    fig = observer._build_figure(midlines_3d)
    # 2 traces (one per fish).
    assert len(fig.data) == 2
    # 3 animation frames.
    assert len(fig.frames) == 3


def test_build_figure_handles_missing_fish(tmp_path: Path) -> None:
    """When a fish is missing in a frame, its trace has empty coordinates."""
    observer = Animation3DObserver(output_dir=tmp_path)

    midlines_3d = [
        {0: _make_spline(0), 1: _make_spline(1)},  # Both fish.
        {0: _make_spline(2)},  # Only fish 0.
    ]

    fig = observer._build_figure(midlines_3d)
    # Frame 1 should have fish 1 trace with empty data.
    frame_1_data = fig.frames[1].data
    fish_1_trace = frame_1_data[1]  # Second trace (fish 1).
    assert len(fish_1_trace.x) == 0
    assert len(fish_1_trace.y) == 0
    assert len(fish_1_trace.z) == 0


def test_html_output_written(tmp_path: Path) -> None:
    """PipelineComplete with context produces animation_3d.html file."""
    observer = Animation3DObserver(output_dir=tmp_path)

    context = SimpleNamespace(
        midlines_3d=[
            {0: _make_spline(0)},
            {0: _make_spline(1)},
        ]
    )

    observer.on_event(
        PipelineComplete(
            run_id="anim_test",
            elapsed_seconds=1.0,
            context=context,
        )
    )

    html_path = tmp_path / "animation_3d.html"
    assert html_path.exists()
    assert html_path.stat().st_size > 0


def test_skips_if_no_context(tmp_path: Path) -> None:
    """No file written when PipelineComplete has no context."""
    observer = Animation3DObserver(output_dir=tmp_path)
    observer.on_event(PipelineComplete(run_id="no_ctx", elapsed_seconds=1.0))

    assert not (tmp_path / "animation_3d.html").exists()


def test_figure_has_animation_controls(tmp_path: Path) -> None:
    """Built figure has updatemenus (play/pause) and sliders."""
    observer = Animation3DObserver(output_dir=tmp_path)

    midlines_3d = [
        {0: _make_spline(0)},
        {0: _make_spline(1)},
    ]

    fig = observer._build_figure(midlines_3d)
    assert fig.layout.updatemenus is not None
    assert len(fig.layout.updatemenus) > 0
    assert fig.layout.sliders is not None
    assert len(fig.layout.sliders) > 0
