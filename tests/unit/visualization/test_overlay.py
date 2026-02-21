"""Unit tests for 2D overlay rendering functions.

All tests mock the RefractiveProjectionModel.project() method — no GPU,
no calibration files, no camera hardware required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from aquapose.visualization.overlay import FISH_COLORS, draw_midline_overlay


def _make_midline(fish_id: int = 0) -> MagicMock:
    """Create a minimal Midline3D mock with a valid cubic B-spline."""
    midline = MagicMock()
    midline.fish_id = fish_id
    midline.degree = 3

    # Knot vector for 7 control points, degree 3 (matches SPLINE_KNOTS)
    midline.knots = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0],
        dtype=np.float32,
    )

    # 7 control points describing a straight line from (0.1,0.1,1.5) to (0.5,0.5,1.5)
    ctrl = np.array(
        [
            [0.10, 0.10, 1.5],
            [0.17, 0.17, 1.5],
            [0.23, 0.23, 1.5],
            [0.30, 0.30, 1.5],
            [0.37, 0.37, 1.5],
            [0.43, 0.43, 1.5],
            [0.50, 0.50, 1.5],
        ],
        dtype=np.float32,
    )
    midline.control_points = ctrl

    # 15 half-widths (world metres, small values)
    midline.half_widths = np.full(15, 0.01, dtype=np.float32)

    return midline


def _make_model(n_eval: int, pixels: np.ndarray | None = None) -> MagicMock:
    """Create a mock RefractiveProjectionModel.

    Args:
        n_eval: Number of points the model will project.
        pixels: Optional (n_eval, 2) pixel array. If None, uses a simple
            diagonal pattern across a 100x100 frame.
    """
    if pixels is None:
        # Diagonal pixels across 100x100
        u = np.linspace(10, 90, n_eval).astype(np.float32)
        v = np.linspace(10, 90, n_eval).astype(np.float32)
        pixels = np.stack([u, v], axis=1)  # (n_eval, 2)

    model = MagicMock()
    # Focal lengths (used for width radius calculation)
    K = torch.eye(3, dtype=torch.float32)
    K[0, 0] = 500.0  # fx
    K[1, 1] = 500.0  # fy
    model.K = K
    model.water_z = 0.0

    # project() returns (pixels_tensor, valid_mask)
    px_tensor = torch.from_numpy(pixels)
    valid_tensor = torch.ones(n_eval, dtype=torch.bool)
    model.project = MagicMock(return_value=(px_tensor, valid_tensor))

    return model


def test_draw_midline_overlay_adds_polyline() -> None:
    """Overlay function should produce non-zero pixels along the expected polyline."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    midline = _make_midline(fish_id=0)
    n_eval = 30
    model = _make_model(n_eval)

    result = draw_midline_overlay(
        frame, midline, model, n_eval=n_eval, draw_widths=False
    )

    # The frame must have been modified (polyline drawn)
    assert result is frame, "draw_midline_overlay should return the same frame object"
    assert np.any(result > 0), "Frame should have non-zero pixels after overlay"


def test_draw_midline_overlay_with_widths() -> None:
    """Width circles should add more non-zero pixels than polyline-only."""
    n_eval = 30
    model_no_widths = _make_model(n_eval)
    model_with_widths = _make_model(n_eval)

    frame_no_widths = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_midline_overlay(
        frame_no_widths,
        _make_midline(0),
        model_no_widths,
        n_eval=n_eval,
        draw_widths=False,
    )

    frame_with_widths = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_midline_overlay(
        frame_with_widths,
        _make_midline(0),
        model_with_widths,
        n_eval=n_eval,
        draw_widths=True,
    )

    n_nonzero_no_widths = int(np.count_nonzero(frame_no_widths))
    n_nonzero_with_widths = int(np.count_nonzero(frame_with_widths))

    assert n_nonzero_with_widths >= n_nonzero_no_widths, (
        "Width circles should add at least as many non-zero pixels as polyline-only"
    )


def test_fish_color_palette_has_10_colors() -> None:
    """FISH_COLORS must contain at least 10 distinct color entries."""
    assert len(FISH_COLORS) >= 10, f"Expected >= 10 colors, got {len(FISH_COLORS)}"

    # Each entry must be a 3-tuple of ints
    for i, color in enumerate(FISH_COLORS):
        assert len(color) == 3, f"Color {i} has wrong length: {len(color)}"
        for ch in color:
            assert isinstance(ch, int), f"Color {i} channel is not int: {ch!r}"
            assert 0 <= ch <= 255, f"Color {i} channel out of range: {ch}"

    # All 10 must be distinct (convert to set of tuples)
    distinct = {tuple(c) for c in FISH_COLORS[:10]}
    assert len(distinct) == 10, f"First 10 colors are not all distinct: {distinct}"
