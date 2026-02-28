"""Unit tests for Overlay2DObserver."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from aquapose.engine.observers import Observer
from aquapose.engine.overlay_observer import Overlay2DObserver


def test_overlay_observer_satisfies_protocol(tmp_path: Path) -> None:
    """Overlay2DObserver satisfies the Observer protocol via isinstance check."""
    observer = Overlay2DObserver(
        output_dir=tmp_path,
        video_dir=tmp_path,
        calibration_path=tmp_path / "cal.json",
    )
    assert isinstance(observer, Observer)


def test_build_mosaic() -> None:
    """_build_mosaic tiles 4 frames into a 2x2 grid."""
    frames = {
        "cam0": np.full((100, 100, 3), 10, dtype=np.uint8),
        "cam1": np.full((100, 100, 3), 20, dtype=np.uint8),
        "cam2": np.full((100, 100, 3), 30, dtype=np.uint8),
        "cam3": np.full((100, 100, 3), 40, dtype=np.uint8),
    }
    camera_ids = ["cam0", "cam1", "cam2", "cam3"]

    mosaic = Overlay2DObserver._build_mosaic(frames, camera_ids)
    assert mosaic.shape == (200, 200, 3)
    # Top-left cell should be cam0's value.
    assert mosaic[0, 0, 0] == 10
    # Top-right cell should be cam1's value.
    assert mosaic[0, 100, 0] == 20
    # Bottom-left cell should be cam2's value.
    assert mosaic[100, 0, 0] == 30
    # Bottom-right cell should be cam3's value.
    assert mosaic[100, 100, 0] == 40


def test_draw_midline() -> None:
    """_draw_midline modifies frame pixels along the polyline."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    points = np.array([[10, 10], [50, 50], [90, 90]], dtype=np.float32)
    color = (0, 255, 0)

    Overlay2DObserver._draw_midline(frame, points, color, thickness=2)

    # Frame should no longer be all zeros.
    assert frame.sum() > 0


def test_reproject_returns_correct_shape() -> None:
    """_reproject_3d_midline returns (N, 2) array with mock model."""
    # Create mock spline with control_points, knots, and degree.
    rng = np.random.RandomState(42)
    spline = SimpleNamespace(
        control_points=rng.randn(7, 3).astype(np.float32),
        knots=np.array(
            [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float64,
        ),
        degree=3,
    )

    # Create mock model that returns 2D projections.
    import torch

    def mock_project(pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = pts.shape[0]
        return torch.rand(n, 2), torch.ones(n, dtype=torch.bool)

    model = MagicMock()
    model.project = mock_project

    result = Overlay2DObserver._reproject_3d_midline(spline, "cam0", model)
    assert result is not None
    assert result.shape == (50, 2)
    assert result.dtype == np.float32


def test_mosaic_dims() -> None:
    """_mosaic_dims computes correct grid dimensions."""
    # 12 cameras in a 4x3 grid.
    h, w = Overlay2DObserver._mosaic_dims(12, 100, 80)
    assert w == 400  # 4 cols * 100
    assert h == 240  # 3 rows * 80

    # 4 cameras in a 2x2 grid.
    h, w = Overlay2DObserver._mosaic_dims(4, 200, 150)
    assert w == 400  # 2 cols * 200
    assert h == 300  # 2 rows * 150


def test_build_mosaic_uneven_count() -> None:
    """_build_mosaic handles non-square camera counts."""
    frames = {f"cam{i}": np.full((50, 50, 3), i * 10, dtype=np.uint8) for i in range(3)}
    camera_ids = ["cam0", "cam1", "cam2"]

    mosaic = Overlay2DObserver._build_mosaic(frames, camera_ids)
    # 3 cameras -> 2x2 grid with one empty cell.
    assert mosaic.shape == (100, 100, 3)
