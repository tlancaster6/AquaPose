"""Tests for core pseudo-label generation functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from aquapose.core.types.reconstruction import Midline3D
from aquapose.training.pseudo_labels import (
    compute_confidence_score,
    generate_fish_labels,
    reproject_spline_keypoints,
)


def _make_midline(
    *,
    n_cameras: int = 5,
    mean_residual: float = 2.0,
    per_camera_residuals: dict[str, float] | None = None,
) -> Midline3D:
    """Create a synthetic Midline3D for testing.

    Builds a simple straight-line B-spline along the X axis from 0 to 0.1 m,
    at z=0.05 (underwater).
    """
    # 7 control points for a degree-3 B-spline
    # Simple line along x-axis at y=0, z=0.05
    control_points = np.zeros((7, 3), dtype=np.float32)
    control_points[:, 0] = np.linspace(0.0, 0.1, 7)
    control_points[:, 2] = 0.05  # underwater

    # Knot vector for degree 3, 7 control points: length = 7 + 3 + 1 = 11
    knots = np.array([0, 0, 0, 0, 0.333, 0.5, 0.667, 1, 1, 1, 1], dtype=np.float32)

    if per_camera_residuals is None:
        per_camera_residuals = {f"cam{i}": 3.0 for i in range(n_cameras)}

    return Midline3D(
        fish_id=1,
        frame_index=0,
        control_points=control_points,
        knots=knots,
        degree=3,
        arc_length=0.1,
        half_widths=np.full(13, 0.005, dtype=np.float32),
        n_cameras=n_cameras,
        mean_residual=mean_residual,
        max_residual=mean_residual * 2,
        is_low_confidence=False,
        per_camera_residuals=per_camera_residuals,
    )


def _make_mock_projection(pixel_x: float = 500.0, pixel_y: float = 300.0) -> MagicMock:
    """Create a mock RefractiveProjectionModel that returns fixed pixels."""
    mock = MagicMock()

    def mock_project(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = points.shape[0]
        # Return linearly spaced pixels to simulate a fish shape
        pixels = torch.zeros(n, 2)
        pixels[:, 0] = torch.linspace(pixel_x - 50, pixel_x + 50, n)
        pixels[:, 1] = torch.full((n,), pixel_y)
        valid = torch.ones(n, dtype=torch.bool)
        return pixels, valid

    mock.project.side_effect = mock_project
    return mock


class TestReprojectSplineKeypoints:
    """Tests for reproject_spline_keypoints."""

    def test_output_shape(self) -> None:
        """Returns (N, 2) pixels and (N,) visibility for N keypoint t-values."""
        midline = _make_midline()
        proj = _make_mock_projection()
        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        pixels, valid = reproject_spline_keypoints(midline, t_values, proj)

        assert pixels.shape == (5, 2)
        assert valid.shape == (5,)
        assert pixels.dtype == np.float64
        assert valid.dtype == bool

    def test_all_valid_for_underwater_points(self) -> None:
        """All points should be valid when projection succeeds."""
        midline = _make_midline()
        proj = _make_mock_projection()
        t_values = [0.0, 0.5, 1.0]

        _, valid = reproject_spline_keypoints(midline, t_values, proj)

        assert valid.all()

    def test_calls_project_with_correct_count(self) -> None:
        """projection_model.project is called with N 3D points."""
        midline = _make_midline()
        proj = _make_mock_projection()
        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        reproject_spline_keypoints(midline, t_values, proj)

        # project should have been called once with (5, 3) tensor
        call_args = proj.project.call_args
        pts_arg = call_args[0][0]
        assert pts_arg.shape == (5, 3)


class TestComputeConfidenceScore:
    """Tests for compute_confidence_score."""

    def test_high_quality_reconstruction(self) -> None:
        """Low residual, many cameras, low variance -> high score."""
        score, metrics = compute_confidence_score(
            mean_residual=1.0,
            n_cameras=8,
            per_camera_residuals={"c1": 1.0, "c2": 1.1, "c3": 0.9},
        )
        assert 0.8 <= score <= 1.0
        assert metrics["residual_score"] == pytest.approx(0.9)
        assert metrics["camera_score"] == pytest.approx(1.0)

    def test_low_quality_reconstruction(self) -> None:
        """High residual, few cameras -> low score."""
        score, metrics = compute_confidence_score(
            mean_residual=20.0,
            n_cameras=2,
            per_camera_residuals={"c1": 15.0, "c2": 25.0},
        )
        assert score < 0.3
        assert metrics["residual_score"] == 0.0
        assert metrics["camera_score"] == 0.0

    def test_score_in_unit_range(self) -> None:
        """Score is always in [0, 1]."""
        score, _ = compute_confidence_score(0.0, 12, {"c1": 0.0})
        assert 0.0 <= score <= 1.0

        score, _ = compute_confidence_score(100.0, 1, None)
        assert 0.0 <= score <= 1.0

    def test_none_per_camera_residuals(self) -> None:
        """None per_camera_residuals yields zero variance."""
        _score, metrics = compute_confidence_score(5.0, 4, None)
        assert metrics["per_camera_variance"] == 0.0

    def test_raw_metrics_keys(self) -> None:
        """Raw metrics dict has all expected keys."""
        _, metrics = compute_confidence_score(2.0, 5, {"c1": 2.0, "c2": 3.0})
        expected_keys = {
            "mean_residual",
            "n_cameras",
            "per_camera_variance",
            "residual_score",
            "camera_score",
            "variance_score",
        }
        assert set(metrics.keys()) == expected_keys


class TestGenerateFishLabels:
    """Tests for generate_fish_labels."""

    def test_returns_none_when_camera_not_in_residuals(self) -> None:
        """Camera not in per_camera_residuals -> None."""
        midline = _make_midline(per_camera_residuals={"cam0": 2.0, "cam1": 3.0})
        proj = _make_mock_projection()

        result = generate_fish_labels(
            midline,
            proj,
            img_w=1920,
            img_h=1080,
            keypoint_t_values=[0.0, 0.5, 1.0],
            lateral_pad=40.0,
            max_camera_residual_px=15.0,
            camera_id="cam99",
        )
        assert result is None

    def test_returns_none_when_residual_exceeds_threshold(self) -> None:
        """Camera residual above threshold -> None."""
        midline = _make_midline(per_camera_residuals={"cam0": 20.0})
        proj = _make_mock_projection()

        result = generate_fish_labels(
            midline,
            proj,
            img_w=1920,
            img_h=1080,
            keypoint_t_values=[0.0, 0.5, 1.0],
            lateral_pad=40.0,
            max_camera_residual_px=15.0,
            camera_id="cam0",
        )
        assert result is None

    def test_returns_none_when_per_camera_residuals_is_none(self) -> None:
        """None per_camera_residuals -> None."""
        midline = _make_midline()
        midline.per_camera_residuals = None
        proj = _make_mock_projection()

        result = generate_fish_labels(
            midline,
            proj,
            img_w=1920,
            img_h=1080,
            keypoint_t_values=[0.0, 0.5, 1.0],
            lateral_pad=40.0,
            max_camera_residual_px=15.0,
            camera_id="cam0",
        )
        assert result is None

    def test_valid_result_has_expected_keys(self) -> None:
        """Good case returns dict with all expected keys."""
        midline = _make_midline(per_camera_residuals={"cam0": 2.0, "cam1": 3.0})
        proj = _make_mock_projection()

        result = generate_fish_labels(
            midline,
            proj,
            img_w=1920,
            img_h=1080,
            keypoint_t_values=[0.0, 0.25, 0.5, 0.75, 1.0],
            lateral_pad=40.0,
            max_camera_residual_px=15.0,
            camera_id="cam0",
        )

        assert result is not None
        expected_keys = {
            "obb_line",
            "pose_line",
            "confidence",
            "raw_metrics",
            "keypoints_2d",
            "visibility",
        }
        assert set(result.keys()) == expected_keys
        assert isinstance(result["obb_line"], str)
        assert isinstance(result["pose_line"], str)
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["raw_metrics"], dict)
        assert result["keypoints_2d"].shape == (5, 2)
        assert result["visibility"].shape == (5,)

    def test_obb_line_format(self) -> None:
        """OBB line has 9 space-separated values (cls + 4 corners x 2)."""
        midline = _make_midline(per_camera_residuals={"cam0": 2.0})
        proj = _make_mock_projection()

        result = generate_fish_labels(
            midline,
            proj,
            img_w=1920,
            img_h=1080,
            keypoint_t_values=[0.0, 0.5, 1.0],
            lateral_pad=40.0,
            max_camera_residual_px=15.0,
            camera_id="cam0",
        )

        assert result is not None
        parts = result["obb_line"].split()
        assert len(parts) == 9

    def test_pose_line_format(self) -> None:
        """Pose line has 5 + 3*N_keypoints space-separated values."""
        midline = _make_midline(per_camera_residuals={"cam0": 2.0})
        proj = _make_mock_projection()
        n_kpts = 5
        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        result = generate_fish_labels(
            midline,
            proj,
            img_w=1920,
            img_h=1080,
            keypoint_t_values=t_values,
            lateral_pad=40.0,
            max_camera_residual_px=15.0,
            camera_id="cam0",
        )

        assert result is not None
        parts = result["pose_line"].split()
        assert len(parts) == 5 + 3 * n_kpts
