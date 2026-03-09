"""Tests for core pseudo-label generation functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from aquapose.core.types.reconstruction import Midline3D
from aquapose.training.pseudo_labels import (
    _passes_bounds_check,
    compute_confidence_score,
    detect_gaps,
    generate_fish_labels,
    generate_gap_fish_labels,
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
            lateral_ratio=0.18,
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
            lateral_ratio=0.18,
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
            lateral_ratio=0.18,
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
            lateral_ratio=0.18,
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
            "lateral_pad",
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
            lateral_ratio=0.18,
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
            lateral_ratio=0.18,
            max_camera_residual_px=15.0,
            camera_id="cam0",
        )

        assert result is not None
        parts = result["pose_line"].split()
        assert len(parts) == 5 + 3 * n_kpts


# ---------------------------------------------------------------------------
# Helpers for gap detection / gap label tests
# ---------------------------------------------------------------------------


def _make_mock_detection(x: float, y: float, w: float, h: float) -> MagicMock:
    """Create a mock Detection with given bbox (x, y, w, h)."""
    det = MagicMock()
    det.bbox = (x, y, w, h)
    return det


def _make_mock_tracklet(
    camera_id: str, frames: tuple, frame_status: tuple
) -> MagicMock:
    """Create a mock Tracklet2D."""
    t = MagicMock()
    t.camera_id = camera_id
    t.frames = frames
    t.frame_status = frame_status
    return t


class TestDetectGaps:
    """Tests for detect_gaps."""

    def test_returns_empty_when_per_camera_residuals_is_none(self) -> None:
        """No residuals -> empty list."""
        midline = _make_midline()
        midline.per_camera_residuals = None
        result = detect_gaps(midline, MagicMock(), {}, {}, {})
        assert result == []

    def test_returns_empty_when_contributing_below_min_cameras(self) -> None:
        """Fewer than min_cameras contributing -> empty list."""
        midline = _make_midline(per_camera_residuals={"cam0": 1.0, "cam1": 2.0})
        midline.n_cameras = 2
        result = detect_gaps(midline, MagicMock(), {}, {}, {}, min_cameras=3)
        assert result == []

    @patch("aquapose.training.pseudo_labels.ghost_point_lookup")
    def test_identifies_gap_cameras(self, mock_gpl: MagicMock) -> None:
        """Cameras visible via LUT but not in per_camera_residuals are gaps."""
        midline = _make_midline(
            n_cameras=3,
            per_camera_residuals={"cam0": 1.0, "cam1": 2.0, "cam2": 1.5},
        )
        # LUT says cam0, cam1, cam2, cam3 can see this fish
        mock_gpl.return_value = [
            [
                ("cam0", 100.0, 200.0),
                ("cam1", 300.0, 400.0),
                ("cam2", 500.0, 600.0),
                ("cam3", 700.0, 800.0),
            ]
        ]
        # Need a projection model for cam3 classification
        mock_proj = _make_mock_projection()
        proj_models = {"cam3": mock_proj}
        # cam3 has no detection -> should be "no-detection"
        result = detect_gaps(midline, MagicMock(), proj_models, {}, {}, min_cameras=3)
        assert len(result) == 1
        cam_ids = [c for c, _ in result]
        assert "cam3" in cam_ids

    @patch("aquapose.training.pseudo_labels.ghost_point_lookup")
    def test_never_flags_contributing_cameras(self, mock_gpl: MagicMock) -> None:
        """Contributing cameras should never appear as gaps."""
        midline = _make_midline(
            n_cameras=3,
            per_camera_residuals={"cam0": 1.0, "cam1": 2.0, "cam2": 1.5},
        )
        mock_gpl.return_value = [
            [("cam0", 100.0, 200.0), ("cam1", 300.0, 400.0), ("cam2", 500.0, 600.0)]
        ]
        result = detect_gaps(midline, MagicMock(), {}, {}, {}, min_cameras=3)
        assert result == []


class TestClassifyGap:
    """Tests for _classify_gap (via detect_gaps integration)."""

    @patch("aquapose.training.pseudo_labels.ghost_point_lookup")
    def test_no_detection_when_no_bbox_contains_centroid(
        self, mock_gpl: MagicMock
    ) -> None:
        """No detection bbox at projected centroid -> 'no-detection'."""
        midline = _make_midline(
            n_cameras=3,
            per_camera_residuals={"cam0": 1.0, "cam1": 2.0, "cam2": 1.5},
        )
        mock_gpl.return_value = [
            [("cam0", 0, 0), ("cam1", 0, 0), ("cam2", 0, 0), ("cam3", 0, 0)]
        ]
        mock_proj = _make_mock_projection(pixel_x=500.0, pixel_y=300.0)
        proj_models = {"cam3": mock_proj}
        # Detection at (0,0,100,100) does NOT contain (500, 300)
        frame_dets = {"cam3": [_make_mock_detection(0, 0, 100, 100)]}
        result = detect_gaps(midline, MagicMock(), proj_models, frame_dets, {})
        assert ("cam3", "no-detection") in result

    @patch("aquapose.training.pseudo_labels.ghost_point_lookup")
    def test_no_tracklet_when_detection_but_no_tracklet(
        self, mock_gpl: MagicMock
    ) -> None:
        """Detection exists but no tracklet -> 'no-tracklet'."""
        midline = _make_midline(
            n_cameras=3,
            per_camera_residuals={"cam0": 1.0, "cam1": 2.0, "cam2": 1.5},
        )
        midline.frame_index = 5
        mock_gpl.return_value = [
            [("cam0", 0, 0), ("cam1", 0, 0), ("cam2", 0, 0), ("cam3", 0, 0)]
        ]
        # Projection returns (500, 300)
        mock_proj = _make_mock_projection(pixel_x=500.0, pixel_y=300.0)
        proj_models = {"cam3": mock_proj}
        # Detection bbox (400, 200, 200, 200) contains (500, 300)
        frame_dets = {"cam3": [_make_mock_detection(400, 200, 200, 200)]}
        # No tracklets for cam3
        frame_tracks = {}
        result = detect_gaps(
            midline, MagicMock(), proj_models, frame_dets, frame_tracks
        )
        assert ("cam3", "no-tracklet") in result

    @patch("aquapose.training.pseudo_labels.ghost_point_lookup")
    def test_failed_midline_when_detection_and_tracklet_exist(
        self, mock_gpl: MagicMock
    ) -> None:
        """Detection + tracklet exist but camera didn't contribute -> 'failed-midline'."""
        midline = _make_midline(
            n_cameras=3,
            per_camera_residuals={"cam0": 1.0, "cam1": 2.0, "cam2": 1.5},
        )
        midline.frame_index = 5
        mock_gpl.return_value = [
            [("cam0", 0, 0), ("cam1", 0, 0), ("cam2", 0, 0), ("cam3", 0, 0)]
        ]
        mock_proj = _make_mock_projection(pixel_x=500.0, pixel_y=300.0)
        proj_models = {"cam3": mock_proj}
        frame_dets = {"cam3": [_make_mock_detection(400, 200, 200, 200)]}
        # Tracklet active and detected at frame 5
        tracklet = _make_mock_tracklet(
            "cam3", (3, 4, 5, 6), ("detected", "detected", "detected", "detected")
        )
        frame_tracks = {"cam3": [tracklet]}
        result = detect_gaps(
            midline, MagicMock(), proj_models, frame_dets, frame_tracks
        )
        assert ("cam3", "failed-midline") in result

    @patch("aquapose.training.pseudo_labels.ghost_point_lookup")
    def test_no_detection_when_projection_invalid(self, mock_gpl: MagicMock) -> None:
        """Invalid projection -> 'no-detection'."""
        midline = _make_midline(
            n_cameras=3,
            per_camera_residuals={"cam0": 1.0, "cam1": 2.0, "cam2": 1.5},
        )
        mock_gpl.return_value = [
            [("cam0", 0, 0), ("cam1", 0, 0), ("cam2", 0, 0), ("cam3", 0, 0)]
        ]
        # Make projection return invalid
        mock_proj = MagicMock()
        mock_proj.project.return_value = (
            torch.zeros(1, 2),
            torch.tensor([False]),
        )
        proj_models = {"cam3": mock_proj}
        result = detect_gaps(midline, MagicMock(), proj_models, {}, {})
        assert ("cam3", "no-detection") in result


class TestGenerateGapFishLabels:
    """Tests for generate_gap_fish_labels."""

    def test_returns_valid_dict_for_good_projection(self) -> None:
        """Valid projection returns dict with obb_line, pose_line, confidence etc."""
        midline = _make_midline(
            per_camera_residuals={"cam0": 2.0, "cam1": 3.0},
        )
        proj = _make_mock_projection()
        result = generate_gap_fish_labels(
            midline,
            proj,
            img_w=1920,
            img_h=1080,
            keypoint_t_values=[0.0, 0.25, 0.5, 0.75, 1.0],
            lateral_ratio=0.18,
        )
        assert result is not None
        expected_keys = {
            "obb_line",
            "pose_line",
            "confidence",
            "raw_metrics",
            "keypoints_2d",
            "visibility",
            "lateral_pad",
        }
        assert set(result.keys()) == expected_keys
        assert 0.0 <= result["confidence"] <= 1.0

    def test_returns_none_when_fewer_than_2_visible(self) -> None:
        """Fewer than 2 visible keypoints -> None."""
        midline = _make_midline()
        # Make projection return only 1 valid point
        mock_proj = MagicMock()

        def mock_project(pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            n = pts.shape[0]
            pixels = torch.zeros(n, 2)
            valid = torch.zeros(n, dtype=torch.bool)
            valid[0] = True  # Only 1 valid
            return pixels, valid

        mock_proj.project.side_effect = mock_project
        result = generate_gap_fish_labels(
            midline,
            mock_proj,
            img_w=1920,
            img_h=1080,
            keypoint_t_values=[0.0, 0.5, 1.0],
            lateral_ratio=0.18,
        )
        assert result is None

    def test_returns_none_when_bounds_check_fails(self) -> None:
        """Bounds check failure -> None."""
        midline = _make_midline()
        # Make projection return pixels far outside image bounds
        mock_proj = MagicMock()

        def mock_project(pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            n = pts.shape[0]
            pixels = torch.full((n, 2), -5000.0)  # Way out of bounds
            valid = torch.ones(n, dtype=torch.bool)
            return pixels, valid

        mock_proj.project.side_effect = mock_project
        result = generate_gap_fish_labels(
            midline,
            mock_proj,
            img_w=1920,
            img_h=1080,
            keypoint_t_values=[0.0, 0.5, 1.0],
            lateral_ratio=0.18,
        )
        assert result is None


class TestPassesBoundsCheck:
    """Tests for _passes_bounds_check."""

    def test_returns_false_when_fewer_than_2_visible(self) -> None:
        """Fewer than 2 visible keypoints -> False."""
        kpts = np.array([[100, 200], [300, 400], [500, 600]], dtype=np.float64)
        vis = np.array([True, False, False])
        assert _passes_bounds_check(kpts, vis, 1920, 1080) is False

    def test_returns_false_when_most_out_of_bounds(self) -> None:
        """Less than 50% visible keypoints in bounds -> False."""
        kpts = np.array(
            [
                [-100, -200],  # out
                [-300, -400],  # out
                [500, 600],  # in
            ],
            dtype=np.float64,
        )
        vis = np.array([True, True, True])
        assert _passes_bounds_check(kpts, vis, 1920, 1080) is False

    def test_returns_true_when_most_in_bounds(self) -> None:
        """At least 50% visible keypoints in bounds -> True."""
        kpts = np.array(
            [
                [100, 200],  # in
                [300, 400],  # in
                [-500, -600],  # out
            ],
            dtype=np.float64,
        )
        vis = np.array([True, True, True])
        assert _passes_bounds_check(kpts, vis, 1920, 1080) is True

    def test_boundary_exact_50_percent(self) -> None:
        """Exactly 50% in bounds -> True (>=50%)."""
        kpts = np.array(
            [
                [100, 200],  # in
                [-100, -200],  # out
            ],
            dtype=np.float64,
        )
        vis = np.array([True, True])
        assert _passes_bounds_check(kpts, vis, 1920, 1080) is True


class TestPartialOobFish:
    """Tests for partial out-of-frame fish handling."""

    def test_oob_keypoints_marked_invisible(self) -> None:
        """Fish with 3/6 keypoints OOB has only 3 visible in result."""
        midline = _make_midline(per_camera_residuals={"cam0": 2.0})
        mock_proj = MagicMock()

        def mock_project(pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            n = pts.shape[0]
            pixels = torch.zeros(n, 2)
            # 3 in-frame, 3 out-of-frame
            pixels[0] = torch.tensor([100.0, 200.0])
            pixels[1] = torch.tensor([200.0, 300.0])
            pixels[2] = torch.tensor([300.0, 400.0])
            pixels[3] = torch.tensor([-50.0, 500.0])  # OOB left
            pixels[4] = torch.tensor([2000.0, 500.0])  # OOB right
            pixels[5] = torch.tensor([500.0, -100.0])  # OOB top
            valid = torch.ones(n, dtype=torch.bool)
            return pixels, valid

        mock_proj.project.side_effect = mock_project

        result = generate_fish_labels(
            midline,
            mock_proj,
            img_w=1920,
            img_h=1080,
            keypoint_t_values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            lateral_ratio=0.18,
            max_camera_residual_px=15.0,
            camera_id="cam0",
        )

        assert result is not None
        # Only 3 keypoints should be visible
        assert result["visibility"].sum() == 3
        assert result["visibility"][0] is np.True_
        assert result["visibility"][1] is np.True_
        assert result["visibility"][2] is np.True_
        assert result["visibility"][3] is np.False_
        assert result["visibility"][4] is np.False_
        assert result["visibility"][5] is np.False_

    def test_obb_annotation_normalized_to_unit_range(self) -> None:
        """OBB annotation values are clipped to [0,1] even if corners exceed image."""
        midline = _make_midline(per_camera_residuals={"cam0": 2.0})
        mock_proj = MagicMock()

        def mock_project(pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            n = pts.shape[0]
            pixels = torch.zeros(n, 2)
            # Fish near left edge — some keypoints in, some out
            pixels[0] = torch.tensor([5.0, 500.0])
            pixels[1] = torch.tensor([50.0, 510.0])
            pixels[2] = torch.tensor([100.0, 520.0])
            pixels[3] = torch.tensor([-30.0, 530.0])  # OOB
            valid = torch.ones(n, dtype=torch.bool)
            return pixels, valid

        mock_proj.project.side_effect = mock_project

        result = generate_fish_labels(
            midline,
            mock_proj,
            img_w=1920,
            img_h=1080,
            keypoint_t_values=[0.0, 0.33, 0.67, 1.0],
            lateral_ratio=0.18,
            max_camera_residual_px=15.0,
            camera_id="cam0",
        )

        assert result is not None
        # OBB annotation has normalized values (may extend past [0,1] for
        # border boxes to preserve rectangular shape)
        obb_tokens = result["obb_line"].split()
        assert len(obb_tokens) == 9

    def test_returns_none_when_all_oob(self) -> None:
        """Fish entirely out of frame returns None."""
        midline = _make_midline(per_camera_residuals={"cam0": 2.0})
        mock_proj = MagicMock()

        def mock_project(pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            n = pts.shape[0]
            pixels = torch.full((n, 2), -100.0)  # All OOB
            valid = torch.ones(n, dtype=torch.bool)
            return pixels, valid

        mock_proj.project.side_effect = mock_project

        result = generate_fish_labels(
            midline,
            mock_proj,
            img_w=1920,
            img_h=1080,
            keypoint_t_values=[0.0, 0.5, 1.0],
            lateral_ratio=0.18,
            max_camera_residual_px=15.0,
            camera_id="cam0",
        )

        assert result is None

    def test_gap_fish_oob_keypoints_marked_invisible(self) -> None:
        """Gap fish with OOB keypoints has them marked invisible."""
        midline = _make_midline(per_camera_residuals={"cam0": 2.0})
        mock_proj = MagicMock()

        def mock_project(pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            n = pts.shape[0]
            pixels = torch.zeros(n, 2)
            pixels[0] = torch.tensor([100.0, 200.0])
            pixels[1] = torch.tensor([200.0, 300.0])
            pixels[2] = torch.tensor([300.0, 400.0])
            pixels[3] = torch.tensor([-50.0, 500.0])  # OOB
            valid = torch.ones(n, dtype=torch.bool)
            return pixels, valid

        mock_proj.project.side_effect = mock_project

        result = generate_gap_fish_labels(
            midline,
            mock_proj,
            img_w=1920,
            img_h=1080,
            keypoint_t_values=[0.0, 0.33, 0.67, 1.0],
            lateral_ratio=0.18,
        )

        assert result is not None
        assert result["visibility"].sum() == 3
        assert result["visibility"][3] is np.False_
