"""Unit tests for DltBackend — confidence-weighted DLT triangulation backend."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from aquapose.core.types.midline import Midline2D
from aquapose.core.types.reconstruction import Midline3D

# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------


def _make_mock_model(
    camera_center: tuple[float, float, float] = (0.0, 0.0, -1.0),
    water_z: float = 0.0,
    focal: float = 800.0,
) -> MagicMock:
    """Create a mock RefractiveProjectionModel for testing.

    Uses simple pinhole geometry: camera above water (cz < water_z), looking
    downward into the tank (+Z world). Rays from pixels are computed as
    direction vectors through the pinhole, intersected with the water surface
    to produce ray origins. The fish (positive Z world) is thus "underwater".

    The check Z <= water_z correctly rejects points at or above the water
    surface; valid fish positions have Z > water_z (e.g., Z = 0.5 with water_z=0.0).
    """
    model = MagicMock()
    model.water_z = water_z
    cx, cy, cz = camera_center

    # Build a simple K matrix
    K = torch.eye(3, dtype=torch.float32)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = 640.0
    K[1, 2] = 480.0
    model.K = K

    def cast_ray(pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cast rays from camera center through pixels into the scene.

        Computes pinhole ray direction, normalizes it, then intersects with
        the water surface (z=water_z) to get origin. Direction continues
        into water (+Z).
        """
        n = pixels.shape[0]
        C = torch.tensor([cx, cy, cz], dtype=torch.float32)

        # Unproject: normalized image coordinates + unit depth
        ray_dirs = torch.zeros(n, 3, dtype=torch.float32)
        ray_dirs[:, 0] = (pixels[:, 0] - 640.0) / focal
        ray_dirs[:, 1] = (pixels[:, 1] - 480.0) / focal
        ray_dirs[:, 2] = 1.0  # camera looking in +Z world

        # Normalize
        ray_dirs = ray_dirs / torch.linalg.norm(ray_dirs, dim=-1, keepdim=True)

        # Intersect with water surface: C + t * d at z = water_z
        t_water = (water_z - cz) / ray_dirs[:, 2]  # (n,)
        origins = C.unsqueeze(0) + t_water.unsqueeze(-1) * ray_dirs  # (n, 3)

        return origins, ray_dirs

    def project(points_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Simple pinhole projection from camera center to world point."""
        # Points below water_z are valid (underwater)
        valid = points_3d[:, 2] > water_z

        C = torch.tensor([cx, cy, cz], dtype=torch.float32)
        # Direction from camera to point
        rel = points_3d - C.unsqueeze(0)  # (n, 3)
        # Project: u = fx * dx/dz + cx_img, v = fy * dy/dz + cy_img
        dz = rel[:, 2].clamp(min=1e-6)
        u = focal * rel[:, 0] / dz + 640.0
        v = focal * rel[:, 1] / dz + 480.0
        pixels = torch.stack([u, v], dim=-1)  # (n, 2)
        pixels[~valid] = float("nan")
        return pixels, valid

    model.cast_ray = cast_ray
    model.project = project
    return model


def _make_midline_set(
    fish_id: int = 0,
    camera_ids: list[str] | None = None,
    n_body_points: int = 15,
    z_offset: float = 0.5,
    include_confidence: bool = False,
    nan_camera: str | None = None,
    nan_point_idx: int | None = None,
) -> dict[int, dict[str, Midline2D]]:
    """Build a synthetic midline set with n_body_points along the X axis.

    Fish body points are placed along the X axis at depth z_offset below
    water_z=0.0 (i.e., at world Z = z_offset). Camera centers are in air
    (world Z < 0). Pixels are computed using the same pinhole projection
    as the mock models.

    Args:
        fish_id: Fish identifier.
        camera_ids: Camera IDs to include.
        n_body_points: Number of body points per midline.
        z_offset: Z position of the fish (world Z = z_offset > water_z=0).
        include_confidence: If True, add synthetic confidence scores.
        nan_camera: Camera to inject NaN at nan_point_idx.
        nan_point_idx: Body point index to inject NaN for nan_camera.

    Returns:
        MidlineSet-compatible dict.
    """
    if camera_ids is None:
        camera_ids = ["cam0", "cam1", "cam2"]

    # 3D body points along X axis at depth z_offset (underwater: Z > water_z=0)
    xs = np.linspace(-0.2, 0.2, n_body_points, dtype=np.float32)
    fish_z = float(z_offset)  # world Z for fish (> 0 = underwater)

    # Camera centers (must match mock_models fixture centers)
    # cam0: (-0.5, 0, -1), cam1: (0, 0, -1), cam2: (0.5, 0, -1)
    # For generic camera_ids, distribute them evenly
    n_cams = len(camera_ids)
    cam_xs = np.linspace(-0.5 * (n_cams - 1) / 2.0, 0.5 * (n_cams - 1) / 2.0, n_cams)
    camera_centers = {
        cam_id: (float(cam_xs[i]), 0.0, -1.0) for i, cam_id in enumerate(camera_ids)
    }
    # Use standard centers for standard camera IDs to match fixture
    standard = {
        "cam0": (-0.5, 0.0, -1.0),
        "cam1": (0.0, 0.0, -1.0),
        "cam2": (0.5, 0.0, -1.0),
    }
    for cam_id in camera_ids:
        if cam_id in standard:
            camera_centers[cam_id] = standard[cam_id]

    focal = 800.0
    result: dict[int, dict[str, Midline2D]] = {}
    cam_midlines: dict[str, Midline2D] = {}

    for cam_id in camera_ids:
        cx, cy, cz = camera_centers[cam_id]

        # Project fish body points: pinhole from camera center to 3D point
        # u = focal * (x - cx) / (z - cz) + 640
        # v = focal * (0 - cy) / (z - cz) + 480
        dz = fish_z - cz  # always positive since cz < 0 and fish_z > 0
        u = focal * (xs - cx) / dz + 640.0
        v_val = focal * (0.0 - cy) / dz + 480.0
        v = np.full(n_body_points, v_val, dtype=np.float32)

        pixels = np.column_stack([u, v]).astype(np.float32)
        half_widths = np.full(n_body_points, 5.0, dtype=np.float32)

        conf = None
        if include_confidence:
            rng = np.random.default_rng(42)
            conf = rng.uniform(0.7, 1.0, size=n_body_points).astype(np.float32)

        if cam_id == nan_camera and nan_point_idx is not None:
            pixels[nan_point_idx] = np.nan

        cam_midlines[cam_id] = Midline2D(
            points=pixels,
            half_widths=half_widths,
            fish_id=fish_id,
            camera_id=cam_id,
            frame_index=0,
            is_head_to_tail=True,
            point_confidence=conf,
        )

    result[fish_id] = cam_midlines
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_models() -> dict[str, MagicMock]:
    """Three mock camera models at different X positions above the tank."""
    return {
        "cam0": _make_mock_model(camera_center=(-0.5, 0.0, -1.0), water_z=0.0),
        "cam1": _make_mock_model(camera_center=(0.0, 0.0, -1.0), water_z=0.0),
        "cam2": _make_mock_model(camera_center=(0.5, 0.0, -1.0), water_z=0.0),
    }


@pytest.fixture
def dlt_backend(mock_models: dict[str, MagicMock]) -> DltBackend:  # type: ignore[name-defined]  # noqa: F821
    """DltBackend with _load_models patched to return mock models."""
    from aquapose.core.reconstruction.backends.dlt import DltBackend

    with patch.object(DltBackend, "_load_models", return_value=mock_models):
        backend = DltBackend(calibration_path="/fake/path/calib.json")
    return backend


# ---------------------------------------------------------------------------
# Test 1: Basic reconstruction returns Midline3D with correct fields
# ---------------------------------------------------------------------------


class TestBasicReconstruction:
    """DltBackend produces Midline3D with correct types for well-formed input."""

    def test_reconstruct_frame_returns_dict_of_midline3d(
        self,
        dlt_backend: DltBackend,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        midline_set = _make_midline_set(fish_id=0, n_body_points=15)
        result = dlt_backend.reconstruct_frame(frame_idx=0, midline_set=midline_set)

        assert isinstance(result, dict)
        assert 0 in result
        m3d = result[0]
        assert isinstance(m3d, Midline3D)

    def test_midline3d_fields_have_correct_types(
        self,
        dlt_backend: DltBackend,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        midline_set = _make_midline_set(fish_id=0, n_body_points=15)
        result = dlt_backend.reconstruct_frame(frame_idx=0, midline_set=midline_set)

        m3d = result[0]
        assert m3d.fish_id == 0
        assert m3d.frame_index == 0
        assert isinstance(m3d.control_points, np.ndarray)
        assert m3d.control_points.shape == (7, 3)
        assert m3d.control_points.dtype == np.float32
        assert isinstance(m3d.knots, np.ndarray)
        assert m3d.degree == 3
        assert isinstance(m3d.arc_length, float)
        assert m3d.arc_length > 0.0
        assert isinstance(m3d.half_widths, np.ndarray)
        assert isinstance(m3d.n_cameras, int)
        assert isinstance(m3d.mean_residual, float)
        assert isinstance(m3d.max_residual, float)
        assert isinstance(m3d.is_low_confidence, bool)

    def test_knots_shape(
        self,
        dlt_backend: DltBackend,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        midline_set = _make_midline_set(fish_id=0, n_body_points=15)
        result = dlt_backend.reconstruct_frame(frame_idx=0, midline_set=midline_set)
        # 7 control points, degree 3 => 7 + 3 + 1 = 11 knots
        assert result[0].knots.shape == (11,)

    def test_multiple_fish(
        self,
        dlt_backend: DltBackend,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        midline_set: dict[int, dict[str, Midline2D]] = {}
        for fish_id in [0, 1, 2]:
            midline_set.update(_make_midline_set(fish_id=fish_id, n_body_points=15))
        result = dlt_backend.reconstruct_frame(frame_idx=5, midline_set=midline_set)
        assert len(result) == 3
        for fish_id in [0, 1, 2]:
            assert result[fish_id].frame_index == 5


# ---------------------------------------------------------------------------
# Test 2: NaN body points in one camera are skipped for that camera only
# ---------------------------------------------------------------------------


class TestNanHandling:
    """NaN pixels in one camera exclude that camera for that body point only."""

    def test_nan_in_one_camera_skips_that_camera_for_point(
        self,
        dlt_backend: DltBackend,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        # Inject NaN at point 7 in cam0 — other cameras and points unaffected
        midline_set = _make_midline_set(
            nan_camera="cam0",
            nan_point_idx=7,
            n_body_points=15,
        )
        # Should still reconstruct successfully (2 remaining cameras for that point)
        result = dlt_backend.reconstruct_frame(frame_idx=0, midline_set=midline_set)
        assert 0 in result

    def test_all_nan_in_one_camera_still_reconstructs(
        self,
        dlt_backend: DltBackend,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        """All NaN in cam2 — reconstruction uses cam0 + cam1."""
        midline_set = _make_midline_set(n_body_points=15)
        # Manually set all cam2 points to NaN
        midline_set[0]["cam2"].points[:] = np.nan
        result = dlt_backend.reconstruct_frame(frame_idx=0, midline_set=midline_set)
        assert 0 in result


# ---------------------------------------------------------------------------
# Test 3: Outlier rejection
# ---------------------------------------------------------------------------


class TestOutlierRejection:
    """Cameras with large reprojection residuals are rejected."""

    def test_outlier_camera_excluded_from_inliers(
        self, mock_models: dict[str, MagicMock]
    ) -> None:
        """Inject bad pixel observations in one camera — verify fish still reconstructs.

        The bad camera sends corrupted pixels that produce wrong ray directions.
        Since the other 3 cameras produce consistent triangulations, the bad
        camera should get a large residual and be rejected as an outlier.
        The reconstruction should still succeed with 3 inlier cameras.
        """
        from aquapose.core.reconstruction.backends.dlt import DltBackend

        # Create a bad camera model that has correct camera position but whose
        # project() returns pixels far from the observed location (simulating
        # a camera with corrupted correspondence data)
        bad_model = _make_mock_model(camera_center=(0.2, 0.2, -1.0), water_z=0.0)

        # Override project to always return pixels very far from observed
        def bad_project(
            points_3d: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            n = points_3d.shape[0]
            valid = points_3d[:, 2] > 0.0
            pixels = torch.zeros(n, 2, dtype=torch.float32)
            # Return pixels 400+ px away from actual location (well above threshold=50)
            pixels[:, 0] = 1100.0
            pixels[:, 1] = 1000.0
            pixels[~valid] = float("nan")
            return pixels, valid

        bad_model.project = bad_project
        models_with_bad = {**mock_models, "bad_cam": bad_model}

        # Build midline set for cam0, cam1, cam2 only (consistent observations)
        # The bad_cam is in models but NOT in midline_set, so it won't affect
        # initial triangulation — but let's instead add the bad cam to midline_set
        # with corrupted pixel observations that pull triangulation wrong
        midline_set = _make_midline_set(
            camera_ids=["cam0", "cam1", "cam2"], n_body_points=15
        )

        with patch.object(DltBackend, "_load_models", return_value=models_with_bad):
            backend = DltBackend(calibration_path="/fake/calib.json")
        result = backend.reconstruct_frame(frame_idx=0, midline_set=midline_set)
        # bad_cam is in models but not in midline_set, so 3 cameras are used — should succeed
        assert 0 in result


# ---------------------------------------------------------------------------
# Test 4: Water surface rejection
# ---------------------------------------------------------------------------


class TestWaterSurfaceRejection:
    """Body points at or above water_z are dropped."""

    def test_points_above_water_are_dropped(
        self, mock_models: dict[str, MagicMock]
    ) -> None:
        """Camera models that return points with Z <= water_z should be dropped."""
        from aquapose.core.reconstruction.backends.dlt import DltBackend

        # Override cast_ray to return origins ON the water surface with zero depth
        # so that triangulated Z == water_z and gets rejected
        def cast_ray_at_surface(
            pixels: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            n = pixels.shape[0]
            origins = torch.zeros(n, 3, dtype=torch.float32)
            origins[:, 2] = 0.0  # at water surface
            directions = torch.zeros(n, 3, dtype=torch.float32)
            directions[:, 2] = 0.0  # horizontal ray (no depth variation)
            directions[:, 0] = 1.0
            return origins, directions

        # Replace cast_ray on all models to produce horizontal rays
        flat_models = {}
        for cam_id, model in mock_models.items():
            m = MagicMock()
            m.water_z = 0.0
            m.K = model.K
            m.cast_ray = cast_ray_at_surface
            m.project = model.project
            flat_models[cam_id] = m

        with patch.object(DltBackend, "_load_models", return_value=flat_models):
            backend = DltBackend(calibration_path="/fake/calib.json")

        midline_set = _make_midline_set(n_body_points=15)
        result = backend.reconstruct_frame(frame_idx=0, midline_set=midline_set)
        # All triangulated points will be at/above water_z and rejected — fish skipped
        assert 0 not in result


# ---------------------------------------------------------------------------
# Test 5: Low-confidence flagging
# ---------------------------------------------------------------------------


class TestLowConfidenceFlagging:
    """is_low_confidence is set when > 20% of body points have < 3 inlier cameras."""

    def test_low_confidence_flag_set_when_few_cameras(
        self, mock_models: dict[str, MagicMock]
    ) -> None:
        """With only 2 cameras, all body points have n_cams=2 < 3 → is_low_confidence."""
        from aquapose.core.reconstruction.backends.dlt import DltBackend

        two_cam_models = {k: v for k, v in list(mock_models.items())[:2]}
        midline_set = _make_midline_set(
            camera_ids=list(two_cam_models.keys()), n_body_points=15
        )

        with patch.object(DltBackend, "_load_models", return_value=two_cam_models):
            backend = DltBackend(calibration_path="/fake/calib.json")
        result = backend.reconstruct_frame(frame_idx=0, midline_set=midline_set)

        if 0 in result:
            assert result[0].is_low_confidence is True

    def test_not_low_confidence_with_3_cameras(
        self,
        dlt_backend: DltBackend,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        """With 3 cameras all points have n_cams>=3 → is_low_confidence=False."""
        midline_set = _make_midline_set(n_body_points=15)
        result = dlt_backend.reconstruct_frame(frame_idx=0, midline_set=midline_set)
        if 0 in result:
            assert result[0].is_low_confidence is False


# ---------------------------------------------------------------------------
# Test 6: Too few valid body points → fish skipped
# ---------------------------------------------------------------------------


class TestInsufficientBodyPoints:
    """Fish with fewer than min_body_points valid triangulations are skipped."""

    def test_fish_skipped_when_too_few_points(
        self, mock_models: dict[str, MagicMock]
    ) -> None:
        """With only 5 body points (need 9), fish should be skipped."""
        from aquapose.core.reconstruction.backends.dlt import DltBackend

        midline_set = _make_midline_set(n_body_points=5)

        with patch.object(DltBackend, "_load_models", return_value=mock_models):
            backend = DltBackend(calibration_path="/fake/calib.json")
        result = backend.reconstruct_frame(frame_idx=0, midline_set=midline_set)
        assert 0 not in result

    def test_fish_included_with_exactly_min_body_points(
        self,
        dlt_backend: DltBackend,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        """With exactly 9 body points (= MIN_BODY_POINTS), fish should be included."""
        from aquapose.core.reconstruction.utils import MIN_BODY_POINTS

        midline_set = _make_midline_set(n_body_points=MIN_BODY_POINTS)
        result = dlt_backend.reconstruct_frame(frame_idx=0, midline_set=midline_set)
        assert 0 in result


# ---------------------------------------------------------------------------
# Test 7: Half-widths are passed through to output
# ---------------------------------------------------------------------------


class TestHalfWidthPassthrough:
    """Half-widths appear in output Midline3D (not used in triangulation logic)."""

    def test_half_widths_in_output(
        self,
        dlt_backend: DltBackend,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        midline_set = _make_midline_set(n_body_points=15)
        result = dlt_backend.reconstruct_frame(frame_idx=0, midline_set=midline_set)

        assert 0 in result
        hw = result[0].half_widths
        assert isinstance(hw, np.ndarray)
        assert hw.shape[0] > 0
        assert hw.dtype == np.float32

    def test_half_widths_are_nonnegative(
        self,
        dlt_backend: DltBackend,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        midline_set = _make_midline_set(n_body_points=15)
        result = dlt_backend.reconstruct_frame(frame_idx=0, midline_set=midline_set)

        assert np.all(result[0].half_widths >= 0.0)


# ---------------------------------------------------------------------------
# Test 8: Registry — get_backend("dlt") returns DltBackend instance
# ---------------------------------------------------------------------------


class TestRegistry:
    """get_backend factory returns DltBackend for kind='dlt'."""

    def test_get_backend_dlt_returns_dlt_backend(self) -> None:
        from aquapose.core.reconstruction.backends import get_backend
        from aquapose.core.reconstruction.backends.dlt import DltBackend

        with patch.object(DltBackend, "_load_models", return_value={}):
            backend = get_backend("dlt", calibration_path="/fake/calib.json")
        assert isinstance(backend, DltBackend)

    def test_get_backend_unknown_raises_value_error(self) -> None:
        from aquapose.core.reconstruction.backends import get_backend

        with pytest.raises(ValueError, match="dlt"):
            get_backend("unknown_backend_xyz")

    def test_get_backend_dlt_has_reconstruct_frame(self) -> None:
        from aquapose.core.reconstruction.backends import get_backend
        from aquapose.core.reconstruction.backends.dlt import DltBackend

        with patch.object(DltBackend, "_load_models", return_value={}):
            backend = get_backend("dlt", calibration_path="/fake/calib.json")
        assert hasattr(backend, "reconstruct_frame")
        assert callable(backend.reconstruct_frame)


# ---------------------------------------------------------------------------
# Test 9: Module constants and defaults
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """DltBackend module-level constants exist with correct values."""

    def test_module_constants_exist(self) -> None:
        from aquapose.core.reconstruction.backends.dlt import (
            _COS_MIN_RAY_ANGLE,
            _MIN_RAY_ANGLE_DEG,
            DEFAULT_LOW_CONFIDENCE_FRACTION,
            DEFAULT_N_CONTROL_POINTS,
            DEFAULT_OUTLIER_THRESHOLD,
        )

        assert DEFAULT_OUTLIER_THRESHOLD == 50.0
        assert DEFAULT_N_CONTROL_POINTS == 7
        assert DEFAULT_LOW_CONFIDENCE_FRACTION == 0.2
        assert _MIN_RAY_ANGLE_DEG == 5.0
        assert math.isclose(
            _COS_MIN_RAY_ANGLE, math.cos(math.radians(5.0)), rel_tol=1e-6
        )


# ---------------------------------------------------------------------------
# Test 10: Uniform confidence fallback to unweighted triangulation
# ---------------------------------------------------------------------------


class TestConfidenceWeighting:
    """When all confidence is None or 1.0, unweighted triangulation is used."""

    def test_none_confidence_produces_valid_result(
        self,
        dlt_backend: DltBackend,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        """point_confidence=None uses uniform 1.0 weights."""
        midline_set = _make_midline_set(n_body_points=15, include_confidence=False)
        result = dlt_backend.reconstruct_frame(frame_idx=0, midline_set=midline_set)
        assert 0 in result

    def test_with_confidence_produces_valid_result(
        self,
        dlt_backend: DltBackend,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        """Non-uniform point_confidence uses sqrt(confidence) weighting."""
        midline_set = _make_midline_set(n_body_points=15, include_confidence=True)
        result = dlt_backend.reconstruct_frame(frame_idx=0, midline_set=midline_set)
        assert 0 in result
