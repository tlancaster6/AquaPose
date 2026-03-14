"""Unit tests for the reconstruction stage evaluator."""

from __future__ import annotations

import ast
import itertools

import numpy as np
import pytest

from aquapose.core.types.reconstruction import Midline3D
from aquapose.evaluation.metrics import Tier1Result, Tier2Result
from aquapose.evaluation.stages.reconstruction import (
    DEFAULT_GRID,
    ReconstructionMetrics,
    compute_z_denoising_metrics,
    evaluate_reconstruction,
)

# ---------------------------------------------------------------------------
# Helper: build a synthetic Midline3D
# ---------------------------------------------------------------------------


def _make_midline3d(
    fish_id: int = 0,
    frame_index: int = 0,
    mean_residual: float = 1.0,
    max_residual: float = 2.0,
    is_low_confidence: bool = False,
    per_camera_residuals: dict[str, float] | None = None,
) -> Midline3D:
    """Build a synthetic Midline3D with minimal valid arrays."""
    return Midline3D(
        fish_id=fish_id,
        frame_index=frame_index,
        control_points=np.zeros((7, 3), dtype=np.float32),
        knots=np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1], dtype=np.float32),
        degree=3,
        arc_length=0.2,
        half_widths=np.zeros(15, dtype=np.float32),
        n_cameras=3,
        mean_residual=mean_residual,
        max_residual=max_residual,
        is_low_confidence=is_low_confidence,
        per_camera_residuals=per_camera_residuals,
    )


# ---------------------------------------------------------------------------
# ReconstructionMetrics identity tests
# ---------------------------------------------------------------------------


def test_reconstruction_metrics_is_not_tier1result() -> None:
    """ReconstructionMetrics is NOT Tier1Result — it is a fresh dataclass."""
    assert ReconstructionMetrics is not Tier1Result
    assert not issubclass(ReconstructionMetrics, Tier1Result)


def test_reconstruction_metrics_is_frozen_dataclass() -> None:
    """ReconstructionMetrics is a frozen dataclass."""
    import dataclasses

    assert dataclasses.is_dataclass(ReconstructionMetrics)
    params = dataclasses.fields(ReconstructionMetrics)
    assert len(params) > 0
    # Frozen: cannot assign to fields
    m = ReconstructionMetrics(
        mean_reprojection_error=0.0,
        max_reprojection_error=0.0,
        fish_reconstructed=0,
        fish_available=0,
        inlier_ratio=1.0,
        low_confidence_flag_rate=0.0,
        tier2_stability=None,
        per_camera_error={},
        per_fish_error={},
    )
    with pytest.raises(AttributeError):  # FrozenInstanceError
        m.mean_reprojection_error = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# evaluate_reconstruction empty input
# ---------------------------------------------------------------------------


def test_evaluate_reconstruction_empty_returns_zeroed() -> None:
    """evaluate_reconstruction([]) returns ReconstructionMetrics with zeroed values."""
    result = evaluate_reconstruction([])
    assert isinstance(result, ReconstructionMetrics)
    assert result.mean_reprojection_error == pytest.approx(0.0)
    assert result.max_reprojection_error == pytest.approx(0.0)
    assert result.fish_reconstructed == 0
    assert result.fish_available == 0
    assert result.inlier_ratio == pytest.approx(1.0)
    assert result.low_confidence_flag_rate == pytest.approx(0.0)
    assert result.tier2_stability is None
    assert result.per_camera_error == {}
    assert result.per_fish_error == {}


# ---------------------------------------------------------------------------
# evaluate_reconstruction known data: mean/max error, fish_reconstructed
# ---------------------------------------------------------------------------


def test_evaluate_reconstruction_known_data_mean_max_error() -> None:
    """evaluate_reconstruction with 2 frames, 3 fish each gives correct aggregates."""
    frame_results = [
        (
            0,
            {
                0: _make_midline3d(
                    fish_id=0, frame_index=0, mean_residual=2.0, max_residual=3.0
                ),
                1: _make_midline3d(
                    fish_id=1, frame_index=0, mean_residual=4.0, max_residual=6.0
                ),
                2: _make_midline3d(
                    fish_id=2, frame_index=0, mean_residual=1.0, max_residual=2.0
                ),
            },
        ),
        (
            10,
            {
                0: _make_midline3d(
                    fish_id=0, frame_index=10, mean_residual=2.0, max_residual=4.0
                ),
                1: _make_midline3d(
                    fish_id=1, frame_index=10, mean_residual=4.0, max_residual=5.0
                ),
                2: _make_midline3d(
                    fish_id=2, frame_index=10, mean_residual=3.0, max_residual=7.0
                ),
            },
        ),
    ]
    result = evaluate_reconstruction(frame_results, fish_available=6)
    assert result.fish_reconstructed == 6
    assert result.fish_available == 6
    # Overall max: per-fish max of max_residuals — fish 2 has max 7.0
    assert result.max_reprojection_error == pytest.approx(7.0)
    # Overall mean: mean of per-fish mean errors
    # fish0: mean(2.0, 2.0) = 2.0, fish1: mean(4.0, 4.0) = 4.0, fish2: mean(1.0, 3.0) = 2.0
    # overall: mean(2.0, 4.0, 2.0) = 2.667
    assert result.mean_reprojection_error == pytest.approx(8.0 / 3.0, rel=1e-5)


# ---------------------------------------------------------------------------
# Low-confidence flag rate and inlier ratio
# ---------------------------------------------------------------------------


def test_evaluate_reconstruction_low_confidence_flag_rate() -> None:
    """2 low-confidence fish out of 10 → flag_rate=0.2, inlier_ratio=0.8."""
    frame_results = [
        (
            i,
            {
                i: _make_midline3d(
                    fish_id=i,
                    frame_index=i,
                    is_low_confidence=(i < 2),
                )
            },
        )
        for i in range(10)
    ]
    result = evaluate_reconstruction(frame_results)
    assert result.low_confidence_flag_rate == pytest.approx(0.2)
    assert result.inlier_ratio == pytest.approx(0.8)


def test_evaluate_reconstruction_all_inliers() -> None:
    """All non-low-confidence → flag_rate=0.0, inlier_ratio=1.0."""
    frame_results = [
        (0, {0: _make_midline3d(is_low_confidence=False)}),
        (1, {1: _make_midline3d(is_low_confidence=False)}),
    ]
    result = evaluate_reconstruction(frame_results)
    assert result.low_confidence_flag_rate == pytest.approx(0.0)
    assert result.inlier_ratio == pytest.approx(1.0)


def test_evaluate_reconstruction_all_low_confidence() -> None:
    """All low-confidence → flag_rate=1.0, inlier_ratio=0.0."""
    frame_results = [
        (0, {0: _make_midline3d(is_low_confidence=True)}),
        (1, {1: _make_midline3d(is_low_confidence=True)}),
    ]
    result = evaluate_reconstruction(frame_results)
    assert result.low_confidence_flag_rate == pytest.approx(1.0)
    assert result.inlier_ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Per-camera and per-fish error dicts
# ---------------------------------------------------------------------------


def test_evaluate_reconstruction_per_camera_error() -> None:
    """per_camera_error is populated from Midline3D.per_camera_residuals."""
    midline = _make_midline3d(
        fish_id=0,
        frame_index=0,
        mean_residual=2.5,
        max_residual=4.0,
        per_camera_residuals={"cam0": 1.0, "cam1": 4.0},
    )
    result = evaluate_reconstruction([(0, {0: midline})])
    assert "cam0" in result.per_camera_error
    assert "cam1" in result.per_camera_error
    assert result.per_camera_error["cam0"]["mean_px"] == pytest.approx(1.0)
    assert result.per_camera_error["cam1"]["mean_px"] == pytest.approx(4.0)


def test_evaluate_reconstruction_per_fish_error() -> None:
    """per_fish_error is populated from Midline3D.mean_residual and max_residual."""
    m0 = _make_midline3d(fish_id=0, frame_index=0, mean_residual=2.0, max_residual=5.0)
    m1 = _make_midline3d(fish_id=0, frame_index=10, mean_residual=4.0, max_residual=8.0)
    result = evaluate_reconstruction([(0, {0: m0}), (10, {0: m1})])
    assert 0 in result.per_fish_error
    assert result.per_fish_error[0]["mean_px"] == pytest.approx(3.0)
    assert result.per_fish_error[0]["max_px"] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# tier2_stability extraction
# ---------------------------------------------------------------------------


def test_evaluate_reconstruction_tier2_stability_none_when_no_tier2() -> None:
    """tier2_stability is None when tier2_result param is omitted."""
    result = evaluate_reconstruction([])
    assert result.tier2_stability is None


def test_evaluate_reconstruction_tier2_stability_none_when_explicitly_none() -> None:
    """tier2_stability is None when tier2_result=None is passed explicitly."""
    result = evaluate_reconstruction([], tier2_result=None)
    assert result.tier2_stability is None


def test_evaluate_reconstruction_tier2_stability_from_tier2_result() -> None:
    """tier2_stability equals max of non-None per_fish_dropout displacements."""
    tier2 = Tier2Result(
        per_fish_dropout={
            0: {"cam0": 0.05, "cam1": 0.02},
            1: {"cam0": 0.10, "cam1": None},
        }
    )
    result = evaluate_reconstruction([], tier2_result=tier2)
    # max of [0.05, 0.02, 0.10] = 0.10
    assert result.tier2_stability == pytest.approx(0.10)


def test_evaluate_reconstruction_tier2_stability_none_when_all_none() -> None:
    """tier2_stability is None when per_fish_dropout contains only None values."""
    tier2 = Tier2Result(
        per_fish_dropout={
            0: {"cam0": None, "cam1": None},
            1: {"cam0": None},
        }
    )
    result = evaluate_reconstruction([], tier2_result=tier2)
    assert result.tier2_stability is None


def test_evaluate_reconstruction_tier2_stability_none_when_empty_dropout() -> None:
    """tier2_stability is None when per_fish_dropout is empty."""
    tier2 = Tier2Result(per_fish_dropout={})
    result = evaluate_reconstruction([], tier2_result=tier2)
    assert result.tier2_stability is None


# ---------------------------------------------------------------------------
# DEFAULT_GRID
# ---------------------------------------------------------------------------


def test_default_grid_has_two_keys() -> None:
    """DEFAULT_GRID has exactly 2 keys."""
    assert len(DEFAULT_GRID) == 2


def test_default_grid_key_names() -> None:
    """DEFAULT_GRID has keys 'outlier_threshold' and 'min_cameras'."""
    assert "outlier_threshold" in DEFAULT_GRID
    assert "min_cameras" in DEFAULT_GRID


def test_default_grid_outlier_threshold_has_19_values() -> None:
    """outlier_threshold has 19 values: 10, 15, 20, ..., 100."""
    values = DEFAULT_GRID["outlier_threshold"]
    assert len(values) == 19
    assert values[0] == pytest.approx(10.0)
    assert values[-1] == pytest.approx(100.0)
    # Step 5.0 between consecutive values
    for a, b in itertools.pairwise(values):
        assert b - a == pytest.approx(5.0)


def test_default_grid_min_cameras_has_three_values() -> None:
    """min_cameras has exactly 3 values."""
    assert len(DEFAULT_GRID["min_cameras"]) == 3


def test_default_grid_values_are_floats() -> None:
    """All DEFAULT_GRID values are floats."""
    for key, vals in DEFAULT_GRID.items():
        for v in vals:
            assert isinstance(v, float), (
                f"Expected float in {key}, got {type(v)}: {v!r}"
            )


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------


def test_to_dict_basic_serialization() -> None:
    """to_dict returns JSON-serializable dict with correct types."""
    result = evaluate_reconstruction([])
    d = result.to_dict()
    assert isinstance(d, dict)
    assert isinstance(d["mean_reprojection_error"], float)
    assert isinstance(d["fish_reconstructed"], int)
    assert d["tier2_stability"] is None


def test_to_dict_tier2_stability_as_float() -> None:
    """to_dict serializes tier2_stability as float when provided."""
    tier2 = Tier2Result(per_fish_dropout={0: {"cam0": 0.07}})
    result = evaluate_reconstruction([], tier2_result=tier2)
    d = result.to_dict()
    assert isinstance(d["tier2_stability"], float)
    assert d["tier2_stability"] == pytest.approx(0.07)


def test_to_dict_per_fish_error_keys_are_str() -> None:
    """to_dict converts per_fish_error integer keys to strings."""
    midline = _make_midline3d(fish_id=42, frame_index=0)
    result = evaluate_reconstruction([(0, {42: midline})])
    d = result.to_dict()
    per_fish = d["per_fish_error"]
    assert isinstance(per_fish, dict)
    for k in per_fish:
        assert isinstance(k, str), f"Expected str key, got {type(k)}: {k!r}"


# ---------------------------------------------------------------------------
# No engine imports
# ---------------------------------------------------------------------------


def test_no_engine_imports_in_reconstruction_evaluator() -> None:
    """reconstruction.py must not import from aquapose.engine."""
    from pathlib import Path

    path = (
        Path(__file__).parent.parent.parent.parent
        / "src"
        / "aquapose"
        / "evaluation"
        / "stages"
        / "reconstruction.py"
    )
    source = path.read_text()
    tree = ast.parse(source)

    engine_imports = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "aquapose.engine" in node.module
            ):
                engine_imports.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "aquapose.engine" in alias.name:
                        engine_imports.append(alias.name)

    assert not engine_imports, f"Found engine imports: {engine_imports}"


# ---------------------------------------------------------------------------
# Z-denoising metrics
# ---------------------------------------------------------------------------


def _make_midline3d_with_cp(
    fish_id: int,
    frame_index: int,
    z_offset: float = 0.5,
    z_noise: float = 0.0,
) -> Midline3D:
    """Build a Midline3D with specific control points for z-denoising tests.

    Control points form a line in x with z = z_offset + noise.
    """
    rng = np.random.default_rng(seed=fish_id * 1000 + frame_index)
    cp = np.zeros((7, 3), dtype=np.float32)
    cp[:, 0] = np.linspace(0.0, 0.1, 7)  # x variation
    cp[:, 2] = z_offset + rng.standard_normal(7).astype(np.float32) * z_noise
    knots = np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1], dtype=np.float32)
    return Midline3D(
        fish_id=fish_id,
        frame_index=frame_index,
        control_points=cp,
        knots=knots,
        degree=3,
        arc_length=0.1,
        half_widths=np.zeros(15, dtype=np.float32),
        n_cameras=3,
        mean_residual=1.0,
        max_residual=2.0,
    )


def test_z_denoising_metrics_empty() -> None:
    """Empty input returns zeroed metrics."""
    result = compute_z_denoising_metrics([])
    assert result.total_fish == 0
    assert result.median_z_range_cm == 0.0
    assert result.mean_z_profile_rms_cm == 0.0


def test_z_denoising_metrics_flat_fish() -> None:
    """Fish with no z-noise has near-zero z-range."""
    frame_results = [
        (i, {0: _make_midline3d_with_cp(0, i, z_offset=0.5, z_noise=0.0)})
        for i in range(10)
    ]
    result = compute_z_denoising_metrics(frame_results)
    assert result.total_fish == 1
    # With zero noise control points, z-range should be very small
    assert result.median_z_range_cm < 0.1


def test_z_denoising_metrics_noisy_fish() -> None:
    """Fish with z-noise has larger z-range than flat fish."""
    flat_results = [
        (i, {0: _make_midline3d_with_cp(0, i, z_offset=0.5, z_noise=0.0)})
        for i in range(10)
    ]
    noisy_results = [
        (i, {0: _make_midline3d_with_cp(0, i, z_offset=0.5, z_noise=0.01)})
        for i in range(10)
    ]

    flat_metrics = compute_z_denoising_metrics(flat_results)
    noisy_metrics = compute_z_denoising_metrics(noisy_results)

    assert noisy_metrics.median_z_range_cm > flat_metrics.median_z_range_cm


def test_z_denoising_metrics_to_dict() -> None:
    """ZDenoisingMetrics.to_dict returns JSON-serializable dict."""
    result = compute_z_denoising_metrics([])
    d = result.to_dict()
    assert isinstance(d, dict)
    assert isinstance(d["median_z_range_cm"], float)
    assert isinstance(d["total_fish"], int)
    assert d["residual_delta_px"] is None


def test_z_denoising_metrics_residual_delta_passthrough() -> None:
    """residual_delta_px is passed through to the result."""
    result = compute_z_denoising_metrics([], residual_delta_px=0.42)
    assert result.residual_delta_px == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Percentile fields (EVAL-01)
# ---------------------------------------------------------------------------


def test_evaluate_reconstruction_percentiles_known_data() -> None:
    """evaluate_reconstruction with known residuals returns correct p50/p90/p95."""
    # Create 10 fish-frames with mean_residuals [1, 2, 3, ..., 10]
    frame_results = [
        (
            i,
            {i: _make_midline3d(fish_id=i, frame_index=i, mean_residual=float(i + 1))},
        )
        for i in range(10)
    ]
    result = evaluate_reconstruction(frame_results)
    residuals = np.array([float(i + 1) for i in range(10)])
    assert result.p50_reprojection_error == pytest.approx(
        float(np.percentile(residuals, 50)), abs=1e-5
    )
    assert result.p90_reprojection_error == pytest.approx(
        float(np.percentile(residuals, 90)), abs=1e-5
    )
    assert result.p95_reprojection_error == pytest.approx(
        float(np.percentile(residuals, 95)), abs=1e-5
    )


def test_evaluate_reconstruction_percentiles_empty_are_none() -> None:
    """evaluate_reconstruction([]) returns None for all percentile fields."""
    result = evaluate_reconstruction([])
    assert result.p50_reprojection_error is None
    assert result.p90_reprojection_error is None
    assert result.p95_reprojection_error is None


def test_to_dict_includes_percentile_fields() -> None:
    """to_dict includes percentile fields as float or None."""
    # Non-empty case
    frame_results = [
        (0, {0: _make_midline3d(fish_id=0, frame_index=0, mean_residual=3.0)}),
    ]
    result = evaluate_reconstruction(frame_results)
    d = result.to_dict()
    assert "p50_reprojection_error" in d
    assert "p90_reprojection_error" in d
    assert "p95_reprojection_error" in d
    assert isinstance(d["p50_reprojection_error"], float)

    # Empty case
    result_empty = evaluate_reconstruction([])
    d_empty = result_empty.to_dict()
    assert d_empty["p50_reprojection_error"] is None
    assert d_empty["p90_reprojection_error"] is None
    assert d_empty["p95_reprojection_error"] is None


def test_reconstruction_metrics_backward_compat_without_percentiles() -> None:
    """ReconstructionMetrics can be constructed without percentile fields."""
    m = ReconstructionMetrics(
        mean_reprojection_error=0.0,
        max_reprojection_error=0.0,
        fish_reconstructed=0,
        fish_available=0,
        inlier_ratio=1.0,
        low_confidence_flag_rate=0.0,
        tier2_stability=None,
        per_camera_error={},
        per_fish_error={},
    )
    assert m.p50_reprojection_error is None
    assert m.p90_reprojection_error is None
    assert m.p95_reprojection_error is None


# ---------------------------------------------------------------------------
# Per-keypoint reprojection error (EVAL-04)
# ---------------------------------------------------------------------------


def _make_mock_projection_model(offset_x: float = 0.0, offset_y: float = 0.0):
    """Create a mock projection model that returns pts_3d[:, :2] + offset.

    This simulates a camera projection that maps 3D (x, y, z) to 2D (x+ox, y+oy)
    with all points valid. Returns a tuple (projected_2d, valid_mask).
    """
    import torch

    class MockProjectionModel:
        def project(self, points_3d: torch.Tensor):
            projected = points_3d[:, :2].clone() + torch.tensor(
                [offset_x, offset_y], dtype=points_3d.dtype
            )
            valid = torch.ones(points_3d.shape[0], dtype=torch.bool)
            return projected, valid

    return MockProjectionModel()


def test_compute_per_point_error_known_data() -> None:
    """compute_per_point_error with known offset returns expected per-point errors."""
    from aquapose.evaluation.stages.reconstruction import compute_per_point_error

    # Create a Midline3D with a simple straight-line spline
    cp = np.zeros((7, 3), dtype=np.float32)
    cp[:, 0] = np.linspace(0.0, 1.0, 7)  # x varies 0..1
    knots = np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1], dtype=np.float32)
    m3d = Midline3D(
        fish_id=0,
        frame_index=0,
        control_points=cp,
        knots=knots,
        degree=3,
        arc_length=1.0,
        half_widths=np.zeros(15, dtype=np.float32),
        n_cameras=2,
        mean_residual=1.0,
        max_residual=2.0,
        per_camera_residuals={"cam0": 1.0, "cam1": 1.0},
    )

    # 2D midline: same as the 3D spline projected (x, y) but sampled at 5 points
    from aquapose.core.types.midline import Midline2D

    n_pts = 5
    u_sample = np.linspace(0, 1, n_pts)
    import scipy.interpolate

    spl = scipy.interpolate.BSpline(knots.astype(np.float64), cp.astype(np.float64), 3)
    pts_2d = spl(u_sample)[:, :2].astype(np.float32)

    midline_2d = Midline2D(
        points=pts_2d,
        half_widths=np.zeros(n_pts, dtype=np.float32),
        fish_id=0,
        camera_id="cam0",
        frame_index=0,
    )

    # Mock projection: adds (3.0, 4.0) offset -> error = 5.0 per point
    model = _make_mock_projection_model(offset_x=3.0, offset_y=4.0)

    frame_results = [(0, {0: m3d})]
    midline_sets_by_frame = {0: {0: {"cam0": midline_2d}}}
    projection_models = {"cam0": model}

    result = compute_per_point_error(
        frame_results, midline_sets_by_frame, projection_models, n_body_points=n_pts
    )
    assert result is not None
    # Each point should have error = sqrt(3^2 + 4^2) = 5.0
    for pt_idx in range(n_pts):
        assert result[pt_idx]["mean_px"] == pytest.approx(5.0, abs=0.1)
        assert result[pt_idx]["p90_px"] == pytest.approx(5.0, abs=0.1)


def test_compute_per_point_error_empty_returns_none() -> None:
    """compute_per_point_error with empty frame_results returns None."""
    from aquapose.evaluation.stages.reconstruction import compute_per_point_error

    result = compute_per_point_error([], {}, {})
    assert result is None


def test_compute_per_point_error_no_matching_models_returns_none() -> None:
    """compute_per_point_error with no matching projection models returns None."""
    from aquapose.evaluation.stages.reconstruction import compute_per_point_error

    m3d = _make_midline3d(per_camera_residuals={"cam0": 1.0})
    frame_results = [(0, {0: m3d})]
    # projection_models has a different camera
    result = compute_per_point_error(frame_results, {0: {0: {}}}, {"cam99": None})
    assert result is None


# ---------------------------------------------------------------------------
# Curvature-stratified quality (EVAL-05)
# ---------------------------------------------------------------------------


def test_compute_curvature_stratified_known_data() -> None:
    """compute_curvature_stratified with enough data returns 4 quartile bins."""
    from aquapose.core.types.midline import Midline2D
    from aquapose.evaluation.stages.reconstruction import compute_curvature_stratified

    # Create 8 fish-frames with varying curvature
    frame_results = []
    midline_sets_by_frame = {}
    for i in range(8):
        m3d = _make_midline3d(fish_id=i, frame_index=i, mean_residual=float(i + 1))
        frame_results.append((i, {i: m3d}))
        # Create 2D midlines with varying curvature (bend angle increases with i)
        n_pts = 10
        t = np.linspace(0, 1, n_pts)
        # Curvature scales with i: more bending = higher curvature
        angle = (i + 1) * 0.1
        pts = np.column_stack([t, np.sin(angle * t * np.pi)]).astype(np.float32)
        midline_2d = Midline2D(
            points=pts * 100,  # scale to pixel coords
            half_widths=np.zeros(n_pts, dtype=np.float32),
            fish_id=i,
            camera_id="cam0",
            frame_index=i,
            point_confidence=np.ones(n_pts, dtype=np.float32),
        )
        midline_sets_by_frame[i] = {i: {"cam0": midline_2d}}

    result = compute_curvature_stratified(frame_results, midline_sets_by_frame)
    assert result is not None
    assert len(result) == 4
    # Check all quartiles have expected keys
    for q_key in ["Q1", "Q2", "Q3", "Q4"]:
        assert q_key in result
        assert "mean_error_px" in result[q_key]
        assert "p90_error_px" in result[q_key]
        assert "count" in result[q_key]
        assert "curvature_range" in result[q_key]
    # Total count should be 8
    total_count = sum(result[q]["count"] for q in ["Q1", "Q2", "Q3", "Q4"])
    assert total_count == 8


def test_compute_curvature_stratified_too_few_returns_none() -> None:
    """compute_curvature_stratified with fewer than 4 samples returns None."""
    from aquapose.core.types.midline import Midline2D
    from aquapose.evaluation.stages.reconstruction import compute_curvature_stratified

    # Only 3 fish-frames
    frame_results = []
    midline_sets_by_frame = {}
    for i in range(3):
        m3d = _make_midline3d(fish_id=i, frame_index=i, mean_residual=1.0)
        frame_results.append((i, {i: m3d}))
        pts = np.zeros((5, 2), dtype=np.float32)
        pts[:, 0] = np.linspace(0, 1, 5)
        midline_2d = Midline2D(
            points=pts,
            half_widths=np.zeros(5, dtype=np.float32),
            fish_id=i,
            camera_id="cam0",
            frame_index=i,
            point_confidence=np.ones(5, dtype=np.float32),
        )
        midline_sets_by_frame[i] = {i: {"cam0": midline_2d}}

    result = compute_curvature_stratified(frame_results, midline_sets_by_frame)
    assert result is None


def test_compute_curvature_stratified_empty_returns_none() -> None:
    """compute_curvature_stratified with empty data returns None."""
    from aquapose.evaluation.stages.reconstruction import compute_curvature_stratified

    result = compute_curvature_stratified([], {})
    assert result is None


# ---------------------------------------------------------------------------
# New optional fields (EVAL-04, EVAL-05) backward compat
# ---------------------------------------------------------------------------


def test_reconstruction_metrics_backward_compat_per_point_curvature() -> None:
    """ReconstructionMetrics can be constructed without per_point_error/curvature_stratified."""
    m = ReconstructionMetrics(
        mean_reprojection_error=0.0,
        max_reprojection_error=0.0,
        fish_reconstructed=0,
        fish_available=0,
        inlier_ratio=1.0,
        low_confidence_flag_rate=0.0,
        tier2_stability=None,
        per_camera_error={},
        per_fish_error={},
    )
    assert m.per_point_error is None
    assert m.curvature_stratified is None


def test_to_dict_includes_per_point_error_and_curvature() -> None:
    """to_dict serializes per_point_error and curvature_stratified correctly."""
    m = ReconstructionMetrics(
        mean_reprojection_error=2.0,
        max_reprojection_error=5.0,
        fish_reconstructed=1,
        fish_available=1,
        inlier_ratio=1.0,
        low_confidence_flag_rate=0.0,
        tier2_stability=None,
        per_camera_error={},
        per_fish_error={},
        per_point_error={0: {"mean_px": 1.5, "p90_px": 3.0}},
        curvature_stratified={
            "Q1": {
                "mean_error_px": 1.0,
                "p90_error_px": 2.0,
                "count": 5,
                "curvature_range": "0.00-0.01",
            }
        },
    )
    d = m.to_dict()
    assert "per_point_error" in d
    assert d["per_point_error"] is not None
    assert "0" in d["per_point_error"]  # int keys converted to str
    assert "curvature_stratified" in d
    assert d["curvature_stratified"] is not None

    # None case
    m2 = ReconstructionMetrics(
        mean_reprojection_error=0.0,
        max_reprojection_error=0.0,
        fish_reconstructed=0,
        fish_available=0,
        inlier_ratio=1.0,
        low_confidence_flag_rate=0.0,
        tier2_stability=None,
        per_camera_error={},
        per_fish_error={},
    )
    d2 = m2.to_dict()
    assert d2["per_point_error"] is None
    assert d2["curvature_stratified"] is None
