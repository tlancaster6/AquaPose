"""Integration tests for run_evaluation harness with synthetic fixtures."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from aquapose.core.types.reconstruction import Midline3D
from aquapose.evaluation.harness import EvalResults, run_evaluation

# ---------------------------------------------------------------------------
# Helpers: build synthetic NPZ fixture
# ---------------------------------------------------------------------------

_N_CAMERAS = 3
_N_FISH = 2
_N_FRAMES = 5
_CAM_IDS = ("cam0", "cam1", "cam2")
_FISH_IDS = (0, 1)
_FRAME_INDICES = (0, 10, 20, 30, 40)


def _make_calib_arrays() -> dict[str, np.ndarray]:
    """Build random-but-valid calibration arrays for an NPZ fixture."""
    rng = np.random.default_rng(42)
    arrays: dict[str, np.ndarray] = {}
    arrays["calib/water_z"] = np.array(0.05, dtype=np.float32)
    arrays["calib/n_air"] = np.array(1.0, dtype=np.float32)
    arrays["calib/n_water"] = np.array(1.333, dtype=np.float32)
    arrays["calib/interface_normal"] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    for cam_id in _CAM_IDS:
        # Identity intrinsic
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = 800.0
        K[1, 1] = 800.0
        K[0, 2] = 400.0
        K[1, 2] = 300.0
        arrays[f"calib/{cam_id}/K_new"] = K
        arrays[f"calib/{cam_id}/R"] = np.eye(3, dtype=np.float32)
        t = rng.standard_normal(3).astype(np.float32)
        arrays[f"calib/{cam_id}/t"] = t
    return arrays


def _make_midline_arrays() -> dict[str, np.ndarray]:
    """Build synthetic midline NPZ arrays for 5 frames, 2 fish, 3 cameras."""
    rng = np.random.default_rng(7)
    arrays: dict[str, np.ndarray] = {}
    for frame_idx in _FRAME_INDICES:
        for fish_id in _FISH_IDS:
            for cam_id in _CAM_IDS:
                prefix = f"midline/{frame_idx}/{fish_id}/{cam_id}"
                arrays[f"{prefix}/points"] = rng.random((10, 2)).astype(np.float32)
                arrays[f"{prefix}/half_widths"] = rng.random(10).astype(np.float32)
                arrays[f"{prefix}/point_confidence"] = np.ones(10, dtype=np.float32)
                arrays[f"{prefix}/is_head_to_tail"] = np.bool_(True)
    return arrays


def _make_meta_arrays(version: str = "2.0") -> dict[str, np.ndarray]:
    """Build fixture metadata arrays."""
    return {
        "meta/version": np.str_(version),
        "meta/camera_ids": np.array(list(_CAM_IDS), dtype=object),
        "meta/frame_indices": np.array(list(_FRAME_INDICES), dtype=np.int64),
        "meta/frame_count": np.int64(50),
        "meta/timestamp": np.str_("2026-03-02T00:00:00Z"),
    }


def _write_fixture(
    path: Path, version: str = "2.0", include_calib: bool = True
) -> Path:
    """Write a synthetic NPZ fixture to ``path`` and return the path."""
    arrays: dict[str, np.ndarray] = {}
    arrays.update(_make_meta_arrays(version))
    arrays.update(_make_midline_arrays())
    if include_calib:
        arrays.update(_make_calib_arrays())
    np.savez_compressed(str(path), **arrays)
    return path


# ---------------------------------------------------------------------------
# Synthetic Midline3D factory (for mock returns)
# ---------------------------------------------------------------------------


def _make_midline3d_result(
    fish_id: int,
    frame_index: int,
    cam_ids: tuple[str, ...] = _CAM_IDS,
) -> Midline3D:
    """Build a synthetic Midline3D with known control points and residuals."""
    rng = np.random.default_rng(fish_id * 100 + frame_index)
    ctrl = rng.random((7, 3)).astype(np.float32)
    per_cam = {cam_id: float(rng.random()) * 5.0 for cam_id in cam_ids}
    return Midline3D(
        fish_id=fish_id,
        frame_index=frame_index,
        control_points=ctrl,
        knots=np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1], dtype=np.float32),
        degree=3,
        arc_length=0.2,
        half_widths=np.zeros(15, dtype=np.float32),
        n_cameras=len(cam_ids),
        mean_residual=float(np.mean(list(per_cam.values()))),
        max_residual=float(np.max(list(per_cam.values()))),
        per_camera_residuals=per_cam,
    )


def _make_triangulate_side_effect(
    cam_ids: tuple[str, ...] = _CAM_IDS,
) -> object:
    """Return a mock side_effect for triangulate_midlines that returns synthetic results.

    Returns a result dict {fish_id: Midline3D} based on the midline_set arg.
    When the reduced set has only 1 camera (leave-one-out with 1 remaining), return {}.
    """

    def _side_effect(
        midline_set: object,
        models: object,
        frame_index: int = 0,
        **kwargs: object,
    ) -> dict[int, Midline3D]:
        # midline_set: fish_id -> cam_id -> Midline2D
        result = {}
        for fish_id, cam_map in midline_set.items():  # type: ignore[union-attr]
            available_cams = tuple(cam_map.keys())
            if len(available_cams) < 2:
                continue  # Not enough cameras — skip
            result[fish_id] = _make_midline3d_result(
                fish_id, frame_index, available_cams
            )
        return result

    return _side_effect


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@mock.patch("aquapose.evaluation.harness.triangulate_midlines")
def test_run_evaluation_returns_eval_results(
    mock_tri: mock.MagicMock, tmp_path: Path
) -> None:
    """run_evaluation returns EvalResults with populated tier1 and tier2."""
    mock_tri.side_effect = _make_triangulate_side_effect()
    fixture_path = _write_fixture(tmp_path / "fixture.npz")
    results = run_evaluation(fixture_path, n_frames=5)
    assert isinstance(results, EvalResults)
    assert isinstance(results.summary_table, str)
    assert len(results.tier1.per_fish) > 0
    assert results.frames_evaluated > 0
    assert results.frames_available == _N_FRAMES


@mock.patch("aquapose.evaluation.harness.triangulate_midlines")
def test_run_evaluation_writes_json(mock_tri: mock.MagicMock, tmp_path: Path) -> None:
    """run_evaluation writes eval_results.json next to the fixture."""
    mock_tri.side_effect = _make_triangulate_side_effect()
    fixture_path = _write_fixture(tmp_path / "fixture.npz")
    results = run_evaluation(fixture_path, n_frames=5)
    assert results.json_path.exists()
    assert results.json_path.parent == tmp_path


@mock.patch("aquapose.evaluation.harness.triangulate_midlines")
def test_run_evaluation_summary_table_has_sections(
    mock_tri: mock.MagicMock, tmp_path: Path
) -> None:
    """Summary table from run_evaluation contains Tier 1 and Tier 2 sections."""
    mock_tri.side_effect = _make_triangulate_side_effect()
    fixture_path = _write_fixture(tmp_path / "fixture.npz")
    results = run_evaluation(fixture_path, n_frames=5)
    assert "Tier 1" in results.summary_table
    assert "Tier 2" in results.summary_table


def test_run_evaluation_raises_for_missing_calib(tmp_path: Path) -> None:
    """run_evaluation raises ValueError when fixture has no CalibBundle (v1.0)."""
    fixture_path = _write_fixture(
        tmp_path / "fixture_v1.npz", version="1.0", include_calib=False
    )
    with pytest.raises(ValueError, match="calib"):
        run_evaluation(fixture_path, n_frames=5)


@mock.patch("aquapose.evaluation.harness.triangulate_midlines")
def test_run_evaluation_custom_output_dir(
    mock_tri: mock.MagicMock, tmp_path: Path
) -> None:
    """run_evaluation writes JSON to custom output_dir when provided."""
    mock_tri.side_effect = _make_triangulate_side_effect()
    fixture_path = _write_fixture(tmp_path / "fixture.npz")
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    results = run_evaluation(fixture_path, n_frames=5, output_dir=output_dir)
    assert results.json_path.parent == output_dir
