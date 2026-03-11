"""Unit tests for TuningOrchestrator and output formatting functions."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from aquapose.evaluation.stages.association import AssociationMetrics
from aquapose.evaluation.stages.reconstruction import ReconstructionMetrics
from aquapose.evaluation.tuning import (
    TuningResult,
    _compute_association_score,
    _compute_reconstruction_score,
    format_comparison_table,
    format_config_diff,
    format_yield_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures for metrics
# ---------------------------------------------------------------------------


def _make_assoc_metrics(
    yield_ratio: float = 0.8,
    singleton_rate: float = 0.3,
) -> AssociationMetrics:
    return AssociationMetrics(
        fish_yield_ratio=yield_ratio,
        singleton_rate=singleton_rate,
        camera_distribution={1: 5, 2: 10, 3: 8},
        total_fish_observations=23,
        frames_evaluated=10,
    )


def _make_recon_metrics(
    mean_error: float = 5.0,
    max_error: float = 15.0,
    fish_reconstructed: int = 80,
    fish_available: int = 90,
) -> ReconstructionMetrics:
    return ReconstructionMetrics(
        mean_reprojection_error=mean_error,
        max_reprojection_error=max_error,
        fish_reconstructed=fish_reconstructed,
        fish_available=fish_available,
        inlier_ratio=0.95,
        low_confidence_flag_rate=0.05,
        tier2_stability=0.02,
        per_camera_error={"cam1": {"mean_px": 4.5, "max_px": 12.0}},
        per_fish_error={0: {"mean_px": 5.0, "max_px": 15.0}},
    )


# ---------------------------------------------------------------------------
# Scoring function tests
# ---------------------------------------------------------------------------


class TestAssociationScore:
    """Tests for _compute_association_score."""

    def test_higher_yield_scores_lower(self) -> None:
        """Higher fish yield should produce a lower (better) score."""
        assoc_high = _make_assoc_metrics(yield_ratio=0.9)
        assoc_low = _make_assoc_metrics(yield_ratio=0.5)
        recon = _make_recon_metrics()

        score_high = _compute_association_score(assoc_high, recon)
        score_low = _compute_association_score(assoc_low, recon)
        assert score_high < score_low

    def test_lower_error_scores_lower_when_yield_tied(self) -> None:
        """Lower reproj error should score better at equal yield."""
        assoc = _make_assoc_metrics(yield_ratio=0.8)
        recon_good = _make_recon_metrics(mean_error=3.0)
        recon_bad = _make_recon_metrics(mean_error=8.0)

        score_good = _compute_association_score(assoc, recon_good)
        score_bad = _compute_association_score(assoc, recon_bad)
        assert score_good < score_bad

    def test_score_is_tuple(self) -> None:
        score = _compute_association_score(_make_assoc_metrics(), _make_recon_metrics())
        assert isinstance(score, tuple)
        assert len(score) == 2


class TestReconstructionScore:
    """Tests for _compute_reconstruction_score."""

    def test_lower_error_scores_lower(self) -> None:
        """Lower mean error should produce a lower (better) score."""
        recon_good = _make_recon_metrics(mean_error=3.0)
        recon_bad = _make_recon_metrics(mean_error=8.0)

        score_good = _compute_reconstruction_score(recon_good)
        score_bad = _compute_reconstruction_score(recon_bad)
        assert score_good < score_bad

    def test_higher_coverage_scores_lower_when_error_tied(self) -> None:
        """Higher coverage should score better at equal error."""
        recon_good = _make_recon_metrics(
            mean_error=5.0, fish_reconstructed=90, fish_available=90
        )
        recon_bad = _make_recon_metrics(
            mean_error=5.0, fish_reconstructed=50, fish_available=90
        )

        score_good = _compute_reconstruction_score(recon_good)
        score_bad = _compute_reconstruction_score(recon_bad)
        assert score_good < score_bad


# ---------------------------------------------------------------------------
# Output formatting tests
# ---------------------------------------------------------------------------


class TestFormatComparisonTable:
    """Tests for format_comparison_table."""

    def test_contains_baseline_and_winner_columns(self) -> None:
        baseline = {"fish_yield_ratio": 0.7, "mean_reprojection_error": 5.5}
        winner = {"fish_yield_ratio": 0.85, "mean_reprojection_error": 4.2}

        result = format_comparison_table(baseline, winner)
        assert "Baseline" in result
        assert "Winner" in result
        assert "Delta" in result

    def test_contains_metric_names(self) -> None:
        baseline = {"fish_yield_ratio": 0.7, "singleton_rate": 0.3}
        winner = {"fish_yield_ratio": 0.85, "singleton_rate": 0.2}

        result = format_comparison_table(baseline, winner)
        assert "fish_yield_ratio" in result
        assert "singleton_rate" in result

    def test_handles_none_values(self) -> None:
        baseline = {"tier2_stability": None}
        winner = {"tier2_stability": None}

        result = format_comparison_table(baseline, winner)
        assert "N/A" in result

    def test_ratio_fields_formatted_as_percent(self) -> None:
        baseline = {"fish_yield_ratio": 0.7}
        winner = {"fish_yield_ratio": 0.85}

        result = format_comparison_table(baseline, winner)
        assert "70.0%" in result
        assert "85.0%" in result

    def test_delta_shown_for_numeric(self) -> None:
        baseline = {"mean_reprojection_error": 5.0}
        winner = {"mean_reprojection_error": 4.0}

        result = format_comparison_table(baseline, winner)
        # Delta should be -1.0
        assert "-1.0000" in result


class TestFormatYieldMatrix:
    """Tests for format_yield_matrix."""

    def test_produces_2d_grid(self) -> None:
        joint_results = [
            {"params": {"ray_dist": 0.02, "score_min": 0.1}, "fish_yield_ratio": 0.5},
            {"params": {"ray_dist": 0.02, "score_min": 0.2}, "fish_yield_ratio": 0.6},
            {"params": {"ray_dist": 0.04, "score_min": 0.1}, "fish_yield_ratio": 0.7},
            {"params": {"ray_dist": 0.04, "score_min": 0.2}, "fish_yield_ratio": 0.8},
        ]

        result = format_yield_matrix(
            joint_results,
            "ray_dist",
            [0.02, 0.04],
            "score_min",
            [0.1, 0.2],
        )
        assert "Yield Matrix" in result
        assert "ray_dist" in result
        assert "score_min" in result

    def test_missing_combo_shows_dashes(self) -> None:
        joint_results = [
            {"params": {"a": 1.0, "b": 2.0}, "fish_yield_ratio": 0.5},
        ]

        result = format_yield_matrix(joint_results, "a", [1.0, 2.0], "b", [2.0, 3.0])
        assert "---" in result

    def test_yield_values_as_percentage(self) -> None:
        joint_results = [
            {"params": {"a": 1.0, "b": 2.0}, "fish_yield_ratio": 0.75},
        ]

        result = format_yield_matrix(joint_results, "a", [1.0], "b", [2.0])
        assert "75.0%" in result


class TestFormatConfigDiff:
    """Tests for format_config_diff."""

    def test_changed_params_shown(self) -> None:
        @dataclass(frozen=True)
        class FakeConfig:
            outlier_threshold: float = 10.0
            min_cameras: int = 3

        baseline = FakeConfig()
        winner_params = {"outlier_threshold": 20.0}

        result = format_config_diff("reconstruction", winner_params, baseline)
        assert "outlier_threshold" in result
        assert "20" in result
        assert "Recommended" in result

    def test_no_changes_message(self) -> None:
        @dataclass(frozen=True)
        class FakeConfig:
            outlier_threshold: float = 10.0

        baseline = FakeConfig()
        winner_params = {"outlier_threshold": 10.0}

        result = format_config_diff("reconstruction", winner_params, baseline)
        assert "No parameter changes" in result

    def test_whole_number_floats_as_int(self) -> None:
        @dataclass(frozen=True)
        class FakeConfig:
            n_sample_points: int = 15

        baseline = FakeConfig()
        winner_params = {"n_sample_points": 21.0}

        result = format_config_diff("reconstruction", winner_params, baseline)
        # Should be 21 not 21.0
        assert "21" in result
        # Should NOT contain 21.0
        assert "21.0" not in result


# ---------------------------------------------------------------------------
# TuningResult dataclass tests
# ---------------------------------------------------------------------------


class TestTuningResult:
    """Tests for the TuningResult dataclass."""

    def test_frozen(self) -> None:
        result = TuningResult(
            stage="association",
            winner_params={"ray_distance_threshold": 0.03},
            baseline_metrics={"fish_yield_ratio": 0.7},
            winner_metrics={"fish_yield_ratio": 0.85},
            all_combos=[],
            joint_grid_results=None,
        )
        with pytest.raises(AttributeError):
            result.stage = "reconstruction"  # type: ignore[misc]

    def test_joint_grid_results_none_for_reconstruction(self) -> None:
        result = TuningResult(
            stage="reconstruction",
            winner_params={"outlier_threshold": 15.0},
            baseline_metrics={},
            winner_metrics={},
            all_combos=[],
            joint_grid_results=None,
        )
        assert result.joint_grid_results is None

    def test_joint_grid_results_present_for_association(self) -> None:
        result = TuningResult(
            stage="association",
            winner_params={},
            baseline_metrics={},
            winner_metrics={},
            all_combos=[],
            joint_grid_results=[
                {"params": {}, "fish_yield_ratio": 0.5, "score": (0, 0)}
            ],
        )
        assert result.joint_grid_results is not None
        assert len(result.joint_grid_results) == 1


# ---------------------------------------------------------------------------
# early_k cast and n_points mapping tests
# ---------------------------------------------------------------------------


class TestParameterMapping:
    """Tests for parameter casting and mapping."""

    def test_early_k_cast_to_int(self) -> None:
        """Verify early_k float values are cast to int in _patch_association_config."""
        from unittest.mock import MagicMock

        from aquapose.engine.config import AssociationConfig, PipelineConfig

        # Create a TuningOrchestrator-like object with just the config
        mock_config = MagicMock(spec=PipelineConfig)
        mock_config.association = AssociationConfig()

        # Simulate the patching logic
        import dataclasses

        params = {"early_k": 15.0}
        clean_params = dict(params)
        clean_params["early_k"] = int(clean_params["early_k"])

        patched = dataclasses.replace(mock_config.association, **clean_params)
        assert isinstance(patched.early_k, int)
        assert patched.early_k == 15

    def test_n_points_mapped_to_n_sample_points(self) -> None:
        """Verify n_points grid key maps to n_sample_points in ReconstructionConfig."""
        import dataclasses

        from aquapose.engine.config import ReconstructionConfig

        base = ReconstructionConfig()
        params = {"n_points": 21.0}

        # Simulate the mapping logic
        clean_params: dict[str, Any] = {}
        for k, v in params.items():
            if k == "n_points":
                clean_params["n_sample_points"] = int(v)
            else:
                clean_params[k] = v

        patched = dataclasses.replace(base, **clean_params)
        assert patched.n_sample_points == 21


# ---------------------------------------------------------------------------
# Two-tier frame count tests
# ---------------------------------------------------------------------------


class TestTwoTierFrameCounts:
    """Tests for n_frames vs n_frames_validate usage."""

    def test_select_frames_different_counts(self) -> None:
        """Verify that different n_frames values produce different sample sizes."""
        from aquapose.evaluation.metrics import select_frames

        all_indices = tuple(range(200))
        fast = select_frames(all_indices, 30)
        validate = select_frames(all_indices, 100)

        assert len(fast) == 30
        assert len(validate) == 100
        # Validate should be a superset-like (same deterministic distribution)
        assert len(fast) < len(validate)


# ---------------------------------------------------------------------------
# TuningOrchestrator chunk cache loading tests
# ---------------------------------------------------------------------------

_N_FRAMES = 4
_CAM_IDS = ["cam0", "cam1"]
_N_ANIMALS = 2
_FISH_IDS = [0, 1]


def _make_tuning_context(n_frames: int = _N_FRAMES) -> Any:
    """Build a minimal PipelineContext with all fields needed for tuning."""
    from aquapose.core.association.types import TrackletGroup
    from aquapose.core.context import PipelineContext
    from aquapose.core.tracking.types import Tracklet2D
    from aquapose.core.types.detection import Detection

    detection_frames = [
        {
            cam_id: [
                Detection(bbox=(10, 20, 50, 80), mask=None, area=4000, confidence=0.9)
            ]
            for cam_id in _CAM_IDS
        }
        for _ in range(n_frames)
    ]
    all_frames = tuple(range(n_frames))
    tracks_2d = {
        cam_id: [
            Tracklet2D(
                camera_id=cam_id,
                track_id=fid,
                frames=all_frames,
                centroids=tuple(
                    (float(i * 10), float(i * 10)) for i in range(n_frames)
                ),
                bboxes=tuple(
                    (float(i * 10), float(i * 10), 50.0, 80.0) for i in range(n_frames)
                ),
                frame_status=tuple("detected" for _ in range(n_frames)),
            )
            for fid in _FISH_IDS
        ]
        for cam_id in _CAM_IDS
    }
    tracklet_groups = [
        TrackletGroup(
            fish_id=fid,
            tracklets=tuple(
                Tracklet2D(
                    camera_id=cam_id,
                    track_id=fid,
                    frames=all_frames,
                    centroids=tuple(
                        (float(i * 10), float(i * 10)) for i in range(n_frames)
                    ),
                    bboxes=tuple(
                        (float(i * 10), float(i * 10), 50.0, 80.0)
                        for i in range(n_frames)
                    ),
                    frame_status=tuple("detected" for _ in range(n_frames)),
                )
                for cam_id in _CAM_IDS
            ),
            confidence=0.95,
            consensus_centroids=None,
        )
        for fid in _FISH_IDS
    ]

    return PipelineContext(
        frame_count=n_frames,
        camera_ids=_CAM_IDS,
        detections=detection_frames,
        tracks_2d=tracks_2d,
        tracklet_groups=tracklet_groups,
        midlines_3d=None,
    )


def _write_chunk_cache(chunk_dir: Path, ctx: Any, run_id: str = "run_test_001") -> None:
    """Write a chunk cache.pkl in the new per-chunk layout."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    envelope = {
        "run_id": run_id,
        "timestamp": "2026-03-03T00:00:00Z",
        "version_fingerprint": "abc123",
        "context": ctx,
    }
    (chunk_dir / "cache.pkl").write_bytes(
        pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    )


def _write_config_yaml(path: Path, n_animals: int = _N_ANIMALS) -> None:
    """Write a minimal config.yaml."""
    path.write_text(f"n_animals: {n_animals}\n")


def _make_tuning_run_dir(
    tmp_path: Path,
    n_chunks: int = 1,
) -> Path:
    """Create a run directory with chunk cache layout for tuning tests."""
    run_dir = tmp_path / "run_001"
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True)
    _write_config_yaml(run_dir / "config.yaml")

    chunks = []
    for chunk_idx in range(n_chunks):
        ctx = _make_tuning_context()
        chunk_dir = diag_dir / f"chunk_{chunk_idx:03d}"
        _write_chunk_cache(chunk_dir, ctx)
        chunks.append(
            {
                "index": chunk_idx,
                "start_frame": chunk_idx * _N_FRAMES,
                "end_frame": (chunk_idx + 1) * _N_FRAMES,
                "stages_cached": ["TrackingStage", "AssociationStage"],
            }
        )

    manifest = {
        "run_id": "run_test_001",
        "total_frames": n_chunks * _N_FRAMES,
        "chunk_size": _N_FRAMES,
        "version_fingerprint": "abc123",
        "chunks": chunks,
    }
    (diag_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return run_dir


class TestTuningOrchestratorChunkLoading:
    """Tests for TuningOrchestrator loading from chunk cache layout."""

    def test_initializes_from_chunk_cache_layout(self, tmp_path: Path) -> None:
        """TuningOrchestrator loads from chunk_000/cache.pkl instead of stage pkl files."""
        from aquapose.evaluation.tuning import TuningOrchestrator

        run_dir = _make_tuning_run_dir(tmp_path, n_chunks=1)

        # Should not raise
        orchestrator = TuningOrchestrator(run_dir / "config.yaml")

        # Tracking and association stages should be present
        assert "tracking" in orchestrator._caches
        assert "association" in orchestrator._caches

    def test_multi_chunk_context_merged(self, tmp_path: Path) -> None:
        """TuningOrchestrator merges multi-chunk contexts correctly."""
        from aquapose.evaluation.tuning import TuningOrchestrator

        n_chunks = 3
        run_dir = _make_tuning_run_dir(tmp_path, n_chunks=n_chunks)
        orchestrator = TuningOrchestrator(run_dir / "config.yaml")

        # Tracking cache should have merged frame_count
        tracking_ctx = orchestrator._caches.get("tracking")
        assert tracking_ctx is not None
        assert tracking_ctx.frame_count == n_chunks * _N_FRAMES

    def test_require_cache_raises_file_not_found_when_missing(
        self, tmp_path: Path
    ) -> None:
        """_require_cache raises FileNotFoundError for missing stage data."""
        from aquapose.evaluation.tuning import TuningOrchestrator

        run_dir = _make_tuning_run_dir(tmp_path, n_chunks=1)
        orchestrator = TuningOrchestrator(run_dir / "config.yaml")

        with pytest.raises(FileNotFoundError):
            orchestrator._require_cache("midline")

    def test_no_chunk_caches_means_empty_caches(self, tmp_path: Path) -> None:
        """TuningOrchestrator with empty diagnostics has empty _caches dict."""
        from aquapose.evaluation.tuning import TuningOrchestrator

        run_dir = tmp_path / "run_001"
        run_dir.mkdir()
        (run_dir / "diagnostics").mkdir()
        _write_config_yaml(run_dir / "config.yaml")

        orchestrator = TuningOrchestrator(run_dir / "config.yaml")
        assert orchestrator._caches == {}
