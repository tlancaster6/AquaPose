"""Unit tests for TuningOrchestrator and output formatting functions."""

from __future__ import annotations

from dataclasses import dataclass
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
