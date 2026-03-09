"""Tests for evaluation comparison module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_eval_results(
    *,
    singleton_rate: float = 0.3,
    fish_yield_ratio: float = 0.6,
    p50_reprojection_error: float = 3.0,
    p90_reprojection_error: float = 5.0,
    total_gaps: int = 10,
    mean_continuity_ratio: float = 0.9,
    per_camera_error: dict | None = None,
) -> dict:
    """Build a minimal eval_results.json-like dict for testing."""
    result: dict = {
        "stages": {
            "association": {
                "singleton_rate": singleton_rate,
                "fish_yield_ratio": fish_yield_ratio,
            },
            "reconstruction": {
                "p50_reprojection_error": p50_reprojection_error,
                "p90_reprojection_error": p90_reprojection_error,
            },
            "fragmentation": {
                "total_gaps": total_gaps,
                "mean_continuity_ratio": mean_continuity_ratio,
            },
        }
    }
    if per_camera_error is not None:
        result["stages"]["reconstruction"]["per_camera_error"] = per_camera_error
    return result


class TestLoadEvalResults:
    """Tests for load_eval_results."""

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        from aquapose.evaluation.compare import load_eval_results

        data = _make_eval_results()
        (tmp_path / "eval_results.json").write_text(json.dumps(data))
        loaded = load_eval_results(tmp_path)
        assert loaded["stages"]["association"]["singleton_rate"] == 0.3

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        from aquapose.evaluation.compare import load_eval_results

        with pytest.raises(FileNotFoundError, match=r"eval_results\.json"):
            load_eval_results(tmp_path)


class TestComputeDeltas:
    """Tests for compute_deltas."""

    def test_delta_math(self) -> None:
        from aquapose.evaluation.compare import compute_deltas

        a = _make_eval_results(singleton_rate=0.3, fish_yield_ratio=0.6)
        b = _make_eval_results(singleton_rate=0.2, fish_yield_ratio=0.8)
        deltas = compute_deltas(a, b)

        sr = deltas["association"]["singleton_rate"]
        assert sr["a"] == pytest.approx(0.3)
        assert sr["b"] == pytest.approx(0.2)
        assert sr["delta"] == pytest.approx(-0.1)
        assert sr["pct_change"] == pytest.approx(-33.333, rel=1e-2)

        fy = deltas["association"]["fish_yield_ratio"]
        assert fy["delta"] == pytest.approx(0.2)

    def test_primary_flag(self) -> None:
        from aquapose.evaluation.compare import compute_deltas

        a = _make_eval_results()
        b = _make_eval_results()
        deltas = compute_deltas(a, b)
        assert deltas["association"]["singleton_rate"]["primary"] is True
        assert deltas["reconstruction"]["p50_reprojection_error"]["primary"] is True
        assert deltas["fragmentation"]["total_gaps"]["primary"] is False

    def test_lower_is_better_directionality(self) -> None:
        """For LOWER_IS_BETTER metrics, delta < 0 means improved=True."""
        from aquapose.evaluation.compare import compute_deltas

        a = _make_eval_results(singleton_rate=0.3)
        b = _make_eval_results(singleton_rate=0.2)
        deltas = compute_deltas(a, b)
        assert deltas["association"]["singleton_rate"]["improved"] is True

    def test_higher_is_better_directionality(self) -> None:
        """For higher-is-better metrics, delta > 0 means improved=True."""
        from aquapose.evaluation.compare import compute_deltas

        a = _make_eval_results(fish_yield_ratio=0.6)
        b = _make_eval_results(fish_yield_ratio=0.8)
        deltas = compute_deltas(a, b)
        assert deltas["association"]["fish_yield_ratio"]["improved"] is True

    def test_division_by_zero(self) -> None:
        """Baseline value near zero produces pct_change=None."""
        from aquapose.evaluation.compare import compute_deltas

        a = _make_eval_results(total_gaps=0)
        b = _make_eval_results(total_gaps=5)
        deltas = compute_deltas(a, b)
        assert deltas["fragmentation"]["total_gaps"]["pct_change"] is None

    def test_dict_valued_metrics_skipped(self) -> None:
        """Dict-valued metrics like per_camera_error should be skipped."""
        from aquapose.evaluation.compare import compute_deltas

        a = _make_eval_results(per_camera_error={"cam0": {"mean_px": 1.0}})
        b = _make_eval_results(per_camera_error={"cam0": {"mean_px": 2.0}})
        deltas = compute_deltas(a, b)
        assert "per_camera_error" not in deltas.get("reconstruction", {})


class TestFormatComparisonTable:
    """Tests for format_comparison_table."""

    def test_returns_nonempty_string_with_metric_names(self) -> None:
        from aquapose.evaluation.compare import format_comparison_table

        a = _make_eval_results()
        b = _make_eval_results(singleton_rate=0.2)
        table = format_comparison_table(a, b, "run_a", "run_b")
        assert isinstance(table, str)
        assert len(table) > 0
        assert "singleton_rate" in table
        assert "association" in table.lower() or "Association" in table


class TestWriteComparisonJson:
    """Tests for write_comparison_json."""

    def test_creates_valid_json(self, tmp_path: Path) -> None:
        from aquapose.evaluation.compare import write_comparison_json

        a = _make_eval_results()
        b = _make_eval_results(singleton_rate=0.2)
        run_a = tmp_path / "run_a"
        run_b = tmp_path / "run_b"
        run_a.mkdir()
        run_b.mkdir()

        out = write_comparison_json(a, b, run_a, run_b, run_b)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "run_a" in data
        assert "run_b" in data
        assert "metrics" in data
        assert "association" in data["metrics"]
