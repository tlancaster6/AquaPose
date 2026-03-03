"""Unit tests for flag_outliers and format_baseline_report in output.py."""

from __future__ import annotations

from aquapose.evaluation.metrics import Tier1Result, Tier2Result
from aquapose.evaluation.output import flag_outliers, format_baseline_report

# ---------------------------------------------------------------------------
# Helpers: synthetic results
# ---------------------------------------------------------------------------


def _make_tier1() -> Tier1Result:
    """Build a minimal Tier1Result with known values."""
    return Tier1Result(
        per_camera={
            "cam0": {"mean_px": 2.5, "max_px": 5.0},
            "cam1": {"mean_px": 3.5, "max_px": 7.0},
        },
        per_fish={
            0: {"mean_px": 2.0, "max_px": 4.0},
            1: {"mean_px": 4.0, "max_px": 8.0},
        },
        overall_mean_px=3.0,
        overall_max_px=8.0,
        fish_reconstructed=4,
        fish_available=5,
    )


def _make_tier2() -> Tier2Result:
    """Build a minimal Tier2Result with known values including a None entry."""
    return Tier2Result(
        per_fish_dropout={
            0: {"cam0": 0.0012, "cam1": None},
            1: {"cam0": 0.0025, "cam1": 0.0008},
        }
    )


# ---------------------------------------------------------------------------
# flag_outliers tests
# ---------------------------------------------------------------------------


def test_flag_outliers_identifies_high_value() -> None:
    """flag_outliers returns keys that exceed mean + 2*std."""
    # 9 baseline values near 1.0, one extreme outlier at 1000.0
    # mean≈100.9, std≈299.7, threshold≈700.3 — outlier is clearly flagged
    values = {f"c{i}": 1.0 for i in range(9)}
    values["outlier"] = 1000.0
    outliers = flag_outliers(values)
    assert "outlier" in outliers
    assert "c0" not in outliers


def test_flag_outliers_empty_for_single_value() -> None:
    """flag_outliers returns empty set for fewer than 2 values."""
    outliers = flag_outliers({"cam0": 5.0})
    assert outliers == set()


def test_flag_outliers_empty_for_uniform_values() -> None:
    """flag_outliers returns empty set when std is zero."""
    values = {"cam0": 2.0, "cam1": 2.0, "cam2": 2.0}
    outliers = flag_outliers(values)
    assert outliers == set()


def test_flag_outliers_custom_threshold() -> None:
    """flag_outliers respects threshold_std parameter."""
    values = {"cam0": 1.0, "cam1": 2.0, "cam2": 3.0}
    # With threshold_std=0.5, more values become outliers
    outliers_tight = flag_outliers(values, threshold_std=0.5)
    outliers_wide = flag_outliers(values, threshold_std=3.0)
    assert len(outliers_tight) >= len(outliers_wide)


# ---------------------------------------------------------------------------
# format_baseline_report tests
# ---------------------------------------------------------------------------


def test_format_baseline_report_contains_headers() -> None:
    """format_baseline_report output contains expected section headers."""
    report = format_baseline_report(_make_tier1(), _make_tier2(), "test.npz", 15, 300)
    assert "Tier 1" in report
    assert "Tier 2" in report
    assert "Baseline" in report


def test_format_baseline_report_contains_fixture_name() -> None:
    """format_baseline_report includes fixture name in output."""
    report = format_baseline_report(
        _make_tier1(), _make_tier2(), "my_fixture.npz", 15, 300
    )
    assert "my_fixture.npz" in report


def test_format_baseline_report_contains_frame_counts() -> None:
    """format_baseline_report includes evaluated and available frame counts."""
    report = format_baseline_report(_make_tier1(), _make_tier2(), "test.npz", 15, 300)
    assert "15" in report
    assert "300" in report


def test_format_baseline_report_contains_na_for_none_tier2() -> None:
    """format_baseline_report shows N/A for None Tier 2 dropout entries."""
    report = format_baseline_report(_make_tier1(), _make_tier2(), "test.npz", 15, 300)
    assert "N/A" in report


def test_format_baseline_report_is_string() -> None:
    """format_baseline_report returns a non-empty string."""
    report = format_baseline_report(_make_tier1(), _make_tier2(), "test.npz", 15, 300)
    assert isinstance(report, str)
    assert len(report) > 0


def test_format_baseline_report_contains_outlier_legend() -> None:
    """format_baseline_report includes outlier legend when data has outliers."""
    report = format_baseline_report(_make_tier1(), _make_tier2(), "test.npz", 15, 300)
    assert "outlier" in report.lower()
