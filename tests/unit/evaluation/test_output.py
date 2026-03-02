"""Unit tests for format_summary_table and write_regression_json in output.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aquapose.evaluation.metrics import Tier1Result, Tier2Result
from aquapose.evaluation.output import format_summary_table, write_regression_json

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
# format_summary_table tests
# ---------------------------------------------------------------------------


def test_summary_table_contains_tier1_header() -> None:
    """format_summary_table output contains 'Tier 1' section header."""
    table = format_summary_table(
        _make_tier1(),
        _make_tier2(),
        "test.npz",
        frames_evaluated=15,
        frames_available=300,
    )
    assert "Tier 1" in table


def test_summary_table_contains_tier2_header() -> None:
    """format_summary_table output contains 'Tier 2' section header."""
    table = format_summary_table(
        _make_tier1(),
        _make_tier2(),
        "test.npz",
        frames_evaluated=15,
        frames_available=300,
    )
    assert "Tier 2" in table


def test_summary_table_contains_overall_row() -> None:
    """format_summary_table output contains an 'OVERALL' aggregate row."""
    table = format_summary_table(
        _make_tier1(),
        _make_tier2(),
        "test.npz",
        frames_evaluated=15,
        frames_available=300,
    )
    assert "OVERALL" in table


def test_summary_table_contains_fixture_name() -> None:
    """format_summary_table output contains the fixture name."""
    table = format_summary_table(
        _make_tier1(),
        _make_tier2(),
        "my_fixture.npz",
        frames_evaluated=15,
        frames_available=300,
    )
    assert "my_fixture.npz" in table


def test_summary_table_contains_frame_counts() -> None:
    """format_summary_table output contains evaluated and available frame counts."""
    table = format_summary_table(
        _make_tier1(),
        _make_tier2(),
        "test.npz",
        frames_evaluated=15,
        frames_available=300,
    )
    assert "15" in table
    assert "300" in table


def test_summary_table_contains_known_camera_ids() -> None:
    """format_summary_table contains camera IDs from Tier1Result."""
    table = format_summary_table(
        _make_tier1(),
        _make_tier2(),
        "test.npz",
        frames_evaluated=15,
        frames_available=300,
    )
    assert "cam0" in table
    assert "cam1" in table


def test_summary_table_na_for_none_tier2() -> None:
    """format_summary_table displays N/A for None Tier 2 dropout entries."""
    table = format_summary_table(
        _make_tier1(),
        _make_tier2(),
        "test.npz",
        frames_evaluated=15,
        frames_available=300,
    )
    assert "N/A" in table


def test_summary_table_is_string() -> None:
    """format_summary_table returns a non-empty string."""
    table = format_summary_table(
        _make_tier1(),
        _make_tier2(),
        "test.npz",
        frames_evaluated=15,
        frames_available=300,
    )
    assert isinstance(table, str)
    assert len(table) > 0


# ---------------------------------------------------------------------------
# write_regression_json tests
# ---------------------------------------------------------------------------


def test_write_regression_json_produces_valid_json(tmp_path: Path) -> None:
    """write_regression_json writes a file loadable with json.load."""
    output_path = tmp_path / "eval_results.json"
    write_regression_json(
        _make_tier1(),
        _make_tier2(),
        fixture_name="test.npz",
        frames_evaluated=15,
        frames_available=300,
        output_path=output_path,
    )
    assert output_path.exists()
    with output_path.open() as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_write_regression_json_has_required_keys(tmp_path: Path) -> None:
    """JSON output contains required top-level keys."""
    output_path = tmp_path / "eval_results.json"
    write_regression_json(
        _make_tier1(),
        _make_tier2(),
        fixture_name="test.npz",
        frames_evaluated=15,
        frames_available=300,
        output_path=output_path,
    )
    with output_path.open() as f:
        data = json.load(f)
    assert "tier1" in data
    assert "tier2" in data
    assert "fixture" in data
    assert "frames_evaluated" in data
    assert "frames_available" in data


def test_write_regression_json_null_for_none_tier2(tmp_path: Path) -> None:
    """JSON output contains null (not absent key) for N/A Tier 2 entries."""
    output_path = tmp_path / "eval_results.json"
    write_regression_json(
        _make_tier1(),
        _make_tier2(),
        fixture_name="test.npz",
        frames_evaluated=15,
        frames_available=300,
        output_path=output_path,
    )
    with output_path.open() as f:
        data = json.load(f)
    # Fish 0, cam1 was None in Tier2Result
    tier2_fish0 = data["tier2"]["per_fish_dropout"]["0"]
    assert "cam1" in tier2_fish0
    assert tier2_fish0["cam1"] is None


def test_write_regression_json_numpy_scalars_serialized(tmp_path: Path) -> None:
    """Numpy scalar values are serialized without TypeError."""
    tier1 = Tier1Result(
        per_camera={"cam0": {"mean_px": np.float32(1.5), "max_px": np.float32(3.0)}},
        per_fish={0: {"mean_px": np.float32(1.5), "max_px": np.float32(3.0)}},
        overall_mean_px=np.float32(1.5),
        overall_max_px=np.float32(3.0),
        fish_reconstructed=1,
        fish_available=1,
    )
    tier2 = Tier2Result(per_fish_dropout={0: {"cam0": np.float32(0.001)}})
    output_path = tmp_path / "eval_results.json"
    # Should not raise
    returned = write_regression_json(
        tier1,
        tier2,
        fixture_name="test.npz",
        frames_evaluated=np.int64(15),
        frames_available=np.int64(300),
        output_path=output_path,
    )
    assert returned == output_path
    with output_path.open() as f:
        data = json.load(f)
    assert data["frames_evaluated"] == 15


def test_write_regression_json_returns_output_path(tmp_path: Path) -> None:
    """write_regression_json returns the output path."""
    output_path = tmp_path / "eval_results.json"
    returned = write_regression_json(
        _make_tier1(),
        _make_tier2(),
        fixture_name="test.npz",
        frames_evaluated=15,
        frames_available=300,
        output_path=output_path,
    )
    assert returned == output_path
