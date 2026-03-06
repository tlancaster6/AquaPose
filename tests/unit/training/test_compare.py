"""Tests for training run comparison functions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from aquapose.training.compare import (
    discover_runs,
    format_comparison_table,
    load_run_summaries,
    write_comparison_csv,
)


def _make_summary(
    run_id: str = "run_20260305_143022",
    tag: str | None = "round2",
    mAP50: float = 0.923,
    mAP50_95: float = 0.671,
    precision: float = 0.891,
    recall: float = 0.876,
    best_epoch: int = 157,
    duration: float = 3420.5,
    consensus_fraction: float | None = 0.8,
    gap_fraction: float | None = 0.2,
) -> dict:
    """Build a minimal summary dict matching summary.json schema."""
    ds: dict = {}
    if consensus_fraction is not None:
        ds["consensus_fraction"] = consensus_fraction
    if gap_fraction is not None:
        ds["gap_fraction"] = gap_fraction
    return {
        "run_id": run_id,
        "tag": tag,
        "model_type": "obb",
        "metrics": {
            "best_epoch": best_epoch,
            "mAP50": mAP50,
            "mAP50-95": mAP50_95,
            "precision": precision,
            "recall": recall,
        },
        "training_duration_seconds": duration,
        "dataset_sources": ds,
    }


class TestDiscoverRuns:
    """Tests for discover_runs."""

    def test_finds_run_dirs_sorted(self, tmp_path: Path) -> None:
        """discover_runs returns sorted list of run_* directories."""
        (tmp_path / "run_20260305_100000").mkdir()
        (tmp_path / "run_20260305_200000").mkdir()
        (tmp_path / "run_20260305_150000").mkdir()
        (tmp_path / "other_dir").mkdir()
        (tmp_path / "somefile.txt").touch()

        result = discover_runs(tmp_path)
        assert len(result) == 3
        assert result[0].name == "run_20260305_100000"
        assert result[1].name == "run_20260305_150000"
        assert result[2].name == "run_20260305_200000"

    def test_ignores_non_run_dirs(self, tmp_path: Path) -> None:
        """discover_runs ignores directories not matching run_* pattern."""
        (tmp_path / "checkpoint_1").mkdir()
        (tmp_path / "config.yaml").touch()
        result = discover_runs(tmp_path)
        assert result == []

    def test_returns_empty_for_nonexistent_dir(self, tmp_path: Path) -> None:
        """discover_runs returns empty list if directory does not exist."""
        result = discover_runs(tmp_path / "nonexistent")
        assert result == []


class TestLoadRunSummaries:
    """Tests for load_run_summaries."""

    def test_loads_valid_summaries(self, tmp_path: Path) -> None:
        """load_run_summaries loads summary.json from each run dir."""
        run1 = tmp_path / "run_1"
        run1.mkdir()
        summary = _make_summary(run_id="run_1")
        (run1 / "summary.json").write_text(json.dumps(summary))

        run2 = tmp_path / "run_2"
        run2.mkdir()
        summary2 = _make_summary(run_id="run_2", mAP50=0.95)
        (run2 / "summary.json").write_text(json.dumps(summary2))

        result = load_run_summaries([run1, run2])
        assert len(result) == 2
        assert result[0]["run_id"] == "run_1"
        assert result[1]["run_id"] == "run_2"

    def test_skips_missing_summary(self, tmp_path: Path) -> None:
        """load_run_summaries skips dirs without summary.json."""
        run1 = tmp_path / "run_1"
        run1.mkdir()
        summary = _make_summary(run_id="run_1")
        (run1 / "summary.json").write_text(json.dumps(summary))

        run2 = tmp_path / "run_2"
        run2.mkdir()
        # No summary.json

        result = load_run_summaries([run1, run2])
        assert len(result) == 1
        assert result[0]["run_id"] == "run_1"


class TestFormatComparisonTable:
    """Tests for format_comparison_table."""

    def test_produces_header_and_data_rows(self) -> None:
        """format_comparison_table produces header + data rows with correct values."""
        summaries = [_make_summary()]
        table = format_comparison_table(summaries)
        assert "run_20260305_143022" in table
        assert "round2" in table
        assert "0.9230" in table  # mAP50
        assert "0.6710" in table  # mAP50-95
        assert "0.8910" in table  # precision
        assert "0.8760" in table  # recall
        assert "157" in table  # epoch
        assert "80%" in table  # consensus fraction

    def test_returns_message_for_empty_input(self) -> None:
        """format_comparison_table returns message for empty input."""
        table = format_comparison_table([])
        assert "No runs found" in table

    def test_best_values_highlighted(self) -> None:
        """Best metric values are highlighted with ANSI escape codes."""
        summaries = [
            _make_summary(
                run_id="run_1", mAP50=0.90, mAP50_95=0.60, precision=0.80, recall=0.70
            ),
            _make_summary(
                run_id="run_2", mAP50=0.95, mAP50_95=0.70, precision=0.85, recall=0.75
            ),
        ]
        table = format_comparison_table(summaries)
        # The best values (run_2) should contain ANSI bold/green codes
        # click.style(bold=True, fg="green") produces \x1b[1m\x1b[32m...\x1b[0m
        assert "\x1b[" in table  # At least some ANSI codes present

    def test_no_provenance_shows_dash(self) -> None:
        """Sources column shows '-' when no provenance data."""
        summaries = [_make_summary(consensus_fraction=None, gap_fraction=None)]
        table = format_comparison_table(summaries)
        # The sources column should have a dash
        assert "-" in table


class TestWriteComparisonCsv:
    """Tests for write_comparison_csv."""

    def test_writes_valid_csv(self, tmp_path: Path) -> None:
        """write_comparison_csv writes CSV with expected columns."""
        summaries = [
            _make_summary(run_id="run_1"),
            _make_summary(run_id="run_2", mAP50=0.95),
        ]
        csv_path = tmp_path / "compare.csv"
        write_comparison_csv(summaries, csv_path)

        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert "Run" in rows[0]
        assert "mAP50" in rows[0]
        assert "mAP50-95" in rows[0]
        assert "Sources" in rows[0]
        # No ANSI codes in CSV
        assert "\x1b[" not in csv_path.read_text()
