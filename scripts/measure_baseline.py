"""Measure reconstruction baseline and persist results as regression reference.

Runs the Phase 41 evaluation harness against the current triangulation backend,
flags outliers exceeding 2 standard deviations, and saves the results as
baseline_results.json and baseline_report.txt next to the fixture file.

Usage:
    python scripts/measure_baseline.py <fixture.npz> [--n-frames N]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from aquapose.evaluation import run_evaluation
from aquapose.evaluation.output import format_baseline_report


def main() -> None:
    """Parse arguments, run evaluation, and persist baseline output."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the AquaPose evaluation harness against a midline fixture and "
            "save outlier-annotated baseline results (baseline_results.json and "
            "baseline_report.txt) next to the fixture file."
        )
    )
    parser.add_argument(
        "fixture_path",
        type=Path,
        help="Path to the midline fixture NPZ file produced by the diagnostic-capture stage.",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=15,
        help="Number of frames to evaluate (default: 15).",
    )
    args = parser.parse_args()

    fixture_path: Path = args.fixture_path.resolve()

    if not fixture_path.exists():
        print(f"Error: fixture file not found: {fixture_path}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Run the evaluation harness
    # ------------------------------------------------------------------
    print(f"Running evaluation on {fixture_path.name} ({args.n_frames} frames) ...")
    results = run_evaluation(fixture_path, n_frames=args.n_frames)

    # ------------------------------------------------------------------
    # Generate and print the outlier-flagged report
    # ------------------------------------------------------------------
    report = format_baseline_report(
        results.tier1,
        results.tier2,
        fixture_name=fixture_path.stem,
        frames_evaluated=results.frames_evaluated,
        frames_available=results.frames_available,
    )
    print(report)

    # ------------------------------------------------------------------
    # Save the human-readable report next to the fixture
    # ------------------------------------------------------------------
    report_path = fixture_path.parent / "baseline_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to: {report_path}")

    # ------------------------------------------------------------------
    # Build the baseline JSON by augmenting the existing eval JSON
    # ------------------------------------------------------------------
    with results.json_path.open("r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    baseline_data["baseline_metadata"] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "fixture_path": str(fixture_path),
        "backend_identifier": (
            "aquapose.core.reconstruction.triangulation.triangulate_midlines"
        ),
    }

    baseline_json_path = fixture_path.parent / "baseline_results.json"
    with baseline_json_path.open("w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)
    print(f"Baseline JSON saved to: {baseline_json_path}")


if __name__ == "__main__":
    main()
