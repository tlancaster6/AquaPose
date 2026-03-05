"""Performance validation script for v3.4 milestone.

Parses pipeline timing reports and eval results, compares pre- vs post-optimization
metrics, and generates a markdown report with speedup ratios and correctness verdict.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path


def parse_timing(path: Path) -> dict[str, float]:
    """Parse a TimingObserver timing.txt into {stage_name: seconds}.

    Args:
        path: Path to a timing.txt file produced by TimingObserver.

    Returns:
        Dict mapping stage name (e.g. "DetectionStage") to elapsed seconds.
        Includes a "TOTAL" key for the total wall-clock time.
    """
    result: dict[str, float] = {}
    text = path.read_text()
    for match in re.finditer(r"^\s+(\S+)\s+([\d.]+)s", text, re.MULTILINE):
        result[match.group(1)] = float(match.group(2))
    return result


def compare_eval(baseline_path: Path, post_path: Path) -> list[dict[str, object]]:
    """Compare two eval_results.json files and return list of failures.

    Args:
        baseline_path: Path to baseline eval_results.json.
        post_path: Path to post-optimization eval_results.json.

    Returns:
        List of failure dicts with keys: stage, metric, baseline, post,
        tolerance, delta. Empty list means all metrics within tolerance.
    """
    tolerances: dict[str, dict[str, float]] = {
        "detection": {"total_detections": 0, "mean_confidence": 0.0},
        "tracking": {"track_count": 0, "detection_coverage": 0.0},
        "association": {"fish_yield_ratio": 0.02, "singleton_rate": 0.05},
        "midline": {"total_midlines": 0, "mean_confidence": 0.001},
        "reconstruction": {
            "mean_reprojection_error": 0.5,
            "fish_reconstructed": 2,
        },
    }

    baseline = json.loads(baseline_path.read_text())
    post = json.loads(post_path.read_text())

    failures: list[dict[str, object]] = []
    for stage, metrics in tolerances.items():
        b_stage = baseline["stages"].get(stage, {})
        p_stage = post["stages"].get(stage, {})
        for metric, tol in metrics.items():
            b_val = b_stage.get(metric)
            p_val = p_stage.get(metric)
            if b_val is None or p_val is None:
                continue
            delta = abs(p_val - b_val)
            if delta > tol:
                failures.append(
                    {
                        "stage": stage,
                        "metric": metric,
                        "baseline": b_val,
                        "post": p_val,
                        "tolerance": tol,
                        "delta": delta,
                    }
                )
    return failures


def generate_report(
    baseline_timing: dict[str, float],
    post_timing: dict[str, float],
    failures: list[dict[str, object]] | None,
    baseline_run: str,
    post_run: str,
    baseline_eval_available: bool,
) -> str:
    """Generate a markdown performance validation report.

    Args:
        baseline_timing: Parsed baseline timing dict.
        post_timing: Parsed post-optimization timing dict.
        failures: List of eval metric failures, or None if eval was skipped.
        baseline_run: Baseline run identifier.
        post_run: Post-optimization run identifier.
        baseline_eval_available: Whether baseline eval data was available.

    Returns:
        Markdown report string.
    """
    date_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")

    lines: list[str] = [
        "# v3.4 Performance Validation Report",
        "",
        f"**Date:** {date_str}",
        f"**Baseline run:** {baseline_run} (pre-optimization)",
        f"**Post-optimization run:** {post_run}",
        "",
        "## Timing Comparison",
        "",
        "| Stage | Before (s) | After (s) | Speedup |",
        "|-------|-----------|----------|---------|",
    ]

    # Stage display order and friendly names
    stage_order = [
        ("DetectionStage", "Detection"),
        ("TrackingStage", "Tracking"),
        ("AssociationStage", "Association"),
        ("MidlineStage", "Midline"),
        ("ReconstructionStage", "Reconstruction"),
    ]

    for key, name in stage_order:
        before = baseline_timing.get(key)
        after = post_timing.get(key)
        if before is not None and after is not None and after > 0:
            speedup = before / after
            lines.append(f"| {name} | {before:.2f} | {after:.2f} | {speedup:.1f}x |")
        elif before is not None:
            after_str = f"{after:.2f}" if after is not None else "N/A"
            lines.append(f"| {name} | {before:.2f} | {after_str} | N/A |")

    # TOTAL row
    b_total = baseline_timing.get("TOTAL")
    p_total = post_timing.get("TOTAL")
    if b_total is not None and p_total is not None and p_total > 0:
        speedup = b_total / p_total
        lines.append(
            f"| **TOTAL** | **{b_total:.2f}** | **{p_total:.2f}** "
            f"| **{speedup:.1f}x** |"
        )

    lines.append("")
    lines.append("## Correctness Validation")
    lines.append("")

    if not baseline_eval_available:
        lines.append("**Result: SKIPPED**")
        lines.append("")
        lines.append(
            "Eval comparison was skipped because baseline eval data was "
            "unavailable (stale caches from pre-optimization code)."
        )
    elif failures is None or len(failures) == 0:
        lines.append("**Result: PASS**")
        lines.append("")
        lines.append("All eval metrics within tolerance.")
    else:
        lines.append("**Result: FAIL**")
        lines.append("")
        lines.append("The following eval metrics exceeded their tolerance thresholds:")
        lines.append("")
        lines.append("| Stage | Metric | Baseline | Post | Tolerance | Delta |")
        lines.append("|-------|--------|----------|------|-----------|-------|")
        for f in failures:
            lines.append(
                f"| {f['stage']} | {f['metric']} | {f['baseline']} "
                f"| {f['post']} | {f['tolerance']} | {f['delta']:.4f} |"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for performance validation."""
    parser = argparse.ArgumentParser(
        description="Compare pre- and post-optimization pipeline performance."
    )
    parser.add_argument(
        "--baseline-timing",
        type=Path,
        required=True,
        help="Path to baseline timing.txt file.",
    )
    parser.add_argument(
        "--post-run-dir",
        type=Path,
        required=True,
        help="Path to post-optimization run directory.",
    )
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=None,
        help="Path to baseline run directory (optional, for eval comparison).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the markdown report.",
    )
    args = parser.parse_args()

    # Parse timing files
    baseline_timing = parse_timing(args.baseline_timing)
    post_timing_path = args.post_run_dir / "timing.txt"
    if not post_timing_path.exists():
        print(f"ERROR: Post-optimization timing.txt not found: {post_timing_path}")
        raise SystemExit(1)
    post_timing = parse_timing(post_timing_path)

    # Extract run IDs
    baseline_run = args.baseline_timing.stem
    # Try to extract run ID from the timing file header
    baseline_text = args.baseline_timing.read_text()
    run_match = re.search(r"run:\s+(\S+)", baseline_text)
    if run_match:
        baseline_run = run_match.group(1)

    post_run = args.post_run_dir.name

    # Compare eval if baseline run dir provided
    failures: list[dict[str, object]] | None = None
    baseline_eval_available = False

    if args.baseline_run_dir is not None:
        baseline_eval = args.baseline_run_dir / "eval_results.json"
        post_eval = args.post_run_dir / "eval_results.json"
        if baseline_eval.exists() and post_eval.exists():
            baseline_eval_available = True
            failures = compare_eval(baseline_eval, post_eval)
        else:
            missing = []
            if not baseline_eval.exists():
                missing.append(f"baseline: {baseline_eval}")
            if not post_eval.exists():
                missing.append(f"post: {post_eval}")
            print(f"WARNING: eval_results.json missing: {', '.join(missing)}")
            print("Skipping eval comparison.")

    # Generate report
    report = generate_report(
        baseline_timing,
        post_timing,
        failures,
        baseline_run,
        post_run,
        baseline_eval_available,
    )

    # Write and print
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
