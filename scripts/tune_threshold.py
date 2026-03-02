"""Grid search + comparison script for DLT outlier threshold tuning.

Sweeps over outlier_threshold values, scores each on reprojection error and
fish yield, then prints a detailed comparison of the top candidates against
the TriangulationBackend baseline.

Usage:
    python scripts/tune_threshold.py <fixture.npz> [--n-frames N] [--top-n K]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from aquapose.evaluation import run_evaluation
from aquapose.evaluation.harness import EvalResults


def _compute_score(results: EvalResults) -> float:
    """Compute a composite score balancing error and fish yield.

    Lower is better. Penalizes yield drops by 10x the deficit so that
    a threshold which throws away data cannot win on error alone.

    Args:
        results: Evaluation results from run_evaluation.

    Returns:
        Composite score (lower is better).
    """
    mean_err = results.tier1.overall_mean_px
    fish_reconstructed = results.tier1.fish_reconstructed
    fish_available = results.tier1.fish_available
    yield_ratio = fish_reconstructed / max(fish_available, 1)
    penalty = 1.0 + max(0.0, 1.0 - yield_ratio) * 10.0
    return mean_err * penalty


def _print_grid_table(
    rows: list[tuple[float, EvalResults, float]],
) -> None:
    """Print the grid search results table sorted by score.

    Args:
        rows: List of (threshold, results, score) tuples, pre-sorted.
    """
    header = (
        f"{'Threshold':>10} | {'Mean Error (px)':>15} | {'Fish Recon':>10} | "
        f"{'Fish Avail':>10} | {'Yield %':>8} | {'Score':>10}"
    )
    print("\n" + "=" * len(header))
    print("Grid Search Results (sorted by score, lower is better)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for threshold, results, score in rows:
        fish_r = results.tier1.fish_reconstructed
        fish_a = results.tier1.fish_available
        yield_pct = 100.0 * fish_r / max(fish_a, 1)
        print(
            f"{threshold:>10.1f} | {results.tier1.overall_mean_px:>15.2f} | "
            f"{fish_r:>10d} | {fish_a:>10d} | {yield_pct:>7.1f}% | {score:>10.2f}"
        )
    print()


def _print_comparison(
    threshold: float,
    dlt_results: EvalResults,
    baseline_results: EvalResults,
) -> None:
    """Print a side-by-side comparison of DLT at a given threshold vs baseline.

    Args:
        threshold: The DLT outlier threshold value.
        dlt_results: Evaluation results for the DLT backend.
        baseline_results: Evaluation results for the triangulation baseline.
    """
    dlt_t1 = dlt_results.tier1
    base_t1 = baseline_results.tier1
    dlt_t2 = dlt_results.tier2
    base_t2 = baseline_results.tier2

    print(
        f"\n=== Comparison: DLT (threshold={threshold:.1f}) vs Baseline (triangulation) ==="
    )
    print()
    print("Overall:")
    dlt_yield = dlt_t1.fish_reconstructed
    dlt_avail = dlt_t1.fish_available
    base_yield = base_t1.fish_reconstructed
    base_avail = base_t1.fish_available
    print(
        f"  DLT:      mean={dlt_t1.overall_mean_px:.2f} px, "
        f"max={dlt_t1.overall_max_px:.2f} px, "
        f"yield={dlt_yield}/{dlt_avail} "
        f"({100.0 * dlt_yield / max(dlt_avail, 1):.0f}%)"
    )
    print(
        f"  Baseline: mean={base_t1.overall_mean_px:.2f} px, "
        f"max={base_t1.overall_max_px:.2f} px, "
        f"yield={base_yield}/{base_avail} "
        f"({100.0 * base_yield / max(base_avail, 1):.0f}%)"
    )

    # Per-fish breakdown
    all_fish_ids = sorted(set(dlt_t1.per_fish.keys()) | set(base_t1.per_fish.keys()))
    print()
    print("Per-Fish Breakdown:")
    print(
        f"  {'Fish':>6} | {'DLT Mean (px)':>14} | {'DLT Status':>10} | "
        f"{'Base Mean (px)':>15} | {'Base Status':>11}"
    )
    print(
        f"  {'------':>6} | {'-' * 14:>14} | {'-' * 10:>10} | {'-' * 15:>15} | {'-' * 11:>11}"
    )
    for fid in all_fish_ids:
        if fid in dlt_t1.per_fish:
            dlt_mean = f"{dlt_t1.per_fish[fid].get('mean_px', 0.0):.2f}"
            dlt_status = "OK"
        else:
            dlt_mean = "N/A"
            dlt_status = "MISSING"
        if fid in base_t1.per_fish:
            base_mean = f"{base_t1.per_fish[fid].get('mean_px', 0.0):.2f}"
            base_status = "OK"
        else:
            base_mean = "N/A"
            base_status = "MISSING"
        print(
            f"  {fid:>6} | {dlt_mean:>14} | {dlt_status:>10} | "
            f"{base_mean:>15} | {base_status:>11}"
        )

    # Tier 2 stability
    dlt_dropout = dlt_t2.per_fish_dropout
    base_dropout = base_t2.per_fish_dropout
    tier2_fish_ids = sorted(set(dlt_dropout.keys()) | set(base_dropout.keys()))
    if tier2_fish_ids:
        print()
        print("Tier 2 Stability (max displacement in mm):")
        print(f"  {'Fish':>6} | {'DLT Max Disp':>13} | {'Base Max Disp':>14}")
        print(f"  {'------':>6} | {'-' * 13:>13} | {'-' * 14:>14}")
        for fid in tier2_fish_ids:
            # DLT max displacement across all dropout cameras
            if fid in dlt_dropout:
                dlt_vals = [
                    v
                    for cam_dict in [dlt_dropout[fid]]
                    for v in cam_dict.values()
                    if v is not None
                ]
                dlt_max = f"{max(dlt_vals) * 1000.0:.2f}" if dlt_vals else "N/A"
            else:
                dlt_max = "N/A"
            # Baseline max displacement
            if fid in base_dropout:
                base_vals = [
                    v
                    for cam_dict in [base_dropout[fid]]
                    for v in cam_dict.values()
                    if v is not None
                ]
                base_max = f"{max(base_vals) * 1000.0:.2f}" if base_vals else "N/A"
            else:
                base_max = "N/A"
            print(f"  {fid:>6} | {dlt_max:>13} | {base_max:>14}")
    print()


def main() -> None:
    """Parse arguments, run grid search, and print comparison report."""
    parser = argparse.ArgumentParser(
        description=(
            "Grid search over DLT outlier_threshold values. Scores candidates "
            "on reprojection error + fish yield, then compares the top N "
            "against the triangulation baseline with full Tier 1 + Tier 2 metrics."
        )
    )
    parser.add_argument(
        "fixture_path",
        type=Path,
        help="Path to the midline fixture NPZ file.",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=15,
        help="Number of frames to evaluate (default: 15).",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=10.0,
        help="Lower bound of grid search (default: 10).",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=100.0,
        help="Upper bound of grid search (default: 100).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=5.0,
        help="Step size for grid search (default: 5).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top candidates for full comparison (default: 3).",
    )
    args = parser.parse_args()

    fixture_path: Path = args.fixture_path.resolve()
    if not fixture_path.exists():
        print(f"Error: fixture file not found: {fixture_path}", file=sys.stderr)
        sys.exit(1)

    # Generate threshold candidates
    thresholds: list[float] = []
    t = args.min_threshold
    while t <= args.max_threshold + 1e-9:
        thresholds.append(t)
        t += args.step

    print(
        f"Grid search: {len(thresholds)} thresholds "
        f"[{args.min_threshold:.0f} .. {args.max_threshold:.0f}] step {args.step:.0f}"
    )
    print(f"Fixture: {fixture_path.name}, frames: {args.n_frames}")

    # ------------------------------------------------------------------
    # Grid search phase
    # ------------------------------------------------------------------
    grid_results: list[tuple[float, EvalResults, float]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, threshold in enumerate(thresholds, 1):
            print(
                f"  [{i}/{len(thresholds)}] threshold={threshold:.1f} ...",
                end=" ",
                flush=True,
            )
            output_dir = Path(tmpdir) / f"t_{threshold:.1f}"
            output_dir.mkdir()
            results = run_evaluation(
                fixture_path,
                n_frames=args.n_frames,
                backend="dlt",
                outlier_threshold=threshold,
                output_dir=output_dir,
            )
            score = _compute_score(results)
            grid_results.append((threshold, results, score))
            print(
                f"mean={results.tier1.overall_mean_px:.2f} px, "
                f"yield={results.tier1.fish_reconstructed}/{results.tier1.fish_available}, "
                f"score={score:.2f}"
            )

    # Sort by score (ascending = best)
    grid_results.sort(key=lambda x: x[2])

    _print_grid_table(grid_results)

    # ------------------------------------------------------------------
    # Full comparison phase: top N candidates vs baseline
    # ------------------------------------------------------------------
    top_n = min(args.top_n, len(grid_results))
    top_candidates = grid_results[:top_n]

    print("Running baseline (triangulation) for comparison ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_results = run_evaluation(
            fixture_path,
            n_frames=args.n_frames,
            backend="triangulation",
            output_dir=Path(tmpdir),
        )

    for threshold, dlt_results, _score in top_candidates:
        _print_comparison(threshold, dlt_results, baseline_results)

    # Final recommendation
    best_threshold, _best_results, best_score = grid_results[0]
    print(f"Recommended threshold: {best_threshold:.1f} px (score: {best_score:.2f})")


if __name__ == "__main__":
    main()
