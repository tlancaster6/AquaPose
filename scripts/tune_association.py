"""Sweep AssociationConfig parameters in priority order to maximise fish yield.

Generates a fresh fixture per config via generate_fixture(), evaluates each
via run_evaluation(skip_tier2=True), and prints a console-only comparison
report. Top-N candidates get full Tier 1 + Tier 2 evaluation.

Usage:
    python scripts/tune_association.py <yaml_config> <fixture_path>
        [--n-frames N] [--top-n K] [--backend BACKEND]
        [--outlier-threshold T] [--target-fish N]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from aquapose.evaluation import run_evaluation
from aquapose.evaluation.harness import EvalResults, generate_fixture
from aquapose.evaluation.metrics import select_frames
from aquapose.io.midline_fixture import load_midline_fixture

# ---------------------------------------------------------------------------
# Sweep ranges (from analysis doc Section 6)
# ---------------------------------------------------------------------------

SWEEP_RANGES: dict[str, list[float]] = {
    "ray_distance_threshold": [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
    "score_min": [0.15, 0.20, 0.25, 0.30, 0.35],
    "ghost_pixel_threshold": [50.0, 10000.0],  # current vs ablated
    "eviction_reproj_threshold": [0.02, 0.03, 0.04, 0.05, 0.08],
}

SECONDARY_RANGES: dict[str, list[float]] = {
    "early_k": [10, 15, 20],
    "leiden_resolution": [0.8, 1.0, 1.2],
}

# Ordered sweep stages (priority order with carry-forward)
PRIMARY_STAGES: list[str] = [
    "ray_distance_threshold",
    "score_min",
    "ghost_pixel_threshold",
    "eviction_reproj_threshold",
]

SECONDARY_STAGES: list[str] = [
    "early_k",
    "leiden_resolution",
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _compute_score(results: EvalResults) -> tuple[float, float]:
    """Return (primary, secondary) where lower is better for both.

    Primary: negative fish yield (higher yield = lower score).
    Secondary: mean reprojection error (lower error = lower score).

    Args:
        results: Evaluation results from run_evaluation.

    Returns:
        Tuple of (negative yield ratio, mean error) for sorting.
    """
    fish_yield = results.tier1.fish_reconstructed / max(results.tier1.fish_available, 1)
    return (-fish_yield, results.tier1.overall_mean_px)


# ---------------------------------------------------------------------------
# Association-level metrics from fixture
# ---------------------------------------------------------------------------


def _compute_association_metrics(
    fixture_path: Path,
    n_frames: int,
) -> dict[str, object]:
    """Compute multi-camera yield and singleton rate from a fixture.

    Args:
        fixture_path: Path to the fixture NPZ file.
        n_frames: Number of frames to sample.

    Returns:
        Dict with keys: camera_distribution (dict of cam_count -> fish_count),
        total_fish_observations, singleton_count, singleton_rate.
    """
    fixture = load_midline_fixture(fixture_path)
    selected = select_frames(fixture.frame_indices, n_frames)
    frame_to_pos = {fi: pos for pos, fi in enumerate(fixture.frame_indices)}

    camera_distribution: dict[int, int] = defaultdict(int)
    total_observations = 0
    singleton_count = 0

    for fi in selected:
        midline_set = fixture.frames[frame_to_pos[fi]]
        for _fish_id, cam_map in midline_set.items():
            n_cams = len(cam_map)
            camera_distribution[n_cams] += 1
            total_observations += 1
            if n_cams == 1:
                singleton_count += 1

    singleton_rate = singleton_count / max(total_observations, 1)
    return {
        "camera_distribution": dict(camera_distribution),
        "total_fish_observations": total_observations,
        "singleton_count": singleton_count,
        "singleton_rate": singleton_rate,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def _print_stage_header(stage_num: int, param_name: str, n_values: int) -> None:
    """Print a stage header.

    Args:
        stage_num: Stage number (1-based).
        param_name: Parameter being swept.
        n_values: Number of values to try.
    """
    print(f"\n{'=' * 70}")
    print(f"Stage {stage_num}: Sweep {param_name} ({n_values} values)")
    print("=" * 70)


def _print_stage_table(
    param_name: str,
    rows: list[tuple[float, EvalResults, tuple[float, float], dict[str, object]]],
) -> None:
    """Print results table for a sweep stage.

    Args:
        param_name: Name of the parameter being swept.
        rows: List of (value, results, score, assoc_metrics) tuples.
    """
    header = (
        f"  {param_name:>25} | {'Fish Recon':>10} | {'Fish Avail':>10} | "
        f"{'Yield %':>8} | {'Mean Err':>10} | {'Singleton %':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for value, results, _score, assoc in rows:
        fish_r = results.tier1.fish_reconstructed
        fish_a = results.tier1.fish_available
        yield_pct = 100.0 * fish_r / max(fish_a, 1)
        singleton_pct = 100.0 * assoc["singleton_rate"]
        print(
            f"  {value:>25.4f} | {fish_r:>10d} | {fish_a:>10d} | "
            f"{yield_pct:>7.1f}% | {results.tier1.overall_mean_px:>10.2f} | "
            f"{singleton_pct:>11.1f}%"
        )
    print()


def _print_camera_distribution(assoc_metrics: dict[str, object]) -> None:
    """Print multi-camera yield distribution.

    Args:
        assoc_metrics: Association metrics dict from _compute_association_metrics.
    """
    dist = assoc_metrics["camera_distribution"]
    total = assoc_metrics["total_fish_observations"]
    if not dist:
        return
    print("  Camera distribution:")
    for n_cams in sorted(dist.keys()):
        count = dist[n_cams]
        pct = 100.0 * count / max(total, 1)
        label = f"{n_cams} cam{'s' if n_cams != 1 else ' '}"
        print(f"    {label}: {count:>5d} ({pct:>5.1f}%)")


def _print_final_report(
    baseline_results: EvalResults,
    baseline_assoc: dict[str, object],
    winner_results: EvalResults,
    winner_assoc: dict[str, object],
    best_config: dict[str, float],
) -> None:
    """Print the final comparison report.

    Args:
        baseline_results: Evaluation results for baseline fixture.
        baseline_assoc: Association metrics for baseline.
        winner_results: Evaluation results for winning config.
        winner_assoc: Association metrics for winner.
        best_config: Dict of parameter -> best value from sweep.
    """
    print("\n" + "=" * 70)
    print("FINAL REPORT: Baseline vs Winner")
    print("=" * 70)

    base_t1 = baseline_results.tier1
    win_t1 = winner_results.tier1

    base_yield = base_t1.fish_reconstructed
    base_avail = base_t1.fish_available
    win_yield = win_t1.fish_reconstructed
    win_avail = win_t1.fish_available

    print(f"\n  {'Metric':<25} | {'Baseline':>12} | {'Winner':>12} | {'Delta':>12}")
    print(f"  {'-' * 25} | {'-' * 12} | {'-' * 12} | {'-' * 12}")

    # Fish yield
    base_yield_pct = 100.0 * base_yield / max(base_avail, 1)
    win_yield_pct = 100.0 * win_yield / max(win_avail, 1)
    delta_yield = win_yield_pct - base_yield_pct
    print(
        f"  {'Fish Recon':<25} | {base_yield:>12d} | {win_yield:>12d} | "
        f"{win_yield - base_yield:>+12d}"
    )
    print(
        f"  {'Fish Avail':<25} | {base_avail:>12d} | {win_avail:>12d} | "
        f"{win_avail - base_avail:>+12d}"
    )
    print(
        f"  {'Yield %':<25} | {base_yield_pct:>11.1f}% | {win_yield_pct:>11.1f}% | "
        f"{delta_yield:>+11.1f}%"
    )

    # Reprojection error
    delta_mean = win_t1.overall_mean_px - base_t1.overall_mean_px
    print(
        f"  {'Mean Error (px)':<25} | {base_t1.overall_mean_px:>12.2f} | "
        f"{win_t1.overall_mean_px:>12.2f} | {delta_mean:>+12.2f}"
    )
    delta_max = win_t1.overall_max_px - base_t1.overall_max_px
    print(
        f"  {'Max Error (px)':<25} | {base_t1.overall_max_px:>12.2f} | "
        f"{win_t1.overall_max_px:>12.2f} | {delta_max:>+12.2f}"
    )

    # Singleton rate
    base_sing = 100.0 * baseline_assoc["singleton_rate"]
    win_sing = 100.0 * winner_assoc["singleton_rate"]
    delta_sing = win_sing - base_sing
    print(
        f"  {'Singleton Rate':<25} | {base_sing:>11.1f}% | {win_sing:>11.1f}% | "
        f"{delta_sing:>+11.1f}%"
    )

    # Tier 2 stability (if available)
    base_t2 = baseline_results.tier2
    win_t2 = winner_results.tier2
    if base_t2.overall_max_displacement is not None:
        base_disp = base_t2.overall_max_displacement * 1000.0
        win_disp_raw = win_t2.overall_max_displacement
        if win_disp_raw is not None:
            win_disp = win_disp_raw * 1000.0
            delta_disp = win_disp - base_disp
            print(
                f"  {'Max Displacement (mm)':<25} | {base_disp:>12.2f} | "
                f"{win_disp:>12.2f} | {delta_disp:>+12.2f}"
            )
        else:
            print(
                f"  {'Max Displacement (mm)':<25} | {base_disp:>12.2f} | "
                f"{'N/A':>12} | {'N/A':>12}"
            )

    # Camera distributions
    print("\n  Baseline camera distribution:")
    _print_camera_distribution(baseline_assoc)
    print("  Winner camera distribution:")
    _print_camera_distribution(winner_assoc)

    # Recommended config
    print("\n  Recommended AssociationConfig updates:")
    for param, value in sorted(best_config.items()):
        if isinstance(value, float) and value == int(value):
            print(f"    {param}: {int(value)}")
        else:
            print(f"    {param}: {value}")
    print()


# ---------------------------------------------------------------------------
# Sweep engine
# ---------------------------------------------------------------------------


def _run_sweep_stage(
    stage_num: int,
    param_name: str,
    values: list[float],
    yaml_config: Path,
    carry_forward: dict[str, float],
    n_frames: int,
    backend: str,
    outlier_threshold: float,
) -> tuple[
    float, list[tuple[float, EvalResults, tuple[float, float], dict[str, object]]]
]:
    """Run a single sweep stage, returning the best value and all rows.

    Args:
        stage_num: Stage number for display.
        param_name: Parameter name to sweep.
        values: List of values to try.
        yaml_config: Path to the pipeline YAML config.
        carry_forward: Dict of previously-locked parameter values.
        n_frames: Number of frames for evaluation.
        backend: Reconstruction backend name.
        outlier_threshold: DLT outlier threshold.

    Returns:
        Tuple of (best_value, rows) where rows contains
        (value, results, score, assoc_metrics) for each tried value.
    """
    _print_stage_header(stage_num, param_name, len(values))

    rows: list[tuple[float, EvalResults, tuple[float, float], dict[str, object]]] = []

    for i, val in enumerate(values, 1):
        overrides = dict(carry_forward)
        overrides[param_name] = val

        print(
            f"  [{i}/{len(values)}] {param_name}={val} ...",
            end=" ",
            flush=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "midline_fixtures.npz"
            fixture_path = generate_fixture(
                yaml_config,
                association_overrides=overrides,
                output_path=output_path,
            )

            eval_output = Path(tmpdir) / "eval"
            eval_output.mkdir()
            results = run_evaluation(
                fixture_path,
                n_frames=n_frames,
                skip_tier2=True,
                backend=backend,
                outlier_threshold=outlier_threshold,
                output_dir=eval_output,
                association_overrides=overrides,
            )

            assoc_metrics = _compute_association_metrics(fixture_path, n_frames)

        score = _compute_score(results)
        rows.append((val, results, score, assoc_metrics))

        fish_r = results.tier1.fish_reconstructed
        fish_a = results.tier1.fish_available
        yield_pct = 100.0 * fish_r / max(fish_a, 1)
        print(
            f"yield={fish_r}/{fish_a} ({yield_pct:.0f}%), "
            f"mean={results.tier1.overall_mean_px:.2f} px, "
            f"singleton={100.0 * assoc_metrics['singleton_rate']:.1f}%"
        )

    # Sort by score tuple (lower is better)
    rows.sort(key=lambda x: x[2])
    _print_stage_table(param_name, rows)

    best_val = rows[0][0]
    best_yield = rows[0][1].tier1.fish_reconstructed
    best_avail = rows[0][1].tier1.fish_available
    best_pct = 100.0 * best_yield / max(best_avail, 1)
    print(f"  >> Best: {param_name}={best_val} (yield={best_pct:.0f}%)")

    return best_val, rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, run sweep, and print comparison report."""
    parser = argparse.ArgumentParser(
        description=(
            "Sweep AssociationConfig parameters in priority order to "
            "maximise multi-camera fish yield. Generates fixtures per config, "
            "evaluates with skip_tier2=True for speed, then runs full evaluation "
            "on top candidates."
        )
    )
    parser.add_argument(
        "yaml_config",
        type=Path,
        help="Path to the pipeline YAML config file.",
    )
    parser.add_argument(
        "fixture_path",
        type=Path,
        help="Path to the baseline midline fixture NPZ file.",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=15,
        help="Number of frames to evaluate (default: 15).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top candidates for full Tier 2 evaluation (default: 3).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="dlt",
        help="Reconstruction backend (default: dlt).",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=50.0,
        help="DLT outlier threshold in pixels (default: 50.0).",
    )
    parser.add_argument(
        "--target-fish",
        type=int,
        default=9,
        help="Target fish count per frame (default: 9).",
    )
    args = parser.parse_args()

    yaml_config: Path = args.yaml_config.resolve()
    fixture_path: Path = args.fixture_path.resolve()

    if not yaml_config.exists():
        print(f"Error: YAML config not found: {yaml_config}", file=sys.stderr)
        sys.exit(1)
    if not fixture_path.exists():
        print(f"Error: fixture file not found: {fixture_path}", file=sys.stderr)
        sys.exit(1)

    print("Association parameter tuning sweep")
    print(f"  Config: {yaml_config.name}")
    print(f"  Baseline fixture: {fixture_path.name}")
    print(f"  Backend: {args.backend}, outlier_threshold: {args.outlier_threshold}")
    print(f"  Target fish: {args.target_fish}, n_frames: {args.n_frames}")

    # ------------------------------------------------------------------
    # Baseline evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Baseline Evaluation (from provided fixture)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_results = run_evaluation(
            fixture_path,
            n_frames=args.n_frames,
            skip_tier2=True,
            backend=args.backend,
            outlier_threshold=args.outlier_threshold,
            output_dir=Path(tmpdir),
        )

    baseline_assoc = _compute_association_metrics(fixture_path, args.n_frames)

    base_yield = baseline_results.tier1.fish_reconstructed
    base_avail = baseline_results.tier1.fish_available
    base_pct = 100.0 * base_yield / max(base_avail, 1)
    print(
        f"  Fish: {base_yield}/{base_avail} ({base_pct:.0f}%), "
        f"mean error: {baseline_results.tier1.overall_mean_px:.2f} px, "
        f"singleton: {100.0 * baseline_assoc['singleton_rate']:.1f}%"
    )
    _print_camera_distribution(baseline_assoc)

    # ------------------------------------------------------------------
    # Sweep stages (priority order with carry-forward)
    # ------------------------------------------------------------------
    carry_forward: dict[str, float] = {}
    all_stage_rows: list[
        tuple[
            str,
            float,
            list[tuple[float, EvalResults, tuple[float, float], dict[str, object]]],
        ]
    ] = []
    stage_num = 0

    # Primary stages
    for param_name in PRIMARY_STAGES:
        stage_num += 1
        values = SWEEP_RANGES[param_name]
        best_val, rows = _run_sweep_stage(
            stage_num,
            param_name,
            values,
            yaml_config,
            carry_forward,
            args.n_frames,
            args.backend,
            args.outlier_threshold,
        )
        carry_forward[param_name] = best_val
        all_stage_rows.append((param_name, best_val, rows))

        # Check target after each stage
        best_row = rows[0]
        best_fish = best_row[1].tier1.fish_reconstructed
        if best_fish >= args.target_fish * args.n_frames:
            print(
                f"\n  ** Target reached! {best_fish} fish >= "
                f"{args.target_fish * args.n_frames} target "
                f"(skipping secondary parameters) **"
            )
            break

    # Secondary stages (only if target not reached)
    best_overall_fish = all_stage_rows[-1][2][0][1].tier1.fish_reconstructed
    if best_overall_fish < args.target_fish * args.n_frames:
        print(
            f"\n  Target not reached ({best_overall_fish} < "
            f"{args.target_fish * args.n_frames}). "
            f"Sweeping secondary parameters..."
        )
        for param_name in SECONDARY_STAGES:
            stage_num += 1
            values_raw = SECONDARY_RANGES[param_name]
            # Convert to float for consistency
            values = [float(v) for v in values_raw]
            best_val, rows = _run_sweep_stage(
                stage_num,
                param_name,
                values,
                yaml_config,
                carry_forward,
                args.n_frames,
                args.backend,
                args.outlier_threshold,
            )
            carry_forward[param_name] = best_val
            all_stage_rows.append((param_name, best_val, rows))

    # ------------------------------------------------------------------
    # Top-N full evaluation (Tier 1 + Tier 2)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Full Evaluation: Top {args.top_n} candidates (with Tier 2)")
    print("=" * 70)

    # Collect all configs and their scores across all stages
    all_configs: list[tuple[dict[str, float], tuple[float, float]]] = []

    # The winning config uses all carry-forward values
    all_configs.append((dict(carry_forward), all_stage_rows[-1][2][0][2]))

    # Also consider configs from intermediate stages (each with their
    # carry-forward state at that point)
    intermediate_carry: dict[str, float] = {}
    for param_name, best_val, rows in all_stage_rows:
        # Add the non-best values from this stage with current carry
        for val, _results, score, _assoc in rows[1:]:
            config_at_point = dict(intermediate_carry)
            config_at_point[param_name] = val
            all_configs.append((config_at_point, score))
        intermediate_carry[param_name] = best_val

    # Sort by score and take top-N
    all_configs.sort(key=lambda x: x[1])
    top_configs = all_configs[: args.top_n]

    top_results: list[tuple[dict[str, float], EvalResults, dict[str, object]]] = []

    for i, (config, _score) in enumerate(top_configs, 1):
        print(f"\n  [{i}/{len(top_configs)}] Config: {config}")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "midline_fixtures.npz"
            gen_fixture = generate_fixture(
                yaml_config,
                association_overrides=config,
                output_path=output_path,
            )

            eval_output = Path(tmpdir) / "eval"
            eval_output.mkdir()
            results = run_evaluation(
                gen_fixture,
                n_frames=args.n_frames,
                skip_tier2=False,
                backend=args.backend,
                outlier_threshold=args.outlier_threshold,
                output_dir=eval_output,
                association_overrides=config,
            )

            assoc_metrics = _compute_association_metrics(gen_fixture, args.n_frames)

        fish_r = results.tier1.fish_reconstructed
        fish_a = results.tier1.fish_available
        yield_pct = 100.0 * fish_r / max(fish_a, 1)
        print(
            f"    yield={fish_r}/{fish_a} ({yield_pct:.0f}%), "
            f"mean={results.tier1.overall_mean_px:.2f} px"
        )
        if results.tier2.overall_max_displacement is not None:
            print(
                f"    max displacement: "
                f"{results.tier2.overall_max_displacement * 1000.0:.2f} mm"
            )
        top_results.append((config, results, assoc_metrics))

    # ------------------------------------------------------------------
    # Full baseline evaluation (with Tier 2) for comparison
    # ------------------------------------------------------------------
    print("\n  Running full baseline evaluation (with Tier 2)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_full = run_evaluation(
            fixture_path,
            n_frames=args.n_frames,
            skip_tier2=False,
            backend=args.backend,
            outlier_threshold=args.outlier_threshold,
            output_dir=Path(tmpdir),
        )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    if top_results:
        winner_config, winner_results, winner_assoc = top_results[0]
        _print_final_report(
            baseline_full,
            baseline_assoc,
            winner_results,
            winner_assoc,
            winner_config,
        )

    # Per-stage summary
    print("Per-stage best values:")
    for param_name, best_val, _rows in all_stage_rows:
        if isinstance(best_val, float) and best_val == int(best_val):
            print(f"  {param_name}: {int(best_val)}")
        else:
            print(f"  {param_name}: {best_val}")

    print()


if __name__ == "__main__":
    main()
