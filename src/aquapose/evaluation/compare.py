"""Cross-run evaluation comparison with delta computation and table formatting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from aquapose.evaluation.output import _NumpySafeEncoder

LOWER_IS_BETTER: set[str] = {
    "singleton_rate",
    "mean_reprojection_error",
    "max_reprojection_error",
    "p50_reprojection_error",
    "p90_reprojection_error",
    "p95_reprojection_error",
    "total_gaps",
    "mean_gap_duration",
    "max_gap_duration",
    "coast_frequency",
    "low_confidence_flag_rate",
    "mean_jitter",
}

PRIMARY_METRICS: set[tuple[str, str]] = {
    ("association", "singleton_rate"),
    ("reconstruction", "p50_reprojection_error"),
    ("reconstruction", "p90_reprojection_error"),
}


def load_eval_results(run_dir: Path) -> dict[str, Any]:
    """Load eval_results.json from a run directory.

    Args:
        run_dir: Path to the run directory containing eval_results.json.

    Returns:
        Parsed JSON dict.

    Raises:
        FileNotFoundError: If eval_results.json does not exist in run_dir.
    """
    results_path = run_dir / "eval_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No eval_results.json in {run_dir}")
    with results_path.open() as f:
        return json.load(f)


def compute_deltas(
    results_a: dict[str, Any], results_b: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute metric deltas between two eval results.

    Iterates all stages present in both results, extracting scalar (int/float)
    metrics. Dict-valued metrics are skipped.

    Args:
        results_a: Baseline eval results dict (with "stages" key).
        results_b: Comparison eval results dict (with "stages" key).

    Returns:
        Nested dict ``{stage: {metric: {a, b, delta, pct_change, improved, primary}}}``.
        ``pct_change`` is None when the baseline value is near zero.
    """
    stages_a = results_a.get("stages", {})
    stages_b = results_b.get("stages", {})
    out: dict[str, dict[str, dict[str, Any]]] = {}

    for stage in sorted(set(stages_a) & set(stages_b)):
        metrics_a = stages_a[stage]
        metrics_b = stages_b[stage]
        stage_deltas: dict[str, dict[str, Any]] = {}

        for metric in sorted(set(metrics_a) & set(metrics_b)):
            val_a = metrics_a[metric]
            val_b = metrics_b[metric]

            # Skip non-scalar values
            if not isinstance(val_a, (int, float)) or not isinstance(
                val_b, (int, float)
            ):
                continue

            delta = val_b - val_a
            pct_change = None if abs(val_a) < 1e-12 else delta / val_a * 100

            lower_better = metric in LOWER_IS_BETTER
            improved = (delta < 0) if lower_better else (delta > 0)
            is_primary = (stage, metric) in PRIMARY_METRICS

            stage_deltas[metric] = {
                "a": val_a,
                "b": val_b,
                "delta": delta,
                "pct_change": pct_change,
                "improved": improved,
                "primary": is_primary,
            }

        if stage_deltas:
            out[stage] = stage_deltas

    return out


def format_comparison_table(
    results_a: dict[str, Any],
    results_b: dict[str, Any],
    run_a_label: str,
    run_b_label: str,
) -> str:
    """Format a terminal comparison table for two eval results.

    Produces a multi-line table with columns: Stage, Metric, Run A, Run B,
    Delta, %Change. Primary metrics are highlighted with color.

    Args:
        results_a: Baseline eval results dict.
        results_b: Comparison eval results dict.
        run_a_label: Display label for run A.
        run_b_label: Display label for run B.

    Returns:
        Formatted table string with ANSI color codes for terminal display.
    """
    deltas = compute_deltas(results_a, results_b)

    # Build rows
    rows: list[dict[str, str]] = []
    for stage in sorted(deltas):
        for metric in sorted(deltas[stage]):
            d = deltas[stage][metric]
            arrow = (
                "\u2193" if d["delta"] < 0 else ("\u2191" if d["delta"] > 0 else " ")
            )
            delta_str = f"{arrow} {d['delta']:+.4f}"
            pct_str = (
                f"{d['pct_change']:+.1f}%" if d["pct_change"] is not None else "N/A"
            )

            row = {
                "Stage": stage.title(),
                "Metric": metric,
                "Run A": f"{d['a']:.4f}" if isinstance(d["a"], float) else str(d["a"]),
                "Run B": f"{d['b']:.4f}" if isinstance(d["b"], float) else str(d["b"]),
                "Delta": delta_str,
                "%Change": pct_str,
            }

            # Highlight primary metrics
            if d["primary"]:
                fg = "green" if d["improved"] else "red"
                for col in ("Run A", "Run B", "Delta", "%Change"):
                    row[col] = click.style(row[col], fg=fg, bold=True)

            rows.append(row)

    if not rows:
        return "No comparable metrics found."

    columns = ["Stage", "Metric", "Run A", "Run B", "Delta", "%Change"]

    # Compute column widths using unstyled text
    col_widths: dict[str, int] = {}
    for col in columns:
        # Use labels for Run A/B headers
        header = (
            run_a_label if col == "Run A" else (run_b_label if col == "Run B" else col)
        )
        header_len = len(header)
        max_data = max((len(click.unstyle(row[col])) for row in rows), default=0)
        col_widths[col] = max(header_len, max_data)

    # Build header
    header_cells = []
    for col in columns:
        header = (
            run_a_label if col == "Run A" else (run_b_label if col == "Run B" else col)
        )
        header_cells.append(click.style(header.ljust(col_widths[col]), bold=True))
    header_line = "  ".join(header_cells)

    # Build separator
    sep_line = "  ".join("-" * col_widths[col] for col in columns)

    # Build data rows
    lines = [header_line, sep_line]
    for row in rows:
        parts = []
        for col in columns:
            cell = row[col]
            pad = col_widths[col] - len(click.unstyle(cell))
            parts.append(cell + " " * pad)
        lines.append("  ".join(parts))

    return "\n".join(lines)


def write_comparison_json(
    results_a: dict[str, Any],
    results_b: dict[str, Any],
    run_a_path: Path,
    run_b_path: Path,
    output_dir: Path,
) -> Path:
    """Write structured comparison JSON to the output directory.

    Args:
        results_a: Baseline eval results dict.
        results_b: Comparison eval results dict.
        run_a_path: Path to run A directory.
        run_b_path: Path to run B directory.
        output_dir: Directory where eval_comparison.json will be written.

    Returns:
        Path to the written eval_comparison.json file.
    """
    deltas = compute_deltas(results_a, results_b)
    comparison = {
        "run_a": {"run_id": run_a_path.name, "path": str(run_a_path)},
        "run_b": {"run_id": run_b_path.name, "path": str(run_b_path)},
        "metrics": deltas,
    }
    output_path = output_dir / "eval_comparison.json"
    with output_path.open("w") as f:
        json.dump(comparison, f, cls=_NumpySafeEncoder, indent=2)
    return output_path
