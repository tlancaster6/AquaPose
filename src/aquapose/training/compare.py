"""Cross-run comparison table and CSV export for training runs."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def discover_runs(training_type_dir: Path) -> list[Path]:
    """List and sort run_* subdirectories by name.

    Args:
        training_type_dir: Directory containing run directories
            (e.g. ``{project_dir}/training/obb/``).

    Returns:
        Sorted list of ``run_*`` directory paths. Empty list if the
        directory does not exist.
    """
    if not training_type_dir.exists():
        return []
    return sorted(
        p
        for p in training_type_dir.iterdir()
        if p.is_dir() and p.name.startswith("run_")
    )


def load_run_summaries(run_dirs: list[Path]) -> list[dict]:
    """Load summary.json from each run directory.

    Args:
        run_dirs: List of run directory paths.

    Returns:
        List of parsed summary dicts. Runs with missing ``summary.json``
        are skipped with a warning.
    """
    summaries: list[dict] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            logger.warning("No summary.json in %s, skipping", run_dir)
            continue
        with open(summary_path) as f:
            summaries.append(json.load(f))
    return summaries


def _format_duration(seconds: float) -> str:
    """Format duration in seconds as 'Xm Ys'."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def _format_sources(dataset_sources: dict) -> str:
    """Format dataset source breakdown as percentage string."""
    cons_frac = dataset_sources.get("consensus_fraction")
    gap_frac = dataset_sources.get("gap_fraction")
    if cons_frac is None and gap_frac is None:
        return "-"
    parts = []
    if cons_frac is not None:
        parts.append(f"{int(cons_frac * 100)}% cons")
    if gap_frac is not None:
        parts.append(f"{int(gap_frac * 100)}% gap")
    return " / ".join(parts) if parts else "-"


_METRIC_COLS = ("mAP50", "mAP50-95", "Prec", "Recall")


def format_comparison_table(summaries: list[dict]) -> str:
    """Build a terminal comparison table with best-value highlighting.

    Metric columns (mAP50, mAP50-95, Prec, Recall) have their best
    values wrapped in ``click.style(bold=True, fg="green")``.

    Args:
        summaries: List of summary dicts from ``load_run_summaries``.

    Returns:
        Formatted table string ready for terminal output, or
        ``"No runs found."`` if the list is empty.
    """
    if not summaries:
        return "No runs found."

    # Extract rows
    rows: list[dict[str, str]] = []
    metric_values: dict[str, list[float]] = {col: [] for col in _METRIC_COLS}

    for s in summaries:
        metrics = s.get("metrics", {})
        m50 = metrics.get("mAP50", 0.0)
        m50_95 = metrics.get("mAP50-95", 0.0)
        prec = metrics.get("precision", 0.0)
        rec = metrics.get("recall", 0.0)

        metric_values["mAP50"].append(m50)
        metric_values["mAP50-95"].append(m50_95)
        metric_values["Prec"].append(prec)
        metric_values["Recall"].append(rec)

        rows.append(
            {
                "Run": s.get("run_id", "?"),
                "Tag": s.get("tag") or "-",
                "mAP50": f"{m50:.4f}",
                "mAP50-95": f"{m50_95:.4f}",
                "Prec": f"{prec:.4f}",
                "Recall": f"{rec:.4f}",
                "Epoch": str(metrics.get("best_epoch", "-")),
                "Duration": _format_duration(s.get("training_duration_seconds", 0)),
                "Sources": _format_sources(s.get("dataset_sources", {})),
            }
        )

    # Find best values per metric column
    best_indices: dict[str, int] = {}
    for col in _METRIC_COLS:
        vals = metric_values[col]
        if vals:
            best_indices[col] = vals.index(max(vals))

    # Highlight best values
    for col in _METRIC_COLS:
        best_idx = best_indices.get(col)
        if best_idx is not None:
            rows[best_idx][col] = click.style(
                rows[best_idx][col], fg="green", bold=True
            )

    # Build table
    columns = [
        "Run",
        "Tag",
        "mAP50",
        "mAP50-95",
        "Prec",
        "Recall",
        "Epoch",
        "Duration",
        "Sources",
    ]

    # Compute column widths (strip ANSI for width calculation)
    col_widths: dict[str, int] = {}
    for col in columns:
        header_len = len(col)
        max_data = max((len(click.unstyle(row[col])) for row in rows), default=0)
        col_widths[col] = max(header_len, max_data)

    # Format header
    header_parts = [
        click.style(col.ljust(col_widths[col]), bold=True) for col in columns
    ]
    header = "  ".join(header_parts)

    # Format data rows
    lines = [header]
    for row in rows:
        parts = []
        for col in columns:
            cell = row[col]
            # Pad based on unstyled width
            pad = col_widths[col] - len(click.unstyle(cell))
            parts.append(cell + " " * pad)
        lines.append("  ".join(parts))

    return "\n".join(lines)


def write_comparison_csv(summaries: list[dict], output_path: Path) -> None:
    """Write comparison data to CSV without ANSI codes.

    Args:
        summaries: List of summary dicts from ``load_run_summaries``.
        output_path: Path to write the CSV file.
    """
    columns = [
        "Run",
        "Tag",
        "mAP50",
        "mAP50-95",
        "Prec",
        "Recall",
        "Epoch",
        "Duration",
        "Sources",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for s in summaries:
            metrics = s.get("metrics", {})
            writer.writerow(
                {
                    "Run": s.get("run_id", "?"),
                    "Tag": s.get("tag") or "-",
                    "mAP50": f"{metrics.get('mAP50', 0.0):.4f}",
                    "mAP50-95": f"{metrics.get('mAP50-95', 0.0):.4f}",
                    "Prec": f"{metrics.get('precision', 0.0):.4f}",
                    "Recall": f"{metrics.get('recall', 0.0):.4f}",
                    "Epoch": str(metrics.get("best_epoch", "-")),
                    "Duration": _format_duration(s.get("training_duration_seconds", 0)),
                    "Sources": _format_sources(s.get("dataset_sources", {})),
                }
            )
