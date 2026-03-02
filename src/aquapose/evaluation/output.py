"""Output formatting for evaluation results: ASCII summary table and regression JSON.

Provides format_summary_table for human-readable output, write_regression_json
for machine-diffable regression data, flag_outliers for statistical outlier
detection, and format_baseline_report for outlier-annotated baseline reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aquapose.evaluation.metrics import Tier1Result, Tier2Result


class _NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalar types to Python builtins."""

    def default(self, obj: Any) -> Any:
        """Convert numpy scalars to native Python types.

        Args:
            obj: Object to serialize.

        Returns:
            Python-native equivalent or delegates to super.
        """
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


def format_summary_table(
    tier1: Tier1Result,
    tier2: Tier2Result,
    fixture_name: str,
    frames_evaluated: int,
    frames_available: int,
) -> str:
    """Format evaluation results as a human-readable ASCII summary table.

    Produces a multi-line string with a Tier 1 reprojection error section
    (per-camera breakdown) and a Tier 2 leave-one-out stability section
    (per-fish dropout table).

    Args:
        tier1: Tier 1 aggregated reprojection error metrics.
        tier2: Tier 2 aggregated leave-one-out displacement metrics.
        fixture_name: Name of the fixture file (for display).
        frames_evaluated: Number of frames evaluated.
        frames_available: Total frames available in the fixture.

    Returns:
        Multi-line ASCII string suitable for printing to stdout.
    """
    lines: list[str] = []

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    lines.append("Evaluation Summary")
    lines.append("==================")
    lines.append(
        f"Fixture: {fixture_name}  |  Frames evaluated: {frames_evaluated} / {frames_available}"
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Tier 1: Reprojection Error
    # ------------------------------------------------------------------
    lines.append("Tier 1: Reprojection Error (pixels)")
    lines.append("-" * 50)
    lines.append(f"{'Camera':<16} {'Mean':>8} {'Max':>8}")
    lines.append(f"{'------':<16} {'----':>8} {'---':>8}")

    for cam_id, cam_stats in sorted(tier1.per_camera.items()):
        mean_px = cam_stats["mean_px"]
        max_px = cam_stats["max_px"]
        lines.append(f"{cam_id:<16} {mean_px:>8.2f} {max_px:>8.2f}")

    lines.append(
        f"{'OVERALL':<16} {tier1.overall_mean_px:>8.2f} {tier1.overall_max_px:>8.2f}"
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Tier 2: Leave-One-Out Stability
    # ------------------------------------------------------------------
    lines.append("Tier 2: Leave-One-Out Stability")
    lines.append("-" * 50)
    lines.append(f"{'Fish':<8} {'Dropout Camera':<18} {'Max Disp (mm)':>14}")
    lines.append(f"{'----':<8} {'--------------':<18} {'-------------':>14}")

    for fish_id in sorted(tier2.per_fish_dropout.keys()):
        dropout_dict = tier2.per_fish_dropout[fish_id]
        for cam_id in sorted(dropout_dict.keys()):
            displacement = dropout_dict[cam_id]
            if displacement is None:
                disp_str = "N/A"
            else:
                # Convert metres to millimetres for display
                disp_mm = displacement * 1000.0
                disp_str = f"{disp_mm:.3f}"
            lines.append(f"{fish_id:<8} {cam_id:<18} {disp_str:>14}")

    lines.append("")

    return "\n".join(lines)


def flag_outliers(values: dict[str, float], threshold_std: float = 2.0) -> set[str]:
    """Return keys whose values exceed mean + threshold_std * std.

    Args:
        values: Mapping of key to numeric value to analyse.
        threshold_std: Number of standard deviations above the mean to use as
            the outlier threshold. Defaults to 2.0.

    Returns:
        Set of keys identified as outliers. Returns an empty set when fewer
        than 2 values are provided or when the standard deviation is zero.
    """
    if len(values) < 2:
        return set()
    vals = list(values.values())
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if std == 0.0:
        return set()
    threshold = mean + threshold_std * std
    return {k for k, v in values.items() if v > threshold}


def format_baseline_report(
    tier1: Tier1Result,
    tier2: Tier2Result,
    fixture_name: str,
    frames_evaluated: int,
    frames_available: int,
) -> str:
    """Format evaluation results as an outlier-annotated baseline report.

    Produces the same sections as format_summary_table but marks entries that
    exceed 2 standard deviations from the mean with an asterisk (``*``).

    Outlier detection is applied to:

    * Tier 1 per-camera mean_px values.
    * Tier 1 per-fish mean_px values.
    * Tier 2 per-fish per-dropout-camera displacement values (within each fish).

    A legend line ``* = outlier (>2 std from mean)`` is appended at the bottom.

    Args:
        tier1: Tier 1 aggregated reprojection error metrics.
        tier2: Tier 2 aggregated leave-one-out displacement metrics.
        fixture_name: Name of the fixture file (for display).
        frames_evaluated: Number of frames evaluated.
        frames_available: Total frames available in the fixture.

    Returns:
        Multi-line ASCII string with outlier markers, suitable for printing
        to stdout or saving as a text report.
    """
    lines: list[str] = []

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    lines.append("Baseline Evaluation Report")
    lines.append("==========================")
    lines.append(
        f"Fixture: {fixture_name}  |  Frames evaluated: {frames_evaluated} / {frames_available}"
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Tier 1: Reprojection Error (with per-camera outlier flagging)
    # ------------------------------------------------------------------
    cam_outliers = flag_outliers(
        {cam: float(stats["mean_px"]) for cam, stats in tier1.per_camera.items()}
    )
    fish_outliers = flag_outliers(
        {str(fid): float(stats["mean_px"]) for fid, stats in tier1.per_fish.items()}
    )

    lines.append("Tier 1: Reprojection Error (pixels)")
    lines.append("-" * 50)
    lines.append(f"{'Camera':<16} {'Mean':>8} {'Max':>8}")
    lines.append(f"{'------':<16} {'----':>8} {'---':>8}")

    for cam_id, cam_stats in sorted(tier1.per_camera.items()):
        mean_px = cam_stats["mean_px"]
        max_px = cam_stats["max_px"]
        marker = " *" if cam_id in cam_outliers else ""
        lines.append(f"{cam_id:<16} {mean_px:>8.2f} {max_px:>8.2f}{marker}")

    lines.append(
        f"{'OVERALL':<16} {tier1.overall_mean_px:>8.2f} {tier1.overall_max_px:>8.2f}"
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Per-fish breakdown
    # ------------------------------------------------------------------
    lines.append("Tier 1: Per-Fish Breakdown (pixels)")
    lines.append("-" * 50)
    lines.append(f"{'Fish':<8} {'Mean':>8} {'Max':>8}")
    lines.append(f"{'----':<8} {'----':>8} {'---':>8}")

    for fish_id in sorted(tier1.per_fish.keys()):
        stats = tier1.per_fish[fish_id]
        mean_px = stats["mean_px"]
        max_px = stats["max_px"]
        marker = " *" if str(fish_id) in fish_outliers else ""
        lines.append(f"{fish_id:<8} {mean_px:>8.2f} {max_px:>8.2f}{marker}")

    lines.append("")

    # ------------------------------------------------------------------
    # Tier 2: Leave-One-Out Stability (with per-fish outlier flagging)
    # ------------------------------------------------------------------
    lines.append("Tier 2: Leave-One-Out Stability")
    lines.append("-" * 50)
    lines.append(f"{'Fish':<8} {'Dropout Camera':<18} {'Max Disp (mm)':>14}")
    lines.append(f"{'----':<8} {'--------------':<18} {'-------------':>14}")

    for fish_id in sorted(tier2.per_fish_dropout.keys()):
        dropout_dict = tier2.per_fish_dropout[fish_id]
        # Compute outliers among non-None displacement values for this fish
        non_none: dict[str, float] = {
            cam: float(val) for cam, val in dropout_dict.items() if val is not None
        }
        tier2_outliers = flag_outliers(non_none)

        for cam_id in sorted(dropout_dict.keys()):
            displacement = dropout_dict[cam_id]
            if displacement is None:
                disp_str = "N/A"
                marker = ""
            else:
                disp_mm = float(displacement) * 1000.0
                disp_str = f"{disp_mm:.3f}"
                marker = " *" if cam_id in tier2_outliers else ""
            lines.append(f"{fish_id:<8} {cam_id:<18} {disp_str:>14}{marker}")

    lines.append("")
    lines.append("* = outlier (>2\u03c3 from mean)")
    lines.append("")

    return "\n".join(lines)


def write_regression_json(
    tier1: Tier1Result,
    tier2: Tier2Result,
    fixture_name: str,
    frames_evaluated: int,
    frames_available: int,
    output_path: Path,
) -> Path:
    """Write evaluation results to a machine-diffable JSON regression file.

    The JSON contains only aggregated metrics (no per-frame detail). All
    numpy scalar types are converted to Python builtins for clean diffs.
    None values for failed Tier 2 dropouts are serialized as JSON null.

    Args:
        tier1: Tier 1 aggregated reprojection error metrics.
        tier2: Tier 2 aggregated leave-one-out displacement metrics.
        fixture_name: Name of the fixture file.
        frames_evaluated: Number of frames evaluated.
        frames_available: Total frames available in the fixture.
        output_path: Path where the JSON file will be written.

    Returns:
        The output_path (for chaining / confirmation).
    """
    # Build Tier 2 structure: fish_id (str key for JSON) -> cam_id -> value or null
    tier2_per_fish: dict[str, dict[str, float | None]] = {}
    for fish_id, dropout_dict in tier2.per_fish_dropout.items():
        fish_key = str(fish_id)
        tier2_per_fish[fish_key] = {}
        for cam_id, displacement in dropout_dict.items():
            if displacement is None:
                tier2_per_fish[fish_key][cam_id] = None
            else:
                tier2_per_fish[fish_key][cam_id] = float(displacement)

    # Build per_fish Tier 1: fish_id (str key) -> stats
    tier1_per_fish: dict[str, dict[str, float]] = {
        str(fid): {"mean_px": float(stats["mean_px"]), "max_px": float(stats["max_px"])}
        for fid, stats in tier1.per_fish.items()
    }

    # Build per_camera Tier 1
    tier1_per_camera: dict[str, dict[str, float]] = {
        cam_id: {"mean_px": float(stats["mean_px"]), "max_px": float(stats["max_px"])}
        for cam_id, stats in tier1.per_camera.items()
    }

    result: dict[str, Any] = {
        "fixture": fixture_name,
        "frames_evaluated": int(frames_evaluated),
        "frames_available": int(frames_available),
        "tier1": {
            "overall_mean_px": float(tier1.overall_mean_px),
            "overall_max_px": float(tier1.overall_max_px),
            "per_camera": tier1_per_camera,
            "per_fish": tier1_per_fish,
        },
        "tier2": {
            "per_fish_dropout": tier2_per_fish,
        },
    }

    output_path = Path(output_path)
    with output_path.open("w") as f:
        json.dump(result, f, cls=_NumpySafeEncoder, indent=2)

    return output_path
