"""Output formatting for evaluation results.

Provides flag_outliers for statistical outlier detection, format_baseline_report
for outlier-annotated baseline reports, and format_eval_report / format_eval_json
for multi-stage EvalRunner results.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from aquapose.evaluation.metrics import Tier1Result, Tier2Result

if TYPE_CHECKING:
    from aquapose.evaluation.runner import EvalRunnerResult


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
    if tier1.fish_available > 0:
        rate = tier1.fish_reconstructed / tier1.fish_available * 100
        lines.append(
            f"Reconstruction rate: {tier1.fish_reconstructed}/{tier1.fish_available} "
            f"fish-frames ({rate:.0f}%)"
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


def format_eval_report(result: EvalRunnerResult) -> str:
    """Format an EvalRunnerResult as a multi-stage human-readable ASCII report.

    Produces a header block with run metadata followed by a summary section
    listing each present stage's primary metric, and then per-stage detail
    sections with full metric tables.

    Args:
        result: Aggregated evaluation result from EvalRunner.run().

    Returns:
        Multi-line ASCII string suitable for printing to stdout.
    """
    lines: list[str] = []
    _SEP = "=" * 50
    _DASH = "-" * 50
    _W_METRIC = 30
    _W_VALUE = 12

    # ------------------------------------------------------------------
    # Header block
    # ------------------------------------------------------------------
    lines.append("Evaluation Report")
    lines.append(_SEP)
    lines.append(
        f"Run: {result.run_id}  |  Frames: {result.frames_evaluated}/{result.frames_available}"
    )
    stages_sorted = sorted(result.stages_present)
    if stages_sorted:
        lines.append(f"Stages: {', '.join(stages_sorted)}")
    lines.append("")

    # ------------------------------------------------------------------
    # Summary: one line per present stage with primary metric
    # ------------------------------------------------------------------
    has_summary = False
    if result.detection is not None:
        d = result.detection
        lines.append(
            f"  Detection:       {d.total_detections} detections,"
            f" mean conf {d.mean_confidence:.3f}"
        )
        has_summary = True
    if result.tracking is not None:
        t = result.tracking
        lines.append(
            f"  Tracking:        {t.track_count} tracks,"
            f" {t.detection_coverage:.1%} coverage"
        )
        has_summary = True
    if result.association is not None:
        a = result.association
        lines.append(
            f"  Association:     yield {a.fish_yield_ratio:.1%},"
            f" singleton {a.singleton_rate:.1%}"
        )
        has_summary = True
    if result.midline is not None:
        m = result.midline
        lines.append(
            f"  Midline:         {m.total_midlines} midlines,"
            f" mean conf {m.mean_confidence:.3f}"
        )
        has_summary = True
    if result.reconstruction is not None:
        r = result.reconstruction
        lines.append(
            f"  Reconstruction:  mean reproj {r.mean_reprojection_error:.2f} px"
        )
        has_summary = True

    if has_summary:
        lines.append("")

    # ------------------------------------------------------------------
    # Per-stage detail sections
    # ------------------------------------------------------------------
    def _row(metric: str, value: object) -> str:
        val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
        return f"  {metric:<{_W_METRIC}} {val_str:>{_W_VALUE}}"

    def _header(name: str) -> list[str]:
        return [
            name,
            _DASH,
            f"  {'Metric':<{_W_METRIC}} {'Value':>{_W_VALUE}}",
        ]

    if result.detection is not None:
        lines.extend(_header("Detection"))
        d = result.detection
        lines.append(_row("total_detections", d.total_detections))
        lines.append(_row("mean_confidence", d.mean_confidence))
        lines.append(_row("std_confidence", d.std_confidence))
        lines.append(_row("mean_jitter", d.mean_jitter))
        lines.append("  per_camera_counts:")
        for cam_id, count in sorted(d.per_camera_counts.items()):
            lines.append(f"    {cam_id:<{_W_METRIC - 2}} {count:>{_W_VALUE}}")
        lines.append("")

    if result.tracking is not None:
        lines.extend(_header("Tracking"))
        t = result.tracking
        lines.append(_row("track_count", t.track_count))
        lines.append(_row("length_median", t.length_median))
        lines.append(_row("length_mean", t.length_mean))
        lines.append(_row("length_min", t.length_min))
        lines.append(_row("length_max", t.length_max))
        lines.append(_row("coast_frequency", t.coast_frequency))
        lines.append(_row("detection_coverage", t.detection_coverage))
        lines.append("")

    if result.association is not None:
        lines.extend(_header("Association"))
        a = result.association
        lines.append(_row("fish_yield_ratio", a.fish_yield_ratio))
        lines.append(_row("singleton_rate", a.singleton_rate))
        lines.append(_row("total_fish_observations", a.total_fish_observations))
        lines.append(_row("frames_evaluated", a.frames_evaluated))
        lines.append("  camera_distribution:")
        for n_cams, count in sorted(a.camera_distribution.items()):
            lines.append(f"    {n_cams} camera(s):{count:>{_W_VALUE}}")
        lines.append("")

    if result.midline is not None:
        lines.extend(_header("Midline"))
        m = result.midline
        lines.append(_row("total_midlines", m.total_midlines))
        lines.append(_row("mean_confidence", m.mean_confidence))
        lines.append(_row("std_confidence", m.std_confidence))
        lines.append(_row("completeness", m.completeness))
        lines.append(_row("temporal_smoothness", m.temporal_smoothness))
        lines.append("")

    if result.reconstruction is not None:
        lines.extend(_header("Reconstruction"))
        r = result.reconstruction
        lines.append("")
        lines.append("  Tier 1: Reprojection Error")
        lines.append("  " + "-" * 30)
        lines.append(_row("mean_reprojection_error", r.mean_reprojection_error))
        lines.append(_row("max_reprojection_error", r.max_reprojection_error))
        lines.append(_row("fish_reconstructed", r.fish_reconstructed))
        lines.append(_row("fish_available", r.fish_available))
        lines.append(_row("inlier_ratio", r.inlier_ratio))
        lines.append(_row("low_confidence_flag_rate", r.low_confidence_flag_rate))
        lines.append("  per_camera_error:")
        for cam_id, stats in sorted(r.per_camera_error.items()):
            lines.append(
                f"    {cam_id:<{_W_METRIC - 2}}"
                f" mean={stats['mean_px']:.2f} max={stats['max_px']:.2f}"
            )
        if r.tier2_stability is not None:
            lines.append("")
            lines.append("  Tier 2: Leave-One-Out Stability")
            lines.append("  " + "-" * 30)
            lines.append(_row("tier2_stability (m)", r.tier2_stability))
        lines.append("")

    return "\n".join(lines)


def format_eval_json(result: EvalRunnerResult) -> str:
    """Format an EvalRunnerResult as a JSON string.

    Uses _NumpySafeEncoder to handle numpy scalar types that may be present
    in metric values. The JSON schema matches result.to_dict().

    Args:
        result: Aggregated evaluation result from EvalRunner.run().

    Returns:
        JSON string with run_id, stages_present, frames_evaluated,
        frames_available, and a "stages" dict mapping stage names to
        their metric dicts. Absent stages are omitted from "stages".
    """
    return json.dumps(result.to_dict(), cls=_NumpySafeEncoder, indent=2)
