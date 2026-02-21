"""Diagnostic Markdown report generator for reconstruction pipeline runs.

Generates a self-contained Markdown document with stage timing, reconstruction
statistics, per-frame fish count summaries, and embedded figure references.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aquapose.reconstruction.triangulation import Midline3D

logger = logging.getLogger(__name__)

# Relative figure paths checked when building embedded references
_FIGURE_PATHS: list[str] = [
    "figures/detection_sample.png",
    "figures/segmentation_sample.png",
    "figures/midline_3d_sample.png",
    "figures/overlay_sample.png",
]


def write_diagnostic_report(
    output_dir: Path,
    stage_timing: dict[str, float],
    midlines_per_frame: list[dict[int, Midline3D]],
    *,
    n_frames: int,
    n_cameras: int,
    mode: str = "diagnostic",
) -> Path:
    """Write a diagnostic Markdown report to ``output_dir/report.md``.

    Builds a self-contained report with:

    1. Header with date, mode, n_frames, n_cameras.
    2. Stage timing table (stage name, wall time, percentage of total).
    3. Reconstruction summary (total fish tracked, mean/max residuals,
       percentage of low-confidence midlines).
    4. Per-frame fish count statistics (min, max, mean, std).
    5. Embedded figure references for any PNG figures that exist under
       ``output_dir/figures/``.

    All figure paths are relative to ``output_dir`` for portability.

    Args:
        output_dir: Directory where ``report.md`` will be written and where
            figure files may exist.
        stage_timing: Wall-clock seconds for each stage, keyed by stage name.
        midlines_per_frame: Per-frame reconstruction results. Each entry maps
            fish_id to Midline3D.
        n_frames: Total number of processed frames.
        n_cameras: Total number of active cameras.
        mode: Pipeline mode string embedded in the report header.

    Returns:
        Path to the written ``report.md`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "report.md"

    # --- Timing statistics ---
    total_time = sum(stage_timing.values())
    timing_rows: list[str] = []
    for stage_name, wall_time in stage_timing.items():
        pct = (wall_time / total_time * 100) if total_time > 0 else 0.0
        timing_rows.append(f"| {stage_name} | {wall_time:.3f} | {pct:.1f}% |")

    # --- Reconstruction statistics ---
    all_midlines: list[Midline3D] = [
        m for frame in midlines_per_frame for m in frame.values()
    ]
    all_fish_ids: set[int] = {m.fish_id for m in all_midlines}
    n_fish_total = len(all_fish_ids)

    if all_midlines:
        mean_residual_global = float(np.mean([m.mean_residual for m in all_midlines]))
        max_residual_global = float(np.max([m.max_residual for m in all_midlines]))
        n_low_conf = sum(1 for m in all_midlines if m.is_low_confidence)
        pct_low_conf = n_low_conf / len(all_midlines) * 100
    else:
        mean_residual_global = 0.0
        max_residual_global = 0.0
        pct_low_conf = 0.0
        n_low_conf = 0

    # --- Per-frame fish counts ---
    fish_counts: list[int] = [len(frame) for frame in midlines_per_frame]
    if fish_counts:
        fc_arr = np.array(fish_counts, dtype=np.float64)
        fc_min = int(fc_arr.min())
        fc_max = int(fc_arr.max())
        fc_mean = float(fc_arr.mean())
        fc_std = float(fc_arr.std())
    else:
        fc_min = fc_max = 0
        fc_mean = fc_std = 0.0

    # --- Figure references (relative paths, only if files exist) ---
    fig_lines: list[str] = []
    for rel_path in _FIGURE_PATHS:
        abs_path = output_dir / rel_path
        if abs_path.exists():
            # Use forward slashes for Markdown portability
            fig_lines.append(f"![{rel_path}]({rel_path})")

    figs_section = (
        "\n".join(fig_lines) if fig_lines else "_No figures found in output_dir._"
    )

    # --- Assemble report ---
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    report_lines: list[str] = [
        "# AquaPose Reconstruction Diagnostic Report",
        "",
        f"**Date:** {now}  ",
        f"**Mode:** {mode}  ",
        f"**Frames processed:** {n_frames}  ",
        f"**Cameras:** {n_cameras}  ",
        "",
        "---",
        "",
        "## Stage Timing",
        "",
        "| Stage | Wall Time (s) | % of Total |",
        "| ----- | ------------- | ---------- |",
        *timing_rows,
        f"| **Total** | **{total_time:.3f}** | **100%** |",
        "",
        "---",
        "",
        "## Reconstruction Summary",
        "",
        f"- **Total unique fish tracked:** {n_fish_total}",
        f"- **Mean reprojection residual:** {mean_residual_global:.2f} px",
        f"- **Max reprojection residual:** {max_residual_global:.2f} px",
        f"- **Low-confidence midlines:** {n_low_conf} / {len(all_midlines)}"
        f" ({pct_low_conf:.1f}%)",
        "",
        "---",
        "",
        "## Per-Frame Fish Count Statistics",
        "",
        f"- **Min:** {fc_min}",
        f"- **Max:** {fc_max}",
        f"- **Mean:** {fc_mean:.2f}",
        f"- **Std:** {fc_std:.2f}",
        "",
        "---",
        "",
        "## Figures",
        "",
        figs_section,
        "",
    ]

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Diagnostic report written to %s", report_path)
    return report_path
