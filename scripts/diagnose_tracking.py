"""Diagnostic script for benchmarking the Cross-View Identity and 3D Tracking pipeline.

Generates a synthetic scenario using the quick-6 synthetic data system, runs
FishTracker frame-by-frame, computes quantitative CLEAR MOT-inspired tracking
metrics by comparing tracker output to synthetic ground truth, and produces
four diagnostic visualizations.

Usage:
    python scripts/diagnose_tracking.py
    python scripts/diagnose_tracking.py --scenario crossing_paths --difficulty 0.7
    python scripts/diagnose_tracking.py --scenario tight_schooling --n-fish 7
    python scripts/diagnose_tracking.py --scenario track_fragmentation --miss-rate 0.4
    python scripts/diagnose_tracking.py --output-dir output/tracking_diagnostic
    python scripts/diagnose_tracking.py --n-cameras 4
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Tracking Metrics Dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrackingMetrics:
    """CLEAR MOT-inspired tracking metrics computed against synthetic GT.

    Attributes:
        n_gt: Total GT object-frames (n_fish * n_frames where fish is in tank).
        true_positives: Frames where a GT fish has a matched confirmed track.
        false_negatives: Frames where a GT fish has no matched confirmed track.
        false_positives: Confirmed track-frames with no GT match.
        id_switches: Number of times a track changes its GT fish assignment.
        fragmentation: Times a GT fish track is interrupted (matched → gap → matched).
        mota: 1 - (FN + FP + ID_switches) / n_gt, clamped to [-1, 1].
        track_purity: Per-track fraction of frames mapping to most-frequent GT fish.
        mean_track_purity: Mean of track_purity values.
        mostly_tracked: GT fish tracked for >= 80% of their lifetime.
        mostly_lost: GT fish tracked for <= 20% of their lifetime.
        n_confirmed_tracks: Total number of distinct confirmed track IDs seen.
        per_fish_tracked_fraction: Dict gt_fish_id -> fraction of frames tracked.
    """

    n_gt: int = 0
    true_positives: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    id_switches: int = 0
    fragmentation: int = 0
    mota: float = 0.0
    track_purity: dict[int, float] = field(default_factory=dict)
    mean_track_purity: float = 0.0
    mostly_tracked: int = 0
    mostly_lost: int = 0
    n_confirmed_tracks: int = 0
    per_fish_tracked_fraction: dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GT Matching
# ---------------------------------------------------------------------------


def _match_gt_to_tracks(
    frame_gt_positions: np.ndarray,
    frame_gt_fish_ids: list[int],
    frame_track_positions: list[np.ndarray],
    frame_track_ids: list[int],
    match_threshold: float = 0.15,
) -> dict[int, int | None]:
    """Match tracker fish_ids to GT fish_ids for one frame.

    Uses greedy nearest-neighbour assignment: compute pairwise 3D distances,
    assign closest pairs first. A match is valid if distance < match_threshold.

    Args:
        frame_gt_positions: GT positions, shape (n_gt, 3).
        frame_gt_fish_ids: GT fish integer IDs, length n_gt.
        frame_track_positions: Tracker positions, list of shape-(3,) arrays.
        frame_track_ids: Tracker fish_ids, same length as frame_track_positions.
        match_threshold: Maximum 3D distance (metres) for a valid match.

    Returns:
        Dict mapping track_fish_id -> gt_fish_id (or None if unmatched).
    """
    result: dict[int, int | None] = {tid: None for tid in frame_track_ids}

    if len(frame_gt_fish_ids) == 0 or len(frame_track_ids) == 0:
        return result

    gt_positions = frame_gt_positions  # (n_gt, 3)
    track_positions = np.stack(frame_track_positions, axis=0)  # (n_tracks, 3)

    # Pairwise distances: (n_gt, n_tracks)
    diffs = gt_positions[:, np.newaxis, :] - track_positions[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)  # (n_gt, n_tracks)

    assigned_gt: set[int] = set()
    assigned_track: set[int] = set()

    while True:
        # Find global minimum unassigned pair
        if len(assigned_gt) == len(frame_gt_fish_ids) or len(assigned_track) == len(
            frame_track_ids
        ):
            break

        min_val = np.inf
        min_gi = -1
        min_ti = -1
        for gi in range(len(frame_gt_fish_ids)):
            if gi in assigned_gt:
                continue
            for ti in range(len(frame_track_ids)):
                if ti in assigned_track:
                    continue
                if dists[gi, ti] < min_val:
                    min_val = dists[gi, ti]
                    min_gi = gi
                    min_ti = ti

        if min_val > match_threshold or min_gi < 0:
            break

        assigned_gt.add(min_gi)
        assigned_track.add(min_ti)
        result[frame_track_ids[min_ti]] = frame_gt_fish_ids[min_gi]

    return result


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_tracking_metrics(
    gt_matching_per_frame: list[dict[int, int | None]],
    gt_positions_per_frame: list[np.ndarray],
    gt_fish_ids_per_frame: list[list[int]],
    n_gt_fish: int,
) -> TrackingMetrics:
    """Compute CLEAR MOT-inspired tracking metrics from per-frame GT matching.

    Args:
        gt_matching_per_frame: Per-frame mapping of track_fish_id -> gt_fish_id
            (or None if unmatched). Length n_frames.
        gt_positions_per_frame: Per-frame GT positions (n_gt, 3) arrays.
        gt_fish_ids_per_frame: Per-frame list of GT fish IDs present.
        n_gt_fish: Total number of ground truth fish.

    Returns:
        TrackingMetrics with all computed fields.
    """
    n_frames = len(gt_matching_per_frame)
    metrics = TrackingMetrics()

    # Total GT object-frames
    metrics.n_gt = sum(len(ids) for ids in gt_fish_ids_per_frame)

    # For each GT fish, track which track_id has been assigned to it per frame
    # gt_fish_id -> list of (track_id | None) per frame
    gt_coverage: dict[int, list[int | None]] = {
        fid: [None] * n_frames for fid in range(n_gt_fish)
    }

    # Per-track: which gt_fish_id does it map to each frame?
    # track_id -> list of (gt_fish_id | None) per frame
    track_gt_history: dict[int, list[int | None]] = {}

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for frame_idx, frame_matching in enumerate(gt_matching_per_frame):
        gt_ids_this_frame = set(gt_fish_ids_per_frame[frame_idx])
        matched_gt_ids: set[int] = set()

        for track_id, gt_id in frame_matching.items():
            if track_id not in track_gt_history:
                track_gt_history[track_id] = [None] * n_frames
            track_gt_history[track_id][frame_idx] = gt_id

            if gt_id is not None and gt_id in gt_ids_this_frame:
                true_positives += 1
                matched_gt_ids.add(gt_id)
                gt_coverage[gt_id][frame_idx] = track_id
            else:
                # Track has no GT match — false positive
                false_positives += 1

        # GT fish with no matched track in this frame
        for gt_id in gt_ids_this_frame:
            if gt_id not in matched_gt_ids:
                false_negatives += 1

    metrics.true_positives = true_positives
    metrics.false_positives = false_positives
    metrics.false_negatives = false_negatives
    metrics.n_confirmed_tracks = len(track_gt_history)

    # ID switches: per track, how many times does its GT assignment change
    # (not counting None -> GT_id or GT_id -> None, only actual identity changes)
    id_switches = 0
    for _track_id, gt_hist in track_gt_history.items():
        prev_gt: int | None = None
        for gt_id in gt_hist:
            if gt_id is not None:
                if prev_gt is not None and gt_id != prev_gt:
                    id_switches += 1
                prev_gt = gt_id
    metrics.id_switches = id_switches

    # Fragmentation: per GT fish, count transitions from matched to unmatched
    # (i.e., number of track gaps)
    fragmentation = 0
    for _gt_id, coverage in gt_coverage.items():
        was_tracked = False
        was_lost = False
        for track_id in coverage:
            if track_id is not None:
                if was_lost:
                    fragmentation += 1  # re-acquired after gap
                    was_lost = False
                was_tracked = True
            else:
                if was_tracked:
                    was_lost = True
    metrics.fragmentation = fragmentation

    # MOTA: 1 - (FN + FP + ID_switches) / n_gt
    if metrics.n_gt > 0:
        mota_raw = 1.0 - (
            (metrics.false_negatives + metrics.false_positives + metrics.id_switches)
            / metrics.n_gt
        )
        metrics.mota = float(np.clip(mota_raw, -1.0, 1.0))
    else:
        metrics.mota = 0.0

    # Track purity: for each track, fraction of frames mapping to most frequent GT fish
    purity_vals: list[float] = []
    for track_id, gt_hist in track_gt_history.items():
        # Count matched frames only
        matched = [g for g in gt_hist if g is not None]
        if not matched:
            metrics.track_purity[track_id] = 0.0
        else:
            from collections import Counter

            counts = Counter(matched)
            most_common_count = counts.most_common(1)[0][1]
            purity = most_common_count / len(matched)
            metrics.track_purity[track_id] = purity
            purity_vals.append(purity)
    metrics.mean_track_purity = float(np.mean(purity_vals)) if purity_vals else 0.0

    # Mostly tracked / mostly lost per GT fish
    mostly_tracked = 0
    mostly_lost = 0
    per_fish_tracked_fraction: dict[int, float] = {}

    for gt_id in range(n_gt_fish):
        gt_frames_present = sum(
            1 for frame_idx, ids in enumerate(gt_fish_ids_per_frame) if gt_id in ids
        )
        if gt_frames_present == 0:
            per_fish_tracked_fraction[gt_id] = 0.0
            continue
        tracked_frames = sum(
            1
            for frame_idx in range(n_frames)
            if gt_coverage[gt_id][frame_idx] is not None
        )
        frac = tracked_frames / gt_frames_present
        per_fish_tracked_fraction[gt_id] = frac
        if frac >= 0.8:
            mostly_tracked += 1
        elif frac <= 0.2:
            mostly_lost += 1

    metrics.mostly_tracked = mostly_tracked
    metrics.mostly_lost = mostly_lost
    metrics.per_fish_tracked_fraction = per_fish_tracked_fraction

    return metrics


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def vis_3d_trajectories(
    gt_states: np.ndarray,
    tracker_positions_per_frame: list[dict[int, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot GT vs tracked 3D trajectories.

    Args:
        gt_states: shape (n_frames, n_fish, 3) — GT XYZ positions.
        tracker_positions_per_frame: Per-frame dict of track_fish_id -> (3,) array.
        output_path: Output PNG path.
    """
    import matplotlib.pyplot as plt

    # Convert BGR FISH_COLORS to RGB floats for matplotlib
    from aquapose.visualization.overlay import FISH_COLORS

    rgb_colors = [(b / 255.0, g / 255.0, r / 255.0) for b, g, r in FISH_COLORS]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    _n_frames, n_fish, _ = gt_states.shape

    # GT trajectories as solid lines
    for fish_idx in range(n_fish):
        color = rgb_colors[fish_idx % len(rgb_colors)]
        xs = gt_states[:, fish_idx, 0]
        ys = gt_states[:, fish_idx, 1]
        zs = gt_states[:, fish_idx, 2]
        ax.plot(xs, ys, zs, color=color, linewidth=2, label=f"GT fish {fish_idx}")
        ax.scatter(xs[0], ys[0], zs[0], color=color, s=50, marker="o")

    # Tracker positions as scatter markers
    track_ids_seen: set[int] = set()
    for frame_positions in tracker_positions_per_frame:
        for track_id, pos in frame_positions.items():
            color = rgb_colors[track_id % len(rgb_colors)]
            ax.scatter(
                pos[0],
                pos[1],
                pos[2],
                color=color,
                s=15,
                marker="x",
                alpha=0.5,
            )
            track_ids_seen.add(track_id)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("GT vs Tracked 3D Trajectories")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close("all")


def vis_detection_overlay_grid(
    dataset_frame_detections: Mapping[str, Sequence[Any]],
    dataset_frame_gt: Mapping[str, Sequence[Any]],
    tracker_positions: dict[int, np.ndarray],
    models: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Draw camera-view grid with GT centroids and detected bbox centers.

    Shows a sample frame with:
    - GT centroids (circles, color by fish_id)
    - Detected bbox centers (squares)
    - Missed detections marked with red X

    Args:
        dataset_frame_detections: detections_per_camera for the sample frame.
        dataset_frame_gt: ground_truth for the sample frame.
        tracker_positions: Confirmed track positions for the sample frame.
        models: Dict of camera_id -> RefractiveProjectionModel.
        output_path: Output PNG path.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from aquapose.visualization.overlay import FISH_COLORS

    rgb_colors = [(b / 255.0, g / 255.0, r / 255.0) for b, g, r in FISH_COLORS]

    cam_ids = list(models.keys())
    n_cams = min(len(cam_ids), 16)
    cam_ids = cam_ids[:n_cams]

    n_cols = min(4, n_cams)
    n_rows = (n_cams + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    from aquapose.synthetic.detection import _image_size_from_model

    for idx, cam_id in enumerate(cam_ids):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        model = models[cam_id]
        img_w, img_h = _image_size_from_model(model)  # type: ignore[arg-type]

        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")
        ax.set_title(cam_id, fontsize=8, color="white", pad=2)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Draw GT centroids (circles)
        gt_entries = dataset_frame_gt.get(cam_id, [])
        for entry in gt_entries:
            u, v = entry.true_centroid_px  # type: ignore[attr-defined]
            fish_id = entry.fish_id  # type: ignore[attr-defined]
            detected = entry.was_detected  # type: ignore[attr-defined]
            color = rgb_colors[fish_id % len(rgb_colors)]
            ax.plot(
                u,
                v,
                "o",
                color=color,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
            if not detected:
                # Missed detection: red X
                ax.plot(u, v, "rx", markersize=12, markeredgewidth=2)

        # Draw detection bbox centers (squares)
        dets = dataset_frame_detections.get(cam_id, [])
        for det in dets:
            x, y, w, h = det.bbox  # type: ignore[attr-defined]
            cx_det = x + w / 2
            cy_det = y + h / 2
            ax.plot(cx_det, cy_det, "s", color="yellow", markersize=6, alpha=0.8)

    # Hide unused axes
    for idx in range(len(cam_ids), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.patch.set_facecolor("#0d0d1a")
    legend_elements = [
        mpatches.Patch(color="white", label="GT centroid (circle)"),
        mpatches.Patch(color="yellow", label="Detected bbox center (square)"),
        mpatches.Patch(color="red", label="Missed detection (X)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=8,
        facecolor="#1a1a2e",
        labelcolor="white",
    )
    fig.suptitle(
        "Camera Detection Overlay Grid (sample frame)", color="white", fontsize=10
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.97))
    plt.savefig(
        output_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close("all")


def vis_id_timeline(
    gt_matching_per_frame: list[dict[int, int | None]],
    gt_coverage_per_frame: list[dict[int, int | None]],
    n_gt_fish: int,
    output_path: Path,
) -> None:
    """Horizontal timeline showing which track_id covers each GT fish per frame.

    X axis = frame, Y axis = GT fish ID. Color = tracker fish_id assigned.
    Gaps (unmatched frames) shown in gray. ID switches marked with red vertical lines.

    Args:
        gt_matching_per_frame: Per-frame dict of track_fish_id -> gt_fish_id.
        gt_coverage_per_frame: Per-frame dict of gt_fish_id -> track_fish_id.
        n_gt_fish: Number of GT fish.
        output_path: Output PNG path.
    """
    import matplotlib.pyplot as plt

    from aquapose.visualization.overlay import FISH_COLORS

    rgb_colors = [(b / 255.0, g / 255.0, r / 255.0) for b, g, r in FISH_COLORS]

    n_frames = len(gt_matching_per_frame)

    fig, axes = plt.subplots(
        n_gt_fish, 1, figsize=(max(10, n_frames // 10), 2 * n_gt_fish), squeeze=False
    )

    id_switch_frames: list[int] = []

    for gt_id in range(n_gt_fish):
        ax = axes[gt_id, 0]
        ax.set_xlim(0, n_frames)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_ylabel(f"GT fish {gt_id}", fontsize=9)

        prev_track_id: int | None = None

        for frame_idx, coverage in enumerate(gt_coverage_per_frame):
            track_id = coverage.get(gt_id)
            if track_id is not None:
                color = rgb_colors[track_id % len(rgb_colors)]
                ax.barh(0, 1, left=frame_idx, height=0.8, color=color, align="center")

                if prev_track_id is not None and track_id != prev_track_id:
                    id_switch_frames.append(frame_idx)
                    ax.axvline(x=frame_idx, color="red", linewidth=1.5, alpha=0.8)
                prev_track_id = track_id
            else:
                ax.barh(
                    0, 1, left=frame_idx, height=0.8, color="#555555", align="center"
                )
                if prev_track_id is not None:
                    prev_track_id = None

        ax.set_xlabel("Frame" if gt_id == n_gt_fish - 1 else "")

    fig.suptitle("ID Consistency Timeline", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close("all")


def vis_metrics_barchart(
    metrics: TrackingMetrics,
    n_frames: int,
    output_path: Path,
) -> None:
    """Bar chart summary of key tracking metrics.

    Args:
        metrics: Computed tracking metrics.
        n_frames: Total frames processed.
        output_path: Output PNG path.
    """
    import matplotlib.pyplot as plt

    if metrics.n_gt > 0:
        tp_pct = 100.0 * metrics.true_positives / metrics.n_gt
        fn_pct = 100.0 * metrics.false_negatives / metrics.n_gt
        fp_pct = 100.0 * metrics.false_positives / max(1, metrics.n_gt)
    else:
        tp_pct = fn_pct = fp_pct = 0.0

    categories = ["MOTA", "TP%", "FN%", "FP%", "Mean Purity"]
    values = [
        max(0.0, metrics.mota) * 100.0,
        tp_pct,
        fn_pct,
        fp_pct,
        metrics.mean_track_purity * 100.0,
    ]
    colors = ["#4CAF50", "#2196F3", "#f44336", "#FF9800", "#9C27B0"]

    _, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, values, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_ylim(0, 110)
    ax.set_ylabel("Value (%)")
    ax.set_title("Tracking Metrics Summary")
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Add raw count annotations
    info_text = (
        f"ID switches: {metrics.id_switches} | "
        f"Fragmentations: {metrics.fragmentation} | "
        f"Mostly tracked: {metrics.mostly_tracked}/{metrics.n_confirmed_tracks}"
    )
    ax.text(
        0.5,
        -0.12,
        info_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close("all")


def write_tracking_report(
    output_path: Path,
    metrics: TrackingMetrics,
    args: argparse.Namespace,
    dataset_metadata: dict[str, object],
    timing: dict[str, float],
    n_frames: int,
) -> None:
    """Write a markdown report summarizing the tracking diagnostic run.

    Args:
        output_path: Output .md path.
        metrics: Computed tracking metrics.
        args: Parsed CLI arguments.
        dataset_metadata: Metadata from the SyntheticDataset.
        timing: Stage timing dict (label -> seconds).
        n_frames: Number of frames processed.
    """
    if metrics.n_gt > 0:
        tp_pct = 100.0 * metrics.true_positives / metrics.n_gt
        fn_pct = 100.0 * metrics.false_negatives / metrics.n_gt
        fp_pct = 100.0 * metrics.false_positives / max(1, metrics.n_gt)
    else:
        tp_pct = fn_pct = fp_pct = 0.0

    n_fish = int(dataset_metadata.get("n_fish") or 0)  # type: ignore[arg-type]
    n_cameras = int(dataset_metadata.get("n_cameras") or 0)  # type: ignore[arg-type]
    total_time = sum(timing.values())

    lines = [
        "# AquaPose Tracking Diagnostic Report",
        "",
        "## Scenario Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Scenario | `{args.scenario}` |",
    ]

    if args.scenario == "crossing_paths":
        lines.append(f"| Difficulty | {args.difficulty} |")
    elif args.scenario == "track_fragmentation":
        lines.append(f"| Miss rate | {args.miss_rate} |")
    elif args.scenario == "tight_schooling":
        lines.append(f"| N fish | {args.n_fish} |")

    lines += [
        f"| Seed | {args.seed} |",
        f"| Cameras | {n_cameras} (fabricated {args.n_cameras}x{args.n_cameras} grid) |",
        f"| Frames | {n_frames} |",
        f"| GT fish | {n_fish} |",
        f"| FishTracker min_hits | {args.min_hits} |",
        f"| FishTracker min_cameras_birth | {args.min_cameras_birth} |",
        "",
        "## Tracking Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| MOTA | {metrics.mota:.4f} |",
        f"| ID switches | {metrics.id_switches} |",
        f"| Fragmentations | {metrics.fragmentation} |",
        f"| True positives | {metrics.true_positives} / {metrics.n_gt} ({tp_pct:.1f}%) |",
        f"| False negatives | {metrics.false_negatives} ({fn_pct:.1f}%) |",
        f"| False positives | {metrics.false_positives} ({fp_pct:.1f}%) |",
        f"| Mostly tracked | {metrics.mostly_tracked} / {n_fish} |",
        f"| Mostly lost | {metrics.mostly_lost} / {n_fish} |",
        f"| Mean track purity | {metrics.mean_track_purity:.4f} |",
        f"| Confirmed tracks | {metrics.n_confirmed_tracks} |",
        "",
        "## Per-Fish Tracking Coverage",
        "",
        "| GT Fish ID | Tracked Fraction | Status |",
        "|-----------|-----------------|--------|",
    ]

    for gt_id, frac in sorted(metrics.per_fish_tracked_fraction.items()):
        if frac >= 0.8:
            status = "Mostly Tracked"
        elif frac <= 0.2:
            status = "Mostly Lost"
        else:
            status = "Partially Tracked"
        lines.append(f"| {gt_id} | {frac:.3f} | {status} |")

    lines += [
        "",
        "## Visualizations",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `3d_trajectories.png` | GT vs tracked 3D trajectories |",
        "| `detection_overlay.png` | Camera grid with GT/detection overlay (sample frame) |",
        "| `id_timeline.png` | Per-GT-fish ID assignment timeline |",
        "| `metrics_barchart.png` | Bar chart of key metrics |",
        "",
        "## Timing",
        "",
        "| Stage | Seconds |",
        "|-------|---------|",
    ]

    for stage, elapsed in timing.items():
        lines.append(f"| {stage} | {elapsed:.2f} |")

    lines += [
        f"| **Total** | **{total_time:.2f}** |",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark FishTracker on synthetic data with GT metrics.",
    )
    parser.add_argument(
        "--scenario",
        choices=[
            "crossing_paths",
            "track_fragmentation",
            "tight_schooling",
            "startle_response",
        ],
        default="crossing_paths",
        help="Scenario preset to run (default: crossing_paths)",
    )
    parser.add_argument(
        "--difficulty",
        type=float,
        default=0.5,
        help="Crossing difficulty for crossing_paths scenario (default: 0.5)",
    )
    parser.add_argument(
        "--miss-rate",
        type=float,
        default=0.25,
        help="Miss rate for track_fragmentation scenario (default: 0.25)",
    )
    parser.add_argument(
        "--n-fish",
        type=int,
        default=5,
        help="Number of fish for tight_schooling scenario (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--n-cameras",
        type=int,
        default=4,
        help="Cameras per axis for fabricated rig (NxN grid, default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/tracking_diagnostic"),
        help="Output directory (default: output/tracking_diagnostic)",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="FishTracker min_hits (default: 3)",
    )
    parser.add_argument(
        "--min-cameras-birth",
        type=int,
        default=2,
        help="FishTracker min_cameras_birth (default: 2)",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="FishTracker expected fish count (default: derived from scenario)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the tracking diagnostic and return exit code."""
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== AquaPose Tracking Diagnostic ===\n")
    print(f"Scenario:       {args.scenario}")
    print(f"Output dir:     {output_dir}")
    print()

    timing: dict[str, float] = {}
    wall_start = time.perf_counter()

    # -------------------------------------------------------------------
    # Build fabricated camera rig
    # -------------------------------------------------------------------
    print(f"Building fabricated {args.n_cameras}x{args.n_cameras} camera rig...")
    from aquapose.synthetic.rig import build_fabricated_rig

    models = build_fabricated_rig(
        n_cameras_x=args.n_cameras,
        n_cameras_y=args.n_cameras,
    )
    print(f"  {len(models)} cameras created.\n")

    # -------------------------------------------------------------------
    # Build scenario kwargs
    # -------------------------------------------------------------------
    scenario_kwargs: dict[str, object] = {"seed": args.seed}
    if args.scenario == "crossing_paths":
        scenario_kwargs["difficulty"] = args.difficulty
    elif args.scenario == "track_fragmentation":
        scenario_kwargs["miss_rate"] = args.miss_rate
    elif args.scenario == "tight_schooling":
        scenario_kwargs["n_fish"] = args.n_fish

    # -------------------------------------------------------------------
    # Regenerate trajectory + dataset (needed for GT positions)
    # -------------------------------------------------------------------
    print(f"Generating scenario '{args.scenario}'...")
    t0 = time.perf_counter()

    from aquapose.synthetic.detection import generate_detection_dataset
    from aquapose.synthetic.scenarios import _SCENARIO_REGISTRY
    from aquapose.synthetic.trajectory import generate_trajectories

    if args.scenario not in _SCENARIO_REGISTRY:
        print(f"ERROR: Unknown scenario '{args.scenario}'.")
        return 1

    scenario_fn = _SCENARIO_REGISTRY[args.scenario]
    traj_cfg, noise_cfg = scenario_fn(**scenario_kwargs)  # type: ignore[call-arg, misc]

    trajectory = generate_trajectories(traj_cfg)
    dataset = generate_detection_dataset(trajectory, models, noise_config=noise_cfg)

    timing["scenario_generation"] = time.perf_counter() - t0
    n_fish = trajectory.n_fish
    n_frames = trajectory.n_frames
    print(
        f"  {n_fish} fish, {n_frames} frames, {len(dataset.frames)} dataset frames.\n"
    )

    # Determine expected_count
    expected_count = args.expected_count if args.expected_count is not None else n_fish

    # -------------------------------------------------------------------
    # FishTracker frame loop
    # -------------------------------------------------------------------
    print("Running FishTracker...")
    t0 = time.perf_counter()

    from aquapose.tracking.tracker import FishTracker

    tracker = FishTracker(
        expected_count=expected_count,
        min_hits=args.min_hits,
        min_cameras_birth=args.min_cameras_birth,
    )

    # Per-frame records
    tracker_positions_per_frame: list[dict[int, np.ndarray]] = []
    gt_positions_per_frame: list[np.ndarray] = []
    gt_fish_ids_per_frame: list[list[int]] = []
    gt_matching_per_frame: list[dict[int, int | None]] = []
    # Per-frame coverage: gt_fish_id -> track_id (for timeline viz)
    gt_coverage_per_frame: list[dict[int, int | None]] = []

    frame_timings: list[float] = []

    for frame in dataset.frames:
        frame_t0 = time.perf_counter()
        confirmed = tracker.update(
            frame.detections_per_camera,
            models,
            frame_index=frame.frame_index,
        )
        frame_timings.append(time.perf_counter() - frame_t0)

        # Extract confirmed track positions
        frame_track_positions: dict[int, np.ndarray] = {}
        for track in confirmed:
            if len(track.positions) > 0:
                frame_track_positions[track.fish_id] = np.array(
                    list(track.positions)[-1], dtype=np.float32
                )
        tracker_positions_per_frame.append(frame_track_positions)

        # Extract GT positions for this frame from trajectory
        gt_state = trajectory.states[frame.frame_index]  # (n_fish, 7)
        gt_pos = gt_state[:, :3]  # (n_fish, 3)
        gt_ids = list(range(n_fish))
        gt_positions_per_frame.append(gt_pos)
        gt_fish_ids_per_frame.append(gt_ids)

        # Match tracker to GT
        if frame_track_positions:
            track_id_list = list(frame_track_positions.keys())
            track_pos_list = [frame_track_positions[tid] for tid in track_id_list]
            matching = _match_gt_to_tracks(
                frame_gt_positions=gt_pos,
                frame_gt_fish_ids=gt_ids,
                frame_track_positions=track_pos_list,
                frame_track_ids=track_id_list,
            )
        else:
            matching = {}
        gt_matching_per_frame.append(matching)

        # Build coverage (inverse mapping gt_id -> track_id)
        coverage: dict[int, int | None] = {gt_id: None for gt_id in gt_ids}
        for track_id, gt_id in matching.items():
            if gt_id is not None:
                coverage[gt_id] = track_id
        gt_coverage_per_frame.append(coverage)

    timing["tracking"] = time.perf_counter() - t0
    mean_frame_ms = 1000.0 * float(np.mean(frame_timings)) if frame_timings else 0.0
    print(f"  {n_frames} frames processed. Mean frame time: {mean_frame_ms:.1f} ms\n")

    # -------------------------------------------------------------------
    # Compute metrics
    # -------------------------------------------------------------------
    print("Computing tracking metrics...")
    t0 = time.perf_counter()
    metrics = compute_tracking_metrics(
        gt_matching_per_frame=gt_matching_per_frame,
        gt_positions_per_frame=gt_positions_per_frame,
        gt_fish_ids_per_frame=gt_fish_ids_per_frame,
        n_gt_fish=n_fish,
    )
    timing["metrics"] = time.perf_counter() - t0

    # -------------------------------------------------------------------
    # Print metrics summary
    # -------------------------------------------------------------------
    n_gt = metrics.n_gt
    tp = metrics.true_positives
    fn = metrics.false_negatives
    fp = metrics.false_positives

    if args.scenario == "crossing_paths":
        scenario_desc = f"crossing_paths (difficulty={args.difficulty})"
    elif args.scenario == "track_fragmentation":
        scenario_desc = f"track_fragmentation (miss_rate={args.miss_rate})"
    elif args.scenario == "tight_schooling":
        scenario_desc = f"tight_schooling (n_fish={args.n_fish})"
    elif args.scenario == "startle_response":
        scenario_desc = "startle_response"
    else:
        scenario_desc = args.scenario

    print("=== Tracking Metrics Summary ===")
    print(f"Scenario:            {scenario_desc}")
    print(f"Frames:              {n_frames}")
    print(f"GT fish:             {n_fish}")
    print(f"Confirmed tracks:    {metrics.n_confirmed_tracks}")
    print(f"MOTA:                {metrics.mota:.4f}")
    print(f"ID switches:         {metrics.id_switches}")
    print(f"Fragmentations:      {metrics.fragmentation}")
    if n_gt > 0:
        print(f"True positives:      {tp} / {n_gt} ({100.0 * tp / n_gt:.1f}%)")
        print(f"False negatives:     {fn} ({100.0 * fn / n_gt:.1f}%)")
        print(f"False positives:     {fp} ({100.0 * fp / max(1, n_gt):.1f}%)")
    else:
        print("True positives:      0 / 0")
        print("False negatives:     0")
        print("False positives:     0")
    print(f"Mostly tracked:      {metrics.mostly_tracked} / {n_fish}")
    print(f"Mostly lost:         {metrics.mostly_lost} / {n_fish}")
    print(f"Mean track purity:   {metrics.mean_track_purity:.4f}")
    print()

    # -------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------
    print("=== Generating Visualizations ===")

    # Prepare data for visualizations
    gt_states_xyz = trajectory.states[:, :, :3]  # (n_frames, n_fish, 3)

    # Pick a sample frame for detection overlay
    sample_frame_idx = len(dataset.frames) // 2
    sample_frame = dataset.frames[sample_frame_idx]

    vis_funcs = [
        (
            "3d_trajectories.png",
            lambda: vis_3d_trajectories(
                gt_states=gt_states_xyz,
                tracker_positions_per_frame=tracker_positions_per_frame,
                output_path=output_dir / "3d_trajectories.png",
            ),
        ),
        (
            "detection_overlay.png",
            lambda: vis_detection_overlay_grid(
                dataset_frame_detections=sample_frame.detections_per_camera,
                dataset_frame_gt=sample_frame.ground_truth,
                tracker_positions=tracker_positions_per_frame[sample_frame_idx],
                models=models,
                output_path=output_dir / "detection_overlay.png",
            ),
        ),
        (
            "id_timeline.png",
            lambda: vis_id_timeline(
                gt_matching_per_frame=gt_matching_per_frame,
                gt_coverage_per_frame=gt_coverage_per_frame,
                n_gt_fish=n_fish,
                output_path=output_dir / "id_timeline.png",
            ),
        ),
        (
            "metrics_barchart.png",
            lambda: vis_metrics_barchart(
                metrics=metrics,
                n_frames=n_frames,
                output_path=output_dir / "metrics_barchart.png",
            ),
        ),
    ]

    t0 = time.perf_counter()
    for name, func in vis_funcs:
        try:
            print(f"  Generating {name}...")
            func()
        except Exception as exc:
            print(f"  [WARN] Failed to generate {name}: {exc}")
    timing["visualizations"] = time.perf_counter() - t0

    # -------------------------------------------------------------------
    # Markdown report
    # -------------------------------------------------------------------
    print("  Generating tracking_report.md...")
    try:
        write_tracking_report(
            output_path=output_dir / "tracking_report.md",
            metrics=metrics,
            args=args,
            dataset_metadata=dataset.metadata,
            timing=timing,
            n_frames=n_frames,
        )
    except Exception as exc:
        print(f"  [WARN] Failed to generate tracking_report.md: {exc}")

    # -------------------------------------------------------------------
    # Final timing summary
    # -------------------------------------------------------------------
    wall_elapsed = time.perf_counter() - wall_start
    print()
    print("=== Timing Summary ===")
    for stage, elapsed in timing.items():
        print(f"  {stage:<25} {elapsed:.2f}s")
    print(f"  {'Total wall time':<25} {wall_elapsed:.2f}s")
    print()
    print(f"Output written to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
