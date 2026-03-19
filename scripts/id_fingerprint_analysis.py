"""Fish identity fingerprinting analysis from 3D midline data.

Analyzes whether per-fish morphometric features (body length, segment ratios)
can detect and resolve known ID swaps in the stitched reconstruction.

Known swaps (from manual review of run_20260318_132848):
  Swap 1: frame ~600,  fish 2 <-> 4
  Swap 2: frame ~2610, fish 0 <-> 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUN_DIR = Path.home() / "aquapose/projects/YH/runs" / "run_20260318_132848"
H5_PATH = RUN_DIR / "midlines_stitched_smoothed.h5"
OUTPUT_DIR = RUN_DIR / "fingerprint_analysis"

KNOWN_SWAPS = [
    {"frame": 600, "fish_a": 2, "fish_b": 4},
    {"frame": 2610, "fish_a": 0, "fish_b": 5},
]

NUM_FISH = 9
FPS = 30
MALE_IDS = {5, 6, 7}  # Fish that start with these IDs are males

_output_dir = OUTPUT_DIR


def _set_output_dir(path: Path):
    global _output_dir
    _output_dir = path


def _get_output_dir() -> Path:
    return _output_dir


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(h5_path: Path) -> dict:
    """Load midline data from H5 file."""
    with h5py.File(h5_path, "r") as f:
        g = f["midlines"]
        return {
            "frame_index": g["frame_index"][:],
            "fish_id": g["fish_id"][:],
            "points": g["points"][:],
        }


def compute_body_length(points: np.ndarray) -> float:
    """Polyline chord length from 6 keypoints. Returns NaN if any point is NaN."""
    if np.isnan(points).any():
        return np.nan
    diffs = np.diff(points, axis=0)
    return np.sqrt((diffs**2).sum(axis=1)).sum()


def compute_segment_lengths(points: np.ndarray) -> np.ndarray | None:
    """Return 5 inter-keypoint segment lengths. None if invalid."""
    if np.isnan(points).any():
        return None
    diffs = np.diff(points, axis=0)
    return np.sqrt((diffs**2).sum(axis=1))


def extract_per_fish_timeseries(data: dict) -> dict[int, dict]:
    """Extract per-fish time series of body length and segment ratios.

    Returns dict[fish_id -> {"frames": array, "body_length": array,
                             "seg_ratios": (N,5) array}]
    """
    frame_index = data["frame_index"]
    fish_id = data["fish_id"]
    points = data["points"]
    n_frames = len(frame_index)

    result = {}
    for fid in range(NUM_FISH):
        frames = []
        lengths = []
        seg_ratios_list = []

        for row in range(n_frames):
            cols = np.where(fish_id[row] == fid)[0]
            if len(cols) == 0:
                continue
            col = cols[0]
            p = points[row, col]
            bl = compute_body_length(p)
            if np.isnan(bl):
                continue
            segs = compute_segment_lengths(p)
            if segs is None:
                continue

            frames.append(frame_index[row])
            lengths.append(bl)
            seg_ratios_list.append(segs / bl)

        result[fid] = {
            "frames": np.array(frames),
            "body_length": np.array(lengths),
            "seg_ratios": np.array(seg_ratios_list),
        }
    return result


def apply_swap_corrections(ts: dict[int, dict]) -> dict[int, dict]:
    """Return a copy of timeseries with known ID swaps corrected.

    After correction, each fish_id consistently refers to the same physical fish.
    """
    corrected = {}
    for fid in range(NUM_FISH):
        corrected[fid] = {
            "frames": ts[fid]["frames"].copy(),
            "body_length": ts[fid]["body_length"].copy(),
            "seg_ratios": ts[fid]["seg_ratios"].copy(),
        }

    # Apply swaps in chronological order
    for swap in sorted(KNOWN_SWAPS, key=lambda s: s["frame"]):
        fa, fb = swap["fish_a"], swap["fish_b"]
        swap_frame = swap["frame"]

        for key in ["body_length", "seg_ratios"]:
            # Get masks for frames >= swap_frame
            mask_a = corrected[fa]["frames"] >= swap_frame
            mask_b = corrected[fb]["frames"] >= swap_frame

            # We need to swap the data after swap_frame between fa and fb.
            # Since frames may not perfectly align, we swap the arrays for
            # frames >= swap_frame.
            vals_a_after = corrected[fa][key][mask_a].copy()
            vals_b_after = corrected[fb][key][mask_b].copy()
            frames_a_after = corrected[fa]["frames"][mask_a].copy()
            frames_b_after = corrected[fb]["frames"][mask_b].copy()

            # Rebuild: fa gets its before + fb's after, fb gets its before + fa's after
            vals_a_before = corrected[fa][key][~mask_a]
            vals_b_before = corrected[fb][key][~mask_b]
            frames_a_before = corrected[fa]["frames"][~mask_a]
            frames_b_before = corrected[fb]["frames"][~mask_b]

            corrected[fa][key] = np.concatenate([vals_a_before, vals_b_after])
            corrected[fb][key] = np.concatenate([vals_b_before, vals_a_after])

        # Also swap frames arrays (do this once, not per key)
        mask_a = corrected[fa]["frames"] >= swap_frame
        mask_b = corrected[fb]["frames"] >= swap_frame
        frames_a_after = corrected[fa]["frames"][mask_a].copy()
        frames_b_after = corrected[fb]["frames"][mask_b].copy()
        frames_a_before = corrected[fa]["frames"][~mask_a]
        frames_b_before = corrected[fb]["frames"][~mask_b]
        corrected[fa]["frames"] = np.concatenate([frames_a_before, frames_b_after])
        corrected[fb]["frames"] = np.concatenate([frames_b_before, frames_a_after])

        # Re-sort by frame
        for fid in [fa, fb]:
            order = np.argsort(corrected[fid]["frames"])
            corrected[fid]["frames"] = corrected[fid]["frames"][order]
            for key in ["body_length", "seg_ratios"]:
                corrected[fid][key] = corrected[fid][key][order]

    return corrected


# ---------------------------------------------------------------------------
# Phase 1a: Pairwise discriminability
# ---------------------------------------------------------------------------


def pairwise_discriminability(ts: dict[int, dict], metric_key: str) -> np.ndarray:
    """Compute Cohen's d for all fish pairs on a given metric.

    Returns (NUM_FISH, NUM_FISH) matrix of absolute Cohen's d values.
    """
    means = {}
    stds = {}
    for fid in range(NUM_FISH):
        vals = ts[fid][metric_key]
        vals = vals[np.isfinite(vals)]
        means[fid] = np.mean(vals)
        stds[fid] = np.std(vals, ddof=1)

    d_matrix = np.zeros((NUM_FISH, NUM_FISH))
    for i in range(NUM_FISH):
        for j in range(i + 1, NUM_FISH):
            pooled_std = np.sqrt((stds[i] ** 2 + stds[j] ** 2) / 2)
            d = abs(means[i] - means[j]) / pooled_std if pooled_std > 0 else 0.0
            d_matrix[i, j] = d
            d_matrix[j, i] = d

    return d_matrix


def print_discriminability_table(d_matrix: np.ndarray, metric_name: str, means: dict):
    """Print pairwise discriminability summary."""
    print(f"\n{'=' * 70}")
    print(f"Pairwise Cohen's d — {metric_name}")
    print(f"{'=' * 70}")

    # Header
    header = "     " + "".join(f"  F{j:d}   " for j in range(NUM_FISH))
    print(header)

    for i in range(NUM_FISH):
        row = f"F{i:d}   "
        for j in range(NUM_FISH):
            if i == j:
                row += "   -   "
            elif j > i:
                d = d_matrix[i, j]
                row += f" {d:5.2f} "
            else:
                row += "       "
        print(row)

    # Summary
    pairs = []
    for i in range(NUM_FISH):
        for j in range(i + 1, NUM_FISH):
            pairs.append((i, j, d_matrix[i, j]))

    pairs.sort(key=lambda x: x[2])

    n_strong = sum(1 for _, _, d in pairs if d >= 1.0)
    n_moderate = sum(1 for _, _, d in pairs if 0.5 <= d < 1.0)
    n_weak = sum(1 for _, _, d in pairs if d < 0.5)

    print(f"\nTotal pairs: {len(pairs)}")
    print(f"  Strong (d >= 1.0):   {n_strong}")
    print(f"  Moderate (0.5-1.0):  {n_moderate}")
    print(f"  Weak (d < 0.5):      {n_weak}")

    print("\nWeakest 5 pairs:")
    for i, j, d in pairs[:5]:
        print(f"  Fish {i} vs {j}: d={d:.3f} (means: {means[i]:.3f}, {means[j]:.3f})")

    print("\nStrongest 5 pairs:")
    for i, j, d in pairs[-5:]:
        print(f"  Fish {i} vs {j}: d={d:.3f} (means: {means[i]:.3f}, {means[j]:.3f})")

    # Specifically check swap pairs
    print("\nSwap pairs:")
    for swap in KNOWN_SWAPS:
        fa, fb = swap["fish_a"], swap["fish_b"]
        d = d_matrix[fa, fb]
        label = "STRONG" if d >= 1.0 else "MODERATE" if d >= 0.5 else "WEAK"
        print(
            f"  Fish {fa} vs {fb} (swap @ frame {swap['frame']}): d={d:.3f} [{label}]"
        )


def run_phase_1a(corrected_ts: dict[int, dict]):
    """Phase 1a: Pairwise discriminability on corrected data."""
    print("\n" + "#" * 70)
    print("# PHASE 1a: Pairwise Discriminability (corrected IDs)")
    print("#" * 70)

    for metric_key, metric_name, scale in [
        ("body_length", "Body Length (cm)", 100),
    ]:
        means = {}
        stds = {}
        for fid in range(NUM_FISH):
            vals = corrected_ts[fid][metric_key]
            vals = vals[np.isfinite(vals)] * scale
            means[fid] = np.mean(vals)
            stds[fid] = np.std(vals, ddof=1)

        d_matrix = pairwise_discriminability(corrected_ts, metric_key)
        print_discriminability_table(d_matrix, metric_name, means)

        # Print per-fish summary
        print(f"\nPer-fish {metric_name}:")
        for fid in range(NUM_FISH):
            print(
                f"  Fish {fid}: mean={means[fid]:.3f}, std={stds[fid]:.3f}, "
                f"cv={stds[fid] / means[fid] * 100:.1f}%"
            )

    # --- Cohen's d heatmap ---
    out = _get_output_dir()
    out.mkdir(parents=True, exist_ok=True)

    d_matrix = pairwise_discriminability(corrected_ts, "body_length")

    # Reorder: females first, then males
    female_ids = sorted(fid for fid in range(NUM_FISH) if fid not in MALE_IDS)
    male_ids = sorted(MALE_IDS)
    order = female_ids + male_ids
    n = len(order)

    # Reorder matrix
    d_reordered = np.zeros((n, n))
    for ri, oi in enumerate(order):
        for rj, oj in enumerate(order):
            d_reordered[ri, rj] = d_matrix[oi, oj]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Mask diagonal
    masked = np.ma.masked_where(np.eye(n, dtype=bool), d_reordered)
    im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=5, aspect="equal")

    # Annotate cells
    for ri in range(n):
        for rj in range(n):
            if ri == rj:
                continue
            val = d_reordered[ri, rj]
            text_color = "white" if val > 3.5 or val < 0.8 else "black"
            ax.text(
                rj,
                ri,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

    # Draw block separators
    n_female = len(female_ids)
    ax.axhline(n_female - 0.5, color="black", linewidth=2)
    ax.axvline(n_female - 0.5, color="black", linewidth=2)

    # Labels with sex
    labels = [f"F{fid} ({'M' if fid in MALE_IDS else 'F'})" for fid in order]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Pairwise Cohen's d — Body Length")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cohen's d")

    # Add threshold annotations on colorbar
    for thresh, label in [(0.5, "moderate"), (1.0, "strong")]:
        cbar.ax.axhline(thresh, color="black", linewidth=0.8, linestyle="--")
        cbar.ax.text(1.5, thresh, f"  {label}", va="center", fontsize=8)

    plt.tight_layout()
    heatmap_path = out / "cohens_d_heatmap.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {heatmap_path}")

    # CSV: Cohen's d matrix (sex-blocked order)
    csv_path = out / "cohens_d_heatmap.csv"
    col_labels = [f"F{fid}_({'M' if fid in MALE_IDS else 'F'})" for fid in order]
    with open(csv_path, "w") as f:
        f.write("," + ",".join(col_labels) + "\n")
        for ri, _oi in enumerate(order):
            row_label = col_labels[ri]
            vals = [f"{d_reordered[ri, rj]:.3f}" if ri != rj else "" for rj in range(n)]
            f.write(row_label + "," + ",".join(vals) + "\n")
    print(f"  Saved: {csv_path}")

    # --- Bar chart: median body length per fish with IQR error bars ---
    out.mkdir(parents=True, exist_ok=True)

    medians = []
    q25s = []
    q75s = []
    colors = []
    labels = []
    for fid in range(NUM_FISH):
        vals = corrected_ts[fid]["body_length"] * 100  # cm
        vals = vals[np.isfinite(vals)]
        med = np.median(vals)
        q25 = np.percentile(vals, 25)
        q75 = np.percentile(vals, 75)
        medians.append(med)
        q25s.append(med - q25)
        q75s.append(q75 - med)
        sex = "M" if fid in MALE_IDS else "F"
        colors.append("#4a90d9" if fid in MALE_IDS else "#d94a4a")
        labels.append(f"Fish {fid} ({sex})")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(NUM_FISH)
    ax.bar(x, medians, color=colors, edgecolor="black", linewidth=0.5)
    ax.errorbar(
        x,
        medians,
        yerr=[q25s, q75s],
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Body Length (cm)")
    ax.set_title("Median Body Length per Fish (corrected IDs, IQR error bars)")

    # Legend for sex
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#d94a4a", edgecolor="black", label="Female"),
        Patch(facecolor="#4a90d9", edgecolor="black", label="Male"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    bar_path = out / "body_length_per_fish.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {bar_path}")

    # CSV: per-fish body length summary
    csv_path = out / "body_length_per_fish.csv"
    with open(csv_path, "w") as f:
        f.write("fish_id,sex,median_cm,q25_cm,q75_cm\n")
        for fid in range(NUM_FISH):
            vals = corrected_ts[fid]["body_length"] * 100
            vals = vals[np.isfinite(vals)]
            sex = "M" if fid in MALE_IDS else "F"
            f.write(
                f"{fid},{sex},{np.median(vals):.3f},"
                f"{np.percentile(vals, 25):.3f},{np.percentile(vals, 75):.3f}\n"
            )
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Phase 1b: Changepoint detection
# ---------------------------------------------------------------------------


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling median with NaN handling."""
    n = len(values)
    result = np.full(n, np.nan)
    half = window // 2
    for i in range(half, n - half):
        chunk = values[i - half : i + half]
        valid = chunk[np.isfinite(chunk)]
        if len(valid) > 10:
            result[i] = np.median(valid)
    return result


def detect_changepoints(
    frames: np.ndarray,
    values: np.ndarray,
    window: int = 300,
    min_gap: int = 100,
) -> list[dict]:
    """Simple rolling-median-difference changepoint detector.

    Computes |median(left_window) - median(right_window)| at each point.
    Returns local maxima above a threshold as candidate changepoints.
    """
    n = len(values)
    if n < 2 * window:
        return []

    # Compute smoothed signal for left/right comparison
    scores = np.full(n, np.nan)
    for i in range(window, n - window):
        left = values[i - window : i]
        right = values[i : i + window]
        left_valid = left[np.isfinite(left)]
        right_valid = right[np.isfinite(right)]
        if len(left_valid) > 10 and len(right_valid) > 10:
            scores[i] = abs(np.median(right_valid) - np.median(left_valid))

    # Find local maxima
    valid_scores = scores[np.isfinite(scores)]
    if len(valid_scores) == 0:
        return []

    threshold = np.percentile(valid_scores, 95)

    # Find local maxima with minimum separation
    candidates = []
    above = np.where(np.isfinite(scores) & (scores >= threshold))[0]
    if len(above) == 0:
        return candidates

    # Group contiguous runs and nearby points, pick the first index of each plateau
    groups = []
    current_group = [above[0]]
    for idx in above[1:]:
        if idx - current_group[-1] <= min_gap:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    groups.append(current_group)

    for group in groups:
        best_idx = group[np.argmax(scores[group])]
        candidates.append(
            {
                "index": best_idx,
                "frame": int(frames[best_idx]),
                "score": float(scores[best_idx]),
                "threshold": float(threshold),
            }
        )

    return candidates


def classify_changepoint(cp_frame: int, fish_id: int, tolerance: int = 200) -> str:
    """Classify a detected changepoint as true_swap, false_positive."""
    for swap in KNOWN_SWAPS:
        if (
            fish_id in (swap["fish_a"], swap["fish_b"])
            and abs(cp_frame - swap["frame"]) <= tolerance
        ):
            return "true_swap"
    return "false_positive"


def _plot_changepoints(
    raw_ts: dict[int, dict],
    metric_key: str,
    metric_name: str,
    scale: float,
    unit: str,
    score_threshold: float,
    out_path: Path,
    title_suffix: str = "",
):
    """Plot changepoint detection for all fish, filtering by score_threshold."""
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    fig.suptitle(
        f"Changepoint Detection — {metric_name}{title_suffix}",
        fontsize=14,
    )

    # First pass: compute rolling medians to determine common y-scale
    all_medians = []
    per_fish_data = {}
    for fid in range(NUM_FISH):
        frames = raw_ts[fid]["frames"]
        values = raw_ts[fid][metric_key] * scale
        med = rolling_median(values, window=300)
        all_medians.append(med[np.isfinite(med)])
        per_fish_data[fid] = (frames, values, med)

    all_med_vals = np.concatenate(all_medians)
    med_min = np.min(all_med_vals)
    med_max = np.max(all_med_vals)
    med_range = med_max - med_min
    y_center = (med_min + med_max) / 2
    y_half = med_range  # 2x the observed range -> full range = 2 * med_range
    y_lo = y_center - y_half
    y_hi = y_center + y_half

    tp_total = 0
    fp_total = 0

    for fid in range(NUM_FISH):
        ax = axes[fid // 3, fid % 3]
        frames, values, med = per_fish_data[fid]

        # Plot raw signal (thinned for readability)
        step = max(1, len(frames) // 2000)
        ax.plot(
            frames[::step] / FPS,
            values[::step],
            ".",
            markersize=0.5,
            alpha=0.5,
            color="dimgray",
        )

        # Plot rolling median
        valid = np.isfinite(med)
        ax.plot(frames[valid] / FPS, med[valid], "-", linewidth=1, color="navy")

        # Detect changepoints and filter by score threshold
        cps = detect_changepoints(frames, values)
        cps = [cp for cp in cps if cp["score"] >= score_threshold]

        for cp in cps:
            label = classify_changepoint(cp["frame"], fid)
            color = "green" if label == "true_swap" else "red"
            marker = "v" if label == "true_swap" else "x"
            if label == "true_swap":
                tp_total += 1
            else:
                fp_total += 1
            ax.axvline(
                cp["frame"] / FPS,
                color=color,
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
            )
            # Plot marker on the median line at changepoint
            if cp["index"] < len(med) and np.isfinite(med[cp["index"]]):
                ax.plot(
                    cp["frame"] / FPS,
                    med[cp["index"]],
                    marker,
                    color=color,
                    markersize=10,
                    zorder=5,
                )

        ax.set_ylim(y_lo, y_hi)

        # Mark known swap frames
        for swap in KNOWN_SWAPS:
            if fid in (swap["fish_a"], swap["fish_b"]):
                ax.axvline(
                    swap["frame"] / FPS,
                    color="darkorange",
                    linestyle=":",
                    alpha=0.7,
                    linewidth=1.5,
                )

        ax.set_title(f"Fish {fid}", fontsize=10)
        ax.set_ylabel(f"{unit}" if fid % 3 == 0 else "")
        if fid // 3 == 2:
            ax.set_xlabel("Time (s)")

    # Legend
    legend_elements = [
        Line2D([0], [0], color="navy", linewidth=1, label="Rolling median"),
        Line2D(
            [0],
            [0],
            color="green",
            linestyle="--",
            linewidth=1.5,
            label="True swap detected",
        ),
        Line2D(
            [0], [0], color="red", linestyle="--", linewidth=1.5, label="False positive"
        ),
        Line2D(
            [0],
            [0],
            color="darkorange",
            linestyle=":",
            alpha=0.7,
            label="Known swap location",
        ),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path} (TP={tp_total}, FP={fp_total})")

    # CSV: rolling median time series per fish
    csv_ts_path = out_path.with_suffix(".csv")
    with open(csv_ts_path, "w") as f:
        f.write("fish_id,frame,time_s,rolling_median_cm\n")
        for fid in range(NUM_FISH):
            frames, values, med = per_fish_data[fid]
            valid = np.isfinite(med)
            for idx in np.where(valid)[0]:
                f.write(f"{fid},{frames[idx]},{frames[idx] / FPS:.2f},{med[idx]:.4f}\n")
    print(f"  Saved: {csv_ts_path}")

    # CSV: detected changepoints
    csv_cp_path = out_path.with_name(out_path.stem + "_changepoints.csv")
    with open(csv_cp_path, "w") as f:
        f.write("fish_id,frame,time_s,score,classification\n")
        for fid in range(NUM_FISH):
            frames, values, med = per_fish_data[fid]
            cps = detect_changepoints(frames, values)
            cps = [cp for cp in cps if cp["score"] >= score_threshold]
            for cp in cps:
                label = classify_changepoint(cp["frame"], fid)
                f.write(
                    f"{fid},{cp['frame']},{cp['frame'] / FPS:.2f},"
                    f"{cp['score']:.4f},{label}\n"
                )
    print(f"  Saved: {csv_cp_path}")


def run_phase_1b(raw_ts: dict[int, dict]):
    """Phase 1b: Changepoint detection on original (uncorrected) data."""
    print("\n" + "#" * 70)
    print("# PHASE 1b: Changepoint Detection (original IDs)")
    print("#" * 70)

    out = _get_output_dir()
    out.mkdir(parents=True, exist_ok=True)

    all_changepoints = {}
    for metric_key, metric_name, scale, unit in [
        ("body_length", "Body Length", 100, "cm"),
    ]:
        print(f"\n--- {metric_name} ---")

        # Collect all changepoint scores to find ROC operating points
        all_scores = []
        for fid in range(NUM_FISH):
            frames = raw_ts[fid]["frames"]
            values = raw_ts[fid][metric_key] * scale
            cps = detect_changepoints(frames, values)
            for cp in cps:
                label = classify_changepoint(cp["frame"], fid)
                all_scores.append({"score": cp["score"], "label": label})
                print(
                    f"  Fish {fid}: frame={cp['frame']} "
                    f"(t={cp['frame'] / FPS:.1f}s), "
                    f"score={cp['score']:.4f}, [{label}]"
                )

        if not all_scores:
            continue

        # Find thresholds for 50% and 100% TPR
        n_true_swaps = len(KNOWN_SWAPS) * 2
        scores_sorted = sorted(set(s["score"] for s in all_scores), reverse=True)

        thresh_50 = None
        thresh_100 = None
        for thresh in scores_sorted:
            detections = [s for s in all_scores if s["score"] >= thresh]
            tp = sum(1 for s in detections if s["label"] == "true_swap")
            tpr = tp / n_true_swaps
            if tpr >= 1.0 and thresh_100 is None:
                thresh_100 = thresh
            if tpr >= 0.5 and thresh_50 is None:
                thresh_50 = thresh

        # Plot at 50% TPR threshold
        if thresh_50 is not None:
            fp_at_50 = sum(
                1
                for s in all_scores
                if s["score"] >= thresh_50 and s["label"] == "false_positive"
            )
            print(f"\n  50% TPR threshold: {thresh_50:.4f} ({fp_at_50} FPs)")
            _plot_changepoints(
                raw_ts,
                metric_key,
                metric_name,
                scale,
                unit,
                score_threshold=thresh_50,
                out_path=out / f"changepoint_{metric_key}_tpr50.png",
                title_suffix=f" (threshold={thresh_50:.1f}, 50% TPR)",
            )

        # Plot at 100% TPR threshold
        if thresh_100 is not None:
            fp_at_100 = sum(
                1
                for s in all_scores
                if s["score"] >= thresh_100 and s["label"] == "false_positive"
            )
            print(f"  100% TPR threshold: {thresh_100:.4f} ({fp_at_100} FPs)")
            _plot_changepoints(
                raw_ts,
                metric_key,
                metric_name,
                scale,
                unit,
                score_threshold=thresh_100,
                out_path=out / f"changepoint_{metric_key}_tpr100.png",
                title_suffix=f" (threshold={thresh_100:.1f}, 100% TPR)",
            )

        all_changepoints[metric_key] = all_scores

    return all_changepoints


# ---------------------------------------------------------------------------
# Phase 2: Segment ratio metrics (pairwise discriminability)
# ---------------------------------------------------------------------------


def run_phase_2(corrected_ts: dict[int, dict]):
    """Phase 2: Additional segment ratio metrics."""
    print("\n" + "#" * 70)
    print("# PHASE 2: Segment Ratio Metrics (corrected IDs)")
    print("#" * 70)

    # Compute additional ratios from seg_ratios (N, 5)
    # seg_ratios columns: [seg0/total, seg1/total, seg2/total, seg3/total, seg4/total]
    # seg0 = nose-to-head, seg4 = pre-tail-to-tail
    # Additional metrics:
    #   - head_ratio: seg0 / total (nose-head proportion)
    #   - tail_ratio: seg4 / total (tail proportion)
    #   - mid_ratio: (seg1+seg2+seg3) / total

    metrics = {}
    for fid in range(NUM_FISH):
        sr = corrected_ts[fid]["seg_ratios"]  # (N, 5)
        valid = np.isfinite(sr).all(axis=1)
        sr = sr[valid]
        metrics.setdefault("head_frac", {})[fid] = sr[:, 0]
        metrics.setdefault("tail_frac", {})[fid] = sr[:, 4]
        metrics.setdefault("head_tail_ratio", {})[fid] = sr[:, 0] / np.where(
            sr[:, 4] > 0, sr[:, 4], np.nan
        )

    for metric_name, fish_vals in metrics.items():
        print(f"\n--- {metric_name} ---")
        # Compute Cohen's d matrix
        means = {}
        stds = {}
        for fid in range(NUM_FISH):
            v = fish_vals[fid]
            v = v[np.isfinite(v)]
            means[fid] = np.mean(v)
            stds[fid] = np.std(v, ddof=1)

        n_strong = 0
        n_moderate = 0
        n_weak = 0
        swap_results = []

        for i in range(NUM_FISH):
            for j in range(i + 1, NUM_FISH):
                pooled_std = np.sqrt((stds[i] ** 2 + stds[j] ** 2) / 2)
                d = abs(means[i] - means[j]) / pooled_std if pooled_std > 0 else 0
                if d >= 1.0:
                    n_strong += 1
                elif d >= 0.5:
                    n_moderate += 1
                else:
                    n_weak += 1

                for swap in KNOWN_SWAPS:
                    if {i, j} == {swap["fish_a"], swap["fish_b"]}:
                        swap_results.append((i, j, d, swap["frame"]))

        print(
            f"  Strong (d>=1.0): {n_strong}, Moderate (0.5-1.0): {n_moderate}, Weak (<0.5): {n_weak}"
        )
        print(
            f"  Per-fish means: {', '.join(f'F{fid}={means[fid]:.4f}' for fid in range(NUM_FISH))}"
        )
        for i, j, d, frame in swap_results:
            label = "STRONG" if d >= 1.0 else "MODERATE" if d >= 0.5 else "WEAK"
            print(f"  Swap pair {i} vs {j} (frame {frame}): d={d:.3f} [{label}]")


# ---------------------------------------------------------------------------
# Phase 3: Per-metric ROC analysis
# ---------------------------------------------------------------------------


def run_phase_3(raw_ts: dict[int, dict]):
    """Phase 3: ROC-like analysis for each metric independently."""
    print("\n" + "#" * 70)
    print("# PHASE 3: Per-Metric ROC Analysis")
    print("#" * 70)

    out = _get_output_dir()
    out.mkdir(parents=True, exist_ok=True)

    for metric_key, metric_name, scale in [
        ("body_length", "Body Length", 100),
    ]:
        print(f"\n--- {metric_name} ---")

        # Collect all changepoint scores across all fish
        all_scores = []
        for fid in range(NUM_FISH):
            frames = raw_ts[fid]["frames"]
            values = raw_ts[fid][metric_key] * scale
            cps = detect_changepoints(frames, values)
            for cp in cps:
                label = classify_changepoint(cp["frame"], fid)
                all_scores.append(
                    {
                        "fish": fid,
                        "frame": cp["frame"],
                        "score": cp["score"],
                        "label": label,
                    }
                )

        if not all_scores:
            print("  No changepoints detected.")
            continue

        # Sweep threshold from min to max score
        scores_arr = np.array([s["score"] for s in all_scores])
        thresholds = np.linspace(scores_arr.min() * 0.9, scores_arr.max() * 1.1, 50)

        n_true_swaps = len(KNOWN_SWAPS) * 2  # each swap involves 2 fish
        tpr_list = []
        fpr_count_list = []

        for thresh in thresholds:
            detections = [s for s in all_scores if s["score"] >= thresh]
            tp = sum(1 for s in detections if s["label"] == "true_swap")
            fp = sum(1 for s in detections if s["label"] == "false_positive")
            tpr_list.append(tp / n_true_swaps if n_true_swaps > 0 else 0)
            fpr_count_list.append(fp)

        # Print summary at a few key operating points
        print(f"  Total true swap signals: {n_true_swaps}")
        print(f"  Total detected changepoints: {len(all_scores)}")
        print(
            f"  True swaps detected: {sum(1 for s in all_scores if s['label'] == 'true_swap')}"
        )
        print(
            f"  False positives: {sum(1 for s in all_scores if s['label'] == 'false_positive')}"
        )

        print(f"\n  {'Threshold':>10s} {'TP':>4s} {'FP':>4s} {'TPR':>6s}")
        for thresh in thresholds[::5]:
            detections = [s for s in all_scores if s["score"] >= thresh]
            tp = sum(1 for s in detections if s["label"] == "true_swap")
            fp = sum(1 for s in detections if s["label"] == "false_positive")
            tpr = tp / n_true_swaps if n_true_swaps > 0 else 0
            print(f"  {thresh:10.5f} {tp:4d} {fp:4d} {tpr:6.2f}")

        # Plot ROC-like curve (TPR vs FP count)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(fpr_count_list, tpr_list, "o-", markersize=3)
        ax.set_xlabel("False Positives (count)")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC-like Curve — {metric_name}")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.grid(True, alpha=0.3)

        out_path = out / f"roc_{metric_key}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")

        # CSV: ROC curve data
        csv_path = out / f"roc_{metric_key}.csv"
        with open(csv_path, "w") as f:
            f.write("threshold,tp,fp,tpr,fp_count\n")
            for thresh in thresholds:
                detections = [s for s in all_scores if s["score"] >= thresh]
                tp = sum(1 for s in detections if s["label"] == "true_swap")
                fp = sum(1 for s in detections if s["label"] == "false_positive")
                tpr = tp / n_true_swaps if n_true_swaps > 0 else 0
                f.write(f"{thresh:.5f},{tp},{fp},{tpr:.4f},{fp}\n")
        print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=H5_PATH, help="Path to H5 file")
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR, help="Output directory"
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_output_dir(output_dir)

    print("Loading data...")
    data = load_data(args.h5)

    print("Extracting per-fish time series...")
    raw_ts = extract_per_fish_timeseries(data)
    for fid in range(NUM_FISH):
        n = len(raw_ts[fid]["frames"])
        bl = raw_ts[fid]["body_length"]
        print(
            f"  Fish {fid}: {n} valid frames, body_length mean={bl.mean() * 100:.2f}cm"
        )

    print("\nApplying swap corrections...")
    corrected_ts = apply_swap_corrections(raw_ts)

    run_phase_1a(corrected_ts)
    run_phase_1b(raw_ts)
    run_phase_2(corrected_ts)
    run_phase_3(raw_ts)

    print("\n" + "=" * 70)
    print("Analysis complete. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
