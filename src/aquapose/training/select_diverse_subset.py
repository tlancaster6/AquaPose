"""Diversity-maximizing subset selection for pseudo-label datasets."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


def _copy_to_split(
    stems: list[str],
    images_src: Path,
    labels_src: Path,
    output_dir: Path,
    split: str,
) -> None:
    """Copy image+label files into a YOLO split directory.

    Args:
        stems: List of file stems to copy.
        images_src: Source images directory.
        labels_src: Source labels directory.
        output_dir: Root output directory (will create images/{split}/ and labels/{split}/).
        split: Split name ("train" or "val").
    """
    img_dst = output_dir / "images" / split
    lbl_dst = output_dir / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        for ext in (".jpg", ".jpeg", ".png"):
            src = images_src / f"{stem}{ext}"
            if src.exists():
                shutil.copy2(src, img_dst / src.name)
                break
        lbl_src = labels_src / f"{stem}.txt"
        if lbl_src.exists():
            shutil.copy2(lbl_src, lbl_dst / lbl_src.name)


def select_obb_subset(
    pseudo_dir: Path,
    output_dir: Path,
    target_count: int = 50,
    val_fraction: float = 0.2,
) -> dict:
    """Select a diverse OBB subset using 3-axis sampling.

    Axes: camera coverage (equal per camera), temporal spread (bins within
    each camera), fish-count tiebreaker (prefer underrepresented counts).

    Args:
        pseudo_dir: Root pseudo-label directory containing ``obb/``.
        output_dir: Output directory for selected files.
        target_count: Target number of OBB images.
        val_fraction: Fraction of selected images for val set (0.0 to skip).

    Returns:
        Selection statistics dict.
    """
    obb_dir = pseudo_dir / "obb"
    conf_path = obb_dir / "confidence.json"
    if not conf_path.exists():
        raise FileNotFoundError(f"No confidence.json in {obb_dir}")

    confidence = json.loads(conf_path.read_text())

    # Parse entries: stem -> (frame_idx, cam_id, fish_count)
    entries: list[dict] = []
    for stem, meta in confidence.items():
        frame_idx = int(stem[:6])
        cam_id = stem[7:]
        fish_count = meta.get("tracked_fish_count", len(meta.get("labels", [])))
        entries.append(
            {
                "stem": stem,
                "frame_idx": frame_idx,
                "cam_id": cam_id,
                "fish_count": fish_count,
            }
        )

    # Group by camera
    by_camera: dict[str, list[dict]] = {}
    for e in entries:
        by_camera.setdefault(e["cam_id"], []).append(e)

    n_cameras = len(by_camera)
    per_camera = target_count // n_cameras
    flex = target_count - per_camera * n_cameras

    # Global fish-count histogram for tiebreaking
    all_counts = [e["fish_count"] for e in entries]
    count_freq = Counter(all_counts)

    selected_stems: list[str] = []
    cam_stats: dict[str, int] = {}

    for cam_id, cam_entries in sorted(by_camera.items()):
        cam_entries.sort(key=lambda e: e["frame_idx"])

        # Divide into temporal bins
        n_bins = min(per_camera, len(cam_entries))
        if n_bins == 0:
            cam_stats[cam_id] = 0
            continue

        bin_edges = np.linspace(0, len(cam_entries), n_bins + 1, dtype=int)

        cam_selected: list[str] = []
        for i in range(n_bins):
            bin_entries = cam_entries[bin_edges[i] : bin_edges[i + 1]]
            if not bin_entries:
                continue
            # Tiebreak: prefer underrepresented fish counts
            best = min(bin_entries, key=lambda e: count_freq[e["fish_count"]])
            cam_selected.append(best["stem"])

        selected_stems.extend(cam_selected)
        cam_stats[cam_id] = len(cam_selected)

    # Flex picks: cameras with most remaining candidates
    if flex > 0:
        selected_set = set(selected_stems)
        remaining = [e for e in entries if e["stem"] not in selected_set]
        # Sort by underrepresented fish count
        remaining.sort(key=lambda e: count_freq[e["fish_count"]])
        for e in remaining[:flex]:
            selected_stems.append(e["stem"])
            cam_stats[e["cam_id"]] = cam_stats.get(e["cam_id"], 0) + 1

    # Split train/val by frame index (later frames -> val)
    selected_entries = [e for e in entries if e["stem"] in set(selected_stems)]
    selected_entries.sort(key=lambda e: e["frame_idx"])

    if val_fraction > 0:
        n_val = max(1, int(len(selected_entries) * val_fraction))
        val_stems = [e["stem"] for e in selected_entries[-n_val:]]
        train_stems = [e["stem"] for e in selected_entries[:-n_val]]
    else:
        val_stems = []
        train_stems = [e["stem"] for e in selected_entries]

    # Copy files
    images_src = obb_dir / "images" / "train"
    labels_src = obb_dir / "labels" / "train"
    _copy_to_split(train_stems, images_src, labels_src, output_dir, "train")
    if val_stems:
        _copy_to_split(val_stems, images_src, labels_src, output_dir, "val")

    # Write filtered confidence
    selected_conf: dict[str, dict] = {}
    for stem in selected_stems:
        selected_conf[stem] = confidence[stem]
    (output_dir / "confidence.json").write_text(
        json.dumps(selected_conf, indent=2), encoding="utf-8"
    )

    return {
        "total_selected": len(selected_stems),
        "total_available": len(entries),
        "train_count": len(train_stems),
        "val_count": len(val_stems),
        "per_camera": cam_stats,
        "n_cameras": n_cameras,
    }


def select_pose_subset(
    pseudo_dir: Path,
    output_dir: Path,
    target_count: int = 320,
    val_fraction: float = 0.2,
) -> dict:
    """Select a diverse pose subset using 2-axis sampling.

    Axes: camera coverage and 2D curvature stratification.
    The last ``val_fraction`` by frame index goes to a ``val/`` subdirectory.

    Args:
        pseudo_dir: Root pseudo-label directory containing ``pose/consensus/``.
        output_dir: Output directory for selected files.
        target_count: Target number of pose crops.
        val_fraction: Fraction of selected crops for secondary val set.

    Returns:
        Selection statistics dict.
    """
    pose_dir = pseudo_dir / "pose" / "consensus"
    conf_path = pose_dir / "confidence.json"
    if not conf_path.exists():
        raise FileNotFoundError(f"No confidence.json in {pose_dir}")

    confidence = json.loads(conf_path.read_text())

    # Build per-crop entries with curvature
    # Confidence keys are {frame}_{cam}, crops are {frame}_{cam}_{fish_idx}
    images_dir = pose_dir / "images" / "train"
    crop_files = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    entries: list[dict] = []
    for crop_path in crop_files:
        stem = crop_path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        conf_key = parts[0]  # {frame}_{cam}
        fish_idx = int(parts[1])
        frame_idx = int(stem[:6])
        cam_id = conf_key[7:]

        # Get curvature from confidence sidecar
        curvature = 0.0
        conf_entry = confidence.get(conf_key, {})
        labels = conf_entry.get("labels", [])
        if fish_idx < len(labels):
            curvature = labels[fish_idx].get("curvature_2d", 0.0)

        entries.append(
            {
                "stem": stem,
                "frame_idx": frame_idx,
                "cam_id": cam_id,
                "fish_idx": fish_idx,
                "curvature": curvature,
            }
        )

    if not entries:
        return {"total_selected": 0, "total_available": 0}

    target_count = min(target_count, len(entries))

    # Curvature quartile bins
    curvatures = np.array([e["curvature"] for e in entries])
    quartiles = np.percentile(curvatures, [25, 50, 75])

    def curvature_bin(c: float) -> int:
        for i, q in enumerate(quartiles):
            if c <= q:
                return i
        return len(quartiles)

    for e in entries:
        e["curv_bin"] = curvature_bin(e["curvature"])

    # Group by (curvature_bin, cam_id)
    groups: dict[tuple[int, str], list[dict]] = {}
    for e in entries:
        key = (e["curv_bin"], e["cam_id"])
        groups.setdefault(key, []).append(e)

    # Proportional allocation across groups
    n_groups = len(groups)
    per_group = max(1, target_count // n_groups)

    selected: list[dict] = []
    for _key, group_entries in sorted(groups.items()):
        # Sort by frame for temporal spread within group
        group_entries.sort(key=lambda e: e["frame_idx"])
        n_pick = min(per_group, len(group_entries))
        if n_pick == 0:
            continue
        # Evenly space selections
        indices = np.linspace(0, len(group_entries) - 1, n_pick, dtype=int)
        for idx in indices:
            selected.append(group_entries[idx])

    # If under target, fill from remaining
    if len(selected) < target_count:
        selected_set = {e["stem"] for e in selected}
        remaining = [e for e in entries if e["stem"] not in selected_set]
        np.random.default_rng(42).shuffle(remaining)
        for e in remaining[: target_count - len(selected)]:
            selected.append(e)

    # If over target, trim
    if len(selected) > target_count:
        selected = selected[:target_count]

    # Split train/val by frame index (later frames -> val)
    selected.sort(key=lambda e: e["frame_idx"])

    if val_fraction > 0:
        n_val = max(1, int(len(selected) * val_fraction))
        val_set = selected[-n_val:]
        train_set = selected[:-n_val]
    else:
        val_set = []
        train_set = list(selected)

    # Copy files
    images_src = pose_dir / "images" / "train"
    labels_src = pose_dir / "labels" / "train"
    _copy_to_split(
        [e["stem"] for e in train_set], images_src, labels_src, output_dir, "train"
    )
    if val_set:
        _copy_to_split(
            [e["stem"] for e in val_set], images_src, labels_src, output_dir, "val"
        )

    # Write filtered confidence
    selected_conf: dict[str, dict] = {}
    for e in selected:
        conf_key = e["stem"].rsplit("_", 1)[0]
        if conf_key in confidence:
            selected_conf[conf_key] = confidence[conf_key]
    (output_dir / "confidence.json").write_text(
        json.dumps(selected_conf, indent=2), encoding="utf-8"
    )

    # Stats
    curv_bin_counts = Counter(e["curv_bin"] for e in selected)
    cam_counts = Counter(e["cam_id"] for e in selected)

    return {
        "total_selected": len(selected),
        "total_available": len(entries),
        "train_count": len(train_set),
        "val_count": len(val_set),
        "per_camera": dict(sorted(cam_counts.items())),
        "per_curvature_bin": dict(sorted(curv_bin_counts.items())),
        "curvature_quartiles": quartiles.tolist(),
    }
