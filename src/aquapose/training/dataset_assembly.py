"""Dataset assembly: pool manual + pseudo-label sources into a YOLO dataset."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def collect_pseudo_labels(
    run_dirs: list[Path],
    base_path: str,
    source: str,
) -> list[dict]:
    """Discover pseudo-labels from run directories.

    Args:
        run_dirs: Pipeline run directories containing ``pseudo_labels/``.
        base_path: Relative path under ``pseudo_labels/`` to the label
            directory, e.g. ``"obb"``, ``"pose/consensus"``, ``"pose/gap"``.
        source: Label source tag for the result dict, e.g. ``"merged"``,
            ``"consensus"``, ``"gap"``.

    Returns:
        List of dicts with keys: ``image_path``, ``label_path``, ``stem``,
        ``confidence``, ``source``, ``run_id``, ``metadata``.
    """
    results: list[dict] = []

    for run_dir in run_dirs:
        run_id = run_dir.name
        conf_path = run_dir / "pseudo_labels" / base_path / "confidence.json"

        if not conf_path.exists():
            logger.warning("No confidence.json at %s, skipping", conf_path)
            continue

        confidence_data = json.loads(conf_path.read_text())

        img_dir = run_dir / "pseudo_labels" / base_path / "images" / "train"
        lbl_dir = run_dir / "pseudo_labels" / base_path / "labels" / "train"

        for img_path in sorted(img_dir.glob("*.jpg")):
            stem = img_path.stem
            label_path = lbl_dir / f"{stem}.txt"

            if not label_path.exists():
                continue

            # Look up confidence from sidecar
            if stem not in confidence_data:
                continue

            entry = confidence_data[stem]
            label_entries = entry.get("labels", [])

            # Mean confidence across fish in this image
            if label_entries:
                mean_conf = sum(e["confidence"] for e in label_entries) / len(
                    label_entries
                )
            else:
                mean_conf = 0.0

            results.append(
                {
                    "image_path": img_path,
                    "label_path": label_path,
                    "stem": stem,
                    "confidence": mean_conf,
                    "source": source,
                    "run_id": run_id,
                    "metadata": entry,
                }
            )

    return results


def filter_by_confidence(
    labels: list[dict],
    min_confidence: float,
) -> list[dict]:
    """Filter pseudo-labels by image-level mean confidence.

    Args:
        labels: Label dicts from :func:`collect_pseudo_labels`.
        min_confidence: Minimum confidence threshold (inclusive).

    Returns:
        Filtered list of label dicts.
    """
    return [lbl for lbl in labels if lbl["confidence"] >= min_confidence]


def filter_by_gap_reason(
    labels: list[dict],
    exclude_reasons: list[str],
) -> list[dict]:
    """Exclude gap labels where ALL fish match excluded gap reasons.

    An image is excluded only if every fish label in it has a ``gap_reason``
    that appears in *exclude_reasons*. Images where at least one fish has a
    different (or missing) gap_reason are kept.

    Args:
        labels: Label dicts with ``metadata`` containing ``labels`` entries.
        exclude_reasons: Gap reasons to exclude.

    Returns:
        Filtered list of label dicts.
    """
    if not exclude_reasons:
        return list(labels)

    exclude_set = set(exclude_reasons)
    result: list[dict] = []

    for lbl in labels:
        fish_labels = lbl.get("metadata", {}).get("labels", [])
        if not fish_labels:
            result.append(lbl)
            continue

        # Keep if ANY fish does NOT match an excluded reason
        all_excluded = all(
            fish.get("gap_reason") in exclude_set for fish in fish_labels
        )
        if not all_excluded:
            result.append(lbl)

    return result


def split_manual_val(
    manual_dir: Path,
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split manual annotations into train/val per-camera stratified.

    Args:
        manual_dir: YOLO-format directory with ``dataset.yaml``, ``images/``,
            ``labels/``.
        val_fraction: Fraction of images per camera to hold out for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_stems, val_stems).
    """
    import numpy as np

    img_dir = manual_dir / "images" / "train"
    stems = sorted(p.stem for p in img_dir.glob("*.jpg"))

    if not stems or val_fraction <= 0.0:
        return stems, []

    # Group by camera (pattern: {frame}_{cam_id})
    cam_groups: dict[str, list[str]] = {}
    for stem in stems:
        parts = stem.rsplit("_", 1)
        cam_id = parts[-1] if len(parts) > 1 else "unknown"
        cam_groups.setdefault(cam_id, []).append(stem)

    rng = np.random.default_rng(seed)
    train_stems: list[str] = []
    val_stems: list[str] = []

    for cam_id in sorted(cam_groups.keys()):
        cam_stems = cam_groups[cam_id]
        n_val = max(1, int(len(cam_stems) * val_fraction))
        n_val = min(n_val, len(cam_stems))

        indices = rng.permutation(len(cam_stems))
        val_indices = set(indices[:n_val].tolist())

        for i, stem in enumerate(cam_stems):
            if i in val_indices:
                val_stems.append(stem)
            else:
                train_stems.append(stem)

    return train_stems, val_stems


def split_pseudo_val(
    labels: list[dict],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split pseudo-labels into train and val portions.

    Args:
        labels: Label dicts from :func:`collect_pseudo_labels`.
        val_fraction: Fraction to hold out for evaluation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_labels, val_labels).
    """
    if not labels or val_fraction <= 0.0:
        return list(labels), []

    import numpy as np

    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(labels) * val_fraction))
    n_val = min(n_val, len(labels))

    indices = rng.permutation(len(labels))
    val_indices = set(indices[:n_val].tolist())

    train: list[dict] = []
    val: list[dict] = []

    for i, lbl in enumerate(labels):
        if i in val_indices:
            val.append(lbl)
        else:
            train.append(lbl)

    return train, val


def _extract_dominant_gap_reason(lbl: dict) -> str | None:
    """Extract the most common gap_reason from a label's metadata.

    Args:
        lbl: Label dict with ``source`` and ``metadata`` keys.

    Returns:
        The dominant gap_reason string, or None for consensus labels or
        labels without gap_reason metadata.
    """
    if lbl["source"] != "gap":
        return None

    fish_labels = lbl.get("metadata", {}).get("labels", [])
    if not fish_labels:
        return None

    from collections import Counter

    reasons = [
        f.get("gap_reason") for f in fish_labels if f.get("gap_reason") is not None
    ]
    if not reasons:
        return None

    # Most common reason (first if tied)
    return Counter(reasons).most_common(1)[0][0]


def _filter_by_frames(
    labels: list[dict],
    selected_frames: dict[str, set[int]],
) -> list[dict]:
    """Filter pseudo-labels by frame index for specific runs.

    For each label, the frame index is parsed as ``int(label["stem"][:6])``.
    Labels whose ``run_id`` is not in *selected_frames* pass through
    unfiltered (per Phase 65 decision: runs not in the dict are kept).

    Args:
        labels: Label dicts with ``stem`` and ``run_id`` keys.
        selected_frames: Mapping of ``run_id`` to allowed frame indices.

    Returns:
        Filtered list of label dicts.
    """
    result: list[dict] = []
    for lbl in labels:
        run_id = lbl["run_id"]
        if run_id not in selected_frames:
            result.append(lbl)
            continue
        frame_idx = int(lbl["stem"][:6])
        if frame_idx in selected_frames[run_id]:
            result.append(lbl)
    return result


def assemble_dataset(
    output_dir: Path,
    manual_dir: Path | None,
    run_dirs: list[Path],
    model_type: str,
    consensus_threshold: float,
    gap_threshold: float,
    exclude_gap_reasons: list[str],
    manual_val_fraction: float,
    pseudo_val_fraction: float,
    seed: int,
    max_frames: int | None = None,
    selected_frames: dict[str, set[int]] | None = None,
    diversity_bin_map: dict[str, dict[int, int]] | None = None,
) -> dict:
    """Assemble a YOLO-format training dataset from manual + pseudo-labels.

    Pools manual annotations and pseudo-labels from multiple pipeline runs,
    applies confidence filtering and gap-reason exclusion, creates train/val
    splits, and writes YOLO-standard output.

    For OBB model_type, reads from the merged ``obb/`` directory and applies
    ``min(consensus_threshold, gap_threshold)`` as a single threshold. For
    pose model_type, reads from ``pose/consensus/`` and ``pose/gap/``
    separately with independent thresholds.

    Args:
        output_dir: Output directory for assembled dataset.
        manual_dir: YOLO-format manual annotation directory, or None.
        run_dirs: Pipeline run directories with ``pseudo_labels/``.
        model_type: ``"obb"`` or ``"pose"``.
        consensus_threshold: Min confidence for consensus labels.
        gap_threshold: Min confidence for gap labels.
        exclude_gap_reasons: Gap reasons to exclude.
        manual_val_fraction: Fraction of manual data for validation.
        pseudo_val_fraction: Fraction of pseudo-labels held out.
        seed: Random seed.
        max_frames: Hard cap on total pseudo-label images. When the
            filtered pool exceeds this, a uniform random subsample is
            taken. Manual annotations are not affected. None means no cap.
        selected_frames: Mapping of ``run_id`` to allowed frame indices.
            When provided, pseudo-labels are filtered by frame index
            (parsed from first 6 chars of stem) after confidence/gap
            filtering but before the *max_frames* cap. Runs not in the
            dict are kept unfiltered. None disables frame filtering.
        diversity_bin_map: Mapping of ``run_id`` to ``{frame_idx: bin_id}``
            from diversity sampling. When provided, a ``curvature_bin``
            field is included in pseudo-label val metadata. None omits it.

    Returns:
        Summary dict with counts per category.
    """
    # Create output directories
    out_img_train = output_dir / "images" / "train"
    out_img_val = output_dir / "images" / "val"
    out_lbl_train = output_dir / "labels" / "train"
    out_lbl_val = output_dir / "labels" / "val"
    for d in [out_img_train, out_img_val, out_lbl_train, out_lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {
        "manual_train": 0,
        "manual_val": 0,
        "consensus_train": 0,
        "consensus_val": 0,
        "gap_train": 0,
        "gap_val": 0,
        "pseudo_train": 0,
        "pseudo_val": 0,
    }

    # --- Manual annotations ---
    if manual_dir is not None and manual_dir.exists():
        manual_train_stems, manual_val_stems = split_manual_val(
            manual_dir, manual_val_fraction, seed
        )

        manual_img_dir = manual_dir / "images" / "train"
        manual_lbl_dir = manual_dir / "labels" / "train"

        # Copy manual train
        for stem in manual_train_stems:
            _copy_if_exists(
                manual_img_dir / f"{stem}.jpg", out_img_train / f"{stem}.jpg"
            )
            _copy_if_exists(
                manual_lbl_dir / f"{stem}.txt", out_lbl_train / f"{stem}.txt"
            )
            counts["manual_train"] += 1

        # Copy manual val
        for stem in manual_val_stems:
            _copy_if_exists(manual_img_dir / f"{stem}.jpg", out_img_val / f"{stem}.jpg")
            _copy_if_exists(manual_lbl_dir / f"{stem}.txt", out_lbl_val / f"{stem}.txt")
            counts["manual_val"] += 1

    # --- Pseudo-labels ---
    if model_type == "obb":
        # Merged OBB: single directory, single threshold
        obb_threshold = min(consensus_threshold, gap_threshold)
        all_pseudo = collect_pseudo_labels(run_dirs, "obb", "merged")
        all_pseudo = filter_by_confidence(all_pseudo, obb_threshold)
        all_pseudo = filter_by_gap_reason(all_pseudo, exclude_gap_reasons)
    else:
        # Pose: separate consensus/gap directories with independent thresholds
        consensus_labels = collect_pseudo_labels(
            run_dirs, "pose/consensus", "consensus"
        )
        consensus_labels = filter_by_confidence(consensus_labels, consensus_threshold)

        gap_labels = collect_pseudo_labels(run_dirs, "pose/gap", "gap")
        gap_labels = filter_by_confidence(gap_labels, gap_threshold)
        gap_labels = filter_by_gap_reason(gap_labels, exclude_gap_reasons)

        all_pseudo = consensus_labels + gap_labels

    # --- Apply frame selection filter ---
    if selected_frames is not None:
        all_pseudo = _filter_by_frames(all_pseudo, selected_frames)

    # --- Apply max_frames cap ---
    if max_frames is not None and len(all_pseudo) > max_frames:
        import numpy as np

        rng = np.random.default_rng(seed)
        indices = rng.choice(len(all_pseudo), size=max_frames, replace=False)
        indices.sort()
        all_pseudo = [all_pseudo[i] for i in indices]
        logger.info(
            "Applied max_frames cap: %d pseudo-labels",
            max_frames,
        )

    pseudo_train, pseudo_val = split_pseudo_val(all_pseudo, pseudo_val_fraction, seed)

    # --- Copy pseudo-label train ---
    for lbl in pseudo_train:
        prefixed_stem = f"{lbl['run_id']}_{lbl['stem']}"
        _copy_if_exists(lbl["image_path"], out_img_train / f"{prefixed_stem}.jpg")
        _copy_if_exists(lbl["label_path"], out_lbl_train / f"{prefixed_stem}.txt")
        if lbl["source"] == "consensus":
            counts["consensus_train"] += 1
        elif lbl["source"] == "gap":
            counts["gap_train"] += 1
        else:
            counts["pseudo_train"] += 1

    # --- Copy pseudo-label val (NOT into val/ -- pseudo val is separate) ---
    pseudo_val_metadata: list[dict] = []
    for lbl in pseudo_val:
        prefixed_stem = f"{lbl['run_id']}_{lbl['stem']}"
        _copy_if_exists(lbl["image_path"], out_img_train / f"{prefixed_stem}.jpg")
        _copy_if_exists(lbl["label_path"], out_lbl_train / f"{prefixed_stem}.txt")
        entry_meta: dict = {
            "stem": prefixed_stem,
            "source": lbl["source"],
            "confidence": lbl["confidence"],
            "run_id": lbl["run_id"],
            "gap_reason": _extract_dominant_gap_reason(lbl),
        }
        if diversity_bin_map is not None:
            run_bins = diversity_bin_map.get(lbl["run_id"])
            if run_bins is not None:
                frame_idx = int(lbl["stem"][:6])
                entry_meta["curvature_bin"] = run_bins.get(frame_idx)
        pseudo_val_metadata.append(entry_meta)
        if lbl["source"] == "consensus":
            counts["consensus_val"] += 1
        elif lbl["source"] == "gap":
            counts["gap_val"] += 1
        else:
            counts["pseudo_val"] += 1

    # --- Write dataset.yaml ---
    dataset_config = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "fish"},
    }
    (output_dir / "dataset.yaml").write_text(
        yaml.dump(dataset_config, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    # --- Write pseudo-label val metadata sidecar ---
    if pseudo_val_metadata:
        (output_dir / "pseudo_val_metadata.json").write_text(
            json.dumps(pseudo_val_metadata, indent=2), encoding="utf-8"
        )

    return counts


def _copy_if_exists(src: Path, dst: Path) -> None:
    """Copy file if source exists, otherwise log warning."""
    if src.exists():
        shutil.copy2(src, dst)
    else:
        logger.warning("Source file not found: %s", src)
