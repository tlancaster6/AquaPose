"""COCO-to-YOLO conversion functions for OBB and Pose datasets."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from .geometry import (
    extrapolate_edge_keypoints,
    format_obb_annotation,
    format_pose_annotation,
    pca_obb,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEYPOINT_NAMES = ["nose", "head", "spine1", "spine2", "spine3", "tail"]
N_KEYPOINTS = 6

# ---------------------------------------------------------------------------
# COCO parsing
# ---------------------------------------------------------------------------


def load_coco(path: Path) -> dict:
    """Load and validate a COCO keypoint JSON file.

    Builds ``image_id -> image_info`` and ``image_id -> list[annotation]``
    lookup dicts and stores them under the keys ``_image_lookup`` and
    ``_ann_lookup`` in the returned dict.

    Args:
        path: Path to COCO JSON file.

    Returns:
        COCO dict with additional ``_image_lookup`` and ``_ann_lookup`` keys.

    Raises:
        ValueError: If the JSON is missing required top-level keys.
    """
    with open(path) as f:
        coco: dict = json.load(f)

    for key in ("images", "annotations"):
        if key not in coco:
            msg = f"COCO JSON missing required key: {key!r}"
            raise ValueError(msg)

    image_lookup: dict[int, dict] = {img["id"]: img for img in coco["images"]}
    ann_lookup: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        ann_lookup.setdefault(ann["image_id"], []).append(ann)

    coco["_image_lookup"] = image_lookup
    coco["_ann_lookup"] = ann_lookup
    return coco


def parse_keypoints(
    ann: dict, n_keypoints: int = N_KEYPOINTS
) -> tuple[np.ndarray, np.ndarray]:
    """Extract keypoint coordinates and visibility from a COCO annotation.

    COCO keypoint format: ``[x, y, v, x, y, v, ...]`` where ``v=0`` means
    not labelled, ``v=1`` means labelled but not visible, ``v=2`` means
    labelled and visible. Any ``v > 0`` is treated as visible here.

    Args:
        ann: COCO annotation dict with a ``keypoints`` field.
        n_keypoints: Expected number of keypoints.

    Returns:
        Tuple of:
        - coords: float64 array of shape ``(N, 2)`` with (x, y) coordinates.
        - visible: bool array of shape ``(N,)``, True if keypoint is visible.
    """
    raw = ann.get("keypoints", [])
    coords = np.zeros((n_keypoints, 2), dtype=np.float64)
    visible = np.zeros(n_keypoints, dtype=bool)

    for k in range(n_keypoints):
        base = k * 3
        if base + 2 < len(raw):
            x, y, v = raw[base], raw[base + 1], raw[base + 2]
            coords[k] = (x, y)
            visible[k] = v > 0

    return coords, visible


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------


def compute_arc_length(coords: np.ndarray, visible: np.ndarray) -> float | None:
    """Sum Euclidean distances between consecutive visible keypoints.

    Args:
        coords: float array of shape ``(N, 2)`` with (x, y) pixel coordinates.
        visible: bool array of shape ``(N,)``, True if the keypoint is visible.

    Returns:
        Total arc length in pixels, or None if fewer than 2 visible keypoints.
    """
    vis_pts = coords[visible]
    if len(vis_pts) < 2:
        return None

    total = 0.0
    prev: np.ndarray | None = None
    for i in range(len(coords)):
        if not visible[i]:
            continue
        pt = coords[i]
        if prev is not None:
            total += float(np.linalg.norm(pt - prev))
        prev = pt

    return total


def compute_median_arc_length(
    annotations: list[dict], n_keypoints: int = N_KEYPOINTS
) -> float:
    """Compute the median arc length over fully-visible fish annotations.

    Args:
        annotations: List of COCO annotation dicts.
        n_keypoints: Required number of visible keypoints.

    Returns:
        Median arc length in pixels.

    Raises:
        ValueError: If no annotation has all keypoints visible.
    """
    lengths: list[float] = []
    for ann in annotations:
        coords, visible = parse_keypoints(ann, n_keypoints)
        if visible.sum() < n_keypoints:
            continue
        arc = compute_arc_length(coords, visible)
        if arc is not None:
            lengths.append(arc)

    if not lengths:
        msg = (
            "No annotations with all keypoints visible found -- cannot compute "
            "median arc length."
        )
        raise ValueError(msg)

    return float(np.median(lengths))


def affine_warp_crop(
    image: np.ndarray,
    obb_corners: np.ndarray,
    crop_w: int,
    crop_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp OBB region to an axis-aligned rectangle.

    Args:
        image: BGR image array of shape ``(H, W, 3)``.
        obb_corners: float64 array of shape ``(4, 2)`` with OBB corner
            coordinates in image space (TL, TR, BR, BL).
        crop_w: Output crop width in pixels.
        crop_h: Output crop height in pixels.

    Returns:
        Tuple of:
        - warped: uint8 BGR crop of shape ``(crop_h, crop_w, 3)``.
        - affine: float64 affine matrix of shape ``(2, 3)``.
    """
    src = np.array([obb_corners[0], obb_corners[1], obb_corners[3]], dtype=np.float32)
    dst = np.array([[0, 0], [crop_w - 1, 0], [0, crop_h - 1]], dtype=np.float32)

    affine = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(image, affine, (crop_w, crop_h))
    return warped, affine.astype(np.float64)


def transform_keypoints(
    coords: np.ndarray,
    visible: np.ndarray,
    affine_matrix: np.ndarray,
    crop_w: int,
    crop_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a 2x3 affine matrix to keypoint coordinates.

    Args:
        coords: float array of shape ``(N, 2)`` with (x, y) pixel coordinates.
        visible: bool array of shape ``(N,)``, True if keypoint is visible.
        affine_matrix: float64 affine transform matrix of shape ``(2, 3)``.
        crop_w: Output crop width in pixels.
        crop_h: Output crop height in pixels.

    Returns:
        Tuple of:
        - coords_out: float64 array of shape ``(N, 2)``.
        - visible_out: bool array of shape ``(N,)``.
    """
    n = len(coords)
    ones = np.ones((n, 1), dtype=np.float64)
    pts_h = np.hstack([coords, ones])

    transformed = (affine_matrix @ pts_h.T).T

    oob = (
        (transformed[:, 0] < 0)
        | (transformed[:, 0] >= crop_w)
        | (transformed[:, 1] < 0)
        | (transformed[:, 1] >= crop_h)
    )

    coords_out = np.clip(transformed, [0, 0], [crop_w - 1, crop_h - 1])
    visible_out = visible & ~oob

    return coords_out, visible_out


# ---------------------------------------------------------------------------
# Temporal splitting
# ---------------------------------------------------------------------------


def parse_frame_index(filename: str) -> int:
    """Extract the frame index from an AquaPose image filename.

    The frame index is the integer after the last underscore in the
    filename stem.  For example,
    ``"e3v82e0-20241005T130000_657000.png"`` yields ``657000``.

    Args:
        filename: Image filename (with or without directory path).

    Returns:
        Integer frame index.

    Raises:
        ValueError: If the stem does not contain an underscore followed
            by a parseable integer.
    """
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) < 2:
        msg = f"Cannot parse frame index from filename: {filename!r}"
        raise ValueError(msg)
    try:
        return int(parts[-1])
    except ValueError:
        msg = f"Cannot parse frame index from filename: {filename!r}"
        raise ValueError(msg) from None


def temporal_split(
    image_ids: list[int],
    image_lookup: dict[int, dict],
    val_fraction: float = 0.2,
) -> tuple[set[int], set[int]]:
    """Split image IDs by frame index, putting the latest frames into val.

    All cameras sharing a frame index stay in the same split.

    Args:
        image_ids: List of COCO image IDs.
        image_lookup: Mapping from image ID to image info dict (must
            contain ``"file_name"``).
        val_fraction: Fraction of unique frame indices to put in val.

    Returns:
        Tuple of ``(train_ids, val_ids)`` as sets.
    """
    # Group image IDs by frame index
    frame_to_ids: dict[int, list[int]] = {}
    for img_id in image_ids:
        file_name = image_lookup[img_id]["file_name"]
        frame_idx = parse_frame_index(file_name)
        frame_to_ids.setdefault(frame_idx, []).append(img_id)

    sorted_frames = sorted(frame_to_ids.keys())
    n_val = max(1, int(len(sorted_frames) * val_fraction))

    val_frames = set(sorted_frames[-n_val:])
    train_ids: set[int] = set()
    val_ids: set[int] = set()

    for frame_idx, ids in frame_to_ids.items():
        if frame_idx in val_frames:
            val_ids.update(ids)
        else:
            train_ids.update(ids)

    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_obb_dataset(
    coco: dict,
    images_dir: Path,
    output_dir: Path,
    median_arc: float,
    lateral_ratio: float,
    edge_factor: float,
    val_split: float,
    seed: int,
    n_keypoints: int = N_KEYPOINTS,
    split_mode: str = "random",
) -> tuple[int, int]:
    """Generate a YOLO-OBB dataset from COCO keypoint annotations.

    Args:
        coco: Loaded COCO dict (from :func:`load_coco`).
        images_dir: Directory containing source images.
        output_dir: Root output directory.
        median_arc: Median fish arc length in pixels.
        lateral_ratio: Fraction of median arc length for lateral OBB padding.
        edge_factor: Threshold multiplier for edge extrapolation.
        val_split: Fraction of images for validation.
        seed: Random seed for reproducible split.
        n_keypoints: Number of keypoints.
        split_mode: Split strategy. ``"random"`` (default) shuffles
            images randomly. ``"temporal"`` groups by frame index and
            puts the latest frames into val.

    Returns:
        Tuple of (n_train, n_val) image counts.
    """
    lateral_pad = median_arc * lateral_ratio

    obb_root = output_dir / "obb"
    for split in ("train", "val"):
        (obb_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (obb_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_lookup: dict[int, dict] = coco["_image_lookup"]
    ann_lookup: dict[int, list[dict]] = coco["_ann_lookup"]

    all_image_ids = list(image_lookup.keys())

    if split_mode == "temporal":
        train_ids, val_ids = temporal_split(all_image_ids, image_lookup, val_split)
    else:
        rng = random.Random(seed)
        rng.shuffle(all_image_ids)
        n_val = (
            max(1, int(len(all_image_ids) * val_split)) if len(all_image_ids) > 1 else 0
        )
        val_ids = set(all_image_ids[:n_val])
        train_ids = set(all_image_ids[n_val:])

    counts: dict[str, int] = {"train": 0, "val": 0}

    for img_id, split in [(i, "val") for i in val_ids] + [
        (i, "train") for i in train_ids
    ]:
        img_info = image_lookup[img_id]
        file_name = img_info["file_name"]
        img_path = images_dir / file_name

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        annotations = ann_lookup.get(img_id, [])
        label_lines: list[str] = []

        for ann in annotations:
            coords, visible = parse_keypoints(ann, n_keypoints)
            if visible.sum() == 0:
                continue

            coords_ext, visible_ext = extrapolate_edge_keypoints(
                coords, visible, img_w, img_h, lateral_pad, edge_factor
            )
            corners = pca_obb(coords_ext, visible_ext, lateral_pad)
            row = format_obb_annotation(corners, img_w, img_h)
            label_lines.append(" ".join(str(v) for v in row))

        stem = Path(file_name).stem
        dest_img = obb_root / "images" / split / Path(file_name).name
        shutil.copy2(str(img_path), str(dest_img))

        label_path = obb_root / "labels" / split / f"{stem}.txt"
        label_path.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
        )

        counts[split] += 1

    dataset_yaml = {
        "path": str(obb_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "fish"},
    }
    yaml_path = obb_root / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    return counts["train"], counts["val"]


def generate_pose_dataset(
    coco: dict,
    images_dir: Path,
    output_dir: Path,
    median_arc: float,
    lateral_ratio: float,
    edge_factor: float,
    crop_w: int,
    crop_h: int,
    min_visible: int,
    val_split: float,
    seed: int,
    split_mode: str = "random",
) -> tuple[int, int]:
    """Generate a YOLO-Pose dataset with affine-warped crops.

    Args:
        coco: Loaded COCO dict (from :func:`load_coco`).
        images_dir: Directory containing source images.
        output_dir: Root output directory.
        median_arc: Median fish arc length in pixels.
        lateral_ratio: Fraction of median arc length for lateral OBB padding.
        edge_factor: Threshold multiplier for edge extrapolation.
        crop_w: Output crop width in pixels.
        crop_h: Output crop height in pixels.
        min_visible: Minimum number of visible keypoints to include.
        val_split: Fraction of crops for validation.
        seed: Random seed for reproducible split.
        split_mode: Split strategy. ``"random"`` (default) shuffles
            crops randomly. ``"temporal"`` groups source images by frame
            index and puts crops from the latest frames into val.

    Returns:
        Tuple of (n_train, n_val) crop counts.
    """
    lateral_pad = median_arc * lateral_ratio

    pose_root = output_dir / "pose"
    tmp_dir = output_dir / "pose" / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (pose_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (pose_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_lookup: dict[int, dict] = coco["_image_lookup"]
    ann_lookup: dict[int, list[dict]] = coco["_ann_lookup"]

    crop_entries: list[tuple[str, list[list[float]], int]] = []
    n_keypoints = 0

    for img_id, img_info in image_lookup.items():
        file_name = img_info["file_name"]
        img_path = images_dir / file_name

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        annotations = ann_lookup.get(img_id, [])
        img_stem = Path(file_name).stem

        for ann_idx, ann in enumerate(annotations):
            coords, visible = parse_keypoints(ann)
            if int(visible.sum()) < min_visible:
                continue

            n_keypoints = len(coords)

            coords_ext, visible_ext = extrapolate_edge_keypoints(
                coords, visible, img_w, img_h, lateral_pad, edge_factor
            )
            corners = pca_obb(coords_ext, visible_ext, lateral_pad)

            warped, affine_mat = affine_warp_crop(img, corners, crop_w, crop_h)

            pose_rows: list[list[float]] = []
            for other_ann in annotations:
                other_coords, other_vis = parse_keypoints(other_ann)
                kp_crop, vis_crop = transform_keypoints(
                    other_coords, other_vis, affine_mat, crop_w, crop_h
                )
                if int(vis_crop.sum()) < min_visible:
                    continue

                crop_obb = pca_obb(kp_crop, vis_crop, lateral_pad)
                crop_obb[:, 0] = np.clip(crop_obb[:, 0], 0, crop_w - 1)
                crop_obb[:, 1] = np.clip(crop_obb[:, 1], 0, crop_h - 1)
                x_min, y_min = crop_obb.min(axis=0)
                x_max, y_max = crop_obb.max(axis=0)
                cx = (x_min + x_max) / 2.0 / crop_w
                cy = (y_min + y_max) / 2.0 / crop_h
                bw = (x_max - x_min) / crop_w
                bh = (y_max - y_min) / crop_h

                pose_rows.append(
                    format_pose_annotation(
                        cx, cy, bw, bh, kp_crop, vis_crop, crop_w, crop_h
                    )
                )

            if not pose_rows:
                continue

            crop_stem = f"{img_stem}_{ann_idx:03d}"
            crop_img_path = tmp_dir / f"{crop_stem}.jpg"
            cv2.imwrite(str(crop_img_path), warped)
            crop_entries.append((crop_stem, pose_rows, n_keypoints))

    if split_mode == "temporal":
        # Group crops by frame index from their source image stem
        frame_to_stems: dict[int, list[str]] = {}
        for crop_stem, _, _ in crop_entries:
            # crop_stem format: "{img_stem}_{ann_idx:03d}"
            # We need the original image stem to extract frame index
            # The ann_idx is always 3 digits, so rsplit on _ to get it
            img_stem_part = crop_stem.rsplit("_", 1)[0]
            frame_idx = parse_frame_index(img_stem_part + ".tmp")
            frame_to_stems.setdefault(frame_idx, []).append(crop_stem)

        sorted_frames = sorted(frame_to_stems.keys())
        n_val_frames = max(1, int(len(sorted_frames) * val_split))
        val_frames = set(sorted_frames[-n_val_frames:])
        val_set: set[str] = set()
        for frame_idx in val_frames:
            val_set.update(frame_to_stems[frame_idx])
    else:
        rng = random.Random(seed)
        rng.shuffle(crop_entries)
        n_val = (
            max(1, int(len(crop_entries) * val_split)) if len(crop_entries) > 1 else 0
        )
        val_set = set(stem for stem, _, _ in crop_entries[:n_val])

    counts: dict[str, int] = {"train": 0, "val": 0}

    for crop_stem, pose_rows, _n_kp in crop_entries:
        split = "val" if crop_stem in val_set else "train"
        src_img = tmp_dir / f"{crop_stem}.jpg"
        dst_img = pose_root / "images" / split / f"{crop_stem}.jpg"
        if src_img.exists():
            shutil.move(str(src_img), str(dst_img))

        label_path = pose_root / "labels" / split / f"{crop_stem}.txt"
        label_lines = [" ".join(str(v) for v in row) for row in pose_rows]
        label_path.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )

        counts[split] += 1

    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    kpt_n = n_keypoints if n_keypoints > 0 else N_KEYPOINTS

    dataset_yaml = {
        "path": str(pose_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "fish"},
        "kpt_shape": [kpt_n, 3],
        "flip_idx": list(range(kpt_n)),
    }
    yaml_path = pose_root / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    return counts["train"], counts["val"]
