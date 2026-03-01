"""Convert COCO-format keypoint annotations into Ultralytics YOLO-OBB and YOLO-Pose datasets."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

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
            raise ValueError(f"COCO JSON missing required key: {key!r}")

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

    Traverses the chain in skeleton order (nose -> head -> spine1 -> spine2 ->
    spine3 -> tail) and accumulates distances between adjacent visible
    keypoints only.

    Args:
        coords: float array of shape ``(N, 2)`` with (x, y) pixel coordinates.
        visible: bool array of shape ``(N,)``, True if the keypoint is visible.

    Returns:
        Total arc length in pixels, or None if fewer than 2 visible keypoints
        exist in the chain.
    """
    vis_pts = coords[visible]
    if len(vis_pts) < 2:
        return None

    total = 0.0
    # Walk chain in order, accumulate distances between consecutive visible pts
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

    Only annotations where all ``n_keypoints`` are visible contribute.

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
        raise ValueError(
            "No annotations with all keypoints visible found — cannot compute "
            "median arc length."
        )

    return float(np.median(lengths))


def pca_obb(
    coords: np.ndarray,
    visible: np.ndarray,
    lateral_pad: float,
) -> np.ndarray:
    """Compute a PCA-derived oriented bounding box for visible keypoints.

    Uses PCA on the visible keypoint coordinates to find the long axis of the
    fish. Projects all visible points onto the PCA axes to determine extent.
    The OBB half-width along the main axis is the max projection plus a small
    epsilon; the half-width perpendicular is ``lateral_pad``.

    Args:
        coords: float array of shape ``(N, 2)`` with (x, y) pixel coordinates.
        visible: bool array of shape ``(N,)``, True if the keypoint is visible.
        lateral_pad: Half-width of OBB in the lateral (perpendicular) direction,
            in pixels.

    Returns:
        OBB corners as float64 array of shape ``(4, 2)``, in clockwise order
        starting from the top-left corner in OBB frame (TL, TR, BR, BL in
        rotated space).
    """
    vis_pts = coords[visible]

    # Degenerate case: 0 or 1 visible keypoint — return a default 20x20 box
    if len(vis_pts) <= 1:
        if len(vis_pts) == 1:
            cx, cy = float(vis_pts[0, 0]), float(vis_pts[0, 1])
        else:
            cx, cy = 0.0, 0.0
        half = 10.0
        return np.array(
            [
                [cx - half, cy - half],
                [cx + half, cy - half],
                [cx + half, cy + half],
                [cx - half, cy + half],
            ],
            dtype=np.float64,
        )

    centroid = vis_pts.mean(axis=0)
    centered = vis_pts - centroid

    if len(vis_pts) >= 2:
        # PCA via SVD
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        main_axis = vh[0]  # first principal component (long axis)
    else:
        main_axis = np.array([1.0, 0.0])

    # Ensure main_axis is a unit vector
    norm = np.linalg.norm(main_axis)
    main_axis = np.array([1.0, 0.0]) if norm < 1e-9 else main_axis / norm

    perp_axis = np.array([-main_axis[1], main_axis[0]])

    # Project visible points onto the main axis to determine extent
    proj_main = centered @ main_axis

    eps = 2.0  # small buffer along main axis
    half_main = float(max(abs(proj_main.max()), abs(proj_main.min()))) + eps
    half_perp = lateral_pad

    # Build corners in OBB frame, then rotate back to image space
    # Order: TL, TR, BR, BL in OBB frame (main axis = horizontal)
    corners_local = np.array(
        [
            [-half_main, -half_perp],
            [half_main, -half_perp],
            [half_main, half_perp],
            [-half_main, half_perp],
        ],
        dtype=np.float64,
    )

    # Rotation matrix: columns are main_axis and perp_axis
    rot = np.stack([main_axis, perp_axis], axis=1)  # (2, 2)
    corners_world = corners_local @ rot.T + centroid  # (4, 2)

    return corners_world


def extrapolate_edge_keypoints(
    coords: np.ndarray,
    visible: np.ndarray,
    img_w: int,
    img_h: int,
    lateral_pad: float,
    edge_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend the polyline toward image edges for near-boundary keypoints.

    Checks whether the first or last visible keypoint in the chain is within
    ``lateral_pad * edge_factor`` pixels of any image boundary. If so,
    extrapolates toward the nearest edge along the local chain direction.

    Args:
        coords: float array of shape ``(N, 2)`` with (x, y) pixel coordinates.
        visible: bool array of shape ``(N,)``, True if the keypoint is visible.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        lateral_pad: Base lateral padding (half-width of OBB) in pixels.
        edge_factor: Multiplier applied to ``lateral_pad`` to set proximity
            threshold. Default is 2.0.

    Returns:
        Tuple of:
        - coords_out: Updated copy of coords with extrapolated keypoints.
        - visible_out: Updated copy of visible with extrapolated points set
            True.
    """
    coords_out = coords.copy()
    visible_out = visible.copy()

    threshold = lateral_pad * edge_factor
    vis_indices = [i for i in range(len(visible)) if visible[i]]

    if len(vis_indices) < 2:
        return coords_out, visible_out

    def _nearest_edge_dist(pt: np.ndarray) -> float:
        """Return distance to nearest image boundary."""
        x, y = pt
        return float(min(x, y, img_w - x, img_h - y))

    def _extrapolate_toward_edge(
        anchor: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Move anchor along direction until it hits the image boundary."""
        # Normalize direction
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return anchor.copy()
        d = direction / norm

        # Find how far we can travel in direction d before hitting boundary
        t_max = float("inf")
        if d[0] > 1e-9:
            t_max = min(t_max, (img_w - anchor[0]) / d[0])
        elif d[0] < -1e-9:
            t_max = min(t_max, -anchor[0] / d[0])
        if d[1] > 1e-9:
            t_max = min(t_max, (img_h - anchor[1]) / d[1])
        elif d[1] < -1e-9:
            t_max = min(t_max, -anchor[1] / d[1])

        if t_max == float("inf") or t_max < 0:
            return anchor.copy()

        new_pt = anchor + d * t_max
        # Clamp to image bounds
        new_pt[0] = float(np.clip(new_pt[0], 0, img_w - 1))
        new_pt[1] = float(np.clip(new_pt[1], 0, img_h - 1))
        return new_pt

    # Check first visible keypoint (nose end)
    first_idx = vis_indices[0]
    first_pt = coords_out[first_idx]
    if _nearest_edge_dist(first_pt) < threshold:
        # Direction: from second visible toward first visible (outward)
        second_idx = vis_indices[1]
        direction = coords_out[first_idx] - coords_out[second_idx]
        new_pt = _extrapolate_toward_edge(first_pt.copy(), direction)
        # Find the first non-visible slot before first_idx to place the new point
        # If none, overwrite first_idx position with extended version
        coords_out[first_idx] = new_pt

    # Check last visible keypoint (tail end)
    last_idx = vis_indices[-1]
    last_pt = coords_out[last_idx]
    if _nearest_edge_dist(last_pt) < threshold:
        second_last_idx = vis_indices[-2]
        direction = coords_out[last_idx] - coords_out[second_last_idx]
        new_pt = _extrapolate_toward_edge(last_pt.copy(), direction)
        coords_out[last_idx] = new_pt

    return coords_out, visible_out


def affine_warp_crop(
    image: np.ndarray,
    obb_corners: np.ndarray,
    crop_w: int,
    crop_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp OBB region to an axis-aligned rectangle.

    Maps 3 of the 4 OBB corners to corresponding destination corners of a
    ``(crop_w, crop_h)`` rectangle using ``cv2.getAffineTransform``.

    The OBB is assumed to have corners in order: TL, TR, BR, BL. The mapping
    ensures the fish is in landscape orientation (long axis = horizontal).

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
    # Source points: TL, TR, BL (3 corners for affine) — must be float32 for cv2
    src = np.array([obb_corners[0], obb_corners[1], obb_corners[3]], dtype=np.float32)
    # Destination: top-left, top-right, bottom-left of crop
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

    Out-of-bounds keypoints (after transform) are marked invisible.

    Args:
        coords: float array of shape ``(N, 2)`` with (x, y) pixel coordinates.
        visible: bool array of shape ``(N,)``, True if keypoint is visible.
        affine_matrix: float64 affine transform matrix of shape ``(2, 3)``.
        crop_w: Output crop width in pixels.
        crop_h: Output crop height in pixels.

    Returns:
        Tuple of:
        - coords_out: float64 array of shape ``(N, 2)`` with transformed
          coordinates, clamped to ``[0, crop_w) x [0, crop_h)``.
        - visible_out: bool array of shape ``(N,)``, marking OOB points
          invisible.
    """
    n = len(coords)
    # Homogeneous coordinates
    ones = np.ones((n, 1), dtype=np.float64)
    pts_h = np.hstack([coords, ones])  # (N, 3)

    transformed = (affine_matrix @ pts_h.T).T  # (N, 2)

    # Check out-of-bounds
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
# Output format functions
# ---------------------------------------------------------------------------


def format_obb_label(
    obb_corners: np.ndarray,
    img_w: int,
    img_h: int,
    class_id: int = 0,
) -> str:
    """Format one OBB annotation in Ultralytics OBB format.

    Format: ``class x1 y1 x2 y2 x3 y3 x4 y4`` with coordinates normalized
    to [0, 1] by image dimensions.

    Args:
        obb_corners: float64 array of shape ``(4, 2)`` with corner (x, y) in
            image pixel space, order: TL, TR, BR, BL.
        img_w: Image width in pixels (for normalization).
        img_h: Image height in pixels (for normalization).
        class_id: YOLO class index.

    Returns:
        Formatted label string (no trailing newline).
    """
    parts = [str(class_id)]
    for corner in obb_corners:
        x_norm = float(np.clip(corner[0] / img_w, 0.0, 1.0))
        y_norm = float(np.clip(corner[1] / img_h, 0.0, 1.0))
        parts.append(f"{x_norm:.6f}")
        parts.append(f"{y_norm:.6f}")
    return " ".join(parts)


def format_pose_label(
    cx: float,
    cy: float,
    w: float,
    h: float,
    keypoints: np.ndarray,
    visible: np.ndarray,
    crop_w: int,
    crop_h: int,
    class_id: int = 0,
) -> str:
    """Format one pose annotation in Ultralytics keypoint format.

    Format: ``class cx cy w h x1 y1 v1 x2 y2 v2 ...`` with bbox and
    keypoints normalized to [0, 1]. Invisible keypoints are output as
    ``0 0 0``. Visible keypoints use COCO visible=2 convention.

    Args:
        cx: Bounding box center x, normalized to [0, 1].
        cy: Bounding box center y, normalized to [0, 1].
        w: Bounding box width, normalized to [0, 1].
        h: Bounding box height, normalized to [0, 1].
        keypoints: float array of shape ``(N, 2)`` with (x, y) in crop pixels.
        visible: bool array of shape ``(N,)``, True if keypoint is visible.
        crop_w: Crop width in pixels (for normalization).
        crop_h: Crop height in pixels (for normalization).
        class_id: YOLO class index.

    Returns:
        Formatted label string (no trailing newline).
    """
    parts = [str(class_id), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]

    for k in range(len(keypoints)):
        if visible[k]:
            x_norm = float(np.clip(keypoints[k, 0] / crop_w, 0.0, 1.0))
            y_norm = float(np.clip(keypoints[k, 1] / crop_h, 0.0, 1.0))
            parts.extend([f"{x_norm:.6f}", f"{y_norm:.6f}", "2"])
        else:
            parts.extend(["0", "0", "0"])

    return " ".join(parts)


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
) -> tuple[int, int]:
    """Generate a YOLO-OBB dataset from COCO keypoint annotations.

    Creates ``output_dir/obb/images/{train,val}/`` and
    ``output_dir/obb/labels/{train,val}/`` with Ultralytics OBB format
    labels. Writes a ``data.yaml`` config file.

    Args:
        coco: Loaded COCO dict (from :func:`load_coco`).
        images_dir: Directory containing source images.
        output_dir: Root output directory.
        median_arc: Median fish arc length in pixels (used for lateral pad).
        lateral_ratio: Fraction of median arc length for lateral OBB padding.
        edge_factor: Threshold multiplier for edge extrapolation.
        val_split: Fraction of images for validation.
        seed: Random seed for reproducible split.

    Returns:
        Tuple of (n_train, n_val) image counts.
    """
    lateral_pad = median_arc * lateral_ratio

    # Create directory structure
    obb_root = output_dir / "obb"
    for split in ("train", "val"):
        (obb_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (obb_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_lookup: dict[int, dict] = coco["_image_lookup"]
    ann_lookup: dict[int, list[dict]] = coco["_ann_lookup"]

    # Shuffle images and split into train/val
    all_image_ids = list(image_lookup.keys())
    rng = random.Random(seed)
    rng.shuffle(all_image_ids)

    n_val = max(1, int(len(all_image_ids) * val_split)) if len(all_image_ids) > 1 else 0
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
            coords, visible = parse_keypoints(ann)
            if visible.sum() == 0:
                continue

            coords_ext, visible_ext = extrapolate_edge_keypoints(
                coords, visible, img_w, img_h, lateral_pad, edge_factor
            )
            corners = pca_obb(coords_ext, visible_ext, lateral_pad)
            label_lines.append(format_obb_label(corners, img_w, img_h))

        # Copy image and write label
        stem = Path(file_name).stem
        dest_img = obb_root / "images" / split / Path(file_name).name
        dest_lbl = obb_root / "labels" / split / f"{stem}.txt"

        shutil.copy2(str(img_path), str(dest_img))
        dest_lbl.write_text("\n".join(label_lines), encoding="utf-8")
        counts[split] += 1

    # Write data.yaml
    yaml_content = (
        "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: fish\n"
    )
    (obb_root / "data.yaml").write_text(yaml_content, encoding="utf-8")

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
) -> tuple[int, int]:
    """Generate a YOLO-Pose dataset with affine-warped crops.

    For each annotation with sufficient visible keypoints, warps the OBB
    region to an axis-aligned crop at ``(crop_w, crop_h)`` and transforms
    keypoint coordinates into crop space. Writes one crop image and one label
    file per annotation.

    Creates ``output_dir/pose/images/{train,val}/`` and
    ``output_dir/pose/labels/{train,val}/`` with Ultralytics pose format
    labels. Writes a ``data.yaml`` config file.

    Args:
        coco: Loaded COCO dict (from :func:`load_coco`).
        images_dir: Directory containing source images.
        output_dir: Root output directory.
        median_arc: Median fish arc length in pixels.
        lateral_ratio: Fraction of median arc length for lateral OBB padding.
        edge_factor: Threshold multiplier for edge extrapolation.
        crop_w: Output crop width in pixels.
        crop_h: Output crop height in pixels.
        min_visible: Minimum number of visible keypoints to include annotation.
        val_split: Fraction of crops for validation.
        seed: Random seed for reproducible split.

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

    crop_stems: list[str] = []

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

            coords_ext, visible_ext = extrapolate_edge_keypoints(
                coords, visible, img_w, img_h, lateral_pad, edge_factor
            )
            corners = pca_obb(coords_ext, visible_ext, lateral_pad)

            warped, affine_mat = affine_warp_crop(img, corners, crop_w, crop_h)
            kp_crop, vis_crop = transform_keypoints(
                coords, visible, affine_mat, crop_w, crop_h
            )

            # Single annotation per crop — bbox is full crop
            label_line = format_pose_label(
                0.5, 0.5, 1.0, 1.0, kp_crop, vis_crop, crop_w, crop_h
            )

            # Save to tmp
            crop_stem = f"{img_stem}_{ann_idx:03d}"
            crop_img_path = tmp_dir / f"{crop_stem}.jpg"
            crop_lbl_path = tmp_dir / f"{crop_stem}.txt"

            cv2.imwrite(str(crop_img_path), warped)
            crop_lbl_path.write_text(label_line, encoding="utf-8")
            crop_stems.append(crop_stem)

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(crop_stems)

    n_val = max(1, int(len(crop_stems) * val_split)) if len(crop_stems) > 1 else 0
    val_stems = set(crop_stems[:n_val])
    train_stems = set(crop_stems[n_val:])

    for stem in crop_stems:
        split = "val" if stem in val_stems else "train"
        src_img = tmp_dir / f"{stem}.jpg"
        src_lbl = tmp_dir / f"{stem}.txt"
        dst_img = pose_root / "images" / split / f"{stem}.jpg"
        dst_lbl = pose_root / "labels" / split / f"{stem}.txt"
        if src_img.exists():
            shutil.move(str(src_img), str(dst_img))
        if src_lbl.exists():
            shutil.move(str(src_lbl), str(dst_lbl))

    # Cleanup tmp
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    # Write data.yaml
    yaml_content = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n"
        "  0: fish\n"
        "kpt_shape: [6, 3]\n"
    )
    (pose_root / "data.yaml").write_text(yaml_content, encoding="utf-8")

    n_train = len(train_stems)
    n_val_actual = len(val_stems)
    return n_train, n_val_actual


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the build_yolo_training_data CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert COCO keypoint annotations into Ultralytics YOLO-OBB "
            "and YOLO-Pose training datasets."
        )
    )
    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="Path to COCO-format keypoint annotation JSON.",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        type=Path,
        help="Directory containing source images referenced in the COCO JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/yolo_training_data/",
        type=Path,
        help="Root output directory for OBB and Pose datasets (default: output/yolo_training_data/).",
    )
    parser.add_argument(
        "--crop-height",
        type=int,
        default=64,
        help="Pose crop height in pixels (default: 64).",
    )
    parser.add_argument(
        "--crop-width",
        type=int,
        default=128,
        help="Pose crop width in pixels (default: 128).",
    )
    parser.add_argument(
        "--lateral-ratio",
        type=float,
        default=0.18,
        help="OBB lateral half-width as fraction of median arc length (default: 0.18).",
    )
    parser.add_argument(
        "--min-visible",
        type=int,
        default=4,
        help="Minimum visible keypoints required for Pose crops (default: 4).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42).",
    )
    parser.add_argument(
        "--edge-threshold-factor",
        type=float,
        default=2.0,
        help="Multiplier on lateral_pad for edge extrapolation threshold (default: 2.0).",
    )
    return parser


def main() -> None:
    """Entry point for the build_yolo_training_data CLI."""
    parser = build_parser()
    args = parser.parse_args()

    annotations_path: Path = args.annotations
    images_dir: Path = args.images_dir
    output_dir: Path = args.output_dir

    if not annotations_path.exists():
        print(
            f"Error: annotations file not found at '{annotations_path}'.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not images_dir.exists():
        print(
            f"Error: images directory not found at '{images_dir}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading COCO JSON: {annotations_path}")
    coco = load_coco(annotations_path)

    all_annotations = coco.get("annotations", [])
    total_images = len(coco.get("images", []))
    total_annotations = len(all_annotations)

    print(f"  Total images      : {total_images}")
    print(f"  Total annotations : {total_annotations}")

    try:
        median_arc = compute_median_arc_length(all_annotations)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    lateral_pad = median_arc * args.lateral_ratio
    print(f"  Median arc length : {median_arc:.1f} px")
    print(f"  Lateral pad       : {lateral_pad:.1f} px")

    print("\nGenerating YOLO-OBB dataset...")
    obb_train, obb_val = generate_obb_dataset(
        coco,
        images_dir=images_dir,
        output_dir=output_dir,
        median_arc=median_arc,
        lateral_ratio=args.lateral_ratio,
        edge_factor=args.edge_threshold_factor,
        val_split=args.val_split,
        seed=args.seed,
    )

    print("\nGenerating YOLO-Pose dataset...")
    pose_train, pose_val = generate_pose_dataset(
        coco,
        images_dir=images_dir,
        output_dir=output_dir,
        median_arc=median_arc,
        lateral_ratio=args.lateral_ratio,
        edge_factor=args.edge_threshold_factor,
        crop_w=args.crop_width,
        crop_h=args.crop_height,
        min_visible=args.min_visible,
        val_split=args.val_split,
        seed=args.seed,
    )

    print("\n=== Generation complete ===")
    print(f"OBB dataset  : train={obb_train}, val={obb_val} images")
    print(f"Pose dataset : train={pose_train}, val={pose_val} crops")
    print(f"Output dir   : {output_dir.resolve()}")


if __name__ == "__main__":
    main()
