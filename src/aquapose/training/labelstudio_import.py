"""Import Label Studio JSON exports back to YOLO-OBB/Pose format."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path


def _rotated_rect_to_obb_corners(
    x_pct: float,
    y_pct: float,
    w_pct: float,
    h_pct: float,
    rotation_deg: float,
    img_w: int,
    img_h: int,
) -> list[float]:
    """Convert Label Studio rotated rectangle to 4 normalized corner points.

    Label Studio stores (x, y) as the top-left anchor in percentages, with
    rotation in degrees around that anchor.

    Args:
        x_pct: Top-left x position (percentage 0-100).
        y_pct: Top-left y position (percentage 0-100).
        w_pct: Width (percentage 0-100).
        h_pct: Height (percentage 0-100).
        rotation_deg: Rotation in degrees.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        [x1,y1, x2,y2, x3,y3, x4,y4] normalized 0-1, order TL,TR,BR,BL.
    """
    # Convert percentages to pixels
    tl_x = x_pct / 100 * img_w
    tl_y = y_pct / 100 * img_h
    w_px = w_pct / 100 * img_w
    h_px = h_pct / 100 * img_h

    angle_rad = math.radians(rotation_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # TL is the anchor (fixed point). Compute other corners by rotating
    # the width/height vectors around TL.
    # TR = TL + width * (cos, sin)
    # BL = TL + height * (-sin, cos)
    # BR = TL + width * (cos, sin) + height * (-sin, cos)
    tr_x = tl_x + w_px * cos_a
    tr_y = tl_y + w_px * sin_a
    bl_x = tl_x - h_px * sin_a
    bl_y = tl_y + h_px * cos_a
    br_x = tr_x - h_px * sin_a
    br_y = tr_y + h_px * cos_a

    # Normalize to 0-1
    return [
        tl_x / img_w,
        tl_y / img_h,
        tr_x / img_w,
        tr_y / img_h,
        br_x / img_w,
        br_y / img_h,
        bl_x / img_w,
        bl_y / img_h,
    ]


def _get_class_id(label_name: str, class_names: dict[int, str]) -> int:
    """Reverse-lookup class ID from label name."""
    for cls_id, name in class_names.items():
        if name == label_name:
            return cls_id
    return 0


def _convert_obb_task(
    results: list[dict], img_w: int, img_h: int, class_names: dict[int, str]
) -> list[str]:
    """Convert Label Studio results to YOLO-OBB label lines."""
    lines = []
    for r in results:
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        label = v.get("rectanglelabels", ["fish"])[0]
        cls_id = _get_class_id(label, class_names)
        corners = _rotated_rect_to_obb_corners(
            v["x"],
            v["y"],
            v["width"],
            v["height"],
            v.get("rotation", 0),
            img_w,
            img_h,
        )
        coords = " ".join(f"{c:.6f}" for c in corners)
        lines.append(f"{cls_id} {coords}")
    return lines


def _resolve_kpt_index(name: str, kpt_names: list[str] | None) -> int | None:
    """Resolve a keypoint label name to its index.

    Supports both numbered names (``"kp0"``) and semantic names (``"nose"``).
    """
    if kpt_names and name in kpt_names:
        return kpt_names.index(name)
    if name.startswith("kp") and name[2:].isdigit():
        return int(name[2:])
    return None


def _convert_pose_task(
    results: list[dict],
    img_w: int,
    img_h: int,
    class_names: dict[int, str],
    num_kpts: int,
    kpt_names: list[str] | None = None,
) -> list[str]:
    """Convert Label Studio results to YOLO-Pose label lines.

    Groups rectanglelabels (bounding box) with their associated keypoints.
    Keypoints are matched to boxes by spatial containment.
    """
    # Separate boxes and keypoints
    boxes = [r for r in results if r.get("type") == "rectanglelabels"]
    keypoints = [r for r in results if r.get("type") == "keypointlabels"]

    lines = []
    for box in boxes:
        v = box["value"]
        label = v.get("rectanglelabels", ["fish"])[0]
        cls_id = _get_class_id(label, class_names)

        # Box center and dimensions (normalized)
        x_pct, y_pct = v["x"], v["y"]
        w_pct, h_pct = v["width"], v["height"]
        cx = (x_pct + w_pct / 2) / 100
        cy = (y_pct + h_pct / 2) / 100
        w = w_pct / 100
        h = h_pct / 100

        # Collect keypoints belonging to this box
        kpt_map: dict[int, tuple[float, float]] = {}
        # Single-box shortcut: assign all keypoints directly (no spatial
        # matching needed for single-fish-per-crop images).
        use_spatial = len(boxes) > 1
        for kp in keypoints:
            kv = kp["value"]
            kp_labels = kv.get("keypointlabels", [])
            if not kp_labels:
                continue
            kp_idx = _resolve_kpt_index(kp_labels[0], kpt_names)
            if kp_idx is None:
                continue

            if use_spatial:
                # Check spatial containment (loose — within box bounds)
                kp_x_pct = kv["x"]
                kp_y_pct = kv["y"]
                if not (
                    x_pct - 5 <= kp_x_pct <= x_pct + w_pct + 5
                    and y_pct - 5 <= kp_y_pct <= y_pct + h_pct + 5
                ):
                    continue

            kpt_map[kp_idx] = (kv["x"] / 100, kv["y"] / 100)

        # Build keypoint string
        kpt_parts = []
        for k in range(num_kpts):
            if k in kpt_map:
                kx, ky = kpt_map[k]
                kpt_parts.extend([f"{kx:.6f}", f"{ky:.6f}", "2"])
            else:
                kpt_parts.extend(["0", "0", "0"])

        coords = f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        kpts = " ".join(kpt_parts)
        lines.append(f"{cls_id} {coords} {kpts}")

    return lines


def _extract_image_name(image_url: str) -> str:
    """Extract the image filename from a Label Studio image URL."""
    # URL format: /data/local-files/?d=.../split/filename.jpg
    if "?d=" in image_url:
        return image_url.split("?d=")[-1].split("/")[-1]
    # Fallback: just take the last path segment
    return image_url.rstrip("/").split("/")[-1]


def import_labelstudio_json(
    json_path: Path,
    output_dir: Path,
    task: str,
    images_dir: Path | None = None,
    class_names: dict[int, str] | None = None,
    num_kpts: int = 6,
    kpt_names: list[str] | None = None,
) -> dict[str, int]:
    """Convert Label Studio JSON export to YOLO-format directory.

    Args:
        json_path: Path to exported Label Studio JSON.
        output_dir: Output directory (will contain images/ and labels/).
        task: ``"obb"`` or ``"pose"``.
        images_dir: Directory containing the original images. If None,
            images are not copied (labels only).
        class_names: Class index to name mapping. Defaults to ``{0: "fish"}``.
        num_kpts: Number of keypoints for pose task.
        kpt_names: Keypoint label names for resolving named keypoints
            (e.g. ``["nose", "head", ...]``). Falls back to ``"kpN"`` format.

    Returns:
        Stats dict with counts.
    """
    if class_names is None:
        class_names = {0: "fish"}

    # Auto-discover keypoint names from dataset.yaml (check next to JSON,
    # then in images_dir, then images_dir parent)
    if kpt_names is None and task == "pose":
        candidates = [json_path.parent / "dataset.yaml"]
        if images_dir is not None:
            candidates.append(images_dir / "dataset.yaml")
            candidates.append(images_dir.parent / "dataset.yaml")
        for ds_yaml_path in candidates:
            if ds_yaml_path.exists():
                import yaml

                ds_cfg = yaml.safe_load(ds_yaml_path.read_text()) or {}
                kpt_names = ds_cfg.get("kpt_names")
                if kpt_names:
                    break

    data = json.loads(json_path.read_text())

    img_out = output_dir / "images" / "train"
    lbl_out = output_dir / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_tasks = 0
    n_annotations = 0
    n_skipped = 0

    for entry in data:
        image_url = entry.get("data", {}).get("image", "")
        image_name = _extract_image_name(image_url)
        stem = Path(image_name).stem

        # Get results from annotations (corrected) or predictions (uncorrected)
        results = []
        annotations = entry.get("annotations", [])
        if annotations:
            # Use the most recent annotation
            results = annotations[-1].get("result", [])
        else:
            predictions = entry.get("predictions", [])
            if predictions:
                results = predictions[-1].get("result", [])

        if not results:
            n_skipped += 1
            continue

        # Get image dimensions from first result
        first = results[0]
        img_w = first.get("original_width", 0)
        img_h = first.get("original_height", 0)
        if img_w == 0 or img_h == 0:
            n_skipped += 1
            continue

        # Convert to YOLO format
        if task == "obb":
            lines = _convert_obb_task(results, img_w, img_h, class_names)
        elif task == "pose":
            lines = _convert_pose_task(
                results, img_w, img_h, class_names, num_kpts, kpt_names
            )
        else:
            raise ValueError(f"Unsupported task: {task}")

        if not lines:
            n_skipped += 1
            continue

        # Write label file
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        n_annotations += len(lines)

        # Copy image if source directory provided
        if images_dir is not None:
            for src_dir in [images_dir, images_dir / "train", images_dir / "val"]:
                src = src_dir / image_name
                if src.exists():
                    shutil.copy2(src, img_out / image_name)
                    break

        n_tasks += 1

    return {
        "tasks_converted": n_tasks,
        "total_annotations": n_annotations,
        "skipped": n_skipped,
    }
