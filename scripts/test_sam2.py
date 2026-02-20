"""Evaluate SAM2 pseudo-label quality against manual ground truth masks.

Loads SAM2 pseudo-label masks from a Label Studio tasks JSON (produced by
verify_pseudo_labels.py generate), loads user-annotated binary PNG ground
truth masks, computes per-frame mask IoU, and saves side-by-side comparison
images.

Usage:
    # Annotate GT masks first (one binary PNG per fish per frame):
    #   {frame_id}_gt_0.png, {frame_id}_gt_1.png, ...  in --gt-dir
    #
    # Then run:
    hatch run python scripts/test_sam2.py --gt-dir output/sam2_gt

    # Full example with explicit paths:
    hatch run python scripts/test_sam2.py \\
        --pseudo-labels-dir output/verify_pseudo_labels \\
        --gt-dir output/sam2_gt \\
        --output-dir output/test_sam2 \\
        --threshold 0.70
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np


def load_gt_masks(gt_dir: Path) -> dict[str, list[np.ndarray]]:
    """Load ground truth binary PNG masks grouped by frame_id.

    Args:
        gt_dir: Directory containing binary PNG files named
            ``{frame_id}_gt_{n}.png``.

    Returns:
        Dict mapping frame_id -> list of uint8 binary masks (0/255).
    """
    gt_dir = Path(gt_dir)
    pattern = re.compile(r"^(.+)_gt_(\d+)\.png$")

    gt_masks: dict[str, list[tuple[int, np.ndarray]]] = {}
    for png_path in sorted(gt_dir.glob("*.png")):
        m = pattern.match(png_path.name)
        if not m:
            print(f"  WARNING: Skipping unrecognised GT filename: {png_path.name}")
            continue
        frame_id = m.group(1)
        idx = int(m.group(2))
        mask = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  WARNING: Cannot read {png_path}")
            continue
        # Normalise to 0/255 binary
        mask_bin = np.where(mask > 127, 255, 0).astype(np.uint8)
        gt_masks.setdefault(frame_id, []).append((idx, mask_bin))

    # Sort each frame's masks by index and drop the index
    return {fid: [m for _, m in sorted(pairs)] for fid, pairs in gt_masks.items()}


def decode_label_studio_rle(rle: list[int], width: int, height: int) -> np.ndarray:
    """Decode a Label Studio brush RLE to a binary uint8 mask.

    Label Studio's mask2rle encodes 4-channel RGBA flat arrays.  Each run
    describes a count of identical bytes in column-major (Fortran) order.

    Args:
        rle: List of integer byte values as produced by mask2rle.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Binary mask (uint8, 0/255) of shape (height, width).
    """
    # Decode flat byte sequence
    flat = np.array(rle, dtype=np.uint8)
    expected = width * height * 4
    if flat.size != expected:
        # Pad or truncate defensively
        if flat.size < expected:
            flat = np.pad(flat, (0, expected - flat.size))
        else:
            flat = flat[:expected]

    # Reshape to (height, width, 4) — Label Studio stores column-major
    # (Fortran order): columns iterate fastest.
    rgba = flat.reshape((width, height, 4), order="C")
    # Transpose to (height, width, 4)
    rgba = rgba.transpose(1, 0, 2)

    # Alpha channel encodes presence; any non-zero alpha means foreground
    alpha = rgba[:, :, 3]
    return np.where(alpha > 0, 255, 0).astype(np.uint8)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute intersection-over-union between two binary masks.

    Args:
        mask_a: Binary mask (any non-zero = foreground).
        mask_b: Binary mask (any non-zero = foreground), same shape as mask_a.

    Returns:
        IoU as float in [0, 1].
    """
    a = mask_a > 0
    b = mask_b > 0
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def greedy_match(
    gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]
) -> list[tuple[int, int, float]]:
    """Greedily match GT masks to prediction masks by descending IoU.

    Args:
        gt_masks: List of ground truth binary masks.
        pred_masks: List of SAM2 prediction binary masks.

    Returns:
        List of (gt_idx, pred_idx, iou) for each matched pair.
    """
    if not gt_masks or not pred_masks:
        return []

    # Build pairwise IoU matrix
    pairs: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_masks):
        for pi, pred in enumerate(pred_masks):
            # Resize pred to match GT if dimensions differ
            if pred.shape != gt.shape:
                pred_resized = cv2.resize(
                    pred,
                    (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                pred_resized = pred
            iou = mask_iou(gt, pred_resized)
            pairs.append((iou, gi, pi))

    # Sort descending by IoU, greedily assign
    pairs.sort(reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for iou, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matches.append((gi, pi, iou))
        matched_gt.add(gi)
        matched_pred.add(pi)

    return matches


def draw_mask_contours(
    image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]
) -> np.ndarray:
    """Draw contours of a binary mask onto an image copy.

    Args:
        image: BGR image to draw on (not modified in place).
        mask: Binary mask (uint8, 0/255).
        color: BGR contour colour tuple.

    Returns:
        Image copy with contours drawn.
    """
    canvas = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, 2)
    return canvas


def make_side_by_side(
    source_frame: np.ndarray,
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    matches: list[tuple[int, int, float]],
    frame_id: str,
    unmatched_gt: list[int],
    unmatched_pred: list[int],
) -> np.ndarray:
    """Build a side-by-side comparison image.

    Left panel: source frame + GT mask contours (red).
    Right panel: source frame + SAM2 mask contours (green).
    Matched pairs show per-pair IoU text on both panels.

    Args:
        source_frame: BGR source image.
        gt_masks: Ground truth binary masks.
        pred_masks: SAM2 prediction masks.
        matches: Output from greedy_match.
        frame_id: Frame identifier string (for title).
        unmatched_gt: GT indices not matched.
        unmatched_pred: Prediction indices not matched.

    Returns:
        Side-by-side BGR image.
    """
    left = source_frame.copy()
    right = source_frame.copy()

    # Draw all GT contours in red (BGR: 0,0,255)
    for mask in gt_masks:
        resized = (
            cv2.resize(
                mask,
                (source_frame.shape[1], source_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            if mask.shape[:2] != source_frame.shape[:2]
            else mask
        )
        left = draw_mask_contours(left, resized, (0, 0, 255))

    # Draw all SAM2 contours in green (BGR: 0,255,0)
    for mask in pred_masks:
        resized = (
            cv2.resize(
                mask,
                (source_frame.shape[1], source_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            if mask.shape[:2] != source_frame.shape[:2]
            else mask
        )
        right = draw_mask_contours(right, resized, (0, 255, 0))

    # Annotate IoU for each match (compute centroid of GT mask for label pos)
    for gi, _pi, iou in matches:
        gt_mask = gt_masks[gi]
        gt_resized = (
            cv2.resize(
                gt_mask,
                (source_frame.shape[1], source_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            if gt_mask.shape[:2] != source_frame.shape[:2]
            else gt_mask
        )

        ys, xs = np.where(gt_resized > 0)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            label = f"IoU={iou:.2f}"
            cv2.putText(
                left,
                label,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                right,
                label,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    # Add panel labels
    label_h = 24
    _h, w = source_frame.shape[:2]
    bar = np.zeros((label_h, w * 2, 3), dtype=np.uint8)
    cv2.putText(
        bar,
        f"{frame_id} | GT (red)",
        (4, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        bar,
        "SAM2 (green)",
        (w + 4, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    side_by_side = np.concatenate([left, right], axis=1)
    return np.concatenate([bar, side_by_side], axis=0)


def load_sam2_masks_from_tasks(
    pseudo_labels_dir: Path,
) -> dict[str, list[np.ndarray]]:
    """Load SAM2 prediction masks from the Label Studio tasks JSON.

    Args:
        pseudo_labels_dir: Directory containing ``*_tasks.json``.

    Returns:
        Dict mapping frame_id -> list of decoded binary masks (0/255).

    Raises:
        FileNotFoundError: If no tasks JSON is found in the directory.
    """
    import json

    pseudo_labels_dir = Path(pseudo_labels_dir)
    tasks_files = list(pseudo_labels_dir.glob("*_tasks.json"))
    if not tasks_files:
        raise FileNotFoundError(f"No *_tasks.json found in {pseudo_labels_dir}")
    tasks_path = tasks_files[0]
    print(f"Loading SAM2 masks from: {tasks_path}")

    with open(tasks_path) as f:
        tasks = json.load(f)

    result: dict[str, list[np.ndarray]] = {}
    for task in tasks:
        frame_id = task["data"]["frame_id"]
        predictions = task.get("predictions", [])
        if not predictions:
            result[frame_id] = []
            continue

        masks: list[np.ndarray] = []
        for pred_result in predictions[0].get("result", []):
            if pred_result.get("type") != "brushlabels":
                continue
            value = pred_result["value"]
            rle = value.get("rle", [])
            w = pred_result.get("original_width", 0)
            h = pred_result.get("original_height", 0)
            if rle and w > 0 and h > 0:
                mask = decode_label_studio_rle(rle, w, h)
                masks.append(mask)
        result[frame_id] = masks

    return result


def find_source_frame(pseudo_labels_dir: Path, frame_id: str) -> np.ndarray | None:
    """Locate and load a source frame image by frame_id.

    Searches in ``pseudo_labels_dir/source_frames/`` and
    ``pseudo_labels_dir/images/`` for a file whose stem matches frame_id.

    Args:
        pseudo_labels_dir: Root of the pseudo-labels output directory.
        frame_id: Frame identifier string.

    Returns:
        BGR image array, or None if not found.
    """
    for subdir in ["source_frames", "images"]:
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = pseudo_labels_dir / subdir / f"{frame_id}{ext}"
            if candidate.exists():
                img = cv2.imread(str(candidate))
                if img is not None:
                    return img
    return None


def main() -> None:
    """Entry point for SAM2 pseudo-label evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SAM2 pseudo-label quality against manual ground truth masks."
        )
    )
    parser.add_argument(
        "--pseudo-labels-dir",
        type=Path,
        default=Path("output/verify_pseudo_labels"),
        help="Directory containing the *_tasks.json from verify_pseudo_labels.py generate",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        required=True,
        help=(
            "Directory of GT binary PNG files named {frame_id}_gt_{n}.png "
            "(white=fish, black=background)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/test_sam2"),
        help="Directory for side-by-side comparison images",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Mean IoU threshold for PASS/FAIL (default 0.70)",
    )
    args = parser.parse_args()

    # --- Validate GT dir ---
    gt_dir: Path = args.gt_dir
    if not gt_dir.exists():
        print(f"ERROR: GT directory does not exist: {gt_dir}")
        sys.exit(2)

    gt_by_frame = load_gt_masks(gt_dir)
    if not gt_by_frame:
        print(
            f"GT directory '{gt_dir}' contains no recognised PNG files "
            "(expected names like {frame_id}_gt_0.png)."
        )
        print("Please create GT masks before running this script.")
        sys.exit(0)

    print(f"Loaded GT masks for {len(gt_by_frame)} frames.")

    # --- Load SAM2 masks ---
    sam2_by_frame = load_sam2_masks_from_tasks(args.pseudo_labels_dir)
    print(f"Loaded SAM2 predictions for {len(sam2_by_frame)} frames from tasks JSON.")

    # --- Output dir ---
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-frame evaluation ---
    all_ious: list[float] = []
    print()
    print(f"{'Frame':40s}  {'GT':>4}  {'Pred':>4}  {'Match':>5}  {'MeanIoU':>8}")
    print("-" * 72)

    for frame_id in sorted(gt_by_frame.keys()):
        gt_masks = gt_by_frame[frame_id]
        pred_masks = sam2_by_frame.get(frame_id, [])

        matches = greedy_match(gt_masks, pred_masks)

        matched_gt_ids = {gi for gi, _, _ in matches}
        matched_pred_ids = {pi for _, pi, _ in matches}
        unmatched_gt = [i for i in range(len(gt_masks)) if i not in matched_gt_ids]
        unmatched_pred = [
            i for i in range(len(pred_masks)) if i not in matched_pred_ids
        ]

        # Unmatched GT masks count as IoU=0
        frame_ious = [iou for _, _, iou in matches] + [0.0] * len(unmatched_gt)
        frame_mean = float(np.mean(frame_ious)) if frame_ious else 0.0
        all_ious.extend(frame_ious)

        num_fp = len(unmatched_pred)
        fp_str = f" (+{num_fp} FP)" if num_fp > 0 else ""

        print(
            f"{frame_id:40s}  {len(gt_masks):>4}  "
            f"{len(pred_masks):>4}  {len(matches):>5}  "
            f"{frame_mean:>8.3f}{fp_str}"
        )

        # --- Side-by-side visual ---
        source_img = find_source_frame(args.pseudo_labels_dir, frame_id)
        if source_img is None:
            # Create a placeholder grey image
            if gt_masks:
                h, w = gt_masks[0].shape[:2]
            elif pred_masks:
                h, w = pred_masks[0].shape[:2]
            else:
                h, w = 480, 640
            source_img = np.full((h, w, 3), 80, dtype=np.uint8)
            cv2.putText(
                source_img,
                "Source image not found",
                (10, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        panel = make_side_by_side(
            source_img,
            gt_masks,
            pred_masks,
            matches,
            frame_id,
            unmatched_gt,
            unmatched_pred,
        )
        out_path = output_dir / f"{frame_id}_comparison.jpg"
        cv2.imwrite(str(out_path), panel)

    # --- Overall summary ---
    print("-" * 72)
    if not all_ious:
        print("No matched pairs found (no GT frames matched predictions).")
        sys.exit(2)

    overall_mean = float(np.mean(all_ious))
    threshold = args.threshold
    status = "PASS" if overall_mean >= threshold else "FAIL"

    print()
    print(f"Overall mean IoU : {overall_mean:.4f}")
    print(f"Threshold        : {threshold:.2f}")
    print(f"Status           : {status}")
    print()
    print(f"Side-by-side comparisons saved to: {output_dir}/")

    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
