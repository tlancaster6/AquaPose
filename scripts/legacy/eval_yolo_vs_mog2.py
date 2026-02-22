"""Compare YOLO and MOG2 fish detectors on the stratified validation set.

Loads ground-truth bounding boxes from ``data/yolo_fish/labels/val/``, runs
both detectors on the corresponding images, computes recall and precision
for each detector, and saves side-by-side annotated images.

Usage
-----
After training YOLO::

    python scripts/eval_yolo_vs_mog2.py \\
        --yolo-weights output/yolo_fish/train_v1/weights/best.pt \\
        --video-dir /path/to/per-camera/videos \\
        --center-camera e3v8340

The ``--video-dir`` flag is required for a fair MOG2 comparison: each val
image is loaded from its source video so that MOG2 can be warmed up on the
preceding frames before evaluation.  If videos are unavailable, omit
``--video-dir`` to run MOG2 without warmup (and note the limitation).

Output
------
- Annotated side-by-side images saved to ``--output-dir``.
- Summary table printed to stdout:
  ``Detector | Recall | Precision | TP | FP | FN``
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for YOLO vs MOG2 evaluation.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO vs MOG2 fish detectors on the validation set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--yolo-weights",
        required=True,
        help="Path to trained YOLO best.pt weights file.",
    )
    parser.add_argument(
        "--data",
        default="data/yolo_fish/dataset.yaml",
        help="Path to dataset YAML file.",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help=(
            "Directory containing per-camera video files (for MOG2 warmup). "
            "Expected naming: {camera_id}.avi / .mp4 / .mkv. "
            "If omitted, MOG2 runs without warmup."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="output/yolo_eval",
        help="Directory to save side-by-side annotated images.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=200,
        help="Number of preceding video frames to feed as MOG2 warmup.",
    )
    parser.add_argument(
        "--center-camera",
        default=None,
        help="Camera ID of the center (top-down) camera, for reporting.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching a detection to a ground-truth box.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def load_gt_boxes(
    label_path: Path, img_w: int, img_h: int
) -> list[tuple[int, int, int, int]]:
    """Load YOLO-format ground-truth boxes from a label file.

    Each line in the file is ``class cx cy w h`` (normalised to [0,1]).

    Args:
        label_path: Path to the ``.txt`` label file.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        List of boxes in ``(x, y, w, h)`` pixel format.
    """
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _cls, cx, cy, bw, bh = parts[:5]
        cx_px = float(cx) * img_w
        cy_px = float(cy) * img_h
        bw_px = float(bw) * img_w
        bh_px = float(bh) * img_h
        x = int(cx_px - bw_px / 2)
        y = int(cy_px - bh_px / 2)
        boxes.append((x, y, int(bw_px), int(bh_px)))
    return boxes


def iou_xywh(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute intersection-over-union for two ``(x, y, w, h)`` boxes.

    Args:
        a: First box as ``(x, y, w, h)``.
        b: Second box as ``(x, y, w, h)``.

    Returns:
        IoU value in ``[0, 1]``.
    """
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = ax1 + a[2], ay1 + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = bx1 + b[2], by1 + b[3]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    gt_boxes: list[tuple[int, int, int, int]],
    det_boxes: list[tuple[int, int, int, int]],
    iou_thr: float,
) -> tuple[int, int, int]:
    """Match detections to ground truth at a given IoU threshold.

    Uses greedy matching: each GT box is matched at most once to the highest-
    IoU detection that exceeds *iou_thr*.

    Args:
        gt_boxes: Ground-truth bounding boxes ``(x, y, w, h)``.
        det_boxes: Detected bounding boxes ``(x, y, w, h)``.
        iou_thr: Minimum IoU to count as a true positive.

    Returns:
        Tuple of ``(tp, fp, fn)`` counts.
    """
    matched_gt = set()
    matched_det = set()
    for di, det in enumerate(det_boxes):
        best_iou = 0.0
        best_gi = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            score = iou_xywh(det, gt)
            if score > best_iou:
                best_iou = score
                best_gi = gi
        if best_iou >= iou_thr and best_gi >= 0:
            matched_gt.add(best_gi)
            matched_det.add(di)

    tp = len(matched_gt)
    fp = len(det_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def parse_camera_and_frame(filename: str) -> tuple[str | None, int | None]:
    """Extract camera ID and frame index from a YOLO val image filename.

    Expected format: ``{camera_id}_frame_{frame_idx:06d}.jpg``

    Args:
        filename: Basename of the image file (with or without extension).

    Returns:
        Tuple of ``(camera_id, frame_idx)``.  Both are ``None`` if the
        filename does not match the expected format.
    """
    stem = Path(filename).stem
    m = re.match(r"^(.+?)_frame_(\d+)$", stem)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def find_video_for_camera(video_dir: Path, camera_id: str) -> Path | None:
    """Search *video_dir* for a video file whose stem matches *camera_id*.

    Args:
        video_dir: Directory to search.
        camera_id: Camera identifier string.

    Returns:
        Path to the first matching video file, or ``None`` if not found.
    """
    for ext in (".avi", ".mp4", ".mkv", ".mov"):
        candidate = video_dir / f"{camera_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def warmup_mog2_from_video(
    detector: object,
    video_path: Path,
    target_frame_idx: int,
    warmup_frames: int,
) -> None:
    """Feed frames before *target_frame_idx* into a MOG2 detector for warmup.

    Reads up to *warmup_frames* frames immediately preceding the target frame
    index from the source video.  Frames are fed via ``detector.apply()`` to
    update the background model without producing detections.

    Args:
        detector: MOG2Detector instance (must have ``apply()`` method).
        video_path: Path to the source video file.
        target_frame_idx: Frame index of the evaluation target frame.
        warmup_frames: Maximum number of preceding frames to feed.
    """
    cap = cv2.VideoCapture(str(video_path))
    start_idx = max(0, target_frame_idx - warmup_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    for _ in range(target_frame_idx - start_idx):
        ret, frame = cap.read()
        if not ret:
            break
        detector.apply(frame)  # type: ignore[attr-defined]
    cap.release()


def draw_boxes(
    image: np.ndarray,
    gt_boxes: list[tuple[int, int, int, int]],
    yolo_boxes: list[tuple[int, int, int, int]],
    mog2_boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Draw GT (green), YOLO (blue), and MOG2 (red) boxes on a copy of *image*.

    Args:
        image: BGR image to annotate.
        gt_boxes: Ground-truth boxes ``(x, y, w, h)``.
        yolo_boxes: YOLO detection boxes ``(x, y, w, h)``.
        mog2_boxes: MOG2 detection boxes ``(x, y, w, h)``.

    Returns:
        Annotated BGR image copy.
    """
    out = image.copy()
    for x, y, w, h in gt_boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for x, y, w, h in yolo_boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for x, y, w, h in mog2_boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return out


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def main() -> None:
    """Run YOLO vs MOG2 evaluation and print the comparison table."""
    args = parse_args()

    from aquapose.segmentation.detector import (
        MOG2Detector,
        YOLODetector,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.video_dir) if args.video_dir else None

    # Discover validation images
    val_images_dir = Path(args.data).parent / "images" / "val"
    val_labels_dir = Path(args.data).parent / "labels" / "val"
    image_paths = sorted(
        p
        for p in val_images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        print(f"ERROR: No validation images found in {val_images_dir}")
        return

    print(f"Found {len(image_paths)} validation images.")

    # Instantiate YOLO detector (single instance, no re-init per frame)
    yolo_detector = YOLODetector(model_path=args.yolo_weights, conf_threshold=args.conf)

    # Accumulators
    yolo_tp = yolo_fp = yolo_fn = 0
    mog2_tp = mog2_fp = mog2_fn = 0

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  WARNING: Could not read {img_path.name}, skipping.")
            continue

        img_h, img_w = frame.shape[:2]
        label_path = val_labels_dir / (img_path.stem + ".txt")
        gt_boxes = load_gt_boxes(label_path, img_w, img_h)

        # YOLO detections
        yolo_dets = yolo_detector.detect(frame)
        yolo_boxes = [d.bbox for d in yolo_dets]

        # MOG2 detections — optionally warmed up from source video
        mog2_detector = MOG2Detector()
        if video_dir is not None:
            camera_id, frame_idx = parse_camera_and_frame(img_path.name)
            if camera_id and frame_idx is not None:
                video_path = find_video_for_camera(video_dir, camera_id)
                if video_path:
                    warmup_mog2_from_video(
                        mog2_detector, video_path, frame_idx, args.warmup_frames
                    )
                else:
                    print(
                        f"  WARNING: No video found for camera '{camera_id}'; running MOG2 without warmup."
                    )
            else:
                print(
                    f"  WARNING: Could not parse camera/frame from '{img_path.name}'; running MOG2 without warmup."
                )

        mog2_dets = mog2_detector.detect(frame)
        mog2_boxes = [d.bbox for d in mog2_dets]

        # Match detections
        tp_y, fp_y, fn_y = match_detections(gt_boxes, yolo_boxes, args.iou_threshold)
        tp_m, fp_m, fn_m = match_detections(gt_boxes, mog2_boxes, args.iou_threshold)

        yolo_tp += tp_y
        yolo_fp += fp_y
        yolo_fn += fn_y
        mog2_tp += tp_m
        mog2_fp += fp_m
        mog2_fn += fn_m

        # Save annotated image
        annotated = draw_boxes(frame, gt_boxes, yolo_boxes, mog2_boxes)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)

    # Compute aggregate metrics
    def _recall(tp: int, fn: int) -> float:
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def _precision(tp: int, fp: int) -> float:
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    yolo_rec = _recall(yolo_tp, yolo_fn)
    yolo_prec = _precision(yolo_tp, yolo_fp)
    mog2_rec = _recall(mog2_tp, mog2_fn)
    mog2_prec = _precision(mog2_tp, mog2_fp)

    # Print results table
    print("\n" + "=" * 60)
    print(f"Evaluation results (IoU threshold: {args.iou_threshold:.2f})")
    print("=" * 60)
    header = f"{'Detector':<10} | {'Recall':>6} | {'Precision':>9} | {'TP':>4} | {'FP':>4} | {'FN':>4}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    print(
        f"{'YOLO':<10} | {yolo_rec:>6.3f} | {yolo_prec:>9.3f} | {yolo_tp:>4} | {yolo_fp:>4} | {yolo_fn:>4}"
    )
    print(
        f"{'MOG2':<10} | {mog2_rec:>6.3f} | {mog2_prec:>9.3f} | {mog2_tp:>4} | {mog2_fp:>4} | {mog2_fn:>4}"
    )
    print("=" * 60)

    if yolo_rec > mog2_rec:
        delta = yolo_rec - mog2_rec
        print(
            f"\nYOLO recall EXCEEDS MOG2 recall by {delta:.3f}. SUCCESS criterion met."
        )
    else:
        delta = mog2_rec - yolo_rec
        print(
            f"\nWARNING: YOLO recall does NOT exceed MOG2 recall (deficit {delta:.3f})."
        )
        print("Consider re-training with yolov8s.pt or more epochs.")

    print(f"\nAnnotated images saved to: {output_dir}/")
    print("Legend: GREEN = ground truth, BLUE = YOLO, RED = MOG2")

    if video_dir is None:
        print(
            "\nNOTE: MOG2 ran WITHOUT video warmup. For a fair comparison, "
            "re-run with --video-dir pointing to per-camera video files. "
            "Without warmup, MOG2 lacks a background model and will likely "
            "show inflated false positives on the first frame it sees."
        )


if __name__ == "__main__":
    main()
