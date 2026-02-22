"""End-to-end U-Net training data pipeline: videos -> COCO dataset -> train."""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from aquapose.calibration import (
    UndistortionMaps,
    compute_undistortion_maps,
    load_calibration_data,
    undistort_image,
)
from aquapose.segmentation import (
    AnnotatedFrame,
    SAMPseudoLabeler,
    compute_crop_region,
    extract_crop,
    filter_mask,
    make_detector,
    stratified_split,
    to_coco_dataset,
)
from aquapose.segmentation.dataset import CropDataset
from aquapose.segmentation.detector import Detection

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SKIP_CAMERAS = {"e3v8250"}  # center top-down, poor mask quality


def _maybe_undistort(frame: np.ndarray, undist: UndistortionMaps | None) -> np.ndarray:
    """Apply undistortion if maps are provided, otherwise return frame as-is."""
    if undist is not None:
        return undistort_image(frame, undist)
    return frame


# ---------------------------------------------------------------------------
# generate subcommand
# ---------------------------------------------------------------------------


def _discover_videos(video_dir: Path) -> list[Path]:
    """Find video files, skipping cameras in SKIP_CAMERAS.

    Args:
        video_dir: Directory containing per-camera video files.

    Returns:
        Sorted list of video paths.
    """
    videos = sorted(
        p
        for p in video_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS and p.stem not in SKIP_CAMERAS
    )
    return videos


def _bbox_grid_cell(
    bbox: tuple[int, int, int, int],
    frame_h: int,
    frame_w: int,
    grid_rows: int,
    grid_cols: int,
) -> tuple[int, int]:
    """Map a detection bbox center to a grid cell.

    Args:
        bbox: Bounding box as (x, y, w, h) in pixel coordinates.
        frame_h: Full frame height.
        frame_w: Full frame width.
        grid_rows: Number of grid rows.
        grid_cols: Number of grid columns.

    Returns:
        (row, col) grid cell index.
    """
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    col = min(int(cx / frame_w * grid_cols), grid_cols - 1)
    row = min(int(cy / frame_h * grid_rows), grid_rows - 1)
    return row, col


def _rank_frames(
    video_path: Path,
    detector: object,
    stride: int,
    grid_rows: int = 3,
    grid_cols: int = 3,
    undist: UndistortionMaps | None = None,
) -> list[tuple[int, int]]:
    """Score all candidate frames and return a priority-ordered list.

    Reads frames at stride intervals, runs YOLO on each, then produces a
    priority ordering using a two-pass strategy:

    1. **Detection-ranked pass**: sorts all candidates by detection count
       descending (crowded frames first).
    2. **Spatial diversity pass**: appends frames that cover grid cells seen in
       the video but missing from pass 1.

    The caller controls how many frames to consume from the returned list
    (e.g. by stopping once a target crop count is reached).

    Args:
        video_path: Path to video file.
        detector: A detector with a ``detect(frame)`` method.
        stride: Frame interval between candidates.
        grid_rows: Number of spatial grid rows for diversity tracking.
        grid_cols: Number of spatial grid columns for diversity tracking.
        undist: Optional precomputed undistortion maps for this camera.

    Returns:
        Priority-ordered list of (frame_position, detection_count) tuples.
        Detection-ranked frames come first, then spatial diversity frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Score all candidate frames and record per-frame grid cell hits
    candidates: list[tuple[int, int]] = []  # (pos, det_count)
    frame_cells: dict[int, set[tuple[int, int]]] = {}  # pos -> set of grid cells
    global_cells: Counter[tuple[int, int]] = Counter()  # total dets per cell

    for pos in range(0, total_frames, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, raw = cap.read()
        if not ret:
            break
        frame = _maybe_undistort(raw, undist)
        dets: list[Detection] = detector.detect(frame)  # type: ignore[union-attr]
        candidates.append((pos, len(dets)))

        cells: set[tuple[int, int]] = set()
        for det in dets:
            cell = _bbox_grid_cell(det.bbox, frame_h, frame_w, grid_rows, grid_cols)
            cells.add(cell)
            global_cells[cell] += 1
        frame_cells[pos] = cells

    cap.release()

    if not candidates:
        return []

    # --- Pass 1: detection-ranked ---
    candidates.sort(key=lambda x: x[1], reverse=True)
    ranked = list(candidates)
    selected_positions = {pos for pos, _ in ranked}

    # --- Pass 2: spatial diversity ---
    # Cells that have detections somewhere but aren't covered by top-ranked frames
    covered_cells: set[tuple[int, int]] = set()
    for pos, _ in ranked:
        covered_cells.update(frame_cells.get(pos, set()))

    accessible_cells = set(global_cells.keys())
    uncovered = accessible_cells - covered_cells
    diversity: list[tuple[int, int]] = []

    if uncovered:
        for pos, count in candidates:
            if not uncovered:
                break
            if pos in selected_positions:
                continue
            hits = frame_cells.get(pos, set()) & uncovered
            if hits:
                diversity.append((pos, count))
                selected_positions.add(pos)
                covered_cells.update(frame_cells.get(pos, set()))
                uncovered -= hits

    return ranked + diversity


def _generate_negative_crops(
    video_path: Path,
    camera_id: str,
    n_negatives: int,
    images_dir: Path,
    frame_stride: int,
    rng: random.Random,
    positive_crop_sizes: list[tuple[int, int]] | None = None,
    cell_histogram: Counter[tuple[int, int]] | None = None,
    grid_rows: int = 3,
    grid_cols: int = 3,
    undist: UndistortionMaps | None = None,
) -> list[AnnotatedFrame]:
    """Sample spatially-matched background crops as negative training examples.

    Crop centers are sampled proportionally to the positive detection grid-cell
    distribution, so negatives see sand and wall backgrounds in the same ratio
    as positive crops. This prevents the model from learning a spatial shortcut
    (e.g. "wall = no fish").

    Args:
        video_path: Path to video file to sample from.
        camera_id: Camera identifier for COCO metadata.
        n_negatives: Number of negative crops to generate.
        images_dir: Directory to save negative crop images.
        frame_stride: Sampling interval between candidate frames.
        rng: Seeded random instance for reproducibility.
        positive_crop_sizes: List of (width, height) from positive crops.
            If provided, negative crop sizes are sampled from this distribution.
            Falls back to 30-200px uniform if empty or None.
        cell_histogram: Counter mapping (row, col) grid cells to positive
            detection counts. When provided, negative crop centers are sampled
            proportionally from this distribution. Falls back to uniform
            sampling when None or empty.
        grid_rows: Number of spatial grid rows.
        grid_cols: Number of spatial grid columns.
        undist: Optional precomputed undistortion maps for this camera.

    Returns:
        List of AnnotatedFrame objects with empty mask lists.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cell_h = frame_h / grid_rows
    cell_w = frame_w / grid_cols

    # Build weighted cell list for spatial sampling
    if cell_histogram:
        cells_list = list(cell_histogram.keys())
        weights = [float(cell_histogram[c]) for c in cells_list]
    else:
        cells_list = None
        weights = None

    negatives: list[AnnotatedFrame] = []
    attempts = 0
    max_attempts = n_negatives * 10

    while len(negatives) < n_negatives and attempts < max_attempts:
        attempts += 1
        frame_pos = rng.randrange(0, total_frames, max(1, frame_stride))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, raw = cap.read()
        if not ret:
            continue
        frame = _maybe_undistort(raw, undist)

        # Sample crop size from the positive crop distribution
        if positive_crop_sizes:
            ref_w, ref_h = rng.choice(positive_crop_sizes)
            crop_w = max(10, int(ref_w * rng.uniform(0.8, 1.2)))
            crop_h = max(10, int(ref_h * rng.uniform(0.8, 1.2)))
        else:
            crop_w = rng.randint(30, min(200, frame_w // 2))
            crop_h = rng.randint(30, min(200, frame_h // 2))

        if frame_h <= crop_h or frame_w <= crop_w:
            continue

        # Sample crop center from the positive spatial distribution
        if cells_list and weights:
            (row, col) = rng.choices(cells_list, weights=weights, k=1)[0]
            # Random center within the chosen cell
            cy = rng.uniform(row * cell_h, (row + 1) * cell_h)
            cx = rng.uniform(col * cell_w, (col + 1) * cell_w)
            x0 = max(0, min(int(cx - crop_w / 2), frame_w - crop_w))
            y0 = max(0, min(int(cy - crop_h / 2), frame_h - crop_h))
        else:
            y0 = rng.randint(0, frame_h - crop_h)
            x0 = rng.randint(0, frame_w - crop_w)

        crop_img = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]

        # Save crop
        crop_name = f"{camera_id}_neg{len(negatives):04d}.jpg"
        crop_path = images_dir / crop_name
        cv2.imwrite(str(crop_path), crop_img)

        negatives.append(
            AnnotatedFrame(
                frame_id=f"{camera_id}_neg{len(negatives):04d}",
                image_path=crop_path,
                masks=[],  # No annotations — pure background
                camera_id=camera_id,
            )
        )

    cap.release()
    return negatives


def _write_split_json(
    annotated_crops: list[AnnotatedFrame],
    output_dir: Path,
    val_fraction: float,
    seed: int,
) -> tuple[Path, Path]:
    """Write train.json and val.json from a per-camera stratified split.

    First writes a combined coco_annotations.json, then loads it as a
    CropDataset, splits by camera using stratified_split, and writes
    separate train.json and val.json files.

    Args:
        annotated_crops: All annotated frames (positive + negative).
        output_dir: Directory containing the combined coco_annotations.json
            and images/ subdirectory.
        val_fraction: Fraction of each camera's images for validation.
        seed: Random seed for split reproducibility.

    Returns:
        Tuple of (train_json_path, val_json_path).
    """
    coco_path = output_dir / "coco_annotations.json"
    images_dir = output_dir / "images"

    # Load the combined COCO to split it
    with open(coco_path) as f:
        full_coco: dict = json.load(f)

    # Build a temporary dataset for splitting (no augment needed)
    ds = CropDataset(coco_path, images_dir)
    train_idx, val_idx = stratified_split(ds, val_fraction=val_fraction, seed=seed)

    # Map image_list positions -> image id sets
    all_images = full_coco["images"]
    all_anns = full_coco["annotations"]

    # Build image_id -> annotation lookup
    ann_by_img: dict[int, list[dict]] = {}
    for ann in all_anns:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    def _subset_coco(indices: list[int]) -> dict:
        subset_images = [all_images[i] for i in indices]
        subset_img_ids = {img["id"] for img in subset_images}
        subset_anns = [ann for ann in all_anns if ann["image_id"] in subset_img_ids]
        return {
            "images": subset_images,
            "annotations": subset_anns,
            "categories": full_coco["categories"],
        }

    train_coco = _subset_coco(train_idx)
    val_coco = _subset_coco(val_idx)

    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"

    with open(train_path, "w") as f:
        json.dump(train_coco, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val_coco, f, indent=2)

    return train_path, val_path


@dataclass
class _CameraStats:
    """Per-camera sampling statistics collected during generation."""

    camera_id: str
    n_positive: int
    n_negative: int
    n_frames_used: int
    n_candidates: int
    cell_histogram: Counter[tuple[int, int]]
    reference_frame: np.ndarray  # first scored frame for heatmap background


def _render_heatmap(
    frame: np.ndarray,
    cell_histogram: Counter[tuple[int, int]],
    grid_rows: int,
    grid_cols: int,
) -> str:
    """Render a spatial heatmap overlaid on a reference frame as a base64 PNG.

    Args:
        frame: BGR reference frame from the video.
        cell_histogram: Counter mapping (row, col) to detection counts.
        grid_rows: Number of grid rows.
        grid_cols: Number of grid columns.

    Returns:
        Base64-encoded PNG string for embedding in markdown.
    """
    overlay = frame.copy()
    frame_h, frame_w = frame.shape[:2]
    cell_h = frame_h / grid_rows
    cell_w = frame_w / grid_cols

    max_count = max(cell_histogram.values()) if cell_histogram else 1

    for (row, col), count in cell_histogram.items():
        alpha = 0.15 + 0.45 * (count / max_count)  # 0.15-0.60 opacity
        x1 = int(col * cell_w)
        y1 = int(row * cell_h)
        x2 = int((col + 1) * cell_w)
        y2 = int((row + 1) * cell_h)

        # Blue-to-red colormap: low counts blue, high counts red
        t = count / max_count
        color = (int(255 * (1 - t)), 0, int(255 * t))  # BGR

        cell_overlay = overlay.copy()
        cv2.rectangle(cell_overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(cell_overlay, alpha, overlay, 1 - alpha, 0, overlay)

        # Draw count text
        label = str(count)
        font_scale = min(cell_w, cell_h) / 120
        thickness = max(1, int(font_scale * 2))
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        tx = x1 + int((cell_w - tw) / 2)
        ty = y1 + int((cell_h + th) / 2)
        cv2.putText(
            overlay,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    # Draw grid lines
    for r in range(1, grid_rows):
        y = int(r * cell_h)
        cv2.line(overlay, (0, y), (frame_w, y), (255, 255, 255), 1)
    for c in range(1, grid_cols):
        x = int(c * cell_w)
        cv2.line(overlay, (x, 0), (x, frame_h), (255, 255, 255), 1)

    # Encode to base64 PNG
    _, buf = cv2.imencode(".png", overlay)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return b64


def _write_sampling_summary(
    output_dir: Path,
    camera_stats: list[_CameraStats],
    args: argparse.Namespace,
    grid_rows: int,
    grid_cols: int,
    total_detections: int,
    train_path: Path,
    val_path: Path,
) -> Path:
    """Write a markdown sampling summary with parameter values, heatmaps, and stats.

    Args:
        output_dir: Output directory for the summary file.
        camera_stats: Per-camera statistics collected during generation.
        args: Parsed CLI arguments.
        grid_rows: Number of spatial grid rows.
        grid_cols: Number of spatial grid columns.
        total_detections: Total YOLO detections across all cameras.
        train_path: Path to the train split JSON.
        val_path: Path to the val split JSON.

    Returns:
        Path to the written summary markdown file.
    """
    lines: list[str] = []
    lines.append("# Sampling Summary\n")

    # Parameters
    lines.append("## Parameters\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| n_samples | {args.n_samples} |")
    lines.append(f"| frame_stride | {args.frame_stride} |")
    lines.append(f"| grid | {grid_rows}x{grid_cols} |")
    lines.append(f"| conf_threshold | {args.conf_threshold} |")
    lines.append(f"| min_conf | {args.min_conf} |")
    lines.append(f"| min_fill | {args.min_fill} |")
    lines.append(f"| max_fill | {args.max_fill} |")
    lines.append(f"| min_area | {args.min_area} |")
    lines.append(f"| neg_fraction | {args.neg_fraction} |")
    lines.append(f"| val_fraction | {args.val_fraction} |")
    lines.append(f"| sam_model | {args.sam_model} |")
    lines.append(f"| seed | {args.seed} |")
    lines.append("")

    # Overall stats
    total_pos = sum(s.n_positive for s in camera_stats)
    total_neg = sum(s.n_negative for s in camera_stats)
    lines.append("## Overall Statistics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Cameras processed | {len(camera_stats)} |")
    lines.append(f"| Total YOLO detections | {total_detections} |")
    lines.append(f"| Positive crops | {total_pos} |")
    lines.append(f"| Negative crops | {total_neg} |")
    lines.append(f"| Total crops | {total_pos + total_neg} |")
    lines.append(f"| Train split | {train_path.name} |")
    lines.append(f"| Val split | {val_path.name} |")
    lines.append("")

    # Per-camera table
    lines.append("## Per-Camera Breakdown\n")
    lines.append("| Camera | Frames Used | Candidates | Positive | Negative | Total |")
    lines.append("|--------|-------------|------------|----------|----------|-------|")
    for s in camera_stats:
        lines.append(
            f"| {s.camera_id} | {s.n_frames_used} | {s.n_candidates} | "
            f"{s.n_positive} | {s.n_negative} | {s.n_positive + s.n_negative} |"
        )
    lines.append("")

    # Spatial heatmaps
    lines.append("## Spatial Distribution Heatmaps\n")
    lines.append("Positive crop counts per grid cell, overlaid on a reference frame.\n")
    for s in camera_stats:
        b64 = _render_heatmap(s.reference_frame, s.cell_histogram, grid_rows, grid_cols)
        lines.append(f"### {s.camera_id}\n")
        lines.append(f"![{s.camera_id} heatmap](data:image/png;base64,{b64})\n")

    summary_path = output_dir / "sampling_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate a cropped COCO dataset from raw videos.

    Args:
        args: Parsed CLI arguments for the generate subcommand.
    """
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    yolo_weights = Path(args.yolo_weights)
    if not yolo_weights.exists():
        print(
            f"Error: YOLO weights not found at '{yolo_weights}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load calibration for undistortion (optional)
    undist_maps: dict[str, UndistortionMaps] = {}
    if args.calibration:
        calib_path = Path(args.calibration)
        if not calib_path.exists():
            print(
                f"Error: calibration file not found at '{calib_path}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        calib = load_calibration_data(calib_path)
        for cam_id, cam_data in calib.cameras.items():
            undist_maps[cam_id] = compute_undistortion_maps(cam_data)
        print(f"Loaded calibration with {len(undist_maps)} camera undistortion maps")

    videos = _discover_videos(video_dir)
    if not videos:
        print(f"No videos found in '{video_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(videos)} cameras (skipping {SKIP_CAMERAS})")

    detector = make_detector(
        "yolo", model_path=yolo_weights, conf_threshold=args.conf_threshold
    )
    labeler = SAMPseudoLabeler(
        model_variant=args.sam_model,
        draw_pseudolabels=args.draw_pseudolabels,
    )

    annotated_crops: list[AnnotatedFrame] = []
    all_camera_stats: list[_CameraStats] = []
    total_detections = 0
    total_kept = 0
    rng = random.Random(args.seed)

    for video_path in videos:
        camera_id = video_path.stem
        cam_undist = undist_maps.get(camera_id)
        print(f"\n[{camera_id}] Scoring candidate frames...")

        ranked = _rank_frames(
            video_path,
            detector,
            args.frame_stride,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            undist=cam_undist,
        )
        if not ranked:
            print(f"  [{camera_id}] No frames could be read — skipping.")
            continue

        print(
            f"  [{camera_id}] {len(ranked)} candidate frames ranked "
            f"(best has {ranked[0][1]} detections)"
        )

        # Re-open video and consume frames until we hit the crop target
        cap = cv2.VideoCapture(str(video_path))
        cam_crops: list[AnnotatedFrame] = []
        cam_crop_sizes: list[tuple[int, int]] = []
        cam_cell_hist: Counter[tuple[int, int]] = Counter()
        n_frames_used = 0
        ref_frame: np.ndarray | None = None

        for frame_pos, _det_count in ranked:
            if len(cam_crops) >= args.n_samples:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, raw = cap.read()
            if not ret:
                continue
            frame = _maybe_undistort(raw, cam_undist)
            n_frames_used += 1
            if ref_frame is None:
                ref_frame = frame.copy()

            frame_h, frame_w = frame.shape[:2]

            # Detect and segment
            detections = detector.detect(frame)
            total_detections += len(detections)

            if not detections:
                continue

            masks = labeler.predict(frame, detections)

            for i, (det, mask) in enumerate(zip(detections, masks, strict=True)):
                if len(cam_crops) >= args.n_samples:
                    break

                # Quality filter using module-level filter_mask
                clean_mask = filter_mask(
                    mask,
                    det,
                    min_conf=args.min_conf,
                    min_fill=args.min_fill,
                    max_fill=args.max_fill,
                    min_area=args.min_area,
                )
                if clean_mask is None:
                    continue

                # Compute crop and extract (no padding — tight to bbox)
                region = compute_crop_region(det.bbox, frame_h, frame_w, padding=0.0)
                crop_img = extract_crop(frame, region)
                crop_mask = extract_crop(clean_mask, region)

                # Save crop image
                crop_name = f"{camera_id}_f{frame_pos:06d}_d{i:02d}.jpg"
                crop_path = images_dir / crop_name
                cv2.imwrite(str(crop_path), crop_img)

                # Build AnnotatedFrame for this single crop
                cam_crops.append(
                    AnnotatedFrame(
                        frame_id=f"{camera_id}_f{frame_pos:06d}_d{i:02d}",
                        image_path=crop_path,
                        masks=[crop_mask],
                        camera_id=camera_id,
                    )
                )
                cam_crop_sizes.append((crop_img.shape[1], crop_img.shape[0]))

                # Track spatial distribution for negative sampling
                cell = _bbox_grid_cell(
                    det.bbox, frame_h, frame_w, args.grid_rows, args.grid_cols
                )
                cam_cell_hist[cell] += 1
                total_kept += 1

        cap.release()

        # Generate negative examples for this camera
        n_pos_cam = len(cam_crops)
        n_neg = max(0, round(n_pos_cam * args.neg_fraction))
        if n_neg > 0:
            negatives = _generate_negative_crops(
                video_path,
                camera_id,
                n_neg,
                images_dir,
                args.frame_stride,
                rng,
                positive_crop_sizes=cam_crop_sizes,
                cell_histogram=cam_cell_hist,
                grid_rows=args.grid_rows,
                grid_cols=args.grid_cols,
                undist=cam_undist,
            )
            cam_crops.extend(negatives)
            print(
                f"  [{camera_id}] {n_pos_cam} positive "
                f"+ {len(negatives)} negative crops from {n_frames_used} frames."
            )
        else:
            negatives = []
            print(
                f"  [{camera_id}] {len(cam_crops)} crops from {n_frames_used} frames."
            )

        # Collect per-camera stats for the sampling summary
        if ref_frame is not None:
            all_camera_stats.append(
                _CameraStats(
                    camera_id=camera_id,
                    n_positive=n_pos_cam,
                    n_negative=len(negatives),
                    n_frames_used=n_frames_used,
                    n_candidates=len(ranked),
                    cell_histogram=cam_cell_hist,
                    reference_frame=ref_frame,
                )
            )

        annotated_crops.extend(cam_crops)

    if not annotated_crops:
        print("No crops survived filtering. Adjust thresholds.", file=sys.stderr)
        sys.exit(1)

    # Export combined COCO JSON
    coco_path = output_dir / "coco_annotations.json"
    to_coco_dataset(annotated_crops, coco_path)

    # Write per-split COCO JSONs
    train_path, val_path = _write_split_json(
        annotated_crops, output_dir, val_fraction=args.val_fraction, seed=args.seed
    )

    # Write sampling summary report
    summary_path = _write_sampling_summary(
        output_dir,
        all_camera_stats,
        args,
        args.grid_rows,
        args.grid_cols,
        total_detections,
        train_path,
        val_path,
    )

    n_pos = sum(1 for af in annotated_crops if af.masks)
    n_neg_total = sum(1 for af in annotated_crops if not af.masks)

    print("\n=== Generation complete ===")
    print(f"Cameras processed   : {len(videos)}")
    print(f"Total detections    : {total_detections}")
    print(f"Positive crops kept : {n_pos}")
    print(f"Negative crops added: {n_neg_total}")
    print(f"Combined COCO JSON  : {coco_path}")
    print(f"Train JSON          : {train_path}")
    print(f"Val JSON            : {val_path}")
    print(f"Images dir          : {images_dir}")
    print(f"Sampling summary    : {summary_path}")


# ---------------------------------------------------------------------------
# train subcommand
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    """Train U-Net on a generated COCO dataset.

    Args:
        args: Parsed CLI arguments for the train subcommand.
    """
    from aquapose.segmentation import evaluate, train

    coco_json = Path(args.coco_json)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)

    train_json: Path | None = Path(args.train_json) if args.train_json else None
    val_json: Path | None = Path(args.val_json) if args.val_json else None

    if not coco_json.exists():
        print(f"Error: COCO JSON not found at '{coco_json}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Training U-Net for {args.epochs} epochs...")
    print(f"  COCO JSON  : {coco_json}")
    if train_json:
        print(f"  Train JSON : {train_json}")
    if val_json:
        print(f"  Val JSON   : {val_json}")
    print(f"  Image root : {image_root}")
    print(f"  Output dir : {output_dir}")

    best_model = train(
        coco_json,
        image_root,
        output_dir,
        train_json=train_json,
        val_json=val_json,
        epochs=args.epochs,
    )
    print(f"\nBest model saved to: {best_model}")

    # Evaluate on val split if provided, otherwise full dataset
    eval_json = val_json if val_json is not None else coco_json
    print(f"\nEvaluating on {'val split' if val_json else 'full dataset'}...")
    results = evaluate(best_model, eval_json, image_root)

    mean_iou = results["mean_iou"]
    num_images = results["num_images"]
    passed = mean_iou >= 0.70

    print("\n=== Evaluation ===")
    print(f"Num instances : {num_images}")
    print(f"Mean IoU      : {mean_iou:.4f}")
    print(f"Result        : {'PASS' if passed else 'FAIL'} (threshold: 0.70)")


# ---------------------------------------------------------------------------
# evaluate subcommand
# ---------------------------------------------------------------------------


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a trained U-Net on a COCO val split.

    Loads the trained model, runs inference on all images in the val JSON,
    computes per-image mask IoU against ground-truth annotations, and prints
    mean mask IoU. The quantitative gate for phase completion is >= 0.90.

    Args:
        args: Parsed CLI arguments for the evaluate subcommand.
    """
    from aquapose.segmentation import evaluate

    model_path = Path(args.model_path)
    val_json = Path(args.val_json)
    image_root = Path(args.image_root)

    if not model_path.exists():
        print(f"Error: model not found at '{model_path}'.", file=sys.stderr)
        sys.exit(1)
    if not val_json.exists():
        print(f"Error: val JSON not found at '{val_json}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating model: {model_path}")
    print(f"Val JSON        : {val_json}")
    print(f"Image root      : {image_root}")

    results = evaluate(model_path, val_json, image_root)

    mean_iou = results["mean_iou"]
    num_images = results["num_images"]
    per_image_iou: list[float] = results["per_image_iou"]  # type: ignore[assignment]
    passed = mean_iou >= 0.90

    print("\n=== Evaluation Results ===")
    print(f"Num images  : {num_images}")
    print(f"Mean IoU    : {mean_iou:.4f}")
    print(f"Min IoU     : {min(per_image_iou, default=0.0):.4f}")
    print(f"Max IoU     : {max(per_image_iou, default=0.0):.4f}")
    print(f"Result      : {'PASS' if passed else 'FAIL'} (threshold: 0.90)")

    if not passed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with generate and train subcommands.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Build U-Net training data from raw videos and train.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    gen = subparsers.add_parser(
        "generate",
        help="Generate cropped COCO dataset from raw videos.",
    )
    gen.add_argument(
        "--video-dir",
        required=True,
        type=Path,
        help="Directory containing per-camera video files.",
    )
    gen.add_argument(
        "--output-dir",
        default="output/training_data/",
        type=Path,
        help="Output directory for images and COCO JSON (default: output/training_data/).",
    )
    gen.add_argument(
        "--yolo-weights",
        default="runs/detect/output/yolo_fish/train_v1/weights/best.pt",
        type=Path,
        help="Path to YOLO weights file.",
    )
    gen.add_argument(
        "--calibration",
        default=None,
        type=Path,
        help="Path to AquaCal calibration JSON for undistortion. "
        "When provided, frames are undistorted before detection and cropping.",
    )
    gen.add_argument(
        "--sam-model",
        default="facebook/sam2.1-hiera-large",
        help="SAM2 model variant (default: facebook/sam2.1-hiera-large).",
    )
    gen.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Target number of positive crops per video (default: 50). "
        "Frames are consumed in priority order until this count is reached.",
    )
    gen.add_argument(
        "--frame-stride",
        type=int,
        default=100,
        help="Stride between candidate frames for scoring (default: 100).",
    )
    gen.add_argument(
        "--grid-rows",
        type=int,
        default=10,
        help="Number of spatial grid rows for diversity sampling (default: 3).",
    )
    gen.add_argument(
        "--grid-cols",
        type=int,
        default=10,
        help="Number of spatial grid columns for diversity sampling (default: 3).",
    )
    gen.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="YOLO confidence for candidate scoring (default: 0.25).",
    )
    gen.add_argument(
        "--min-conf",
        type=float,
        default=0.25,
        help="Minimum YOLO confidence for kept detections (default: 0.25).",
    )
    gen.add_argument(
        "--min-fill",
        type=float,
        default=0.20,
        help="Minimum mask fill ratio of bbox area (default: 0.20).",
    )
    gen.add_argument(
        "--max-fill",
        type=float,
        default=0.70,
        help="Maximum mask fill ratio of bbox area (default: 0.70).",
    )
    gen.add_argument(
        "--min-area",
        type=int,
        default=200,
        help="Minimum mask pixel area in full-frame coords (default: 200).",
    )
    gen.add_argument(
        "--neg-fraction",
        type=float,
        default=0.1,
        help="Fraction of negative (background) examples relative to positives "
        "per camera (default: 0.1 = ~10%%).",
    )
    gen.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Per-camera validation split fraction (default: 0.2 = 20%%).",
    )
    gen.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split and negative sampling (default: 42).",
    )
    gen.add_argument(
        "--draw-pseudolabels",
        action="store_true",
        default=False,
        help="Save annotated debug crops to <output-dir>/debug/ for visual inspection.",
    )

    # --- train ---
    tr = subparsers.add_parser(
        "train",
        help="Train U-Net on generated COCO dataset.",
    )
    tr.add_argument(
        "--coco-json",
        required=True,
        type=Path,
        help="Path to COCO-format annotation JSON.",
    )
    tr.add_argument(
        "--image-root",
        required=True,
        type=Path,
        help="Root directory containing crop images.",
    )
    tr.add_argument(
        "--output-dir",
        default="output/unet/",
        type=Path,
        help="Directory for model checkpoints (default: output/unet/).",
    )
    tr.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs (default: 40).",
    )
    tr.add_argument(
        "--train-json",
        default=None,
        type=Path,
        help="Optional path to pre-split train COCO JSON (e.g. train.json from generate). "
        "When provided with --val-json, --coco-json is used only for fallback.",
    )
    tr.add_argument(
        "--val-json",
        default=None,
        type=Path,
        help="Optional path to pre-split val COCO JSON (e.g. val.json from generate). "
        "When provided with --train-json, --coco-json is used only for fallback.",
    )

    # --- evaluate ---
    ev = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained U-Net on a COCO val split.",
    )
    ev.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Path to saved U-Net model checkpoint (best_model.pth).",
    )
    ev.add_argument(
        "--val-json",
        required=True,
        type=Path,
        help="Path to COCO-format val annotation JSON.",
    )
    ev.add_argument(
        "--image-root",
        required=True,
        type=Path,
        help="Root directory containing crop images.",
    )

    return parser


def main() -> None:
    """Entry point for the build_training_data CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
