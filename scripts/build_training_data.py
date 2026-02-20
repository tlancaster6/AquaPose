"""End-to-end Mask R-CNN training data pipeline: videos -> COCO dataset -> train."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2

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

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SKIP_CAMERAS = {"e3v8250"}  # center top-down, poor mask quality


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


def _score_frames(
    video_path: Path,
    detector: object,
    max_frames: int,
    stride: int,
) -> list[tuple[int, int]]:
    """Score candidate frames by detection count and return top-N frame indices.

    Reads frames at stride intervals, runs YOLO on each, and returns the
    frame positions with the most detections.

    Args:
        video_path: Path to video file.
        detector: A detector with a ``detect(frame)`` method.
        max_frames: Number of frames to select.
        stride: Frame interval between candidates.

    Returns:
        List of (frame_position, detection_count) tuples, sorted by
        position, for the top-N frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    candidates: list[tuple[int, int]] = []

    for pos in range(0, total_frames, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame)  # type: ignore[union-attr]
        candidates.append((pos, len(dets)))

    cap.release()

    # Keep top-N by detection count, then sort by position for stable order
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = candidates[:max_frames]
    selected.sort(key=lambda x: x[0])
    return selected


def _generate_negative_crops(
    video_path: Path,
    camera_id: str,
    n_negatives: int,
    images_dir: Path,
    frame_stride: int,
    rng: random.Random,
) -> list[AnnotatedFrame]:
    """Sample random background crops from frames with no fish.

    Reads video frames at stride intervals, skips frames where a detector
    would fire, and crops random patches of plausible fish sizes as negative
    training examples.

    Args:
        video_path: Path to video file to sample from.
        camera_id: Camera identifier for COCO metadata.
        n_negatives: Number of negative crops to generate.
        images_dir: Directory to save negative crop images.
        frame_stride: Sampling interval between candidate frames.
        rng: Seeded random instance for reproducibility.

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

    negatives: list[AnnotatedFrame] = []
    attempts = 0
    max_attempts = n_negatives * 10

    while len(negatives) < n_negatives and attempts < max_attempts:
        attempts += 1
        frame_pos = rng.randrange(0, total_frames, max(1, frame_stride))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_h, frame_w = frame.shape[:2]

        # Sample a random crop of plausible fish size (100-300px in each dim)
        crop_h = rng.randint(100, min(300, frame_h // 2))
        crop_w = rng.randint(100, min(300, frame_w // 2))
        if frame_h <= crop_h or frame_w <= crop_w:
            continue

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

    videos = _discover_videos(video_dir)
    if not videos:
        print(f"No videos found in '{video_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(videos)} cameras (skipping {SKIP_CAMERAS})")

    detector = make_detector(
        "yolo", model_path=yolo_weights, conf_threshold=args.conf_threshold
    )
    labeler = SAMPseudoLabeler(model_variant=args.sam_model)

    annotated_crops: list[AnnotatedFrame] = []
    total_detections = 0
    total_kept = 0
    rng = random.Random(args.seed)

    for video_path in videos:
        camera_id = video_path.stem
        print(f"\n[{camera_id}] Scoring candidate frames...")

        scored = _score_frames(video_path, detector, args.max_frames, args.frame_stride)
        if not scored:
            print(f"  [{camera_id}] No frames could be read — skipping.")
            continue

        print(
            f"  [{camera_id}] Selected {len(scored)} frames "
            f"(best has {scored[0][1] if scored else 0} detections)"
        )

        # Re-open video to read selected frames at full quality
        cap = cv2.VideoCapture(str(video_path))
        cam_crops: list[AnnotatedFrame] = []

        for frame_pos, _det_count in scored:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_h, frame_w = frame.shape[:2]

            # Detect and segment
            detections = detector.detect(frame)
            total_detections += len(detections)

            if not detections:
                continue

            masks = labeler.predict(frame, detections)

            for i, (det, mask) in enumerate(zip(detections, masks, strict=True)):
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

                # Compute crop and extract
                region = compute_crop_region(det.bbox, frame_h, frame_w, padding=0.25)
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
                total_kept += 1

        cap.release()

        # Generate negative examples for this camera
        n_neg = max(0, round(len(cam_crops) * args.neg_fraction))
        if n_neg > 0:
            negatives = _generate_negative_crops(
                video_path, camera_id, n_neg, images_dir, args.frame_stride, rng
            )
            cam_crops.extend(negatives)
            print(
                f"  [{camera_id}] {total_kept} positive + {len(negatives)} negative crops."
            )
        else:
            print(f"  [{camera_id}] {len(cam_crops)} crop instances kept.")

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


# ---------------------------------------------------------------------------
# train subcommand
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    """Train Mask R-CNN on a generated COCO dataset.

    Args:
        args: Parsed CLI arguments for the train subcommand.
    """
    from aquapose.segmentation import evaluate, train

    coco_json = Path(args.coco_json)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)

    if not coco_json.exists():
        print(f"Error: COCO JSON not found at '{coco_json}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Training Mask R-CNN for {args.epochs} epochs...")
    print(f"  COCO JSON  : {coco_json}")
    print(f"  Image root : {image_root}")
    print(f"  Output dir : {output_dir}")

    best_model = train(coco_json, image_root, output_dir, epochs=args.epochs)
    print(f"\nBest model saved to: {best_model}")

    # Evaluate
    print("\nEvaluating on full dataset...")
    results = evaluate(best_model, coco_json, image_root)

    mean_iou = results["mean_iou"]
    num_images = results["num_images"]
    passed = mean_iou >= 0.70

    print("\n=== Evaluation ===")
    print(f"Num instances : {num_images}")
    print(f"Mean IoU      : {mean_iou:.4f}")
    print(f"Result        : {'PASS' if passed else 'FAIL'} (threshold: 0.70)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with generate and train subcommands.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Build Mask R-CNN training data from raw videos and train.",
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
        "--sam-model",
        default="facebook/sam2.1-hiera-large",
        help="SAM2 model variant (default: facebook/sam2.1-hiera-large).",
    )
    gen.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Max frames per camera to select (default: 10).",
    )
    gen.add_argument(
        "--frame-stride",
        type=int,
        default=100,
        help="Stride between candidate frames for scoring (default: 100).",
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
        default=0.4,
        help="Minimum YOLO confidence for kept detections (default: 0.4).",
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

    # --- train ---
    tr = subparsers.add_parser(
        "train",
        help="Train Mask R-CNN on generated COCO dataset.",
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
        default="output/maskrcnn/",
        type=Path,
        help="Directory for model checkpoints (default: output/maskrcnn/).",
    )
    tr.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs (default: 40).",
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


if __name__ == "__main__":
    main()
