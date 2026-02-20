"""End-to-end pseudo-label generation: detector -> SAM2 -> Label Studio export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from aquapose.segmentation import (
    SAMPseudoLabeler,
    export_to_label_studio,
    make_detector,
)
from aquapose.segmentation.pseudo_labeler import FrameAnnotation


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate pseudo-label masks from video frames using a configurable "
            "detector (YOLO or MOG2) followed by SAM2 refinement, then export to "
            "Label Studio."
        )
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        type=Path,
        help="Path to directory containing per-camera video files.",
    )
    parser.add_argument(
        "--detector",
        default="yolo",
        choices=["mog2", "yolo"],
        help="Detector type to use: 'mog2' or 'yolo' (default: yolo).",
    )
    parser.add_argument(
        "--yolo-weights",
        default="runs/detect/output/yolo_fish/train_v1/weights/best.pt",
        type=Path,
        help=(
            "Path to YOLO weights file (.pt). Required when --detector yolo. "
            "(default: runs/detect/output/yolo_fish/train_v1/weights/best.pt)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="output/pseudo_labels/",
        type=Path,
        help="Directory for Label Studio export (default: output/pseudo_labels/).",
    )
    parser.add_argument(
        "--sam-model",
        default="facebook/sam2.1-hiera-large",
        help="SAM2 model variant from Hugging Face hub (default: facebook/sam2.1-hiera-large).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=100,
        help="Process every Nth frame from each video (default: 100).",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=200,
        help="Number of warmup frames for MOG2 background model (ignored for YOLO, default: 200).",
    )
    parser.add_argument(
        "--max-frames-per-camera",
        type=int,
        default=10,
        help="Maximum frames to process per camera (default: 10).",
    )
    parser.add_argument(
        "--center-camera",
        default="e3v8340",
        help="Center camera ID, used for MOG2 warmup distinction (default: e3v8340).",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (default: 0.25).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full pseudo-label generation pipeline.

    Reads video files from --video-dir, detects fish in sampled frames using
    the selected detector, refines detections with SAM2, builds FrameAnnotation
    objects, and exports them to Label Studio JSON format.
    """
    args = parse_args()

    # Validate YOLO weights before loading any models
    if args.detector == "yolo" and not Path(args.yolo_weights).exists():
        print(
            f"Error: YOLO weights not found at '{args.yolo_weights}'. "
            "Provide a valid path via --yolo-weights.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create detector
    if args.detector == "yolo":
        detector = make_detector(
            "yolo",
            model_path=args.yolo_weights,
            conf_threshold=args.conf_threshold,
        )
    else:
        detector = make_detector("mog2")

    # Create SAM2 labeler (lazily loads model on first predict call)
    labeler = SAMPseudoLabeler(model_variant=args.sam_model)

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Collect all video files
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = sorted(
        p
        for p in Path(args.video_dir).iterdir()
        if p.suffix.lower() in video_extensions
    )

    if not video_files:
        print(
            f"No video files found in '{args.video_dir}'. "
            "Supported formats: .mp4, .avi, .mov, .mkv",
            file=sys.stderr,
        )
        sys.exit(1)

    all_frames: list[FrameAnnotation] = []
    total_detections = 0
    total_masks = 0

    for video_path in video_files:
        camera_id = video_path.stem
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(
                f"Warning: Cannot open video '{video_path}' — skipping.",
                file=sys.stderr,
            )
            continue

        # MOG2 warmup: feed warmup frames to build background model
        if args.detector == "mog2":
            print(
                f"  [{camera_id}] Warming up MOG2 with {args.warmup_frames} frames..."
            )
            warmup_count = 0
            while warmup_count < args.warmup_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                detector.apply(frame)  # type: ignore[union-attr]
                warmup_count += 1

        # Sample frames at stride intervals
        print(f"  [{camera_id}] Sampling frames (stride={args.frame_stride})...")
        frame_idx = 0
        sampled_count = 0
        camera_frames: list[FrameAnnotation] = []

        while sampled_count < args.max_frames_per_camera:
            # Seek to next target frame position
            target_position = (
                args.warmup_frames if args.detector == "mog2" else 0
            ) + frame_idx * args.frame_stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_position)
            ret, frame = cap.read()
            if not ret:
                break

            # Detect fish
            detections = detector.detect(frame)
            total_detections += len(detections)

            # Refine detections with SAM2
            masks: list[np.ndarray] = []
            if detections:
                masks = labeler.predict(frame, detections)
                total_masks += len(masks)

            # Save frame as JPEG
            frame_name = f"{camera_id}_frame_{target_position:06d}.jpg"
            frame_path = images_dir / frame_name
            cv2.imwrite(str(frame_path), frame)

            annotation = FrameAnnotation(
                frame_id=f"{camera_id}_frame_{target_position:06d}",
                image_path=frame_path,
                masks=masks,
                camera_id=camera_id,
            )
            camera_frames.append(annotation)
            sampled_count += 1
            frame_idx += 1

        cap.release()
        all_frames.extend(camera_frames)
        print(
            f"  [{camera_id}] {sampled_count} frames sampled, "
            f"{sum(len(f.masks) for f in camera_frames)} masks generated."
        )

    if not all_frames:
        print(
            "No frames were processed. Check video files and detector settings.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Export to Label Studio
    tasks_path = export_to_label_studio(all_frames, output_dir)

    # Summary
    print()
    print("=== Pseudo-label generation complete ===")
    print(f"Cameras processed : {len(video_files)}")
    print(f"Total frames      : {len(all_frames)}")
    print(f"Total detections  : {total_detections}")
    print(f"Total masks       : {total_masks}")
    print(f"Tasks JSON        : {tasks_path}")


if __name__ == "__main__":
    main()
