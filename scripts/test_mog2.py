"""Consolidated MOG2 detection diagnostic: numeric metrics + side-by-side visual output.

Processes camera MP4 files (default: center e3v8250 and side e3v83eb), warms up the
MOG2 background model, samples evenly-spaced frames after warm-up, and saves annotated
side-by-side JPEG stills.

Left panel: original frame with detection bounding box overlays and detection count.
Right panel: colorized raw MOG2 mask (green=foreground 255, blue=shadow 127) with
    detection contour outlines.

Prints per-frame detection counts and per-camera summary statistics.
No automated pass/fail — quality is assessed visually from the output stills.

Usage:
    python scripts/test_mog2.py \\
        --data-root "C:/Users/tucke/Desktop/Aqua/AquaPose" \\
        --output-dir output/test_mog2

    # Run all 13 cameras:
    python scripts/test_mog2.py \\
        --data-root "C:/Users/tucke/Desktop/Aqua/AquaPose" \\
        --cameras all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from aquapose.segmentation.detector import MOG2Detector

# Distinct colors for up to 10 detections per frame (BGR)
COLORS = [
    (0, 0, 255),  # red
    (0, 255, 0),  # green
    (255, 0, 0),  # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (0, 128, 255),  # orange
    (128, 0, 255),  # purple
    (0, 255, 128),  # spring green
    (255, 128, 0),  # sky blue
]

# Default cameras: center + one side camera
DEFAULT_CAMERAS = ["e3v8250", "e3v83eb"]


def parse_camera_id(filename: str) -> str:
    """Extract camera ID from video filename (stem before first '-').

    Args:
        filename: Video filename stem, e.g. 'e3v831e-20260218T145915-150429'.

    Returns:
        Camera identifier, e.g. 'e3v831e'.
    """
    return filename.split("-")[0]


def sample_indices(start: int, end: int, count: int) -> list[int]:
    """Return evenly-spaced frame indices in [start, end).

    Args:
        start: First valid index (inclusive).
        end: Last valid index (exclusive).
        count: Number of indices to return.

    Returns:
        List of integer frame indices, sorted ascending.
    """
    if count <= 0 or start >= end:
        return []
    span = end - start
    if count >= span:
        return list(range(start, end))
    return [start + int(i * span / count) for i in range(count)]


def draw_left_panel(
    frame: np.ndarray,
    detections: list,
    fg_mask: np.ndarray,
) -> np.ndarray:
    """Render the left diagnostic panel: original frame with detection overlays.

    Applies a faint green tint over foreground pixels and a faint blue tint over
    shadow pixels, then draws per-detection bounding boxes and contours in distinct
    colors with detection count text.

    Args:
        frame: BGR image as uint8 array of shape (H, W, 3).
        detections: List of Detection objects from MOG2Detector.detect().
        fg_mask: Raw MOG2 mask (uint8); 255=foreground, 127=shadow, 0=background.

    Returns:
        Annotated BGR image of the same shape as *frame*.
    """
    overlay = frame.copy()

    # Faint tint over foreground and shadow regions
    overlay[fg_mask == 255] = (
        overlay[fg_mask == 255].astype(np.float32) * 0.6
        + np.array([0, 255, 0], dtype=np.float32) * 0.4
    ).astype(np.uint8)
    overlay[fg_mask == 127] = (
        overlay[fg_mask == 127].astype(np.float32) * 0.75
        + np.array([255, 100, 0], dtype=np.float32) * 0.25
    ).astype(np.uint8)

    for j, det in enumerate(detections):
        color = COLORS[j % len(COLORS)]
        contours, _ = cv2.findContours(
            det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2)
        bx, by, bw, bh = det.bbox
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, 2)
        cv2.putText(
            overlay,
            f"#{j} {det.area}px",
            (bx, max(by - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    n_det = len(detections)
    cv2.putText(
        overlay,
        f"Detections: {n_det}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return overlay


def draw_right_panel(
    frame: np.ndarray,
    detections: list,
    fg_mask: np.ndarray,
) -> np.ndarray:
    """Render the right diagnostic panel: colorized raw MOG2 mask.

    Green pixels are foreground (255), blue-ish pixels are shadow (127). Detection
    contours are drawn on top in distinct colors.

    Args:
        frame: BGR image used only for shape reference.
        detections: List of Detection objects from MOG2Detector.detect().
        fg_mask: Raw MOG2 mask (uint8); 255=foreground, 127=shadow, 0=background.

    Returns:
        Colorized mask panel as a BGR image of the same shape as *frame*.
    """
    right = np.zeros_like(frame)
    right[fg_mask == 255] = (0, 255, 0)  # green = foreground
    right[fg_mask == 127] = (255, 100, 0)  # blue-ish = shadow

    for j, det in enumerate(detections):
        color = COLORS[j % len(COLORS)]
        contours, _ = cv2.findContours(
            det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(right, contours, -1, color, 2)

    cv2.putText(
        right,
        "GREEN = foreground (255)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        right,
        "BLUE = shadow (127)",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 100, 0),
        2,
    )
    cv2.putText(
        right,
        "Colored outlines = detections",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return right


def process_camera(
    video_path: Path,
    output_dir: Path,
    warmup_frames: int,
    sample_count: int,
) -> list[dict]:
    """Process a single camera video: warm up, sample, detect, save side-by-side stills.

    For each sampled frame, saves a JPEG composed of the annotated left panel
    (bboxes + detection count) and colorized right panel (raw MOG2 mask) at
    half scale. Intermediate frames between sample points are fed to the
    background model to keep it updated.

    Args:
        video_path: Path to the camera's MP4 file.
        output_dir: Root output directory. Per-camera subdirectory is created.
        warmup_frames: Number of frames to feed without detection for background warmup.
        sample_count: Number of frames to sample for detection after warmup.

    Returns:
        List of dicts with keys camera_id, frame_idx, num_detections
        for each sampled frame.
    """
    camera_id = parse_camera_id(video_path.stem)
    cam_output = output_dir / camera_id
    cam_output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Camera {camera_id}: {total_frames} total frames")

    # Use production-default parameters (history=500, var_threshold=12, min_area=200)
    detector = MOG2Detector()

    # Warm-up phase: feed frames without detection to stabilize background model
    print(f"  Warming up on first {warmup_frames} frames...")
    for i in range(min(warmup_frames, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        detector.apply(frame)
        if (i + 1) % 100 == 0:
            print(f"    warmup {i + 1}/{warmup_frames}")

    # Sample evenly-spaced frames after warmup
    detect_start = warmup_frames
    indices = sample_indices(detect_start, total_frames, sample_count)
    print(f"  Sampling {len(indices)} frames after warm-up")

    results: list[dict] = []
    current_pos = warmup_frames  # number of frames already consumed

    for target_idx in indices:
        frames_to_skip = target_idx - current_pos
        if frames_to_skip < 0:
            # Seek back if needed (should not happen with sorted indices)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            current_pos = target_idx
            frames_to_skip = 0

        # Feed intermediate frames to keep background model updated
        for _ in range(frames_to_skip):
            ret, frame = cap.read()
            if not ret:
                break
            detector.apply(frame)
            current_pos += 1

        ret, frame = cap.read()
        if not ret:
            print(f"    Frame {target_idx}: could not read")
            break
        current_pos += 1

        h, w = frame.shape[:2]

        # Detect fish in this frame
        detections = detector.detect(frame)

        # Re-extract raw MOG2 mask for visualization only (learningRate=0 avoids
        # updating the background model a second time — acceptable in diagnostic scripts
        # that access the internal _mog2 attribute)
        fg_mask = detector._mog2.apply(frame, learningRate=0)

        # Build side-by-side panels at half scale
        left = draw_left_panel(frame, detections, fg_mask)
        right = draw_right_panel(frame, detections, fg_mask)

        scale = 0.5
        left_small = cv2.resize(left, (int(w * scale), int(h * scale)))
        right_small = cv2.resize(right, (int(w * scale), int(h * scale)))
        combined = np.hstack([left_small, right_small])

        out_path = cam_output / f"frame_{target_idx:06d}.jpg"
        cv2.imwrite(str(out_path), combined)

        n_det = len(detections)
        print(f"    {camera_id} frame {target_idx:06d}: {n_det} detections")

        results.append(
            {
                "camera_id": camera_id,
                "frame_idx": target_idx,
                "num_detections": n_det,
            }
        )

    cap.release()
    return results


def print_summary(all_results: list[dict]) -> None:
    """Print per-camera detection statistics.

    Args:
        all_results: List of per-frame result dicts from all cameras.
    """
    cameras = sorted(set(r["camera_id"] for r in all_results))

    print("\n" + "=" * 55)
    print("PER-CAMERA SUMMARY")
    print("=" * 55)
    print(
        f"{'Camera':<12} {'Frames':<8} {'Avg Det':<10} {'Min Det':<10} {'Max Det':<10}"
    )
    print("-" * 55)

    for cam in cameras:
        cam_results = [r for r in all_results if r["camera_id"] == cam]
        dets = [r["num_detections"] for r in cam_results]
        print(
            f"{cam:<12} {len(dets):<8} {np.mean(dets):<10.1f} "
            f"{min(dets):<10} {max(dets):<10}"
        )

    total_frames = len(all_results)
    total_dets = sum(r["num_detections"] for r in all_results)
    print(f"\nTotal: {total_dets} detections across {total_frames} sampled frames")


def main() -> None:
    """Entry point: parse args, process selected cameras, print detection metrics."""
    parser = argparse.ArgumentParser(
        description="MOG2 detection diagnostic: numeric metrics + side-by-side visual output."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("C:/Users/tucke/Desktop/Aqua/AquaPose"),
        help="Root directory containing raw_videos/ (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/test_mog2"),
        help="Directory for annotated output JPEG stills (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=500,
        help="Number of frames for MOG2 background warm-up (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=10,
        help="Number of frames to sample per camera after warm-up (default: %(default)s)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=",".join(DEFAULT_CAMERAS),
        help=(
            "Comma-separated list of camera IDs to process, or 'all' for every camera. "
            f"(default: {','.join(DEFAULT_CAMERAS)})"
        ),
    )
    args = parser.parse_args()

    video_dir = args.data_root / "raw_videos"
    all_videos = sorted(video_dir.glob("*.mp4"))
    if not all_videos:
        print(f"No MP4 files found in {video_dir}")
        return

    # Filter by camera selection
    if args.cameras.strip().lower() == "all":
        videos = all_videos
    else:
        selected = {c.strip() for c in args.cameras.split(",")}
        videos = [v for v in all_videos if parse_camera_id(v.stem) in selected]
        missing = selected - {parse_camera_id(v.stem) for v in videos}
        if missing:
            print(f"WARNING: Camera IDs not found in {video_dir}: {sorted(missing)}")

    if not videos:
        print(f"No matching videos found in {video_dir}")
        return

    print(f"Processing {len(videos)} camera(s) from {video_dir}")
    print(f"Output: {args.output_dir.resolve()}")
    print(
        f"Warmup: {args.warmup_frames} frames, Sample: {args.sample_count} frames/camera\n"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for video_path in videos:
        print(f"\nProcessing {video_path.name}...")
        results = process_camera(
            video_path, args.output_dir, args.warmup_frames, args.sample_count
        )
        all_results.extend(results)

    if not all_results:
        print("No results collected. Check that video files are readable.")
        return

    print_summary(all_results)
    print(f"\nAnnotated stills saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
