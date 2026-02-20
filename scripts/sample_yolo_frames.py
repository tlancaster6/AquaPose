"""MOG2-guided frame sampling for YOLO annotation dataset creation.

Samples diverse frames from multi-camera video files using MOG2 detection counts
to ensure hard cases (0, 1, 2, many fish) are represented across all cameras.
Exports frames as JPEGs to a flat images/ directory ready for Label Studio import.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running as a standalone script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aquapose.segmentation import MOG2Detector


def find_video_files(video_dir: Path) -> dict[str, Path]:
    """Discover video files in a directory, keyed by camera ID.

    The camera ID is taken as the video file stem (filename without extension).

    Args:
        video_dir: Directory containing per-camera video files.

    Returns:
        Mapping from camera_id to video file path.
    """
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".mts", ".m2ts"}
    videos: dict[str, Path] = {}
    for path in sorted(video_dir.iterdir()):
        if path.suffix.lower() in extensions:
            videos[path.stem] = path
    return videos


def warm_up_detector(
    detector: MOG2Detector, cap: cv2.VideoCapture, n_frames: int
) -> None:
    """Feed warmup frames into MOG2 to build a stable background model.

    Args:
        detector: MOG2Detector to warm up.
        cap: OpenCV VideoCapture positioned at frame 0.
        n_frames: Number of frames to feed (detector.apply, not detect).
    """
    fed = 0
    while fed < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        detector.apply(frame)
        fed += 1


def bin_frames(
    detection_counts: list[tuple[int, int]],
) -> dict[str, list[int]]:
    """Bin frame indices by MOG2 detection count.

    Args:
        detection_counts: List of (frame_idx, mog2_count) tuples.

    Returns:
        Dict mapping bin label to list of frame indices.
        Bins: "0", "1", "2", "3+".
    """
    bins: dict[str, list[int]] = {"0": [], "1": [], "2": [], "3+": []}
    for frame_idx, count in detection_counts:
        if count == 0:
            bins["0"].append(frame_idx)
        elif count == 1:
            bins["1"].append(frame_idx)
        elif count == 2:
            bins["2"].append(frame_idx)
        else:
            bins["3+"].append(frame_idx)
    return bins


def sample_from_bins(
    bins: dict[str, list[int]], n_total: int, rng: np.random.Generator
) -> list[int]:
    """Select n_total frame indices from bins with balanced diversity.

    Strategy:
    1. Allocate at least 1 frame to each non-empty bin.
    2. Distribute remaining quota proportionally to bin size (largest bins get more).

    Args:
        bins: Dict mapping bin label to list of candidate frame indices.
        n_total: Total number of frames to select.
        rng: NumPy random Generator for reproducible sampling.

    Returns:
        Sorted list of selected frame indices.
    """
    non_empty = {k: v for k, v in bins.items() if v}
    if not non_empty:
        return []

    n_bins = len(non_empty)
    if n_total <= 0:
        return []

    # Step 1: guarantee at least 1 per non-empty bin
    quota: dict[str, int] = {k: 1 for k in non_empty}
    remaining = n_total - n_bins

    if remaining < 0:
        # Fewer slots than bins — sample 1 from each of the n_total largest bins
        sorted_keys = sorted(non_empty, key=lambda k: len(non_empty[k]), reverse=True)
        selected_keys = sorted_keys[:n_total]
        quota = {k: 1 for k in selected_keys}
    elif remaining > 0:
        # Step 2: distribute remaining quota proportional to bin size
        total_frames = sum(len(v) for v in non_empty.values())
        fractional = {
            k: len(non_empty[k]) / total_frames * remaining for k in non_empty
        }
        # Integer allocation — floor first, give leftover to largest fractions
        floored = {k: int(f) for k, f in fractional.items()}
        leftover = remaining - sum(floored.values())
        # Sort by fractional remainder descending to assign leftover
        by_remainder = sorted(
            non_empty, key=lambda k: fractional[k] - floored[k], reverse=True
        )
        for i, k in enumerate(by_remainder):
            if i < leftover:
                floored[k] += 1
        for k, extra in floored.items():
            quota[k] += extra

    selected: list[int] = []
    for k, n in quota.items():
        pool = non_empty[k]
        n = min(n, len(pool))
        chosen = rng.choice(pool, size=n, replace=False).tolist()
        selected.extend(chosen)

    return sorted(selected)


def scan_camera(
    camera_id: str,
    video_path: Path,
    warmup_frames: int,
    sample_stride: int,
    n_sample: int,
    output_dir: Path,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Process one camera video: warm up MOG2, scan, sample, and export frames.

    Args:
        camera_id: Camera identifier string (used in output filenames).
        video_path: Path to the video file.
        warmup_frames: Number of frames to feed for background model warmup.
        sample_stride: Check every Nth frame after warmup.
        n_sample: Number of frames to export for this camera.
        output_dir: Directory to write exported JPEG files.
        rng: NumPy random Generator for reproducible sampling.

    Returns:
        Dict with keys "exported", "bin_0", "bin_1", "bin_2", "bin_3plus",
        "total_scanned" for summary reporting.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}", flush=True)
        return {
            "exported": 0,
            "bin_0": 0,
            "bin_1": 0,
            "bin_2": 0,
            "bin_3plus": 0,
            "total_scanned": 0,
        }

    detector = MOG2Detector()

    # Warmup phase
    warm_up_detector(detector, cap, warmup_frames)

    # Scan phase — stride through remaining frames
    detection_counts: list[tuple[int, int]] = []

    # Reset and seek to warmup end (more reliable than counting reads)
    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_frames)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride_counter = 0

    while True:
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos >= total_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        current_pos = pos  # frame index before read

        if stride_counter % sample_stride == 0:
            detections = detector.detect(frame)
            detection_counts.append((current_pos, len(detections)))
        else:
            # Still advance the background model for non-sampled frames
            detector.apply(frame)

        stride_counter += 1

    # Bin and sample
    bins = bin_frames(detection_counts)
    selected_indices = sample_from_bins(bins, n_sample, rng)

    # Build a map from frame_idx → frame data (re-read selected frames)
    selected_set = set(selected_indices)
    frames_to_export: dict[int, np.ndarray] = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    while selected_set:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in selected_set:
            frames_to_export[idx] = frame
            selected_set.discard(idx)
        idx += 1

    cap.release()

    # Export frames as JPEG
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for fidx in sorted(frames_to_export):
        filename = f"{camera_id}_frame_{fidx:06d}.jpg"
        out_path = images_dir / filename
        cv2.imwrite(str(out_path), frames_to_export[fidx])
        exported += 1

    return {
        "exported": exported,
        "bin_0": len(bins["0"]),
        "bin_1": len(bins["1"]),
        "bin_2": len(bins["2"]),
        "bin_3plus": len(bins["3+"]),
        "total_scanned": len(detection_counts),
    }


def print_summary_table(
    results: dict[str, dict[str, int]], center_camera: str | None
) -> None:
    """Print a summary table of per-camera sampling results.

    Args:
        results: Mapping from camera_id to result dict from scan_camera.
        center_camera: Camera ID of the center camera, if any.
    """
    header = f"{'Camera':<16} {'Type':<8} {'Scanned':>8} {'Bin0':>6} {'Bin1':>6} {'Bin2':>6} {'Bin3+':>6} {'Exported':>9}"
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    total_exported = 0
    for cam_id, r in sorted(results.items()):
        cam_type = "center" if cam_id == center_camera else "ring"
        print(
            f"{cam_id:<16} {cam_type:<8} {r['total_scanned']:>8} "
            f"{r['bin_0']:>6} {r['bin_1']:>6} {r['bin_2']:>6} {r['bin_3plus']:>6} "
            f"{r['exported']:>9}"
        )
        total_exported += r["exported"]
    print("-" * len(header))
    print(
        f"{'TOTAL':<16} {'':>8} {sum(r['total_scanned'] for r in results.values()):>8} "
        f"{'':>6} {'':>6} {'':>6} {'':>6} {total_exported:>9}"
    )
    print("=" * len(header))
    print()
    print(f"Total frames exported: {total_exported}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sample diverse frames from per-camera video files using MOG2 "
            "detection counts for diversity. Exports JPEGs ready for Label Studio "
            "bounding box annotation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        type=Path,
        help="Directory containing per-camera video files (one file per camera).",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data/yolo_fish_raw"),
        type=Path,
        help="Root output directory. Frames are written to <output-dir>/images/.",
    )
    parser.add_argument(
        "--center-camera",
        default=None,
        help=(
            "Camera ID (video file stem) of the center/top-down camera. "
            "Center camera gets 30 sampled frames; ring cameras get 10 each."
        ),
    )
    parser.add_argument(
        "--warmup-frames",
        default=200,
        type=int,
        help="Number of frames fed to MOG2 for background model warmup (no detection).",
    )
    parser.add_argument(
        "--sample-stride",
        default=100,
        type=int,
        help="Check every Nth frame for MOG2 detection count during scanning.",
    )
    parser.add_argument(
        "--ring-frames",
        default=10,
        type=int,
        help="Number of frames to sample per ring camera.",
    )
    parser.add_argument(
        "--center-frames",
        default=30,
        type=int,
        help="Number of frames to sample from the center camera.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reproducible frame sampling.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args, scan each camera, export sampled frames."""
    args = parse_args()

    video_dir = args.video_dir
    if not video_dir.is_dir():
        print(f"[ERROR] --video-dir is not a directory: {video_dir}", file=sys.stderr)
        sys.exit(1)

    videos = find_video_files(video_dir)
    if not videos:
        print(f"[ERROR] No video files found in: {video_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(videos)} camera video(s) in {video_dir}")
    if args.center_camera and args.center_camera not in videos:
        print(
            f"[WARNING] --center-camera '{args.center_camera}' not found in video files. "
            f"Available cameras: {', '.join(sorted(videos))}",
            file=sys.stderr,
        )

    rng = np.random.default_rng(args.seed)
    results: dict[str, dict[str, int]] = {}

    for camera_id, video_path in sorted(videos.items()):
        is_center = camera_id == args.center_camera
        n_sample = args.center_frames if is_center else args.ring_frames
        cam_type = "center" if is_center else "ring"
        print(
            f"Processing {camera_id} ({cam_type}, target={n_sample} frames) ... ",
            end="",
            flush=True,
        )
        result = scan_camera(
            camera_id=camera_id,
            video_path=video_path,
            warmup_frames=args.warmup_frames,
            sample_stride=args.sample_stride,
            n_sample=n_sample,
            output_dir=args.output_dir,
            rng=rng,
        )
        results[camera_id] = result
        print(f"exported {result['exported']} frames", flush=True)

    print_summary_table(results, args.center_camera)
    print(f"\nFrames written to: {args.output_dir / 'images'}")
    print(
        "\nNext steps:\n"
        "  1. Import the images/ directory into Label Studio.\n"
        "  2. Annotate all frames with fish bounding boxes.\n"
        "  3. Export from Label Studio as 'YOLO' format.\n"
        "  4. Organize into data/yolo_fish/{images,labels}/{train,val}/\n"
        "     with 80/20 stratified split per camera.\n"
        "  5. Create data/yolo_fish/dataset.yaml.\n"
    )


if __name__ == "__main__":
    main()
