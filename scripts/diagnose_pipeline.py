"""Diagnostic script for the full 5-stage AquaPose reconstruction pipeline.

Calls pipeline stages directly to retain all intermediate data, then invokes
10 per-stage diagnostic visualizations. Outputs are written to a diagnostics
subdirectory alongside the standard HDF5 output.

Usage:
    python scripts/diagnose_pipeline.py
    python scripts/diagnose_pipeline.py --stop-frame 100
    python scripts/diagnose_pipeline.py --video-dir /path/to/videos --output-dir output/diag
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Default paths (override via CLI arguments)
# ---------------------------------------------------------------------------
DEFAULT_VIDEO_DIR = Path("C:/Users/tucke/Desktop/Aqua/AquaPose/videos/core_videos")
DEFAULT_CALIBRATION = Path("C:/Users/tucke/Desktop/Aqua/AquaPose/calibration.json")
DEFAULT_UNET_WEIGHTS = Path("C:/Users/tucke/Desktop/Aqua/AquaPose/unet/best_model.pth")
DEFAULT_YOLO_WEIGHTS = Path(
    "C:/Users/tucke/Desktop/Aqua/AquaPose/yolo/train_v2/weights/best.pt"
)
DEFAULT_OUTPUT_DIR = Path("output/e2e_diagnostic")
DEFAULT_STOP_FRAME = 30

# Camera to exclude (centre top-down camera — poor mask quality)
_SKIP_CAMERA_ID = "e3v8250"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the AquaPose pipeline with per-stage diagnostic visualizations.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="Directory containing video files",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="Path to calibration.json",
    )
    parser.add_argument(
        "--unet-weights",
        type=Path,
        default=DEFAULT_UNET_WEIGHTS,
        help="Path to U-Net weights (.pth)",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=DEFAULT_YOLO_WEIGHTS,
        help="Path to YOLO weights (.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for pipeline output",
    )
    parser.add_argument(
        "--stop-frame",
        type=int,
        default=DEFAULT_STOP_FRAME,
        help="Number of frames to process",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="Camera IDs for claiming overlay (default: top 4 by count)",
    )
    return parser.parse_args()


def check_paths(args: argparse.Namespace) -> bool:
    """Verify all required input paths exist. Returns True if all OK."""
    ok = True
    for label, path in [
        ("Video directory", args.video_dir),
        ("Calibration JSON", args.calibration),
        ("U-Net weights", args.unet_weights),
        ("YOLO weights", args.yolo_weights),
    ]:
        if path.exists():
            print(f"  [OK]   {label}: {path}")
        else:
            print(f"  [MISS] {label}: {path}")
            ok = False
    return ok


def print_timing(stage_timing: dict[str, float]) -> None:
    """Print a formatted timing table."""
    total = sum(stage_timing.values())
    print(f"\n{'Stage':<25} {'Seconds':>10} {'% Total':>10}")
    print("-" * 47)
    for stage_name, elapsed in stage_timing.items():
        pct = 100.0 * elapsed / total if total > 0 else 0.0
        print(f"{stage_name:<25} {elapsed:>10.2f} {pct:>9.1f}%")
    print("-" * 47)
    print(f"{'TOTAL':<25} {total:>10.2f} {'100.0':>9}%")


def main() -> int:
    """Run pipeline stages directly and generate all diagnostic visualizations."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    print("=== AquaPose Pipeline Diagnostics ===\n")
    print("Checking input paths...")
    if not check_paths(args):
        print("\nAborting: one or more required paths are missing.")
        return 1

    # -----------------------------------------------------------------------
    # Setup (replicates orchestrator logic)
    # -----------------------------------------------------------------------
    from aquapose.calibration.loader import (
        compute_undistortion_maps,
        load_calibration_data,
    )
    from aquapose.calibration.projection import RefractiveProjectionModel
    from aquapose.io.midline_writer import Midline3DWriter
    from aquapose.io.video import VideoSet
    from aquapose.pipeline.stages import (
        run_detection,
        run_midline_extraction,
        run_segmentation,
        run_triangulation,
    )
    from aquapose.reconstruction.midline import MidlineExtractor
    from aquapose.segmentation.model import UNetSegmentor
    from aquapose.tracking.tracker import FishTracker
    from aquapose.visualization.diagnostics import (
        TrackSnapshot,
        vis_arclength_histogram,
        vis_claiming_overlay,
        vis_confidence_histogram,
        vis_detection_grid,
        vis_midline_extraction_montage,
        vis_residual_heatmap,
        vis_skip_reason_pie,
        vis_spline_camera_overlay,
        write_diagnostic_report,
    )
    from aquapose.visualization.plot3d import render_3d_animation

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Discover camera videos
    video_paths: dict[str, Path] = {}
    for suffix in ("*.avi", "*.mp4"):
        for p in args.video_dir.glob(suffix):
            camera_id = p.stem.split("-")[0]
            if camera_id == _SKIP_CAMERA_ID:
                logger.info("Skipping excluded camera: %s", camera_id)
                continue
            video_paths[camera_id] = p

    if not video_paths:
        print(f"No .avi/.mp4 files found in {args.video_dir}")
        return 1

    print(f"Found {len(video_paths)} cameras: {sorted(video_paths)}")

    # Load calibration + undistortion
    calib = load_calibration_data(args.calibration)
    undist_maps = {}
    models: dict[str, RefractiveProjectionModel] = {}
    for cam_id in video_paths:
        if cam_id not in calib.cameras:
            logger.warning("Camera %r not in calibration; skipping", cam_id)
            continue
        cam_data = calib.cameras[cam_id]
        maps = compute_undistortion_maps(cam_data)
        undist_maps[cam_id] = maps
        models[cam_id] = RefractiveProjectionModel(
            K=maps.K_new,
            R=cam_data.R,
            t=cam_data.t,
            water_z=calib.water_z,
            normal=calib.interface_normal,
            n_air=calib.n_air,
            n_water=calib.n_water,
        )

    if not models:
        print("No cameras matched between video_dir and calibration.")
        return 1

    # Create stateful objects
    max_fish = 9
    tracker = FishTracker(expected_count=max_fish)
    extractor = MidlineExtractor()
    segmentor = UNetSegmentor(weights_path=args.unet_weights)

    stage_timing: dict[str, float] = {}

    # -----------------------------------------------------------------------
    # Stage 1: Detection
    # -----------------------------------------------------------------------
    print("\nRunning Stage 1: Detection...")
    t0 = time.perf_counter()
    with VideoSet(video_paths, undistortion=undist_maps) as det_video_set:
        detections_per_frame = run_detection(
            video_set=det_video_set,
            stop_frame=args.stop_frame,
            detector_kind="yolo",
            model_path=args.yolo_weights,
        )
    stage_timing["detection"] = time.perf_counter() - t0

    # -----------------------------------------------------------------------
    # Stage 2: Segmentation
    # -----------------------------------------------------------------------
    print("Running Stage 2: Segmentation...")
    t0 = time.perf_counter()
    with VideoSet(video_paths, undistortion=undist_maps) as seg_video_set:
        masks_per_frame = run_segmentation(
            detections_per_frame=detections_per_frame,
            video_set=seg_video_set,
            segmentor=segmentor,
            stop_frame=args.stop_frame,
        )
    stage_timing["segmentation"] = time.perf_counter() - t0

    # -----------------------------------------------------------------------
    # Stage 3: Tracking (manual loop for full snapshots)
    # -----------------------------------------------------------------------
    print("Running Stage 3: Tracking (with snapshots)...")
    t0 = time.perf_counter()
    tracks_per_frame: list[list] = []
    snapshots_per_frame: list[list[TrackSnapshot]] = []

    for frame_idx, frame_dets in enumerate(detections_per_frame):
        confirmed = tracker.update(frame_dets, models, frame_index=frame_idx)
        tracks_per_frame.append(confirmed)

        # Snapshot ALL live tracks (not just confirmed)
        frame_snapshots: list[TrackSnapshot] = []
        for track in tracker.tracks:
            if len(track.positions) > 0:
                frame_snapshots.append(
                    TrackSnapshot(
                        fish_id=track.fish_id,
                        position=np.array(list(track.positions)[-1]),
                        state=track.state,
                        camera_detections=dict(track.camera_detections),
                    )
                )
        snapshots_per_frame.append(frame_snapshots)

    stage_timing["tracking"] = time.perf_counter() - t0

    # -----------------------------------------------------------------------
    # Stage 4: Midline Extraction
    # -----------------------------------------------------------------------
    print("Running Stage 4: Midline Extraction...")
    t0 = time.perf_counter()
    midline_sets = run_midline_extraction(
        tracks_per_frame=tracks_per_frame,
        masks_per_frame=masks_per_frame,
        detections_per_frame=detections_per_frame,
        extractor=extractor,
    )
    stage_timing["midline_extraction"] = time.perf_counter() - t0

    # -----------------------------------------------------------------------
    # Stage 5: Triangulation
    # -----------------------------------------------------------------------
    print("Running Stage 5: Triangulation...")
    t0 = time.perf_counter()
    midlines_3d = run_triangulation(
        midline_sets=midline_sets,
        models=models,
    )
    stage_timing["triangulation"] = time.perf_counter() - t0

    # -----------------------------------------------------------------------
    # Write HDF5 output
    # -----------------------------------------------------------------------
    print("Writing HDF5 output...")
    t0 = time.perf_counter()
    h5_path = output_dir / "midlines_3d.h5"
    with Midline3DWriter(h5_path, max_fish=max_fish) as writer:
        for frame_idx, frame_midlines in enumerate(midlines_3d):
            writer.write_frame(frame_idx, frame_midlines)
    stage_timing["hdf5_write"] = time.perf_counter() - t0

    # -----------------------------------------------------------------------
    # Timing summary
    # -----------------------------------------------------------------------
    print("\n=== Stage Timing Summary ===")
    print_timing(stage_timing)

    n_frames = len(midlines_3d)
    non_empty = sum(1 for f in midlines_3d if f)
    print(f"\nFrames processed: {n_frames}")
    print(f"Frames with 3D midlines: {non_empty}/{n_frames}")

    # -----------------------------------------------------------------------
    # Diagnostic Visualizations
    # -----------------------------------------------------------------------
    print("\n=== Generating Diagnostic Visualizations ===")

    vis_funcs = [
        (
            "detection_grid.png",
            lambda: vis_detection_grid(
                detections_per_frame, video_paths, diag_dir / "detection_grid.png"
            ),
        ),
        (
            "confidence_histogram.png",
            lambda: vis_confidence_histogram(
                detections_per_frame, diag_dir / "confidence_histogram.png"
            ),
        ),
        (
            "3d_animation.mp4",
            lambda: render_3d_animation(midlines_3d, diag_dir / "3d_animation"),
        ),
        (
            "claiming_overlay.mp4",
            lambda: vis_claiming_overlay(
                snapshots_per_frame,
                detections_per_frame,
                video_paths,
                models,
                diag_dir / "claiming_overlay.mp4",
                cameras=args.cameras,
            ),
        ),
        (
            "midline_montage.png",
            lambda: vis_midline_extraction_montage(
                tracks_per_frame,
                masks_per_frame,
                detections_per_frame,
                video_paths,
                diag_dir / "midline_montage.png",
            ),
        ),
        (
            "skip_reasons.png",
            lambda: vis_skip_reason_pie(
                tracks_per_frame, masks_per_frame, diag_dir / "skip_reasons.png"
            ),
        ),
        (
            "residual_heatmap.png",
            lambda: vis_residual_heatmap(
                midlines_3d, diag_dir / "residual_heatmap.png"
            ),
        ),
        (
            "arclength_histogram.png",
            lambda: vis_arclength_histogram(
                midlines_3d, diag_dir / "arclength_histogram.png"
            ),
        ),
        (
            "spline_overlay.png",
            lambda: vis_spline_camera_overlay(
                midlines_3d, models, video_paths, diag_dir / "spline_overlay.png"
            ),
        ),
    ]

    for name, func in vis_funcs:
        try:
            print(f"  Generating {name}...")
            func()
        except Exception as exc:
            print(f"  [WARN] Failed to generate {name}: {exc}")
            logger.exception("Failed to generate %s", name)

    # -----------------------------------------------------------------------
    # Quantitative Markdown Report
    # -----------------------------------------------------------------------
    print("  Generating report.md...")
    try:
        write_diagnostic_report(
            output_path=diag_dir / "report.md",
            stage_timing=stage_timing,
            detections_per_frame=detections_per_frame,
            masks_per_frame=masks_per_frame,
            snapshots_per_frame=snapshots_per_frame,
            tracks_per_frame=tracks_per_frame,
            midlines_3d_per_frame=midlines_3d,
            n_cameras=len(models),
        )
    except Exception as exc:
        print(f"  [WARN] Failed to generate report.md: {exc}")
        logger.exception("Failed to generate report.md")

    print(f"\nDiagnostics written to: {diag_dir}")
    print(f"HDF5 output: {h5_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
