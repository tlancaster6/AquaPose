"""Diagnostic script for the full 5-stage AquaPose reconstruction pipeline.

Calls pipeline stages directly to retain all intermediate data, then invokes
10 per-stage diagnostic visualizations. Outputs are written to a diagnostics
subdirectory alongside the standard HDF5 output.

Usage:
    python scripts/diagnose_pipeline.py
    python scripts/diagnose_pipeline.py --stop-frame 100
    python scripts/diagnose_pipeline.py --video-dir /path/to/videos --output-dir output/diag
    python scripts/diagnose_pipeline.py --synthetic --stop-frame 5
    python scripts/diagnose_pipeline.py --synthetic --n-fish 3 --method curve
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
DEFAULT_UNET_WEIGHTS = Path(
    "C:/Users/tucke/Desktop/Aqua/AquaPose/unet/run2/best_model.pth"
)
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
    parser.add_argument(
        "--method",
        choices=["triangulation", "curve"],
        default="triangulation",
        help="Reconstruction method: triangulation (current) or curve (new optimizer)",
    )
    # Synthetic data arguments
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="Use synthetic fish data instead of real video (bypasses stages 1-4)",
    )
    parser.add_argument(
        "--n-fish",
        type=int,
        default=1,
        help="Number of synthetic fish (only with --synthetic)",
    )
    parser.add_argument(
        "--n-synthetic-cameras",
        type=int,
        default=4,
        help=(
            "Number of cameras per axis for fabricated rig (NxN grid, "
            "only with --synthetic without --calibration)"
        ),
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


def print_timing(
    stage_timing: dict[str, float],
    sub_timing: dict[str, dict[str, float]] | None = None,
) -> None:
    """Print a formatted timing table with optional sub-stage breakdowns.

    Args:
        stage_timing: Top-level stage name to elapsed seconds.
        sub_timing: Optional mapping of stage name to dict of sub-step timings.
            When present, sub-steps are printed indented beneath the parent stage.
    """
    total = sum(stage_timing.values())
    print(f"\n{'Stage':<35} {'Seconds':>10} {'% Total':>10}")
    print("-" * 57)
    for stage_name, elapsed in stage_timing.items():
        pct = 100.0 * elapsed / total if total > 0 else 0.0
        print(f"{stage_name:<35} {elapsed:>10.2f} {pct:>9.1f}%")
        if sub_timing and stage_name in sub_timing:
            for sub_name, sub_elapsed in sub_timing[stage_name].items():
                sub_pct = 100.0 * sub_elapsed / total if total > 0 else 0.0
                print(f"  {sub_name:<33} {sub_elapsed:>10.2f} {sub_pct:>9.1f}%")
    print("-" * 57)
    print(f"{'TOTAL':<35} {total:>10.2f} {'100.0':>9}%")


def _print_ground_truth_comparison(
    midlines_3d: list[dict[int, object]],
    ground_truths: list[dict[int, object]],
) -> None:
    """Print ground truth comparison between reconstructed and synthetic midlines.

    For each fish in each frame, computes the mean Euclidean distance between
    the reconstructed B-spline control points and the ground truth control
    points (after matching by fish_id). Results are printed per-fish and as
    an overall mean, in millimetres.

    Args:
        midlines_3d: Reconstructed Midline3D dicts, one per frame.
        ground_truths: Ground truth Midline3D dicts, one per frame.
    """
    from aquapose.reconstruction.triangulation import Midline3D

    print("\n=== Synthetic Ground Truth Comparison ===")

    per_fish_errors: dict[int, list[float]] = {}

    for frame_idx, (recon_frame, gt_frame) in enumerate(
        zip(midlines_3d, ground_truths, strict=True)
    ):
        for fish_id, gt_midline in gt_frame.items():
            if fish_id not in recon_frame:
                logger.debug(
                    "Frame %d: fish %d not in reconstruction (skipped)",
                    frame_idx,
                    fish_id,
                )
                continue

            recon_midline = recon_frame[fish_id]
            if not isinstance(recon_midline, Midline3D) or not isinstance(
                gt_midline, Midline3D
            ):
                continue

            # Compute mean distance between control points (in metres)
            diff = recon_midline.control_points - gt_midline.control_points
            dist_per_ctrl = np.linalg.norm(diff, axis=1)  # shape (SPLINE_N_CTRL,)
            mean_dist_m = float(np.mean(dist_per_ctrl))
            mean_dist_mm = mean_dist_m * 1000.0

            if fish_id not in per_fish_errors:
                per_fish_errors[fish_id] = []
            per_fish_errors[fish_id].append(mean_dist_mm)

    if not per_fish_errors:
        print("  No reconstructed fish matched ground truth fish IDs.")
        return

    all_errors: list[float] = []
    for fish_id in sorted(per_fish_errors):
        errors = per_fish_errors[fish_id]
        mean_err = float(np.mean(errors))
        all_errors.extend(errors)
        print(
            f"  Fish {fish_id}: mean control-point error = {mean_err:.2f} mm "
            f"over {len(errors)} frame(s)"
        )

    overall_mean = float(np.mean(all_errors))
    print(f"\n  Overall mean control-point error: {overall_mean:.2f} mm")


def main() -> int:
    """Run pipeline stages directly and generate all diagnostic visualizations."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    print("=== AquaPose Pipeline Diagnostics ===\n")

    if args.synthetic:
        print("Mode: SYNTHETIC (bypassing stages 1-4)\n")
        return _run_synthetic(args)

    print("Checking input paths...")
    if not check_paths(args):
        print("\nAborting: one or more required paths are missing.")
        return 1

    return _run_real(args)


def _run_synthetic(args: argparse.Namespace) -> int:
    """Run the pipeline in synthetic mode, bypassing stages 1-4.

    Generates synthetic fish midlines from configurable fish shapes, feeds
    them directly to Stage 5 (triangulation or curve optimizer), and computes
    a ground truth comparison after reconstruction.

    Args:
        args: Parsed command-line arguments with synthetic-mode flags set.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    from aquapose.calibration.projection import RefractiveProjectionModel
    from aquapose.io.midline_writer import Midline3DWriter
    from aquapose.synthetic import (
        FishConfig,
        build_fabricated_rig,
        generate_synthetic_midline_sets,
    )
    from aquapose.visualization.diagnostics import (
        vis_arclength_histogram,
        vis_residual_heatmap,
        vis_synthetic_3d_comparison,
        vis_synthetic_camera_overlays,
        vis_synthetic_error_distribution,
        write_synthetic_report,
    )
    from aquapose.visualization.plot3d import render_3d_animation

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    stage_timing: dict[str, float] = {}

    # -----------------------------------------------------------------------
    # Build camera models
    # -----------------------------------------------------------------------
    models: dict[str, RefractiveProjectionModel]
    if args.calibration.exists():
        print(f"  Loading calibration from: {args.calibration}")
        from aquapose.calibration.loader import (
            compute_undistortion_maps,
            load_calibration_data,
        )

        calib = load_calibration_data(args.calibration)
        models = {}
        for cam_id, cam_data in calib.cameras.items():
            if cam_id == _SKIP_CAMERA_ID:
                continue
            maps = compute_undistortion_maps(cam_data)
            models[cam_id] = RefractiveProjectionModel(
                K=maps.K_new,
                R=cam_data.R,
                t=cam_data.t,
                water_z=calib.water_z,
                normal=calib.interface_normal,
                n_air=calib.n_air,
                n_water=calib.n_water,
            )
        print(f"  Using {len(models)} cameras from calibration.")
    else:
        n = args.n_synthetic_cameras
        models = build_fabricated_rig(n_cameras_x=n, n_cameras_y=n)
        print(f"  Using fabricated {n}x{n} rig ({len(models)} cameras).")

    # -----------------------------------------------------------------------
    # Build fish configs
    # -----------------------------------------------------------------------
    fish_configs: list[FishConfig] = []
    for i in range(args.n_fish):
        x_pos = i * 0.1 - (args.n_fish - 1) * 0.05
        curvature = 0.0 if i % 2 == 0 else 15.0
        fish_configs.append(
            FishConfig(
                position=(x_pos, 0.0, 1.25),
                heading_rad=0.0,
                curvature=curvature,
                scale=0.085,
            )
        )
    print(f"  Generating {args.n_fish} synthetic fish, {args.stop_frame} frame(s).\n")

    # -----------------------------------------------------------------------
    # Synthetic midline generation (replaces stages 1-4)
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    midline_sets, ground_truths = generate_synthetic_midline_sets(
        models=models,
        fish_configs=fish_configs,
        n_frames=args.stop_frame,
        frame_start=0,
    )
    stage_timing["synthetic_generation"] = time.perf_counter() - t0
    print(f"Generated {len(midline_sets)} synthetic frame(s).")

    # -----------------------------------------------------------------------
    # Stage 5: Triangulation / Curve Optimizer
    # -----------------------------------------------------------------------
    print(f"\nRunning Stage 5: Triangulation ({args.method})...")

    midlines_3d: list[dict[int, object]] = []
    sub_timing: dict[str, dict[str, float]] = {}

    if args.method == "curve":
        from aquapose.reconstruction.curve_optimizer import (
            CurveOptimizer,
            CurveOptimizerConfig,
        )

        t0 = time.perf_counter()
        optimizer = CurveOptimizer(config=CurveOptimizerConfig(max_depth=2.0))
        for frame_idx, midline_set in enumerate(midline_sets):
            results = optimizer.optimize_midlines(
                midline_set, models, frame_index=frame_idx
            )
            midlines_3d.append(results)
        stage_timing["triangulation"] = time.perf_counter() - t0

    else:
        from aquapose.reconstruction.triangulation import triangulate_midlines

        t0 = time.perf_counter()
        for frame_idx, midline_set in enumerate(midline_sets):
            frame_results = triangulate_midlines(
                midline_set, models, frame_index=frame_idx, max_depth=2.0
            )
            midlines_3d.append(frame_results)
        stage_timing["triangulation"] = time.perf_counter() - t0

    # -----------------------------------------------------------------------
    # Write HDF5 output
    # -----------------------------------------------------------------------
    print("Writing HDF5 output...")
    max_fish = max(args.n_fish, 1)
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
    print_timing(stage_timing, sub_timing=sub_timing)

    n_frames = len(midlines_3d)
    non_empty = sum(1 for f in midlines_3d if f)
    print(f"\nFrames processed: {n_frames}")
    print(f"Frames with 3D midlines: {non_empty}/{n_frames}")

    # -----------------------------------------------------------------------
    # Diagnostic Visualizations (synthetic-compatible subset)
    # -----------------------------------------------------------------------
    print("\n=== Generating Diagnostic Visualizations ===")

    vis_funcs_syn = [
        (
            "3d_animation",
            lambda: render_3d_animation(midlines_3d, diag_dir / "3d_animation"),
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
            "synthetic_3d_comparison.png",
            lambda: vis_synthetic_3d_comparison(
                midlines_3d, ground_truths, diag_dir / "synthetic_3d_comparison.png"
            ),
        ),
        (
            "synthetic_camera_overlays/",
            lambda: vis_synthetic_camera_overlays(
                midlines_3d,
                ground_truths,
                models,
                diag_dir / "synthetic_camera_overlays",
            ),
        ),
        (
            "synthetic_error_distribution.png",
            lambda: vis_synthetic_error_distribution(
                midlines_3d,
                ground_truths,
                diag_dir / "synthetic_error_distribution.png",
            ),
        ),
    ]

    for name, func in vis_funcs_syn:
        try:
            print(f"  Generating {name}...")
            func()
        except Exception as exc:
            print(f"  [WARN] Failed to generate {name}: {exc}")
            logger.exception("Failed to generate %s", name)

    # -----------------------------------------------------------------------
    # Ground truth comparison
    # -----------------------------------------------------------------------
    _print_ground_truth_comparison(midlines_3d, ground_truths)

    # -----------------------------------------------------------------------
    # Synthetic Markdown Report
    # -----------------------------------------------------------------------
    print("  Generating synthetic_report.md...")
    try:
        write_synthetic_report(
            output_path=diag_dir / "synthetic_report.md",
            stage_timing=stage_timing,
            midlines_3d=midlines_3d,
            ground_truths=ground_truths,
            models=models,
            fish_configs=fish_configs,
            method=args.method,
            diag_dir=diag_dir,
        )
    except Exception as exc:
        print(f"  [WARN] Failed to generate synthetic_report.md: {exc}")
        logger.exception("Failed to generate synthetic_report.md")

    print(f"\nDiagnostics written to: {diag_dir}")
    print(f"HDF5 output: {h5_path}")
    return 0


def _run_real(args: argparse.Namespace) -> int:
    """Run the full 5-stage pipeline on real video data.

    Args:
        args: Parsed command-line arguments (synthetic=False).

    Returns:
        Exit code (0 = success, 1 = failure).
    """
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
        vis_per_camera_spline_overlays,
        vis_residual_heatmap,
        vis_skip_reason_pie,
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
    # Stage 5: Triangulation / Curve Optimizer (with sub-step timing)
    # -----------------------------------------------------------------------
    print(f"Running Stage 5: Triangulation ({args.method})...")

    midlines_3d: list[dict[int, object]] = []
    sub_timing: dict[str, dict[str, float]] = {}

    if args.method == "curve":
        from aquapose.reconstruction.curve_optimizer import (
            CurveOptimizer,
            CurveOptimizerConfig,
        )

        t0 = time.perf_counter()
        optimizer = CurveOptimizer(config=CurveOptimizerConfig(max_depth=2.0))
        for frame_idx, midline_set in enumerate(midline_sets):
            results = optimizer.optimize_midlines(
                midline_set, models, frame_index=frame_idx
            )
            midlines_3d.append(results)
        stage_timing["triangulation"] = time.perf_counter() - t0

    else:
        from aquapose.reconstruction.triangulation import triangulate_midlines

        t0 = time.perf_counter()

        for frame_idx, midline_set in enumerate(midline_sets):
            frame_results = triangulate_midlines(
                midline_set, models, frame_index=frame_idx, max_depth=0.5
            )
            midlines_3d.append(frame_results)

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
    print_timing(stage_timing, sub_timing=sub_timing)

    n_frames = len(midlines_3d)
    non_empty = sum(1 for f in midlines_3d if f)
    print(f"\nFrames processed: {n_frames}")
    print(f"Frames with 3D midlines: {non_empty}/{n_frames}")

    # -----------------------------------------------------------------------
    # Diagnostic Visualizations
    # -----------------------------------------------------------------------
    print("\n=== Generating Diagnostic Visualizations ===")

    with VideoSet(video_paths, undistortion=undist_maps) as vis_video_set:
        vis_funcs = [
            (
                "detection_grid.png",
                lambda: vis_detection_grid(
                    detections_per_frame,
                    vis_video_set,
                    diag_dir / "detection_grid.png",
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
                    vis_video_set,
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
                    vis_video_set,
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
                "spline_overlays/",
                lambda: vis_per_camera_spline_overlays(
                    frame_index=len(midlines_3d) - 1,
                    frame_3d=midlines_3d[-1],
                    frame_2d=midline_sets[-1] if midline_sets else {},
                    models=models,
                    video_set=vis_video_set,
                    output_dir=diag_dir / "spline_overlays",
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
