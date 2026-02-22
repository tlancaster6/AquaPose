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
    # Stage 5: Triangulation (with sub-step timing)
    # -----------------------------------------------------------------------
    print("Running Stage 5: Triangulation...")
    import scipy.interpolate as _scipy_interp
    import torch

    from aquapose.reconstruction.triangulation import (
        DEFAULT_INLIER_THRESHOLD,
        MIN_BODY_POINTS,
        N_SAMPLE_POINTS,
        SPLINE_K,
        SPLINE_KNOTS,
        _align_midline_orientations,
        _fit_spline,
        _pixel_half_width_to_metres,
        _triangulate_body_point,
    )

    tri_sub = {
        "orientation_alignment": 0.0,
        "ray_cast_triangulate": 0.0,
        "spline_fitting": 0.0,
        "half_width_conversion": 0.0,
    }
    t0 = time.perf_counter()
    midlines_3d: list[dict[int, object]] = []

    for frame_idx, midline_set in enumerate(midline_sets):
        results: dict[int, object] = {}
        for fish_id, cam_midlines in midline_set.items():
            # --- Sub-step: orientation alignment ---
            ts = time.perf_counter()
            cam_midlines = _align_midline_orientations(
                cam_midlines, models, DEFAULT_INLIER_THRESHOLD
            )
            tri_sub["orientation_alignment"] += time.perf_counter() - ts

            # --- Sub-step: per-body-point ray cast + triangulate ---
            ts = time.perf_counter()
            valid_indices: list[int] = []
            pts_3d_list: list[np.ndarray] = []
            per_point_residuals: list[float] = []
            per_point_n_cams: list[int] = []
            per_point_inlier_ids: list[list[str]] = []
            per_point_hw_px: list[float] = []
            per_point_depths: list[float] = []

            for i in range(N_SAMPLE_POINTS):
                pixels: dict[str, torch.Tensor] = {}
                hw_px_list: list[float] = []
                for cam_id, midline in cam_midlines.items():
                    if cam_id not in models:
                        continue
                    pixels[cam_id] = torch.from_numpy(midline.points[i]).float()
                    hw_px_list.append(float(midline.half_widths[i]))

                result = _triangulate_body_point(
                    pixels, models, DEFAULT_INLIER_THRESHOLD
                )
                if result is None:
                    continue

                pt3d, inlier_ids, mean_res = result
                pt3d_np = pt3d.detach().cpu().numpy().astype(np.float64)

                valid_indices.append(i)
                pts_3d_list.append(pt3d_np)
                per_point_residuals.append(mean_res)
                per_point_n_cams.append(len(inlier_ids))
                per_point_inlier_ids.append(inlier_ids)

                avg_hw_px = float(np.mean(hw_px_list)) if hw_px_list else 0.0
                per_point_hw_px.append(avg_hw_px)

                water_z = next(iter(models.values())).water_z
                depth_m = max(0.0, float(pt3d_np[2]) - water_z)
                per_point_depths.append(depth_m)
            tri_sub["ray_cast_triangulate"] += time.perf_counter() - ts

            if len(valid_indices) < MIN_BODY_POINTS:
                continue

            # --- Sub-step: spline fitting ---
            ts = time.perf_counter()
            u_param = np.array(
                [i / (N_SAMPLE_POINTS - 1) for i in valid_indices], dtype=np.float64
            )
            pts_3d_arr = np.stack(pts_3d_list, axis=0)

            spline_result = _fit_spline(u_param, pts_3d_arr)
            tri_sub["spline_fitting"] += time.perf_counter() - ts

            if spline_result is None:
                continue

            control_points, arc_length = spline_result

            # --- Sub-step: half-width conversion ---
            ts = time.perf_counter()
            hw_metres_valid: list[float] = []
            for _idx, (hw_px, depth_m, inlier_ids) in enumerate(
                zip(
                    per_point_hw_px,
                    per_point_depths,
                    per_point_inlier_ids,
                    strict=True,
                )
            ):
                if inlier_ids:
                    cam_model = models[inlier_ids[0]]
                    focal_px = float(
                        (cam_model.K[0, 0].item() + cam_model.K[1, 1].item()) / 2.0
                    )
                else:
                    focal_px = next(iter(models.values())).K[0, 0].item()
                hw_m = _pixel_half_width_to_metres(hw_px, depth_m, focal_px)
                hw_metres_valid.append(hw_m)

            hw_metres_all = np.zeros(N_SAMPLE_POINTS, dtype=np.float32)
            if len(valid_indices) >= 2:
                fill_bounds: tuple[float, float] = (
                    hw_metres_valid[0],
                    hw_metres_valid[-1],
                )
                interp = _scipy_interp.interp1d(
                    u_param,
                    np.array(hw_metres_valid, dtype=np.float64),
                    kind="linear",
                    bounds_error=False,
                    fill_value=fill_bounds,  # type: ignore[arg-type]
                )
                u_all = np.linspace(0.0, 1.0, N_SAMPLE_POINTS)
                hw_metres_all = interp(u_all).astype(np.float32)
            elif len(valid_indices) == 1:
                hw_metres_all[:] = hw_metres_valid[0]
            tri_sub["half_width_conversion"] += time.perf_counter() - ts

            # Build Midline3D
            # Spline-based residuals: reproject fitted spline into each
            # observing camera and compare against 2D midline observations.
            import scipy.interpolate as _sp_interp

            from aquapose.reconstruction.triangulation import Midline3D

            _spline_obj = _sp_interp.BSpline(
                SPLINE_KNOTS, control_points.astype(np.float64), SPLINE_K
            )
            _u_sample = np.linspace(0.0, 1.0, N_SAMPLE_POINTS)
            _spline_pts_3d = torch.from_numpy(_spline_obj(_u_sample).astype(np.float32))
            _all_res: list[float] = []
            _cam_res: dict[str, float] = {}
            _active_cams = [c for c in cam_midlines if c in models]
            for _cid in _active_cams:
                _proj_px, _valid = models[_cid].project(_spline_pts_3d)
                _proj_np = _proj_px.detach().cpu().numpy()
                _valid_np = _valid.detach().cpu().numpy()
                _obs = cam_midlines[_cid].points
                _cam_errs: list[float] = []
                for _j in range(N_SAMPLE_POINTS):
                    if _valid_np[_j] and not np.any(np.isnan(_proj_np[_j])):
                        _e = float(np.linalg.norm(_proj_np[_j] - _obs[_j]))
                        _cam_errs.append(_e)
                        _all_res.append(_e)
                if _cam_errs:
                    _cam_res[_cid] = float(np.mean(_cam_errs))
            mean_residual = float(np.mean(_all_res)) if _all_res else 0.0
            max_residual_val = float(np.max(_all_res)) if _all_res else 0.0

            min_n_cams = min(per_point_n_cams) if per_point_n_cams else 0

            n_weak = sum(1 for nc in per_point_n_cams if nc < 3)
            is_low_confidence = n_weak > 0.2 * len(per_point_n_cams)

            midline_3d = Midline3D(
                fish_id=fish_id,
                frame_index=frame_idx,
                control_points=control_points,
                knots=SPLINE_KNOTS.astype(np.float32),
                degree=SPLINE_K,
                arc_length=arc_length,
                half_widths=hw_metres_all,
                n_cameras=min_n_cams,
                mean_residual=mean_residual,
                max_residual=max_residual_val,
                is_low_confidence=is_low_confidence,
                per_camera_residuals=_cam_res,
            )
            results[fish_id] = midline_3d

        midlines_3d.append(results)

    stage_timing["triangulation"] = time.perf_counter() - t0
    sub_timing: dict[str, dict[str, float]] = {"triangulation": tri_sub}

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
