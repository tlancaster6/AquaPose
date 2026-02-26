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
import json
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

            # Compute mean distance between control points (orientation-invariant)
            fwd_errors = np.linalg.norm(
                recon_midline.control_points - gt_midline.control_points, axis=1
            )
            rev_errors = np.linalg.norm(
                recon_midline.control_points[::-1] - gt_midline.control_points,
                axis=1,
            )
            dist_per_ctrl = (
                rev_errors if rev_errors.mean() < fwd_errors.mean() else fwd_errors
            )
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


# ---------------------------------------------------------------------------
# Raw data serialization helpers
# ---------------------------------------------------------------------------


def _save_metadata(
    raw_dir: Path,
    args: argparse.Namespace,
    stage_timing: dict[str, float],
    n_cameras: int,
) -> None:
    """Save run metadata as JSON."""
    meta = {
        "method": args.method,
        "stop_frame": args.stop_frame,
        "n_cameras": n_cameras,
        "synthetic": args.synthetic,
        "stage_timing": stage_timing,
    }
    if args.synthetic:
        meta["n_fish"] = args.n_fish
        meta["n_synthetic_cameras"] = args.n_synthetic_cameras
    else:
        meta["video_dir"] = str(args.video_dir)
        meta["calibration"] = str(args.calibration)
    (raw_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def _save_models(
    raw_dir: Path,
    models: dict[str, object],
) -> None:
    """Save camera projection parameters as npz."""
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, object] = {}
    for cam_id, model in models.items():
        arrays[f"K_{cam_id}"] = model.K.cpu().numpy()
        arrays[f"R_{cam_id}"] = model.R.cpu().numpy()
        arrays[f"t_{cam_id}"] = model.t.cpu().numpy()
        arrays[f"normal_{cam_id}"] = model.normal.cpu().numpy()
        # Scalars are the same across cameras; just grab from last
        scalars["water_z"] = float(model.water_z)
        scalars["n_air"] = float(model.n_air)
        scalars["n_water"] = float(model.n_water)
    # Store camera list and scalars as small arrays
    arrays["camera_ids"] = np.array(sorted(models.keys()))
    arrays["water_z"] = np.array([scalars["water_z"]])
    arrays["n_air"] = np.array([scalars["n_air"]])
    arrays["n_water"] = np.array([scalars["n_water"]])
    np.savez_compressed(raw_dir / "models.npz", **arrays)


def _save_video_frames(
    raw_dir: Path,
    video_set: object,
    n_frames: int,
    cam_ids: list[str],
) -> None:
    """Save undistorted video frames as one compressed npz per camera."""
    frames_dir = raw_dir / "video_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Collect all frames first, grouped by camera
    cam_frames: dict[str, dict[str, np.ndarray]] = {c: {} for c in cam_ids}
    for frame_idx in range(n_frames):
        frame_dict = video_set.read_frame(frame_idx)
        for cam_id in cam_ids:
            if cam_id in frame_dict:
                cam_frames[cam_id][f"frame_{frame_idx:04d}"] = frame_dict[cam_id]
    for cam_id, frames in cam_frames.items():
        if frames:
            np.savez_compressed(frames_dir / f"{cam_id}.npz", **frames)


def _save_detections(
    raw_dir: Path,
    detections_per_frame: list[dict[str, list]],
) -> None:
    """Save detection scalars as JSON and mask arrays as compressed npz."""
    # JSON for scalars (bbox, area, confidence)
    json_data: list[dict[str, list[dict]]] = []
    mask_dir = raw_dir / "detection_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame_dets in enumerate(detections_per_frame):
        frame_json: dict[str, list[dict]] = {}
        for cam_id, dets in frame_dets.items():
            cam_json: list[dict] = []
            mask_arrays: dict[str, np.ndarray] = {}
            for det_idx, det in enumerate(dets):
                cam_json.append(
                    {
                        "bbox": list(det.bbox),
                        "area": det.area,
                        "confidence": det.confidence,
                    }
                )
                if det.mask is not None:
                    mask_arrays[f"mask_{det_idx}"] = det.mask
            frame_json[cam_id] = cam_json
            if mask_arrays:
                np.savez_compressed(
                    mask_dir / f"frame_{frame_idx:04d}_{cam_id}.npz",
                    **mask_arrays,
                )
        json_data.append(frame_json)

    (raw_dir / "detections.json").write_text(json.dumps(json_data, indent=1))


def _save_segmentation_masks(
    raw_dir: Path,
    masks_per_frame: list[dict[str, list]],
) -> None:
    """Save post-segmentation mask+crop pairs as compressed npz."""
    seg_dir = raw_dir / "segmentation_masks"
    seg_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame_masks in enumerate(masks_per_frame):
        for cam_id, mask_crop_list in frame_masks.items():
            arrays: dict[str, np.ndarray] = {}
            crop_data: list[dict] = []
            for i, (mask, crop) in enumerate(mask_crop_list):
                arrays[f"mask_{i}"] = mask
                crop_data.append(
                    {
                        "x1": crop.x1,
                        "y1": crop.y1,
                        "x2": crop.x2,
                        "y2": crop.y2,
                        "frame_h": crop.frame_h,
                        "frame_w": crop.frame_w,
                    }
                )
            if arrays:
                np.savez_compressed(
                    seg_dir / f"frame_{frame_idx:04d}_{cam_id}.npz",
                    **arrays,
                )
                # Store crop metadata as companion JSON
                crop_path = seg_dir / f"frame_{frame_idx:04d}_{cam_id}_crops.json"
                crop_path.write_text(json.dumps(crop_data))


def _save_track_snapshots(
    raw_dir: Path,
    snapshots_per_frame: list[list],
) -> None:
    """Save per-frame TrackSnapshot data as JSON."""
    json_data: list[list[dict]] = []
    for frame_snapshots in snapshots_per_frame:
        frame_json: list[dict] = []
        for snap in frame_snapshots:
            frame_json.append(
                {
                    "fish_id": snap.fish_id,
                    "position": snap.position.tolist(),
                    "state": snap.state.value,
                    "camera_detections": snap.camera_detections,
                }
            )
        json_data.append(frame_json)
    (raw_dir / "track_snapshots.json").write_text(json.dumps(json_data, indent=1))


def _save_tracks(
    raw_dir: Path,
    tracks_per_frame: list[list],
) -> None:
    """Save per-frame confirmed track data as JSON."""
    json_data: list[list[dict]] = []
    for frame_tracks in tracks_per_frame:
        frame_json: list[dict] = []
        for track in frame_tracks:
            frame_json.append(
                {
                    "fish_id": track.fish_id,
                    "state": track.state.value,
                    "camera_detections": track.camera_detections,
                    "bboxes": {cam: list(bb) for cam, bb in track.bboxes.items()},
                    "position": list(track.positions)[-1].tolist()
                    if track.positions
                    else None,
                    "n_cameras": track.n_cameras,
                    "confidence": track.confidence,
                }
            )
        json_data.append(frame_json)
    (raw_dir / "tracks.json").write_text(json.dumps(json_data, indent=1))


def _save_midlines_2d(
    raw_dir: Path,
    midline_sets: list[dict[int, dict[str, object]]],
) -> None:
    """Save 2D midline points/half_widths as compressed npz."""
    arrays: dict[str, np.ndarray] = {}
    meta: list[dict] = []
    for frame_idx, frame_set in enumerate(midline_sets):
        for fish_id, cam_dict in frame_set.items():
            for cam_id, m2d in cam_dict.items():
                key = f"pts_{frame_idx}_{fish_id}_{cam_id}"
                arrays[key] = m2d.points
                arrays[f"hw_{frame_idx}_{fish_id}_{cam_id}"] = m2d.half_widths
                meta.append(
                    {
                        "frame": frame_idx,
                        "fish_id": fish_id,
                        "camera_id": cam_id,
                        "is_head_to_tail": m2d.is_head_to_tail,
                    }
                )
    np.savez_compressed(raw_dir / "midlines_2d.npz", **arrays)
    (raw_dir / "midlines_2d_meta.json").write_text(json.dumps(meta, indent=1))


def _save_midlines_3d(
    raw_dir: Path,
    midlines_3d: list[dict[int, object]],
) -> None:
    """Save 3D midline data as compressed npz + companion JSON."""
    from aquapose.reconstruction.triangulation import Midline3D

    arrays: dict[str, np.ndarray] = {}
    meta: list[dict] = []
    for frame_idx, frame_dict in enumerate(midlines_3d):
        for fish_id, m3d in frame_dict.items():
            if not isinstance(m3d, Midline3D):
                continue
            prefix = f"{frame_idx}_{fish_id}"
            arrays[f"ctrl_{prefix}"] = m3d.control_points
            arrays[f"knots_{prefix}"] = m3d.knots
            arrays[f"hw_{prefix}"] = m3d.half_widths
            meta.append(
                {
                    "frame": frame_idx,
                    "fish_id": fish_id,
                    "arc_length": m3d.arc_length,
                    "mean_residual": m3d.mean_residual,
                    "max_residual": m3d.max_residual,
                    "n_cameras": m3d.n_cameras,
                    "degree": m3d.degree,
                    "is_low_confidence": m3d.is_low_confidence,
                    "per_camera_residuals": m3d.per_camera_residuals,
                }
            )
    np.savez_compressed(raw_dir / "midlines_3d.npz", **arrays)
    (raw_dir / "midlines_3d_meta.json").write_text(json.dumps(meta, indent=1))


def _save_ground_truths(
    raw_dir: Path,
    ground_truths: list[dict[int, object]],
) -> None:
    """Save ground truth midlines (same format as midlines_3d)."""
    from aquapose.reconstruction.triangulation import Midline3D

    arrays: dict[str, np.ndarray] = {}
    meta: list[dict] = []
    for frame_idx, frame_dict in enumerate(ground_truths):
        for fish_id, m3d in frame_dict.items():
            if not isinstance(m3d, Midline3D):
                continue
            prefix = f"{frame_idx}_{fish_id}"
            arrays[f"ctrl_{prefix}"] = m3d.control_points
            arrays[f"knots_{prefix}"] = m3d.knots
            arrays[f"hw_{prefix}"] = m3d.half_widths
            meta.append(
                {
                    "frame": frame_idx,
                    "fish_id": fish_id,
                    "arc_length": m3d.arc_length,
                    "mean_residual": m3d.mean_residual,
                    "max_residual": m3d.max_residual,
                    "n_cameras": m3d.n_cameras,
                    "degree": m3d.degree,
                }
            )
    np.savez_compressed(raw_dir / "ground_truths.npz", **arrays)
    (raw_dir / "ground_truths_meta.json").write_text(json.dumps(meta, indent=1))


def _save_fish_configs(
    raw_dir: Path,
    fish_configs: list,
) -> None:
    """Save FishConfig parameters as JSON."""
    from dataclasses import asdict

    configs = [asdict(fc) for fc in fish_configs]
    # Convert tuples to lists for JSON
    for cfg in configs:
        cfg["position"] = list(cfg["position"])
        cfg["velocity"] = list(cfg["velocity"])
    (raw_dir / "fish_configs.json").write_text(json.dumps(configs, indent=2))


def _save_optimizer_snapshots(
    raw_dir: Path,
    snapshots: list,
) -> None:
    """Save OptimizerSnapshot arrays as compressed npz."""
    arrays: dict[str, np.ndarray] = {}
    meta: list[dict] = []
    for i, snap in enumerate(snapshots):
        arrays[f"obs_2d_{i}"] = snap.obs_2d
        arrays[f"cold_start_ctrl_{i}"] = snap.cold_start_ctrl
        arrays[f"post_coarse_ctrl_{i}"] = snap.post_coarse_ctrl
        arrays[f"post_fine_ctrl_{i}"] = snap.post_fine_ctrl
        meta.append(
            {
                "index": i,
                "fish_id": snap.fish_id,
                "best_cam_id": snap.best_cam_id,
                "cold_start_loss": snap.cold_start_loss,
                "coarse_loss": snap.coarse_loss,
                "fine_loss": snap.fine_loss,
            }
        )
    np.savez_compressed(raw_dir / "optimizer_snapshots.npz", **arrays)
    (raw_dir / "optimizer_snapshots_meta.json").write_text(json.dumps(meta, indent=2))


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
        vis_optimizer_progression,
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
    # Build fish configs — diverse shapes to expose failure modes
    # -----------------------------------------------------------------------
    # Each fish gets a unique combination of curvature, heading, sinusoidal
    # amplitude, and drift. The pattern cycles through shape archetypes.
    # Sinusoidal amplitudes are ~15-25% of body length for clear visibility.
    _SHAPE_PRESETS: list[dict[str, float]] = [
        # 0: straight — baseline
        {"curvature": 0.0, "sinusoidal_amplitude": 0.0, "sinusoidal_periods": 1.0},
        # 1: gentle arc (C-shape)
        {"curvature": 10.0, "sinusoidal_amplitude": 0.0, "sinusoidal_periods": 1.0},
        # 2: S-curve — full sine period, large amplitude
        {"curvature": 0.0, "sinusoidal_amplitude": 0.015, "sinusoidal_periods": 1.0},
        # 3: tight arc
        {"curvature": 25.0, "sinusoidal_amplitude": 0.0, "sinusoidal_periods": 1.0},
        # 4: compound arc + S-wave
        {"curvature": 8.0, "sinusoidal_amplitude": 0.012, "sinusoidal_periods": 1.0},
        # 5: reverse arc
        {"curvature": -15.0, "sinusoidal_amplitude": 0.0, "sinusoidal_periods": 1.0},
        # 6: C-curve via half-period sine (one-sided bend)
        {"curvature": 0.0, "sinusoidal_amplitude": 0.015, "sinusoidal_periods": 0.5},
        # 7: gentle S-curve (smaller amplitude)
        {"curvature": 0.0, "sinusoidal_amplitude": 0.010, "sinusoidal_periods": 1.0},
        # 8: double-S (two full sine periods)
        {"curvature": 0.0, "sinusoidal_amplitude": 0.012, "sinusoidal_periods": 2.0},
        # 9: reverse tight arc
        {"curvature": -25.0, "sinusoidal_amplitude": 0.0, "sinusoidal_periods": 1.0},
        # 10: compound reverse arc + S-wave
        {"curvature": -10.0, "sinusoidal_amplitude": 0.010, "sinusoidal_periods": 1.0},
        # 11: very tight arc (near biomechanical limit)
        {"curvature": 35.0, "sinusoidal_amplitude": 0.0, "sinusoidal_periods": 1.0},
    ]

    fish_configs: list[FishConfig] = []
    for i in range(args.n_fish):
        # Spread fish in a line along X, offset in Y for variety
        x_pos = i * 0.1 - (args.n_fish - 1) * 0.05
        y_pos = 0.015 * (i % 3 - 1)  # slight Y scatter
        # Vary heading across fish (spread over ~180 degrees)
        heading = i * np.pi / max(args.n_fish, 1)
        # Cycle through shape presets
        preset = _SHAPE_PRESETS[i % len(_SHAPE_PRESETS)]
        # Alternate drift: odd fish drift, even fish stationary
        if i % 2 == 0:
            velocity = (0.0, 0.0, 0.0)
            angular_vel = 0.0
        else:
            velocity = (0.002, 0.001, 0.0)
            angular_vel = 0.05
        fish_configs.append(
            FishConfig(
                position=(x_pos, y_pos, 1.25),
                heading_rad=heading,
                curvature=preset["curvature"],
                sinusoidal_amplitude=preset["sinusoidal_amplitude"],
                sinusoidal_periods=preset["sinusoidal_periods"],
                scale=0.085,
                velocity=velocity,
                angular_velocity=angular_vel,
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
    # Save raw data for figure reproduction
    # -----------------------------------------------------------------------
    print("Saving raw data...")
    t0 = time.perf_counter()
    raw_dir = output_dir / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _save_models(raw_dir, models)
    _save_midlines_3d(raw_dir, midlines_3d)
    _save_ground_truths(raw_dir, ground_truths)
    _save_fish_configs(raw_dir, fish_configs)
    if args.method == "curve" and optimizer.snapshots:
        _save_optimizer_snapshots(raw_dir, optimizer.snapshots)
    _save_metadata(raw_dir, args, stage_timing, n_cameras=len(models))
    stage_timing["raw_data_save"] = time.perf_counter() - t0

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

    # Optimizer progression visualization (curve method only, frame 0)
    if args.method == "curve" and optimizer.snapshots:
        vis_funcs_syn_progression = [
            (
                "optimizer_progression.png",
                lambda: vis_optimizer_progression(
                    optimizer.snapshots, models, diag_dir / "optimizer_progression.png"
                ),
            ),
        ]
    else:
        vis_funcs_syn_progression = []

    vis_funcs_syn = [
        *vis_funcs_syn_progression,
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
    print(f"Raw data written to: {raw_dir}")
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
        vis_optimizer_progression,
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
    # Save raw data for figure reproduction
    # -----------------------------------------------------------------------
    print("Saving raw data...")
    t0 = time.perf_counter()
    raw_dir = output_dir / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _save_models(raw_dir, models)
    _save_detections(raw_dir, detections_per_frame)
    _save_segmentation_masks(raw_dir, masks_per_frame)
    _save_track_snapshots(raw_dir, snapshots_per_frame)
    _save_tracks(raw_dir, tracks_per_frame)
    _save_midlines_2d(raw_dir, midline_sets)
    _save_midlines_3d(raw_dir, midlines_3d)
    if args.method == "curve" and optimizer.snapshots:
        _save_optimizer_snapshots(raw_dir, optimizer.snapshots)
    stage_timing["raw_data_save"] = time.perf_counter() - t0

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
        # Save video frames inside VideoSet context
        print("  Saving video frames...")
        try:
            _save_video_frames(
                raw_dir, vis_video_set, args.stop_frame, list(models.keys())
            )
        except Exception as exc:
            print(f"  [WARN] Failed to save video frames: {exc}")
            logger.exception("Failed to save video frames")
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

        # Optimizer progression visualization (curve method only, frame 0)
        if args.method == "curve" and optimizer.snapshots:
            vis_funcs.append(
                (
                    "optimizer_progression.png",
                    lambda: vis_optimizer_progression(
                        optimizer.snapshots,
                        models,
                        diag_dir / "optimizer_progression.png",
                    ),
                )
            )

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

    # Save metadata last (includes final timing)
    _save_metadata(raw_dir, args, stage_timing, n_cameras=len(models))

    print(f"\nDiagnostics written to: {diag_dir}")
    print(f"Raw data written to: {raw_dir}")
    print(f"HDF5 output: {h5_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
