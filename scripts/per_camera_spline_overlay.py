"""Per-camera spline overlay diagnostic for 3D triangulation debugging.

Runs the full 5-stage pipeline for 30 frames, then generates per-camera
overlay images at frame 29 showing:
  - Undistorted camera frame as background
  - 2D midline points (from midline extraction) as colored dots
  - 3D spline reprojections as colored polylines with width indicators
  - Per-fish annotation: fish_id, residual, arc_length

If the 2D dots land on the fish but the 3D lines don't, the triangulation
is broken. If both miss, it's a calibration or coordinate-space issue.

Usage:
    python scripts/per_camera_spline_overlay.py
    python scripts/per_camera_spline_overlay.py --stop-frame 60 --target-frame 59
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Defaults (same as diagnose_pipeline.py)
# ---------------------------------------------------------------------------
DEFAULT_VIDEO_DIR = Path("C:/Users/tucke/Desktop/Aqua/AquaPose/videos/core_videos")
DEFAULT_CALIBRATION = Path("C:/Users/tucke/Desktop/Aqua/AquaPose/calibration.json")
DEFAULT_UNET_WEIGHTS = Path("C:/Users/tucke/Desktop/Aqua/AquaPose/unet/best_model.pth")
DEFAULT_YOLO_WEIGHTS = Path(
    "C:/Users/tucke/Desktop/Aqua/AquaPose/yolo/train_v2/weights/best.pt"
)
DEFAULT_OUTPUT_DIR = Path("output/e2e_diagnostic")
DEFAULT_STOP_FRAME = 30
DEFAULT_TARGET_FRAME = 29

_SKIP_CAMERA_ID = "e3v8250"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate per-camera spline overlay diagnostics.",
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
        help="Directory for output",
    )
    parser.add_argument(
        "--stop-frame",
        type=int,
        default=DEFAULT_STOP_FRAME,
        help="Number of frames to process",
    )
    parser.add_argument(
        "--target-frame",
        type=int,
        default=DEFAULT_TARGET_FRAME,
        help="Frame index to generate overlays for (default: 29)",
    )
    return parser.parse_args()


def draw_2d_midline_dots(
    frame: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    *,
    radius: int = 4,
) -> None:
    """Draw 2D midline sample points as filled circles.

    Args:
        frame: BGR image, modified in-place.
        points: 2D pixel coords, shape (N, 2).
        color: BGR color.
        radius: Circle radius in pixels.
    """
    for pt in points:
        px, py = round(float(pt[0])), round(float(pt[1]))
        cv2.circle(frame, (px, py), radius, color, -1)
        # White outline for visibility
        cv2.circle(frame, (px, py), radius, (255, 255, 255), 1)


def annotate_fish(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw a text label with a dark background for readability.

    Args:
        frame: BGR image, modified in-place.
        text: Label string.
        position: (x, y) pixel position for text origin.
        color: BGR text color.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    # Clamp to frame bounds
    x = max(0, min(x, frame.shape[1] - tw - 4))
    y = max(th + 4, min(y, frame.shape[0] - 4))
    cv2.rectangle(
        frame,
        (x - 2, y - th - 4),
        (x + tw + 2, y + baseline + 2),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def main() -> int:
    """Run pipeline and generate per-camera spline overlay images."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    print("=== Per-Camera Spline Overlay Diagnostic ===\n")

    # Check paths
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
            print("\nAborting: required path missing.")
            return 1

    if args.target_frame >= args.stop_frame:
        print(
            f"Error: --target-frame ({args.target_frame}) must be < "
            f"--stop-frame ({args.stop_frame})"
        )
        return 1

    # -----------------------------------------------------------------------
    # Imports (heavy, deferred)
    # -----------------------------------------------------------------------
    import scipy.interpolate as _scipy_interp
    import torch

    from aquapose.calibration.loader import (
        compute_undistortion_maps,
        load_calibration_data,
    )
    from aquapose.calibration.projection import RefractiveProjectionModel
    from aquapose.io.video import VideoSet
    from aquapose.pipeline.stages import (
        run_detection,
        run_midline_extraction,
        run_segmentation,
    )
    from aquapose.reconstruction.midline import MidlineExtractor
    from aquapose.reconstruction.triangulation import (
        DEFAULT_INLIER_THRESHOLD,
        MIN_BODY_POINTS,
        N_SAMPLE_POINTS,
        SPLINE_K,
        SPLINE_KNOTS,
        Midline3D,
        _align_midline_orientations,
        _fit_spline,
        _pixel_half_width_to_metres,
        _triangulate_body_point,
    )
    from aquapose.segmentation.model import UNetSegmentor
    from aquapose.tracking.tracker import FishTracker
    from aquapose.visualization.overlay import FISH_COLORS, draw_midline_overlay

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    output_dir = args.output_dir
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Discover videos
    video_paths: dict[str, Path] = {}
    for suffix in ("*.avi", "*.mp4"):
        for p in args.video_dir.glob(suffix):
            camera_id = p.stem.split("-")[0]
            if camera_id == _SKIP_CAMERA_ID:
                continue
            video_paths[camera_id] = p

    if not video_paths:
        print(f"No videos found in {args.video_dir}")
        return 1

    print(f"Found {len(video_paths)} cameras: {sorted(video_paths)}")

    # Calibration + undistortion
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

    # -----------------------------------------------------------------------
    # Stage 1: Detection
    # -----------------------------------------------------------------------
    print(f"\nRunning pipeline for {args.stop_frame} frames...")
    t0 = time.perf_counter()

    print("  Stage 1: Detection...")
    with VideoSet(video_paths, undistortion=undist_maps) as det_vs:
        detections_per_frame = run_detection(
            video_set=det_vs,
            stop_frame=args.stop_frame,
            detector_kind="yolo",
            model_path=args.yolo_weights,
        )

    # -----------------------------------------------------------------------
    # Stage 2: Segmentation
    # -----------------------------------------------------------------------
    print("  Stage 2: Segmentation...")
    segmentor = UNetSegmentor(weights_path=args.unet_weights)
    with VideoSet(video_paths, undistortion=undist_maps) as seg_vs:
        masks_per_frame = run_segmentation(
            detections_per_frame=detections_per_frame,
            video_set=seg_vs,
            segmentor=segmentor,
            stop_frame=args.stop_frame,
        )

    # -----------------------------------------------------------------------
    # Stage 3: Tracking
    # -----------------------------------------------------------------------
    print("  Stage 3: Tracking...")
    tracker = FishTracker(expected_count=9)
    tracks_per_frame: list[list] = []
    for frame_idx, frame_dets in enumerate(detections_per_frame):
        confirmed = tracker.update(frame_dets, models, frame_index=frame_idx)
        tracks_per_frame.append(confirmed)

    # -----------------------------------------------------------------------
    # Stage 4: Midline Extraction
    # -----------------------------------------------------------------------
    print("  Stage 4: Midline Extraction...")
    extractor = MidlineExtractor()
    midline_sets = run_midline_extraction(
        tracks_per_frame=tracks_per_frame,
        masks_per_frame=masks_per_frame,
        detections_per_frame=detections_per_frame,
        extractor=extractor,
    )

    # -----------------------------------------------------------------------
    # Stage 5: Triangulation (same logic as diagnose_pipeline.py)
    # -----------------------------------------------------------------------
    print("  Stage 5: Triangulation...")
    midlines_3d: list[dict[int, Midline3D]] = []

    for frame_idx, midline_set in enumerate(midline_sets):
        results: dict[int, Midline3D] = {}
        for fish_id, cam_midlines in midline_set.items():
            cam_midlines = _align_midline_orientations(
                cam_midlines, models, DEFAULT_INLIER_THRESHOLD
            )

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

            if len(valid_indices) < MIN_BODY_POINTS:
                continue

            u_param = np.array(
                [i / (N_SAMPLE_POINTS - 1) for i in valid_indices], dtype=np.float64
            )
            pts_3d_arr = np.stack(pts_3d_list, axis=0)

            spline_result = _fit_spline(u_param, pts_3d_arr)
            if spline_result is None:
                continue

            control_points, arc_length = spline_result

            # Half-width conversion
            hw_metres_valid: list[float] = []
            for hw_px, depth_m, inlier_ids in zip(
                per_point_hw_px, per_point_depths, per_point_inlier_ids, strict=True
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
                fill_bounds = (hw_metres_valid[0], hw_metres_valid[-1])
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

            # Spline-based residuals
            _spline_obj = _scipy_interp.BSpline(
                SPLINE_KNOTS, control_points.astype(np.float64), SPLINE_K
            )
            _u_sample = np.linspace(0.0, 1.0, N_SAMPLE_POINTS)
            _spline_pts_3d = torch.from_numpy(_spline_obj(_u_sample).astype(np.float32))
            _all_res: list[float] = []
            _active_cams = [c for c in cam_midlines if c in models]
            for _cid in _active_cams:
                _proj_px, _valid = models[_cid].project(_spline_pts_3d)
                _proj_np = _proj_px.detach().cpu().numpy()
                _valid_np = _valid.detach().cpu().numpy()
                _obs = cam_midlines[_cid].points
                for _j in range(N_SAMPLE_POINTS):
                    if _valid_np[_j] and not np.any(np.isnan(_proj_np[_j])):
                        _all_res.append(float(np.linalg.norm(_proj_np[_j] - _obs[_j])))
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
            )
            results[fish_id] = midline_3d

        midlines_3d.append(results)

    elapsed = time.perf_counter() - t0
    print(f"Pipeline complete in {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Generate per-camera overlay images at target frame
    # -----------------------------------------------------------------------
    target = args.target_frame
    print(f"\n=== Generating overlays for frame {target} ===")

    if target >= len(midlines_3d):
        print(f"Error: target frame {target} out of range (have {len(midlines_3d)})")
        return 1

    frame_3d = midlines_3d[target]
    frame_2d = midline_sets[target] if target < len(midline_sets) else {}

    print(
        f"  3D midlines at frame {target}: {len(frame_3d)} fish "
        f"(IDs: {sorted(frame_3d.keys())})"
    )
    print(
        f"  2D midline sets at frame {target}: {len(frame_2d)} fish "
        f"(IDs: {sorted(frame_2d.keys())})"
    )

    # Print per-fish 3D metrics
    for fid, m3d in sorted(frame_3d.items()):
        conf = "LOW" if m3d.is_low_confidence else "OK"
        print(
            f"    fish {fid}: residual={m3d.mean_residual:.1f}px  "
            f"arc={m3d.arc_length:.4f}m  cams={m3d.n_cameras}  [{conf}]"
        )

    # Read undistorted frames for the target frame
    print("\n  Reading undistorted frames...")
    with VideoSet(video_paths, undistortion=undist_maps) as vs:
        frames = vs.read_frame(target)

    n_saved = 0
    for cam_id in sorted(models.keys()):
        if cam_id not in frames:
            logger.warning("No frame for camera %s", cam_id)
            continue

        frame = frames[cam_id].copy()
        model = models[cam_id]

        # --- Draw 3D spline reprojections (thick colored lines) ---
        for fish_id, m3d in sorted(frame_3d.items()):
            color = FISH_COLORS[fish_id % len(FISH_COLORS)]
            draw_midline_overlay(
                frame,
                m3d,
                model,
                color=color,
                thickness=3,
                n_eval=40,
                draw_widths=True,
            )

            # Annotate with fish_id, residual, arc_length near the spline head
            # Project the first control point to find label position
            spline = _scipy_interp.BSpline(
                m3d.knots.astype(np.float64),
                m3d.control_points.astype(np.float64),
                m3d.degree,
            )
            head_3d = spline(0.0).astype(np.float32)
            head_px, head_valid = model.project(torch.from_numpy(head_3d.reshape(1, 3)))
            if head_valid[0]:
                hx = round(float(head_px[0, 0]))
                hy = round(float(head_px[0, 1]))
                label = (
                    f"F{fish_id} res={m3d.mean_residual:.0f}px "
                    f"arc={m3d.arc_length:.3f}m"
                )
                annotate_fish(frame, label, (hx + 5, hy - 10), color)

        # --- Draw 2D midline dots (smaller, with white outline) ---
        for fish_id, cam_midlines in frame_2d.items():
            if cam_id not in cam_midlines:
                continue
            m2d = cam_midlines[cam_id]
            color = FISH_COLORS[fish_id % len(FISH_COLORS)]
            # Use a slightly different shade for 2D dots (brighter)
            draw_2d_midline_dots(frame, m2d.points, color, radius=4)

        # --- Camera label ---
        cv2.putText(
            frame,
            f"Camera: {cam_id}  Frame: {target}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        # Legend
        cv2.putText(
            frame,
            "Lines=3D spline  Dots=2D midline",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        out_path = diag_dir / f"spline_overlay_{cam_id}.png"
        cv2.imwrite(str(out_path), frame)
        n_saved += 1
        print(f"  Saved: {out_path}")

    print(f"\nDone. {n_saved} overlay images written to {diag_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
