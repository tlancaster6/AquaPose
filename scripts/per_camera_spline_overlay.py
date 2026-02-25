"""Per-camera spline overlay diagnostic for 3D reconstruction debugging.

Runs the full 5-stage pipeline for 30 frames, then generates per-camera
overlay images at a target frame showing:
  - Undistorted camera frame as background
  - 2D midline points (from midline extraction) as colored dots
  - 3D spline reprojections as colored polylines with width indicators
  - Per-fish annotation: fish_id, residual, arc_length

If the 2D dots land on the fish but the 3D lines don't, the reconstruction
is broken. If both miss, it's a calibration or coordinate-space issue.

Usage:
    python scripts/per_camera_spline_overlay.py
    python scripts/per_camera_spline_overlay.py --stop-frame 60 --target-frame 59
    python scripts/per_camera_spline_overlay.py --method curve
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
DEFAULT_UNET_WEIGHTS = Path(
    "C:/Users/tucke/Desktop/Aqua/AquaPose/unet/run2/best_model.pth"
)
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
    parser.add_argument(
        "--method",
        choices=["triangulation", "curve"],
        default="triangulation",
        help="Reconstruction method: triangulation (current) or curve (new optimizer)",
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
    import scipy.interpolate
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
    # Stage 5: Reconstruction (triangulation or curve optimizer)
    # -----------------------------------------------------------------------
    print(f"  Stage 5: Reconstruction ({args.method})...")
    midlines_3d: list[dict[int, object]] = []

    if args.method == "curve":
        from aquapose.reconstruction.curve_optimizer import (
            CurveOptimizer,
            CurveOptimizerConfig,
        )

        optimizer = CurveOptimizer(config=CurveOptimizerConfig(max_depth=2.0))
        for frame_idx, midline_set in enumerate(midline_sets):
            results = optimizer.optimize_midlines(
                midline_set, models, frame_index=frame_idx
            )
            midlines_3d.append(results)
    else:
        from aquapose.reconstruction.triangulation import triangulate_midlines

        for frame_idx, midline_set in enumerate(midline_sets):
            frame_results = triangulate_midlines(
                midline_set, models, frame_index=frame_idx, max_depth=0.5
            )
            midlines_3d.append(frame_results)

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
            spline = scipy.interpolate.BSpline(
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
            f"Camera: {cam_id}  Frame: {target}  Method: {args.method}",
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
