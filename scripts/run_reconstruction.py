"""End-to-end reconstruction CLI: detect fish, initialize 3D state, optimize, validate holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add project root to path so the script can be run directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aquapose.calibration.loader import load_calibration_data
from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.initialization import init_fish_states_from_masks
from aquapose.optimization import (
    FishOptimizer,
    RefractiveSilhouetteRenderer,
    compute_angular_diversity_weights,
    render_overlay,
    run_holdout_validation,
)
from aquapose.segmentation.detector import make_detector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end per-fish 3D reconstruction with cross-view holdout validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Path to directory containing per-camera video files.",
    )
    parser.add_argument(
        "--calibration-json",
        required=True,
        help="Path to AquaCal calibration JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/reconstruction",
        help="Directory to write results (overlays, metrics).",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=100,
        help="Number of video frames to process.",
    )
    parser.add_argument(
        "--detector",
        choices=["yolo", "mog2"],
        default="yolo",
        help="Fish detection backend.",
    )
    parser.add_argument(
        "--yolo-weights",
        default=None,
        help="Path to YOLO weights file (required when --detector=yolo).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1e-4,
        help="Silhouette renderer sigma (edge softness).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam optimizer learning rate.",
    )
    parser.add_argument(
        "--max-iters-first",
        type=int,
        default=300,
        help="Max optimization iterations for the first frame.",
    )
    parser.add_argument(
        "--max-iters-warmstart",
        type=int,
        default=50,
        help="Max optimization iterations for subsequent frames.",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        nargs=2,
        default=[240, 320],
        metavar=("H", "W"),
        help="Renderer output resolution (H W).",
    )
    return parser.parse_args()


def _build_projection_model(cam_data, calib: any) -> RefractiveProjectionModel:
    """Build a RefractiveProjectionModel from CalibrationData for one camera."""
    return RefractiveProjectionModel(
        K=cam_data.K,
        R=cam_data.R,
        t=cam_data.t,
        water_z=calib.water_z,
        normal=calib.interface_normal,
        n_air=calib.n_air,
        n_water=calib.n_water,
    )


def _open_video_captures(
    video_dir: Path, camera_ids: list[str]
) -> dict[str, cv2.VideoCapture]:
    """Open VideoCapture for each camera by matching files in video_dir.

    Looks for files whose stem matches (case-insensitive) or contains the camera ID.

    Args:
        video_dir: Directory containing video files.
        camera_ids: List of camera identifiers to match.

    Returns:
        Dict mapping camera_id -> VideoCapture (only cameras that have a matching file).
    """
    captures: dict[str, cv2.VideoCapture] = {}
    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".h264", ".hevc"}

    video_files = [
        p for p in video_dir.iterdir() if p.suffix.lower() in video_extensions
    ]

    for cam_id in camera_ids:
        # Try exact stem match first, then substring match.
        matched = None
        for vf in video_files:
            if vf.stem.lower() == cam_id.lower() or cam_id.lower() in vf.stem.lower():
                matched = vf
                break

        if matched is None:
            print(f"  [WARN] No video found for camera '{cam_id}' in {video_dir}")
            continue

        cap = cv2.VideoCapture(str(matched))
        if not cap.isOpened():
            print(f"  [WARN] Cannot open video '{matched}' for camera '{cam_id}'")
            continue

        captures[cam_id] = cap
        print(f"  Opened: {matched.name} -> {cam_id}")

    return captures


def _read_frame(
    captures: dict[str, cv2.VideoCapture], camera_ids: list[str]
) -> dict[str, np.ndarray | None]:
    """Read one frame from each VideoCapture.

    Args:
        captures: Dict camera_id -> VideoCapture.
        camera_ids: Ordered list of camera IDs.

    Returns:
        Dict camera_id -> BGR frame (or None if read failed / camera missing).
    """
    frames = {}
    for cam_id in camera_ids:
        cap = captures.get(cam_id)
        if cap is None:
            frames[cam_id] = None
            continue
        ret, frame = cap.read()
        frames[cam_id] = frame if ret else None
    return frames


def _detect_masks(
    frames: dict[str, np.ndarray | None],
    camera_ids: list[str],
    detectors: dict[str, any],
) -> dict[str, np.ndarray | None]:
    """Run fish detection on each camera frame.

    Args:
        frames: Dict camera_id -> BGR frame (or None).
        camera_ids: Ordered camera IDs.
        detectors: Dict camera_id -> detector instance.

    Returns:
        Dict camera_id -> binary mask (H, W, uint8) or None if no detection.
    """
    masks: dict[str, np.ndarray | None] = {}
    for cam_id in camera_ids:
        frame = frames.get(cam_id)
        detector = detectors.get(cam_id)
        if frame is None or detector is None:
            masks[cam_id] = None
            continue
        detections = detector.detect(frame)
        if detections:
            # Use the first (highest confidence) detection mask.
            masks[cam_id] = detections[0].mask.astype(np.uint8)
        else:
            masks[cam_id] = None
    return masks


def _mask_to_tensor(mask: np.ndarray | None, H: int, W: int) -> torch.Tensor:
    """Convert a detection mask to a float32 tensor at render resolution.

    Args:
        mask: Binary mask (h, w) uint8 or None.
        H: Target height.
        W: Target width.

    Returns:
        Float32 tensor, shape (H, W), values in {0, 1}.
    """
    if mask is None:
        return torch.zeros(H, W)
    resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(resized.astype(np.float32) / 255.0)


def main() -> None:
    """Run end-to-end reconstruction on real video data."""
    args = _parse_args()

    # -----------------------------------------------------------------------
    # Setup output directories
    # -----------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    render_H, render_W = args.render_size

    # -----------------------------------------------------------------------
    # Load calibration
    # -----------------------------------------------------------------------
    print(f"\nLoading calibration from: {args.calibration_json}")
    calib = load_calibration_data(args.calibration_json)

    # Use ring cameras (exclude auxiliary center camera for optimization).
    camera_ids = calib.ring_cameras
    print(f"  Using {len(camera_ids)} ring cameras: {camera_ids}")

    # Build projection models.
    cameras = [_build_projection_model(calib.cameras[cid], calib) for cid in camera_ids]

    # Build renderer.
    renderer = RefractiveSilhouetteRenderer(
        image_size=(render_H, render_W),
        sigma=args.sigma,
    )

    # Compute angular diversity weights.
    camera_weights = compute_angular_diversity_weights(cameras, camera_ids)

    # Build optimizer.
    loss_weights = {"iou": 1.0, "gravity": 0.05, "morph": 0.2}
    optimizer = FishOptimizer(
        renderer=renderer,
        loss_weights=loss_weights,
        camera_weights=camera_weights,
        lr=args.lr,
        max_iters_first=args.max_iters_first,
        max_iters_warmstart=args.max_iters_warmstart,
    )

    # -----------------------------------------------------------------------
    # Open video captures
    # -----------------------------------------------------------------------
    video_dir = Path(args.video_dir)
    print(f"\nOpening videos from: {video_dir}")
    captures = _open_video_captures(video_dir, camera_ids)

    if not captures:
        print("ERROR: No video files found. Check --video-dir.")
        sys.exit(1)

    active_camera_ids = [cid for cid in camera_ids if cid in captures]
    active_cameras = [cameras[camera_ids.index(cid)] for cid in active_camera_ids]

    # -----------------------------------------------------------------------
    # Build detectors per camera
    # -----------------------------------------------------------------------
    print(f"\nBuilding detectors (--detector={args.detector})")
    detectors: dict[str, any] = {}
    for cam_id in active_camera_ids:
        if args.detector == "yolo":
            if args.yolo_weights is None:
                print("ERROR: --yolo-weights required when --detector=yolo")
                sys.exit(1)
            detectors[cam_id] = make_detector("yolo", model_path=args.yolo_weights)
        else:
            detectors[cam_id] = make_detector("mog2")

    # -----------------------------------------------------------------------
    # Process frames
    # -----------------------------------------------------------------------
    print(f"\nProcessing {args.n_frames} frames...")
    frames_data: list[dict] = []
    all_masks_per_frame: list[dict[str, np.ndarray | None]] = []
    raw_frames_per_camera: dict[str, list[np.ndarray | None]] = {
        cid: [] for cid in active_camera_ids
    }

    for frame_idx in range(args.n_frames):
        raw_frames = _read_frame(captures, active_camera_ids)
        masks_np = _detect_masks(raw_frames, active_camera_ids, detectors)

        target_masks = {
            cid: _mask_to_tensor(masks_np.get(cid), render_H, render_W)
            for cid in active_camera_ids
        }
        crop_regions: dict[str, tuple[int, int, int, int] | None] = {
            cid: None for cid in active_camera_ids
        }

        frames_data.append(
            {
                "target_masks": target_masks,
                "crop_regions": crop_regions,
            }
        )
        all_masks_per_frame.append(masks_np)

        for cid in active_camera_ids:
            raw_frames_per_camera[cid].append(raw_frames.get(cid))

        if (frame_idx + 1) % 10 == 0:
            n_detected = sum(
                1 for cid in active_camera_ids if masks_np.get(cid) is not None
            )
            print(
                f"  Frame {frame_idx + 1}/{args.n_frames}: {n_detected}/{len(active_camera_ids)} cameras have detections"
            )

    # Close video captures.
    for cap in captures.values():
        cap.release()

    # -----------------------------------------------------------------------
    # Cold-start initialization from first frame
    # -----------------------------------------------------------------------
    print("\nInitializing fish state from first frame masks...")
    # init_fish_states_from_masks expects: masks_per_camera[camera_idx][fish_idx]
    masks_per_camera = [[all_masks_per_frame[0].get(cid)] for cid in active_camera_ids]

    try:
        init_states = init_fish_states_from_masks(masks_per_camera, active_cameras)
        if init_states:
            init_state = init_states[0]
            print(
                f"  Initialized state: p={init_state.p.tolist()}, s={init_state.s.item():.3f}"
            )
        else:
            print("  WARNING: No fish detected in first frame — using default init.")
            from aquapose.mesh.state import FishState

            init_state = FishState(
                p=torch.tensor([0.0, 0.0, 1.5]),
                psi=torch.tensor(0.0),
                theta=torch.tensor(0.0),
                kappa=torch.tensor(0.0),
                s=torch.tensor(0.15),
            )
    except Exception as e:
        print(f"  WARNING: Initialization failed ({e}) — using default init.")
        from aquapose.mesh.state import FishState

        init_state = FishState(
            p=torch.tensor([0.0, 0.0, 1.5]),
            psi=torch.tensor(0.0),
            theta=torch.tensor(0.0),
            kappa=torch.tensor(0.0),
            s=torch.tensor(0.15),
        )

    # -----------------------------------------------------------------------
    # Optimize sequence
    # -----------------------------------------------------------------------
    print(
        f"\nOptimizing {args.n_frames} frames (first-frame: {args.max_iters_first} iters, warm-start: {args.max_iters_warmstart} iters)..."
    )
    optimized_states = optimizer.optimize_sequence(
        init_state, frames_data, active_cameras, active_camera_ids
    )
    print(f"  Optimized {len(optimized_states)} frames.")

    # -----------------------------------------------------------------------
    # Cross-view holdout validation
    # -----------------------------------------------------------------------
    print("\nRunning cross-view holdout validation...")
    metrics = run_holdout_validation(
        optimized_states,
        frames_data,
        active_cameras,
        active_camera_ids,
        renderer,
        optimizer,
    )

    # -----------------------------------------------------------------------
    # Save overlay images
    # -----------------------------------------------------------------------
    print(f"\nSaving visual overlays to: {overlay_dir}")
    n_overlays = min(args.n_frames, 10)  # Save first 10 frames to avoid disk bloat.
    for frame_idx in range(n_overlays):
        state = optimized_states[frame_idx]
        for cam_id in active_camera_ids:
            raw_frame = raw_frames_per_camera[cam_id][frame_idx]
            if raw_frame is None:
                continue

            # Render mesh alpha into this camera.
            with torch.no_grad():
                from aquapose.mesh.builder import build_fish_mesh

                meshes = build_fish_mesh([state])
                cam_model = active_cameras[active_camera_ids.index(cam_id)]
                alpha_maps = renderer.render(meshes, [cam_model], [cam_id])
                alpha = alpha_maps[cam_id].numpy()

            # Resize alpha to match raw frame resolution.
            fh, fw = raw_frame.shape[:2]
            alpha_resized = cv2.resize(alpha, (fw, fh), interpolation=cv2.INTER_LINEAR)

            overlay = render_overlay(
                raw_frame, alpha_resized, color=(0, 255, 0), opacity=0.4
            )

            filename = overlay_dir / f"frame{frame_idx:04d}_{cam_id}.jpg"
            cv2.imwrite(str(filename), overlay)

    print(f"  Saved overlays for {n_overlays} frames.")

    # -----------------------------------------------------------------------
    # Save metrics JSON
    # -----------------------------------------------------------------------
    metrics_path = output_dir / "metrics.json"
    metrics_serializable = {
        "global_mean_iou": metrics["global_mean_iou"],
        "min_camera_iou": metrics["min_camera_iou"],
        "per_camera_iou": metrics["per_camera_iou"],
        "target_met_080": metrics["target_met_080"],
        "target_met_060_floor": metrics["target_met_060_floor"],
        "n_frames_processed": args.n_frames,
        "n_cameras": len(active_camera_ids),
        "camera_ids": active_camera_ids,
        "pass_fail": "PASS"
        if (metrics["target_met_080"] and metrics["target_met_060_floor"])
        else "FAIL",
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"\n{'=' * 50}")
    print(
        f"RESULT: {'PASS' if metrics_serializable['pass_fail'] == 'PASS' else 'FAIL'}"
    )
    print(
        f"  Global mean holdout IoU: {metrics['global_mean_iou']:.4f}  (target: >= 0.80)"
    )
    print(
        f"  Min camera holdout IoU:  {metrics['min_camera_iou']:.4f}  (floor:  >= 0.60)"
    )
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
