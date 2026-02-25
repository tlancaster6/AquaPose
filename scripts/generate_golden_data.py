"""Generate frozen golden reference outputs from the v1.0 pipeline.

Runs the v1.0 pipeline stage-by-stage on a fixed clip with a fixed random seed,
saving each stage's output as a ``.pt`` file in ``tests/golden/``.

These fixture files serve as the regression baseline for Phase 15-16 stage
migrations — any ported stage must produce numerically equivalent results.

Usage::

    python scripts/generate_golden_data.py \\
        --video-dir /path/to/raw_videos \\
        --calibration /path/to/calibration.json \\
        --output-dir tests/golden/ \\
        --stop-frame 30 \\
        --detector-kind yolo \\
        --yolo-weights runs/detect/output/yolo_fish/train_v1/weights/best.pt \\
        --unet-weights /path/to/unet/best_model.pth \\
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Determinism setup — MUST happen before any other imports that touch CUDA
# ---------------------------------------------------------------------------


def _set_deterministic_seeds(seed: int) -> None:
    """Set global random seeds for reproducibility.

    Args:
        seed: Integer seed value applied to all RNG sources.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_golden_data")

# Camera to exclude (centre top-down camera — poor mask quality)
_SKIP_CAMERA_ID = "e3v8250"


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Generate frozen golden reference outputs from the v1.0 pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory containing per-camera .avi or .mp4 files.",
    )
    parser.add_argument(
        "--calibration",
        required=True,
        type=Path,
        metavar="JSON",
        help="Path to AquaCal calibration JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("tests/golden/"),
        type=Path,
        metavar="DIR",
        help="Directory to write .pt fixture files.",
    )
    parser.add_argument(
        "--stop-frame",
        default=30,
        type=int,
        metavar="N",
        help="Number of frames to process (first N frames).",
    )
    parser.add_argument(
        "--detector-kind",
        default="yolo",
        choices=["yolo", "mog2"],
        help="Detector type to use.",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        metavar="PTH",
        help="Path to YOLO weights file (required when --detector-kind=yolo).",
    )
    parser.add_argument(
        "--unet-weights",
        type=Path,
        metavar="PTH",
        help="Path to U-Net weights .pth file.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Global random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-fish",
        default=9,
        type=int,
        help="Maximum number of fish slots for the tracker.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run golden data generation.

    Args:
        argv: Optional list of CLI argument strings. If None, uses sys.argv.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # --- Determinism setup (before any pipeline imports) ---
    _set_deterministic_seeds(args.seed)
    logger.info("Seeds set: seed=%d", args.seed)

    # --- Validate arguments ---
    if args.detector_kind == "yolo" and args.yolo_weights is None:
        parser.error("--yolo-weights is required when --detector-kind=yolo")

    if not args.video_dir.exists():
        logger.error("video-dir does not exist: %s", args.video_dir)
        return 1
    if not args.calibration.exists():
        logger.error("calibration does not exist: %s", args.calibration)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", args.output_dir.resolve())

    # --- Now import pipeline modules (after seed setup) ---
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
        run_tracking,
        run_triangulation,
    )
    from aquapose.reconstruction.midline import MidlineExtractor
    from aquapose.segmentation.model import UNetSegmentor
    from aquapose.tracking.tracker import FishTracker

    # --- Discover camera videos (same logic as orchestrator.py) ---
    video_paths: dict[str, Path] = {}
    for suffix in ("*.avi", "*.mp4"):
        for p in args.video_dir.glob(suffix):
            camera_id = p.stem.split("-")[0]
            if camera_id == _SKIP_CAMERA_ID:
                logger.info("Skipping excluded camera: %s", camera_id)
                continue
            video_paths[camera_id] = p

    if not video_paths:
        logger.error(
            "No .avi/.mp4 files found in %s (after excluding %r)",
            args.video_dir,
            _SKIP_CAMERA_ID,
        )
        return 1

    camera_ids = sorted(video_paths)
    logger.info("Found %d cameras: %s", len(camera_ids), camera_ids)

    # --- Load calibration + undistortion maps ---
    logger.info("Loading calibration from %s", args.calibration)
    calib = load_calibration_data(args.calibration)

    undist_maps: dict[str, object] = {}
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
        logger.error("No cameras matched between video_dir and calibration.")
        return 1

    # --- Create stateful objects (once, same as orchestrator.py) ---
    tracker = FishTracker(expected_count=args.max_fish)
    extractor = MidlineExtractor()
    segmentor = UNetSegmentor(weights_path=args.unet_weights)

    # --- Build detector kwargs ---
    detector_kwargs: dict[str, object] = {}
    if args.detector_kind == "yolo" and args.yolo_weights is not None:
        detector_kwargs["model_path"] = args.yolo_weights

    # ===================================================================
    # Stage 1: Detection
    # ===================================================================
    logger.info("=== Stage 1: Detection (stop_frame=%d) ===", args.stop_frame)
    t0 = time.perf_counter()
    video_set = VideoSet(video_paths, undistortion=undist_maps)
    with video_set:
        detections_per_frame = run_detection(
            video_set=video_set,
            stop_frame=args.stop_frame,
            detector_kind=args.detector_kind,
            **detector_kwargs,
        )
    elapsed = time.perf_counter() - t0
    logger.info(
        "Stage 1 complete: %d frames, %d cameras, %.2fs",
        len(detections_per_frame),
        len(models),
        elapsed,
    )

    det_path = args.output_dir / "golden_detection.pt"
    torch.save(detections_per_frame, det_path)
    logger.info("Saved golden_detection.pt (%d bytes)", det_path.stat().st_size)

    # ===================================================================
    # Stage 2: Segmentation
    # ===================================================================
    logger.info("=== Stage 2: Segmentation ===")
    t0 = time.perf_counter()
    with VideoSet(video_paths, undistortion=undist_maps) as seg_video_set:
        masks_per_frame = run_segmentation(
            detections_per_frame=detections_per_frame,
            video_set=seg_video_set,
            segmentor=segmentor,
            stop_frame=args.stop_frame,
        )
    elapsed = time.perf_counter() - t0
    logger.info(
        "Stage 2 complete: %d frames, %d cameras, %.2fs",
        len(masks_per_frame),
        len(models),
        elapsed,
    )

    seg_path = args.output_dir / "golden_segmentation.pt"
    torch.save(masks_per_frame, seg_path)
    logger.info("Saved golden_segmentation.pt (%d bytes)", seg_path.stat().st_size)

    # ===================================================================
    # Stage 3: Tracking
    # ===================================================================
    logger.info("=== Stage 3: Tracking ===")
    t0 = time.perf_counter()
    tracks_per_frame = run_tracking(
        detections_per_frame=detections_per_frame,
        models=models,
        tracker=tracker,
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "Stage 3 complete: %d frames, %.2fs",
        len(tracks_per_frame),
        elapsed,
    )

    trk_path = args.output_dir / "golden_tracking.pt"
    torch.save(tracks_per_frame, trk_path)
    logger.info("Saved golden_tracking.pt (%d bytes)", trk_path.stat().st_size)

    # ===================================================================
    # Stage 4: Midline Extraction
    # ===================================================================
    logger.info("=== Stage 4: Midline Extraction ===")
    t0 = time.perf_counter()
    midline_sets = run_midline_extraction(
        tracks_per_frame=tracks_per_frame,
        masks_per_frame=masks_per_frame,
        detections_per_frame=detections_per_frame,
        extractor=extractor,
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "Stage 4 complete: %d frames, %.2fs",
        len(midline_sets),
        elapsed,
    )

    mid_path = args.output_dir / "golden_midline_extraction.pt"
    torch.save(midline_sets, mid_path)
    logger.info(
        "Saved golden_midline_extraction.pt (%d bytes)", mid_path.stat().st_size
    )

    # ===================================================================
    # Stage 5: Triangulation
    # ===================================================================
    logger.info("=== Stage 5: Triangulation ===")
    t0 = time.perf_counter()
    midlines_3d = run_triangulation(
        midline_sets=midline_sets,
        models=models,
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "Stage 5 complete: %d frames, %.2fs",
        len(midlines_3d),
        elapsed,
    )

    tri_path = args.output_dir / "golden_triangulation.pt"
    torch.save(midlines_3d, tri_path)
    logger.info("Saved golden_triangulation.pt (%d bytes)", tri_path.stat().st_size)

    # ===================================================================
    # Metadata
    # ===================================================================
    cuda_version: str
    gpu_name: str
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda or "N/A"
        gpu_name = torch.cuda.get_device_name(0)
    else:
        cuda_version = "N/A"
        gpu_name = "N/A"

    metadata: dict[str, object] = {
        "seed": args.seed,
        "stop_frame": args.stop_frame,
        "detector_kind": args.detector_kind,
        "max_fish": args.max_fish,
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "numpy_version": np.__version__,
        "camera_ids": camera_ids,
        "frame_count": len(detections_per_frame),
        "generation_timestamp": datetime.now(UTC).isoformat(),
    }

    meta_path = args.output_dir / "metadata.pt"
    torch.save(metadata, meta_path)
    logger.info("Saved metadata.pt (%d bytes)", meta_path.stat().st_size)

    # ===================================================================
    # Summary
    # ===================================================================
    total_bytes = sum(
        p.stat().st_size
        for p in [det_path, seg_path, trk_path, mid_path, tri_path, meta_path]
    )
    logger.info(
        "Golden data generation complete. Total size: %.1f MB. Files in: %s",
        total_bytes / (1024 * 1024),
        args.output_dir.resolve(),
    )
    logger.info(
        "Commit with: git add tests/golden/ && git commit -m "
        '"data(14): commit golden reference outputs from v1.0 pipeline"'
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
