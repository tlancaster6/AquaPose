"""Generate frozen golden reference outputs using the v2.0 PosePipeline.

Runs PosePipeline on a fixed clip with a fixed random seed, saving each
stage's output as a ``.pt`` file in ``tests/golden/``.

This script was updated in Phase 16 to use the new PosePipeline engine
(build_stages + PosePipeline.run()) instead of the v1.0 stage functions.
The CLI interface and output file names are unchanged.

These fixture files serve as the regression baseline for Phase 16
verification — any ported stage must produce numerically equivalent results.

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
import gzip
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


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Generate frozen golden reference outputs using the v2.0 PosePipeline.",
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
    """Run golden data generation using the v2.0 PosePipeline.

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

    # --- Import pipeline modules (after seed setup) ---
    from aquapose.engine.config import load_config
    from aquapose.engine.pipeline import PosePipeline, build_stages

    # --- Build PipelineConfig from CLI args ---
    overrides: dict[str, object] = {
        "video_dir": str(args.video_dir),
        "calibration_path": str(args.calibration),
        "detection.detector_kind": args.detector_kind,
        "detection.stop_frame": args.stop_frame,
        "tracking.max_fish": args.max_fish,
    }
    if args.yolo_weights is not None:
        overrides["detection.model_path"] = str(args.yolo_weights)
    if args.unet_weights is not None:
        overrides["midline.weights_path"] = str(args.unet_weights)

    config = load_config(cli_overrides=overrides, run_id="golden_data_generation")

    # --- Build and run PosePipeline ---
    logger.info(
        "Running PosePipeline (stop_frame=%d, detector=%s, seed=%d)",
        args.stop_frame,
        args.detector_kind,
        args.seed,
    )
    t0_total = time.perf_counter()
    stages = build_stages(config)
    pipeline = PosePipeline(stages=stages, config=config)
    context = pipeline.run()
    total_elapsed = time.perf_counter() - t0_total

    logger.info(
        "PosePipeline complete: %d frames, %d cameras, %.2fs total",
        context.frame_count,
        len(context.camera_ids or []),
        total_elapsed,
    )

    # ===================================================================
    # Extract and save per-stage golden outputs
    # ===================================================================

    # Stage 1: Detection
    det_path = args.output_dir / "golden_detection.pt"
    torch.save(context.detections, det_path)
    logger.info("Saved golden_detection.pt (%d bytes)", det_path.stat().st_size)

    # Stage 2: Segmentation (masks extracted from annotated_detections)
    # annotated_detections stores AnnotatedDetection objects with .mask and .crop_region
    # Reformat to legacy list[dict[str, list[tuple[ndarray, CropRegion]]]] format
    masks_per_frame = []
    for frame in context.annotated_detections or []:
        frame_masks: dict[str, list] = {}
        for cam_id, det_list in frame.items():
            cam_masks = []
            for det in det_list:
                mask = getattr(det, "mask", None)
                crop_region = getattr(det, "crop_region", None)
                if mask is not None and crop_region is not None:
                    cam_masks.append((mask, crop_region))
            frame_masks[cam_id] = cam_masks
        masks_per_frame.append(frame_masks)

    seg_path = args.output_dir / "golden_segmentation.pt.gz"
    with gzip.open(seg_path, "wb", compresslevel=6) as f:
        torch.save(masks_per_frame, f)
    logger.info("Saved golden_segmentation.pt.gz (%d bytes)", seg_path.stat().st_size)

    # Stage 4: Tracking
    trk_path = args.output_dir / "golden_tracking.pt"
    torch.save(context.tracks, trk_path)
    logger.info("Saved golden_tracking.pt (%d bytes)", trk_path.stat().st_size)

    # Stage 2 (midlines): Extract Midline2D from annotated_detections
    # Legacy format: list[dict[int, dict[str, Midline2D]]] (MidlineSet per frame)
    # New pipeline produces midlines per detection (before tracking) so fish_id
    # is not available at Stage 2. We use detection index as a proxy fish_id.
    midline_sets = []
    for frame in context.annotated_detections or []:
        frame_midlines: dict[int, dict[str, object]] = {}
        det_idx = 0
        for cam_id, det_list in frame.items():
            for det in det_list:
                midline = getattr(det, "midline", None)
                if midline is not None:
                    fish_proxy_id = det_idx
                    if fish_proxy_id not in frame_midlines:
                        frame_midlines[fish_proxy_id] = {}
                    frame_midlines[fish_proxy_id][cam_id] = midline
                det_idx += 1
        midline_sets.append(frame_midlines)

    mid_path = args.output_dir / "golden_midline_extraction.pt"
    torch.save(midline_sets, mid_path)
    logger.info(
        "Saved golden_midline_extraction.pt (%d bytes)", mid_path.stat().st_size
    )

    # Stage 5: Triangulation (Reconstruction)
    tri_path = args.output_dir / "golden_triangulation.pt"
    torch.save(context.midlines_3d, tri_path)
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
        "camera_ids": context.camera_ids,
        "frame_count": context.frame_count,
        "generation_timestamp": datetime.now(UTC).isoformat(),
    }

    meta_path = args.output_dir / "metadata.pt"
    torch.save(metadata, meta_path)
    logger.info("Saved metadata.pt (%d bytes)", meta_path.stat().st_size)

    # ===================================================================
    # Summary
    # ===================================================================
    all_paths = [det_path, seg_path, trk_path, mid_path, tri_path, meta_path]
    total_bytes = sum(p.stat().st_size for p in all_paths)
    logger.info(
        "Golden data generation complete. Total size: %.1f MB. Files in: %s",
        total_bytes / (1024 * 1024),
        args.output_dir.resolve(),
    )
    logger.info(
        "Commit with: git add tests/golden/ && git commit -m "
        '"data(16): commit golden reference outputs from v2.0 PosePipeline"'
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
