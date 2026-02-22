"""Pipeline orchestrator for the 5-stage fish 3D reconstruction pipeline.

Provides the :func:`reconstruct` entry point that chains detection,
segmentation, tracking, midline extraction, and triangulation into a single
callable API with HDF5 output.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aquapose.reconstruction.triangulation import Midline3D

logger = logging.getLogger(__name__)

# Camera to exclude (centre top-down camera — poor mask quality)
_SKIP_CAMERA_ID = "e3v8250"


@dataclass
class ReconstructResult:
    """Result of a :func:`reconstruct` run.

    Attributes:
        output_dir: Directory where HDF5 output was written.
        midlines_3d: Per-frame triangulation results. Indexed by frame_idx,
            each entry maps fish_id to :class:`~aquapose.reconstruction.triangulation.Midline3D`.
        stage_timing: Wall-clock seconds spent in each stage, keyed by stage name.
    """

    output_dir: Path
    midlines_3d: list[dict[int, Midline3D]]
    stage_timing: dict[str, float] = field(default_factory=dict)


def reconstruct(
    video_dir: Path,
    calibration_path: Path,
    output_dir: Path,
    *,
    stop_frame: int | None = None,
    detector_kind: str = "yolo",
    unet_weights: Path | None = None,
    max_fish: int = 9,
    **detector_kwargs: object,
) -> ReconstructResult:
    """Run the full 5-stage fish 3D reconstruction pipeline.

    Discovers camera videos, loads calibration, creates stateful objects
    once, chains all five stages, writes HDF5 output, and returns results
    with timing information.

    Stateful objects (:class:`~aquapose.tracking.tracker.FishTracker` and
    :class:`~aquapose.reconstruction.midline.MidlineExtractor`) are
    instantiated once and persist across all frames.

    Args:
        video_dir: Directory containing per-camera ``.avi`` or ``.mp4`` files.
            Camera ID is inferred from the filename stem.
        calibration_path: Path to the AquaCal calibration JSON file.
        output_dir: Directory to write ``midlines_3d.h5``.
            Created if it does not exist.
        stop_frame: If provided, process only the first ``stop_frame`` frames.
        detector_kind: Detector type — ``"yolo"`` (default) or ``"mog2"``.
        unet_weights: Optional path to U-Net weights ``.pth`` file. If None,
            uses ImageNet-pretrained backbone (for testing only).
        max_fish: Maximum number of fish slots in HDF5 output and tracker
            population constraint.
        **detector_kwargs: Additional kwargs forwarded to the detector
            constructor (e.g. ``model_path`` for YOLO).

    Returns:
        :class:`ReconstructResult` with midlines, output_dir, and stage timing.

    Raises:
        FileNotFoundError: If ``video_dir`` or ``calibration_path`` do not exist.
        ValueError: If no valid camera videos are found.
    """
    from aquapose.calibration.loader import (
        compute_undistortion_maps,
        load_calibration_data,
    )
    from aquapose.calibration.projection import RefractiveProjectionModel
    from aquapose.io.midline_writer import Midline3DWriter
    from aquapose.io.video import VideoSet
    from aquapose.reconstruction.midline import MidlineExtractor
    from aquapose.segmentation.model import UNetSegmentor
    from aquapose.tracking.tracker import FishTracker

    from .stages import (
        run_detection,
        run_midline_extraction,
        run_segmentation,
        run_tracking,
        run_triangulation,
    )

    video_dir = Path(video_dir)
    calibration_path = Path(calibration_path)
    output_dir = Path(output_dir)

    if not video_dir.exists():
        raise FileNotFoundError(f"video_dir does not exist: {video_dir}")
    if not calibration_path.exists():
        raise FileNotFoundError(f"calibration_path does not exist: {calibration_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Discover camera videos ---
    video_paths: dict[str, Path] = {}
    for suffix in ("*.avi", "*.mp4"):
        for p in video_dir.glob(suffix):
            camera_id = p.stem.split("-")[0]
            if camera_id == _SKIP_CAMERA_ID:
                logger.info("Skipping excluded camera: %s", camera_id)
                continue
            video_paths[camera_id] = p

    if not video_paths:
        raise ValueError(
            f"No .avi/.mp4 files found in {video_dir} (after excluding {_SKIP_CAMERA_ID!r})"
        )

    logger.info("Found %d cameras: %s", len(video_paths), sorted(video_paths))

    # --- Load calibration + undistortion ---
    calib = load_calibration_data(calibration_path)

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
        raise ValueError("No cameras matched between video_dir and calibration.")

    # --- Create stateful objects once ---
    tracker = FishTracker(expected_count=max_fish)
    extractor = MidlineExtractor()
    segmentor = UNetSegmentor(weights_path=unet_weights)

    stage_timing: dict[str, float] = {}

    # --- Stage 1: Detection ---
    t0 = time.perf_counter()
    video_set = VideoSet(video_paths, undistortion=undist_maps)
    with video_set:
        detections_per_frame = run_detection(
            video_set=video_set,
            stop_frame=stop_frame,
            detector_kind=detector_kind,
            **detector_kwargs,
        )
    stage_timing["detection"] = time.perf_counter() - t0
    logger.info("Stage 1 (detection): %.2fs", stage_timing["detection"])

    # --- Stage 2: Segmentation ---
    t0 = time.perf_counter()
    with VideoSet(video_paths, undistortion=undist_maps) as seg_video_set:
        masks_per_frame = run_segmentation(
            detections_per_frame=detections_per_frame,
            video_set=seg_video_set,
            segmentor=segmentor,
            stop_frame=stop_frame,
        )
    stage_timing["segmentation"] = time.perf_counter() - t0
    logger.info("Stage 2 (segmentation): %.2fs", stage_timing["segmentation"])

    # --- Stage 3: Tracking ---
    t0 = time.perf_counter()
    tracks_per_frame = run_tracking(
        detections_per_frame=detections_per_frame,
        models=models,
        tracker=tracker,
    )
    stage_timing["tracking"] = time.perf_counter() - t0
    logger.info("Stage 3 (tracking): %.2fs", stage_timing["tracking"])

    # --- Stage 4: Midline Extraction ---
    t0 = time.perf_counter()
    midline_sets = run_midline_extraction(
        tracks_per_frame=tracks_per_frame,
        masks_per_frame=masks_per_frame,
        detections_per_frame=detections_per_frame,
        extractor=extractor,
    )
    stage_timing["midline_extraction"] = time.perf_counter() - t0
    logger.info(
        "Stage 4 (midline_extraction): %.2fs", stage_timing["midline_extraction"]
    )

    # --- Stage 5: Triangulation ---
    t0 = time.perf_counter()
    midlines_3d = run_triangulation(
        midline_sets=midline_sets,
        models=models,
    )
    stage_timing["triangulation"] = time.perf_counter() - t0
    logger.info("Stage 5 (triangulation): %.2fs", stage_timing["triangulation"])

    # --- Write HDF5 output ---
    t0 = time.perf_counter()
    h5_path = output_dir / "midlines_3d.h5"
    with Midline3DWriter(h5_path, max_fish=max_fish) as writer:
        for frame_idx, frame_midlines in enumerate(midlines_3d):
            writer.write_frame(frame_idx, frame_midlines)
    stage_timing["hdf5_write"] = time.perf_counter() - t0
    logger.info("HDF5 written to %s (%.2fs)", h5_path, stage_timing["hdf5_write"])

    return ReconstructResult(
        output_dir=output_dir,
        midlines_3d=midlines_3d,
        stage_timing=stage_timing,
    )
