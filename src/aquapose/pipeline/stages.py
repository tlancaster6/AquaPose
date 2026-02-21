"""Per-stage batch functions for the 5-stage fish reconstruction pipeline.

Each function is independently callable with typed inputs and outputs,
enabling modular testing and incremental pipeline execution.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from aquapose.segmentation.detector import Detection, make_detector

if TYPE_CHECKING:
    from aquapose.calibration.projection import RefractiveProjectionModel
    from aquapose.reconstruction.midline import MidlineExtractor
    from aquapose.reconstruction.triangulation import Midline3D, MidlineSet
    from aquapose.segmentation.crop import CropRegion
    from aquapose.segmentation.model import UNetSegmentor
    from aquapose.tracking.tracker import FishTrack, FishTracker

logger = logging.getLogger(__name__)


def run_detection(
    video_paths: dict[str, Path],
    stop_frame: int | None = None,
    detector_kind: str = "mog2",
    **detector_kwargs: object,
) -> list[dict[str, list[Detection]]]:
    """Detect fish in all cameras across all frames.

    Creates one detector per camera and reads all videos in lockstep across
    cameras, processing each frame and collecting detections.

    Args:
        video_paths: Mapping from camera_id to video file path.
        stop_frame: If provided, stop after this many frames (exclusive).
            If None, process the entire video.
        detector_kind: Detector type — ``"mog2"`` or ``"yolo"``.
        **detector_kwargs: Forwarded to the detector constructor.

    Returns:
        List indexed by frame_idx. Each entry is a dict mapping camera_id
        to list of :class:`~aquapose.segmentation.detector.Detection`.
    """
    t0 = time.perf_counter()
    camera_ids = list(video_paths.keys())

    # Create one detector per camera
    detectors = {
        cam: make_detector(detector_kind, **detector_kwargs) for cam in camera_ids
    }

    # Open all video captures
    captures: dict[str, cv2.VideoCapture] = {}
    try:
        for cam, path in video_paths.items():
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            captures[cam] = cap

        detections_per_frame: list[dict[str, list[Detection]]] = []
        frame_idx = 0

        while True:
            if stop_frame is not None and frame_idx >= stop_frame:
                break

            frame_dets: dict[str, list[Detection]] = {}
            all_eof = True

            for cam in camera_ids:
                ret, frame = captures[cam].read()
                if not ret:
                    frame_dets[cam] = []
                else:
                    all_eof = False
                    frame_dets[cam] = detectors[cam].detect(frame)

            if all_eof:
                break

            detections_per_frame.append(frame_dets)
            frame_idx += 1

    finally:
        for cap in captures.values():
            cap.release()

    elapsed = time.perf_counter() - t0
    logger.info(
        "run_detection: %d frames, %d cameras, %.2fs",
        len(detections_per_frame),
        len(camera_ids),
        elapsed,
    )
    return detections_per_frame


def run_segmentation(
    detections_per_frame: list[dict[str, list[Detection]]],
    video_paths: dict[str, Path],
    segmentor: UNetSegmentor,
    stop_frame: int | None = None,
) -> list[dict[str, list[tuple[np.ndarray, CropRegion]]]]:
    """Segment detected fish crops across all frames.

    Re-reads video frames (detection stage does not store raw frames to save
    memory), crops each detection bounding box, and runs the segmentor on each
    crop to produce binary masks in crop-space.

    Args:
        detections_per_frame: Output of :func:`run_detection`.
        video_paths: Mapping from camera_id to video file path.
        segmentor: Instantiated :class:`~aquapose.segmentation.model.UNetSegmentor`.
        stop_frame: If provided, stop after this many frames (exclusive).

    Returns:
        List indexed by frame_idx. Each entry is a dict mapping camera_id to
        a list of ``(mask_uint8, CropRegion)`` tuples, one per detection.
        ``mask_uint8`` is a uint8 array with values 0 or 255 in crop-space.
    """
    from aquapose.segmentation.crop import compute_crop_region, extract_crop

    t0 = time.perf_counter()
    camera_ids = list(video_paths.keys())
    n_frames = (
        min(len(detections_per_frame), stop_frame)
        if stop_frame is not None
        else len(detections_per_frame)
    )

    # Open all video captures
    captures: dict[str, cv2.VideoCapture] = {}
    try:
        for cam, path in video_paths.items():
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            captures[cam] = cap

        masks_per_frame: list[dict[str, list[tuple[np.ndarray, CropRegion]]]] = []

        for frame_idx in range(n_frames):
            frame_dets = detections_per_frame[frame_idx]
            frame_masks: dict[str, list[tuple[np.ndarray, CropRegion]]] = {}

            for cam in camera_ids:
                ret, frame = captures[cam].read()
                if not ret:
                    frame_masks[cam] = []
                    continue

                cam_dets = frame_dets.get(cam, [])
                if not cam_dets:
                    frame_masks[cam] = []
                    continue

                h_frame, w_frame = frame.shape[:2]

                # Build crops for all detections in this camera
                crops: list[np.ndarray] = []
                crop_regions: list[CropRegion] = []

                for det in cam_dets:
                    region = compute_crop_region(
                        bbox=det.bbox,
                        frame_h=h_frame,
                        frame_w=w_frame,
                    )
                    crop = extract_crop(frame, region)
                    crops.append(crop)
                    crop_regions.append(region)

                # Run segmentor on all crops at once
                results = segmentor.segment(crops, crop_regions)

                # Extract (mask, crop_region) per detection
                cam_masks: list[tuple[np.ndarray, CropRegion]] = []
                for seg_results, region in zip(results, crop_regions, strict=True):
                    if seg_results:
                        cam_masks.append((seg_results[0].mask, region))
                    else:
                        # Fallback: empty mask
                        empty = np.zeros((region.height, region.width), dtype=np.uint8)
                        cam_masks.append((empty, region))

                frame_masks[cam] = cam_masks

            masks_per_frame.append(frame_masks)

    finally:
        for cap in captures.values():
            cap.release()

    elapsed = time.perf_counter() - t0
    logger.info(
        "run_segmentation: %d frames, %d cameras, %.2fs",
        len(masks_per_frame),
        len(camera_ids),
        elapsed,
    )
    return masks_per_frame


def run_tracking(
    detections_per_frame: list[dict[str, list[Detection]]],
    models: dict[str, RefractiveProjectionModel],
    tracker: FishTracker,
) -> list[list[FishTrack]]:
    """Track fish across all frames using the provided stateful FishTracker.

    The tracker MUST be externally instantiated and passed in. This function
    does NOT create a new tracker, preserving caller-controlled state.

    Args:
        detections_per_frame: Output of :func:`run_detection`.
        models: Per-camera :class:`~aquapose.calibration.projection.RefractiveProjectionModel`.
        tracker: An externally-created :class:`~aquapose.tracking.tracker.FishTracker`
            instance. State is mutated in place.

    Returns:
        List indexed by frame_idx of confirmed :class:`~aquapose.tracking.tracker.FishTrack`
        objects from each frame.
    """
    t0 = time.perf_counter()
    tracks_per_frame: list[list[FishTrack]] = []

    for frame_idx, frame_dets in enumerate(detections_per_frame):
        confirmed = tracker.update(frame_dets, models, frame_index=frame_idx)
        tracks_per_frame.append(confirmed)

    elapsed = time.perf_counter() - t0
    logger.info(
        "run_tracking: %d frames, tracker.frame_count=%d, %.2fs",
        len(tracks_per_frame),
        tracker.frame_count,
        elapsed,
    )
    return tracks_per_frame


def run_midline_extraction(
    tracks_per_frame: list[list[FishTrack]],
    masks_per_frame: list[dict[str, list[tuple[np.ndarray, CropRegion]]]],
    detections_per_frame: list[dict[str, list[Detection]]],
    models: dict[str, RefractiveProjectionModel],
    extractor: MidlineExtractor,
) -> list[MidlineSet]:
    """Extract 2D midlines for all tracked fish across all frames.

    The extractor MUST be externally instantiated and passed in. This function
    does NOT create a new extractor, preserving caller-controlled orientation
    state and back-correction buffers.

    Args:
        tracks_per_frame: Output of :func:`run_tracking`.
        masks_per_frame: Output of :func:`run_segmentation`.
        detections_per_frame: Output of :func:`run_detection`.
        models: Per-camera :class:`~aquapose.calibration.projection.RefractiveProjectionModel`.
        extractor: An externally-created :class:`~aquapose.reconstruction.midline.MidlineExtractor`
            instance. State is mutated in place.

    Returns:
        List indexed by frame_idx of :data:`~aquapose.reconstruction.triangulation.MidlineSet`
        (``dict[int, dict[str, Midline2D]]``).
    """
    t0 = time.perf_counter()
    midline_sets: list[MidlineSet] = []

    for frame_idx, (tracks, frame_masks, frame_dets) in enumerate(
        zip(tracks_per_frame, masks_per_frame, detections_per_frame, strict=True)
    ):
        # Build per-camera masks and crop_regions from the stored tuples
        masks_per_camera: dict[str, list[np.ndarray]] = {}
        crop_regions_per_camera: dict[str, list[CropRegion]] = {}

        for cam, mask_crop_list in frame_masks.items():
            masks_per_camera[cam] = [mc[0] for mc in mask_crop_list]
            crop_regions_per_camera[cam] = [mc[1] for mc in mask_crop_list]

        midline_set = extractor.extract_midlines(
            tracks=tracks,
            masks_per_camera=masks_per_camera,
            crop_regions_per_camera=crop_regions_per_camera,
            detections_per_camera=frame_dets,
            projection_models=models,
            frame_index=frame_idx,
        )
        midline_sets.append(midline_set)

    elapsed = time.perf_counter() - t0
    logger.info("run_midline_extraction: %d frames, %.2fs", len(midline_sets), elapsed)
    return midline_sets


def run_triangulation(
    midline_sets: list[MidlineSet],
    models: dict[str, RefractiveProjectionModel],
) -> list[dict[int, Midline3D]]:
    """Triangulate 3D midlines from 2D midline sets across all frames.

    Args:
        midline_sets: Output of :func:`run_midline_extraction`. One
            :data:`~aquapose.reconstruction.triangulation.MidlineSet` per frame.
        models: Per-camera :class:`~aquapose.calibration.projection.RefractiveProjectionModel`.

    Returns:
        List indexed by frame_idx of dicts mapping fish_id to
        :class:`~aquapose.reconstruction.triangulation.Midline3D`.
    """
    from aquapose.reconstruction.triangulation import triangulate_midlines

    t0 = time.perf_counter()
    results_per_frame: list[dict[int, Midline3D]] = []

    for frame_idx, midline_set in enumerate(midline_sets):
        frame_result = triangulate_midlines(midline_set, models, frame_index=frame_idx)
        results_per_frame.append(frame_result)

    elapsed = time.perf_counter() - t0
    logger.info("run_triangulation: %d frames, %.2fs", len(results_per_frame), elapsed)
    return results_per_frame
