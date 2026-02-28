"""MidlineStage — Stage 4 of the v2.1 5-stage AquaPose pipeline.

Reads detection bounding boxes from Stage 1, crops and segments each detection
via U-Net, then extracts 15-point 2D midlines with half-widths via
skeletonization + BFS pruning. When tracklet_groups are available from
Stage 3 (Association), only processes detections belonging to confirmed
tracklet groups and applies head-tail orientation resolution.

Populates PipelineContext.annotated_detections.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aquapose.core.context import PipelineContext
from aquapose.core.midline.backends import get_backend
from aquapose.io.discovery import discover_camera_videos

__all__ = ["MidlineStage"]

logger = logging.getLogger(__name__)


class MidlineStage:
    """Stage 4: Segments detections and extracts 2D midlines, populates context.

    Runs after Association (Stage 3). For each frame, for each camera, crops
    each detection and runs the configured backend to produce binary masks and
    15-point 2D midlines. When tracklet_groups are available, only processes
    detections belonging to confirmed tracklet groups and applies head-tail
    orientation resolution using cross-camera geometry, velocity, and
    temporal prior signals.

    The backend is created eagerly at construction time. A missing weights file
    raises :class:`FileNotFoundError` immediately.

    Camera and video discovery:
    - Glob ``*.avi`` and ``*.mp4`` in *video_dir*
    - Camera ID = ``stem.split("-")[0]``
    - All cameras in the input directory are processed (no internal filtering)

    Calibration data is loaded at construction for undistortion map computation.

    Args:
        video_dir: Directory containing per-camera video files.
        calibration_path: Path to the AquaCal calibration JSON file.
        weights_path: Path to U-Net model weights file. Raises FileNotFoundError
            if the path does not exist (None uses pretrained ImageNet encoder).
        confidence_threshold: Minimum confidence for mask acceptance.
        backend: Backend kind -- "segment_then_extract" (default) or
            "direct_pose" (raises NotImplementedError).
        device: PyTorch device string (e.g. "cuda", "cpu").
        n_points: Number of midline points per detection.
        min_area: Minimum mask area (pixels) to attempt midline extraction.
        lut_config: Optional LUT configuration for ForwardLUT loading (needed
            for orientation resolution). None skips orientation resolution.
        midline_config: Optional MidlineConfig-like object with orientation
            weights (speed_threshold, orientation_weight_*). None uses defaults.

    Raises:
        FileNotFoundError: If *video_dir*, *calibration_path*, or U-Net weights
            do not exist.
        ValueError: If no valid camera videos are found.
        NotImplementedError: If *backend* is ``"direct_pose"``.

    """

    def __init__(
        self,
        video_dir: str | Path,
        calibration_path: str | Path,
        weights_path: str | None = None,
        confidence_threshold: float = 0.5,
        backend: str = "segment_then_extract",
        device: str = "cuda",
        n_points: int = 15,
        min_area: int = 300,
        lut_config: Any | None = None,
        midline_config: Any | None = None,
    ) -> None:
        from aquapose.calibration.loader import (
            compute_undistortion_maps,
            load_calibration_data,
        )

        self._video_dir = Path(video_dir)
        self._calibration_path = Path(calibration_path)
        self._lut_config = lut_config
        self._midline_config = midline_config

        # Validate paths eagerly
        if not self._video_dir.exists():
            raise FileNotFoundError(f"video_dir does not exist: {self._video_dir}")
        if not self._calibration_path.exists():
            raise FileNotFoundError(
                f"calibration_path does not exist: {self._calibration_path}",
            )

        # Discover camera videos
        video_paths = discover_camera_videos(self._video_dir)

        if not video_paths:
            raise ValueError(f"No .avi/.mp4 files found in {self._video_dir}")

        logger.info(
            "MidlineStage: found %d cameras: %s",
            len(video_paths),
            sorted(video_paths),
        )

        # Load calibration and compute undistortion maps
        calib = load_calibration_data(self._calibration_path)
        undist_maps = {}
        for cam_id in video_paths:
            if cam_id not in calib.cameras:
                logger.warning("Camera %r not in calibration; skipping", cam_id)
                continue
            undist_maps[cam_id] = compute_undistortion_maps(calib.cameras[cam_id])

        # Only keep cameras with both video and calibration
        self._video_paths: dict[str, Path] = {
            cam_id: p for cam_id, p in video_paths.items() if cam_id in undist_maps
        }
        self._undist_maps = undist_maps

        if not self._video_paths:
            raise ValueError("No cameras matched between video_dir and calibration.")

        # Eagerly create backend (fail-fast on missing weights, unsupported kind)
        self._backend = get_backend(
            backend,
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            n_points=n_points,
            min_area=min_area,
            device=device,
        )

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run midline extraction across all cameras for all frames.

        When ``context.tracklet_groups`` is available, only processes detections
        that belong to tracklet groups (i.e., confirmed fish identities). After
        extraction, applies head-tail orientation resolution per fish per frame.

        When ``tracklet_groups`` is empty or None, falls back to processing all
        detections (original behavior, no orientation resolution).

        Populates ``context.annotated_detections`` with per-frame per-camera
        AnnotatedDetection lists.

        Args:
            context: Accumulated pipeline state from prior stages. Must have
                ``detections`` and ``camera_ids`` populated by Stage 1.

        Returns:
            The same *context* object with ``annotated_detections`` populated.

        Raises:
            ValueError: If ``context.detections`` or ``context.camera_ids`` is
                None (Stage 1 has not yet run).

        """
        from aquapose.io.video import VideoSet

        detections = context.get("detections")
        camera_ids = context.get("camera_ids")

        # Check for tracklet groups from Association stage
        tracklet_groups = context.tracklet_groups
        has_groups = tracklet_groups is not None and len(tracklet_groups) > 0

        if has_groups:
            logger.info(
                "MidlineStage: %d tracklet groups available, "
                "filtering detections to confirmed fish",
                len(tracklet_groups),  # type: ignore[arg-type]
            )
            # Build tracklet-detection index for filtering
            group_det_index = _build_group_detection_index(
                tracklet_groups,
                detections,  # type: ignore[arg-type]
            )
        else:
            logger.warning(
                "MidlineStage: no tracklet_groups available, "
                "processing all detections (fallback mode)"
            )
            group_det_index = None

        t0 = time.perf_counter()

        video_set = VideoSet(self._video_paths, undistortion=self._undist_maps)
        annotated_per_frame: list[dict[str, list]] = []

        with video_set:
            for frame_idx, frames in video_set:
                if frame_idx >= len(detections):  # type: ignore[arg-type]
                    break

                frame_dets = detections[frame_idx]  # type: ignore[index]

                if group_det_index is not None:
                    # Filter to only detections in tracklet groups
                    filtered_dets = _filter_frame_detections(
                        frame_idx, frame_dets, group_det_index
                    )
                else:
                    filtered_dets = frame_dets

                annotated = self._backend.process_frame(  # type: ignore[union-attr]
                    frame_idx=frame_idx,
                    frame_dets=filtered_dets,
                    frames=frames,
                    camera_ids=camera_ids,  # type: ignore[arg-type]
                )
                annotated_per_frame.append(annotated)

        # Apply orientation resolution if LUTs and tracklet groups available
        if has_groups and self._lut_config is not None:
            annotated_per_frame = self._apply_orientation(
                annotated_per_frame,
                tracklet_groups,  # type: ignore[arg-type]
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            "MidlineStage.run: %d frames, %d cameras, %.2fs",
            len(annotated_per_frame),
            len(camera_ids),  # type: ignore[arg-type]
            elapsed,
        )

        context.annotated_detections = annotated_per_frame
        return context

    def _apply_orientation(
        self,
        annotated_per_frame: list[dict[str, list]],
        tracklet_groups: list,
    ) -> list[dict[str, list]]:
        """Apply head-tail orientation resolution to extracted midlines.

        For each fish (TrackletGroup) in each frame, collects midline points
        across cameras and resolves head-tail orientation using cross-camera
        geometry, velocity alignment, and temporal prior.

        Args:
            annotated_per_frame: Per-frame per-camera AnnotatedDetection lists.
            tracklet_groups: TrackletGroup list from Association stage.

        Returns:
            Updated annotated_per_frame with orientation-corrected midlines.
        """
        from aquapose.calibration.luts import load_forward_luts
        from aquapose.core.midline.orientation import resolve_orientation

        try:
            forward_luts = load_forward_luts(
                str(self._calibration_path), self._lut_config
            )
        except Exception:
            logger.warning("Failed to load ForwardLUTs for orientation -- skipping")
            return annotated_per_frame

        # Use midline_config for orientation weights, or default config
        config = self._midline_config
        if config is None:
            config = _DefaultOrientationConfig()

        # Track per-fish orientation across frames
        prev_orientations: dict[int, int] = {}

        for frame_idx, frame_annotated in enumerate(annotated_per_frame):
            for group in tracklet_groups:
                fish_id = group.fish_id

                # Collect midline points for this fish across cameras
                midline_points_by_cam: dict[str, np.ndarray] = {}
                velocity = None
                speed = 0.0

                for tracklet in group.tracklets:
                    cam_id = tracklet.camera_id
                    frame_map = {f: i for i, f in enumerate(tracklet.frames)}

                    if frame_idx not in frame_map:
                        continue

                    # Find matching annotated detection for this camera/frame
                    if cam_id not in frame_annotated:
                        continue

                    tidx = frame_map[frame_idx]
                    centroid = tracklet.centroids[tidx]

                    # Find closest annotated detection by centroid
                    best_ad = _find_matching_annotated(
                        frame_annotated[cam_id], centroid
                    )

                    if best_ad is not None and best_ad.midline is not None:
                        midline_points_by_cam[cam_id] = np.array(
                            best_ad.midline.points, dtype=np.float64
                        )

                    # Extract velocity from primary camera (first with data)
                    if velocity is None and tidx > 0:
                        prev_centroid = tracklet.centroids[tidx - 1]
                        dx = centroid[0] - prev_centroid[0]
                        dy = centroid[1] - prev_centroid[1]
                        velocity = (float(dx), float(dy))
                        speed = float(np.sqrt(dx * dx + dy * dy))

                if not midline_points_by_cam:
                    continue

                # Resolve orientation
                prev_orient = prev_orientations.get(fish_id)
                corrected, orientation = resolve_orientation(
                    midline_points_by_cam,
                    forward_luts,
                    velocity=velocity,
                    prev_orientation=prev_orient,
                    speed=speed,
                    config=config,
                )
                prev_orientations[fish_id] = orientation

                # Apply corrections back to annotated detections
                for cam_id, corrected_pts in corrected.items():
                    if cam_id not in frame_annotated:
                        continue
                    for ad in frame_annotated[cam_id]:
                        if ad.midline is not None:
                            ad_pts = np.array(ad.midline.points, dtype=np.float64)
                            if np.allclose(
                                ad_pts,
                                midline_points_by_cam.get(cam_id, ad_pts),
                                atol=1.0,
                            ):
                                # Replace midline points with corrected version
                                ad.midline = type(ad.midline)(
                                    points=np.asarray(corrected_pts, dtype=np.float32),
                                    half_widths=ad.midline.half_widths,
                                    fish_id=ad.midline.fish_id,
                                    camera_id=ad.midline.camera_id,
                                    frame_index=ad.midline.frame_index,
                                    is_head_to_tail=ad.midline.is_head_to_tail,
                                )
                                break

        return annotated_per_frame


# ---------------------------------------------------------------------------
# Default orientation config (avoids engine import in core/)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DefaultOrientationConfig:
    """Fallback orientation config when no midline_config is provided.

    Mirrors MidlineConfig orientation defaults without importing from engine/.
    """

    speed_threshold: float = 2.0
    orientation_weight_geometric: float = 1.0
    orientation_weight_velocity: float = 0.5
    orientation_weight_temporal: float = 0.3


# ---------------------------------------------------------------------------
# Internal helpers for tracklet-group filtering
# ---------------------------------------------------------------------------


def _build_group_detection_index(
    tracklet_groups: list,
    detections: list[dict[str, list]],
) -> dict[int, dict[str, set[int]]]:
    """Build a frame-indexed lookup of which detections belong to tracklet groups.

    For each frame, for each camera, records the detection indices that match
    a tracklet centroid in one of the groups.

    Args:
        tracklet_groups: TrackletGroup list from Association stage.
        detections: Per-frame per-camera detection lists from Stage 1.

    Returns:
        Dict mapping frame_idx -> {cam_id -> set of detection indices}.
    """
    index: dict[int, dict[str, set[int]]] = {}

    for group in tracklet_groups:
        for tracklet in group.tracklets:
            cam_id = tracklet.camera_id
            frame_map = {f: i for i, f in enumerate(tracklet.frames)}

            for frame_idx, tidx in frame_map.items():
                if frame_idx >= len(detections):
                    continue

                frame_dets = detections[frame_idx]
                if cam_id not in frame_dets:
                    continue

                centroid = tracklet.centroids[tidx]
                det_list = frame_dets[cam_id]

                # Find closest detection by centroid
                best_idx = _find_closest_detection_idx(det_list, centroid)
                if best_idx is not None:
                    if frame_idx not in index:
                        index[frame_idx] = {}
                    if cam_id not in index[frame_idx]:
                        index[frame_idx][cam_id] = set()
                    index[frame_idx][cam_id].add(best_idx)

    return index


def _find_closest_detection_idx(
    det_list: list,
    centroid: tuple[float, float],
    tolerance: float = 5.0,
) -> int | None:
    """Find the detection index closest to a tracklet centroid.

    Args:
        det_list: List of Detection objects from Stage 1.
        centroid: (u, v) centroid from the tracklet.
        tolerance: Maximum pixel distance for matching.

    Returns:
        Index of the closest detection, or None if none within tolerance.
    """
    best_idx = None
    best_dist = tolerance

    for i, det in enumerate(det_list):
        if hasattr(det, "centroid"):
            c = det.centroid
            dist = ((c[0] - centroid[0]) ** 2 + (c[1] - centroid[1]) ** 2) ** 0.5
        elif hasattr(det, "bbox"):
            bx, by, bw, bh = det.bbox
            cx = bx + bw / 2
            cy = by + bh / 2
            dist = ((cx - centroid[0]) ** 2 + (cy - centroid[1]) ** 2) ** 0.5
        else:
            continue

        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx


def _filter_frame_detections(
    frame_idx: int,
    frame_dets: dict[str, list],
    group_det_index: dict[int, dict[str, set[int]]],
) -> dict[str, list]:
    """Filter a frame's detections to only include those in tracklet groups.

    Args:
        frame_idx: Current frame index.
        frame_dets: Per-camera detection lists for this frame.
        group_det_index: Pre-computed index from _build_group_detection_index.

    Returns:
        Filtered per-camera detection dict.
    """
    if frame_idx not in group_det_index:
        return {}

    frame_index = group_det_index[frame_idx]
    filtered: dict[str, list] = {}

    for cam_id, det_list in frame_dets.items():
        if cam_id not in frame_index:
            continue
        kept_indices = frame_index[cam_id]
        filtered[cam_id] = [det for i, det in enumerate(det_list) if i in kept_indices]

    return filtered


def _find_matching_annotated(
    annotated_list: list,
    centroid: tuple[float, float],
    tolerance: float = 10.0,
) -> object | None:
    """Find the AnnotatedDetection closest to a tracklet centroid.

    Args:
        annotated_list: List of AnnotatedDetection objects.
        centroid: (u, v) centroid from the tracklet.
        tolerance: Maximum pixel distance for matching.

    Returns:
        Closest AnnotatedDetection, or None if none within tolerance.
    """
    best_ad = None
    best_dist = tolerance

    for ad in annotated_list:
        if hasattr(ad, "detection") and hasattr(ad.detection, "centroid"):
            c = ad.detection.centroid
            dist = ((c[0] - centroid[0]) ** 2 + (c[1] - centroid[1]) ** 2) ** 0.5
        elif hasattr(ad, "detection") and hasattr(ad.detection, "bbox"):
            bx, by, bw, bh = ad.detection.bbox
            cx = bx + bw / 2
            cy = by + bh / 2
            dist = ((cx - centroid[0]) ** 2 + (cy - centroid[1]) ** 2) ** 0.5
        else:
            continue

        if dist < best_dist:
            best_dist = dist
            best_ad = ad

    return best_ad
