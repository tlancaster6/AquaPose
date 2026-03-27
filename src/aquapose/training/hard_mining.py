"""Hard example mining: identify frames and fish for manual annotation.

Provides pure mining functions that take loaded midline data and return
dataclass lists describing hard examples. No file I/O — CLI and writing
are handled by :mod:`aquapose.training.pseudo_label_cli`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from aquapose.core.types.reconstruction import Midline3D
from aquapose.training.pseudo_labels import compute_curvature

if TYPE_CHECKING:
    from aquapose.calibration.luts import InverseLUT
    from aquapose.core.tracking.types import Tracklet2D

__all__ = [
    "HighCurvatureExample",
    "HighResidualExample",
    "LowCamerasExample",
    "TrackingGapExample",
    "compute_thresholds",
    "mine_high_curvature",
    "mine_high_residual",
    "mine_low_cameras",
    "mine_tracking_gaps",
]


@dataclass
class TrackingGapExample:
    """A frame where a fish was coasting/missing in a specific camera."""

    frame_idx: int
    camera_id: str
    track_id: int
    gap_length: int
    gap_position: int  # 0-based position within the gap


@dataclass
class LowCamerasExample:
    """A fish reconstructed from too few cameras."""

    frame_idx: int
    fish_id: int
    n_cameras: int
    contributing_cameras: list[str] = field(default_factory=list)
    visible_gap_cameras: list[str] = field(default_factory=list)


@dataclass
class HighCurvatureExample:
    """A fish with unusually high 3D curvature."""

    frame_idx: int
    fish_id: int
    curvature: float


@dataclass
class HighResidualExample:
    """A fish with high mean reprojection residual."""

    frame_idx: int
    fish_id: int
    mean_residual: float
    n_cameras: int


def _midline_curvature(midline: Midline3D) -> float | None:
    """Compute curvature from whichever point representation is available."""
    pts = (
        midline.control_points if midline.control_points is not None else midline.points
    )
    if pts is None or pts.shape[0] < 3:
        return None
    # Filter out rows with any NaN (partial triangulation failures)
    valid = ~np.any(np.isnan(pts), axis=1)
    pts = pts[valid]
    if pts.shape[0] < 3:
        return None
    return compute_curvature(pts)


def compute_thresholds(
    midlines_3d: list[dict[int, Midline3D]],
    curvature_percentile: float = 95.0,
    residual_percentile: float = 95.0,
) -> tuple[float, float]:
    """Pre-scan midlines to compute percentile-based thresholds.

    Args:
        midlines_3d: Per-frame dicts mapping fish_id to Midline3D.
        curvature_percentile: Percentile for curvature threshold (0-100).
        residual_percentile: Percentile for residual threshold (0-100).

    Returns:
        Tuple of (curvature_threshold, residual_threshold).
    """
    curvatures: list[float] = []
    residuals: list[float] = []

    for fish_dict in midlines_3d:
        if not fish_dict:
            continue
        for midline in fish_dict.values():
            curv = _midline_curvature(midline)
            if curv is not None:
                curvatures.append(curv)
            if midline.mean_residual >= 0:
                residuals.append(midline.mean_residual)

    curv_thresh = (
        float(np.percentile(curvatures, curvature_percentile)) if curvatures else 0.0
    )
    resid_thresh = (
        float(np.percentile(residuals, residual_percentile)) if residuals else 0.0
    )

    return curv_thresh, resid_thresh


def mine_tracking_gaps(
    tracks_2d: dict[str, list[Tracklet2D]],
    temporal_step: int = 1,
    max_examples: int | None = None,
    min_gap_length: int = 1,
) -> list[TrackingGapExample]:
    """Find frames where the detector missed a tracked fish in a specific camera.

    Scans each tracklet's ``frame_status`` for runs of ``"coasted"`` frames
    (detector missed but tracker predicted). Emits only the **first frame**
    of each gap to avoid redundancy. Sorted by longest gaps first.

    Args:
        tracks_2d: Per-camera tracklet lists (``context.tracks_2d``).
        temporal_step: Process every Nth frame.
        max_examples: Cap on returned examples (after sorting).
        min_gap_length: Minimum consecutive coasted frames to count as a gap.

    Returns:
        Examples sorted by longest gap first, then by frame index.
    """
    examples: list[TrackingGapExample] = []

    for cam_id, tracklets in tracks_2d.items():
        for tracklet in tracklets:
            frames = tracklet.frames
            statuses = tracklet.frame_status

            # Scan for runs of "coasted" frames
            i = 0
            while i < len(statuses):
                if statuses[i] == "coasted":
                    # Start of a gap — find the end
                    gap_start = i
                    while i < len(statuses) and statuses[i] == "coasted":
                        i += 1
                    gap_len = i - gap_start

                    if gap_len >= min_gap_length:
                        first_frame = frames[gap_start]
                        if first_frame % temporal_step == 0:
                            examples.append(
                                TrackingGapExample(
                                    frame_idx=first_frame,
                                    camera_id=cam_id,
                                    track_id=tracklet.track_id,
                                    gap_length=gap_len,
                                    gap_position=0,
                                )
                            )
                else:
                    i += 1

    examples.sort(key=lambda e: (-e.gap_length, e.frame_idx))
    if max_examples is not None:
        examples = examples[:max_examples]
    return examples


def mine_low_cameras(
    midlines_3d: list[dict[int, Midline3D]],
    inverse_lut: InverseLUT,
    max_cameras: int = 2,
    temporal_step: int = 1,
    max_examples: int | None = None,
) -> list[LowCamerasExample]:
    """Find fish reconstructed from too few cameras with InverseLUT filtering.

    For each low-camera fish, queries the InverseLUT to determine which
    cameras *should* geometrically see the fish, then reports only cameras
    that are visible but didn't contribute.

    Args:
        midlines_3d: Per-frame dicts mapping fish_id to Midline3D.
        inverse_lut: InverseLUT for visibility queries.
        max_cameras: Include fish with n_cameras <= this value.
        temporal_step: Process every Nth frame.
        max_examples: Cap on returned examples (after sorting).

    Returns:
        Examples sorted by fewest cameras first. Only includes examples
        where at least one geometrically visible camera didn't contribute.
    """
    from aquapose.calibration.luts import ghost_point_lookup

    examples: list[LowCamerasExample] = []

    for frame_idx, fish_dict in enumerate(midlines_3d):
        if frame_idx % temporal_step != 0:
            continue
        if not fish_dict:
            continue
        for fish_id, midline in fish_dict.items():
            if midline.n_cameras > max_cameras:
                continue

            contributing = (
                set(midline.per_camera_residuals.keys())
                if midline.per_camera_residuals
                else set()
            )

            # Compute 3D centroid for visibility query
            pts = midline.points
            if pts is None or np.all(np.isnan(pts)):
                continue
            centroid = np.nanmean(pts, axis=0)
            if np.any(np.isnan(centroid)):
                continue

            centroid_tensor = torch.from_numpy(centroid[None].astype(np.float32))
            visible_list = ghost_point_lookup(inverse_lut, centroid_tensor)
            visible_cam_ids = {cam_id for cam_id, _, _ in visible_list[0]}

            gap_cameras = sorted(visible_cam_ids - contributing)
            if not gap_cameras:
                continue

            examples.append(
                LowCamerasExample(
                    frame_idx=frame_idx,
                    fish_id=fish_id,
                    n_cameras=midline.n_cameras,
                    contributing_cameras=sorted(contributing),
                    visible_gap_cameras=gap_cameras,
                )
            )

    examples.sort(key=lambda e: e.n_cameras)
    if max_examples is not None:
        examples = examples[:max_examples]
    return examples


def mine_high_curvature(
    midlines_3d: list[dict[int, Midline3D]],
    curvature_threshold: float,
    temporal_step: int = 1,
    max_examples: int | None = None,
) -> list[HighCurvatureExample]:
    """Find fish with curvature above *curvature_threshold*.

    Args:
        midlines_3d: Per-frame dicts mapping fish_id to Midline3D.
        curvature_threshold: Minimum curvature to include.
        temporal_step: Process every Nth frame.
        max_examples: Cap on returned examples (after sorting).

    Returns:
        Examples sorted by highest curvature first.
    """
    examples: list[HighCurvatureExample] = []

    for frame_idx, fish_dict in enumerate(midlines_3d):
        if frame_idx % temporal_step != 0:
            continue
        if not fish_dict:
            continue
        for fish_id, midline in fish_dict.items():
            curv = _midline_curvature(midline)
            if curv is not None and curv >= curvature_threshold:
                examples.append(
                    HighCurvatureExample(
                        frame_idx=frame_idx,
                        fish_id=fish_id,
                        curvature=curv,
                    )
                )

    examples.sort(key=lambda e: e.curvature, reverse=True)
    if max_examples is not None:
        examples = examples[:max_examples]
    return examples


def mine_high_residual(
    midlines_3d: list[dict[int, Midline3D]],
    residual_threshold: float,
    temporal_step: int = 1,
    max_examples: int | None = None,
) -> list[HighResidualExample]:
    """Find fish with mean reprojection residual above *residual_threshold*.

    Args:
        midlines_3d: Per-frame dicts mapping fish_id to Midline3D.
        residual_threshold: Minimum mean residual in pixels.
        temporal_step: Process every Nth frame.
        max_examples: Cap on returned examples (after sorting).

    Returns:
        Examples sorted by highest residual first.
    """
    examples: list[HighResidualExample] = []

    for frame_idx, fish_dict in enumerate(midlines_3d):
        if frame_idx % temporal_step != 0:
            continue
        if not fish_dict:
            continue
        for fish_id, midline in fish_dict.items():
            if midline.mean_residual >= residual_threshold:
                examples.append(
                    HighResidualExample(
                        frame_idx=frame_idx,
                        fish_id=fish_id,
                        mean_residual=midline.mean_residual,
                        n_cameras=midline.n_cameras,
                    )
                )

    examples.sort(key=lambda e: e.mean_residual, reverse=True)
    if max_examples is not None:
        examples = examples[:max_examples]
    return examples
