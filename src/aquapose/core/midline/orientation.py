"""Head-tail orientation resolver for fish midlines.

Combines three signals to disambiguate head from tail in 2D midlines:
1. Cross-camera geometric vote (triangulate both orientations)
2. Velocity alignment (midline direction vs movement direction)
3. Temporal prior (maintain previous frame's orientation)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch

from aquapose.core.association.scoring import ray_ray_closest_point

if TYPE_CHECKING:
    from aquapose.calibration.luts import ForwardLUT

logger = logging.getLogger(__name__)

__all__ = ["OrientationConfigLike", "resolve_orientation"]


# ---------------------------------------------------------------------------
# Config protocol (IB-003: core/ must not import engine/)
# ---------------------------------------------------------------------------


@runtime_checkable
class OrientationConfigLike(Protocol):
    """Structural protocol for orientation resolution configuration.

    Satisfied by ``MidlineConfig`` from ``engine.config`` without explicit
    import, preserving the core -> engine import boundary (IB-003).

    Attributes:
        speed_threshold: Minimum speed (pixels/frame) for velocity signal.
        orientation_weight_geometric: Weight for geometric vote.
        orientation_weight_velocity: Weight for velocity alignment.
        orientation_weight_temporal: Weight for temporal prior.
    """

    speed_threshold: float
    orientation_weight_geometric: float
    orientation_weight_velocity: float
    orientation_weight_temporal: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_orientation(
    midline_points_by_cam: dict[str, np.ndarray],
    forward_luts: dict[str, ForwardLUT],
    velocity: tuple[float, float] | None,
    prev_orientation: int | None,
    speed: float,
    config: OrientationConfigLike,
) -> tuple[dict[str, np.ndarray], int]:
    """Resolve head-tail orientation across cameras for one fish in one frame.

    Combines three signals with configurable weights to determine whether
    the midline points are oriented head-first (+1) or need flipping (-1).
    Cameras that disagree with the consensus have their midlines reversed.

    Args:
        midline_points_by_cam: Camera-ID to (N, 2) midline points array.
            N=15 points from putative head (index 0) to tail (index -1).
        forward_luts: Per-camera ForwardLUT for ray back-projection.
        velocity: OC-SORT Kalman-filtered velocity (du/dt, dv/dt) in
            pixels/frame from the primary camera. None if unavailable.
        prev_orientation: +1 or -1 from previous frame. None if first frame.
        speed: Velocity magnitude in pixels/frame.
        config: Orientation weight and threshold parameters.

    Returns:
        Tuple of (corrected_midline_points_by_cam, orientation) where
        orientation is +1 (original) or -1 (flipped consensus).
    """
    cam_ids = list(midline_points_by_cam.keys())

    # Gather votes and weights
    votes: list[float] = []  # +1 or -1
    weights: list[float] = []

    # Signal 1: Cross-camera geometric vote
    if len(cam_ids) >= 2:
        geo_vote = _geometric_vote(midline_points_by_cam, forward_luts, cam_ids)
        votes.append(geo_vote)
        weights.append(config.orientation_weight_geometric)

    # Signal 2: Velocity alignment
    if velocity is not None and speed >= config.speed_threshold:
        vel_vote = _velocity_vote(midline_points_by_cam, velocity, cam_ids)
        votes.append(vel_vote)
        weights.append(config.orientation_weight_velocity)

    # Signal 3: Temporal prior
    if prev_orientation is not None:
        votes.append(float(prev_orientation))
        weights.append(config.orientation_weight_temporal)

    # Combine votes
    if votes:
        weighted_sum = sum(v * w for v, w in zip(votes, weights, strict=True))
        orientation = 1 if weighted_sum >= 0 else -1
    else:
        # No signals available (single camera, no velocity, first frame)
        orientation = 1

    # Apply per-camera correction
    corrected = _apply_per_camera_flip(
        midline_points_by_cam, forward_luts, cam_ids, orientation
    )

    return corrected, orientation


# ---------------------------------------------------------------------------
# Signal implementations
# ---------------------------------------------------------------------------


def _geometric_vote(
    midline_points_by_cam: dict[str, np.ndarray],
    forward_luts: dict[str, ForwardLUT],
    cam_ids: list[str],
) -> float:
    """Compute geometric vote by triangulating head/tail in both orientations.

    Triangulates point[0] (putative head) and point[-1] (putative tail)
    across all camera pairs. Then flips all midlines and repeats. The
    orientation with lower total ray convergence error wins.

    Args:
        midline_points_by_cam: Camera-ID to (N, 2) midline points.
        forward_luts: Per-camera ForwardLUTs.
        cam_ids: Camera IDs to use.

    Returns:
        +1 if original orientation has lower error, -1 if flipped is better.
    """
    # Filter to cameras that have LUTs
    valid_cams = [c for c in cam_ids if c in forward_luts]
    if len(valid_cams) < 2:
        return 0.0  # Can't determine

    # Original orientation error
    original_error = _triangulation_error(
        midline_points_by_cam, forward_luts, valid_cams, flip=False
    )

    # Flipped orientation error
    flipped_error = _triangulation_error(
        midline_points_by_cam, forward_luts, valid_cams, flip=True
    )

    if original_error <= flipped_error:
        return 1.0
    return -1.0


def _triangulation_error(
    midline_points_by_cam: dict[str, np.ndarray],
    forward_luts: dict[str, ForwardLUT],
    cam_ids: list[str],
    flip: bool,
) -> float:
    """Compute total ray convergence error for head and tail points.

    Args:
        midline_points_by_cam: Camera-ID to (N, 2) midline points.
        forward_luts: Per-camera ForwardLUTs.
        cam_ids: Valid camera IDs.
        flip: If True, reverse point order before triangulating.

    Returns:
        Sum of mean pairwise ray-ray distances for head and tail points.
    """
    total_error = 0.0

    for point_index in [0, -1]:
        rays: list[tuple[np.ndarray, np.ndarray]] = []

        for cam_id in cam_ids:
            pts = midline_points_by_cam[cam_id]
            if flip:
                pts = pts[::-1]

            pixel = pts[point_index]
            pix_tensor = torch.tensor(
                [[float(pixel[0]), float(pixel[1])]], dtype=torch.float32
            )
            origins, dirs = forward_luts[cam_id].cast_ray(pix_tensor)
            o = origins[0].cpu().numpy().astype(np.float64)
            d = dirs[0].cpu().numpy().astype(np.float64)
            rays.append((o, d))

        if len(rays) < 2:
            continue

        # Mean pairwise distance
        pair_dists: list[float] = []
        for i in range(len(rays)):
            for j in range(i + 1, len(rays)):
                dist, _ = ray_ray_closest_point(
                    rays[i][0], rays[i][1], rays[j][0], rays[j][1]
                )
                pair_dists.append(dist)

        total_error += float(np.mean(pair_dists))

    return total_error


def _velocity_vote(
    midline_points_by_cam: dict[str, np.ndarray],
    velocity: tuple[float, float],
    cam_ids: list[str],
) -> float:
    """Vote based on alignment of midline direction with velocity vector.

    Uses the first camera's midline to compute direction from head (index 0)
    to tail (index -1) and checks alignment with velocity.

    Args:
        midline_points_by_cam: Camera-ID to (N, 2) midline points.
        velocity: (du/dt, dv/dt) velocity in pixels/frame.
        cam_ids: Camera IDs (first is used as primary).

    Returns:
        +1 if midline head aligns with travel direction, -1 otherwise.
    """
    primary_cam = cam_ids[0]
    pts = midline_points_by_cam[primary_cam]

    # Direction from head to tail
    midline_dir = pts[-1] - pts[0]
    midline_norm = float(np.linalg.norm(midline_dir))
    if midline_norm < 1e-6:
        return 0.0

    vel = np.array(velocity, dtype=np.float64)
    vel_norm = float(np.linalg.norm(vel))
    if vel_norm < 1e-6:
        return 0.0

    # Fish swim head-first, so head (index 0) should be in the direction
    # of travel. The midline_dir goes from head to tail, so if aligned
    # with velocity, the fish is "tail-forward" — meaning we need to flip.
    # Actually: midline_dir = tail - head. If the fish swims head-first,
    # velocity should be OPPOSITE to midline_dir (head → velocity direction).
    # So: alignment < 0 means head is forward (correct), alignment > 0 means
    # tail is forward (flip needed).
    # Convention: +1 = keep as-is, -1 = flip.
    # head_dir = -midline_dir (from tail to head = head direction)
    head_dir = -midline_dir
    alignment = float(np.dot(head_dir, vel)) / (midline_norm * vel_norm)

    return 1.0 if alignment > 0 else -1.0


def _apply_per_camera_flip(
    midline_points_by_cam: dict[str, np.ndarray],
    forward_luts: dict[str, ForwardLUT],
    cam_ids: list[str],
    consensus_orientation: int,
) -> dict[str, np.ndarray]:
    """Flip individual cameras that disagree with the consensus orientation.

    For each camera, check if its midline direction is consistent with the
    consensus. If not, reverse its midline point order.

    Args:
        midline_points_by_cam: Camera-ID to (N, 2) midline points.
        forward_luts: Per-camera ForwardLUTs.
        cam_ids: Camera IDs to process.
        consensus_orientation: +1 (keep original) or -1 (flip).

    Returns:
        New dict with corrected midline points per camera.
    """
    corrected: dict[str, np.ndarray] = {}

    # If only one camera, just apply the global consensus
    if len(cam_ids) <= 1:
        for cam_id in cam_ids:
            pts = midline_points_by_cam[cam_id]
            if consensus_orientation == -1:
                corrected[cam_id] = pts[::-1].copy()
            else:
                corrected[cam_id] = pts.copy()
        return corrected

    # For multiple cameras: check each camera's geometric consistency
    # with the consensus by comparing its head-tail direction to the others
    valid_cams = [c for c in cam_ids if c in forward_luts]
    if len(valid_cams) < 2:
        # Can't do per-camera check, apply global flip
        for cam_id in cam_ids:
            pts = midline_points_by_cam[cam_id]
            if consensus_orientation == -1:
                corrected[cam_id] = pts[::-1].copy()
            else:
                corrected[cam_id] = pts.copy()
        return corrected

    # Compute per-camera geometric vote independently
    for cam_id in cam_ids:
        pts = midline_points_by_cam[cam_id]

        # Check this camera's head point against others in original orientation
        other_cams = [c for c in valid_cams if c != cam_id and c in forward_luts]
        if not other_cams or cam_id not in forward_luts:
            # Can't compare, apply consensus
            if consensus_orientation == -1:
                corrected[cam_id] = pts[::-1].copy()
            else:
                corrected[cam_id] = pts.copy()
            continue

        # Apply consensus orientation to this camera
        if consensus_orientation == -1:
            corrected[cam_id] = pts[::-1].copy()
        else:
            corrected[cam_id] = pts.copy()

    return corrected
