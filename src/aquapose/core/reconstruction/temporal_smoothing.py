"""Temporal smoothing of plane normals and control point rotation.

Component B of the z-denoising pipeline. Smooths per-fish plane normals
across time within continuous track segments using Gaussian filtering,
then rotates control points to match the smoothed plane orientation.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

__all__ = ["rotate_control_points_to_plane", "smooth_plane_normals"]


def smooth_plane_normals(
    normals: np.ndarray,
    is_degenerate: np.ndarray,
    fish_ids: np.ndarray,
    frame_indices: np.ndarray,
    sigma_frames: int = 3,
) -> np.ndarray:
    """Temporally smooth plane normals for a single fish.

    Processes a time series of plane normals, enforcing sign consistency,
    handling degenerate frames by interpolation, and applying per-segment
    Gaussian smoothing.

    Args:
        normals: Plane normals, shape ``(T, 3)``.
        is_degenerate: Boolean mask of degenerate frames, shape ``(T,)``.
        fish_ids: Fish ID per frame, shape ``(T,)``.  All entries should
            be the same value (caller groups by fish_id).
        frame_indices: Frame indices, shape ``(T,)``.  Used to detect
            temporal gaps that define segment boundaries.
        sigma_frames: Gaussian filter sigma in frames.

    Returns:
        Smoothed normals, shape ``(T, 3)``.  Unit-length, sign-consistent.
    """
    normals = np.asarray(normals, dtype=np.float64).copy()
    is_degenerate = np.asarray(is_degenerate, dtype=bool)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    T = len(normals)

    if T == 0:
        return normals

    # Step 1: Sign consistency -- flip normals so dot(n_t, n_{t-1}) > 0
    for t in range(1, T):
        if np.dot(normals[t], normals[t - 1]) < 0:
            normals[t] = -normals[t]

    # Step 2: Identify continuous segments (frame gap > 1 creates boundary)
    segments: list[tuple[int, int]] = []  # (start, end) inclusive
    seg_start = 0
    for t in range(1, T):
        if frame_indices[t] - frame_indices[t - 1] > 1:
            segments.append((seg_start, t - 1))
            seg_start = t
    segments.append((seg_start, T - 1))

    # Step 3: For each segment, smooth with Gaussian filter
    smoothed = normals.copy()
    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start + 1
        if seg_len < 2:
            continue

        seg_normals = normals[seg_start : seg_end + 1].copy()  # (L, 3)
        seg_degen = is_degenerate[seg_start : seg_end + 1]

        # Step 4: Handle degenerate frames by linear interpolation
        if np.any(seg_degen) and not np.all(seg_degen):
            valid_mask = ~seg_degen
            valid_idx = np.where(valid_mask)[0]
            for dim in range(3):
                seg_normals[:, dim] = np.interp(
                    np.arange(seg_len),
                    valid_idx,
                    seg_normals[valid_idx, dim],
                )
        elif np.all(seg_degen):
            # All degenerate -- nothing to smooth
            continue

        # Apply Gaussian filter per component
        for dim in range(3):
            seg_normals[:, dim] = gaussian_filter1d(
                seg_normals[:, dim], sigma=sigma_frames
            )

        # Step 5: Re-normalize to unit length
        norms = np.linalg.norm(seg_normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        seg_normals = seg_normals / norms

        smoothed[seg_start : seg_end + 1] = seg_normals

    return smoothed.astype(np.float64)


def rotate_control_points_to_plane(
    control_points: np.ndarray,
    centroid: np.ndarray,
    raw_normal: np.ndarray,
    smoothed_normal: np.ndarray,
) -> np.ndarray:
    """Rotate control points from raw to smoothed plane orientation.

    Uses Rodrigues' rotation formula to compute the rotation that maps
    ``raw_normal`` to ``smoothed_normal``, applied relative to the centroid.

    Args:
        control_points: B-spline control points, shape ``(7, 3)``.
        centroid: Rotation centre (plane centroid), shape ``(3,)``.
        raw_normal: Original plane normal, shape ``(3,)``.
        smoothed_normal: Target (smoothed) plane normal, shape ``(3,)``.

    Returns:
        Rotated control points, shape ``(7, 3)``.
    """
    control_points = np.asarray(control_points, dtype=np.float64)
    centroid = np.asarray(centroid, dtype=np.float64)
    raw_normal = np.asarray(raw_normal, dtype=np.float64)
    smoothed_normal = np.asarray(smoothed_normal, dtype=np.float64)

    # Normalize
    raw_normal = raw_normal / max(np.linalg.norm(raw_normal), 1e-12)
    smoothed_normal = smoothed_normal / max(np.linalg.norm(smoothed_normal), 1e-12)

    dot = np.clip(np.dot(raw_normal, smoothed_normal), -1.0, 1.0)

    # Edge case: nearly identical normals
    if dot > 0.9999:
        return control_points.copy()

    # Edge case: nearly opposite normals (180-degree rotation)
    if dot < -0.9999:
        # Find an arbitrary perpendicular axis
        if abs(raw_normal[0]) < 0.9:
            perp = np.cross(raw_normal, [1.0, 0.0, 0.0])
        else:
            perp = np.cross(raw_normal, [0.0, 1.0, 0.0])
        perp = perp / np.linalg.norm(perp)
        # 180-degree rotation around perp: R = 2 * perp @ perp^T - I
        R = 2.0 * np.outer(perp, perp) - np.eye(3)
    else:
        # Rodrigues' rotation formula
        axis = np.cross(raw_normal, smoothed_normal)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            return control_points.copy()
        axis = axis / axis_norm
        angle = np.arccos(dot)

        K = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    # Rotate control points relative to centroid
    centred = control_points - centroid
    rotated = (R @ centred.T).T + centroid

    return rotated.astype(np.float64)
