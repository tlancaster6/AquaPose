"""Temporal smoothing of centroid z for z-denoising.

Smooths per-fish centroid z across time within continuous track segments
using Gaussian filtering, then shifts control points to match.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

__all__ = ["smooth_centroid_z"]


def smooth_centroid_z(
    centroid_z: np.ndarray,
    frame_indices: np.ndarray,
    sigma_frames: int = 3,
) -> np.ndarray:
    """Temporally smooth centroid z values for a single fish.

    Processes a time series of centroid z values, handling temporal gaps
    that define segment boundaries, and applying per-segment Gaussian
    smoothing.

    Args:
        centroid_z: Centroid z per frame, shape ``(T,)``.
        frame_indices: Frame indices, shape ``(T,)``.  Used to detect
            temporal gaps that define segment boundaries.
        sigma_frames: Gaussian filter sigma in frames.

    Returns:
        Smoothed centroid z values, shape ``(T,)``.
    """
    centroid_z = np.asarray(centroid_z, dtype=np.float64).copy()
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    T = len(centroid_z)

    if T < 2:
        return centroid_z

    # Identify continuous segments (frame gap > 1 creates boundary)
    segments: list[tuple[int, int]] = []
    seg_start = 0
    for t in range(1, T):
        if frame_indices[t] - frame_indices[t - 1] > 1:
            segments.append((seg_start, t - 1))
            seg_start = t
    segments.append((seg_start, T - 1))

    smoothed = centroid_z.copy()
    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start + 1
        if seg_len < 2:
            continue
        seg = centroid_z[seg_start : seg_end + 1].copy()
        nan_mask = np.isnan(seg)
        if nan_mask.all():
            continue
        # Interpolate through NaN gaps before filtering
        if nan_mask.any():
            valid_idx = np.where(~nan_mask)[0]
            seg[nan_mask] = np.interp(np.where(nan_mask)[0], valid_idx, seg[valid_idx])
        smoothed[seg_start : seg_end + 1] = gaussian_filter1d(seg, sigma=sigma_frames)
        # Restore NaN at originally-missing positions
        smoothed[seg_start : seg_end + 1][nan_mask] = np.nan

    return smoothed
