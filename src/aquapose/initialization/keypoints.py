"""PCA-based keypoint extraction from binary masks."""

from __future__ import annotations

import numpy as np


def extract_keypoints(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract center and endpoint keypoints from a binary mask using PCA.

    Steps:
        1. Get binary mask pixel coordinates as Nx2 array (u, v format)
        2. Centroid (mean) = center keypoint
        3. PCA on coordinates: compute 2x2 covariance, use np.linalg.eigh
        4. Major axis = eigenvector of largest eigenvalue
        5. Project all pixels onto major axis → min/max = two endpoint keypoints
        6. Enforce canonical sign: ensure endpoint_a is the one with max projection

    Args:
        mask: Binary mask, shape (H, W), uint8 (0/255) or bool.

    Returns:
        center: Centroid pixel coord, shape (2,), float32. (u, v) format.
        endpoint_a: First endpoint along major axis, shape (2,), float32.
            Always the endpoint with maximum projection onto major axis.
        endpoint_b: Second endpoint along major axis, shape (2,), float32.

    Raises:
        ValueError: If mask has no foreground pixels.
    """
    # Get foreground pixel coordinates
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        raise ValueError("Empty mask: no foreground pixels found.")

    # (u, v) = (col, row) format
    coords = np.stack([cols, rows], axis=1).astype(np.float32)  # (N, 2)

    # Centroid
    centroid = coords.mean(axis=0)  # (2,)

    # Handle degenerate single-pixel case
    if len(coords) == 1:
        center = centroid.astype(np.float32)
        return center, center.copy(), center.copy()

    # PCA: 2x2 covariance matrix
    centered = coords - centroid  # (N, 2)
    cov = (centered.T @ centered) / len(coords)  # (2, 2)

    # eigh returns eigenvalues ascending; last eigenvector = major axis
    _eigenvalues, eigenvectors = np.linalg.eigh(cov)
    major_axis = eigenvectors[:, -1]  # (2,) - eigenvector of largest eigenvalue

    # Project all pixels onto major axis
    projections = centered @ major_axis  # (N,)

    # Enforce canonical sign: endpoint_a should have max projection
    # If the point with max projection is actually the most negative, negate axis
    if projections.max() < abs(projections.min()):
        major_axis = -major_axis
        projections = -projections

    # Endpoints at extremes of projection
    endpoint_a = (centroid + major_axis * projections.max()).astype(np.float32)
    endpoint_b = (centroid + major_axis * projections.min()).astype(np.float32)
    center = centroid.astype(np.float32)

    return center, endpoint_a, endpoint_b


def extract_keypoints_batch(
    masks: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract keypoints from multiple masks (batch-first API).

    Args:
        masks: List of binary masks, each shape (H, W), uint8 or bool.

    Returns:
        List of (center, endpoint_a, endpoint_b) tuples, one per mask.

    Raises:
        ValueError: If any mask has no foreground pixels.
    """
    return [extract_keypoints(mask) for mask in masks]
