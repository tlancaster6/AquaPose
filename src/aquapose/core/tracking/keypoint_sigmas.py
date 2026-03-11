"""Per-keypoint OKS sigmas for the 6-keypoint fish pose model.

Provides DEFAULT_SIGMAS (hard-coded reasonable priors) and compute_keypoint_sigmas
(derives sigmas empirically from annotation records using the COCO methodology).

Keypoint indices:
    0 = nose, 1 = head, 2 = spine1, 3 = spine2, 4 = spine3, 5 = tail
"""

from __future__ import annotations

import numpy as np

__all__ = ["DEFAULT_SIGMAS", "compute_keypoint_sigmas"]

# Hard-coded priors for the 6-keypoint fish model.
# Endpoints (nose=0, tail=5) have larger sigmas because they are anatomically
# more variable and more likely to be occluded.  Mid-body keypoints (spine1-3)
# are more constrained and benefit from tighter matching windows.
# These values will be refined by running compute_keypoint_sigmas on the dataset.
DEFAULT_SIGMAS: np.ndarray = np.array(
    [0.08, 0.06, 0.04, 0.04, 0.05, 0.07],
    dtype=np.float64,
)


def compute_keypoint_sigmas(annotations: list[dict]) -> np.ndarray:
    """Compute per-keypoint OKS sigmas from annotation records.

    Implements the COCO methodology: for each keypoint k, compute the standard
    deviation of ||kpt_k - mean_kpt_k|| / sqrt(obb_area) over all instances.
    This normalises positional variance by fish scale.

    Args:
        annotations: List of dicts, each with:
            - ``keypoints``: np.ndarray of shape ``(K, 2)``, float — 2-D positions.
            - ``obb_area``: float — OBB area (pixels²) used as scale proxy.

    Returns:
        np.ndarray of shape ``(K,)`` containing positive sigma values.
        If fewer than 2 annotations are provided, returns DEFAULT_SIGMAS.
    """
    if len(annotations) < 2:
        return DEFAULT_SIGMAS.copy()

    # Stack keypoints: (N, K, 2)
    all_kpts = np.stack([a["keypoints"] for a in annotations], axis=0).astype(
        np.float64
    )
    areas = np.array([a["obb_area"] for a in annotations], dtype=np.float64)

    _n, _k, _ = all_kpts.shape

    # Compute per-keypoint mean position (K, 2)
    mean_kpts = all_kpts.mean(axis=0)  # (K, 2)

    # Displacement from mean: (N, K, 2)
    displacements = all_kpts - mean_kpts[np.newaxis, :, :]  # (N, K, 2)

    # Euclidean distance: (N, K)
    distances = np.linalg.norm(displacements, axis=2)  # (N, K)

    # Normalise by sqrt(area): (N, K)
    sqrt_areas = np.sqrt(np.maximum(areas, 1.0))[:, np.newaxis]  # (N, 1)
    normalised = distances / sqrt_areas  # (N, K)

    # Per-keypoint sigma = std of normalised distances
    sigmas = normalised.std(axis=0)  # (K,)

    # Guard against zero sigmas (degenerate annotations)
    min_sigma = 1e-4
    sigmas = np.maximum(sigmas, min_sigma)

    return sigmas
