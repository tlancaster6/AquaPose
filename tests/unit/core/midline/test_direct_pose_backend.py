"""Unit tests for PoseEstimationBackend (v3.7 raw-keypoint batch API).

Validates:
- PoseEstimationBackend instantiates without error and without loading any model
- process_batch with no model returns (None, None) for every crop
- process_batch with empty crops returns empty list
- Backend accepts expected constructor kwargs
"""

from __future__ import annotations

import numpy as np

from aquapose.core.pose.backends.pose_estimation import PoseEstimationBackend
from aquapose.core.types.crop import AffineCrop
from aquapose.core.types.detection import Detection

_DEFAULT_T = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _make_affine_crop(crop_w: int = 128, crop_h: int = 64) -> AffineCrop:
    """Build an AffineCrop with an identity-like affine matrix."""
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    image = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    return AffineCrop(
        image=image, M=M, crop_size=(crop_w, crop_h), frame_shape=(480, 640)
    )


def test_pose_estimation_no_model_returns_none_tuples() -> None:
    """PoseEstimationBackend with no model returns (None, None) for every crop."""
    backend = PoseEstimationBackend(keypoint_t_values=_DEFAULT_T)

    det1 = Detection(bbox=(10, 10, 50, 50), mask=None, area=2500, confidence=0.9)
    det2 = Detection(bbox=(60, 60, 30, 30), mask=None, area=900, confidence=0.8)

    crops = [_make_affine_crop(), _make_affine_crop()]
    metadata = [(det1, "cam1", 0), (det2, "cam1", 0)]

    result = backend.process_batch(crops, metadata)

    assert len(result) == 2
    for kpts_xy, kpts_conf in result:
        assert kpts_xy is None, "Stub must return None kpts_xy when no model"
        assert kpts_conf is None, "Stub must return None kpts_conf when no model"


def test_pose_estimation_stub_accepts_minimal_args() -> None:
    """PoseEstimationBackend instantiates with keypoint_t_values only."""
    backend = PoseEstimationBackend(keypoint_t_values=_DEFAULT_T)
    assert backend is not None


def test_pose_estimation_stub_accepts_kwargs() -> None:
    """PoseEstimationBackend accepts all explicit kwargs (API compatibility)."""
    backend = PoseEstimationBackend(
        weights_path=None,
        device="cpu",
        n_points=15,
        n_keypoints=6,
        keypoint_t_values=_DEFAULT_T,
        confidence_floor=0.3,
        min_observed_keypoints=3,
        crop_size=(128, 64),
    )
    assert backend is not None


def test_pose_estimation_empty_crops_returns_empty_list() -> None:
    """process_batch with empty crops returns empty list."""
    backend = PoseEstimationBackend(keypoint_t_values=_DEFAULT_T)
    result = backend.process_batch([], [])
    assert result == []
