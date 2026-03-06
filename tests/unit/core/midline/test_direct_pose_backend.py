"""Unit tests for PoseEstimationBackend (no-op stub).

Validates:
- PoseEstimationBackend instantiates without error and without loading any model
- process_frame returns AnnotatedDetection(midline=None) for every detection
- process_frame signature matches what MidlineStage expects
"""

from __future__ import annotations

import numpy as np

from aquapose.core.midline.backends.pose_estimation import PoseEstimationBackend
from aquapose.core.midline.types import AnnotatedDetection
from aquapose.core.types.detection import Detection

_DEFAULT_T = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def test_pose_estimation_stub_returns_none_midlines() -> None:
    """PoseEstimationBackend stub instantiates cleanly and returns midline=None."""
    backend = PoseEstimationBackend(keypoint_t_values=_DEFAULT_T)

    det1 = Detection(bbox=(10, 10, 50, 50), mask=None, area=2500, confidence=0.9)
    det2 = Detection(bbox=(60, 60, 30, 30), mask=None, area=900, confidence=0.8)
    frame_dets: dict[str, list[Detection]] = {"cam1": [det1, det2]}
    frames: dict[str, np.ndarray] = {"cam1": np.zeros((480, 640, 3), dtype=np.uint8)}

    result = backend.process_frame(
        frame_idx=0,
        frame_dets=frame_dets,
        frames=frames,
        camera_ids=["cam1"],
    )

    assert "cam1" in result
    assert len(result["cam1"]) == 2
    for ann in result["cam1"]:
        assert isinstance(ann, AnnotatedDetection)
        assert ann.midline is None, "Stub must return midline=None for all detections"


def test_pose_estimation_stub_accepts_minimal_args() -> None:
    """PoseEstimationBackend instantiates with keypoint_t_values only."""
    backend = PoseEstimationBackend(keypoint_t_values=_DEFAULT_T)
    assert backend is not None


def test_pose_estimation_stub_accepts_kwargs() -> None:
    """PoseEstimationBackend accepts all explicit kwargs (API compatibility).

    Uses weights_path=None since non-existent paths now raise FileNotFoundError.
    """
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


def test_pose_estimation_stub_empty_camera() -> None:
    """process_frame handles cameras with no detections gracefully."""
    import numpy as np

    backend = PoseEstimationBackend(keypoint_t_values=_DEFAULT_T)
    frame_dets: dict[str, list[Detection]] = {"cam1": [], "cam2": []}
    frames: dict[str, np.ndarray] = {
        "cam1": np.zeros((480, 640, 3), dtype=np.uint8),
        "cam2": np.zeros((480, 640, 3), dtype=np.uint8),
    }

    result = backend.process_frame(
        frame_idx=5,
        frame_dets=frame_dets,
        frames=frames,
        camera_ids=["cam1", "cam2"],
    )

    assert result["cam1"] == []
    assert result["cam2"] == []


def test_pose_estimation_stub_annotated_detection_fields() -> None:
    """AnnotatedDetection objects have correct camera_id and frame_index."""
    backend = PoseEstimationBackend(keypoint_t_values=_DEFAULT_T)

    det = Detection(bbox=(0, 0, 100, 100), mask=None, area=10000, confidence=0.95)
    frame_dets: dict[str, list[Detection]] = {"camA": [det]}
    frames: dict[str, np.ndarray] = {"camA": np.zeros((480, 640, 3), dtype=np.uint8)}

    result = backend.process_frame(
        frame_idx=42,
        frame_dets=frame_dets,
        frames=frames,
        camera_ids=["camA"],
    )

    ann = result["camA"][0]
    assert ann.camera_id == "camA"
    assert ann.frame_index == 42
    assert ann.detection is det
    assert ann.mask is None
    assert ann.crop_region is None
