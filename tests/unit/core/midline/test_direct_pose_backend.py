"""Unit tests for DirectPoseBackend (no-op stub).

Validates:
- DirectPoseBackend instantiates without error and without loading any model
- process_frame returns AnnotatedDetection(midline=None) for every detection
- process_frame signature matches what MidlineStage expects
"""

from __future__ import annotations

import numpy as np

from aquapose.core.midline.backends.direct_pose import DirectPoseBackend
from aquapose.core.midline.types import AnnotatedDetection
from aquapose.segmentation.detector import Detection


def test_direct_pose_stub_returns_none_midlines() -> None:
    """DirectPoseBackend stub instantiates cleanly and returns midline=None."""
    backend = DirectPoseBackend()

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


def test_direct_pose_stub_accepts_no_args() -> None:
    """DirectPoseBackend instantiates with no arguments."""
    backend = DirectPoseBackend()
    assert backend is not None


def test_direct_pose_stub_accepts_kwargs() -> None:
    """DirectPoseBackend accepts and ignores all kwargs (API compatibility)."""
    backend = DirectPoseBackend(
        weights_path="nonexistent.pth",
        device="cpu",
        n_points=15,
        n_keypoints=6,
        confidence_floor=0.1,
        min_observed_keypoints=3,
        crop_size=(128, 64),
    )
    assert backend is not None


def test_direct_pose_stub_empty_camera() -> None:
    """process_frame handles cameras with no detections gracefully."""
    import numpy as np

    backend = DirectPoseBackend()
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


def test_direct_pose_stub_annotated_detection_fields() -> None:
    """AnnotatedDetection objects have correct camera_id and frame_index."""
    backend = DirectPoseBackend()

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
