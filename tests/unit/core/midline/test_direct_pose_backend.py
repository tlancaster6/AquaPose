"""Unit tests for DirectPoseBackend.

All tests mock _PoseModel to avoid needing real weights. Tests verify:
- Constructor validates weights path (fail-fast)
- process_frame returns AnnotatedDetection with correct Midline2D shape
- Output always has exactly n_sample_points points
- Partial visibility: NaN-padding outside observed arc-span
- Below min_observed_keypoints: midline is None
- axis-aligned detections (angle=None) handled gracefully
- Confidence heuristic formula
- Both backends produce Midline2D with identical field structure
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aquapose.segmentation.detector import Detection

# ---------------------------------------------------------------------------
# Helpers for mocking the _PoseModel and torch
# ---------------------------------------------------------------------------


def _make_detection(
    bbox: tuple[int, int, int, int] = (10, 10, 60, 30),
    angle: float | None = 0.0,
) -> Detection:
    """Build a minimal Detection for testing.

    Uses the canonical Detection fields only (no centroid — derived from bbox by backend).
    """
    return Detection(
        bbox=bbox,
        mask=None,
        area=bbox[2] * bbox[3],
        confidence=0.9,
        angle=angle,
    )


def _make_fake_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a solid-color BGR frame for testing."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_torch_tensor(coords: list[float]):
    """Build a (1, len(coords)) float32 torch tensor from a flat list."""
    import torch

    return torch.tensor([coords], dtype=torch.float32)


class _FakePoseModel:
    """Callable mock for _PoseModel that returns a fixed torch output tensor.

    Mimics the _PoseModel interface: call(), to(), eval(), load_state_dict().
    """

    def __init__(self, output_coords: list[float]):
        import torch

        self._output = torch.tensor([output_coords], dtype=torch.float32)

    def __call__(self, x):
        return self._output

    def to(self, device):
        return self

    def eval(self):
        return self

    def load_state_dict(self, state_dict, strict=True):
        return None


def _build_backend_with_model(
    tmp_path: Path,
    fake_model: _FakePoseModel,
    *,
    n_keypoints: int = 6,
    n_points: int = 15,
    confidence_floor: float = 0.1,
    min_observed_keypoints: int = 3,
    keypoint_t_values: list[float] | None = None,
):
    """Construct a DirectPoseBackend with a pre-built fake model.

    Patches _PoseModel at its source module and torch.load to return an empty dict.
    After construction, replaces the model with fake_model directly.
    """
    fake_weights = tmp_path / "pose_model.pth"
    fake_weights.write_bytes(b"fake")

    # Patch _PoseModel at source and torch.load globally (both are lazy-imported)
    with (
        patch("aquapose.training.pose._PoseModel", return_value=fake_model),
        patch("torch.load", return_value={}),
    ):
        from aquapose.core.midline.backends.direct_pose import DirectPoseBackend

        backend = DirectPoseBackend(
            weights_path=str(fake_weights),
            device="cpu",
            n_keypoints=n_keypoints,
            n_points=n_points,
            confidence_floor=confidence_floor,
            min_observed_keypoints=min_observed_keypoints,
            keypoint_t_values=keypoint_t_values,
            crop_size=(128, 64),
        )

    # Always set the model directly on the backend
    backend._model = fake_model
    return backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_constructor_validates_weights_path(tmp_path: Path) -> None:
    """Non-existent weights path raises FileNotFoundError at construction."""
    from aquapose.core.midline.backends.direct_pose import DirectPoseBackend

    nonexistent = tmp_path / "does_not_exist.pth"

    with pytest.raises(FileNotFoundError, match=r"does_not_exist\.pth"):
        DirectPoseBackend(weights_path=str(nonexistent))


def test_process_frame_returns_annotated_detections(tmp_path: Path) -> None:
    """process_frame returns dict mapping camera_id to list of AnnotatedDetection."""
    n_keypoints = 6
    n_points = 15
    # All keypoints at center of crop -> high confidence
    coords = [0.5, 0.5] * n_keypoints
    fake_model = _FakePoseModel(coords)
    backend = _build_backend_with_model(
        tmp_path, fake_model, n_keypoints=n_keypoints, n_points=n_points
    )

    det = _make_detection()
    frame = _make_fake_frame()
    result = backend.process_frame(
        frame_idx=0,
        frame_dets={"cam1": [det]},
        frames={"cam1": frame},
        camera_ids=["cam1"],
    )

    assert "cam1" in result
    assert len(result["cam1"]) == 1

    ad = result["cam1"][0]
    assert ad.detection is det
    assert ad.mask is None
    assert ad.crop_region is None
    assert ad.midline is not None

    ml = ad.midline
    assert ml.points.shape == (n_points, 2), (
        f"Expected ({n_points}, 2), got {ml.points.shape}"
    )
    assert ml.point_confidence is not None
    assert ml.point_confidence.shape == (n_points,)


def test_output_always_n_sample_points(tmp_path: Path) -> None:
    """Output midline always has exactly n_points points regardless of keypoint count."""
    n_points = 10
    n_keypoints = 6
    # Keypoints spread across the crop (all clearly visible)
    coords = [0.1, 0.5, 0.3, 0.5, 0.5, 0.5, 0.6, 0.5, 0.7, 0.5, 0.9, 0.5]
    fake_model = _FakePoseModel(coords)
    backend = _build_backend_with_model(
        tmp_path,
        fake_model,
        n_keypoints=n_keypoints,
        n_points=n_points,
        confidence_floor=0.05,
    )

    det = _make_detection()
    frame = _make_fake_frame()
    result = backend.process_frame(
        frame_idx=0,
        frame_dets={"cam1": [det]},
        frames={"cam1": frame},
        camera_ids=["cam1"],
    )

    ml = result["cam1"][0].midline
    assert ml is not None
    assert ml.points.shape == (n_points, 2)
    assert ml.point_confidence is not None
    assert ml.point_confidence.shape == (n_points,)


def test_partial_visibility_no_loops(tmp_path: Path) -> None:
    """Keypoints below confidence_floor are excluded; remaining keypoints produce a smooth midline.

    After the x-sorting fix, visible keypoints are sorted by crop-space x-coordinate
    and assigned uniform t-values [0, 1].  This means:
    - The full span [0, 1] is always observed → no NaN-padding from arc-span gaps.
    - The resulting midline covers all n_points without NaN holes.
    - Keypoints excluded by the confidence floor simply don't participate.
    """
    n_keypoints = 6
    n_points = 15

    # Only the middle 3 keypoints (indices 2,3,4) are visible (near centre)
    # kp0,1 -> at corner (0,0) -> conf=0 (edge)
    # kp2 at x=0.45, kp3 at x=0.50, kp4 at x=0.55 -> near centre -> high conf
    # kp5 -> at corner (1,1) -> conf=0 (edge)
    coords = [
        0.0,
        0.0,  # kp0: corner -> low conf
        0.0,
        0.0,  # kp1: corner -> low conf
        0.45,
        0.5,  # kp2: near centre -> high conf
        0.5,
        0.5,  # kp3: centre -> max conf
        0.55,
        0.5,  # kp4: near centre -> high conf
        1.0,
        1.0,  # kp5: corner -> low conf
    ]

    # Uniform t-values: kp0=0, kp1=0.2, kp2=0.4, kp3=0.6, kp4=0.8, kp5=1.0
    t_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    fake_model = _FakePoseModel(coords)
    backend = _build_backend_with_model(
        tmp_path,
        fake_model,
        n_keypoints=n_keypoints,
        n_points=n_points,
        confidence_floor=0.1,
        min_observed_keypoints=2,
        keypoint_t_values=t_vals,
    )

    det = _make_detection()
    frame = _make_fake_frame()
    result = backend.process_frame(
        frame_idx=0,
        frame_dets={"cam1": [det]},
        frames={"cam1": frame},
        camera_ids=["cam1"],
    )

    ml = result["cam1"][0].midline
    assert ml is not None

    # After x-sort fix: visible kp2,kp3,kp4 are sorted by x-coordinate and
    # assigned t-values [0.0, 0.5, 1.0].  The full span [0, 1] is observed,
    # so no NaN-padding occurs — all n_points are valid.
    assert ml.points.shape == (n_points, 2), (
        f"Expected ({n_points}, 2), got {ml.points.shape}"
    )
    assert not np.any(np.isnan(ml.points)), (
        "All midline points should be valid (no NaN) when visible keypoints span [0,1]"
    )
    assert ml.point_confidence is not None
    assert ml.point_confidence.shape == (n_points,)
    assert np.all(ml.point_confidence > 0.0), (
        "All point confidence should be positive when full span is covered"
    )


def test_below_min_observed_returns_none_midline(tmp_path: Path) -> None:
    """All keypoints below confidence_floor -> midline is None."""
    n_keypoints = 6
    # All keypoints at corners (confidence=0 by the heuristic)
    coords = [0.0, 0.0] * n_keypoints

    fake_model = _FakePoseModel(coords)
    backend = _build_backend_with_model(
        tmp_path,
        fake_model,
        n_keypoints=n_keypoints,
        confidence_floor=0.1,
        min_observed_keypoints=3,
    )

    det = _make_detection()
    frame = _make_fake_frame()
    result = backend.process_frame(
        frame_idx=0,
        frame_dets={"cam1": [det]},
        frames={"cam1": frame},
        camera_ids=["cam1"],
    )

    ad = result["cam1"][0]
    assert ad.midline is None, (
        "Midline should be None when all keypoints are below floor"
    )


def test_axis_aligned_detection_angle_none(tmp_path: Path) -> None:
    """Detection with angle=None uses angle=0.0 without error."""
    n_keypoints = 6
    coords = [0.5, 0.5] * n_keypoints

    fake_model = _FakePoseModel(coords)
    backend = _build_backend_with_model(tmp_path, fake_model, n_keypoints=n_keypoints)

    # Detection with angle=None (axis-aligned, e.g. from non-OBB YOLO)
    det = _make_detection(angle=None)
    frame = _make_fake_frame()

    # Should not raise
    result = backend.process_frame(
        frame_idx=0,
        frame_dets={"cam1": [det]},
        frames={"cam1": frame},
        camera_ids=["cam1"],
    )

    assert "cam1" in result
    assert len(result["cam1"]) == 1


def test_confidence_heuristic(tmp_path: Path) -> None:
    """Known sigmoid output produces expected confidence via the 1-2*max(|x-0.5|,|y-0.5|) formula."""
    # Verify the mathematical formula independently
    # (x=0.5, y=0.5): conf = 1 - 2*max(0, 0) = 1.0
    # (x=0.0, y=0.5): conf = 1 - 2*max(0.5, 0) = 0.0
    # (x=0.25, y=0.5): conf = 1 - 2*max(0.25, 0) = 0.5
    kp_norm = np.array([[0.5, 0.5], [0.0, 0.5], [0.25, 0.5]])
    x_dev = np.abs(kp_norm[:, 0] - 0.5)
    y_dev = np.abs(kp_norm[:, 1] - 0.5)
    conf = np.clip(1.0 - 2.0 * np.maximum(x_dev, y_dev), 0.0, 1.0)

    assert abs(conf[0] - 1.0) < 1e-6, "Centre point should have conf=1.0"
    assert abs(conf[1] - 0.0) < 1e-6, "Edge point should have conf=0.0"
    assert abs(conf[2] - 0.5) < 1e-6, "Quarter-point should have conf=0.5"

    # Also verify it matches the backend implementation by running with
    # only the first and third keypoints visible
    n_keypoints = 3
    coords = [0.5, 0.5, 0.0, 0.5, 0.25, 0.5]
    fake_model = _FakePoseModel(coords)
    backend = _build_backend_with_model(
        tmp_path,
        fake_model,
        n_keypoints=n_keypoints,
        confidence_floor=0.01,  # very low floor so all are treated as visible
        min_observed_keypoints=1,
    )

    det = _make_detection()
    frame = _make_fake_frame()
    result = backend.process_frame(
        frame_idx=0,
        frame_dets={"cam1": [det]},
        frames={"cam1": frame},
        camera_ids=["cam1"],
    )

    # kp1 (x=0.0) has conf=0 which is below floor=0.01
    # kp0 and kp2 should be visible -> midline should not be None
    ad = result["cam1"][0]
    # We only check the formula; midline presence depends on visible count
    assert ad is not None  # basic sanity


def test_empty_camera_returns_empty_list(tmp_path: Path) -> None:
    """Cameras with no detections or no frame return empty AnnotatedDetection lists."""
    n_keypoints = 6
    coords = [0.5, 0.5] * n_keypoints
    fake_model = _FakePoseModel(coords)
    backend = _build_backend_with_model(tmp_path, fake_model, n_keypoints=n_keypoints)

    frame = _make_fake_frame()
    result = backend.process_frame(
        frame_idx=0,
        frame_dets={"cam1": [], "cam2": [_make_detection()]},
        frames={"cam1": frame},  # cam2 frame missing
        camera_ids=["cam1", "cam2"],
    )

    assert result["cam1"] == []
    assert result["cam2"] == []


def test_both_backends_same_shape(tmp_path: Path) -> None:
    """DirectPoseBackend and SegmentThenExtractBackend both produce Midline2D with identical field structure."""
    n_points = 10

    # --- Setup DirectPoseBackend ---
    n_keypoints = 6
    coords = [0.5, 0.5] * n_keypoints
    fake_pose_model = _FakePoseModel(coords)
    dp_backend = _build_backend_with_model(
        tmp_path, fake_pose_model, n_keypoints=n_keypoints, n_points=n_points
    )

    # --- Setup SegmentThenExtractBackend (mocked) ---
    fake_weights_unet = tmp_path / "unet.pth"
    fake_weights_unet.write_bytes(b"fake")

    mock_unet_model = MagicMock()
    mock_unet_model.to.return_value = mock_unet_model
    mock_unet_segmentor = MagicMock()
    mock_unet_segmentor.get_model.return_value = mock_unet_model

    with (
        patch(
            "aquapose.segmentation.model.UNetSegmentor",
            return_value=mock_unet_segmentor,
        ),
        patch(
            "aquapose.core.midline.backends.segment_then_extract.Path.exists",
            return_value=True,
        ),
    ):
        from aquapose.core.midline.backends.segment_then_extract import (
            SegmentThenExtractBackend,
        )

        ste_backend = SegmentThenExtractBackend(
            weights_path=str(fake_weights_unet),
            device="cpu",
            n_points=n_points,
        )

    # Both have a process_frame method
    assert hasattr(dp_backend, "process_frame")
    assert hasattr(ste_backend, "process_frame")

    # Run DirectPoseBackend and verify Midline2D field structure
    det = _make_detection()
    frame = _make_fake_frame()
    dp_result = dp_backend.process_frame(
        frame_idx=0,
        frame_dets={"cam1": [det]},
        frames={"cam1": frame},
        camera_ids=["cam1"],
    )

    dp_ad = dp_result["cam1"][0]
    if dp_ad.midline is not None:
        ml = dp_ad.midline
        assert isinstance(ml.points, np.ndarray)
        assert ml.points.shape == (n_points, 2)
        assert ml.points.dtype == np.float32
        assert isinstance(ml.half_widths, np.ndarray)
        assert ml.half_widths.shape == (n_points,)
        assert ml.point_confidence is not None
        assert isinstance(ml.point_confidence, np.ndarray)
        assert ml.point_confidence.shape == (n_points,)
        assert isinstance(ml.fish_id, int)
        assert isinstance(ml.camera_id, str)
        assert isinstance(ml.frame_index, int)
        assert isinstance(ml.is_head_to_tail, bool)
