"""Unit tests for PoseEstimationBackend.

Validates:
- Instantiation with weights_path=None logs warning, no crash
- process_frame with no model returns all midline=None
- process_frame with mocked YOLO model produces Midline2D with correct shapes
- _keypoints_to_midline directly: output shape and endpoint values
- confidence_floor filtering: too few visible keypoints returns midline=None
- Exactly min_observed_keypoints visible keypoints: midline produced
- Missing det.angle (None) falls back to 0.0 without crashing
- extract_affine_crop and invert_affine_points mocked to avoid real images
- No import from aquapose.engine at runtime
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aquapose.core.midline.backends.pose_estimation import (
    PoseEstimationBackend,
    _keypoints_to_midline,
)
from aquapose.core.midline.types import AnnotatedDetection
from aquapose.core.types.crop import AffineCrop
from aquapose.core.types.detection import Detection

_DEFAULT_T = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_det(
    bbox: tuple[int, int, int, int] = (10, 10, 60, 20),
    angle: float | None = 0.0,
) -> Detection:
    """Create a minimal Detection for testing."""
    return Detection(
        bbox=bbox,
        mask=None,
        area=int(bbox[2] * bbox[3]),
        confidence=0.9,
        angle=angle,
    )


def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a dummy BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_affine_crop(crop_w: int = 128, crop_h: int = 64) -> AffineCrop:
    """Build an AffineCrop with an identity-like affine matrix."""
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    image = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    return AffineCrop(
        image=image, M=M, crop_size=(crop_w, crop_h), frame_shape=(480, 640)
    )


def _make_yolo_keypoints_result(
    n_keypoints: int = 6,
    confidences: np.ndarray | None = None,
) -> list[object]:
    """Build a mock YOLO-pose result with keypoints.

    Args:
        n_keypoints: Number of keypoints to produce.
        confidences: Per-keypoint confidence array, shape (n_keypoints,).
            Defaults to all-ones (fully confident).

    Returns:
        List with one mock result object that has .keypoints.xy and .keypoints.conf.
    """
    if confidences is None:
        confidences = np.ones(n_keypoints, dtype=np.float32)

    # Build evenly-spaced keypoints across a 64x128 crop
    xs = np.linspace(10.0, 118.0, n_keypoints, dtype=np.float32)
    ys = np.full(n_keypoints, 32.0, dtype=np.float32)
    kpts = np.stack([xs, ys], axis=1)  # (K, 2)

    def _tensor(arr: np.ndarray) -> MagicMock:
        t = MagicMock()
        t.cpu.return_value.numpy.return_value = arr
        return t

    # mock result[0].keypoints.xy[0].cpu().numpy() → (K, 2)
    mock_kp = MagicMock()
    mock_kp.xy = [_tensor(kpts)]  # xy[0].cpu().numpy() → (K, 2)
    mock_kp.conf = [_tensor(confidences)]  # conf[0].cpu().numpy() → (K,)

    mock_result = MagicMock()
    mock_result.keypoints = mock_kp

    return [mock_result]


# ---------------------------------------------------------------------------
# Instantiation tests
# ---------------------------------------------------------------------------


def test_instantiation_no_weights_path(caplog: pytest.LogCaptureFixture) -> None:
    """PoseEstimationBackend with weights_path=None logs warning, no crash."""
    import logging

    with caplog.at_level(
        logging.WARNING, logger="aquapose.core.midline.backends.pose_estimation"
    ):
        backend = PoseEstimationBackend(weights_path=None, keypoint_t_values=_DEFAULT_T)

    assert backend._model is None
    assert any("no weights_path" in msg for msg in caplog.messages)


def test_instantiation_nonexistent_weights() -> None:
    """Non-existent weights_path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="weights not found"):
        PoseEstimationBackend(
            weights_path="/nonexistent/path/model.pt", keypoint_t_values=_DEFAULT_T
        )


def test_instantiation_default_kwargs() -> None:
    """Default attribute values are stored correctly."""
    backend = PoseEstimationBackend(keypoint_t_values=_DEFAULT_T)
    assert backend.n_points == 15
    assert backend.n_keypoints == 6
    assert backend.confidence_floor == 0.3
    assert backend.min_observed_keypoints == 3
    assert backend.crop_size == (128, 64)
    assert len(backend.keypoint_t_values) == 6


def test_instantiation_custom_t_values() -> None:
    """Custom keypoint_t_values are stored correctly."""
    t = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    backend = PoseEstimationBackend(keypoint_t_values=t)
    np.testing.assert_allclose(backend._keypoint_t_values, t)


def test_instantiation_none_t_values_raises() -> None:
    """keypoint_t_values=None raises ValueError."""
    with pytest.raises(ValueError, match="keypoint_t_values"):
        PoseEstimationBackend(n_keypoints=4)


# ---------------------------------------------------------------------------
# _keypoints_to_midline unit tests
# ---------------------------------------------------------------------------


def test_keypoints_to_midline_output_shape() -> None:
    """_keypoints_to_midline returns correct output shapes."""
    n_kpts = 6
    n_points = 15
    kpts_xy = np.stack(
        [np.linspace(10.0, 100.0, n_kpts), np.full(n_kpts, 32.0)], axis=1
    ).astype(np.float32)
    t_values = np.linspace(0.0, 1.0, n_kpts, dtype=np.float32)
    confidences = np.ones(n_kpts, dtype=np.float32)

    xy, conf = _keypoints_to_midline(kpts_xy, t_values, confidences, n_points)

    assert xy.shape == (n_points, 2)
    assert conf.shape == (n_points,)
    assert xy.dtype == np.float32
    assert conf.dtype == np.float32


def test_keypoints_to_midline_endpoint_values() -> None:
    """Output endpoints match input endpoints when t spans [0, 1]."""
    n_kpts = 6
    kpts_xy = np.stack(
        [np.linspace(0.0, 100.0, n_kpts), np.zeros(n_kpts)], axis=1
    ).astype(np.float32)
    t_values = np.linspace(0.0, 1.0, n_kpts, dtype=np.float32)
    confidences = np.linspace(0.5, 1.0, n_kpts, dtype=np.float32)

    xy, _conf = _keypoints_to_midline(kpts_xy, t_values, confidences, 11)

    # First output point should be near first input point
    np.testing.assert_allclose(xy[0], kpts_xy[0], atol=0.5)
    # Last output point should be near last input point
    np.testing.assert_allclose(xy[-1], kpts_xy[-1], atol=0.5)


def test_keypoints_to_midline_monotone_x() -> None:
    """Interpolated x values are monotonically increasing for sorted inputs."""
    kpts_xy = np.stack([np.linspace(10.0, 90.0, 6), np.full(6, 32.0)], axis=1).astype(
        np.float32
    )
    t_values = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    confidences = np.ones(6, dtype=np.float32)

    xy, _ = _keypoints_to_midline(kpts_xy, t_values, confidences, 15)

    diffs = np.diff(xy[:, 0])
    assert np.all(diffs >= -1e-4), "x values should be monotonically increasing"


# ---------------------------------------------------------------------------
# No-model behavior
# ---------------------------------------------------------------------------


def test_process_frame_no_model_returns_none_midlines() -> None:
    """process_frame with no model returns midline=None for every detection."""
    backend = PoseEstimationBackend(weights_path=None, keypoint_t_values=_DEFAULT_T)

    det1 = _make_det()
    det2 = _make_det(bbox=(100, 100, 60, 20))
    frame_dets: dict[str, list[Detection]] = {"cam1": [det1, det2]}
    frames: dict[str, np.ndarray] = {"cam1": _make_frame()}

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
        assert ann.midline is None


def test_process_frame_no_model_empty_camera() -> None:
    """process_frame with no detections returns empty list."""
    backend = PoseEstimationBackend(weights_path=None, keypoint_t_values=_DEFAULT_T)
    result = backend.process_frame(
        frame_idx=0,
        frame_dets={"cam1": []},
        frames={"cam1": _make_frame()},
        camera_ids=["cam1"],
    )
    assert result["cam1"] == []


# ---------------------------------------------------------------------------
# Mocked YOLO inference — success path
# ---------------------------------------------------------------------------


@patch("aquapose.core.midline.backends.pose_estimation.invert_affine_points")
@patch("aquapose.core.midline.backends.pose_estimation.extract_affine_crop")
def test_process_frame_with_mock_model_produces_midline(
    mock_extract: MagicMock,
    mock_invert: MagicMock,
) -> None:
    """process_frame with mocked YOLO model produces AnnotatedDetection with Midline2D."""
    n_points = 15
    n_keypoints = 6
    crop_w, crop_h = 128, 64

    affine_crop = _make_affine_crop(crop_w, crop_h)
    mock_extract.return_value = affine_crop
    mock_invert.side_effect = lambda pts, M: pts.astype(np.float32) + 5.0

    backend = PoseEstimationBackend(
        weights_path=None,
        n_points=n_points,
        n_keypoints=n_keypoints,
        keypoint_t_values=_DEFAULT_T,
    )
    mock_model = MagicMock()
    mock_model.predict.return_value = _make_yolo_keypoints_result(n_keypoints)
    backend._model = mock_model

    det = _make_det()
    result = backend.process_frame(
        frame_idx=7,
        frame_dets={"cam1": [det]},
        frames={"cam1": _make_frame()},
        camera_ids=["cam1"],
    )

    ann = result["cam1"][0]
    assert isinstance(ann, AnnotatedDetection)
    assert ann.midline is not None, "Expected non-None midline with mock model"
    assert ann.midline.points.shape == (n_points, 2)
    assert ann.midline.half_widths.shape == (n_points,)
    assert ann.midline.point_confidence is not None
    assert ann.midline.point_confidence.shape == (n_points,)
    assert ann.midline.camera_id == "cam1"
    assert ann.midline.frame_index == 7


# ---------------------------------------------------------------------------
# Confidence filtering
# ---------------------------------------------------------------------------


@patch("aquapose.core.midline.backends.pose_estimation.invert_affine_points")
@patch("aquapose.core.midline.backends.pose_estimation.extract_affine_crop")
def test_too_few_visible_keypoints_returns_none(
    mock_extract: MagicMock,
    mock_invert: MagicMock,
) -> None:
    """When fewer than min_observed_keypoints pass confidence_floor, midline is None."""
    affine_crop = _make_affine_crop()
    mock_extract.return_value = affine_crop
    mock_invert.side_effect = lambda pts, M: pts.astype(np.float32)

    # 4 of 6 keypoints below floor, only 2 visible < min_observed(3)
    confidences = np.array([0.05, 0.05, 0.05, 0.05, 0.9, 0.9], dtype=np.float32)

    backend = PoseEstimationBackend(
        weights_path=None,
        confidence_floor=0.3,
        min_observed_keypoints=3,
        keypoint_t_values=_DEFAULT_T,
    )
    mock_model = MagicMock()
    mock_model.predict.return_value = _make_yolo_keypoints_result(6, confidences)
    backend._model = mock_model

    det = _make_det()
    result = backend.process_frame(
        frame_idx=0,
        frame_dets={"cam1": [det]},
        frames={"cam1": _make_frame()},
        camera_ids=["cam1"],
    )

    assert result["cam1"][0].midline is None


@patch("aquapose.core.midline.backends.pose_estimation.invert_affine_points")
@patch("aquapose.core.midline.backends.pose_estimation.extract_affine_crop")
def test_exactly_min_visible_keypoints_produces_midline(
    mock_extract: MagicMock,
    mock_invert: MagicMock,
) -> None:
    """Exactly min_observed_keypoints visible keypoints produces a midline."""
    affine_crop = _make_affine_crop()
    mock_extract.return_value = affine_crop
    mock_invert.side_effect = lambda pts, M: pts.astype(np.float32)

    # Exactly 3 keypoints above floor (min_observed=3)
    confidences = np.array([0.05, 0.05, 0.05, 0.9, 0.9, 0.9], dtype=np.float32)

    backend = PoseEstimationBackend(
        weights_path=None,
        confidence_floor=0.3,
        min_observed_keypoints=3,
        keypoint_t_values=_DEFAULT_T,
    )
    mock_model = MagicMock()
    mock_model.predict.return_value = _make_yolo_keypoints_result(6, confidences)
    backend._model = mock_model

    det = _make_det()
    result = backend.process_frame(
        frame_idx=0,
        frame_dets={"cam1": [det]},
        frames={"cam1": _make_frame()},
        camera_ids=["cam1"],
    )

    assert result["cam1"][0].midline is not None


# ---------------------------------------------------------------------------
# det.angle = None fallback
# ---------------------------------------------------------------------------


@patch("aquapose.core.midline.backends.pose_estimation.invert_affine_points")
@patch("aquapose.core.midline.backends.pose_estimation.extract_affine_crop")
def test_none_angle_falls_back_to_zero(
    mock_extract: MagicMock,
    mock_invert: MagicMock,
) -> None:
    """Detection with angle=None does not crash — extract_affine_crop called with 0.0."""
    affine_crop = _make_affine_crop()
    mock_extract.return_value = affine_crop
    mock_invert.side_effect = lambda pts, M: pts.astype(np.float32)

    backend = PoseEstimationBackend(weights_path=None, keypoint_t_values=_DEFAULT_T)
    mock_model = MagicMock()
    mock_model.predict.return_value = _make_yolo_keypoints_result(6)
    backend._model = mock_model

    det = _make_det(angle=None)
    result = backend.process_frame(
        frame_idx=0,
        frame_dets={"cam1": [det]},
        frames={"cam1": _make_frame()},
        camera_ids=["cam1"],
    )

    assert "cam1" in result
    # Verify extract_affine_crop was called with angle_math_rad=0.0
    assert mock_extract.called
    call_kwargs = mock_extract.call_args
    angle_passed = call_kwargs.kwargs.get("angle_math_rad")
    assert angle_passed == 0.0, (
        f"Expected angle_math_rad=0.0, got {angle_passed!r}; kwargs={call_kwargs.kwargs}"
    )


# ---------------------------------------------------------------------------
# _keypoints_to_midline NaN-out extrapolation tests
# ---------------------------------------------------------------------------


def test_keypoints_to_midline_nan_outside_visible_range() -> None:
    """Points outside the visible keypoint t-range are NaN with confidence 0."""
    # 4 keypoints at t_values = [0.2, 0.4, 0.6, 0.8] (nose and tail dropped)
    n_kpts = 4
    n_points = 15
    t_values = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    kpts_xy = np.stack(
        [np.linspace(20.0, 80.0, n_kpts), np.full(n_kpts, 32.0)], axis=1
    ).astype(np.float32)
    confidences = np.ones(n_kpts, dtype=np.float32)

    xy, conf = _keypoints_to_midline(kpts_xy, t_values, confidences, n_points)

    # t_eval = linspace(0, 1, 15): indices 0..14, step ~0.0714
    # t < 0.2: indices where t_eval < 0.2 → indices 0, 1 (t=0.0, 0.0714)
    # t > 0.8: indices where t_eval > 0.8 → indices 13, 14 (t=0.857, 1.0)
    t_eval = np.linspace(0.0, 1.0, n_points)
    outside = (t_eval < 0.2) | (t_eval > 0.8)
    inside = ~outside

    # Outside range: NaN xy and confidence 0
    assert np.all(np.isnan(xy[outside, 0])), "x outside range should be NaN"
    assert np.all(np.isnan(xy[outside, 1])), "y outside range should be NaN"
    assert np.all(conf[outside] == 0.0), "confidence outside range should be 0"

    # Inside range: finite values and positive confidence
    assert np.all(np.isfinite(xy[inside, 0])), "x inside range should be finite"
    assert np.all(np.isfinite(xy[inside, 1])), "y inside range should be finite"
    assert np.all(conf[inside] > 0.0), "confidence inside range should be positive"


def test_keypoints_to_midline_full_range_no_nan() -> None:
    """When t spans [0, 1], no NaN in output — all points finite with positive confidence."""
    n_kpts = 6
    n_points = 15
    t_values = np.linspace(0.0, 1.0, n_kpts, dtype=np.float32)
    kpts_xy = np.stack(
        [np.linspace(10.0, 100.0, n_kpts), np.full(n_kpts, 32.0)], axis=1
    ).astype(np.float32)
    confidences = np.ones(n_kpts, dtype=np.float32)

    xy, conf = _keypoints_to_midline(kpts_xy, t_values, confidences, n_points)

    assert not np.any(np.isnan(xy)), "No NaN expected when t spans [0, 1]"
    assert np.all(np.isfinite(xy)), "All xy should be finite"
    assert np.all(conf > 0.0), "All confidence should be positive"


def test_keypoints_to_midline_tail_only_dropped() -> None:
    """When tail is dropped (t spans [0.0, 0.6]), only points beyond t=0.6 are NaN."""
    n_kpts = 4
    n_points = 15
    t_values = np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float32)
    kpts_xy = np.stack(
        [np.linspace(10.0, 70.0, n_kpts), np.full(n_kpts, 32.0)], axis=1
    ).astype(np.float32)
    confidences = np.ones(n_kpts, dtype=np.float32)

    xy, conf = _keypoints_to_midline(kpts_xy, t_values, confidences, n_points)

    t_eval = np.linspace(0.0, 1.0, n_points)
    outside = t_eval > 0.6
    inside = ~outside

    # Points beyond t=0.6 should be NaN
    assert np.all(np.isnan(xy[outside, 0])), "x beyond t=0.6 should be NaN"
    assert np.all(np.isnan(xy[outside, 1])), "y beyond t=0.6 should be NaN"
    assert np.all(conf[outside] == 0.0), "confidence beyond t=0.6 should be 0"

    # Points within [0.0, 0.6] should be finite
    assert np.all(np.isfinite(xy[inside, 0])), "x within range should be finite"
    assert np.all(np.isfinite(xy[inside, 1])), "y within range should be finite"


def test_keypoints_to_midline_shape_preserved() -> None:
    """Output shape is (n_points, 2) and (n_points,) regardless of NaN presence."""
    n_kpts = 3
    n_points = 11
    t_values = np.array([0.3, 0.5, 0.7], dtype=np.float32)
    kpts_xy = np.stack(
        [np.linspace(30.0, 70.0, n_kpts), np.full(n_kpts, 32.0)], axis=1
    ).astype(np.float32)
    confidences = np.ones(n_kpts, dtype=np.float32)

    xy, conf = _keypoints_to_midline(kpts_xy, t_values, confidences, n_points)

    assert xy.shape == (n_points, 2), f"Expected ({n_points}, 2), got {xy.shape}"
    assert conf.shape == (n_points,), f"Expected ({n_points},), got {conf.shape}"
    assert xy.dtype == np.float32
    assert conf.dtype == np.float32


# ---------------------------------------------------------------------------
# Import boundary test
# ---------------------------------------------------------------------------


def test_no_engine_import() -> None:
    """pose_estimation.py does not import from aquapose.engine at runtime."""
    import ast
    import pathlib

    import aquapose.core.midline.backends.pose_estimation as pose_mod

    source = pathlib.Path(pose_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "aquapose.engine" not in node.module, (
                f"pose_estimation.py imports from aquapose.engine: {node.module}"
            )
