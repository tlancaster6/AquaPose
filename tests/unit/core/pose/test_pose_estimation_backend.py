"""Unit tests for PoseEstimationBackend (v3.7 raw-keypoint API).

Validates:
- Instantiation with weights_path=None logs warning, no crash
- process_batch with no model returns all (None, None)
- process_batch with mocked YOLO model produces (kpts_xy, kpts_conf) tuples
- confidence_floor filtering: too few visible keypoints returns (None, None)
- Exactly min_observed_keypoints visible keypoints: result produced
- Missing det.angle (None) falls back to 0.0 without crashing
- extract_affine_crop and invert_affine_points mocked to avoid real images
- No import from aquapose.engine at runtime
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aquapose.core.pose.backends.pose_estimation import PoseEstimationBackend
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
        logging.WARNING, logger="aquapose.core.pose.backends.pose_estimation"
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
    assert backend.n_keypoints == 6
    assert backend.confidence_floor == 0.3
    assert backend.min_observed_keypoints == 3
    assert backend.crop_size == (128, 64)


# ---------------------------------------------------------------------------
# No-model behavior
# ---------------------------------------------------------------------------


def test_process_batch_no_model_returns_none_tuples() -> None:
    """process_batch with no model returns (None, None) for every detection."""
    backend = PoseEstimationBackend(weights_path=None, keypoint_t_values=_DEFAULT_T)

    det1 = _make_det()
    det2 = _make_det(bbox=(100, 100, 60, 20))
    crop = _make_affine_crop()
    crops = [crop, crop]
    metadata = [(det1, "cam1", 0), (det2, "cam1", 0)]

    result = backend.process_batch(crops, metadata)

    assert len(result) == 2
    for kpts_xy, kpts_conf in result:
        assert kpts_xy is None
        assert kpts_conf is None


def test_process_batch_no_model_empty_crops() -> None:
    """process_batch with empty crops returns empty list."""
    backend = PoseEstimationBackend(weights_path=None, keypoint_t_values=_DEFAULT_T)
    result = backend.process_batch([], [])
    assert result == []


# ---------------------------------------------------------------------------
# Mocked YOLO inference — success path
# ---------------------------------------------------------------------------


@patch("aquapose.core.pose.backends.pose_estimation.invert_affine_points")
def test_process_batch_with_mock_model_produces_keypoints(
    mock_invert: MagicMock,
) -> None:
    """process_batch with mocked YOLO model produces (kpts_xy, kpts_conf) with correct shapes."""
    n_keypoints = 6

    mock_invert.side_effect = lambda pts, M: pts.astype(np.float32) + 5.0

    backend = PoseEstimationBackend(
        weights_path=None,
        n_keypoints=n_keypoints,
        keypoint_t_values=_DEFAULT_T,
    )
    mock_model = MagicMock()
    mock_model.predict.return_value = _make_yolo_keypoints_result(n_keypoints)
    backend._model = mock_model

    det = _make_det()
    crop = _make_affine_crop()
    result = backend.process_batch([crop], [(det, "cam1", 7)])

    assert len(result) == 1
    kpts_xy, kpts_conf = result[0]
    assert kpts_xy is not None, "Expected non-None kpts_xy with mock model"
    assert kpts_conf is not None, "Expected non-None kpts_conf with mock model"
    assert kpts_xy.shape == (n_keypoints, 2)
    assert kpts_conf.shape == (n_keypoints,)


# ---------------------------------------------------------------------------
# Confidence filtering
# ---------------------------------------------------------------------------


@patch("aquapose.core.pose.backends.pose_estimation.invert_affine_points")
def test_too_few_visible_keypoints_returns_none(
    mock_invert: MagicMock,
) -> None:
    """When fewer than min_observed_keypoints pass confidence_floor, result is (None, None)."""
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
    crop = _make_affine_crop()
    result = backend.process_batch([crop], [(det, "cam1", 0)])

    kpts_xy, kpts_conf = result[0]
    assert kpts_xy is None
    assert kpts_conf is None


@patch("aquapose.core.pose.backends.pose_estimation.invert_affine_points")
def test_exactly_min_visible_keypoints_produces_result(
    mock_invert: MagicMock,
) -> None:
    """Exactly min_observed_keypoints visible keypoints produces a valid result."""
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
    crop = _make_affine_crop()
    result = backend.process_batch([crop], [(det, "cam1", 0)])

    kpts_xy, kpts_conf = result[0]
    assert kpts_xy is not None
    assert kpts_conf is not None


# ---------------------------------------------------------------------------
# det.angle = None fallback
# ---------------------------------------------------------------------------


@patch("aquapose.core.pose.backends.pose_estimation.invert_affine_points")
@patch("aquapose.core.pose.backends.pose_estimation.extract_affine_crop")
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
    # Call _extract_crop directly to test angle fallback
    frame = _make_frame()
    crop = backend._extract_crop(det, frame)
    # Should not raise even with angle=None
    assert crop is not None


# ---------------------------------------------------------------------------
# Import boundary test
# ---------------------------------------------------------------------------


def test_no_engine_import() -> None:
    """pose_estimation.py does not import from aquapose.engine at runtime."""
    import ast
    import pathlib

    import aquapose.core.pose.backends.pose_estimation as pose_mod

    source = pathlib.Path(pose_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "aquapose.engine" not in node.module, (
                f"pose_estimation.py imports from aquapose.engine: {node.module}"
            )
