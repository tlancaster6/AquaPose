"""Unit tests for SegmentationBackend.

Validates:
- Instantiation with weights_path=None logs warning, no crash
- process_frame with no model returns all midline=None
- process_frame with mocked YOLO model produces AnnotatedDetection with Midline2D
- Mask area below min_area returns midline=None
- Missing det.angle falls back to 0.0 without crashing
- extract_affine_crop and invert_affine_points are mocked to avoid real images
- No import from aquapose.engine at runtime
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aquapose.core.midline.backends.segmentation import SegmentationBackend
from aquapose.core.midline.types import AnnotatedDetection
from aquapose.core.types.crop import AffineCrop
from aquapose.core.types.detection import Detection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_det(
    bbox: tuple[int, int, int, int] = (10, 10, 60, 20),
    angle: float | None = 0.0,
    obb_points: np.ndarray | None = None,
) -> Detection:
    """Create a minimal Detection for testing."""
    return Detection(
        bbox=bbox,
        mask=None,
        area=int(bbox[2] * bbox[3]),
        confidence=0.9,
        angle=angle,
        obb_points=obb_points,
    )


def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a dummy BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_affine_crop(crop_w: int = 128, crop_h: int = 64) -> AffineCrop:
    """Build an AffineCrop with an identity-like affine matrix for testing."""
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    image = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    return AffineCrop(
        image=image, M=M, crop_size=(crop_w, crop_h), frame_shape=(480, 640)
    )


def _make_yolo_masks_result(
    crop_h: int, crop_w: int, area_fraction: float = 0.5
) -> list[object]:
    """Build a mock YOLO result with a single mask tensor of given fill fraction.

    Args:
        crop_h: Mask height in pixels.
        crop_w: Mask width in pixels.
        area_fraction: Fraction of mask pixels set to 1.0 (filled).

    Returns:
        List with one mock result object that has .masks.data.
    """
    # Create a mask that is a thin horizontal rectangle to yield a valid skeleton
    mask_data = np.zeros((crop_h, crop_w), dtype=np.float32)
    # Fill a tall-enough central stripe so skeletonization works
    row_start = crop_h // 4
    row_end = 3 * crop_h // 4
    mask_data[row_start:row_end, 4 : crop_w - 4] = 1.0

    # Wrap in a mock tensor
    mock_tensor = MagicMock()
    mock_tensor.cpu.return_value.numpy.return_value = mask_data

    mock_masks = MagicMock()
    mock_masks.data = [mock_tensor]

    mock_result = MagicMock()
    mock_result.masks = mock_masks

    return [mock_result]


# ---------------------------------------------------------------------------
# Instantiation tests
# ---------------------------------------------------------------------------


def test_instantiation_no_weights_path(caplog: pytest.LogCaptureFixture) -> None:
    """SegmentationBackend with weights_path=None logs warning, no crash."""
    import logging

    with caplog.at_level(
        logging.WARNING, logger="aquapose.core.midline.backends.segmentation"
    ):
        backend = SegmentationBackend(weights_path=None)

    assert backend._model is None
    assert any("no weights_path" in msg for msg in caplog.messages)


def test_instantiation_nonexistent_weights() -> None:
    """SegmentationBackend with a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="weights not found"):
        SegmentationBackend(weights_path="/nonexistent/path/model.pt")


def test_instantiation_default_kwargs() -> None:
    """Default attribute values are stored correctly."""
    backend = SegmentationBackend()
    assert backend.confidence_threshold == 0.5
    assert backend.n_points == 15
    assert backend.min_area == 300
    assert backend.device == "cuda"
    assert backend.crop_size == (128, 64)


# ---------------------------------------------------------------------------
# No-model behavior
# ---------------------------------------------------------------------------


def test_process_frame_no_model_returns_none_midlines() -> None:
    """process_frame with no model returns midline=None for every detection."""
    backend = SegmentationBackend(weights_path=None)

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
    backend = SegmentationBackend(weights_path=None)
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


@patch("aquapose.core.midline.backends.segmentation.invert_affine_points")
@patch("aquapose.core.midline.backends.segmentation.extract_affine_crop")
def test_process_frame_with_mock_model_produces_midline(
    mock_extract: MagicMock,
    mock_invert: MagicMock,
) -> None:
    """process_frame with mocked YOLO model produces AnnotatedDetection with Midline2D."""
    crop_w, crop_h = 128, 64
    n_points = 15

    # Set up mock AffineCrop
    affine_crop = _make_affine_crop(crop_w, crop_h)
    mock_extract.return_value = affine_crop

    # mock invert_affine_points to return same points shifted slightly
    def _identity_invert(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
        return pts.astype(np.float32) + 10.0

    mock_invert.side_effect = _identity_invert

    # Build a backend with a mock model
    backend = SegmentationBackend(weights_path=None, n_points=n_points, min_area=50)
    mock_model = MagicMock()
    mock_model.predict.return_value = _make_yolo_masks_result(crop_h, crop_w)
    backend._model = mock_model

    det = _make_det()
    result = backend.process_frame(
        frame_idx=5,
        frame_dets={"cam1": [det]},
        frames={"cam1": _make_frame()},
        camera_ids=["cam1"],
    )

    ann = result["cam1"][0]
    assert isinstance(ann, AnnotatedDetection)
    assert ann.midline is not None, "Expected non-None midline with mock model"
    assert ann.midline.points.shape == (n_points, 2)
    assert ann.midline.half_widths.shape == (n_points,)
    assert ann.midline.camera_id == "cam1"
    assert ann.midline.frame_index == 5
    assert ann.mask is not None


# ---------------------------------------------------------------------------
# Mask area threshold
# ---------------------------------------------------------------------------


@patch("aquapose.core.midline.backends.segmentation.invert_affine_points")
@patch("aquapose.core.midline.backends.segmentation.extract_affine_crop")
def test_mask_below_min_area_returns_none(
    mock_extract: MagicMock,
    mock_invert: MagicMock,
) -> None:
    """When mask area < min_area, midline is None."""
    crop_w, crop_h = 128, 64

    affine_crop = _make_affine_crop(crop_w, crop_h)
    mock_extract.return_value = affine_crop
    mock_invert.side_effect = lambda pts, M: pts.astype(np.float32)

    # Create a tiny mask (< min_area=300)
    tiny_mask = np.zeros((crop_h, crop_w), dtype=np.float32)
    tiny_mask[30:32, 60:65] = 1.0  # only 10 pixels

    mock_tensor = MagicMock()
    mock_tensor.cpu.return_value.numpy.return_value = tiny_mask
    mock_masks = MagicMock()
    mock_masks.data = [mock_tensor]
    mock_result = MagicMock()
    mock_result.masks = mock_masks

    backend = SegmentationBackend(weights_path=None, min_area=300)
    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    backend._model = mock_model

    det = _make_det()
    result = backend.process_frame(
        frame_idx=0,
        frame_dets={"cam1": [det]},
        frames={"cam1": _make_frame()},
        camera_ids=["cam1"],
    )

    assert result["cam1"][0].midline is None


# ---------------------------------------------------------------------------
# det.angle = None fallback
# ---------------------------------------------------------------------------


@patch("aquapose.core.midline.backends.segmentation.invert_affine_points")
@patch("aquapose.core.midline.backends.segmentation.extract_affine_crop")
def test_none_angle_falls_back_to_zero(
    mock_extract: MagicMock,
    mock_invert: MagicMock,
) -> None:
    """Detection with angle=None does not crash — defaults to 0.0 in crop extraction."""
    crop_w, crop_h = 128, 64
    affine_crop = _make_affine_crop(crop_w, crop_h)
    mock_extract.return_value = affine_crop
    mock_invert.side_effect = lambda pts, M: pts.astype(np.float32)

    # Use a mock model so crop extraction is actually called
    backend = SegmentationBackend(weights_path=None)
    mock_model = MagicMock()
    mock_model.predict.return_value = _make_yolo_masks_result(crop_h, crop_w)
    backend._model = mock_model

    det = _make_det(angle=None)
    result = backend.process_frame(
        frame_idx=0,
        frame_dets={"cam1": [det]},
        frames={"cam1": _make_frame()},
        camera_ids=["cam1"],
    )

    # No exception — angle=None handled gracefully
    assert "cam1" in result

    # Verify extract_affine_crop was called with angle_math_rad=0.0
    assert mock_extract.called
    call_kwargs = mock_extract.call_args
    angle_passed = call_kwargs.kwargs.get("angle_math_rad")
    assert angle_passed == 0.0, (
        f"Expected angle_math_rad=0.0, got {angle_passed!r}; kwargs={call_kwargs.kwargs}"
    )


# ---------------------------------------------------------------------------
# Import boundary test
# ---------------------------------------------------------------------------


def test_no_engine_import() -> None:
    """segmentation.py does not import from aquapose.engine at runtime."""
    import aquapose.core.midline.backends.segmentation as seg_mod

    engine_imports = [name for name in vars(seg_mod) if "engine" in name.lower()]
    assert not engine_imports, (
        f"segmentation.py appears to have engine imports: {engine_imports}"
    )

    # More robust: check the module's __file__ source for 'aquapose.engine'
    import ast
    import pathlib

    source = pathlib.Path(seg_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "aquapose.engine" not in node.module, (
                f"segmentation.py imports from aquapose.engine: {node.module}"
            )
