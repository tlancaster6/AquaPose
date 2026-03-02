"""Unit tests for YOLO fish detector and Detection dataclass."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aquapose.core.detection.backends.yolo import YOLODetector, make_detector
from aquapose.core.types.detection import Detection


class TestDetection:
    """Tests for the Detection dataclass."""

    def test_detection_fields(self) -> None:
        mask = np.zeros((480, 640), dtype=np.uint8)
        det = Detection(bbox=(10, 20, 50, 60), mask=mask, area=3000, confidence=1.0)
        assert det.bbox == (10, 20, 50, 60)
        assert det.area == 3000
        assert det.confidence == 1.0
        assert det.mask.shape == (480, 640)

    def test_detection_angle_defaults_none(self) -> None:
        """angle and obb_points default to None for non-OBB detectors."""
        det = Detection(bbox=(0, 0, 10, 10), mask=None, area=100, confidence=0.9)
        assert det.angle is None
        assert det.obb_points is None

    def test_detection_with_obb_fields(self) -> None:
        """Detection can be constructed with OBB angle and corner points."""
        obb = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        det = Detection(
            bbox=(0, 0, 10, 10),
            mask=None,
            area=100,
            confidence=0.9,
            angle=0.5,
            obb_points=obb,
        )
        assert det.angle == 0.5
        assert det.obb_points is not None
        assert det.obb_points.shape == (4, 2)


# ---------------------------------------------------------------------------
# YOLODetector and make_detector tests
# ---------------------------------------------------------------------------


def _make_yolo_mock(
    boxes_xyxy: list[tuple[float, float, float, float]], confs: list[float]
) -> MagicMock:
    """Build a mock ultralytics YOLO that returns the given boxes on predict()."""
    mock_yolo_cls = MagicMock()
    mock_model_instance = MagicMock()
    mock_yolo_cls.return_value = mock_model_instance

    mock_boxes = []
    for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs, strict=True):
        box = MagicMock()
        box.xyxy = [MagicMock()]
        box.xyxy[0].tolist.return_value = [x1, y1, x2, y2]
        box.conf = [MagicMock()]
        box.conf[0].__float__ = MagicMock(return_value=conf)
        mock_boxes.append(box)

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_model_instance.predict.return_value = [mock_result]

    return mock_yolo_cls


@pytest.fixture
def mock_yolo_cls() -> MagicMock:
    """Mock ultralytics.YOLO class that returns two fish boxes on predict()."""
    return _make_yolo_mock(
        boxes_xyxy=[(100.0, 200.0, 300.0, 400.0)],
        confs=[0.87],
    )


@pytest.fixture
def yolo_detector(mock_yolo_cls: MagicMock) -> YOLODetector:
    """YOLODetector with patched ultralytics.YOLO."""
    with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
        return YOLODetector(model_path="dummy.pt")


class TestYOLODetectorInit:
    """Tests for YOLODetector construction via mocked ultralytics."""

    def test_instantiation_with_mock(self, mock_yolo_cls: MagicMock) -> None:
        """YOLODetector can be created when ultralytics is mocked."""
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            det = YOLODetector(model_path="dummy.pt")
        assert det is not None

    def test_custom_conf_threshold(self, mock_yolo_cls: MagicMock) -> None:
        """Custom conf_threshold is stored."""
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            det = YOLODetector(model_path="dummy.pt", conf_threshold=0.5)
        assert det._conf == 0.5

    def test_custom_padding_fraction(self, mock_yolo_cls: MagicMock) -> None:
        """Custom padding_fraction is stored."""
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            det = YOLODetector(model_path="dummy.pt", padding_fraction=0.2)
        assert det._padding_fraction == 0.2


class TestYOLODetectorDetect:
    """Tests for YOLODetector.detect() output format."""

    def test_returns_list_of_detection(self, mock_yolo_cls: MagicMock) -> None:
        """detect() returns a list of Detection objects."""
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            det = YOLODetector(model_path="dummy.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert isinstance(results, list)
        assert all(isinstance(r, Detection) for r in results)

    def test_bbox_format_xywh(self, mock_yolo_cls: MagicMock) -> None:
        """detect() returns bbox as (x, y, w, h) with positive w and h."""
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            det = YOLODetector(model_path="dummy.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        x, y, w, h = results[0].bbox
        assert w > 0
        assert h > 0
        assert x >= 0
        assert y >= 0

    def test_mask_is_none(self, mock_yolo_cls: MagicMock) -> None:
        """YOLO detections have no mask (bbox-only)."""
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            det = YOLODetector(model_path="dummy.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        assert results[0].mask is None

    def test_confidence_in_range(self, mock_yolo_cls: MagicMock) -> None:
        """detect() confidence is in [0, 1]."""
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            det = YOLODetector(model_path="dummy.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        assert 0.0 <= results[0].confidence <= 1.0

    def test_empty_frame_returns_empty_list(self) -> None:
        """detect() returns empty list when YOLO finds no boxes."""
        mock_cls = _make_yolo_mock(boxes_xyxy=[], confs=[])
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_cls)}):
            det = YOLODetector(model_path="dummy.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert results == []

    def test_bbox_clipped_to_frame_on_edge(self) -> None:
        """Padded bbox at frame edge is clamped to frame bounds."""
        # Box at top-left corner — padding would go negative without clamping
        mock_cls = _make_yolo_mock(boxes_xyxy=[(0.0, 0.0, 100.0, 80.0)], confs=[0.9])
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_cls)}):
            det = YOLODetector(model_path="dummy.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        x, y, w, h = results[0].bbox
        assert x >= 0
        assert y >= 0
        assert x + w <= 640
        assert y + h <= 480


class TestMakeDetector:
    """Tests for make_detector factory function."""

    def test_make_yolo_returns_yolo_detector(self) -> None:
        """make_detector('yolo', model_path=...) returns a YOLODetector instance."""
        mock_cls = _make_yolo_mock(boxes_xyxy=[], confs=[])
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_cls)}):
            result = make_detector("yolo", model_path="dummy.pt")
        assert isinstance(result, YOLODetector)

    def test_make_detector_rejects_mog2(self) -> None:
        """make_detector('mog2') raises ValueError with descriptive message."""
        with pytest.raises(ValueError, match="Unknown detector_kind") as exc_info:
            make_detector("mog2")
        assert "mog2" in str(exc_info.value)

    def test_make_detector_unknown_raises_value_error(self) -> None:
        """make_detector with unrecognized kind raises ValueError."""
        with pytest.raises(ValueError, match="Unknown detector_kind"):
            make_detector("bad")
