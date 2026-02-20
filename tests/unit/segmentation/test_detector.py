"""Unit tests for MOG2 fish detector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aquapose.segmentation.detector import (
    Detection,
    MOG2Detector,
    YOLODetector,
    make_detector,
)


@pytest.fixture
def detector() -> MOG2Detector:
    """Create a default MOG2Detector instance."""
    return MOG2Detector()


@pytest.fixture
def blank_frame() -> np.ndarray:
    """A plain gray frame (background)."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def frame_with_fish(blank_frame: np.ndarray) -> np.ndarray:
    """A frame with a bright white rectangle simulating a fish."""
    frame = blank_frame.copy()
    # Draw a bright white rectangle (fish-like blob)
    frame[200:260, 300:400] = 255
    return frame


@pytest.fixture
def frame_with_multiple_fish(blank_frame: np.ndarray) -> np.ndarray:
    """A frame with two separated bright blobs."""
    frame = blank_frame.copy()
    frame[100:160, 100:200] = 255  # Fish 1
    frame[300:360, 400:500] = 255  # Fish 2
    return frame


@pytest.fixture
def warmed_detector(detector: MOG2Detector, blank_frame: np.ndarray) -> MOG2Detector:
    """A detector with a stable background model from blank frames."""
    # Feed many blank frames to build a stable background
    detector.warm_up([blank_frame] * 50)
    return detector


class TestDetection:
    """Tests for the Detection dataclass."""

    def test_detection_fields(self) -> None:
        mask = np.zeros((480, 640), dtype=np.uint8)
        det = Detection(bbox=(10, 20, 50, 60), mask=mask, area=3000, confidence=1.0)
        assert det.bbox == (10, 20, 50, 60)
        assert det.area == 3000
        assert det.confidence == 1.0
        assert det.mask.shape == (480, 640)


class TestMOG2DetectorInit:
    """Tests for MOG2Detector construction."""

    def test_default_construction(self) -> None:
        detector = MOG2Detector()
        assert detector is not None

    def test_custom_params(self) -> None:
        detector = MOG2Detector(
            history=200,
            var_threshold=16,
            detect_shadows=False,
            min_area=500,
            padding_fraction=0.2,
        )
        assert detector is not None


class TestDetectEmptyFrame:
    """Tests for detection on frames with no fish."""

    def test_empty_frame_returns_no_detections(
        self, warmed_detector: MOG2Detector, blank_frame: np.ndarray
    ) -> None:
        detections = warmed_detector.detect(blank_frame)
        assert isinstance(detections, list)
        assert len(detections) == 0


class TestDetectSingleFish:
    """Tests for detection of a single fish."""

    def test_single_fish_detected(
        self, warmed_detector: MOG2Detector, frame_with_fish: np.ndarray
    ) -> None:
        detections = warmed_detector.detect(frame_with_fish)
        assert len(detections) >= 1
        det = detections[0]
        assert isinstance(det, Detection)
        assert det.area > 0
        assert det.confidence == 1.0

    def test_single_fish_bbox_format(
        self, warmed_detector: MOG2Detector, frame_with_fish: np.ndarray
    ) -> None:
        detections = warmed_detector.detect(frame_with_fish)
        assert len(detections) >= 1
        x, y, w, h = detections[0].bbox
        assert w > 0
        assert h > 0
        # Bbox should be near the fish location (200:260, 300:400) with padding
        assert x < 400
        assert y < 260

    def test_single_fish_mask_shape(
        self, warmed_detector: MOG2Detector, frame_with_fish: np.ndarray
    ) -> None:
        detections = warmed_detector.detect(frame_with_fish)
        assert len(detections) >= 1
        mask = detections[0].mask
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8
        # Mask should have some nonzero pixels
        assert np.any(mask > 0)


class TestDetectMultipleFish:
    """Tests for detection of multiple fish."""

    def test_multiple_fish_detected(
        self, warmed_detector: MOG2Detector, frame_with_multiple_fish: np.ndarray
    ) -> None:
        detections = warmed_detector.detect(frame_with_multiple_fish)
        assert len(detections) >= 2


class TestNoiseFiltering:
    """Tests for small-blob filtering."""

    def test_small_noise_filtered(
        self, warmed_detector: MOG2Detector, blank_frame: np.ndarray
    ) -> None:
        """A tiny bright spot should be filtered by min_area."""
        frame = blank_frame.copy()
        # 5x5 bright spot = 25 pixels, well below default min_area=200
        frame[100:105, 100:105] = 255
        detections = warmed_detector.detect(frame)
        assert len(detections) == 0


class TestEdgePadding:
    """Tests for bbox padding at frame edges."""

    def test_bbox_clipped_to_frame_bounds(
        self, warmed_detector: MOG2Detector, blank_frame: np.ndarray
    ) -> None:
        """A fish near the edge should have its padded bbox clipped."""
        frame = blank_frame.copy()
        # Fish near top-left corner
        frame[0:60, 0:100] = 255
        detections = warmed_detector.detect(frame)
        assert len(detections) >= 1
        x, y, w, h = detections[0].bbox
        # Bbox should not go negative (clipped to frame)
        assert x >= 0
        assert y >= 0
        assert x + w <= 640
        assert y + h <= 480


class TestWarmUp:
    """Tests for the warm_up method."""

    def test_warm_up_accepts_frames(
        self, detector: MOG2Detector, blank_frame: np.ndarray
    ) -> None:
        """warm_up should consume frames without error."""
        detector.warm_up([blank_frame] * 10)

    def test_warm_up_builds_background(
        self, detector: MOG2Detector, blank_frame: np.ndarray
    ) -> None:
        """After warm-up, blank frames should not trigger detections."""
        detector.warm_up([blank_frame] * 50)
        detections = detector.detect(blank_frame)
        assert len(detections) == 0


class TestShadowExclusion:
    """Tests for shadow pixel exclusion via thresholding at 254."""

    def test_shadow_pixels_excluded_from_mask(
        self, warmed_detector: MOG2Detector, blank_frame: np.ndarray
    ) -> None:
        """The detector thresholds MOG2 output at 254 so shadow pixels (127) are excluded.

        We verify the contract indirectly: after warm-up on background, feed a
        frame with a bright blob. The detected mask should only contain definite
        foreground pixels (255 in MOG2 output), not shadow pixels (127).
        """
        frame = blank_frame.copy()
        frame[200:260, 300:400] = 255
        detections = warmed_detector.detect(frame)
        assert len(detections) >= 1
        # Mask values should be binary 0 or 255 (no intermediate shadow values)
        mask = detections[0].mask
        unique_vals = set(np.unique(mask))
        assert unique_vals <= {0, 255}


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
    with patch("aquapose.segmentation.detector.YOLODetector.__init__.__globals__", {}):
        pass
    # Patch at the module import point inside __init__
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

    def test_mask_is_full_frame(self, mock_yolo_cls: MagicMock) -> None:
        """detect() mask has the same shape as the input frame (H, W)."""
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            det = YOLODetector(model_path="dummy.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        mask = results[0].mask
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8

    def test_mask_values_binary(self, mock_yolo_cls: MagicMock) -> None:
        """detect() mask contains only 0 and 255."""
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            det = YOLODetector(model_path="dummy.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        unique_vals = set(np.unique(results[0].mask))
        assert unique_vals <= {0, 255}

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

    def test_make_mog2_returns_mog2_detector(self) -> None:
        """make_detector('mog2') returns a MOG2Detector instance."""
        result = make_detector("mog2")
        assert isinstance(result, MOG2Detector)

    def test_make_mog2_forwards_kwargs(self) -> None:
        """make_detector('mog2', min_area=500) passes kwargs to MOG2Detector."""
        result = make_detector("mog2", min_area=500)
        assert isinstance(result, MOG2Detector)
        assert result._min_area == 500

    def test_make_yolo_returns_yolo_detector(self) -> None:
        """make_detector('yolo', model_path=...) returns a YOLODetector instance."""
        mock_cls = _make_yolo_mock(boxes_xyxy=[], confs=[])
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_cls)}):
            result = make_detector("yolo", model_path="dummy.pt")
        assert isinstance(result, YOLODetector)

    def test_make_detector_unknown_raises_value_error(self) -> None:
        """make_detector with unrecognized kind raises ValueError."""
        with pytest.raises(ValueError, match="Unknown detector kind"):
            make_detector("bad")
