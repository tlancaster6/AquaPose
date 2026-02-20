"""Unit tests for MOG2 fish detector."""

import numpy as np
import pytest

from aquapose.segmentation.detector import Detection, MOG2Detector


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
