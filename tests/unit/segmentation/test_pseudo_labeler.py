"""Unit tests for SAM2 pseudo-labeler."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from aquapose.segmentation.detector import Detection
from aquapose.segmentation.pseudo_labeler import (
    FrameAnnotation,
    SAMPseudoLabeler,
    _mask_to_logits,
)


class TestMaskToLogits:
    """Tests for the mask-to-logits conversion utility."""

    def test_output_shape(self) -> None:
        mask = np.zeros((480, 640), dtype=np.uint8)
        logits = _mask_to_logits(mask, target_size=256)
        assert logits.shape == (1, 256, 256)

    def test_foreground_gets_positive_logits(self) -> None:
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        logits = _mask_to_logits(mask)
        assert np.all(logits > 0)

    def test_background_gets_negative_logits(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        logits = _mask_to_logits(mask)
        assert np.all(logits < 0)

    def test_mixed_mask_has_both_signs(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        logits = _mask_to_logits(mask)
        assert np.any(logits > 0)
        assert np.any(logits < 0)

    def test_custom_target_size(self) -> None:
        mask = np.zeros((480, 640), dtype=np.uint8)
        logits = _mask_to_logits(mask, target_size=128)
        assert logits.shape == (1, 128, 128)


class TestSAMPseudoLabelerInit:
    """Tests for SAMPseudoLabeler construction."""

    def test_default_construction(self) -> None:
        labeler = SAMPseudoLabeler()
        assert labeler._model_variant == "facebook/sam2.1-hiera-large"
        assert labeler._predictor is None

    def test_custom_variant(self) -> None:
        labeler = SAMPseudoLabeler(
            model_variant="facebook/sam2.1-hiera-small", device="cpu"
        )
        assert labeler._model_variant == "facebook/sam2.1-hiera-small"
        assert labeler._device == "cpu"


class TestSAMPseudoLabelerPredict:
    """Tests for SAMPseudoLabeler.predict with mocked SAM2."""

    def test_empty_detections_returns_empty(self) -> None:
        labeler = SAMPseudoLabeler()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = labeler.predict(image, [])
        assert result == []

    def test_predict_returns_masks_for_each_detection(self) -> None:
        """Mock SAM2 to verify predict returns one mask per detection."""
        labeler = SAMPseudoLabeler()

        # Create mock predictor
        mock_predictor = MagicMock()
        # SAM2 predict returns (masks, scores, logits)
        # masks shape: (1, H, W) with multimask_output=False
        mock_mask = np.ones((1, 480, 640), dtype=np.float32)
        mock_predictor.predict.return_value = (
            mock_mask,
            np.array([0.99]),
            np.zeros((1, 256, 256)),
        )
        labeler._predictor = mock_predictor

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        det1 = Detection(
            bbox=(100, 100, 50, 50),
            mask=np.zeros((480, 640), dtype=np.uint8),
            area=2500,
            confidence=1.0,
        )
        det2 = Detection(
            bbox=(300, 300, 60, 40),
            mask=np.zeros((480, 640), dtype=np.uint8),
            area=2400,
            confidence=1.0,
        )

        results = labeler.predict(image, [det1, det2])

        assert len(results) == 2
        assert all(isinstance(m, np.ndarray) for m in results)
        assert all(m.shape == (480, 640) for m in results)
        assert all(m.dtype == np.uint8 for m in results)

    def test_bbox_conversion_to_sam2_format(self) -> None:
        """Verify bbox (x,y,w,h) is converted to [x1,y1,x2,y2] for SAM2."""
        labeler = SAMPseudoLabeler()
        mock_predictor = MagicMock()
        mock_mask = np.ones((1, 480, 640), dtype=np.float32)
        mock_predictor.predict.return_value = (
            mock_mask,
            np.array([0.99]),
            np.zeros((1, 256, 256)),
        )
        labeler._predictor = mock_predictor

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        det = Detection(
            bbox=(100, 200, 50, 60),
            mask=np.zeros((480, 640), dtype=np.uint8),
            area=3000,
            confidence=1.0,
        )

        labeler.predict(image, [det])

        # Check the box arg passed to predict
        call_kwargs = mock_predictor.predict.call_args[1]
        box = call_kwargs["box"]
        np.testing.assert_array_equal(box, [100, 200, 150, 260])


class TestFrameAnnotation:
    """Tests for the FrameAnnotation dataclass."""

    def test_fields(self) -> None:
        from pathlib import Path

        fa = FrameAnnotation(
            frame_id="cam01_f001",
            image_path=Path("/tmp/frame.jpg"),
            masks=[np.zeros((100, 100), dtype=np.uint8)],
            camera_id="cam01",
        )
        assert fa.frame_id == "cam01_f001"
        assert fa.camera_id == "cam01"
        assert len(fa.masks) == 1
