"""Unit tests for Mask R-CNN segmentation model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aquapose.segmentation.model import MaskRCNNSegmentor, SegmentationResult


class TestSegmentationResult:
    """Tests for the SegmentationResult dataclass."""

    def test_fields(self) -> None:
        result = SegmentationResult(
            bbox=(10, 20, 100, 200),
            mask_rle={"size": [480, 640], "counts": b"abc"},
            confidence=0.95,
            label=1,
        )
        assert result.bbox == (10, 20, 100, 200)
        assert result.confidence == 0.95
        assert result.label == 1


class TestMaskRCNNSegmentorConstruction:
    """Tests for model construction (no GPU needed)."""

    @pytest.mark.slow
    def test_default_construction(self) -> None:
        """Model builds with default params (downloads pretrained weights)."""
        segmentor = MaskRCNNSegmentor()
        model = segmentor.get_model()
        assert model is not None

    @pytest.mark.slow
    def test_custom_num_classes(self) -> None:
        segmentor = MaskRCNNSegmentor(num_classes=3)
        model = segmentor.get_model()
        # Check box predictor has correct num classes
        assert model.roi_heads.box_predictor.cls_score.out_features == 3

    @pytest.mark.slow
    def test_get_model_returns_module(self) -> None:
        segmentor = MaskRCNNSegmentor()
        model = segmentor.get_model()
        assert isinstance(model, torch.nn.Module)


class TestMaskRCNNSegmentorPredict:
    """Tests for model prediction (uses small random images)."""

    @pytest.mark.slow
    def test_predict_returns_list_per_image(self) -> None:
        segmentor = MaskRCNNSegmentor(confidence_threshold=0.0)
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)]
        results = segmentor.predict(images)

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], list)

    @pytest.mark.slow
    def test_predict_batch_of_images(self) -> None:
        segmentor = MaskRCNNSegmentor(confidence_threshold=0.0)
        images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)
        ]
        results = segmentor.predict(images)
        assert len(results) == 3

    @pytest.mark.slow
    def test_segmentation_result_has_rle(self) -> None:
        """If any detection found, it should have valid RLE format."""
        segmentor = MaskRCNNSegmentor(confidence_threshold=0.0)
        images = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)]
        results = segmentor.predict(images)

        for det in results[0]:
            assert isinstance(det, SegmentationResult)
            assert "counts" in det.mask_rle
            assert "size" in det.mask_rle
            assert isinstance(det.confidence, float)
            assert isinstance(det.label, int)

    @pytest.mark.slow
    def test_confidence_filtering(self) -> None:
        """High threshold should filter out low-confidence detections."""
        segmentor = MaskRCNNSegmentor(confidence_threshold=0.99)
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)]
        results = segmentor.predict(images)

        # On random noise, unlikely to have very high confidence detections
        for det in results[0]:
            assert det.confidence >= 0.99
