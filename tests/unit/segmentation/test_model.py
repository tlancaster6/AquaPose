"""Unit tests for segmentation models."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aquapose.segmentation.crop import CropRegion
from aquapose.segmentation.model import (
    MaskRCNNSegmentor,
    SegmentationResult,
    UNetSegmentor,
)


class TestSegmentationResult:
    """Tests for the SegmentationResult dataclass."""

    def test_fields(self) -> None:
        region = CropRegion(x1=0, y1=0, x2=100, y2=200, frame_h=480, frame_w=640)
        mask = np.zeros((200, 100), dtype=np.uint8)
        result = SegmentationResult(
            bbox=(10, 20, 100, 200),
            mask=mask,
            confidence=0.95,
            label=1,
            crop_region=region,
        )
        assert result.bbox == (10, 20, 100, 200)
        assert result.confidence == 0.95
        assert result.label == 1
        assert result.mask is mask
        assert result.crop_region is region

    def test_no_mask_rle_field(self) -> None:
        """SegmentationResult must not have mask_rle — it was removed."""
        region = CropRegion(x1=0, y1=0, x2=10, y2=10, frame_h=100, frame_w=100)
        result = SegmentationResult(
            bbox=(0, 0, 10, 10),
            mask=np.zeros((10, 10), dtype=np.uint8),
            confidence=0.5,
            label=1,
            crop_region=region,
        )
        assert not hasattr(result, "mask_rle")


class TestUNetSegmentorConstruction:
    """Tests for model construction (no GPU needed)."""

    def test_default_construction(self) -> None:
        """Model builds with default params (ImageNet-pretrained encoder)."""
        segmentor = UNetSegmentor()
        model = segmentor.get_model()
        assert model is not None

    def test_get_model_returns_module(self) -> None:
        segmentor = UNetSegmentor()
        model = segmentor.get_model()
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass_shape(self) -> None:
        """Model forward pass produces correct output shape."""
        segmentor = UNetSegmentor()
        model = segmentor.get_model()
        model.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 128, 128)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestUNetSegmentorSegment:
    """Tests for the primary segment() inference method."""

    def test_segment_returns_list_per_crop(self) -> None:
        segmentor = UNetSegmentor(confidence_threshold=0.0)
        crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        region = CropRegion(x1=50, y1=50, x2=150, y2=150, frame_h=480, frame_w=640)

        results = segmentor.segment([crop], [region])

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], list)

    def test_segment_batch_of_different_sizes(self) -> None:
        """Batch with crops of different sizes — resize handles this."""
        segmentor = UNetSegmentor(confidence_threshold=0.0)
        crops = [
            np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (120, 150, 3), dtype=np.uint8),
            np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8),
        ]
        regions = [
            CropRegion(x1=0, y1=0, x2=100, y2=80, frame_h=480, frame_w=640),
            CropRegion(x1=100, y1=0, x2=250, y2=120, frame_h=480, frame_w=640),
            CropRegion(x1=200, y1=100, x2=296, y2=164, frame_h=480, frame_w=640),
        ]
        results = segmentor.segment(crops, regions)
        assert len(results) == 3

    def test_segment_result_has_crop_space_mask_and_region(self) -> None:
        """Results must carry crop-space mask (NOT full-frame) and CropRegion."""
        segmentor = UNetSegmentor(confidence_threshold=0.0)
        h, w = 80, 100
        crop = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        region = CropRegion(
            x1=10, y1=20, x2=10 + w, y2=20 + h, frame_h=480, frame_w=640
        )

        results = segmentor.segment([crop], [region])

        for det in results[0]:
            assert isinstance(det, SegmentationResult)
            # mask must be crop-sized, not 128x128 or frame-sized
            assert det.mask.shape == (h, w)
            assert det.mask.dtype == np.uint8
            # mask values should be 0 or 255
            unique_vals = set(np.unique(det.mask).tolist())
            assert unique_vals.issubset({0, 255})
            # crop_region must be attached
            assert det.crop_region is region
            assert isinstance(det.confidence, float)
            assert isinstance(det.label, int)
            assert det.label == 1

    def test_segment_mismatched_lengths_raises(self) -> None:
        segmentor = UNetSegmentor(confidence_threshold=0.0)
        crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        region = CropRegion(x1=0, y1=0, x2=100, y2=100, frame_h=480, frame_w=640)

        with pytest.raises(
            ValueError, match="crops and crop_regions must have equal length"
        ):
            segmentor.segment([crop, crop], [region])

    def test_confidence_filtering_in_segment(self) -> None:
        """High threshold should filter out low-confidence predictions."""
        segmentor = UNetSegmentor(confidence_threshold=0.99)
        crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        region = CropRegion(x1=0, y1=0, x2=100, y2=100, frame_h=480, frame_w=640)
        results = segmentor.segment([crop], [region])
        # With random noise input + high threshold, likely empty
        for det in results[0]:
            assert det.confidence >= 0.99


class TestUNetSegmentorPredict:
    """Tests for predict() backward-compatible entry point."""

    def test_predict_returns_list_per_image(self) -> None:
        segmentor = UNetSegmentor(confidence_threshold=0.0)
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)]
        results = segmentor.predict(images)

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], list)

    def test_predict_batch_of_images(self) -> None:
        segmentor = UNetSegmentor(confidence_threshold=0.0)
        images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)
        ]
        results = segmentor.predict(images)
        assert len(results) == 3

    def test_predict_result_has_mask_ndarray(self) -> None:
        """predict() should return SegmentationResult with mask ndarray."""
        segmentor = UNetSegmentor(confidence_threshold=0.0)
        images = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)]
        results = segmentor.predict(images)

        for det in results[0]:
            assert isinstance(det, SegmentationResult)
            assert isinstance(det.mask, np.ndarray)
            assert det.mask.dtype == np.uint8
            assert isinstance(det.confidence, float)
            assert isinstance(det.label, int)
            # crop_region covers full image
            assert det.crop_region.x1 == 0
            assert det.crop_region.y1 == 0
            assert det.crop_region.x2 == 128
            assert det.crop_region.y2 == 128

    def test_predict_trivial_crop_region(self) -> None:
        """predict() attaches a trivial CropRegion covering the full image."""
        segmentor = UNetSegmentor(confidence_threshold=0.0)
        h, w = 80, 120
        image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        results = segmentor.predict([image])

        assert len(results) == 1
        for det in results[0]:
            r = det.crop_region
            assert r.x1 == 0
            assert r.y1 == 0
            assert r.x2 == w
            assert r.y2 == h
            assert r.frame_h == h
            assert r.frame_w == w


class TestMaskRCNNSegmentorBackwardCompat:
    """Verify MaskRCNNSegmentor is still importable."""

    def test_maskrcnn_importable(self) -> None:
        assert MaskRCNNSegmentor is not None
