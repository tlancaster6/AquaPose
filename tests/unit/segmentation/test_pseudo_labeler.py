"""Unit tests for SAM2 pseudo-labeler."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from aquapose.segmentation.detector import Detection
from aquapose.segmentation.pseudo_labeler import (
    AnnotatedFrame,
    FrameAnnotation,
    SAMPseudoLabeler,
    _select_largest_mask,
    filter_mask,
    to_coco_dataset,
)


@pytest.fixture
def sample_mask() -> np.ndarray:
    """A simple binary mask for testing."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 30:70] = 255
    return mask


@pytest.fixture
def sample_detection() -> Detection:
    """A detection with a large confident bbox."""
    return Detection(
        bbox=(30, 20, 40, 60),
        mask=np.zeros((100, 100), dtype=np.uint8),
        area=2400,
        confidence=0.9,
    )


class TestSelectLargestMask:
    """Tests for _select_largest_mask helper."""

    def test_single_mask_returns_binarized(self) -> None:
        masks = np.zeros((1, 50, 50), dtype=np.float32)
        masks[0, 10:40, 10:40] = 1.0
        result = _select_largest_mask(masks)
        assert result.shape == (50, 50)
        assert result.dtype == np.uint8
        assert result.max() == 255

    def test_selects_largest_of_two(self) -> None:
        masks = np.zeros((2, 50, 50), dtype=np.float32)
        # Mask 0: small region
        masks[0, 10:15, 10:15] = 1.0
        # Mask 1: large region
        masks[1, 5:45, 5:45] = 1.0
        result = _select_largest_mask(masks)
        # Should return mask 1 (larger)
        assert result[25, 25] == 255
        assert result[12, 12] == 255  # also inside mask 1's region

    def test_all_zeros_returns_zeros(self) -> None:
        masks = np.zeros((2, 30, 30), dtype=np.float32)
        result = _select_largest_mask(masks)
        assert result.max() == 0


class TestFilterMask:
    """Tests for the filter_mask quality filtering function."""

    def _make_mask(self, h: int = 100, w: int = 100, fill: bool = True) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        if fill:
            # Mask covers ~50% of a 60x60 bbox region (within fill ratio bounds)
            mask[30:60, 30:60] = 255  # 900 pixels
        return mask

    def _make_det(
        self,
        bbox: tuple[int, int, int, int] = (25, 25, 60, 60),
        confidence: float = 0.9,
    ) -> Detection:
        # bbox_area = 3600, mask = 900px → fill = 0.25 (between 0.15 and 0.85)
        return Detection(
            bbox=bbox,
            mask=np.zeros((100, 100), dtype=np.uint8),
            area=bbox[2] * bbox[3],
            confidence=confidence,
        )

    def test_good_mask_passes(self) -> None:
        mask = self._make_mask()
        det = self._make_det()
        result = filter_mask(mask, det)
        assert result is not None

    def test_low_confidence_rejected(self) -> None:
        mask = self._make_mask()
        det = self._make_det(confidence=0.1)
        result = filter_mask(mask, det, min_conf=0.3)
        assert result is None

    def test_too_small_area_rejected(self) -> None:
        # Tiny mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50:53, 50:53] = 255  # 9 pixels
        det = self._make_det()
        result = filter_mask(mask, det, min_area=100)
        assert result is None

    def test_fill_ratio_too_low_rejected(self) -> None:
        # Mask with very few pixels relative to a large bbox
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50:55, 50:55] = 255  # 25 pixels
        det = self._make_det(bbox=(0, 0, 90, 90))  # bbox area = 8100
        result = filter_mask(mask, det, min_fill=0.15, min_area=1)
        assert result is None

    def test_fill_ratio_too_high_rejected(self) -> None:
        # Mask covering almost the whole frame — exceeds bbox
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        det = self._make_det(bbox=(30, 20, 40, 60))  # bbox area = 2400
        # fill = 10000 / 2400 >> max_fill=0.85
        result = filter_mask(mask, det, max_fill=0.85)
        assert result is None

    def test_empty_mask_rejected(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        det = self._make_det()
        result = filter_mask(mask, det)
        assert result is None

    def test_largest_connected_component_kept(self) -> None:
        # Two disconnected blobs — only the larger should survive
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:15, 10:15] = 255  # small blob (25 px)
        mask[40:80, 30:70] = 255  # large blob (1600 px)
        det = self._make_det(bbox=(10, 10, 70, 75), confidence=0.9)
        result = filter_mask(mask, det, min_area=50)
        assert result is not None
        # Small blob region should be zeroed out
        assert result[12, 12] == 0
        # Large blob region should survive
        assert result[60, 50] == 255

    def test_zero_bbox_area_rejected(self) -> None:
        mask = self._make_mask()
        det = self._make_det(bbox=(30, 20, 0, 60))  # bw=0 → area=0
        result = filter_mask(mask, det)
        assert result is None


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

    def test_draw_pseudolabels_flag(self) -> None:
        labeler = SAMPseudoLabeler(draw_pseudolabels=True)
        assert labeler._draw_pseudolabels is True

    def test_no_mask_prompt_attribute(self) -> None:
        """Verify _mask_to_logits no longer exists in the module."""
        import aquapose.segmentation.pseudo_labeler as mod

        assert not hasattr(mod, "_mask_to_logits"), (
            "_mask_to_logits should have been removed"
        )


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

    def test_predict_uses_box_only_no_mask_input(self) -> None:
        """Verify predict() does not pass mask_input to SAM2."""
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
            bbox=(100, 100, 50, 50),
            mask=np.zeros((480, 640), dtype=np.uint8),
            area=2500,
            confidence=1.0,
        )
        labeler.predict(image, [det])

        call_kwargs = mock_predictor.predict.call_args[1]
        assert "mask_input" not in call_kwargs, (
            "mask_input should not be passed (box-only mode)"
        )

    def test_bbox_conversion_to_sam2_format(self) -> None:
        """Verify bbox (x,y,w,h) is converted to crop-relative [x1,y1,x2,y2] for SAM2.

        SAMPseudoLabeler crops the image around the bbox before calling SAM2,
        so the box coordinates passed to predict() are relative to the crop origin.
        """
        from aquapose.segmentation.crop import compute_crop_region

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
        bbox = (100, 200, 50, 60)
        det = Detection(
            bbox=bbox,
            mask=np.zeros((480, 640), dtype=np.uint8),
            area=3000,
            confidence=1.0,
        )

        labeler.predict(image, [det])

        # Compute the expected crop-relative box
        region = compute_crop_region(bbox, 480, 640, padding=0.25)
        bx, by, bw, bh = bbox
        expected_box = np.array(
            [bx - region.x1, by - region.y1, bx + bw - region.x1, by + bh - region.y1],
            dtype=np.float32,
        )

        # Check the box arg passed to predict is crop-relative
        call_kwargs = mock_predictor.predict.call_args[1]
        box = call_kwargs["box"]
        np.testing.assert_array_equal(box, expected_box)


class TestFrameAnnotation:
    """Tests for the FrameAnnotation dataclass."""

    def test_fields(self) -> None:
        fa = FrameAnnotation(
            frame_id="cam01_f001",
            image_path=Path("/tmp/frame.jpg"),
            masks=[np.zeros((100, 100), dtype=np.uint8)],
            camera_id="cam01",
        )
        assert fa.frame_id == "cam01_f001"
        assert fa.camera_id == "cam01"
        assert len(fa.masks) == 1


class TestToCOCODataset:
    """Tests for COCO JSON conversion."""

    def test_creates_valid_coco_json(
        self, tmp_path: Path, sample_mask: np.ndarray
    ) -> None:
        frames = [
            AnnotatedFrame(
                frame_id="cam01_f001",
                image_path=Path("frame001.jpg"),
                masks=[sample_mask],
                camera_id="cam01",
            )
        ]

        output_path = tmp_path / "coco.json"
        result_path = to_coco_dataset(frames, output_path)
        assert result_path.exists()

        with open(result_path) as f:
            coco = json.load(f)

        assert "images" in coco
        assert "annotations" in coco
        assert "categories" in coco
        assert len(coco["images"]) == 1
        assert len(coco["annotations"]) == 1
        assert coco["categories"] == [{"id": 1, "name": "fish"}]

    def test_annotation_has_required_fields(
        self, tmp_path: Path, sample_mask: np.ndarray
    ) -> None:
        frames = [
            AnnotatedFrame(
                frame_id="cam01_f001",
                image_path=Path("frame001.jpg"),
                masks=[sample_mask],
                camera_id="cam01",
            )
        ]

        output_path = tmp_path / "coco.json"
        to_coco_dataset(frames, output_path)

        with open(output_path) as f:
            coco = json.load(f)

        ann = coco["annotations"][0]
        assert "id" in ann
        assert "image_id" in ann
        assert "category_id" in ann
        assert ann["category_id"] == 1
        assert "segmentation" in ann
        assert "bbox" in ann
        assert "area" in ann
        assert ann["iscrowd"] == 0

    def test_rle_is_valid_pycocotools_format(
        self, tmp_path: Path, sample_mask: np.ndarray
    ) -> None:
        """Verify the RLE can be decoded back to a mask."""
        import pycocotools.mask as mask_util

        frames = [
            AnnotatedFrame(
                frame_id="cam01_f001",
                image_path=Path("frame001.jpg"),
                masks=[sample_mask],
                camera_id="cam01",
            )
        ]

        output_path = tmp_path / "coco.json"
        to_coco_dataset(frames, output_path)

        with open(output_path) as f:
            coco = json.load(f)

        rle = coco["annotations"][0]["segmentation"]
        # Convert counts back to bytes for pycocotools
        rle_bytes = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
        decoded = mask_util.decode(rle_bytes)
        assert decoded.shape == (100, 100)
        assert np.any(decoded > 0)

    def test_multiple_masks_per_frame(
        self, tmp_path: Path, sample_mask: np.ndarray
    ) -> None:
        frames = [
            AnnotatedFrame(
                frame_id="cam01_f001",
                image_path=Path("frame001.jpg"),
                masks=[sample_mask, sample_mask],
                camera_id="cam01",
            )
        ]

        output_path = tmp_path / "coco.json"
        to_coco_dataset(frames, output_path)

        with open(output_path) as f:
            coco = json.load(f)

        assert len(coco["annotations"]) == 2
        assert coco["annotations"][0]["id"] == 1
        assert coco["annotations"][1]["id"] == 2

    def test_negative_frame_has_no_annotations(self, tmp_path: Path) -> None:
        frames = [
            AnnotatedFrame(
                frame_id="cam01_f003",
                image_path=Path("frame003.jpg"),
                masks=[],
                camera_id="cam01",
            )
        ]

        output_path = tmp_path / "coco.json"
        to_coco_dataset(frames, output_path)

        with open(output_path) as f:
            coco = json.load(f)

        assert len(coco["images"]) == 1
        assert len(coco["annotations"]) == 0
