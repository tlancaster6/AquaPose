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
    _mask_to_logits,
    to_coco_dataset,
)


@pytest.fixture
def sample_mask() -> np.ndarray:
    """A simple binary mask for testing."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 30:70] = 255
    return mask


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
