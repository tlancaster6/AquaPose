"""Unit tests for COCO-format training dataset."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch

from aquapose.segmentation.dataset import CropDataset


def _create_coco_fixture(
    tmp_path: Path, num_images: int = 2, num_anns_per_image: int = 1
) -> tuple[Path, Path]:
    """Create a minimal COCO JSON fixture with synthetic images and masks.

    Returns:
        Tuple of (coco_json_path, image_root_path).
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    images = []
    annotations = []
    ann_id = 1

    for img_idx in range(num_images):
        img_name = f"frame_{img_idx:03d}.jpg"
        h, w = 100, 100
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / img_name), img)

        images.append(
            {"id": img_idx + 1, "file_name": img_name, "width": w, "height": h}
        )

        for ann_idx in range(num_anns_per_image):
            # Create a simple rectangular mask
            mask = np.zeros((h, w), dtype=np.uint8)
            y0 = 20 + ann_idx * 10
            mask[y0 : y0 + 30, 25:75] = 1
            mask_f = np.asfortranarray(mask)
            rle = mask_util.encode(mask_f)

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_idx + 1,
                    "category_id": 1,
                    "segmentation": {
                        "size": rle["size"],
                        "counts": rle["counts"].decode("utf-8"),
                    },
                    "bbox": [25, y0, 50, 30],
                    "area": float(mask_util.area(rle)),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "fish"}],
    }

    coco_path = tmp_path / "coco.json"
    with open(coco_path, "w") as f:
        json.dump(coco, f)

    return coco_path, images_dir


class TestCropDatasetBasic:
    """Basic dataset loading and indexing tests."""

    def test_length_matches_images(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path, num_images=3)
        ds = CropDataset(coco_path, image_root)
        assert len(ds) == 3

    def test_getitem_returns_tensor_and_dict(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path)
        ds = CropDataset(coco_path, image_root, crop_size=256)
        image, target = ds[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(target, dict)
        assert "boxes" in target
        assert "labels" in target
        assert "masks" in target

    def test_image_tensor_shape(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path)
        ds = CropDataset(coco_path, image_root, crop_size=256)
        image, _ = ds[0]

        assert image.shape == (3, 256, 256)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_boxes_format_xyxy(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path)
        ds = CropDataset(coco_path, image_root, crop_size=256)
        _, target = ds[0]

        boxes = target["boxes"]
        assert boxes.shape[1] == 4
        assert boxes.dtype == torch.float32
        # x2 > x1, y2 > y1
        assert (boxes[:, 2] > boxes[:, 0]).all()
        assert (boxes[:, 3] > boxes[:, 1]).all()

    def test_labels_are_ones(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path)
        ds = CropDataset(coco_path, image_root)
        _, target = ds[0]

        labels = target["labels"]
        assert labels.dtype == torch.int64
        assert (labels == 1).all()

    def test_masks_shape_matches_crop_size(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path)
        ds = CropDataset(coco_path, image_root, crop_size=128)
        _, target = ds[0]

        masks = target["masks"]
        assert masks.shape[1:] == (128, 128)
        assert masks.dtype == torch.uint8


class TestCropDatasetNegativeFrames:
    """Test handling of images with no annotations (negative frames)."""

    def test_negative_frame_returns_empty_targets(self, tmp_path: Path) -> None:
        # Create fixture with one image having no annotations
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "empty.jpg"), img)

        coco = {
            "images": [
                {"id": 1, "file_name": "empty.jpg", "width": 100, "height": 100}
            ],
            "annotations": [],
            "categories": [{"id": 1, "name": "fish"}],
        }
        coco_path = tmp_path / "coco.json"
        with open(coco_path, "w") as f:
            json.dump(coco, f)

        ds = CropDataset(coco_path, images_dir, crop_size=256)
        image, target = ds[0]

        assert image.shape == (3, 256, 256)
        assert target["boxes"].shape == (0, 4)
        assert target["labels"].shape == (0,)
        assert target["masks"].shape == (0, 256, 256)


class TestCropDatasetAugmentation:
    """Test that augmentation doesn't crash and produces valid output."""

    def test_augmentation_produces_valid_output(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path)
        ds = CropDataset(coco_path, image_root, crop_size=256, augment=True)

        # Run multiple times to hit different augmentation paths
        for _ in range(5):
            image, target = ds[0]
            assert image.shape == (3, 256, 256)
            assert image.dtype == torch.float32
            assert target["boxes"].shape[1] == 4
            assert target["masks"].shape[0] == target["boxes"].shape[0]


class TestCropDatasetMultipleAnnotations:
    """Test frames with multiple annotations."""

    def test_multiple_annotations_per_image(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(
            tmp_path, num_images=1, num_anns_per_image=3
        )
        ds = CropDataset(coco_path, image_root)
        _, target = ds[0]

        assert target["boxes"].shape[0] == 3
        assert target["labels"].shape[0] == 3
        assert target["masks"].shape[0] == 3
