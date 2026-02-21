"""Unit tests for COCO-format training datasets."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch

from aquapose.segmentation.dataset import (
    BinaryMaskDataset,
    CropDataset,
    stratified_split,
)


def _create_coco_fixture(
    tmp_path: Path,
    num_images: int = 2,
    num_anns_per_image: int = 1,
    img_size: tuple[int, int] = (100, 100),
    camera_id: str = "cam01",
) -> tuple[Path, Path]:
    """Create a minimal COCO JSON fixture with synthetic images and masks.

    Args:
        tmp_path: Temporary directory for test files.
        num_images: Number of images to generate.
        num_anns_per_image: Number of annotations per image.
        img_size: (height, width) of generated images.
        camera_id: Camera identifier to embed in COCO image entries.

    Returns:
        Tuple of (coco_json_path, image_root_path).
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)

    images = []
    annotations = []
    ann_id = 1
    h, w = img_size

    for img_idx in range(num_images):
        img_name = f"frame_{img_idx:03d}.jpg"
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / img_name), img)

        images.append(
            {
                "id": img_idx + 1,
                "file_name": img_name,
                "width": w,
                "height": h,
                "camera_id": camera_id,
            }
        )

        for ann_idx in range(num_anns_per_image):
            # Create a simple rectangular mask (~25% fill relative to bbox)
            mask = np.zeros((h, w), dtype=np.uint8)
            y0 = min(20 + ann_idx * 10, h - 30)
            mask[y0 : y0 + 30, 25 : min(75, w)] = 1
            mask_f = np.asfortranarray(mask)
            rle = mask_util.encode(mask_f)

            bw = min(75, w) - 25
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_idx + 1,
                    "category_id": 1,
                    "segmentation": {
                        "size": rle["size"],
                        "counts": rle["counts"].decode("utf-8"),
                    },
                    "bbox": [25, y0, bw, 30],
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


def _create_multi_camera_fixture(
    tmp_path: Path,
    cameras: dict[str, int],
    img_size: tuple[int, int] = (80, 80),
) -> tuple[Path, Path]:
    """Create a COCO fixture with multiple cameras.

    Args:
        tmp_path: Temporary directory.
        cameras: Mapping from camera_id to number of images for that camera.
        img_size: (height, width) for all generated images.

    Returns:
        Tuple of (coco_json_path, image_root_path).
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)

    images = []
    annotations = []
    img_id = 1
    ann_id = 1
    h, w = img_size

    for cam_id, count in cameras.items():
        for i in range(count):
            img_name = f"{cam_id}_{i:03d}.jpg"
            img = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.imwrite(str(images_dir / img_name), img)
            images.append(
                {
                    "id": img_id,
                    "file_name": img_name,
                    "width": w,
                    "height": h,
                    "camera_id": cam_id,
                }
            )

            # One annotation per image
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[10:40, 10:40] = 1
            mask_f = np.asfortranarray(mask)
            rle = mask_util.encode(mask_f)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": {
                        "size": rle["size"],
                        "counts": rle["counts"].decode("utf-8"),
                    },
                    "bbox": [10, 10, 30, 30],
                    "area": float(mask_util.area(rle)),
                    "iscrowd": 0,
                }
            )
            img_id += 1
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


# ===================================================================
# CropDataset tests (existing, for backward compat with Mask R-CNN)
# ===================================================================


class TestCropDatasetBasic:
    """Basic dataset loading and indexing tests."""

    def test_length_matches_images(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path, num_images=3)
        ds = CropDataset(coco_path, image_root)
        assert len(ds) == 3

    def test_getitem_returns_tensor_and_dict(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path)
        ds = CropDataset(coco_path, image_root)
        image, target = ds[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(target, dict)
        assert "boxes" in target
        assert "labels" in target
        assert "masks" in target

    def test_image_tensor_is_float_in_zero_one(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path)
        ds = CropDataset(coco_path, image_root)
        image, _ = ds[0]

        assert image.dtype == torch.float32
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_image_tensor_has_three_channels(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path)
        ds = CropDataset(coco_path, image_root)
        image, _ = ds[0]
        assert image.shape[0] == 3  # C, H, W format

    def test_boxes_format_xyxy(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path)
        ds = CropDataset(coco_path, image_root)
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


class TestCropDatasetVariableSize:
    """Test that crops are loaded at native resolution (no forced 256x256)."""

    def test_native_resolution_preserved_small(self, tmp_path: Path) -> None:
        """Images smaller than 256 should NOT be upscaled."""
        coco_path, image_root = _create_coco_fixture(
            tmp_path, num_images=1, img_size=(64, 80)
        )
        ds = CropDataset(coco_path, image_root)
        image, _ = ds[0]
        # Should be (3, 64, 80) not (3, 256, 256)
        assert image.shape == (3, 64, 80)

    def test_native_resolution_preserved_large(self, tmp_path: Path) -> None:
        """Images larger than 256 should NOT be downscaled."""
        coco_path, image_root = _create_coco_fixture(
            tmp_path, num_images=1, img_size=(400, 300)
        )
        ds = CropDataset(coco_path, image_root)
        image, _ = ds[0]
        assert image.shape == (3, 400, 300)

    def test_different_sizes_in_same_dataset(self, tmp_path: Path) -> None:
        """Dataset can contain crops of different sizes."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Two images with different sizes
        img1 = np.zeros((50, 70, 3), dtype=np.uint8)
        img2 = np.zeros((120, 90, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "img1.jpg"), img1)
        cv2.imwrite(str(images_dir / "img2.jpg"), img2)

        coco: dict = {
            "images": [
                {
                    "id": 1,
                    "file_name": "img1.jpg",
                    "width": 70,
                    "height": 50,
                    "camera_id": "cam01",
                },
                {
                    "id": 2,
                    "file_name": "img2.jpg",
                    "width": 90,
                    "height": 120,
                    "camera_id": "cam01",
                },
            ],
            "annotations": [],
            "categories": [{"id": 1, "name": "fish"}],
        }
        coco_path = tmp_path / "coco.json"
        with open(coco_path, "w") as f:
            json.dump(coco, f)

        ds = CropDataset(coco_path, images_dir)
        img0, _ = ds[0]
        img1_t, _ = ds[1]

        assert img0.shape == (3, 50, 70)
        assert img1_t.shape == (3, 120, 90)

    def test_masks_match_native_image_size(self, tmp_path: Path) -> None:
        """Mask spatial dims should match the native image size."""
        coco_path, image_root = _create_coco_fixture(
            tmp_path, num_images=1, img_size=(80, 60)
        )
        ds = CropDataset(coco_path, image_root)
        image, target = ds[0]

        h, w = image.shape[1], image.shape[2]
        masks = target["masks"]
        assert masks.shape[1] == h
        assert masks.shape[2] == w


class TestCropDatasetNegativeFrames:
    """Test handling of images with no annotations (negative frames)."""

    def test_negative_frame_returns_empty_targets(self, tmp_path: Path) -> None:
        """Image with no annotations returns empty tensors (native size)."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        img = np.zeros((80, 60, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "empty.jpg"), img)

        coco = {
            "images": [
                {
                    "id": 1,
                    "file_name": "empty.jpg",
                    "width": 60,
                    "height": 80,
                    "camera_id": "cam01",
                }
            ],
            "annotations": [],
            "categories": [{"id": 1, "name": "fish"}],
        }
        coco_path = tmp_path / "coco.json"
        with open(coco_path, "w") as f:
            json.dump(coco, f)

        ds = CropDataset(coco_path, images_dir)
        image, target = ds[0]

        # Image at native size (80, 60)
        assert image.shape == (3, 80, 60)
        # Empty targets
        assert target["boxes"].shape == (0, 4)
        assert target["labels"].shape == (0,)
        # Empty masks: (0, H, W) matching image spatial dims
        assert target["masks"].shape[0] == 0
        assert target["masks"].shape[1] == 80
        assert target["masks"].shape[2] == 60

    def test_mixed_positive_negative_dataset(self, tmp_path: Path) -> None:
        """Dataset with both positive and negative images loads correctly."""
        coco_path, image_root = _create_coco_fixture(tmp_path, num_images=2)

        # Add a third image with no annotations (negative)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(image_root / "neg.jpg"), img)

        with open(coco_path) as f:
            coco = json.load(f)
        coco["images"].append(
            {
                "id": 3,
                "file_name": "neg.jpg",
                "width": 100,
                "height": 100,
                "camera_id": "cam01",
            }
        )
        with open(coco_path, "w") as f:
            json.dump(coco, f)

        ds = CropDataset(coco_path, image_root)
        assert len(ds) == 3

        # Last item is the negative frame
        _, target = ds[2]
        assert target["boxes"].shape == (0, 4)


class TestCropDatasetAugmentation:
    """Test that augmentation doesn't crash and produces valid output."""

    def test_augmentation_produces_valid_output(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path)
        ds = CropDataset(coco_path, image_root, augment=True)

        # Run multiple times to hit different augmentation paths
        for _ in range(5):
            image, target = ds[0]
            assert image.dtype == torch.float32
            assert image.shape[0] == 3
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


# ===================================================================
# BinaryMaskDataset tests
# ===================================================================


class TestBinaryMaskDatasetBasic:
    """Basic BinaryMaskDataset loading tests."""

    def test_length_matches_images(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path, num_images=3)
        ds = BinaryMaskDataset(coco_path, image_root)
        assert len(ds) == 3

    def test_getitem_returns_image_and_mask_tensors(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path)
        ds = BinaryMaskDataset(coco_path, image_root)
        image, mask = ds[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_fixed_128_output_shape(self, tmp_path: Path) -> None:
        """Output is always 128x128 regardless of input size."""
        coco_path, image_root = _create_coco_fixture(
            tmp_path, num_images=1, img_size=(64, 80)
        )
        ds = BinaryMaskDataset(coco_path, image_root)
        image, mask = ds[0]

        assert image.shape == (3, 128, 128)
        assert mask.shape == (1, 128, 128)

    def test_image_tensor_is_float_in_zero_one(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path)
        ds = BinaryMaskDataset(coco_path, image_root)
        image, _ = ds[0]

        assert image.dtype == torch.float32
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_mask_tensor_is_binary_float(self, tmp_path: Path) -> None:
        """Mask values should be 0.0 or 1.0."""
        coco_path, image_root = _create_coco_fixture(tmp_path)
        ds = BinaryMaskDataset(coco_path, image_root)
        _, mask = ds[0]

        assert mask.dtype == torch.float32
        unique_vals = set(torch.unique(mask).tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_mask_has_foreground_for_positive_example(self, tmp_path: Path) -> None:
        """Positive examples should have non-zero mask pixels."""
        coco_path, image_root = _create_coco_fixture(tmp_path)
        ds = BinaryMaskDataset(coco_path, image_root)
        _, mask = ds[0]

        assert mask.sum() > 0


class TestBinaryMaskDatasetNegative:
    """Test negative (no-annotation) examples."""

    def test_negative_frame_returns_zero_mask(self, tmp_path: Path) -> None:
        """Image with no annotations returns all-zero mask."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        img = np.zeros((80, 60, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "empty.jpg"), img)

        coco = {
            "images": [
                {
                    "id": 1,
                    "file_name": "empty.jpg",
                    "width": 60,
                    "height": 80,
                    "camera_id": "cam01",
                }
            ],
            "annotations": [],
            "categories": [{"id": 1, "name": "fish"}],
        }
        coco_path = tmp_path / "coco.json"
        with open(coco_path, "w") as f:
            json.dump(coco, f)

        ds = BinaryMaskDataset(coco_path, images_dir)
        image, mask = ds[0]

        assert image.shape == (3, 128, 128)
        assert mask.shape == (1, 128, 128)
        assert mask.sum() == 0.0


class TestBinaryMaskDatasetAugmentation:
    """Test augmentation on BinaryMaskDataset."""

    def test_augmentation_produces_valid_output(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(tmp_path)
        ds = BinaryMaskDataset(coco_path, image_root, augment=True)

        for _ in range(5):
            image, mask = ds[0]
            assert image.shape == (3, 128, 128)
            assert mask.shape == (1, 128, 128)
            assert image.dtype == torch.float32
            assert mask.dtype == torch.float32


class TestBinaryMaskDatasetMergesMasks:
    """Test that multiple annotations are merged into a single binary mask."""

    def test_multiple_annotations_merged(self, tmp_path: Path) -> None:
        coco_path, image_root = _create_coco_fixture(
            tmp_path, num_images=1, num_anns_per_image=2
        )
        ds = BinaryMaskDataset(coco_path, image_root)
        _, mask = ds[0]

        # Single binary mask (merged), not multi-instance
        assert mask.shape == (1, 128, 128)
        assert mask.sum() > 0


# ===================================================================
# stratified_split tests (works with both dataset types)
# ===================================================================


class TestStratifiedSplit:
    """Tests for per-camera stratified split function."""

    def test_returns_all_indices(self, tmp_path: Path) -> None:
        """Train + val indices should cover all dataset indices."""
        cameras = {"cam01": 10, "cam02": 8}
        coco_path, image_root = _create_multi_camera_fixture(tmp_path, cameras)
        ds = CropDataset(coco_path, image_root)

        train_idx, val_idx = stratified_split(ds, val_fraction=0.2)

        all_indices = sorted(train_idx + val_idx)
        assert all_indices == list(range(len(ds)))

    def test_no_overlap_between_splits(self, tmp_path: Path) -> None:
        """Train and val sets must be disjoint."""
        cameras = {"cam01": 10, "cam02": 10}
        coco_path, image_root = _create_multi_camera_fixture(tmp_path, cameras)
        ds = CropDataset(coco_path, image_root)

        train_idx, val_idx = stratified_split(ds, val_fraction=0.2)

        assert len(set(train_idx) & set(val_idx)) == 0

    def test_val_fraction_is_proportional(self, tmp_path: Path) -> None:
        """Val set should be roughly val_fraction of total dataset."""
        cameras = {"cam01": 20, "cam02": 20}
        coco_path, image_root = _create_multi_camera_fixture(tmp_path, cameras)
        ds = CropDataset(coco_path, image_root)

        _train_idx, val_idx = stratified_split(ds, val_fraction=0.2)

        total = len(ds)
        val_ratio = len(val_idx) / total
        # Should be close to 0.2 (within a few % for small datasets)
        assert 0.15 <= val_ratio <= 0.3

    def test_each_camera_represented_in_val(self, tmp_path: Path) -> None:
        """Every camera should contribute at least 1 image to val set."""
        cameras = {"cam01": 10, "cam02": 10, "cam03": 10}
        coco_path, image_root = _create_multi_camera_fixture(tmp_path, cameras)
        ds = CropDataset(coco_path, image_root)

        _, val_idx = stratified_split(ds, val_fraction=0.2)

        # Which cameras appear in val?
        val_cameras = {ds._images[i].get("camera_id") for i in val_idx}
        assert val_cameras == {"cam01", "cam02", "cam03"}

    def test_seed_produces_deterministic_split(self, tmp_path: Path) -> None:
        """Same seed should produce identical splits."""
        cameras = {"cam01": 15, "cam02": 15}
        coco_path, image_root = _create_multi_camera_fixture(tmp_path, cameras)
        ds = CropDataset(coco_path, image_root)

        train1, val1 = stratified_split(ds, val_fraction=0.2, seed=42)
        train2, val2 = stratified_split(ds, val_fraction=0.2, seed=42)

        assert sorted(train1) == sorted(train2)
        assert sorted(val1) == sorted(val2)

    def test_different_seeds_produce_different_splits(self, tmp_path: Path) -> None:
        """Different seeds should (typically) produce different splits."""
        cameras = {"cam01": 20, "cam02": 20}
        coco_path, image_root = _create_multi_camera_fixture(tmp_path, cameras)
        ds = CropDataset(coco_path, image_root)

        _, val1 = stratified_split(ds, val_fraction=0.2, seed=42)
        _, val2 = stratified_split(ds, val_fraction=0.2, seed=99)

        assert sorted(val1) != sorted(val2)

    def test_single_camera_dataset(self, tmp_path: Path) -> None:
        """Single-camera dataset is split like a standard holdout."""
        cameras = {"cam01": 10}
        coco_path, image_root = _create_multi_camera_fixture(tmp_path, cameras)
        ds = CropDataset(coco_path, image_root)

        train_idx, val_idx = stratified_split(ds, val_fraction=0.2)

        assert len(train_idx) + len(val_idx) == 10
        assert len(val_idx) >= 1

    def test_stratified_split_with_binary_mask_dataset(self, tmp_path: Path) -> None:
        """stratified_split works with BinaryMaskDataset too."""
        cameras = {"cam01": 10, "cam02": 8}
        coco_path, image_root = _create_multi_camera_fixture(tmp_path, cameras)
        ds = BinaryMaskDataset(coco_path, image_root)

        train_idx, val_idx = stratified_split(ds, val_fraction=0.2)

        all_indices = sorted(train_idx + val_idx)
        assert all_indices == list(range(len(ds)))
