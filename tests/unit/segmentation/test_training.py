"""Unit tests for U-Net training (migrated from segmentation.training to training.unet)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util

from aquapose.training.unet import train_unet


def _create_training_fixture(
    tmp_path: Path,
    num_images: int = 3,
    variable_sizes: bool = False,
    camera_ids: list[str] | None = None,
) -> Path:
    """Create a minimal COCO fixture for training tests using data_dir convention.

    Creates images in tmp_path/images/ and writes annotations.json into tmp_path.

    Args:
        tmp_path: Temporary directory for test files.
        num_images: Number of images to create.
        variable_sizes: If True, each image has a different (H, W).
        camera_ids: Optional per-image camera_id values for stratified split tests.

    Returns:
        data_dir (tmp_path) where annotations.json and images/ reside.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)

    images = []
    annotations = []
    ann_id = 1

    base_sizes = [(64, 64), (80, 96), (72, 56), (88, 104), (60, 68)]

    for img_idx in range(num_images):
        img_name = f"images/frame_{img_idx:03d}.jpg"
        if variable_sizes:
            h, w = base_sizes[img_idx % len(base_sizes)]
        else:
            h, w = 64, 64

        # Create image with a visible blob (not pure noise)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[10 : h - 10, 10 : w - 10] = 200  # bright rectangle
        cv2.imwrite(str(tmp_path / img_name), img)

        img_entry: dict = {
            "id": img_idx + 1,
            "file_name": img_name,
            "width": w,
            "height": h,
        }
        if camera_ids is not None:
            img_entry["camera_id"] = camera_ids[img_idx % len(camera_ids)]
        images.append(img_entry)

        # Create mask matching the bright region
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[10 : h - 10, 10 : w - 10] = 1
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
                "bbox": [10, 10, w - 20, h - 20],
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

    coco_path = tmp_path / "annotations.json"
    with open(coco_path, "w") as f:
        json.dump(coco, f)

    return tmp_path


def _create_presplit_fixture(
    tmp_path: Path,
    num_images: int = 4,
) -> Path:
    """Create a COCO fixture with pre-split train.json and val.json.

    Args:
        tmp_path: Temporary directory.
        num_images: Total number of images (at least 2).

    Returns:
        data_dir with train.json, val.json, and images/.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)

    all_images = []
    annotations = []
    ann_id = 1

    for img_idx in range(num_images):
        img_name = f"images/frame_{img_idx:03d}.jpg"
        h, w = 64, 64
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[10:54, 10:54] = 200
        cv2.imwrite(str(tmp_path / img_name), img)

        all_images.append(
            {"id": img_idx + 1, "file_name": img_name, "width": w, "height": h}
        )

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[10:54, 10:54] = 1
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
                "bbox": [10, 10, 44, 44],
                "area": float(mask_util.area(rle)),
                "iscrowd": 0,
            }
        )
        ann_id += 1

    val_img_ids = {all_images[0]["id"]}
    train_img_ids = {img["id"] for img in all_images[1:]}

    def _filter(ids: set[int]) -> dict:
        imgs = [i for i in all_images if i["id"] in ids]
        anns = [a for a in annotations if a["image_id"] in ids]
        return {
            "images": imgs,
            "annotations": anns,
            "categories": [{"id": 1, "name": "fish"}],
        }

    with open(tmp_path / "train.json", "w") as f:
        json.dump(_filter(train_img_ids), f)
    with open(tmp_path / "val.json", "w") as f:
        json.dump(_filter(val_img_ids), f)

    return tmp_path


class TestTrainUnet:
    """Tests for train_unet() using the data_dir convention."""

    def test_train_one_epoch(self, tmp_path: Path) -> None:
        """Train for 1 epoch and verify output model file exists."""
        data_dir = _create_training_fixture(tmp_path)
        output_dir = tmp_path / "output"

        best_model = train_unet(
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=1,
            batch_size=1,
            num_workers=0,
            device="cpu",
        )

        assert isinstance(best_model, Path)
        assert best_model.exists()
        assert (output_dir / "last_model.pth").exists()

    def test_train_uses_stratified_split(self, tmp_path: Path) -> None:
        """train_unet() without pre-split JSON uses stratified_split."""
        data_dir = _create_training_fixture(
            tmp_path,
            num_images=6,
            camera_ids=["cam_a", "cam_b"],
        )
        output_dir = tmp_path / "output"

        best_model = train_unet(
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=1,
            batch_size=1,
            val_split=0.33,
            num_workers=0,
            device="cpu",
        )
        assert best_model.exists()

    def test_train_with_presplit_json(self, tmp_path: Path) -> None:
        """train_unet() with train.json/val.json uses those files directly."""
        data_dir = _create_presplit_fixture(tmp_path, num_images=4)
        output_dir = tmp_path / "output"

        best_model = train_unet(
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=1,
            batch_size=1,
            num_workers=0,
            device="cpu",
        )
        assert best_model.exists()

    def test_train_variable_size_crops(self, tmp_path: Path) -> None:
        """Training works when dataset contains crops of different sizes."""
        data_dir = _create_training_fixture(tmp_path, num_images=4, variable_sizes=True)
        output_dir = tmp_path / "output"

        best_model = train_unet(
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=1,
            batch_size=2,
            num_workers=0,
            device="cpu",
        )
        assert best_model.exists()

    def test_train_metrics_csv_created(self, tmp_path: Path) -> None:
        """train_unet() should create a metrics.csv in output_dir."""
        data_dir = _create_training_fixture(tmp_path)
        output_dir = tmp_path / "output"

        train_unet(
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=1,
            batch_size=1,
            num_workers=0,
            device="cpu",
        )
        assert (output_dir / "metrics.csv").exists()
