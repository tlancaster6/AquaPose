"""Unit tests for U-Net training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util

from aquapose.segmentation.training import evaluate, train


def _create_training_fixture(
    tmp_path: Path,
    num_images: int = 3,
    variable_sizes: bool = False,
    camera_ids: list[str] | None = None,
) -> tuple[Path, Path]:
    """Create a minimal COCO fixture for training tests.

    Args:
        tmp_path: Temporary directory for test files.
        num_images: Number of images to create.
        variable_sizes: If True, each image has a different (H, W).
        camera_ids: Optional per-image camera_id values for stratified split tests.

    Returns:
        Tuple of (coco_json_path, image_root_path).
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    images = []
    annotations = []
    ann_id = 1

    base_sizes = [(64, 64), (80, 96), (72, 56), (88, 104), (60, 68)]

    for img_idx in range(num_images):
        img_name = f"frame_{img_idx:03d}.jpg"
        if variable_sizes:
            h, w = base_sizes[img_idx % len(base_sizes)]
        else:
            h, w = 64, 64

        # Create image with a visible blob (not pure noise)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[10 : h - 10, 10 : w - 10] = 200  # bright rectangle
        cv2.imwrite(str(images_dir / img_name), img)

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

    coco_path = tmp_path / "coco.json"
    with open(coco_path, "w") as f:
        json.dump(coco, f)

    return coco_path, images_dir


def _split_coco_json(
    coco_path: Path,
    train_path: Path,
    val_path: Path,
    n_val: int = 1,
) -> None:
    """Split a COCO JSON into train and val files for testing."""
    with open(coco_path) as f:
        coco = json.load(f)

    all_img_ids = [img["id"] for img in coco["images"]]
    val_ids = set(all_img_ids[:n_val])
    train_ids = set(all_img_ids[n_val:])

    def _filter(coco_data: dict, keep_ids: set[int]) -> dict:
        imgs = [i for i in coco_data["images"] if i["id"] in keep_ids]
        anns = [a for a in coco_data["annotations"] if a["image_id"] in keep_ids]
        return {
            "images": imgs,
            "annotations": anns,
            "categories": coco_data["categories"],
        }

    with open(train_path, "w") as f:
        json.dump(_filter(coco, train_ids), f)
    with open(val_path, "w") as f:
        json.dump(_filter(coco, val_ids), f)


class TestTrain:
    """Tests for the train() function."""

    def test_train_one_epoch(self, tmp_path: Path) -> None:
        """Train for 1 epoch and verify output model file exists."""
        coco_path, image_root = _create_training_fixture(tmp_path)
        output_dir = tmp_path / "output"

        best_model = train(
            coco_path,
            image_root,
            output_dir,
            epochs=1,
            batch_size=1,
            num_workers=0,
            device="cpu",
        )

        assert isinstance(best_model, Path)
        assert best_model.exists()
        assert (output_dir / "final_model.pth").exists()

    def test_train_uses_stratified_split(self, tmp_path: Path) -> None:
        """train() without pre-split JSON uses stratified_split."""
        coco_path, image_root = _create_training_fixture(
            tmp_path,
            num_images=6,
            camera_ids=["cam_a", "cam_b"],
        )
        output_dir = tmp_path / "output"

        best_model = train(
            coco_path,
            image_root,
            output_dir,
            epochs=1,
            batch_size=1,
            val_split=0.33,
            num_workers=0,
            device="cpu",
        )
        assert best_model.exists()

    def test_train_with_presplit_json(self, tmp_path: Path) -> None:
        """train() with train_json/val_json uses those files directly."""
        coco_path, image_root = _create_training_fixture(tmp_path, num_images=4)
        train_json = tmp_path / "train.json"
        val_json = tmp_path / "val.json"
        _split_coco_json(coco_path, train_json, val_json, n_val=1)

        output_dir = tmp_path / "output"
        best_model = train(
            coco_path,
            image_root,
            output_dir,
            train_json=train_json,
            val_json=val_json,
            epochs=1,
            batch_size=1,
            num_workers=0,
            device="cpu",
        )
        assert best_model.exists()

    def test_train_variable_size_crops(self, tmp_path: Path) -> None:
        """Training works when dataset contains crops of different sizes."""
        coco_path, image_root = _create_training_fixture(
            tmp_path, num_images=4, variable_sizes=True
        )
        output_dir = tmp_path / "output"

        best_model = train(
            coco_path,
            image_root,
            output_dir,
            epochs=1,
            batch_size=2,
            num_workers=0,
            device="cpu",
        )
        assert best_model.exists()


class TestEvaluate:
    """Tests for the evaluate() function."""

    def test_evaluate_returns_expected_keys(self, tmp_path: Path) -> None:
        """Evaluate after 1-epoch training and check return dict structure."""
        coco_path, image_root = _create_training_fixture(tmp_path)
        output_dir = tmp_path / "output"

        best_model = train(
            coco_path,
            image_root,
            output_dir,
            epochs=1,
            batch_size=1,
            num_workers=0,
            device="cpu",
        )

        result = evaluate(best_model, coco_path, image_root, device="cpu")

        assert "mean_iou" in result
        assert "per_image_iou" in result
        assert "num_images" in result
        assert isinstance(result["mean_iou"], float)
        assert isinstance(result["per_image_iou"], list)
        assert isinstance(result["num_images"], int)
        assert result["num_images"] > 0

    def test_evaluate_variable_size_crops(self, tmp_path: Path) -> None:
        """evaluate() works with datasets containing different-sized crop images."""
        coco_path, image_root = _create_training_fixture(
            tmp_path, num_images=4, variable_sizes=True
        )
        output_dir = tmp_path / "output"

        best_model = train(
            coco_path,
            image_root,
            output_dir,
            epochs=1,
            batch_size=1,
            num_workers=0,
            device="cpu",
        )

        result = evaluate(best_model, coco_path, image_root, device="cpu")
        assert result["num_images"] == 4
