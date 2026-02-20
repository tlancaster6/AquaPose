"""Unit tests for Mask R-CNN training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import pytest

from aquapose.segmentation.training import evaluate, train


def _create_training_fixture(tmp_path: Path, num_images: int = 3) -> tuple[Path, Path]:
    """Create a minimal COCO fixture for training tests.

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
        h, w = 64, 64
        # Create image with a visible blob (not pure noise)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[20:50, 15:50] = 200  # bright rectangle
        cv2.imwrite(str(images_dir / img_name), img)

        images.append(
            {"id": img_idx + 1, "file_name": img_name, "width": w, "height": h}
        )

        # Create mask matching the bright region
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[20:50, 15:50] = 1
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
                "bbox": [15, 20, 35, 30],
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


@pytest.mark.slow
class TestTrain:
    """Tests for the train() function (slow -- instantiates Mask R-CNN)."""

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
            device="cpu",
        )

        assert isinstance(best_model, Path)
        assert best_model.exists()
        assert (output_dir / "final_model.pth").exists()


@pytest.mark.slow
class TestEvaluate:
    """Tests for the evaluate() function (slow -- loads Mask R-CNN)."""

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
