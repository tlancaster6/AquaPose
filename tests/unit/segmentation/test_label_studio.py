"""Unit tests for Label Studio export/import utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aquapose.segmentation.label_studio import (
    export_to_label_studio,
    import_from_label_studio,
    to_coco_dataset,
)
from aquapose.segmentation.pseudo_labeler import AnnotatedFrame, FrameAnnotation


@pytest.fixture
def sample_mask() -> np.ndarray:
    """A simple binary mask for testing."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 30:70] = 255
    return mask


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a dummy image file for testing."""
    import cv2

    img_path = tmp_path / "test_frame.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture
def sample_frames(sample_mask: np.ndarray, sample_image: Path) -> list[FrameAnnotation]:
    """Sample FrameAnnotation list for export testing."""
    return [
        FrameAnnotation(
            frame_id="cam01_f001",
            image_path=sample_image,
            masks=[sample_mask],
            camera_id="cam01",
        ),
        FrameAnnotation(
            frame_id="cam01_f002",
            image_path=sample_image,
            masks=[sample_mask, sample_mask],
            camera_id="cam01",
        ),
    ]


class TestExportToLabelStudio:
    """Tests for Label Studio task JSON export."""

    def test_creates_tasks_json(
        self, tmp_path: Path, sample_frames: list[FrameAnnotation]
    ) -> None:
        tasks_path = export_to_label_studio(
            sample_frames, tmp_path / "output", project_name="test"
        )
        assert tasks_path.exists()
        assert tasks_path.name == "test_tasks.json"

    def test_tasks_json_structure(
        self, tmp_path: Path, sample_frames: list[FrameAnnotation]
    ) -> None:
        tasks_path = export_to_label_studio(sample_frames, tmp_path / "output")
        with open(tasks_path) as f:
            tasks = json.load(f)

        assert isinstance(tasks, list)
        assert len(tasks) == 2

        # First task should have one prediction result
        task0 = tasks[0]
        assert "data" in task0
        assert "predictions" in task0
        assert len(task0["predictions"]) == 1
        results = task0["predictions"][0]["result"]
        assert len(results) == 1
        assert results[0]["type"] == "brushlabels"
        assert results[0]["value"]["format"] == "rle"
        assert "rle" in results[0]["value"]
        assert results[0]["value"]["brushlabels"] == ["fish"]

    def test_multiple_masks_in_single_frame(
        self, tmp_path: Path, sample_frames: list[FrameAnnotation]
    ) -> None:
        tasks_path = export_to_label_studio(sample_frames, tmp_path / "output")
        with open(tasks_path) as f:
            tasks = json.load(f)

        # Second task has 2 masks
        task1 = tasks[1]
        results = task1["predictions"][0]["result"]
        assert len(results) == 2

    def test_negative_frame_has_empty_predictions(
        self, tmp_path: Path, sample_image: Path
    ) -> None:
        frames = [
            FrameAnnotation(
                frame_id="cam01_f003",
                image_path=sample_image,
                masks=[],
                camera_id="cam01",
            )
        ]
        tasks_path = export_to_label_studio(frames, tmp_path / "output")
        with open(tasks_path) as f:
            tasks = json.load(f)

        assert len(tasks) == 1
        assert tasks[0]["predictions"] == []

    def test_images_copied_to_output(
        self, tmp_path: Path, sample_frames: list[FrameAnnotation]
    ) -> None:
        export_to_label_studio(sample_frames, tmp_path / "output")
        images_dir = tmp_path / "output" / "images"
        assert images_dir.exists()
        assert (images_dir / "test_frame.jpg").exists()

    def test_data_contains_metadata(
        self, tmp_path: Path, sample_frames: list[FrameAnnotation]
    ) -> None:
        tasks_path = export_to_label_studio(sample_frames, tmp_path / "output")
        with open(tasks_path) as f:
            tasks = json.load(f)

        assert tasks[0]["data"]["frame_id"] == "cam01_f001"
        assert tasks[0]["data"]["camera_id"] == "cam01"


class TestImportFromLabelStudio:
    """Tests for Label Studio annotation import."""

    def test_import_basic_structure(self, tmp_path: Path) -> None:
        """Test import of a basic Label Studio export."""
        from label_studio_converter.brush import mask2rle

        # Create a synthetic mask and encode it
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 255
        rle = mask2rle(mask)

        ls_export = [
            {
                "data": {
                    "image": "/data/local-files/?d=images/frame.jpg",
                    "frame_id": "cam01_f001",
                    "camera_id": "cam01",
                },
                "annotations": [
                    {
                        "result": [
                            {
                                "type": "brushlabels",
                                "value": {
                                    "format": "rle",
                                    "rle": rle,
                                    "brushlabels": ["fish"],
                                },
                                "original_width": 100,
                                "original_height": 100,
                                "from_name": "label",
                                "to_name": "image",
                            }
                        ]
                    }
                ],
            }
        ]

        export_path = tmp_path / "export.json"
        with open(export_path, "w") as f:
            json.dump(ls_export, f)

        frames = import_from_label_studio(export_path)
        assert len(frames) == 1
        assert frames[0].frame_id == "cam01_f001"
        assert frames[0].camera_id == "cam01"
        assert len(frames[0].masks) == 1
        assert frames[0].masks[0].dtype == np.uint8

    def test_import_empty_annotations(self, tmp_path: Path) -> None:
        """Negative frames with no annotations produce empty mask list."""
        ls_export = [
            {
                "data": {
                    "image": "/data/local-files/?d=images/frame.jpg",
                    "frame_id": "cam01_f003",
                    "camera_id": "cam01",
                },
                "annotations": [],
            }
        ]

        export_path = tmp_path / "export.json"
        with open(export_path, "w") as f:
            json.dump(ls_export, f)

        frames = import_from_label_studio(export_path)
        assert len(frames) == 1
        assert frames[0].masks == []


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
