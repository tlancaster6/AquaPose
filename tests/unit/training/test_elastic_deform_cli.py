"""Tests for elastic deformation CLI and YOLO output writer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


def _make_yolo_dir(tmp_path: Path) -> Path:
    """Create a minimal YOLO directory with 1 synthetic image and pose label."""
    input_dir = tmp_path / "input"
    img_dir = input_dir / "images" / "train"
    lbl_dir = input_dir / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Synthetic 100x60 image
    img = np.random.randint(0, 255, (60, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "000001_cam01.jpg"), img)

    # Pose label: cls cx cy w h x1 y1 v1 ... (6 keypoints, all visible)
    # Keypoints in normalized coords across a 100x60 image
    coords_norm = [
        (0.15, 0.5, 2),
        (0.30, 0.5, 2),
        (0.45, 0.5, 2),
        (0.60, 0.5, 2),
        (0.75, 0.5, 2),
        (0.90, 0.5, 2),
    ]
    parts = ["0", "0.5", "0.5", "0.8", "0.6"]  # cls cx cy w h
    for x, y, v in coords_norm:
        parts.extend([str(x), str(y), str(v)])
    label_line = " ".join(parts)
    (lbl_dir / "000001_cam01.txt").write_text(label_line + "\n")

    return input_dir


class TestWriteYoloDataset:
    """Tests for write_yolo_dataset output structure."""

    def test_output_file_counts(self, tmp_path: Path) -> None:
        from aquapose.training.elastic_deform import write_yolo_dataset

        input_dir = _make_yolo_dir(tmp_path)
        output_dir = tmp_path / "output"
        write_yolo_dataset(input_dir, output_dir, lateral_pad=15.0)

        out_imgs = list((output_dir / "images" / "train").glob("*.jpg"))
        out_lbls = list((output_dir / "labels" / "train").glob("*.txt"))
        # 1 original + 4 variants = 5
        assert len(out_imgs) == 5
        assert len(out_lbls) == 5

    def test_dataset_yaml_valid(self, tmp_path: Path) -> None:
        from aquapose.training.elastic_deform import write_yolo_dataset

        input_dir = _make_yolo_dir(tmp_path)
        output_dir = tmp_path / "output"
        write_yolo_dataset(input_dir, output_dir, lateral_pad=15.0)

        yaml_path = output_dir / "dataset.yaml"
        assert yaml_path.exists()
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        assert "path" in data
        assert data["train"] == "images/train"
        assert data["names"] == {0: "fish"}
        assert data["nc"] == 1
        assert data["kpt_shape"] == [6, 3]

    def test_variant_filenames_have_tags(self, tmp_path: Path) -> None:
        from aquapose.training.elastic_deform import write_yolo_dataset

        input_dir = _make_yolo_dir(tmp_path)
        output_dir = tmp_path / "output"
        write_yolo_dataset(input_dir, output_dir, lateral_pad=15.0)

        stems = {p.stem for p in (output_dir / "images" / "train").glob("*.jpg")}
        assert "000001_cam01" in stems  # original
        assert "000001_cam01_c_pos" in stems
        assert "000001_cam01_c_neg" in stems
        assert "000001_cam01_s_pos" in stems
        assert "000001_cam01_s_neg" in stems

    def test_original_image_copied(self, tmp_path: Path) -> None:
        from aquapose.training.elastic_deform import write_yolo_dataset

        input_dir = _make_yolo_dir(tmp_path)
        output_dir = tmp_path / "output"
        write_yolo_dataset(input_dir, output_dir, lateral_pad=15.0)

        # Original image should be copied
        orig_img = output_dir / "images" / "train" / "000001_cam01.jpg"
        assert orig_img.exists()
        # Original label should be copied
        orig_lbl = output_dir / "labels" / "train" / "000001_cam01.txt"
        assert orig_lbl.exists()

    def test_label_values_normalized(self, tmp_path: Path) -> None:
        from aquapose.training.elastic_deform import write_yolo_dataset

        input_dir = _make_yolo_dir(tmp_path)
        output_dir = tmp_path / "output"
        write_yolo_dataset(input_dir, output_dir, lateral_pad=15.0)

        # Check a variant label has values in [0, 1] (except class and visibility)
        lbl_path = output_dir / "labels" / "train" / "000001_cam01_c_pos.txt"
        line = lbl_path.read_text().strip()
        vals = [float(v) for v in line.split()]
        # cls is 0, then cx cy w h (normalized), then x y v triplets
        for i, val in enumerate(vals):
            if i == 0:
                continue  # class id
            if (i - 5) >= 0 and (i - 5) % 3 == 2:
                continue  # visibility flag
            assert 0.0 <= val <= 1.0, f"Label value at index {i} = {val} out of [0,1]"


class TestParsePoseLabel:
    """Tests for parse_pose_label."""

    def test_parse_valid_label(self, tmp_path: Path) -> None:
        from aquapose.training.elastic_deform import parse_pose_label

        label_path = tmp_path / "test.txt"
        # cls cx cy w h x1 y1 v1 x2 y2 v2 ... (6 keypoints)
        parts = ["0", "0.5", "0.5", "0.8", "0.6"]
        for i in range(6):
            parts.extend([str(0.1 + i * 0.15), "0.5", "2"])
        label_path.write_text(" ".join(parts) + "\n")

        coords, visible = parse_pose_label(label_path, 100, 60)
        assert coords.shape == (6, 2)
        assert visible.shape == (6,)
        assert visible.all()
        # Coords should be denormalized to pixel space
        assert coords[0, 0] == pytest.approx(10.0, abs=0.1)  # 0.1 * 100
