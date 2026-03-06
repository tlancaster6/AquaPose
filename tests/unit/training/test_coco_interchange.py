"""Unit tests for COCO Keypoints interchange format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from aquapose.training.coco_interchange import (
    coco_to_yolo_pose,
    write_coco_keypoints,
    yolo_pose_to_coco,
)
from PIL import Image

N_KP = 13


def _make_yolo_pose_dir(tmp_path: Path, n_kp: int = N_KP) -> Path:
    """Create a minimal YOLO-Pose directory with 1 image and 2 fish."""
    pose_dir = tmp_path / "pose"
    img_dir = pose_dir / "images" / "train"
    lbl_dir = pose_dir / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Create a small dummy JPEG image (128x64)
    img = Image.new("RGB", (128, 64), color="blue")
    img.save(img_dir / "000001_cam0_000.jpg")

    # Two fish annotations: class cx cy w h  (then n_kp * (x y v))
    # Fish 1: bbox centered at (0.5, 0.5), size (0.4, 0.3)
    kp1 = []
    for i in range(n_kp):
        # Visible keypoints at various normalized positions
        kp1.extend([0.1 + i * 0.06, 0.2 + i * 0.04, 2])
    line1 = "0 0.5 0.5 0.4 0.3 " + " ".join(f"{v}" for v in kp1)

    # Fish 2: bbox at (0.3, 0.7), size (0.2, 0.15)
    kp2 = []
    for i in range(n_kp):
        kp2.extend([0.2 + i * 0.05, 0.3 + i * 0.03, 2])
    line2 = "0 0.3 0.7 0.2 0.15 " + " ".join(f"{v}" for v in kp2)

    (lbl_dir / "000001_cam0_000.txt").write_text(f"{line1}\n{line2}\n")

    return pose_dir


def test_yolo_pose_to_coco_basic(tmp_path: Path) -> None:
    """Convert minimal YOLO-Pose dir and verify COCO structure."""
    pose_dir = _make_yolo_pose_dir(tmp_path)
    coco = yolo_pose_to_coco(pose_dir, N_KP)

    # Structure checks
    assert "images" in coco
    assert "annotations" in coco
    assert "categories" in coco

    # 1 image
    assert len(coco["images"]) == 1
    img_entry = coco["images"][0]
    assert img_entry["width"] == 128
    assert img_entry["height"] == 64
    assert "000001_cam0_000.jpg" in img_entry["file_name"]

    # 2 annotations
    assert len(coco["annotations"]) == 2

    ann = coco["annotations"][0]
    # bbox should be absolute pixels [x_min, y_min, w, h]
    # Original: cx=0.5, cy=0.5, w=0.4, h=0.3 on 128x64
    # x_min = (0.5 - 0.2) * 128 = 38.4, y_min = (0.5 - 0.15) * 64 = 22.4
    # w = 0.4 * 128 = 51.2, h = 0.3 * 64 = 19.2
    assert len(ann["bbox"]) == 4
    np.testing.assert_allclose(ann["bbox"], [38.4, 22.4, 51.2, 19.2], atol=1e-5)

    # Keypoints should be absolute pixels, length = n_kp * 3
    assert len(ann["keypoints"]) == N_KP * 3
    # First keypoint: (0.1 * 128, 0.2 * 64, 2) = (12.8, 12.8, 2)
    np.testing.assert_allclose(ann["keypoints"][:3], [12.8, 12.8, 2], atol=1e-5)

    # Category
    assert len(coco["categories"]) == 1
    cat = coco["categories"][0]
    assert len(cat["keypoints"]) == N_KP
    assert cat["keypoints"][0] == "kp_0"
    assert cat["skeleton"] == []

    # IDs are sequential
    assert ann["id"] == 1
    assert coco["annotations"][1]["id"] == 2


def test_yolo_pose_to_coco_invisible_keypoints(tmp_path: Path) -> None:
    """YOLO '0 0 0' entries map to COCO '0 0 0' (not-labeled)."""
    pose_dir = tmp_path / "pose"
    img_dir = pose_dir / "images" / "train"
    lbl_dir = pose_dir / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_dir / "test.jpg")

    # 1 fish, first 3 keypoints invisible, rest visible
    parts = ["0", "0.5", "0.5", "0.4", "0.3"]
    for i in range(N_KP):
        if i < 3:
            parts.extend(["0", "0", "0"])
        else:
            parts.extend([f"{0.1 + i * 0.05}", f"{0.2 + i * 0.04}", "2"])
    (lbl_dir / "test.txt").write_text(" ".join(parts) + "\n")

    coco = yolo_pose_to_coco(pose_dir, N_KP)
    ann = coco["annotations"][0]
    kps = ann["keypoints"]

    # First 3 keypoints should be (0, 0, 0)
    for i in range(3):
        assert kps[i * 3] == 0
        assert kps[i * 3 + 1] == 0
        assert kps[i * 3 + 2] == 0

    # 4th keypoint should be visible (v=2) with non-zero coords
    assert kps[9 + 2] == 2
    assert kps[9] > 0  # x
    assert kps[10] > 0  # y


def test_coco_to_yolo_pose_basic(tmp_path: Path) -> None:
    """Convert COCO JSON with 2 images and 3 annotations to YOLO labels."""
    coco = {
        "images": [
            {"id": 1, "file_name": "img_a.jpg", "width": 200, "height": 100},
            {"id": 2, "file_name": "img_b.jpg", "width": 400, "height": 200},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [50, 20, 40, 30],  # absolute [x_min, y_min, w, h]
                "keypoints": [100, 50, 2, 120, 60, 2, 0, 0, 0],
                "num_keypoints": 2,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "keypoints": [20, 20, 2, 0, 0, 0, 30, 30, 2],
                "num_keypoints": 2,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [100, 50, 80, 60],
                "keypoints": [200, 100, 2, 0, 0, 0, 300, 150, 2],
                "num_keypoints": 2,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "fish", "keypoints": ["kp_0", "kp_1", "kp_2"]}
        ],
    }
    coco_path = tmp_path / "coco_keypoints.json"
    coco_path.write_text(json.dumps(coco))

    out_dir = tmp_path / "labels"
    out_dir.mkdir()

    n = coco_to_yolo_pose(coco_path, out_dir)
    assert n == 2  # 2 images -> 2 files

    # Check img_a.txt
    lines_a = (out_dir / "img_a.txt").read_text().strip().split("\n")
    assert len(lines_a) == 2  # 2 annotations for image 1

    parts = lines_a[0].split()
    # class=0, cx=(50+20)/200=0.35, cy=(20+15)/100=0.35, w=40/200=0.2, h=30/100=0.3
    np.testing.assert_allclose(float(parts[1]), 0.35, atol=1e-5)  # cx
    np.testing.assert_allclose(float(parts[2]), 0.35, atol=1e-5)  # cy
    np.testing.assert_allclose(float(parts[3]), 0.2, atol=1e-5)  # w
    np.testing.assert_allclose(float(parts[4]), 0.3, atol=1e-5)  # h
    # kp0: (100/200, 50/100, 2) = (0.5, 0.5, 2)
    np.testing.assert_allclose(float(parts[5]), 0.5, atol=1e-5)
    np.testing.assert_allclose(float(parts[6]), 0.5, atol=1e-5)
    assert int(float(parts[7])) == 2

    # Check img_b.txt
    lines_b = (out_dir / "img_b.txt").read_text().strip().split("\n")
    assert len(lines_b) == 1  # 1 annotation for image 2


def test_coco_to_yolo_pose_invisible(tmp_path: Path) -> None:
    """COCO visibility=0 maps to YOLO '0 0 0'."""
    coco = {
        "images": [
            {"id": 1, "file_name": "img.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 30, 30],
                "keypoints": [50, 50, 2, 0, 0, 0, 70, 70, 1],
                "num_keypoints": 2,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "fish", "keypoints": ["kp_0", "kp_1", "kp_2"]}
        ],
    }
    coco_path = tmp_path / "coco.json"
    coco_path.write_text(json.dumps(coco))
    out_dir = tmp_path / "labels"
    out_dir.mkdir()

    coco_to_yolo_pose(coco_path, out_dir)

    parts = (out_dir / "img.txt").read_text().strip().split()
    # kp_1 was (0,0,0) -> YOLO should also be (0,0,0)
    # kp_1 starts at index 5 + 3 = 8
    np.testing.assert_allclose(float(parts[8]), 0.0, atol=1e-9)
    np.testing.assert_allclose(float(parts[9]), 0.0, atol=1e-9)
    np.testing.assert_allclose(float(parts[10]), 0.0, atol=1e-9)


def test_round_trip(tmp_path: Path) -> None:
    """YOLO -> COCO -> YOLO produces identical labels within float tolerance."""
    pose_dir = _make_yolo_pose_dir(tmp_path)

    # Read original labels
    orig_label = (pose_dir / "labels" / "train" / "000001_cam0_000.txt").read_text()

    # YOLO -> COCO
    coco_path = write_coco_keypoints(pose_dir, N_KP)
    assert coco_path.exists()

    # COCO -> YOLO
    rt_labels_dir = tmp_path / "round_trip_labels"
    rt_labels_dir.mkdir()
    coco_to_yolo_pose(coco_path, rt_labels_dir)

    # Compare
    rt_label = (rt_labels_dir / "000001_cam0_000.txt").read_text()

    orig_lines = orig_label.strip().split("\n")
    rt_lines = rt_label.strip().split("\n")
    assert len(orig_lines) == len(rt_lines)

    for orig_line, rt_line in zip(orig_lines, rt_lines, strict=True):
        orig_vals = [float(v) for v in orig_line.split()]
        rt_vals = [float(v) for v in rt_line.split()]
        np.testing.assert_allclose(orig_vals, rt_vals, atol=1e-5)
