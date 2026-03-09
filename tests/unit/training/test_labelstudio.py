"""Tests for Label Studio export/import roundtrip."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from PIL import Image

from aquapose.training.coco_convert import KEYPOINT_NAMES
from aquapose.training.labelstudio_export import export_labelstudio_json
from aquapose.training.labelstudio_import import import_labelstudio_json


def _make_rect_corners(
    cx_px: float,
    cy_px: float,
    w_px: float,
    h_px: float,
    angle_deg: float,
    img_w: int,
    img_h: int,
) -> str:
    """Generate OBB corners (TL,TR,BR,BL) for a true rectangle in pixel space.

    Returns a YOLO-OBB label line with normalized coordinates.
    """
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    # Half-extents along rotated axes (pixel space)
    dx_w, dy_w = w_px / 2 * cos_a, w_px / 2 * sin_a
    dx_h, dy_h = -h_px / 2 * sin_a, h_px / 2 * cos_a
    tl = (cx_px - dx_w - dx_h, cy_px - dy_w - dy_h)
    tr = (cx_px + dx_w - dx_h, cy_px + dy_w - dy_h)
    br = (cx_px + dx_w + dx_h, cy_px + dy_w + dy_h)
    bl = (cx_px - dx_w + dx_h, cy_px - dy_w + dy_h)
    # Normalize
    coords = " ".join(
        f"{pt[0] / img_w:.6f} {pt[1] / img_h:.6f}" for pt in [tl, tr, br, bl]
    )
    return f"0 {coords}"


@pytest.fixture
def obb_dataset(tmp_path):
    """Create a minimal YOLO-OBB dataset with a true rectangle."""
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    img = Image.new("RGB", (200, 100), color="black")
    img.save(img_dir / "test.jpg")

    # True rectangle in pixel space: center=(60,40), 80x20px, rotated 15°
    line = _make_rect_corners(60, 40, 80, 20, 15.0, 200, 100)
    lbl_dir.joinpath("test.txt").write_text(line + "\n")
    return tmp_path


@pytest.fixture
def pose_dataset(tmp_path):
    """Create a minimal YOLO-Pose dataset."""
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    img = Image.new("RGB", (128, 64), color="black")
    img.save(img_dir / "test.jpg")

    # Pose label: class cx cy w h  kp0_x kp0_y kp0_v ... kp5
    lbl_dir.joinpath("test.txt").write_text(
        "0 0.5 0.5 0.8 0.6 "
        "0.1 0.5 2 0.25 0.45 2 0.4 0.4 2 0.55 0.42 2 0.7 0.48 2 0.9 0.55 2\n"
    )
    return tmp_path


class TestOBBRoundtrip:
    def test_export_import_roundtrip(self, obb_dataset, tmp_path):
        """OBB corners survive export→import roundtrip."""
        json_path = tmp_path / "export.json"
        export_labelstudio_json(
            obb_dataset,
            json_path,
            "obb",
            image_url_prefix="/data/local-files/?d=test/images/",
        )

        # Verify JSON was created
        data = json.loads(json_path.read_text())
        assert len(data) == 1
        assert len(data[0]["predictions"][0]["result"]) == 1

        # Import back
        out_dir = tmp_path / "imported"
        stats = import_labelstudio_json(
            json_path,
            out_dir,
            "obb",
            images_dir=obb_dataset / "images" / "train",
        )
        assert stats["tasks_converted"] == 1

        # Compare corners
        original = obb_dataset / "labels" / "train" / "test.txt"
        imported = out_dir / "labels" / "train" / "test.txt"
        orig_tokens = original.read_text().strip().split()
        imp_tokens = imported.read_text().strip().split()

        orig_corners = np.array([float(t) for t in orig_tokens[1:]])
        imp_corners = np.array([float(t) for t in imp_tokens[1:]])
        np.testing.assert_allclose(orig_corners, imp_corners, atol=1e-4)

    def test_multiple_boxes(self, tmp_path):
        """Multiple OBBs in one image roundtrip correctly."""
        img_dir = tmp_path / "images" / "train"
        lbl_dir = tmp_path / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        img = Image.new("RGB", (200, 100), color="black")
        img.save(img_dir / "multi.jpg")

        line1 = _make_rect_corners(60, 40, 80, 20, 15.0, 200, 100)
        line2 = _make_rect_corners(140, 30, 60, 15, -20.0, 200, 100)
        lbl_dir.joinpath("multi.txt").write_text(f"{line1}\n{line2}\n")

        json_path = tmp_path / "export.json"
        export_labelstudio_json(
            tmp_path,
            json_path,
            "obb",
            image_url_prefix="/data/local-files/?d=",
        )

        out_dir = tmp_path / "imported"
        stats = import_labelstudio_json(json_path, out_dir, "obb")
        assert stats["total_annotations"] == 2

        original = lbl_dir / "multi.txt"
        imported = out_dir / "labels" / "train" / "multi.txt"
        orig_lines = original.read_text().strip().splitlines()
        imp_lines = imported.read_text().strip().splitlines()
        assert len(imp_lines) == len(orig_lines)

        for orig_line, imp_line in zip(orig_lines, imp_lines, strict=True):
            orig_vals = np.array([float(t) for t in orig_line.split()[1:]])
            imp_vals = np.array([float(t) for t in imp_line.split()[1:]])
            np.testing.assert_allclose(orig_vals, imp_vals, atol=1e-4)


class TestPoseRoundtrip:
    def test_export_import_roundtrip(self, pose_dataset, tmp_path):
        """Pose bbox and keypoints survive export→import roundtrip."""
        json_path = tmp_path / "export.json"
        export_labelstudio_json(
            pose_dataset,
            json_path,
            "pose",
            image_url_prefix="/data/local-files/?d=test/images/",
            kpt_names=KEYPOINT_NAMES,
        )

        data = json.loads(json_path.read_text())
        assert len(data) == 1
        # 1 bbox + 6 keypoints
        results = data[0]["predictions"][0]["result"]
        assert len(results) == 7
        # Verify keypoint names are semantic, not "kpN"
        kpt_results = [r for r in results if r["type"] == "keypointlabels"]
        kpt_labels = [r["value"]["keypointlabels"][0] for r in kpt_results]
        assert kpt_labels == ["nose", "head", "spine1", "spine2", "spine3", "tail"]

        out_dir = tmp_path / "imported"
        stats = import_labelstudio_json(
            json_path,
            out_dir,
            "pose",
            images_dir=pose_dataset / "images" / "train",
            kpt_names=KEYPOINT_NAMES,
        )
        assert stats["tasks_converted"] == 1

        original = pose_dataset / "labels" / "train" / "test.txt"
        imported = out_dir / "labels" / "train" / "test.txt"
        orig_tokens = original.read_text().strip().split()
        imp_tokens = imported.read_text().strip().split()

        # Compare bbox (cx, cy, w, h)
        orig_bbox = np.array([float(t) for t in orig_tokens[1:5]])
        imp_bbox = np.array([float(t) for t in imp_tokens[1:5]])
        np.testing.assert_allclose(orig_bbox, imp_bbox, atol=1e-4)

        # Compare keypoints (x, y pairs — skip visibility flag comparison
        # since import always writes 2 for visible)
        for k in range(6):
            orig_base = 5 + k * 3
            imp_base = 5 + k * 3
            orig_xy = np.array(
                [float(orig_tokens[orig_base]), float(orig_tokens[orig_base + 1])]
            )
            imp_xy = np.array(
                [float(imp_tokens[imp_base]), float(imp_tokens[imp_base + 1])]
            )
            np.testing.assert_allclose(orig_xy, imp_xy, atol=1e-4)

    def test_invisible_keypoints_preserved(self, tmp_path):
        """Invisible keypoints (vis=0) are not exported and come back as 0."""
        img_dir = tmp_path / "images" / "train"
        lbl_dir = tmp_path / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        img = Image.new("RGB", (128, 64), color="black")
        img.save(img_dir / "partial.jpg")

        # kp2 and kp4 invisible
        lbl_dir.joinpath("partial.txt").write_text(
            "0 0.5 0.5 0.8 0.6 "
            "0.1 0.5 2 0.25 0.45 2 0.0 0.0 0 0.55 0.42 2 0.0 0.0 0 0.9 0.55 2\n"
        )

        json_path = tmp_path / "export.json"
        export_labelstudio_json(
            tmp_path,
            json_path,
            "pose",
            image_url_prefix="/data/local-files/?d=",
        )

        data = json.loads(json_path.read_text())
        # 1 bbox + 4 visible keypoints
        assert len(data[0]["predictions"][0]["result"]) == 5

        out_dir = tmp_path / "imported"
        import_labelstudio_json(json_path, out_dir, "pose")

        imported = out_dir / "labels" / "train" / "partial.txt"
        tokens = imported.read_text().strip().split()
        # kp2 (index 2) should be 0 0 0
        assert tokens[5 + 2 * 3] == "0"
        assert tokens[5 + 2 * 3 + 1] == "0"
        assert tokens[5 + 2 * 3 + 2] == "0"
        # kp4 (index 4) should be 0 0 0
        assert tokens[5 + 4 * 3] == "0"
        assert tokens[5 + 4 * 3 + 1] == "0"
        assert tokens[5 + 4 * 3 + 2] == "0"
