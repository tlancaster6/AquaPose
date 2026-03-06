"""Unit tests for COCO-to-YOLO conversion geometry and format functions."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from aquapose.training.coco_convert import (
    compute_arc_length,
    compute_median_arc_length,
    generate_obb_dataset,
    generate_pose_dataset,
    load_coco,
    parse_keypoints,
    transform_keypoints,
)
from aquapose.training.geometry import (
    extrapolate_edge_keypoints,
    format_obb_annotation,
    format_pose_annotation,
    pca_obb,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 6  # number of keypoints


def _make_ann(keypoints: list[float]) -> dict:
    """Build a minimal COCO annotation dict."""
    return {"image_id": 1, "keypoints": keypoints, "num_keypoints": N}


def _full_kp_flat(coords: list[tuple[float, float]]) -> list[float]:
    """Build a flat COCO keypoints list with all v=2 (visible)."""
    flat: list[float] = []
    for x, y in coords:
        flat.extend([x, y, 2])
    return flat


# ---------------------------------------------------------------------------
# parse_keypoints
# ---------------------------------------------------------------------------


class TestParseKeypoints:
    def test_all_visible(self) -> None:
        pts = [
            (10.0, 20.0),
            (30.0, 40.0),
            (50.0, 60.0),
            (70.0, 80.0),
            (90.0, 100.0),
            (110.0, 120.0),
        ]
        ann = _make_ann(_full_kp_flat(pts))
        coords, visible = parse_keypoints(ann)
        assert coords.shape == (N, 2)
        assert visible.shape == (N,)
        assert visible.all()
        np.testing.assert_allclose(coords[0], [10.0, 20.0])
        np.testing.assert_allclose(coords[-1], [110.0, 120.0])

    def test_mixed_visibility(self) -> None:
        # v=0 -> invisible, v=1 -> visible, v=2 -> visible
        kps = [10, 20, 0, 30, 40, 1, 50, 60, 2, 70, 80, 0, 90, 100, 2, 110, 120, 1]
        ann = _make_ann(kps)
        _coords, visible = parse_keypoints(ann)
        assert visible[0] is np.bool_(False)
        assert visible[1] is np.bool_(True)
        assert visible[2] is np.bool_(True)
        assert visible[3] is np.bool_(False)
        assert visible[4] is np.bool_(True)
        assert visible[5] is np.bool_(True)

    def test_empty_keypoints(self) -> None:
        ann = _make_ann([])
        coords, visible = parse_keypoints(ann)
        assert not visible.any()
        assert coords.shape == (N, 2)

    def test_partial_keypoints(self) -> None:
        # Only first 3 keypoints provided
        kps = [10, 20, 2, 30, 40, 2, 50, 60, 2]
        ann = _make_ann(kps)
        _coords, visible = parse_keypoints(ann, n_keypoints=N)
        assert visible[:3].all()
        assert not visible[3:].any()


# ---------------------------------------------------------------------------
# compute_arc_length
# ---------------------------------------------------------------------------


class TestComputeArcLength:
    def test_collinear_3_pts(self) -> None:
        # 3 visible points in a horizontal line spaced 10px apart
        coords = np.array(
            [[0, 0], [10, 0], [20, 0], [0, 0], [0, 0], [0, 0]], dtype=float
        )
        visible = np.array([True, True, True, False, False, False])
        arc = compute_arc_length(coords, visible)
        assert arc is not None
        np.testing.assert_allclose(arc, 20.0, atol=1e-6)

    def test_diagonal(self) -> None:
        coords = np.array([[0, 0], [3, 4], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=float)
        visible = np.array([True, True, False, False, False, False])
        arc = compute_arc_length(coords, visible)
        assert arc is not None
        np.testing.assert_allclose(arc, 5.0, atol=1e-6)

    def test_fewer_than_2_visible_returns_none(self) -> None:
        coords = np.zeros((N, 2))
        visible = np.array([True, False, False, False, False, False])
        assert compute_arc_length(coords, visible) is None

    def test_no_visible_returns_none(self) -> None:
        coords = np.zeros((N, 2))
        visible = np.zeros(N, dtype=bool)
        assert compute_arc_length(coords, visible) is None


# ---------------------------------------------------------------------------
# pca_obb
# ---------------------------------------------------------------------------


class TestPcaObb:
    def test_returns_4_corners(self) -> None:
        coords = np.array(
            [[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0]], dtype=float
        )
        visible = np.ones(N, dtype=bool)
        corners = pca_obb(coords, visible, lateral_pad=10.0)
        assert corners.shape == (4, 2)

    def test_horizontal_fish_orientation(self) -> None:
        # Fish aligned along x-axis — OBB should be wider than tall
        coords = np.array(
            [[0, 0], [20, 0], [40, 0], [60, 0], [80, 0], [100, 0]], dtype=float
        )
        visible = np.ones(N, dtype=bool)
        corners = pca_obb(coords, visible, lateral_pad=5.0)
        # Compute bounding rect of the corners
        min_pt = corners.min(axis=0)
        max_pt = corners.max(axis=0)
        width = max_pt[0] - min_pt[0]
        height = max_pt[1] - min_pt[1]
        # Width (along x) should exceed height (lateral pad = 5 each side = 10)
        assert width > height

    def test_encloses_keypoints(self) -> None:
        coords = np.array(
            [[10, 10], [20, 12], [30, 14], [40, 16], [50, 18], [60, 20]], dtype=float
        )
        visible = np.ones(N, dtype=bool)
        corners = pca_obb(coords, visible, lateral_pad=15.0)
        # The centroid should be inside the OBB approximate bounds
        centroid = corners.mean(axis=0)
        # OBB center should be close to keypoint centroid
        kp_centroid = coords.mean(axis=0)
        np.testing.assert_allclose(centroid, kp_centroid, atol=5.0)

    def test_degenerate_single_visible(self) -> None:
        coords = np.array(
            [[50, 50], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=float
        )
        visible = np.array([True, False, False, False, False, False])
        corners = pca_obb(coords, visible, lateral_pad=10.0)
        assert corners.shape == (4, 2)
        # Should return a 20x20 box centered at the visible point
        center = corners.mean(axis=0)
        np.testing.assert_allclose(center, [50.0, 50.0], atol=1.0)

    def test_degenerate_no_visible(self) -> None:
        coords = np.zeros((N, 2))
        visible = np.zeros(N, dtype=bool)
        corners = pca_obb(coords, visible, lateral_pad=10.0)
        assert corners.shape == (4, 2)


# ---------------------------------------------------------------------------
# extrapolate_edge_keypoints
# ---------------------------------------------------------------------------


class TestExtrapolateEdgeKeypoints:
    def test_nose_near_left_edge_is_extended(self) -> None:
        # Nose at x=3, well inside threshold of lateral_pad * edge_factor = 20
        coords = np.array(
            [[3, 50], [30, 50], [60, 50], [90, 50], [120, 50], [150, 50]], dtype=float
        )
        visible = np.ones(N, dtype=bool)
        coords_out, vis_out = extrapolate_edge_keypoints(
            coords, visible, img_w=200, img_h=100, lateral_pad=5.0, edge_factor=2.0
        )
        # First keypoint should be pushed closer to or at the left edge
        assert coords_out[0, 0] <= coords[0, 0]
        assert vis_out.all()

    def test_mid_frame_keypoints_not_extended(self) -> None:
        # All keypoints well away from borders
        coords = np.array(
            [[100, 100], [150, 100], [200, 100], [250, 100], [300, 100], [350, 100]],
            dtype=float,
        )
        visible = np.ones(N, dtype=bool)
        coords_out, _vis_out = extrapolate_edge_keypoints(
            coords, visible, img_w=500, img_h=200, lateral_pad=5.0, edge_factor=2.0
        )
        # No change expected — all keypoints are far from edge
        np.testing.assert_allclose(coords_out, coords)

    def test_returns_copies(self) -> None:
        coords = np.array(
            [[3, 50], [30, 50], [60, 50], [90, 50], [120, 50], [150, 50]], dtype=float
        )
        visible = np.ones(N, dtype=bool)
        coords_out, vis_out = extrapolate_edge_keypoints(
            coords, visible, img_w=200, img_h=100, lateral_pad=5.0
        )
        # Originals should be unchanged
        assert coords[0, 0] == 3.0
        assert coords_out is not coords
        assert vis_out is not visible


# ---------------------------------------------------------------------------
# format_obb_annotation — flat list format
# ---------------------------------------------------------------------------


class TestFormatObbAnnotation:
    def test_returns_flat_list(self) -> None:
        """format_obb_annotation returns a flat list [cls, x1, y1, ..., x4, y4]."""
        corners = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=float)
        row = format_obb_annotation(corners, img_w=100, img_h=50)
        assert isinstance(row, list)
        # cls + 4 corners * 2 coords = 9 values
        assert len(row) == 9
        assert row[0] == 0.0  # class_id

    def test_normalized_values(self) -> None:
        corners = np.array([[50, 25], [100, 25], [100, 50], [50, 50]], dtype=float)
        row = format_obb_annotation(corners, img_w=200, img_h=100)
        # All values except class_id should be in [0, 1]
        for v in row[1:]:
            assert 0.0 <= v <= 1.0

    def test_custom_class_id(self) -> None:
        corners = np.zeros((4, 2))
        row = format_obb_annotation(corners, img_w=100, img_h=100, class_id=3)
        assert row[0] == 3.0


# ---------------------------------------------------------------------------
# format_pose_annotation — flat list format
# ---------------------------------------------------------------------------


class TestFormatPoseAnnotation:
    def test_returns_flat_list(self) -> None:
        """format_pose_annotation returns a flat list [cls, cx, cy, w, h, x1, y1, v1, ...]."""
        kps = np.array(
            [[16, 8], [32, 8], [48, 8], [64, 8], [80, 8], [96, 8]], dtype=float
        )
        vis = np.ones(N, dtype=bool)
        row = format_pose_annotation(
            0.5, 0.5, 1.0, 1.0, kps, vis, crop_w=128, crop_h=64
        )
        assert isinstance(row, list)
        # cls + 4 bbox + N * 3 = 5 + 18 = 23 values
        assert len(row) == 5 + N * 3
        assert row[0] == 0.0  # class_id
        assert row[1] == 0.5  # cx
        assert row[2] == 0.5  # cy
        assert row[3] == 1.0  # w
        assert row[4] == 1.0  # h

    def test_visible_keypoints_have_v2(self) -> None:
        kps = np.array(
            [[16, 8], [32, 8], [48, 8], [64, 8], [80, 8], [96, 8]], dtype=float
        )
        vis = np.ones(N, dtype=bool)
        row = format_pose_annotation(
            0.5, 0.5, 1.0, 1.0, kps, vis, crop_w=128, crop_h=64
        )
        # Every 3rd value starting at index 7 is the visibility flag
        for k in range(N):
            v_flag = row[5 + k * 3 + 2]
            assert v_flag == 2

    def test_invisible_keypoints_are_zeros(self) -> None:
        kps = np.zeros((N, 2), dtype=float)
        vis = np.zeros(N, dtype=bool)
        row = format_pose_annotation(
            0.5, 0.5, 1.0, 1.0, kps, vis, crop_w=128, crop_h=64
        )
        for k in range(N):
            x = row[5 + k * 3]
            y = row[5 + k * 3 + 1]
            v = row[5 + k * 3 + 2]
            assert x == 0
            assert y == 0
            assert v == 0

    def test_mixed_visibility(self) -> None:
        kps = np.array([[16, 8], [0, 0], [48, 8], [0, 0], [80, 8], [0, 0]], dtype=float)
        vis = np.array([True, False, True, False, True, False])
        row = format_pose_annotation(
            0.5, 0.5, 1.0, 1.0, kps, vis, crop_w=128, crop_h=64
        )
        for k in range(N):
            v_flag = row[5 + k * 3 + 2]
            if vis[k]:
                assert v_flag == 2
            else:
                assert v_flag == 0


# ---------------------------------------------------------------------------
# transform_keypoints
# ---------------------------------------------------------------------------


class TestTransformKeypoints:
    def test_identity(self) -> None:
        coords = np.array(
            [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100], [110, 120]], dtype=float
        )
        visible = np.ones(N, dtype=bool)
        affine = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        out, vis_out = transform_keypoints(
            coords, visible, affine, crop_w=200, crop_h=200
        )
        np.testing.assert_allclose(out, coords, atol=1e-6)
        assert vis_out.all()

    def test_translation(self) -> None:
        coords = np.array(
            [[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0]], dtype=float
        )
        visible = np.ones(N, dtype=bool)
        # Translate by (5, 10)
        affine = np.array([[1, 0, 5], [0, 1, 10]], dtype=float)
        out, _vis_out = transform_keypoints(
            coords, visible, affine, crop_w=200, crop_h=200
        )
        np.testing.assert_allclose(out[:, 0], coords[:, 0] + 5, atol=1e-6)
        np.testing.assert_allclose(out[:, 1], coords[:, 1] + 10, atol=1e-6)

    def test_oob_marked_invisible(self) -> None:
        # Point at (150, 0) — will be out of 100x100 crop
        coords = np.array(
            [[150, 0], [10, 10], [20, 20], [30, 30], [40, 40], [50, 50]], dtype=float
        )
        visible = np.ones(N, dtype=bool)
        affine = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        _out, vis_out = transform_keypoints(
            coords, visible, affine, crop_w=100, crop_h=100
        )
        assert not vis_out[0]  # out of bounds
        assert vis_out[1:].all()


# ---------------------------------------------------------------------------
# Integration test: full pipeline (txt+yaml format)
# ---------------------------------------------------------------------------


class TestIntegrationPipeline:
    def _build_coco_json(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create a minimal COCO JSON with 2 images and 3 annotations."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Create two dummy images (200x100 BGR)
        img1 = np.zeros((100, 200, 3), dtype=np.uint8)
        img2 = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "img1.jpg"), img1)
        cv2.imwrite(str(images_dir / "img2.jpg"), img2)

        # Build COCO JSON: 2 images, 3 annotations (2 on img1, 1 on img2)
        # All keypoints fully visible, spread across image
        kps_full = _full_kp_flat(
            [(20, 50), (40, 50), (80, 50), (120, 50), (160, 50), (180, 50)]
        )
        coco = {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 200, "height": 100},
                {"id": 2, "file_name": "img2.jpg", "width": 200, "height": 100},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "keypoints": kps_full, "num_keypoints": N},
                {"id": 2, "image_id": 1, "keypoints": kps_full, "num_keypoints": N},
                {"id": 3, "image_id": 2, "keypoints": kps_full, "num_keypoints": N},
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "fish",
                    "keypoints": ["nose", "head", "spine1", "spine2", "spine3", "tail"],
                }
            ],
        }
        coco_path = tmp_path / "annotations.json"
        with open(coco_path, "w") as f:
            json.dump(coco, f)

        return coco_path, images_dir

    def test_obb_dataset_structure(self, tmp_path: Path) -> None:
        """OBB dataset produces labels/ dir with .txt files and dataset.yaml."""
        coco_path, images_dir = self._build_coco_json(tmp_path)
        coco = load_coco(coco_path)
        output_dir = tmp_path / "output"

        median_arc = compute_median_arc_length(coco["annotations"])
        n_train, n_val = generate_obb_dataset(
            coco,
            images_dir=images_dir,
            output_dir=output_dir,
            median_arc=median_arc,
            lateral_ratio=0.18,
            edge_factor=2.0,
            val_split=0.5,
            seed=42,
        )

        obb_root = output_dir / "obb"
        assert (obb_root / "dataset.yaml").exists()
        assert (obb_root / "images" / "train").is_dir()
        assert (obb_root / "images" / "val").is_dir()
        assert (obb_root / "labels" / "train").is_dir()
        assert (obb_root / "labels" / "val").is_dir()

        # Total images should be 2
        assert n_train + n_val == 2

        # Each image should have a corresponding .txt label file
        all_imgs = list((obb_root / "images" / "train").glob("*.jpg")) + list(
            (obb_root / "images" / "val").glob("*.jpg")
        )
        for img_path in all_imgs:
            label_path = (
                obb_root / "labels" / img_path.parent.name / (img_path.stem + ".txt")
            )
            assert label_path.exists(), f"Missing label for {img_path}"
            # Each non-empty line should have 9 space-separated values (cls + 4 corners * 2)
            lines = [ln for ln in label_path.read_text().splitlines() if ln.strip()]
            for line in lines:
                parts = line.split()
                assert len(parts) == 9, f"Expected 9 values, got {len(parts)}: {line}"

    def test_pose_dataset_structure(self, tmp_path: Path) -> None:
        """Pose dataset produces labels/ dir with .txt files and dataset.yaml."""
        coco_path, images_dir = self._build_coco_json(tmp_path)
        coco = load_coco(coco_path)
        output_dir = tmp_path / "output"

        median_arc = compute_median_arc_length(coco["annotations"])
        n_train, n_val = generate_pose_dataset(
            coco,
            images_dir=images_dir,
            output_dir=output_dir,
            median_arc=median_arc,
            lateral_ratio=0.18,
            edge_factor=2.0,
            crop_w=128,
            crop_h=64,
            min_visible=4,
            val_split=0.5,
            seed=42,
        )

        pose_root = output_dir / "pose"
        assert (pose_root / "dataset.yaml").exists()
        assert (pose_root / "images" / "train").is_dir()
        assert (pose_root / "images" / "val").is_dir()
        assert (pose_root / "labels" / "train").is_dir()
        assert (pose_root / "labels" / "val").is_dir()

        # 3 annotations in input, all with 6/6 visible -> 3 crops total
        assert n_train + n_val == 3

        # Check crop images have the right size
        all_crops = list((pose_root / "images" / "train").glob("*.jpg")) + list(
            (pose_root / "images" / "val").glob("*.jpg")
        )
        for crop_path in all_crops:
            img = cv2.imread(str(crop_path))
            assert img is not None
            h, w = img.shape[:2]
            assert w == 128
            assert h == 64

        # Each crop should have a .txt label with correct format
        # cls cx cy w h + N * 3 = 5 + 18 = 23 values per line
        for crop_path in all_crops:
            label_path = (
                pose_root / "labels" / crop_path.parent.name / (crop_path.stem + ".txt")
            )
            assert label_path.exists()
            lines = [ln for ln in label_path.read_text().splitlines() if ln.strip()]
            assert len(lines) >= 1, "Each crop has at least one pose annotation"
            for line in lines:
                parts = line.split()
                assert len(parts) == 5 + N * 3

    def test_dataset_yaml_content_obb(self, tmp_path: Path) -> None:
        """OBB dataset.yaml has correct keys and no kpt_shape."""
        coco_path, images_dir = self._build_coco_json(tmp_path)
        coco = load_coco(coco_path)
        output_dir = tmp_path / "output"
        median_arc = compute_median_arc_length(coco["annotations"])

        generate_obb_dataset(
            coco, images_dir, output_dir, median_arc, 0.18, 2.0, 0.5, 42
        )

        obb_yaml_path = output_dir / "obb" / "dataset.yaml"
        obb_yaml = yaml.safe_load(obb_yaml_path.read_text())

        assert "path" in obb_yaml
        assert obb_yaml["train"] == "images/train"
        assert obb_yaml["val"] == "images/val"
        assert obb_yaml["nc"] == 1
        assert "kpt_shape" not in obb_yaml
        assert "flip_idx" not in obb_yaml

    def test_dataset_yaml_content_pose(self, tmp_path: Path) -> None:
        """Pose dataset.yaml has kpt_shape and flip_idx."""
        coco_path, images_dir = self._build_coco_json(tmp_path)
        coco = load_coco(coco_path)
        output_dir = tmp_path / "output"
        median_arc = compute_median_arc_length(coco["annotations"])

        generate_pose_dataset(
            coco, images_dir, output_dir, median_arc, 0.18, 2.0, 128, 64, 4, 0.5, 42
        )

        pose_yaml_path = output_dir / "pose" / "dataset.yaml"
        pose_yaml = yaml.safe_load(pose_yaml_path.read_text())

        assert pose_yaml["train"] == "images/train"
        assert pose_yaml["val"] == "images/val"
        assert pose_yaml["nc"] == 1
        assert pose_yaml["kpt_shape"] == [6, 3]
        assert "flip_idx" in pose_yaml
        assert len(pose_yaml["flip_idx"]) == 6
