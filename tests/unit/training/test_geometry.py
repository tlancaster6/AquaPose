"""Tests for promoted geometry functions in training.geometry."""

from __future__ import annotations

import numpy as np
import pytest

from aquapose.training.geometry import (
    extrapolate_edge_keypoints,
    format_obb_annotation,
    format_pose_annotation,
    pca_obb,
)


class TestPcaObb:
    """Tests for pca_obb function."""

    def test_shape_and_order(self) -> None:
        """OBB output has shape (4, 2) for a normal case."""
        coords = np.array(
            [[100, 200], [150, 210], [200, 220], [250, 230], [300, 240]],
            dtype=np.float64,
        )
        visible = np.array([True, True, True, True, True])
        corners = pca_obb(coords, visible, lateral_pad=30.0)
        assert corners.shape == (4, 2)
        assert corners.dtype == np.float64

    def test_degenerate_single_point(self) -> None:
        """Single visible keypoint returns a default 20x20 box."""
        coords = np.array([[100, 200], [0, 0]], dtype=np.float64)
        visible = np.array([True, False])
        corners = pca_obb(coords, visible, lateral_pad=30.0)
        assert corners.shape == (4, 2)
        # Should be centered at (100, 200) with half=10
        np.testing.assert_allclose(corners[0], [90, 190])
        np.testing.assert_allclose(corners[2], [110, 210])

    def test_degenerate_no_visible(self) -> None:
        """No visible keypoints returns a default box at origin."""
        coords = np.array([[100, 200]], dtype=np.float64)
        visible = np.array([False])
        corners = pca_obb(coords, visible, lateral_pad=30.0)
        assert corners.shape == (4, 2)
        np.testing.assert_allclose(corners[0], [-10, -10])

    def test_lateral_pad_affects_width(self) -> None:
        """Larger lateral_pad produces a wider OBB."""
        coords = np.array([[100, 200], [200, 200]], dtype=np.float64)
        visible = np.array([True, True])
        corners_small = pca_obb(coords, visible, lateral_pad=10.0)
        corners_large = pca_obb(coords, visible, lateral_pad=50.0)
        # The perpendicular extent should be larger for larger lateral_pad
        height_small = np.linalg.norm(corners_small[0] - corners_small[3])
        height_large = np.linalg.norm(corners_large[0] - corners_large[3])
        assert height_large > height_small


class TestExtrapolateEdgeKeypoints:
    """Tests for extrapolate_edge_keypoints function."""

    def test_no_extrapolation_when_far_from_edge(self) -> None:
        """Keypoints far from edges are unchanged."""
        coords = np.array([[500, 500], [600, 500], [700, 500]], dtype=np.float64)
        visible = np.array([True, True, True])
        out_coords, out_vis = extrapolate_edge_keypoints(
            coords, visible, img_w=1920, img_h=1080, lateral_pad=40.0
        )
        np.testing.assert_array_equal(out_coords, coords)
        np.testing.assert_array_equal(out_vis, visible)

    def test_extrapolation_near_left_edge(self) -> None:
        """First keypoint near left edge gets snapped to x=0."""
        coords = np.array([[10, 500], [100, 500], [200, 500]], dtype=np.float64)
        visible = np.array([True, True, True])
        out_coords, _out_vis = extrapolate_edge_keypoints(
            coords, visible, img_w=1920, img_h=1080, lateral_pad=40.0
        )
        # First point should be snapped to x=0 (nearest edge)
        assert out_coords[0, 0] == 0.0

    def test_single_visible_unchanged(self) -> None:
        """With fewer than 2 visible keypoints, no extrapolation occurs."""
        coords = np.array([[10, 500], [100, 500]], dtype=np.float64)
        visible = np.array([True, False])
        out_coords, _out_vis = extrapolate_edge_keypoints(
            coords, visible, img_w=1920, img_h=1080, lateral_pad=40.0
        )
        np.testing.assert_array_equal(out_coords, coords)


class TestFormatObbAnnotation:
    """Tests for format_obb_annotation function."""

    def test_output_format(self) -> None:
        """Returns [cls, x1, y1, ..., x4, y4] with 9 elements."""
        corners = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64)
        row = format_obb_annotation(corners, img_w=1000, img_h=500)
        assert len(row) == 9
        assert row[0] == 0.0  # class_id

    def test_normalized_values(self) -> None:
        """Corner coordinates are normalized to [0, 1]."""
        corners = np.array(
            [[100, 200], [300, 200], [300, 400], [100, 400]], dtype=np.float64
        )
        row = format_obb_annotation(corners, img_w=1000, img_h=1000)
        # x1=100/1000=0.1, y1=200/1000=0.2
        assert row[1] == pytest.approx(0.1)
        assert row[2] == pytest.approx(0.2)

    def test_clipping(self) -> None:
        """Out-of-bounds corners are clipped to [0, 1]."""
        corners = np.array(
            [[-50, -50], [1500, -50], [1500, 1500], [-50, 1500]],
            dtype=np.float64,
        )
        row = format_obb_annotation(corners, img_w=1000, img_h=1000)
        for v in row[1:]:
            assert 0.0 <= v <= 1.0


class TestFormatPoseAnnotation:
    """Tests for format_pose_annotation function."""

    def test_output_format(self) -> None:
        """Returns [cls, cx, cy, w, h, x1, y1, v1, ...] with correct length."""
        keypoints = np.array([[100, 200], [300, 400], [500, 600]], dtype=np.float64)
        visible = np.array([True, False, True])
        row = format_pose_annotation(
            cx=0.5,
            cy=0.5,
            w=0.3,
            h=0.2,
            keypoints=keypoints,
            visible=visible,
            crop_w=1000,
            crop_h=1000,
        )
        # 5 (header) + 3 keypoints * 3 values each = 14
        assert len(row) == 14
        assert row[0] == 0.0  # class_id

    def test_invisible_keypoints_are_zero(self) -> None:
        """Invisible keypoints output as 0, 0, 0."""
        keypoints = np.array([[100, 200], [300, 400]], dtype=np.float64)
        visible = np.array([True, False])
        row = format_pose_annotation(
            cx=0.5,
            cy=0.5,
            w=0.3,
            h=0.2,
            keypoints=keypoints,
            visible=visible,
            crop_w=1000,
            crop_h=1000,
        )
        # Second keypoint (invisible): indices 8, 9, 10
        assert row[8] == 0
        assert row[9] == 0
        assert row[10] == 0

    def test_visible_keypoints_normalized(self) -> None:
        """Visible keypoints are normalized and have visibility=2."""
        keypoints = np.array([[500, 250]], dtype=np.float64)
        visible = np.array([True])
        row = format_pose_annotation(
            cx=0.5,
            cy=0.5,
            w=0.3,
            h=0.2,
            keypoints=keypoints,
            visible=visible,
            crop_w=1000,
            crop_h=500,
        )
        # x=500/1000=0.5, y=250/500=0.5, vis=2
        assert row[5] == pytest.approx(0.5)
        assert row[6] == pytest.approx(0.5)
        assert row[7] == 2
