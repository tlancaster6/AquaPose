"""Unit tests for elastic midline deformation module."""

from __future__ import annotations

import numpy as np
import pytest

# --- Keypoint deformation tests ---


class TestDeformKeypointsCCurve:
    """Tests for C-curve (uniform arc) keypoint deformation."""

    @pytest.fixture
    def collinear_keypoints(self) -> np.ndarray:
        """6 collinear keypoints along x-axis at y=100."""
        return np.array(
            [[50, 100], [70, 100], [90, 100], [110, 100], [130, 100], [150, 100]],
            dtype=np.float64,
        )

    def test_zero_angle_is_identity(self, collinear_keypoints: np.ndarray) -> None:
        from aquapose.training.elastic_deform import deform_keypoints_c_curve

        result = deform_keypoints_c_curve(collinear_keypoints, 0.0)
        np.testing.assert_allclose(result, collinear_keypoints, atol=1e-10)

    def test_positive_angle_symmetric_endpoints(
        self, collinear_keypoints: np.ndarray
    ) -> None:
        from aquapose.training.elastic_deform import deform_keypoints_c_curve

        result = deform_keypoints_c_curve(collinear_keypoints, 20.0)
        # Centroid should be preserved
        np.testing.assert_allclose(
            result.mean(axis=0), collinear_keypoints.mean(axis=0), atol=1e-6
        )
        # Middle keypoints should have max lateral displacement
        lateral_disp = np.abs(result[:, 1] - collinear_keypoints[:, 1])
        mid_indices = [2, 3]
        edge_indices = [0, 5]
        assert lateral_disp[mid_indices].mean() > lateral_disp[edge_indices].mean()

    def test_negative_angle_mirrors_positive(
        self, collinear_keypoints: np.ndarray
    ) -> None:
        from aquapose.training.elastic_deform import deform_keypoints_c_curve

        pos = deform_keypoints_c_curve(collinear_keypoints, 20.0)
        neg = deform_keypoints_c_curve(collinear_keypoints, -20.0)
        # Mirror about the midline (x-axis for collinear points at y=100)
        centroid_y = collinear_keypoints[:, 1].mean()
        pos_lateral = pos[:, 1] - centroid_y
        neg_lateral = neg[:, 1] - centroid_y
        np.testing.assert_allclose(pos_lateral, -neg_lateral, atol=1e-6)

    def test_output_shape_matches_input(self, collinear_keypoints: np.ndarray) -> None:
        from aquapose.training.elastic_deform import deform_keypoints_c_curve

        result = deform_keypoints_c_curve(collinear_keypoints, 15.0)
        assert result.shape == collinear_keypoints.shape


class TestDeformKeypointsSCurve:
    """Tests for S-curve (sinusoidal) keypoint deformation."""

    @pytest.fixture
    def collinear_keypoints(self) -> np.ndarray:
        """6 collinear keypoints along x-axis at y=100."""
        return np.array(
            [[50, 100], [70, 100], [90, 100], [110, 100], [130, 100], [150, 100]],
            dtype=np.float64,
        )

    def test_centroid_preserved(self, collinear_keypoints: np.ndarray) -> None:
        from aquapose.training.elastic_deform import deform_keypoints_s_curve

        result = deform_keypoints_s_curve(collinear_keypoints, 20.0)
        np.testing.assert_allclose(
            result.mean(axis=0), collinear_keypoints.mean(axis=0), atol=1e-6
        )

    def test_endpoints_near_zero_displacement(
        self, collinear_keypoints: np.ndarray
    ) -> None:
        from aquapose.training.elastic_deform import deform_keypoints_s_curve

        result = deform_keypoints_s_curve(collinear_keypoints, 20.0)
        # Endpoints should have smaller displacement than middle
        lateral_disp = np.abs(result[:, 1] - collinear_keypoints[:, 1])
        assert lateral_disp[0] < lateral_disp[2]
        assert lateral_disp[5] < lateral_disp[3]

    def test_negative_amplitude_mirrors(self, collinear_keypoints: np.ndarray) -> None:
        from aquapose.training.elastic_deform import deform_keypoints_s_curve

        pos = deform_keypoints_s_curve(collinear_keypoints, 20.0)
        neg = deform_keypoints_s_curve(collinear_keypoints, -20.0)
        centroid_y = collinear_keypoints[:, 1].mean()
        pos_lateral = pos[:, 1] - centroid_y
        neg_lateral = neg[:, 1] - centroid_y
        np.testing.assert_allclose(pos_lateral, -neg_lateral, atol=1e-6)

    def test_output_shape_matches_input(self, collinear_keypoints: np.ndarray) -> None:
        from aquapose.training.elastic_deform import deform_keypoints_s_curve

        result = deform_keypoints_s_curve(collinear_keypoints, 15.0)
        assert result.shape == collinear_keypoints.shape


# --- TPS image warp tests ---


class TestTpsWarpImage:
    """Tests for thin-plate spline image warping."""

    def test_identity_mapping_preserves_image(self) -> None:
        from aquapose.training.elastic_deform import tps_warp_image

        img = np.random.randint(0, 255, (60, 100, 3), dtype=np.uint8)
        pts = np.array(
            [[20, 10], [40, 10], [60, 30], [80, 30], [50, 50], [70, 50]],
            dtype=np.float64,
        )
        result = tps_warp_image(img, pts, pts, 100, 60)
        assert result.shape == img.shape
        # Identity should produce very similar output
        np.testing.assert_allclose(result.astype(float), img.astype(float), atol=5.0)

    def test_non_identity_changes_image(self) -> None:
        from aquapose.training.elastic_deform import tps_warp_image

        img = np.zeros((60, 100, 3), dtype=np.uint8)
        img[25:35, 45:55] = 255  # white square in center
        src = np.array(
            [[20, 10], [40, 10], [60, 30], [80, 30], [50, 50], [70, 50]],
            dtype=np.float64,
        )
        dst = src.copy()
        dst[:, 1] += 5  # shift all points down by 5
        result = tps_warp_image(img, src, dst, 100, 60)
        assert result.shape == img.shape
        # Should differ from original due to warp
        assert not np.array_equal(result, img)

    def test_output_shape_correct(self) -> None:
        from aquapose.training.elastic_deform import tps_warp_image

        img = np.zeros((60, 100, 3), dtype=np.uint8)
        pts = np.array(
            [[20, 10], [40, 10], [60, 30], [80, 30], [50, 50], [70, 50]],
            dtype=np.float64,
        )
        result = tps_warp_image(img, pts, pts, 100, 60)
        assert result.shape == (60, 100, 3)


# --- Label generation tests ---


class TestGenerateDeformedLabels:
    """Tests for deformed label generation."""

    def test_labels_have_normalized_values(self) -> None:
        from aquapose.training.elastic_deform import generate_deformed_labels

        coords = np.array(
            [[20, 10], [35, 12], [50, 15], [65, 18], [80, 15], [95, 10]],
            dtype=np.float64,
        )
        visible = np.ones(6, dtype=bool)
        result = generate_deformed_labels(coords, visible, 120, 60, 15.0)

        assert "obb_line" in result
        assert "pose_line" in result
        assert "obb_corners" in result

        # OBB line: [cls, x1, y1, ..., x4, y4] -> 9 values
        obb = result["obb_line"]
        assert len(obb) == 9
        # All coordinate values should be in [0, 1]
        for val in obb[1:]:
            assert 0.0 <= val <= 1.0, f"OBB value {val} out of [0,1]"

        # Pose line: [cls, cx, cy, w, h, x1, y1, v1, ...] -> 5 + 6*3 = 23
        pose = result["pose_line"]
        assert len(pose) == 23
        # bbox values normalized
        for val in pose[1:5]:
            assert 0.0 <= val <= 1.0, f"Pose bbox value {val} out of [0,1]"

    def test_obb_corners_shape(self) -> None:
        from aquapose.training.elastic_deform import generate_deformed_labels

        coords = np.array(
            [[20, 10], [35, 12], [50, 15], [65, 18], [80, 15], [95, 10]],
            dtype=np.float64,
        )
        visible = np.ones(6, dtype=bool)
        result = generate_deformed_labels(coords, visible, 120, 60, 15.0)
        assert result["obb_corners"].shape == (4, 2)


# --- Variant generation tests ---


class TestGenerateVariants:
    """Tests for high-level variant generation."""

    def test_four_variants_returned(self) -> None:
        from aquapose.training.elastic_deform import generate_variants

        img = np.random.randint(0, 255, (60, 100, 3), dtype=np.uint8)
        coords = np.array(
            [[15, 30], [30, 30], [45, 30], [60, 30], [75, 30], [90, 30]],
            dtype=np.float64,
        )
        visible = np.ones(6, dtype=bool)
        variants = generate_variants(img, coords, visible, 100, 60, 15.0)
        assert len(variants) == 4

    def test_variant_tags_correct(self) -> None:
        from aquapose.training.elastic_deform import generate_variants

        img = np.random.randint(0, 255, (60, 100, 3), dtype=np.uint8)
        coords = np.array(
            [[15, 30], [30, 30], [45, 30], [60, 30], [75, 30], [90, 30]],
            dtype=np.float64,
        )
        visible = np.ones(6, dtype=bool)
        variants = generate_variants(img, coords, visible, 100, 60, 15.0)
        tags = {v["variant_tag"] for v in variants}
        assert tags == {"c_pos", "c_neg", "s_pos", "s_neg"}

    def test_variant_dict_keys(self) -> None:
        from aquapose.training.elastic_deform import generate_variants

        img = np.random.randint(0, 255, (60, 100, 3), dtype=np.uint8)
        coords = np.array(
            [[15, 30], [30, 30], [45, 30], [60, 30], [75, 30], [90, 30]],
            dtype=np.float64,
        )
        visible = np.ones(6, dtype=bool)
        variants = generate_variants(img, coords, visible, 100, 60, 15.0)
        expected_keys = {
            "image",
            "coords",
            "visible",
            "obb_line",
            "pose_line",
            "variant_tag",
        }
        for v in variants:
            assert set(v.keys()) == expected_keys

    def test_variant_images_have_correct_shape(self) -> None:
        from aquapose.training.elastic_deform import generate_variants

        img = np.random.randint(0, 255, (60, 100, 3), dtype=np.uint8)
        coords = np.array(
            [[15, 30], [30, 30], [45, 30], [60, 30], [75, 30], [90, 30]],
            dtype=np.float64,
        )
        visible = np.ones(6, dtype=bool)
        variants = generate_variants(img, coords, visible, 100, 60, 15.0)
        for v in variants:
            assert v["image"].shape == img.shape
