"""Unit tests for PCA-based keypoint extraction from binary masks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aquapose.initialization.keypoints import extract_keypoints, extract_keypoints_batch


class TestHorizontalRectangle:
    """Test keypoint extraction from a horizontal bar mask."""

    def setup_method(self):
        # 200 wide, 20 tall horizontal bar at center of a 300x100 canvas
        self.mask = np.zeros((100, 300), dtype=np.uint8)
        self.mask[40:60, 50:250] = 255  # rows 40-59, cols 50-249

    def test_center_near_centroid(self):
        center, _, _ = extract_keypoints(self.mask)
        # Centroid in (u, v) = (col, row): u ~149.5, v ~49.5
        assert center.shape == (2,)
        assert abs(center[0] - 149.5) < 1.0, f"u center = {center[0]}"
        assert abs(center[1] - 49.5) < 1.0, f"v center = {center[1]}"

    def test_major_axis_is_horizontal(self):
        _center, ep_a, ep_b = extract_keypoints(self.mask)
        # Endpoints should differ mainly in u (horizontal), not v
        assert abs(ep_a[0] - ep_b[0]) > 100, "Endpoints should span horizontal axis"
        assert abs(ep_a[1] - ep_b[1]) < 5, "Endpoints should have similar v coords"

    def test_output_dtype_float32(self):
        center, ep_a, ep_b = extract_keypoints(self.mask)
        assert center.dtype == np.float32
        assert ep_a.dtype == np.float32
        assert ep_b.dtype == np.float32


class TestVerticalRectangle:
    """Test keypoint extraction from a vertical bar mask."""

    def setup_method(self):
        # 20 wide, 200 tall vertical bar
        self.mask = np.zeros((300, 100), dtype=np.uint8)
        self.mask[50:250, 40:60] = 255  # rows 50-249, cols 40-59

    def test_major_axis_is_vertical(self):
        _center, ep_a, ep_b = extract_keypoints(self.mask)
        # Endpoints should differ mainly in v (vertical), not u
        assert abs(ep_a[1] - ep_b[1]) > 100, "Endpoints should span vertical axis"
        assert abs(ep_a[0] - ep_b[0]) < 5, "Endpoints should have similar u coords"


class TestDiagonalEllipse:
    """Test keypoint extraction from a rotated elliptical mask."""

    def setup_method(self):
        # Create a diagonal ellipse (45-degree orientation)
        H, W = 200, 200
        self.mask = np.zeros((H, W), dtype=np.uint8)
        cx, cy = W // 2, H // 2
        for row in range(H):
            for col in range(W):
                # Ellipse rotated 45 degrees: semi-major=60, semi-minor=20
                dx = col - cx
                dy = row - cy
                # Rotate by -45 degrees
                angle = -math.pi / 4
                dx_r = dx * math.cos(angle) - dy * math.sin(angle)
                dy_r = dx * math.sin(angle) + dy * math.cos(angle)
                if (dx_r / 60) ** 2 + (dy_r / 20) ** 2 <= 1.0:
                    self.mask[row, col] = 255

    def test_major_axis_follows_ellipse(self):
        _center, ep_a, ep_b = extract_keypoints(self.mask)
        # Axis vector (ep_a - ep_b) should be close to 45 degrees
        axis = ep_a - ep_b
        angle = math.degrees(math.atan2(abs(axis[1]), abs(axis[0])))
        # Should be close to 45 degrees (allowing generous tolerance for discrete pixels)
        assert 30 < angle < 60, f"Axis angle = {angle:.1f} deg (expected near 45)"


class TestEndpointDistance:
    """Test that endpoints span the mask extent along major axis."""

    def test_endpoint_distance_matches_mask_length(self):
        # Horizontal bar: 200 pixels wide (cols 50-249)
        mask = np.zeros((100, 300), dtype=np.uint8)
        mask[40:60, 50:250] = 255

        _center, ep_a, ep_b = extract_keypoints(mask)
        # Distance between endpoints should approximately match mask width (200 px)
        dist = np.linalg.norm(ep_a - ep_b)
        # Major axis spans the 200-wide rectangle (some rounding from discrete pixels)
        assert abs(dist - 199.0) < 5.0, f"Endpoint distance = {dist:.1f}, expected ~199"


class TestCanonicalSign:
    """Test that canonical sign enforcement makes results deterministic."""

    def test_deterministic_results(self):
        # Same mask should always produce identical output
        mask = np.zeros((100, 300), dtype=np.uint8)
        mask[40:60, 50:250] = 255

        results = [extract_keypoints(mask) for _ in range(5)]
        for i in range(1, 5):
            assert np.allclose(results[0][0], results[i][0]), "center not deterministic"
            assert np.allclose(results[0][1], results[i][1]), (
                "endpoint_a not deterministic"
            )
            assert np.allclose(results[0][2], results[i][2]), (
                "endpoint_b not deterministic"
            )

    def test_endpoint_a_has_max_projection(self):
        # endpoint_a should be the one with maximum projection onto major axis
        mask = np.zeros((100, 300), dtype=np.uint8)
        mask[40:60, 50:250] = 255

        center, ep_a, ep_b = extract_keypoints(mask)
        # ep_a should have projection > ep_b (canonical sign)
        # Compute a rough major axis direction
        axis = ep_a - center
        # ep_b should project negatively relative to ep_a
        proj_b = np.dot(ep_b - center, axis / (np.linalg.norm(axis) + 1e-8))
        assert proj_b < 0, f"ep_b should project negatively: {proj_b}"


class TestSinglePixelMask:
    """Test degenerate case: single-pixel mask."""

    def test_single_pixel_no_crash(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255

        center, ep_a, ep_b = extract_keypoints(mask)
        # All three keypoints should be that pixel
        assert np.allclose(center, [50.0, 50.0], atol=1.0)
        assert np.allclose(ep_a, center, atol=1.0)
        assert np.allclose(ep_b, center, atol=1.0)


class TestEmptyMask:
    """Test that empty mask raises ValueError."""

    def test_empty_mask_raises(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"[Ee]mpty|[Nn]o foreground|[Nn]o pixel"):
            extract_keypoints(mask)

    def test_bool_mask_accepted(self):
        mask = np.zeros((100, 200), dtype=bool)
        mask[40:60, 50:150] = True
        # Should work with bool masks
        center, _ep_a, _ep_b = extract_keypoints(mask)
        assert center.shape == (2,)


class TestBatchExtraction:
    """Test batch extraction API."""

    def test_batch_returns_list_of_tuples(self):
        masks = [
            np.zeros((100, 300), dtype=np.uint8),
            np.zeros((100, 300), dtype=np.uint8),
            np.zeros((100, 300), dtype=np.uint8),
        ]
        masks[0][40:60, 50:250] = 255
        masks[1][30:70, 30:270] = 255
        masks[2][45:55, 60:240] = 255

        results = extract_keypoints_batch(masks)
        assert len(results) == 3
        for center, ep_a, ep_b in results:
            assert center.shape == (2,)
            assert ep_a.shape == (2,)
            assert ep_b.shape == (2,)

    def test_batch_single_mask(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[40:60, 50:150] = 255
        results = extract_keypoints_batch([mask])
        assert len(results) == 1
