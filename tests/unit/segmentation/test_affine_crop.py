"""Tests for affine crop utilities in segmentation/crop.py.

Validates:
- extract_affine_crop() produces correctly-sized, rotation-aligned crops
- invert_affine_point() back-projects with < 1px round-trip error
- invert_affine_points() batch API works identically
- Zero-fill for out-of-bounds regions
- Round-trip accuracy across multiple rotation angles (DET-03)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aquapose.segmentation.crop import (
    AffineCrop,
    extract_affine_crop,
    invert_affine_point,
    invert_affine_points,
)

# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "crop_size",
    [(256, 128), (64, 64), (320, 160), (100, 50)],
)
def test_affine_crop_output_shape(crop_size: tuple[int, int]) -> None:
    """AffineCrop.image.shape[:2] matches (crop_h, crop_w) for various crop sizes."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = extract_affine_crop(
        frame=frame,
        center_xy=(320.0, 240.0),
        angle_math_rad=0.0,
        obb_w=100.0,
        obb_h=40.0,
        crop_size=crop_size,
    )

    assert isinstance(result, AffineCrop)
    crop_w, crop_h = crop_size
    assert result.image.shape[:2] == (crop_h, crop_w), (
        f"Expected shape ({crop_h}, {crop_w}), got {result.image.shape[:2]}"
    )
    assert result.crop_size == crop_size
    assert result.M.shape == (2, 3)


# ---------------------------------------------------------------------------
# Identity rotation (angle = 0)
# ---------------------------------------------------------------------------


def test_extract_affine_crop_identity_rotation() -> None:
    """angle=0 crop should contain the frame region around bbox centre."""
    # Create a frame with a distinct pattern at the centre
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    # Place a bright rectangle centred at (150, 100)
    frame[90:110, 140:160] = 200  # bright patch around (150, 100)

    result = extract_affine_crop(
        frame=frame,
        center_xy=(150.0, 100.0),
        angle_math_rad=0.0,
        obb_w=60.0,
        obb_h=30.0,
        crop_size=(60, 30),
    )

    # With identity rotation and centre at crop centre, the bright patch
    # should appear at the crop centre (crop_w/2, crop_h/2) = (30, 15)
    crop_h, crop_w = result.image.shape[:2]
    centre_region = result.image[
        crop_h // 2 - 5 : crop_h // 2 + 5,
        crop_w // 2 - 5 : crop_w // 2 + 5,
    ]
    assert centre_region.max() > 100, (
        "Bright patch should appear at crop centre for zero-rotation"
    )


# ---------------------------------------------------------------------------
# 90-degree rotation
# ---------------------------------------------------------------------------


def test_extract_affine_crop_90_degree_rotation() -> None:
    """angle=pi/2 (90 deg CCW): a horizontal bar in the frame appears vertical in crop."""
    # Create a frame with a horizontal bar (wide, narrow height)
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    # Horizontal bar at y=140..160, spanning x=150..250
    frame[140:160, 150:250] = 180

    # With angle=pi/2 (90 deg CCW), rotating the crop by 90 CW should
    # make the horizontal bar appear vertical in the crop output
    result = extract_affine_crop(
        frame=frame,
        center_xy=(200.0, 150.0),  # centre of horizontal bar
        angle_math_rad=math.pi / 2,
        obb_w=100.0,
        obb_h=20.0,
        crop_size=(120, 120),
    )

    crop = result.image
    _crop_h, crop_w = crop.shape[:2]

    # In the rotated crop, the bar should appear vertical at the crop centre
    # Check that a vertical strip at centre has high values
    cx = crop_w // 2
    vertical_strip = crop[:, max(0, cx - 5) : cx + 5]

    # The vertical strip (through the crop centre) should have high brightness
    assert vertical_strip.max() > 100, (
        "After 90-deg rotation, vertical strip should contain bright pixels"
    )


# ---------------------------------------------------------------------------
# Round-trip accuracy — single point (DET-03)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "angle_rad",
    [0.0, math.pi / 4, math.pi / 2, -math.pi / 3, math.pi / 6, -math.pi],
)
def test_round_trip_accuracy(angle_rad: float) -> None:
    """Crop -> invert_affine_point round-trip error < 1.0 pixel."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    center_xy = (320.0, 240.0)
    crop_size = (256, 128)

    result = extract_affine_crop(
        frame=frame,
        center_xy=center_xy,
        angle_math_rad=angle_rad,
        obb_w=150.0,
        obb_h=50.0,
        crop_size=crop_size,
    )

    # Test multiple points in crop space
    test_points_frame = [
        (320.0, 240.0),  # OBB centre
        (200.0, 150.0),
        (400.0, 300.0),
        (100.0, 100.0),
        (500.0, 350.0),
    ]

    for fx, fy in test_points_frame:
        # Forward: frame -> crop using M (affine matrix-vector multiply)
        pt_crop_cv = result.M @ np.array([fx, fy, 1.0])
        crop_x, crop_y = pt_crop_cv[0], pt_crop_cv[1]

        # Backward: crop -> frame using inverse
        fx_recovered, fy_recovered = invert_affine_point((crop_x, crop_y), result.M)

        error = math.sqrt((fx_recovered - fx) ** 2 + (fy_recovered - fy) ** 2)
        assert error < 1.0, (
            f"Round-trip error {error:.4f}px for angle={angle_rad:.4f}rad, "
            f"point=({fx}, {fy}). Expected < 1.0px."
        )


# ---------------------------------------------------------------------------
# Round-trip accuracy — batch API (DET-03)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "angle_rad",
    [0.0, math.pi / 4, math.pi / 2, -math.pi / 3],
)
def test_round_trip_batch(angle_rad: float) -> None:
    """invert_affine_points batch API gives same result as single-point version."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    crop_size = (256, 128)

    result = extract_affine_crop(
        frame=frame,
        center_xy=(320.0, 240.0),
        angle_math_rad=angle_rad,
        obb_w=150.0,
        obb_h=50.0,
        crop_size=crop_size,
    )

    # Generate test crop points by forward-projecting frame points
    frame_points = np.array(
        [[200.0, 150.0], [300.0, 200.0], [400.0, 300.0], [150.0, 100.0]],
        dtype=np.float64,
    )

    # Forward project all points using M
    ones = np.ones((len(frame_points), 1))
    pts_h = np.hstack([frame_points, ones])  # (N, 3)
    crop_points = (result.M @ pts_h.T).T  # (N, 2)

    # Batch back-projection
    recovered_batch = invert_affine_points(crop_points, result.M)

    # Single-point back-projection for comparison
    for i, (cx, cy) in enumerate(crop_points):
        rx, ry = invert_affine_point((cx, cy), result.M)
        assert abs(recovered_batch[i, 0] - rx) < 1e-9, "Batch x mismatch"
        assert abs(recovered_batch[i, 1] - ry) < 1e-9, "Batch y mismatch"

    # Verify round-trip error < 1.0px for batch results
    errors = np.sqrt(np.sum((recovered_batch - frame_points) ** 2, axis=1))
    assert np.all(errors < 1.0), (
        f"Batch round-trip errors > 1.0px: {errors[errors >= 1.0]}"
    )


# ---------------------------------------------------------------------------
# Border fill
# ---------------------------------------------------------------------------


def test_affine_crop_border_fill_is_zero() -> None:
    """Pixels outside the source frame are zero-filled."""
    # Small frame, large crop — most of crop canvas will be outside source
    frame = np.ones((50, 50, 3), dtype=np.uint8) * 127  # mid-gray frame

    large_crop_size = (400, 400)
    result = extract_affine_crop(
        frame=frame,
        center_xy=(25.0, 25.0),  # centre of small frame
        angle_math_rad=0.0,
        obb_w=50.0,
        obb_h=50.0,
        crop_size=large_crop_size,
    )

    crop = result.image
    # Check the four corners are zero (they are far outside the source frame)
    corners = [
        crop[0, 0],
        crop[0, -1],
        crop[-1, 0],
        crop[-1, -1],
    ]
    for corner in corners:
        assert np.all(corner == 0), f"Corner pixel should be zero, got {corner}"

    # Check that centre region has non-zero values (from the actual frame)
    cy, cx = 200, 200
    centre_patch = crop[cy - 10 : cy + 10, cx - 10 : cx + 10]
    assert centre_patch.max() > 0, "Centre of crop should contain frame pixels"
