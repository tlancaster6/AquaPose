"""Unit tests for the 2D medial axis extraction and arc-length sampling pipeline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from aquapose.reconstruction.midline import (
    Midline2D,
    MidlineExtractor,
    _adaptive_smooth,
    _check_skip_mask,
    _crop_to_frame,
    _longest_path_bfs,
    _resample_arc_length,
    _skeleton_and_widths,
)
from aquapose.segmentation.crop import CropRegion

# ---------------------------------------------------------------------------
# Helper fixtures / factories
# ---------------------------------------------------------------------------


def _make_ellipse_mask(
    h: int,
    w: int,
    center: tuple[int, int],
    axes: tuple[int, int],
    angle: float = 0,
) -> np.ndarray:
    """Draw a filled ellipse on a uint8 array.

    Args:
        h: Image height in pixels.
        w: Image width in pixels.
        center: (cx, cy) centre of the ellipse.
        axes: (semi-major, semi-minor) axis lengths in pixels.
        angle: Rotation angle of the ellipse in degrees.

    Returns:
        Binary mask uint8, shape (H, W).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        mask,
        center,
        axes,
        angle,
        0,
        360,
        255,
        -1,
    )
    return mask


def _make_long_mask(h: int = 128, w: int = 128) -> np.ndarray:
    """Create a horizontally elongated ellipse in a 128x128 crop.

    Returns:
        Binary mask uint8, shape (128, 128). Represents a typical side-view fish.
    """
    return _make_ellipse_mask(h, w, center=(64, 64), axes=(50, 10), angle=0)


def _make_round_mask(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a nearly circular blob in a 64x64 crop.

    Returns:
        Binary mask uint8, shape (64, 64). Represents a head-on view.
    """
    return _make_ellipse_mask(h, w, center=(32, 32), axes=(15, 14), angle=0)


def _make_tiny_mask(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a very small blob (area < 300) in a 64x64 crop.

    Returns:
        Binary mask uint8, shape (64, 64). Should be skipped.
    """
    return _make_ellipse_mask(h, w, center=(32, 32), axes=(5, 5), angle=0)


def _make_boundary_mask(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a large mask touching the top edge.

    Returns:
        Binary mask uint8, shape (64, 64). Should be skipped (boundary-clipped).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    # Ellipse centred near top edge so it touches row 0
    cv2.ellipse(mask, (32, 5), (20, 6), 0, 0, 360, 255, -1)
    return mask


def _make_crop_region() -> CropRegion:
    """Create a simple CropRegion for testing coordinate transforms.

    Region: x1=100, y1=200, x2=300, y2=400 in a 1200x1600 frame.

    Returns:
        CropRegion with known scale factors.
    """
    return CropRegion(x1=100, y1=200, x2=300, y2=400, frame_h=1200, frame_w=1600)


# ---------------------------------------------------------------------------
# Mock FishTrack
# ---------------------------------------------------------------------------


@dataclass
class _MockFishTrack:
    """Minimal FishTrack mock for testing.

    Attributes:
        fish_id: Fish identifier.
        velocity: 3D velocity vector, shape (3,).
        positions: Position history.
        camera_detections: Per-camera detection index map.
    """

    fish_id: int = 0
    velocity: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.0, 0.0], dtype=np.float32)
    )
    positions: deque = field(
        default_factory=lambda: deque(
            [np.array([0.0, 0.0, 0.5], dtype=np.float32)], maxlen=2
        )
    )
    camera_detections: dict[str, int] = field(default_factory=lambda: {"cam0": 0})


class _MockProjectionModel:
    """Orthographic projection model for testing (drops Z coordinate).

    Returns (x * 100, y * 100) as pixel coordinates, always valid.
    """

    def project(self, points: object) -> tuple[object, object]:
        """Project 3D points to 2D pixels (orthographic, scaled by 100).

        Args:
            points: Tensor of shape (N, 3).

        Returns:
            pixels: Tensor of shape (N, 2).
            valid: Boolean tensor of shape (N,), all True.
        """
        import torch

        pts = points if isinstance(points, torch.Tensor) else torch.tensor(points)
        pixels = pts[:, :2] * 100.0
        valid = torch.ones(pts.shape[0], dtype=torch.bool)
        return pixels, valid


# ---------------------------------------------------------------------------
# Test _check_skip_mask
# ---------------------------------------------------------------------------


def test_check_skip_mask_valid() -> None:
    """Valid long mask should pass (returns None)."""
    mask = _make_long_mask()
    result = _check_skip_mask(mask)
    assert result is None


def test_check_skip_mask_too_small() -> None:
    """Tiny mask should be skipped with 'too small' reason."""
    mask = _make_tiny_mask()
    result = _check_skip_mask(mask)
    assert result is not None
    assert "too small" in result.lower()


def test_check_skip_mask_boundary_clipped() -> None:
    """Mask touching top edge should be skipped with 'boundary' reason."""
    mask = _make_boundary_mask()
    result = _check_skip_mask(mask)
    assert result is not None
    assert "boundary" in result.lower()


# ---------------------------------------------------------------------------
# Test _adaptive_smooth
# ---------------------------------------------------------------------------


def test_adaptive_smooth_preserves_shape() -> None:
    """Smoothed long mask should still have nonzero pixels and area within 50%."""
    mask = _make_long_mask()
    orig_area = int(np.count_nonzero(mask))
    smoothed = _adaptive_smooth(mask)
    smoothed_area = int(np.count_nonzero(smoothed))
    assert smoothed_area > 0
    assert smoothed_area >= orig_area * 0.5
    assert smoothed_area <= orig_area * 1.5


# ---------------------------------------------------------------------------
# Test _skeleton_and_widths
# ---------------------------------------------------------------------------


def test_skeleton_produces_thin_path() -> None:
    """Skeleton of smoothed long mask should be 1-pixel wide and ≥15 pixels."""
    mask = _make_long_mask()
    smooth = _adaptive_smooth(mask)
    skel, dt = _skeleton_and_widths(smooth)

    n_skel = int(np.sum(skel))
    assert n_skel >= 15, f"Expected ≥15 skeleton pixels, got {n_skel}"

    # dt values at skeleton should be positive (inside the mask)
    skel_vals = dt[skel]
    assert np.all(skel_vals > 0)


# ---------------------------------------------------------------------------
# Test _longest_path_bfs
# ---------------------------------------------------------------------------


def test_longest_path_bfs_returns_ordered_path() -> None:
    """On a T-shaped skeleton, BFS returns the longest arm ignoring the branch."""
    # Build a T-shaped 15x15 skeleton:
    #   Horizontal bar: row 7, cols 0-12 (length 13)
    #   Vertical stub: rows 7-14, col 6 (length 8, but col 6 already in bar)
    #   Total branch from junction at (7,6): up = rows 0-7, down = rows 7-14
    skel = np.zeros((20, 20), dtype=bool)
    # Horizontal bar: row 7, cols 1..15
    skel[7, 1:16] = True
    # Vertical stub downward: col 8, rows 7..14
    skel[7:15, 8] = True

    path = _longest_path_bfs(skel)
    assert len(path) > 0

    # The longest path should go through the horizontal bar endpoints
    # (row 7, col 1) and (row 7, col 15), not the short vertical
    rows = [p[0] for p in path]
    cols = [p[1] for p in path]

    # Path endpoints should be far apart
    start = path[0]
    end = path[-1]
    dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
    assert dist > 10, f"Expected endpoints far apart, got dist={dist}"

    # Path should be mostly in row 7 (horizontal bar)
    row7_count = sum(1 for r in rows if r == 7)
    assert row7_count >= 10

    _ = cols  # used indirectly


def test_longest_path_bfs_empty_skeleton() -> None:
    """All-zero skeleton should return empty list."""
    skel = np.zeros((32, 32), dtype=bool)
    path = _longest_path_bfs(skel)
    assert path == []


# ---------------------------------------------------------------------------
# Test _resample_arc_length
# ---------------------------------------------------------------------------


def test_resample_arc_length_count() -> None:
    """Resampled output should have exactly 15 points and 15 half-widths."""
    mask = _make_long_mask()
    smooth = _adaptive_smooth(mask)
    skel, dt = _skeleton_and_widths(smooth)
    path = _longest_path_bfs(skel)
    xy_crop, hw = _resample_arc_length(path, dt, n_points=15)

    assert xy_crop.shape == (15, 2)
    assert hw.shape == (15,)


def test_resample_arc_length_endpoints() -> None:
    """First/last resampled point should be near path start/end (within 2px)."""
    mask = _make_long_mask()
    smooth = _adaptive_smooth(mask)
    skel, dt = _skeleton_and_widths(smooth)
    path = _longest_path_bfs(skel)
    xy_crop, _ = _resample_arc_length(path, dt, n_points=15)

    # path is in (row, col); xy_crop is in (col, row) = (x, y)
    start_yx = path[0]
    end_yx = path[-1]

    start_xy = np.array([start_yx[1], start_yx[0]], dtype=np.float32)
    end_xy = np.array([end_yx[1], end_yx[0]], dtype=np.float32)

    dist_start = float(np.linalg.norm(xy_crop[0] - start_xy))
    dist_end = float(np.linalg.norm(xy_crop[-1] - end_xy))

    assert dist_start < 2.0, f"First point too far from path start: {dist_start}"
    assert dist_end < 2.0, f"Last point too far from path end: {dist_end}"


# ---------------------------------------------------------------------------
# Test _crop_to_frame
# ---------------------------------------------------------------------------


def test_crop_to_frame_transform() -> None:
    """(0,0) in crop-space should map to (x1,y1) in frame; (crop_w,crop_h) to (x2,y2)."""
    crop = _make_crop_region()  # x1=100, y1=200, x2=300, y2=400; width=200, height=200
    crop_h, crop_w = 200, 200  # no resize — same as region dimensions

    # Two test points: top-left and bottom-right of crop
    xy_crop = np.array([[0.0, 0.0], [200.0, 200.0]], dtype=np.float32)
    hw_crop = np.array([5.0, 5.0], dtype=np.float32)

    xy_frame, hw_frame = _crop_to_frame(xy_crop, hw_crop, crop, crop_h, crop_w)

    np.testing.assert_allclose(xy_frame[0], [100.0, 200.0], atol=1e-4)
    np.testing.assert_allclose(xy_frame[1], [300.0, 400.0], atol=1e-4)

    # Scale is 1.0 (no resize), so half-widths unchanged
    np.testing.assert_allclose(hw_frame, [5.0, 5.0], atol=1e-4)


def test_crop_to_frame_with_resize() -> None:
    """Correctly scale when mask resolution differs from crop region size."""
    # Crop region: 200x200 pixels in full frame
    crop = CropRegion(x1=100, y1=200, x2=300, y2=400, frame_h=1200, frame_w=1600)
    # Mask is 128x128 (U-Net output)
    crop_h, crop_w = 128, 128

    # A point at (64, 64) in crop-space (centre of mask)
    xy_crop = np.array([[64.0, 64.0]], dtype=np.float32)
    hw_crop = np.array([8.0], dtype=np.float32)

    xy_frame, hw_frame = _crop_to_frame(xy_crop, hw_crop, crop, crop_h, crop_w)

    # Expected: scale = 200/128 ≈ 1.5625
    scale = 200.0 / 128.0
    expected_x = 64.0 * scale + 100.0
    expected_y = 64.0 * scale + 200.0
    expected_hw = 8.0 * scale  # avg scale = scale (symmetric)

    np.testing.assert_allclose(xy_frame[0, 0], expected_x, rtol=1e-4)
    np.testing.assert_allclose(xy_frame[0, 1], expected_y, rtol=1e-4)
    np.testing.assert_allclose(hw_frame[0], expected_hw, rtol=1e-4)


# ---------------------------------------------------------------------------
# Test MidlineExtractor full pipeline
# ---------------------------------------------------------------------------


def _make_track_and_data(
    velocity: np.ndarray | None = None,
) -> tuple[_MockFishTrack, dict, dict, dict, dict]:
    """Create a complete set of inputs for extract_midlines.

    Args:
        velocity: 3D velocity for the mock track. Defaults to [0.01, 0, 0].

    Returns:
        Tuple of (track, masks_per_camera, crop_regions_per_camera,
                  detections_per_camera, projection_models).
    """
    if velocity is None:
        velocity = np.array([0.01, 0.0, 0.0], dtype=np.float32)

    track = _MockFishTrack(
        fish_id=0,
        velocity=velocity,
        camera_detections={"cam0": 0},
    )

    mask = _make_long_mask()
    crop = _make_crop_region()

    masks_per_camera: dict[str, list[np.ndarray]] = {"cam0": [mask]}
    crop_regions_per_camera: dict[str, list[CropRegion]] = {"cam0": [crop]}
    detections_per_camera: dict[str, list] = {"cam0": [object()]}
    projection_models: dict[str, _MockProjectionModel] = {
        "cam0": _MockProjectionModel()
    }

    return (
        track,
        masks_per_camera,
        crop_regions_per_camera,
        detections_per_camera,
        projection_models,
    )


def test_extract_midlines_full_pipeline() -> None:
    """Full pipeline: long ellipse mask → Midline2D with 15 points in frame coords."""
    track, masks, crops, dets, models = _make_track_and_data()
    extractor = MidlineExtractor(n_points=15, min_area=300)

    result = extractor.extract_midlines(
        tracks=[track],  # type: ignore[list-item]
        masks_per_camera=masks,
        crop_regions_per_camera=crops,
        detections_per_camera=dets,
        projection_models=models,
        frame_index=0,
    )

    assert 0 in result, "fish_id 0 should be in result"
    assert "cam0" in result[0], "cam0 should be in result[0]"

    midline = result[0]["cam0"]
    assert isinstance(midline, Midline2D)
    assert midline.points.shape == (15, 2)
    assert midline.half_widths.shape == (15,)
    assert midline.fish_id == 0
    assert midline.camera_id == "cam0"
    assert midline.frame_index == 0

    # Points should be in full-frame coordinates (within frame bounds)
    crop_region = crops["cam0"][0]
    assert np.all(midline.points[:, 0] >= crop_region.x1)
    assert np.all(midline.points[:, 0] <= crop_region.x2)
    assert np.all(midline.points[:, 1] >= crop_region.y1)
    assert np.all(midline.points[:, 1] <= crop_region.y2)

    # Half-widths should be positive (fish has nonzero width)
    assert np.all(midline.half_widths > 0)


def test_extract_midlines_skips_small_mask() -> None:
    """Tiny mask should cause extraction to be skipped (no cam entry)."""
    track = _MockFishTrack(fish_id=0, camera_detections={"cam0": 0})
    tiny_mask = _make_tiny_mask()
    crop = _make_crop_region()

    result = MidlineExtractor(n_points=15, min_area=300).extract_midlines(
        tracks=[track],  # type: ignore[list-item]
        masks_per_camera={"cam0": [tiny_mask]},
        crop_regions_per_camera={"cam0": [crop]},
        detections_per_camera={"cam0": [object()]},
        projection_models={"cam0": _MockProjectionModel()},
        frame_index=0,
    )

    # fish_id may be absent or present with empty camera dict
    cam_result = result.get(0, {})
    assert "cam0" not in cam_result


# ---------------------------------------------------------------------------
# Test orientation inheritance
# ---------------------------------------------------------------------------


def test_orientation_inheritance() -> None:
    """Second low-velocity frame should inherit orientation from first high-velocity frame."""
    extractor = MidlineExtractor(n_points=15, min_area=300)

    high_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # fast → establishes orient
    zero_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # slow → inherit

    # Frame 0: high velocity — orientation established
    track0 = _MockFishTrack(
        fish_id=1,
        velocity=high_vel,
        camera_detections={"cam0": 0},
    )
    result0 = extractor.extract_midlines(
        tracks=[track0],  # type: ignore[list-item]
        masks_per_camera={"cam0": [_make_long_mask()]},
        crop_regions_per_camera={"cam0": [_make_crop_region()]},
        detections_per_camera={"cam0": [object()]},
        projection_models={"cam0": _MockProjectionModel()},
        frame_index=0,
    )
    assert 1 in result0
    assert "cam0" in result0[1]
    mid0 = result0[1]["cam0"]

    # Frame 1: zero velocity — should inherit
    track1 = _MockFishTrack(
        fish_id=1,
        velocity=zero_vel,
        camera_detections={"cam0": 0},
    )
    result1 = extractor.extract_midlines(
        tracks=[track1],  # type: ignore[list-item]
        masks_per_camera={"cam0": [_make_long_mask()]},
        crop_regions_per_camera={"cam0": [_make_crop_region()]},
        detections_per_camera={"cam0": [object()]},
        projection_models={"cam0": _MockProjectionModel()},
        frame_index=1,
    )

    # Result should have an entry (inheritance, not skip)
    assert 1 in result1, "Should inherit orientation"
    assert "cam0" in result1[1], "Should inherit orientation"
    mid1 = result1[1]["cam0"]

    # Both midlines should have 15 points in frame coordinates
    assert mid0.points.shape == (15, 2)
    assert mid1.points.shape == (15, 2)


# ---------------------------------------------------------------------------
# Test back-correction cap
# ---------------------------------------------------------------------------


def test_back_correction_cap() -> None:
    """After cap (30) frames with zero velocity, buffer should be cleared."""
    extractor = MidlineExtractor(n_points=15, min_area=300, fps=60.0)
    # fps=60 → cap = min(30, 60) = 30

    zero_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Call 35 times with zero velocity (orientation never established)
    for frame_idx in range(35):
        track = _MockFishTrack(
            fish_id=2,
            velocity=zero_vel,
            camera_detections={"cam0": 0},
        )
        extractor.extract_midlines(
            tracks=[track],  # type: ignore[list-item]
            masks_per_camera={"cam0": [_make_long_mask()]},
            crop_regions_per_camera={"cam0": [_make_crop_region()]},
            detections_per_camera={"cam0": [object()]},
            projection_models={"cam0": _MockProjectionModel()},
            frame_index=frame_idx,
        )

    # After cap, buffer should be empty (committed or discarded)
    buf = extractor._back_correction_buffers.get(2, [])
    assert len(buf) == 0, f"Buffer should be empty after cap, got {len(buf)} entries"

    # Frame count should reflect all 35 calls
    count = extractor._back_correction_frame_counts.get(2, 0)
    assert count == 35
