"""Tests for holdout IoU evaluation and visual overlay rendering."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from aquapose.mesh.state import FishState
from aquapose.optimization.validation import (
    evaluate_holdout_iou,
    render_overlay,
    run_holdout_validation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_state(x: float = 0.0, y: float = 0.0, z: float = 1.5) -> FishState:
    """Create a minimal FishState at the given position."""
    return FishState(
        p=torch.tensor([x, y, z]),
        psi=torch.tensor(0.0),
        theta=torch.tensor(0.0),
        kappa=torch.tensor(0.0),
        s=torch.tensor(0.15),
    )


def _make_mock_renderer(alpha_value: float = 0.8, H: int = 64, W: int = 64):
    """Build a mock renderer that returns a constant alpha map."""
    renderer = MagicMock()

    def _render(meshes, cameras, camera_ids, **kwargs):
        result = {}
        for cam_id in camera_ids:
            result[cam_id] = torch.full((H, W), alpha_value)
        return result

    renderer.render.side_effect = _render
    return renderer


def _make_mock_camera():
    """Build a minimal mock camera (content irrelevant for holdout tests)."""
    return MagicMock()


# ---------------------------------------------------------------------------
# evaluate_holdout_iou tests
# ---------------------------------------------------------------------------


def test_evaluate_holdout_iou_perfect():
    """Alpha == target mask -> IoU should be near 1.0."""
    H, W = 32, 32
    renderer = _make_mock_renderer(alpha_value=1.0, H=H, W=W)

    state = _make_state()
    camera = _make_mock_camera()
    # Perfect mask: all ones (matches the renderer's alpha=1.0).
    mask = torch.ones(H, W)

    iou = evaluate_holdout_iou(state, camera, mask, renderer)

    assert iou > 0.95, f"Expected IoU near 1.0 for perfect overlap, got {iou:.4f}"


def test_evaluate_holdout_iou_no_overlap():
    """Disjoint prediction and target -> IoU should be near 0.0."""
    H, W = 32, 32

    # Renderer predicts alpha=1 in left half (mocked).
    # Target mask has ones only in right half -> no overlap.
    renderer = MagicMock()

    def _render_left(meshes, cameras, camera_ids, **kwargs):
        alpha = torch.zeros(H, W)
        alpha[:, : W // 2] = 1.0  # left half
        return {cam_id: alpha for cam_id in camera_ids}

    renderer.render.side_effect = _render_left

    state = _make_state()
    camera = _make_mock_camera()
    mask = torch.zeros(H, W)
    mask[:, W // 2 :] = 1.0  # right half

    iou = evaluate_holdout_iou(state, camera, mask, renderer)

    assert iou < 0.05, f"Expected IoU near 0.0 for disjoint masks, got {iou:.4f}"


def test_evaluate_holdout_iou_with_crop_region():
    """Crop region restricts IoU computation to a subregion."""
    H, W = 64, 64
    renderer = _make_mock_renderer(alpha_value=1.0, H=H, W=W)

    state = _make_state()
    camera = _make_mock_camera()
    mask = torch.ones(H, W)
    crop_region = (10, 10, 40, 40)

    iou = evaluate_holdout_iou(state, camera, mask, renderer, crop_region=crop_region)

    # Perfect overlap in crop region -> IoU near 1.0.
    assert iou > 0.95, f"Expected IoU near 1.0 with crop region, got {iou:.4f}"


def test_evaluate_holdout_iou_returns_float():
    """Return type must be a Python float, not a tensor."""
    H, W = 16, 16
    renderer = _make_mock_renderer(alpha_value=0.5, H=H, W=W)
    state = _make_state()
    camera = _make_mock_camera()
    mask = torch.ones(H, W) * 0.5

    iou = evaluate_holdout_iou(state, camera, mask, renderer)

    assert isinstance(iou, float), f"Expected float, got {type(iou)}"


# ---------------------------------------------------------------------------
# render_overlay tests
# ---------------------------------------------------------------------------


def test_render_overlay_shape():
    """Output shape must match input frame shape."""
    H, W = 120, 160
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    alpha = np.ones((H, W), dtype=np.float32) * 0.5

    result = render_overlay(frame, alpha)

    assert result.shape == (H, W, 3), f"Expected {(H, W, 3)}, got {result.shape}"


def test_render_overlay_dtype():
    """Output must be uint8."""
    H, W = 64, 64
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    alpha = np.zeros((H, W), dtype=np.float32)

    result = render_overlay(frame, alpha)

    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"


def test_render_overlay_modifies_pixels():
    """Output must differ from input where alpha > 0."""
    H, W = 64, 64
    frame = np.zeros((H, W, 3), dtype=np.uint8)  # black frame
    alpha = np.zeros((H, W), dtype=np.float32)
    alpha[20:40, 20:40] = 1.0  # bright region in center

    result = render_overlay(frame, alpha, color=(0, 255, 0), opacity=1.0)

    # Pixels under alpha=1 should be green (non-zero G channel).
    assert result[30, 30, 1] > 0, "Expected green overlay where alpha=1"
    # Background (alpha=0) should remain unchanged (black).
    assert result[0, 0, 0] == 0
    assert result[0, 0, 1] == 0
    assert result[0, 0, 2] == 0


def test_render_overlay_zero_alpha_unchanged():
    """Pixels with alpha=0 must be unchanged from input frame."""
    H, W = 64, 64
    frame = np.full((H, W, 3), 128, dtype=np.uint8)  # mid-gray frame
    alpha = np.zeros((H, W), dtype=np.float32)  # all zeros -> no overlay

    result = render_overlay(frame, alpha, opacity=0.9)

    np.testing.assert_array_equal(result, frame)


def test_render_overlay_with_crop_region():
    """Crop region correctly places crop-sized alpha into full frame."""
    H, W = 128, 128
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    crop_region = (10, 20, 50, 80)  # y1, x1, y2, x2
    crop_h = 50 - 10
    crop_w = 80 - 20
    alpha_crop = np.ones((crop_h, crop_w), dtype=np.float32)

    # In BGR, (0, 0, 255) is red (R channel = index 2).
    result = render_overlay(
        frame, alpha_crop, crop_region=crop_region, color=(0, 0, 255), opacity=1.0
    )

    # Inside crop region -> should be overlaid (non-zero R channel; index 2 in BGR is red).
    assert result[30, 50, 2] > 0, "Expected red overlay inside crop region"
    # Outside crop region -> should remain black.
    assert result[0, 0, 0] == 0
    assert result[0, 0, 1] == 0
    assert result[0, 0, 2] == 0


# ---------------------------------------------------------------------------
# run_holdout_validation tests
# ---------------------------------------------------------------------------


def test_run_holdout_validation_structure():
    """Returned dict must have all expected keys with correct types."""
    H, W = 32, 32
    n_frames = 4
    camera_ids = ["cam_a", "cam_b", "cam_c"]
    cameras = [_make_mock_camera() for _ in camera_ids]

    mask = torch.ones(H, W)
    frames_data = [
        {
            "target_masks": {cid: mask.clone() for cid in camera_ids},
            "crop_regions": {cid: None for cid in camera_ids},
        }
        for _ in range(n_frames)
    ]

    states = [_make_state(z=float(i) * 0.1 + 1.0) for i in range(n_frames)]
    renderer = _make_mock_renderer(alpha_value=1.0, H=H, W=W)
    mock_optimizer = MagicMock()

    result = run_holdout_validation(
        states, frames_data, cameras, camera_ids, renderer, mock_optimizer
    )

    # Check all required keys are present.
    required_keys = [
        "global_mean_iou",
        "per_camera_iou",
        "per_frame_iou",
        "min_camera_iou",
        "target_met_080",
        "target_met_060_floor",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    assert isinstance(result["global_mean_iou"], float)
    assert isinstance(result["per_camera_iou"], dict)
    assert isinstance(result["per_frame_iou"], list)
    assert isinstance(result["min_camera_iou"], float)
    assert isinstance(result["target_met_080"], bool)
    assert isinstance(result["target_met_060_floor"], bool)


def test_run_holdout_validation_per_frame_entries():
    """per_frame_iou entries must have expected fields."""
    H, W = 32, 32
    camera_ids = ["cam_a", "cam_b"]
    cameras = [_make_mock_camera() for _ in camera_ids]
    mask = torch.ones(H, W)
    n_frames = 4
    frames_data = [
        {
            "target_masks": {cid: mask.clone() for cid in camera_ids},
            "crop_regions": {cid: None for cid in camera_ids},
        }
        for _ in range(n_frames)
    ]
    states = [_make_state() for _ in range(n_frames)]
    renderer = _make_mock_renderer(alpha_value=0.9, H=H, W=W)
    mock_optimizer = MagicMock()

    result = run_holdout_validation(
        states, frames_data, cameras, camera_ids, renderer, mock_optimizer
    )

    for entry in result["per_frame_iou"]:
        assert "frame_idx" in entry
        assert "held_out_camera" in entry
        assert "iou" in entry
        assert isinstance(entry["iou"], float)


def test_run_holdout_validation_perfect_iou():
    """Perfect overlap (alpha=1, mask=1) -> global_mean_iou near 1.0."""
    H, W = 32, 32
    camera_ids = ["cam_a", "cam_b", "cam_c"]
    cameras = [_make_mock_camera() for _ in camera_ids]
    mask = torch.ones(H, W)
    n_frames = 6
    frames_data = [
        {
            "target_masks": {cid: mask.clone() for cid in camera_ids},
            "crop_regions": {cid: None for cid in camera_ids},
        }
        for _ in range(n_frames)
    ]
    states = [_make_state() for _ in range(n_frames)]
    renderer = _make_mock_renderer(alpha_value=1.0, H=H, W=W)
    mock_optimizer = MagicMock()

    result = run_holdout_validation(
        states, frames_data, cameras, camera_ids, renderer, mock_optimizer
    )

    assert result["global_mean_iou"] > 0.95
    assert result["target_met_080"] is True
