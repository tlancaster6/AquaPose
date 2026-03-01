"""Unit tests for the pose regression model and training utilities."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from aquapose.training.pose import (
    KeypointDataset,
    _freeze_encoder,
    _load_backbone_weights,
    _masked_mse_loss,
    _mean_keypoint_error,
    _PoseModel,
    train_pose,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N_KP = 6
_IMG_SIZE = 32  # small synthetic images


@pytest.fixture
def coco_keypoint_dir(tmp_path: Path) -> Path:
    """Synthetic COCO keypoint dataset with 6 small images.

    Provides:
    - 6 solid-color 32x32 PNG images
    - annotations.json with 6 annotations (6 keypoints each)
    - Annotation 0: all 6 keypoints visible, safe coords in [5, 27]
    - Annotation 1: all 6 keypoints visible, different coords
    - Annotation 2: all 6 keypoints visible
    - Annotation 3: all 6 keypoints visible
    - Annotation 4: 3/6 keypoints visible (v=0 for the other 3) — partial visibility
    - Annotation 5: 1 keypoint has negative y (simulates real-data edge case from RESEARCH.md)
    """
    rng = np.random.default_rng(42)

    images_info = []
    annotations = []

    for i in range(6):
        fname = f"img_{i:03d}.png"
        # Solid-color image (different color per image)
        color = rng.integers(50, 200, size=3, dtype=np.uint8)
        img_array = np.full((32, 32, 3), color, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / fname), img_array)

        images_info.append({"id": i, "file_name": fname, "height": 32, "width": 32})

        # Build keypoint array [x, y, v, ...]
        kps: list[float] = []
        if i < 4:
            # All 6 visible, safe coords
            for _ in range(_N_KP):
                x = float(rng.integers(5, 28))
                y = float(rng.integers(5, 28))
                kps.extend([x, y, 2.0])
        elif i == 4:
            # 3 visible, 3 invisible
            for k in range(_N_KP):
                x = float(rng.integers(5, 28))
                y = float(rng.integers(5, 28))
                v = 2.0 if k < 3 else 0.0
                kps.extend([x, y, v])
        else:
            # i == 5: keypoint 0 has slightly negative y (real-data edge case)
            kps.extend([16.0, -0.9, 2.0])  # negative y — OOB in 128px space
            for _ in range(_N_KP - 1):
                x = float(rng.integers(5, 28))
                y = float(rng.integers(5, 28))
                kps.extend([x, y, 2.0])

        annotations.append(
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "keypoints": kps,
                "num_keypoints": _N_KP,
            }
        )

    coco = {
        "images": images_info,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "fish", "keypoints": [f"kp{k}" for k in range(_N_KP)]}
        ],
    }

    with open(tmp_path / "annotations.json", "w") as f:
        json.dump(coco, f)

    return tmp_path


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def test_pose_model_output_shape_default() -> None:
    """_PoseModel should output (B, n_keypoints*2) with default n_keypoints=6."""
    model = _PoseModel(n_keypoints=6, pretrained=False)
    model.eval()
    x = torch.zeros(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 12), f"Expected (2, 12), got {out.shape}"


def test_pose_model_output_shape_custom_keypoints() -> None:
    """_PoseModel should output (B, n_keypoints*2) for custom n_keypoints."""
    model = _PoseModel(n_keypoints=10, pretrained=False)
    model.eval()
    x = torch.zeros(3, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (3, 20), f"Expected (3, 20), got {out.shape}"


def test_pose_model_output_in_zero_one() -> None:
    """_PoseModel output should be in [0, 1] (Sigmoid final activation)."""
    model = _PoseModel(n_keypoints=6, pretrained=False)
    model.eval()
    x = torch.randn(4, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert float(out.min()) >= 0.0, "Output below 0"
    assert float(out.max()) <= 1.0, "Output above 1"


def test_pose_model_encoder_parameters_non_empty() -> None:
    """encoder_parameters() should return a non-empty list."""
    model = _PoseModel(n_keypoints=6, pretrained=False)
    enc_params = model.encoder_parameters()
    assert len(enc_params) > 0


def test_pose_model_head_parameters_non_empty() -> None:
    """head_parameters() should return a non-empty list."""
    model = _PoseModel(n_keypoints=6, pretrained=False)
    head_params = model.head_parameters()
    assert len(head_params) > 0


# ---------------------------------------------------------------------------
# Backbone weight loading
# ---------------------------------------------------------------------------


def _make_fake_unet_state_dict() -> dict[str, torch.Tensor]:
    """Build a minimal fake state-dict with enc* keys for testing."""
    from aquapose.segmentation.model import _UNet

    unet = _UNet(pretrained=False)
    # Return only encoder keys so _load_backbone_weights can find them
    return {k: v for k, v in unet.state_dict().items() if k.startswith("enc")}


def test_load_backbone_weights_sets_encoder_params(
    tmp_path: Path,
) -> None:
    """_load_backbone_weights should populate encoder params from a saved state-dict."""
    # Build a real _UNet state-dict and save it (enc* keys present)
    from aquapose.segmentation.model import _UNet

    unet = _UNet(pretrained=False)
    weights_path = tmp_path / "unet.pth"
    torch.save(unet.state_dict(), weights_path)

    # Build a fresh _PoseModel and load backbone weights
    model = _PoseModel(n_keypoints=6, pretrained=False)
    _load_backbone_weights(model, weights_path)

    # Verify encoder params match the saved UNet encoder params
    unet_enc_state = {k: v for k, v in unet.state_dict().items() if k.startswith("enc")}
    model_enc_state = {
        k: v for k, v in model.state_dict().items() if k.startswith("enc")
    }
    assert set(unet_enc_state.keys()) == set(model_enc_state.keys()), (
        "Encoder key sets should match"
    )
    for key in unet_enc_state:
        assert torch.allclose(unet_enc_state[key], model_enc_state[key]), (
            f"Encoder param {key!r} not loaded correctly"
        )


# ---------------------------------------------------------------------------
# Freeze / unfreeze behaviour
# ---------------------------------------------------------------------------


def test_freeze_encoder_disables_grad() -> None:
    """After _freeze_encoder, all encoder params should have requires_grad=False."""
    model = _PoseModel(n_keypoints=6, pretrained=False)
    # Ensure all params start as trainable
    for p in model.parameters():
        p.requires_grad_(True)

    _freeze_encoder(model)

    for param in model.encoder_parameters():
        assert not param.requires_grad, "Encoder param should be frozen"


def test_freeze_encoder_leaves_head_trainable() -> None:
    """_freeze_encoder should not affect head parameters."""
    model = _PoseModel(n_keypoints=6, pretrained=False)
    _freeze_encoder(model)

    for param in model.head_parameters():
        assert param.requires_grad, "Head param should remain trainable after freeze"


def test_no_freeze_encoder_all_params_trainable() -> None:
    """Without calling _freeze_encoder, all params should have requires_grad=True."""
    model = _PoseModel(n_keypoints=6, pretrained=False)
    for param in model.encoder_parameters():
        assert param.requires_grad, "Encoder param should be trainable by default"
    for param in model.head_parameters():
        assert param.requires_grad, "Head param should be trainable by default"


def test_backbone_weights_with_unfreeze_encoder_stays_trainable(
    tmp_path: Path,
) -> None:
    """When backbone_weights + unfreeze=True, encoder params remain trainable."""
    from aquapose.segmentation.model import _UNet

    unet = _UNet(pretrained=False)
    weights_path = tmp_path / "unet.pth"
    torch.save(unet.state_dict(), weights_path)

    # Simulate train_pose unfreeze=True: load weights, do NOT freeze
    model = _PoseModel(n_keypoints=6, pretrained=False)
    _load_backbone_weights(model, weights_path)
    # unfreeze=True means we skip _freeze_encoder — encoder stays trainable

    for param in model.encoder_parameters():
        assert param.requires_grad, (
            "Encoder param should be trainable when unfreeze=True"
        )


def test_backbone_weights_without_unfreeze_encoder_is_frozen(
    tmp_path: Path,
) -> None:
    """When backbone_weights + unfreeze=False (default), encoder params are frozen."""
    from aquapose.segmentation.model import _UNet

    unet = _UNet(pretrained=False)
    weights_path = tmp_path / "unet.pth"
    torch.save(unet.state_dict(), weights_path)

    # Simulate train_pose unfreeze=False: load weights then freeze
    model = _PoseModel(n_keypoints=6, pretrained=False)
    _load_backbone_weights(model, weights_path)
    _freeze_encoder(model)

    for param in model.encoder_parameters():
        assert not param.requires_grad, (
            "Encoder param should be frozen when unfreeze=False"
        )


# ---------------------------------------------------------------------------
# Metric utility (original tests — backward compatibility)
# ---------------------------------------------------------------------------


def test_mean_keypoint_error_zero_for_perfect_prediction() -> None:
    """_mean_keypoint_error should return 0.0 for identical pred and target."""
    t = torch.rand(4, 12)  # 6 keypoints
    assert _mean_keypoint_error(t, t) < 1e-6


def test_mean_keypoint_error_positive_for_different_tensors() -> None:
    """_mean_keypoint_error should return a positive value for differing tensors."""
    pred = torch.zeros(4, 12)
    target = torch.ones(4, 12)
    err = _mean_keypoint_error(pred, target)
    assert err > 0.0


# ---------------------------------------------------------------------------
# KeypointDataset 3-tuple return
# ---------------------------------------------------------------------------


def test_keypoint_dataset_clean_returns_3tuple(coco_keypoint_dir: Path) -> None:
    """KeypointDataset(augment=False) returns a 3-tuple with correct shapes."""
    ds = KeypointDataset(
        coco_keypoint_dir / "annotations.json",
        coco_keypoint_dir,
        n_keypoints=_N_KP,
        augment=False,
    )
    item = ds[0]
    assert len(item) == 3, "Expected 3-tuple (image, keypoints, visibility)"
    image, keypoints, visibility = item
    assert image.shape == (3, 64, 128), f"Image shape: {image.shape}"
    assert keypoints.shape == (_N_KP * 2,), f"Keypoints shape: {keypoints.shape}"
    assert visibility.shape == (_N_KP,), f"Visibility shape: {visibility.shape}"
    assert image.dtype == torch.float32
    assert keypoints.dtype == torch.float32
    assert visibility.dtype == torch.bool


def test_keypoint_dataset_augmented_returns_3tuple(coco_keypoint_dir: Path) -> None:
    """KeypointDataset(augment=True) returns a 3-tuple with correct shapes."""
    ds = KeypointDataset(
        coco_keypoint_dir / "annotations.json",
        coco_keypoint_dir,
        n_keypoints=_N_KP,
        augment=True,
    )
    item = ds[0]
    assert len(item) == 3, "Expected 3-tuple (image, keypoints, visibility)"
    image, keypoints, visibility = item
    assert image.shape == (3, 64, 128), f"Image shape: {image.shape}"
    assert keypoints.shape == (_N_KP * 2,), f"Keypoints shape: {keypoints.shape}"
    assert visibility.shape == (_N_KP,), f"Visibility shape: {visibility.shape}"


def test_keypoint_dataset_clean_visibility_from_coco(coco_keypoint_dir: Path) -> None:
    """Clean path derives visibility from COCO v flags and OOB check."""
    ds = KeypointDataset(
        coco_keypoint_dir / "annotations.json",
        coco_keypoint_dir,
        n_keypoints=_N_KP,
        augment=False,
    )
    # Annotation index 4 has exactly 3/6 visible keypoints
    _, _, vis4 = ds[4]
    assert int(vis4.sum()) == 3, (
        f"Expected 3 visible keypoints for partial annotation, got {vis4.sum()}"
    )

    # Annotation index 5 has keypoint 0 with y=-0.9 (negative → OOB after scaling)
    _, _, vis5 = ds[5]
    # Keypoint 0 should be invisible because y < 0 in pixel space
    assert not vis5[0].item(), (
        "Keypoint 0 with negative y should be marked invisible via OOB check"
    )


def test_keypoint_dataset_augmented_changes_image(coco_keypoint_dir: Path) -> None:
    """Augmented dataset applies transforms — most calls produce different images."""
    ds = KeypointDataset(
        coco_keypoint_dir / "annotations.json",
        coco_keypoint_dir,
        n_keypoints=_N_KP,
        augment=True,
    )
    # Collect 10 augmented versions of image 0
    images = [ds[0][0] for _ in range(10)]
    # Check that not all images are identical (augmentation is active)
    first = images[0]
    n_different = sum(1 for img in images[1:] if not torch.allclose(img, first))
    assert n_different > 0, (
        "Augmentation should produce at least some distinct images across 10 calls"
    )


def test_augmentation_oob_keypoints_zeroed(coco_keypoint_dir: Path) -> None:
    """For invisible keypoints, coordinates should be 0.0."""
    ds = KeypointDataset(
        coco_keypoint_dir / "annotations.json",
        coco_keypoint_dir,
        n_keypoints=_N_KP,
        augment=False,
    )
    # Check across all samples
    for idx in range(len(ds)):
        _, kps, vis = ds[idx]
        for k in range(_N_KP):
            if not vis[k].item():
                x_val = float(kps[k * 2])
                y_val = float(kps[k * 2 + 1])
                assert x_val == 0.0, (
                    f"Sample {idx}, keypoint {k}: x should be 0.0 when invisible, got {x_val}"
                )
                assert y_val == 0.0, (
                    f"Sample {idx}, keypoint {k}: y should be 0.0 when invisible, got {y_val}"
                )


# ---------------------------------------------------------------------------
# Masked MSE loss
# ---------------------------------------------------------------------------


def test_masked_mse_loss_zero_for_perfect_prediction() -> None:
    """_masked_mse_loss(t, t, all_true) returns ~0.0."""
    B, n_kp = 4, 6
    t = torch.rand(B, n_kp * 2)
    vis = torch.ones(B, n_kp, dtype=torch.bool)
    loss = _masked_mse_loss(t, t, vis)
    assert float(loss) < 1e-7, f"Expected ~0 loss for perfect prediction, got {loss}"


def test_masked_mse_loss_all_invisible_returns_zero() -> None:
    """_masked_mse_loss with all-invisible visibility returns 0.0."""
    B, n_kp = 4, 6
    pred = torch.rand(B, n_kp * 2)
    target = torch.rand(B, n_kp * 2)
    vis = torch.zeros(B, n_kp, dtype=torch.bool)
    loss = _masked_mse_loss(pred, target, vis)
    assert float(loss) == 0.0, f"Expected 0.0 loss when all invisible, got {loss}"


def test_masked_mse_loss_ignores_invisible_keypoints() -> None:
    """_masked_mse_loss only accumulates error on visible keypoints."""
    B, n_kp = 2, 6
    # pred differs from target by 1.0 on all keypoints
    pred = torch.zeros(B, n_kp * 2)
    target = torch.ones(B, n_kp * 2)

    # Only first half (3 keypoints) visible
    vis = torch.zeros(B, n_kp, dtype=torch.bool)
    vis[:, :3] = True

    masked_loss = float(_masked_mse_loss(pred, target, vis))

    # MSE on the visible half only: all coords differ by 1.0 → MSE = 1.0
    # n_visible = B * 3_keypoints * 2_coords = 12, sum_sq = 12 * 1.0 = 12, loss = 1.0
    expected = 1.0
    assert abs(masked_loss - expected) < 1e-5, (
        f"Masked loss should be {expected}, got {masked_loss}"
    )

    # Full loss (all visible) should also be 1.0 (all coords differ by 1.0)
    vis_all = torch.ones(B, n_kp, dtype=torch.bool)
    full_loss = float(_masked_mse_loss(pred, target, vis_all))
    assert abs(full_loss - 1.0) < 1e-5, f"Full loss should be 1.0, got {full_loss}"


# ---------------------------------------------------------------------------
# Masked validation metric
# ---------------------------------------------------------------------------


def test_mean_keypoint_error_masked() -> None:
    """_mean_keypoint_error with partial visibility differs from unmasked version."""
    B, n_kp = 4, 6
    pred = torch.zeros(B, n_kp * 2)
    # Target: first half keypoints at (1,1), second half at (0.5, 0.5)
    target = torch.zeros(B, n_kp * 2)
    target[:, :n_kp] = 1.0  # first 3 x-coords = 1
    target[:, n_kp:] = 0.5  # last 3 x-coords = 0.5

    # Only visible: first 3 keypoints
    vis = torch.zeros(B, n_kp, dtype=torch.bool)
    vis[:, :3] = True

    masked_err = _mean_keypoint_error(pred, target, vis)
    unmasked_err = _mean_keypoint_error(pred, target)

    # They should differ because invisible keypoints are excluded from masked
    assert masked_err != unmasked_err, (
        "Masked and unmasked errors should differ for partial visibility"
    )


def test_mean_keypoint_error_backward_compat() -> None:
    """_mean_keypoint_error(pred, target) still works without visibility."""
    pred = torch.zeros(4, 12)
    target = torch.ones(4, 12)
    err = _mean_keypoint_error(pred, target)
    # All coords differ by 1.0, each keypoint has dist=sqrt(2), mean = sqrt(2)
    assert abs(err - math.sqrt(2)) < 1e-5, (
        f"Expected sqrt(2) = {math.sqrt(2):.5f}, got {err:.5f}"
    )


# ---------------------------------------------------------------------------
# Smoke test: train_pose with augmented data
# ---------------------------------------------------------------------------


def test_train_pose_augmented_smoke(coco_keypoint_dir: Path, tmp_path: Path) -> None:
    """train_pose with augmented dataset completes 2 epochs and saves best_model.pth."""
    out_dir = tmp_path / "out"
    best_path = train_pose(
        data_dir=coco_keypoint_dir,
        output_dir=out_dir,
        epochs=2,
        batch_size=2,
        patience=0,
        num_workers=0,
        n_keypoints=_N_KP,
    )
    assert best_path.exists(), f"best_model.pth not found at {best_path}"
    assert best_path.name == "best_model.pth"
