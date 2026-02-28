"""Unit tests for the pose regression model and training utilities."""

from __future__ import annotations

from pathlib import Path

import torch

from aquapose.training.pose import (
    _freeze_encoder,
    _load_backbone_weights,
    _mean_keypoint_error,
    _PoseModel,
)

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
# Metric utility
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
