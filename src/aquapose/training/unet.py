"""U-Net training loop for fish binary segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset

from aquapose.segmentation.model import UNET_INPUT_SIZE, UNetSegmentor

from .common import EarlyStopping, MetricsLogger, make_loader, save_best_and_last
from .datasets import BinaryMaskDataset, stratified_split


def _dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute soft Dice loss.

    Args:
        pred: Predicted probabilities (B, 1, H, W) in [0, 1].
        target: Ground truth binary mask (B, 1, H, W) in {0, 1}.

    Returns:
        Scalar Dice loss (1 - Dice coefficient).
    """
    smooth = 1.0
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )


def _bce_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined BCE + Dice loss (equally weighted).

    Args:
        pred: Predicted probabilities (B, 1, H, W) in [0, 1].
        target: Ground truth binary mask (B, 1, H, W) in {0, 1}.

    Returns:
        Scalar combined loss.
    """
    bce = F.binary_cross_entropy(pred, target)
    dice = _dice_loss(pred, target)
    return bce + dice


def _compute_val_iou(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,  # type: ignore[type-arg]
    device: str,
) -> float:
    """Compute mean binary IoU on validation set.

    Args:
        model: U-Net model in eval mode.
        val_loader: Validation data loader yielding (images, masks).
        device: Torch device string.

    Returns:
        Mean binary mask IoU across all validation images.
    """
    model.eval()
    ious: list[float] = []

    with torch.no_grad():
        for images, gt_masks in val_loader:
            images = images.to(device)
            pred = model(images)  # (B, 1, 128, 128)

            pred_binary = (pred > 0.5).float().cpu()
            gt_masks_cpu = gt_masks.float()

            for i in range(pred_binary.shape[0]):
                p = pred_binary[i, 0]
                g = gt_masks_cpu[i, 0]

                intersection = (p * g).sum()
                union = ((p + g) > 0).float().sum()

                if union == 0:
                    # Both empty — perfect match
                    ious.append(1.0)
                else:
                    ious.append(float(intersection / union))

    return float(np.mean(ious)) if ious else 0.0


def train_unet(
    data_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    val_split: float = 0.2,
    patience: int = 20,
    num_workers: int = 4,
    device: str | None = None,
    input_size: tuple[int, int] = UNET_INPUT_SIZE,
) -> Path:
    """Train U-Net on COCO-format fish crop annotations.

    Fine-tunes from ImageNet-pretrained MobileNetV3-Small encoder.
    Saves best model (by validation IoU) and last model to output_dir.
    All crops are resized to 128x128 for training.

    Looks for ``annotations.json`` in ``data_dir``. If ``train.json`` and
    ``val.json`` are present in ``data_dir``, they are used directly as
    pre-split datasets; otherwise ``annotations.json`` is split with
    per-camera stratification.

    Args:
        data_dir: Directory containing COCO JSON (``annotations.json``) and
            images. Pre-split ``train.json`` / ``val.json`` are used when present.
        output_dir: Directory for model checkpoints and metrics CSV.
        epochs: Number of training epochs.
        batch_size: Images per training batch.
        lr: Initial learning rate for AdamW (decoder). Encoder gets lr * 0.1.
        val_split: Fraction of data for validation when using stratified split.
        patience: Early-stopping patience (epochs without val IoU improvement).
            Set to 0 to disable early stopping.
        num_workers: DataLoader worker processes.
        device: Torch device string. Auto-detected if None.

    Returns:
        Path to the best model checkpoint file.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve dataset paths
    train_json = data_dir / "train.json"
    val_json = data_dir / "val.json"
    annotations_json = data_dir / "annotations.json"

    # Build train/val datasets
    if train_json.exists() and val_json.exists():
        train_dataset: (
            BinaryMaskDataset
            | Subset[  # type: ignore[type-arg]
                tuple[torch.Tensor, torch.Tensor]
            ]
        ) = BinaryMaskDataset(train_json, data_dir, augment=True)
        val_dataset: (
            BinaryMaskDataset
            | Subset[  # type: ignore[type-arg]
                tuple[torch.Tensor, torch.Tensor]
            ]
        ) = BinaryMaskDataset(val_json, data_dir, augment=False)
    else:
        full_dataset = BinaryMaskDataset(annotations_json, data_dir, augment=True)
        train_indices, val_indices = stratified_split(
            full_dataset, val_fraction=val_split, seed=42
        )
        train_dataset = Subset(full_dataset, train_indices)

        val_dataset_base = BinaryMaskDataset(annotations_json, data_dir, augment=False)
        val_dataset = Subset(val_dataset_base, val_indices)

    train_loader = make_loader(
        train_dataset, batch_size, shuffle=True, device=device, num_workers=num_workers
    )
    val_loader = make_loader(
        val_dataset, batch_size, shuffle=False, device=device, num_workers=num_workers
    )

    # Build model with differential LR: encoder gets lr * 0.1, decoder gets lr
    segmentor = UNetSegmentor()
    model = segmentor.get_model()
    model.to(device)

    encoder_params: list[torch.Tensor] = []
    decoder_params: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("enc"):
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": lr * 0.1},
            {"params": decoder_params, "lr": lr},
        ],
        weight_decay=1e-3,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    early_stopper = EarlyStopping(patience=patience, mode="max")
    logger = MetricsLogger(
        output_dir, fields=["train_loss", "val_iou", "lr_enc", "lr_dec"]
    )
    logger.set_total_epochs(epochs)

    best_iou = 0.0
    best_model_path = output_dir / "best_model.pth"

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            pred = model(images)
            loss = _bce_dice_loss(pred, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += float(loss.detach())
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate every epoch
        val_iou = _compute_val_iou(model, val_loader, device)
        enc_lr = float(optimizer.param_groups[0]["lr"])
        dec_lr = float(optimizer.param_groups[1]["lr"])

        logger.log(
            epoch + 1,
            train_loss=avg_loss,
            val_iou=val_iou,
            lr_enc=enc_lr,
            lr_dec=dec_lr,
        )

        # Save best/last checkpoints
        _, best_iou = save_best_and_last(
            model, output_dir, val_iou, best_iou, metric_name="val_iou"
        )

        # Early stopping
        if early_stopper.step(val_iou):
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(no val IoU improvement for {patience} epochs). "
                f"Best val IoU: {early_stopper.best:.4f}"
            )
            break

    # Ensure best_model.pth exists even if no improvement was ever recorded
    if not best_model_path.exists():
        torch.save(model.state_dict(), best_model_path)

    return best_model_path
