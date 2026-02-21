"""U-Net training and evaluation for fish binary segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from .dataset import BinaryMaskDataset, stratified_split
from .model import UNetSegmentor


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


def train(
    coco_json: Path,
    image_root: Path,
    output_dir: Path,
    *,
    train_json: Path | None = None,
    val_json: Path | None = None,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    val_split: float = 0.2,
    patience: int = 20,
    num_workers: int = 4,
    device: str | None = None,
) -> Path:
    """Train U-Net on COCO-format fish crop annotations.

    Fine-tunes from ImageNet-pretrained MobileNetV3-Small encoder.
    Saves best model (by validation IoU) and final model to output_dir.
    All crops are resized to 128x128 for training.

    When ``train_json`` and ``val_json`` are both provided, they are used
    directly as pre-split datasets. Otherwise, ``coco_json`` is split
    with :func:`~.dataset.stratified_split` using per-camera stratification.

    Args:
        coco_json: Path to combined COCO-format annotation JSON.
        image_root: Root directory containing source images.
        output_dir: Directory to save model checkpoints.
        train_json: Optional path to pre-split training COCO JSON.
        val_json: Optional path to pre-split validation COCO JSON.
        epochs: Number of training epochs.
        batch_size: Images per training batch.
        lr: Initial learning rate for AdamW.
        val_split: Fraction of data for validation when using stratified split.
        patience: Early-stopping patience (epochs without val IoU improvement).
        num_workers: DataLoader worker processes for parallel image loading.
        device: Torch device string. Auto-detects if None.

    Returns:
        Path to the best model checkpoint file.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build train/val datasets
    if train_json is not None and val_json is not None:
        train_dataset: BinaryMaskDataset | Subset[tuple[torch.Tensor, torch.Tensor]] = (
            BinaryMaskDataset(train_json, image_root, augment=True)
        )  # type: ignore[assignment]
        val_dataset_no_aug: (
            BinaryMaskDataset | Subset[tuple[torch.Tensor, torch.Tensor]]
        ) = BinaryMaskDataset(val_json, image_root, augment=False)  # type: ignore[assignment]
    else:
        full_dataset = BinaryMaskDataset(coco_json, image_root, augment=True)
        train_indices, val_indices = stratified_split(
            full_dataset, val_fraction=val_split, seed=42
        )
        train_dataset = Subset(full_dataset, train_indices)

        val_dataset_base = BinaryMaskDataset(coco_json, image_root, augment=False)
        val_dataset_no_aug = Subset(val_dataset_base, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset_no_aug,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # Build model
    segmentor = UNetSegmentor()
    model = segmentor.get_model()
    model.to(device)

    # Differential LR: pretrained encoder gets 1/10th the decoder LR
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

    best_iou = 0.0
    epochs_without_improvement = 0
    best_model_path = output_dir / "best_model.pth"
    final_model_path = output_dir / "final_model.pth"

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
        enc_lr = optimizer.param_groups[0]["lr"]
        dec_lr = optimizer.param_groups[1]["lr"]
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train_loss: {avg_loss:.4f} - val_iou: {val_iou:.4f} - "
            f"lr_enc: {enc_lr:.6f} - lr_dec: {dec_lr:.6f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        # Early stopping
        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(no val IoU improvement for {patience} epochs). "
                f"Best val IoU: {best_iou:.4f}"
            )
            break

    # Save final model
    torch.save(model.state_dict(), final_model_path)

    # If no validation was better, save current as best too
    if not best_model_path.exists():
        torch.save(model.state_dict(), best_model_path)

    return best_model_path


def _compute_val_iou(
    model: torch.nn.Module,
    val_loader: DataLoader,  # type: ignore[type-arg]
    device: str,
) -> float:
    """Compute mean binary IoU on validation set.

    Args:
        model: U-Net model.
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


def evaluate(
    model_path: Path,
    coco_json: Path,
    image_root: Path,
    *,
    device: str | None = None,
) -> dict[str, float | list[float] | int]:
    """Evaluate a trained U-Net model on a dataset.

    Args:
        model_path: Path to saved model state_dict.
        coco_json: Path to COCO-format annotation JSON.
        image_root: Root directory containing source images.
        device: Torch device string. Auto-detects if None.

    Returns:
        Dict with 'mean_iou' (float), 'per_image_iou' (list[float]),
        and 'num_images' (int).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    segmentor = UNetSegmentor(weights_path=model_path, confidence_threshold=0.5)
    model = segmentor.get_model()
    model.to(device)
    model.eval()

    dataset = BinaryMaskDataset(coco_json, image_root, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    per_image_iou: list[float] = []

    with torch.no_grad():
        for images, gt_masks in loader:
            images = images.to(device)
            pred = model(images)  # (1, 1, 128, 128)

            pred_binary = (pred > 0.5).float().cpu()
            gt_float = gt_masks.float()

            p = pred_binary[0, 0]
            g = gt_float[0, 0]

            intersection = (p * g).sum()
            union = ((p + g) > 0).float().sum()

            if union == 0:
                per_image_iou.append(1.0)
            else:
                per_image_iou.append(float(intersection / union))

    mean_iou = float(np.mean(per_image_iou)) if per_image_iou else 0.0

    return {
        "mean_iou": mean_iou,
        "per_image_iou": per_image_iou,
        "num_images": len(per_image_iou),
    }
