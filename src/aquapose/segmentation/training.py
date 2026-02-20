"""Mask R-CNN training and evaluation for fish segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset, random_split

from .dataset import CropDataset
from .model import MaskRCNNSegmentor


def _collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[tuple[torch.Tensor, ...], tuple[dict[str, torch.Tensor], ...]]:
    """Custom collate for Mask R-CNN (expects list of tuples, not batched).

    Args:
        batch: List of (image, target) tuples from dataset.

    Returns:
        Tuple of (images_tuple, targets_tuple).
    """
    return tuple(zip(*batch, strict=True))  # type: ignore[return-value]


def train(
    coco_json: Path,
    image_root: Path,
    output_dir: Path,
    *,
    epochs: int = 40,
    batch_size: int = 4,
    lr: float = 0.005,
    val_split: float = 0.2,
    crop_size: int = 256,
    device: str | None = None,
) -> Path:
    """Train Mask R-CNN on COCO-format fish annotations.

    Fine-tunes from ImageNet-pretrained ResNet-50 backbone. Saves best
    model (by validation IoU) and final model to output_dir.

    Args:
        coco_json: Path to COCO-format annotation JSON.
        image_root: Root directory containing source images.
        output_dir: Directory to save model checkpoints.
        epochs: Number of training epochs.
        batch_size: Images per training batch.
        lr: Initial learning rate for SGD.
        val_split: Fraction of data to use for validation.
        crop_size: Target spatial dimension for training crops.
        device: Torch device string. Auto-detects if None.

    Returns:
        Path to the best model checkpoint file.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    full_dataset = CropDataset(coco_json, image_root, crop_size, augment=True)
    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    # Validation subset should not augment -- wrap with augment=False dataset
    val_dataset = CropDataset(coco_json, image_root, crop_size, augment=False)
    val_indices = val_subset.indices
    val_subset_no_aug = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_subset_no_aug,
        batch_size=1,
        shuffle=False,
        collate_fn=_collate_fn,
    )

    # Build model
    segmentor = MaskRCNNSegmentor(num_classes=2)
    model = segmentor.get_model()
    model.to(device)

    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    step_size = max(1, int(epochs * 0.8))
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)

    best_iou = 0.0
    best_model_path = output_dir / "best_model.pth"
    final_model_path = output_dir / "final_model.pth"

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, targets in train_loader:
            images_dev = [img.to(device) for img in images]
            targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Skip batches with no annotations (all negative frames)
            if all(t["boxes"].shape[0] == 0 for t in targets_dev):
                continue

            loss_dict = model(images_dev, targets_dev)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()  # type: ignore[union-attr]
            optimizer.step()

            epoch_loss += float(losses)  # type: ignore[arg-type]
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate every 5 epochs
        val_iou = 0.0
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_iou = _compute_val_iou(model, val_loader, device)
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"train_loss: {avg_loss:.4f} - val_iou: {val_iou:.4f}"
            )

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), best_model_path)
        else:
            print(f"Epoch {epoch + 1}/{epochs} - train_loss: {avg_loss:.4f}")

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
    """Compute mean mask IoU on validation set.

    Args:
        model: Mask R-CNN model in eval mode.
        val_loader: Validation data loader.
        device: Torch device string.

    Returns:
        Mean mask IoU across all validation images.
    """
    model.eval()
    ious: list[float] = []

    with torch.no_grad():
        for images, targets in val_loader:
            images_dev = [img.to(device) for img in images]
            outputs = model(images_dev)

            for output, target in zip(outputs, targets, strict=True):
                gt_masks = target["masks"].numpy()
                if gt_masks.shape[0] == 0:
                    continue

                pred_masks = output["masks"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()

                if pred_masks.shape[0] == 0:
                    ious.append(0.0)
                    continue

                # Take top-K predictions matching GT count
                k = min(gt_masks.shape[0], pred_masks.shape[0])
                top_k_idx = np.argsort(pred_scores)[::-1][:k]
                pred_binary = (pred_masks[top_k_idx, 0] > 0.5).astype(np.uint8)

                # Compute IoU for each GT-pred pair (greedy matching)
                for gi in range(gt_masks.shape[0]):
                    if gi < k:
                        intersection = (gt_masks[gi] & pred_binary[gi]).sum()
                        union = (gt_masks[gi] | pred_binary[gi]).sum()
                        iou = float(intersection) / max(float(union), 1.0)
                        ious.append(iou)
                    else:
                        ious.append(0.0)

    return float(np.mean(ious)) if ious else 0.0


def evaluate(
    model_path: Path,
    coco_json: Path,
    image_root: Path,
    *,
    crop_size: int = 256,
    device: str | None = None,
) -> dict[str, float | list[float] | int]:
    """Evaluate a trained model on a dataset.

    Args:
        model_path: Path to saved model state_dict.
        coco_json: Path to COCO-format annotation JSON.
        image_root: Root directory containing source images.
        crop_size: Target spatial dimension for evaluation crops.
        device: Torch device string. Auto-detects if None.

    Returns:
        Dict with 'mean_iou' (float), 'per_image_iou' (list[float]),
        and 'num_images' (int).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    segmentor = MaskRCNNSegmentor(
        num_classes=2, weights_path=model_path, confidence_threshold=0.1
    )
    model = segmentor.get_model()
    model.to(device)
    model.eval()

    dataset = CropDataset(coco_json, image_root, crop_size, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_fn)

    per_image_iou: list[float] = []

    with torch.no_grad():
        for images, targets in loader:
            images_dev = [img.to(device) for img in images]
            outputs = model(images_dev)

            for output, target in zip(outputs, targets, strict=True):
                gt_masks = target["masks"].numpy()
                if gt_masks.shape[0] == 0:
                    per_image_iou.append(1.0)  # No GT, no predictions expected
                    continue

                pred_masks = output["masks"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()

                if pred_masks.shape[0] == 0:
                    per_image_iou.append(0.0)
                    continue

                # Match predictions to GT
                k = min(gt_masks.shape[0], pred_masks.shape[0])
                top_k_idx = np.argsort(pred_scores)[::-1][:k]
                pred_binary = (pred_masks[top_k_idx, 0] > 0.5).astype(np.uint8)

                img_ious = []
                for gi in range(gt_masks.shape[0]):
                    if gi < k:
                        intersection = (gt_masks[gi] & pred_binary[gi]).sum()
                        union = (gt_masks[gi] | pred_binary[gi]).sum()
                        iou = float(intersection) / max(float(union), 1.0)
                        img_ious.append(iou)
                    else:
                        img_ious.append(0.0)

                per_image_iou.append(float(np.mean(img_ious)))

    mean_iou = float(np.mean(per_image_iou)) if per_image_iou else 0.0

    return {
        "mean_iou": mean_iou,
        "per_image_iou": per_image_iou,
        "num_images": len(per_image_iou),
    }
