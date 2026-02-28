"""Pose/keypoint regression training with optional frozen-backbone transfer."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset

from aquapose.segmentation.model import _UNet

from .common import EarlyStopping, MetricsLogger, make_loader, save_best_and_last
from .datasets import stratified_split

# Fixed input size — matches U-Net training pipeline
_INPUT_SIZE = 128


class _PoseModel(nn.Module):
    """U-Net encoder + regression head for keypoint coordinate prediction.

    The encoder is taken from :class:`~aquapose.segmentation.model._UNet`.
    The decoder is discarded and replaced with a lightweight regression head
    that outputs flattened (x, y) coordinates normalised to [0, 1] in crop
    space.

    Architecture:
        encoder (MobileNetV3-Small, enc0-enc4) -> AdaptiveAvgPool2d(1)
        -> flatten -> Linear(96, 256) -> ReLU -> Linear(256, n_keypoints*2)
        -> Sigmoid

    Args:
        n_keypoints: Number of anatomical keypoints.  Output has shape
            ``(B, n_keypoints * 2)``.
        pretrained: Whether to load ImageNet weights for the encoder when no
            ``backbone_weights`` are supplied.
    """

    # Channel width at the bottleneck of MobileNetV3-Small (enc4 output)
    _ENC4_CHANNELS: int = 96

    def __init__(self, n_keypoints: int = 6, pretrained: bool = True) -> None:
        super().__init__()
        # Borrow only the encoder from _UNet — discard decoder and output conv
        unet = _UNet(pretrained=pretrained)
        self.enc0 = unet.enc0
        self.enc1 = unet.enc1
        self.enc2 = unet.enc2
        self.enc3 = unet.enc3
        self.enc4 = unet.enc4

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self._ENC4_CHANNELS, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_keypoints * 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and regression head.

        Args:
            x: Input tensor of shape (B, 3, H, W) in [0, 1].

        Returns:
            Keypoint coordinates of shape (B, n_keypoints * 2), normalised
            to [0, 1] in crop space.  Even indices are x, odd indices are y.
        """
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return self.head(e4)

    def encoder_parameters(self) -> list[nn.Parameter]:
        """Return all encoder parameters.

        Returns:
            List of parameters belonging to enc0-enc4.
        """
        params: list[nn.Parameter] = []
        for enc in (self.enc0, self.enc1, self.enc2, self.enc3, self.enc4):
            params.extend(enc.parameters())
        return params

    def head_parameters(self) -> list[nn.Parameter]:
        """Return regression-head parameters.

        Returns:
            List of parameters belonging to the regression head.
        """
        return list(self.head.parameters())


class KeypointDataset(Dataset):  # type: ignore[type-arg]
    """COCO-format keypoint dataset for pose regression training.

    Each sample is an ``(image_tensor, keypoints_tensor)`` pair.  Keypoint
    coordinates are normalised to [0, 1] in crop space.  Keypoints with
    ``visibility == 0`` are treated as invisible; their coordinates are set
    to 0.0 so the network has a well-defined learning target even when
    annotations are partial.

    Args:
        coco_json: Path to a COCO-format JSON with keypoint annotations.
        image_root: Root directory containing source images.
        n_keypoints: Expected number of keypoints per instance.
        input_size: Square image size (pixels) passed to the model.
    """

    def __init__(
        self,
        coco_json: Path,
        image_root: Path,
        n_keypoints: int = 6,
        input_size: int = _INPUT_SIZE,
    ) -> None:
        self._image_root = Path(image_root)
        self._n_keypoints = n_keypoints
        self._input_size = input_size

        with open(coco_json) as f:
            coco = json.load(f)

        self._images: list[dict] = coco["images"]  # type: ignore[assignment]

        # Build annotation index: image_id -> first annotation with keypoints
        self._ann_index: dict[int, dict] = {}  # type: ignore[assignment]
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self._ann_index and ann.get("keypoints"):
                self._ann_index[img_id] = ann

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load image and normalised keypoint coordinates.

        Args:
            idx: Index into the image list.

        Returns:
            Tuple of:
            - image_tensor: float32 [3, H, W] in [0, 1]
            - keypoints_tensor: float32 [n_keypoints * 2] normalised to [0, 1]
        """
        img_info = self._images[idx]
        img_id = img_info["id"]
        sz = self._input_size

        # Load and resize image
        img_path = self._image_root / img_info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            h = img_info.get("height", sz)
            w = img_info.get("width", sz)
            image = np.zeros((h, w, 3), dtype=np.uint8)
        orig_h, orig_w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (sz, sz), interpolation=cv2.INTER_LINEAR)

        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        # Decode keypoints from annotation
        ann = self._ann_index.get(img_id)
        kp_flat = np.zeros(self._n_keypoints * 2, dtype=np.float32)

        if ann is not None:
            raw_kps = ann["keypoints"]  # [x, y, v, x, y, v, ...]
            for k in range(self._n_keypoints):
                base = k * 3
                if base + 2 < len(raw_kps):
                    x, y, v = raw_kps[base], raw_kps[base + 1], raw_kps[base + 2]
                    if v > 0 and orig_w > 0 and orig_h > 0:
                        kp_flat[k * 2] = float(x) / orig_w
                        kp_flat[k * 2 + 1] = float(y) / orig_h

        keypoints_tensor = torch.from_numpy(kp_flat)
        return image_tensor, keypoints_tensor


def _load_backbone_weights(model: _PoseModel, weights_path: Path) -> None:
    """Load encoder weights from a U-Net checkpoint into the pose model.

    Extracts keys starting with ``enc`` from the saved state-dict and loads
    them into the matching encoder submodules.

    Args:
        model: The ``_PoseModel`` to load weights into.
        weights_path: Path to a saved state-dict (e.g. from ``train_unet``).
    """
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    encoder_state: dict[str, torch.Tensor] = {
        k: v for k, v in state_dict.items() if k.startswith("enc")
    }
    # Load with strict=False so missing decoder keys don't raise
    model.load_state_dict(encoder_state, strict=False)


def _freeze_encoder(model: _PoseModel) -> None:
    """Freeze all encoder parameters (requires_grad=False).

    Args:
        model: The ``_PoseModel`` whose encoder will be frozen.
    """
    for param in model.encoder_parameters():
        param.requires_grad = False


def _mean_keypoint_error(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Compute mean Euclidean distance between predicted and target keypoints.

    Args:
        pred: Predicted keypoints (B, n_keypoints * 2) in [0, 1].
        target: Ground-truth keypoints (B, n_keypoints * 2) in [0, 1].

    Returns:
        Mean Euclidean distance in normalised coordinates.
    """
    # Reshape to (B, n_keypoints, 2)
    n = pred.shape[1] // 2
    p = pred.view(-1, n, 2)
    t = target.view(-1, n, 2)
    dist = torch.sqrt(((p - t) ** 2).sum(dim=-1))  # (B, n_keypoints)
    return float(dist.mean().item())


def train_pose(
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
    backbone_weights: Path | None = None,
    unfreeze: bool = False,
    n_keypoints: int = 6,
) -> Path:
    """Train a pose regression model on COCO-format keypoint annotations.

    Builds a ``_PoseModel`` (U-Net encoder + regression head) and trains it
    with MSE loss between predicted and ground-truth keypoint coordinates.

    Transfer learning behaviour:

    - ``backbone_weights`` only: encoder weights loaded, encoder frozen.
    - ``backbone_weights`` + ``unfreeze``: encoder weights loaded, encoder
      trainable with 1/10th the head learning rate (differential LR).
    - No ``backbone_weights``: trains from ImageNet-pretrained encoder
      (encoder fully trainable, differential LR applied).

    Looks for ``annotations.json`` in ``data_dir``.  If ``train.json`` and
    ``val.json`` are present they are used directly.

    Args:
        data_dir: Directory with COCO keypoint JSON (``annotations.json``)
            and images.
        output_dir: Directory for model checkpoints and ``metrics.csv``.
        epochs: Number of training epochs.
        batch_size: Images per batch.
        lr: Base learning rate for the regression head.  Encoder gets
            ``lr * 0.1`` when unfrozen.
        val_split: Validation fraction when splitting ``annotations.json``.
        patience: Early-stopping patience (epochs without improvement in
            validation keypoint error).  0 disables early stopping.
        num_workers: DataLoader workers.
        device: Torch device string.  Auto-detected if None.
        backbone_weights: Optional path to a U-Net state-dict from which
            encoder weights (keys starting with ``enc``) are loaded.
        unfreeze: When True (only meaningful with ``backbone_weights``),
            the encoder is not frozen — fine-tuning with differential LR.
        n_keypoints: Number of anatomical keypoints to regress.

    Returns:
        Path to the best model checkpoint file.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve dataset JSON paths
    train_json = data_dir / "train.json"
    val_json = data_dir / "val.json"
    annotations_json = data_dir / "annotations.json"

    if train_json.exists() and val_json.exists():
        train_dataset: KeypointDataset | Subset[tuple[torch.Tensor, torch.Tensor]] = (
            KeypointDataset(train_json, data_dir, n_keypoints=n_keypoints)
        )
        val_dataset: KeypointDataset | Subset[tuple[torch.Tensor, torch.Tensor]] = (
            KeypointDataset(val_json, data_dir, n_keypoints=n_keypoints)
        )
    else:
        full_dataset = KeypointDataset(
            annotations_json, data_dir, n_keypoints=n_keypoints
        )
        train_indices, val_indices = stratified_split(
            full_dataset,
            val_fraction=val_split,
            seed=42,
        )
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset_base = KeypointDataset(
            annotations_json, data_dir, n_keypoints=n_keypoints
        )
        val_dataset = Subset(val_dataset_base, val_indices)

    train_loader = make_loader(
        train_dataset, batch_size, shuffle=True, device=device, num_workers=num_workers
    )
    val_loader = make_loader(
        val_dataset, batch_size, shuffle=False, device=device, num_workers=num_workers
    )

    # Build model
    pretrained_encoder = backbone_weights is None
    model = _PoseModel(n_keypoints=n_keypoints, pretrained=pretrained_encoder)

    if backbone_weights is not None:
        _load_backbone_weights(model, backbone_weights)
        if not unfreeze:
            _freeze_encoder(model)

    model.to(device)

    # Set up optimizer: differential LR when encoder is trainable
    encoder_params = [p for p in model.encoder_parameters() if p.requires_grad]
    head_params = list(model.head_parameters())

    if encoder_params:
        # Encoder trainable — use differential LR
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": lr * 0.1},
                {"params": head_params, "lr": lr},
            ],
            weight_decay=1e-4,
        )
    else:
        # Encoder frozen — only optimise head
        optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=1e-4)

    early_stopper = EarlyStopping(patience=patience, mode="min")
    logger = MetricsLogger(output_dir, fields=["train_loss", "val_error"])
    logger.set_total_epochs(epochs)

    best_error = float("inf")
    best_model_path = output_dir / "best_model.pth"

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            pred = model(images)
            loss = torch.nn.functional.mse_loss(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += float(loss.detach())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation: mean keypoint error (Euclidean distance in [0,1] coords)
        model.eval()
        val_preds: list[torch.Tensor] = []
        val_targets: list[torch.Tensor] = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                pred = model(images)
                val_preds.append(pred.cpu())
                val_targets.append(targets.cpu())

        if val_preds:
            all_pred = torch.cat(val_preds, dim=0)
            all_tgt = torch.cat(val_targets, dim=0)
            val_error = _mean_keypoint_error(all_pred, all_tgt)
        else:
            val_error = float("nan")

        logger.log(epoch + 1, train_loss=avg_loss, val_error=val_error)

        _, best_error = save_best_and_last(
            model, output_dir, val_error, best_error, metric_name="val_error_loss"
        )

        if early_stopper.step(val_error):
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(no improvement for {patience} epochs). "
                f"Best val error: {early_stopper.best:.4f}"
            )
            break

    if not best_model_path.exists():
        torch.save(model.state_dict(), best_model_path)

    return best_model_path
