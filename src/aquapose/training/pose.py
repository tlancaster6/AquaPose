"""Pose/keypoint regression training with optional frozen-backbone transfer."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import tv_tensors
from torchvision.transforms import v2

from aquapose.segmentation.model import _UNet

from .common import EarlyStopping, MetricsLogger, make_loader, save_best_and_last
from .datasets import _load_image, stratified_split

# Fixed input size — matches U-Net training pipeline
_INPUT_SIZE = 128

# Module-level augmentation transform applied in KeypointDataset augmented path.
# Geometric transforms (flip + affine) + photometric jitter in a single compose.
_AUGMENT_TRANSFORM = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    ]
)


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

    Each sample is an ``(image_tensor, keypoints_tensor, visibility_mask)``
    tuple.  Keypoint coordinates are normalised to [0, 1] in crop space.
    Keypoints with ``visibility == 0`` (COCO convention) or that fall
    out-of-bounds after augmentation are treated as invisible; their
    coordinates are set to 0.0 so the masked loss can ignore them.

    Args:
        coco_json: Path to a COCO-format JSON with keypoint annotations.
        image_root: Root directory containing source images.
        n_keypoints: Expected number of keypoints per instance.
        input_size: Square image size (pixels) passed to the model.
        augment: Whether to apply :data:`_AUGMENT_TRANSFORM` to each sample.
            When False the clean image is returned with visibility derived
            purely from COCO ``v`` flags.
    """

    def __init__(
        self,
        coco_json: Path,
        image_root: Path,
        n_keypoints: int = 6,
        input_size: int = _INPUT_SIZE,
        augment: bool = False,
    ) -> None:
        self._image_root = Path(image_root)
        self._n_keypoints = n_keypoints
        self._input_size = input_size
        self._augment = augment

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load image and normalised keypoint coordinates with visibility mask.

        Args:
            idx: Index into the image list.

        Returns:
            Tuple of:
            - image_tensor: float32 [3, H, W] in [0, 1]
            - keypoints_tensor: float32 [n_keypoints * 2] normalised to [0, 1].
              Invisible keypoints have coordinates zeroed.
            - visibility_mask: bool [n_keypoints]. True = visible and
              in-bounds.
        """
        img_info = self._images[idx]
        img_id = img_info["id"]
        sz = self._input_size

        # Load and resize image
        image = _load_image(
            self._image_root / img_info["file_name"],
            img_info.get("height", sz),
            img_info.get("width", sz),
        )
        orig_h, orig_w = image.shape[:2]
        image_resized = cv2.resize(image, (sz, sz), interpolation=cv2.INTER_LINEAR)

        # Decode keypoints: map COCO pixel coords into 128x128 pixel space
        ann = self._ann_index.get(img_id)
        kp_pixel = torch.zeros(self._n_keypoints, 2, dtype=torch.float32)
        visibility = torch.zeros(self._n_keypoints, dtype=torch.bool)

        if ann is not None:
            raw_kps = ann["keypoints"]  # [x, y, v, x, y, v, ...]
            for k in range(self._n_keypoints):
                base = k * 3
                if base + 2 < len(raw_kps):
                    x, y, v = raw_kps[base], raw_kps[base + 1], raw_kps[base + 2]
                    if v > 0 and orig_w > 0 and orig_h > 0:
                        x_px = float(x) / orig_w * sz
                        y_px = float(y) / orig_h * sz
                        kp_pixel[k] = torch.tensor([x_px, y_px])
                        visibility[k] = True

        if self._augment:
            # Build uint8 image tensor for v2 transforms (ColorJitter needs uint8)
            img_uint8 = torch.from_numpy(image_resized).permute(2, 0, 1)  # (3, H, W)
            img_tv = tv_tensors.Image(img_uint8)
            kps_tv = tv_tensors.KeyPoints(kp_pixel, canvas_size=(sz, sz))

            vis_aug = visibility
            kp_out = kp_pixel.clone()

            for _ in range(10):
                img_aug, kps_aug = _AUGMENT_TRANSFORM(img_tv, kps_tv)
                x_aug = kps_aug[:, 0]
                y_aug = kps_aug[:, 1]
                oob = (x_aug < 0) | (x_aug >= sz) | (y_aug < 0) | (y_aug >= sz)
                vis_aug = visibility & ~oob
                kp_out = torch.as_tensor(kps_aug).clone()
                if int(vis_aug.sum()) >= 3:
                    break

            # Zero out invisible coordinates, normalize to [0, 1]
            kp_out[~vis_aug] = 0.0
            kp_flat = (kp_out / sz).view(-1)
            image_tensor = img_aug.float() / 255.0  # type: ignore[union-attr]

        else:
            # Clean path: OOB check from COCO coordinates, no transform applied
            x_coords = kp_pixel[:, 0]
            y_coords = kp_pixel[:, 1]
            oob = (x_coords < 0) | (x_coords >= sz) | (y_coords < 0) | (y_coords >= sz)
            vis_aug = visibility & ~oob
            kp_out = kp_pixel.clone()
            kp_out[~vis_aug] = 0.0
            kp_flat = (kp_out / sz).view(-1)
            image_tensor = (
                torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
            )

        return image_tensor, kp_flat, vis_aug


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


def _masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """MSE loss computed only on visible keypoints.

    Args:
        pred: (B, n_keypoints * 2) predicted coordinates in [0, 1].
        target: (B, n_keypoints * 2) ground-truth coordinates in [0, 1].
        visibility: (B, n_keypoints) bool mask, True = visible.

    Returns:
        Scalar loss tensor.
    """
    vis_exp = visibility.float().repeat_interleave(2, dim=1)  # (B, n_kp*2)
    diff_sq = (pred - target) ** 2
    n_visible = vis_exp.sum().clamp(min=1.0)
    return (diff_sq * vis_exp).sum() / n_visible


def _mean_keypoint_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor | None = None,
) -> float:
    """Compute mean Euclidean distance between predicted and target keypoints.

    When ``visibility`` is provided, only visible keypoints contribute to the
    mean.  When ``visibility`` is None, all keypoints are included (backward
    compatible with existing callers).

    Args:
        pred: Predicted keypoints (B, n_keypoints * 2) in [0, 1].
        target: Ground-truth keypoints (B, n_keypoints * 2) in [0, 1].
        visibility: Optional (B, n_keypoints) bool mask.  True = visible.
            When provided, distances are averaged over visible keypoints only.

    Returns:
        Mean Euclidean distance in normalised coordinates.
    """
    # Reshape to (B, n_keypoints, 2)
    n = pred.shape[1] // 2
    p = pred.view(-1, n, 2)
    t = target.view(-1, n, 2)
    dist = torch.sqrt(((p - t) ** 2).sum(dim=-1))  # (B, n_keypoints)

    if visibility is not None:
        v = visibility.float()  # (B, n_keypoints)
        n_visible = v.sum().clamp(min=1.0)
        return float((dist * v).sum() / n_visible)

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
    with masked MSE loss between predicted and ground-truth keypoint
    coordinates.  Only visible keypoints contribute to the loss, preventing
    the model from learning to predict (0, 0) for missing annotations.

    Training data is doubled via a ``ConcatDataset`` of clean and augmented
    subsets of the training split.  Validation uses clean images only.

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

    train_dataset: Dataset  # type: ignore[type-arg]
    val_dataset: Dataset  # type: ignore[type-arg]

    if train_json.exists() and val_json.exists():
        # Pre-split: combine clean + augmented for training, clean for val
        train_clean = KeypointDataset(
            train_json, data_dir, n_keypoints=n_keypoints, augment=False
        )
        train_aug = KeypointDataset(
            train_json, data_dir, n_keypoints=n_keypoints, augment=True
        )
        train_dataset = ConcatDataset([train_clean, train_aug])
        val_dataset = KeypointDataset(
            val_json, data_dir, n_keypoints=n_keypoints, augment=False
        )
    else:
        # Single annotations.json: split once on clean dataset, build ConcatDataset
        train_clean = KeypointDataset(
            annotations_json, data_dir, n_keypoints=n_keypoints, augment=False
        )
        train_aug = KeypointDataset(
            annotations_json, data_dir, n_keypoints=n_keypoints, augment=True
        )
        train_indices, val_indices = stratified_split(
            train_clean,
            val_fraction=val_split,
            seed=42,
        )
        # 2x effective epoch size: 1 clean + 1 augmented per sample
        train_dataset = ConcatDataset(
            [Subset(train_clean, train_indices), Subset(train_aug, train_indices)]
        )
        val_dataset = Subset(train_clean, val_indices)

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

        for images, targets, visibility in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            visibility = visibility.to(device)

            pred = model(images)
            loss = _masked_mse_loss(pred, targets, visibility)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += float(loss.detach())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation: masked mean keypoint error (Euclidean distance in [0,1] coords)
        model.eval()
        val_preds: list[torch.Tensor] = []
        val_targets: list[torch.Tensor] = []
        val_vis: list[torch.Tensor] = []
        with torch.no_grad():
            for images, targets, visibility in val_loader:
                images = images.to(device)
                pred = model(images)
                val_preds.append(pred.cpu())
                val_targets.append(targets.cpu())
                val_vis.append(visibility.cpu())

        if val_preds:
            all_pred = torch.cat(val_preds, dim=0)
            all_tgt = torch.cat(val_targets, dim=0)
            all_vis = torch.cat(val_vis, dim=0)
            val_error = _mean_keypoint_error(all_pred, all_tgt, all_vis)
        else:
            val_error = float("nan")

        logger.log(epoch + 1, train_loss=avg_loss, val_error=val_error)

        _, best_error = save_best_and_last(
            model, output_dir, val_error, best_error, mode="min"
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
