"""ReID projection head training with SubCenterArcFace metric learning.

Implements backbone feature caching, projection head training, and AUC-gated
evaluation for fine-tuning MegaDescriptor-T embeddings to discriminate
individual fish (especially females) in a multi-camera aquarium rig.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.samplers import MPerClassSampler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from aquapose.core.reid.embedder import FishEmbedder

logger = logging.getLogger(__name__)

# Default female fish IDs for the YH project.
DEFAULT_FEMALE_IDS: frozenset[int] = frozenset({0, 1, 2, 3, 4, 8})


@dataclass(frozen=True)
class ReidTrainingConfig:
    """Configuration for ReID projection head training.

    Attributes:
        reid_crops_dir: Path to miner output with group_NNN directories.
        output_dir: Where to save model checkpoint and feature cache.
        embedding_dim: Output dimension of the projection head.
        hidden_dim: Hidden layer dimension of the projection head.
        backbone_dim: Input dimension from the backbone (MegaDescriptor-T = 768).
        num_classes: Number of fish identities.
        sub_centers: Number of sub-centers per identity in ArcFace.
        arcface_margin: Angular margin in degrees for ArcFace loss.
        arcface_scale: Scale factor for ArcFace loss.
        epochs: Maximum training epochs.
        patience: Early stopping patience (epochs without AUC improvement).
            Set to 0 to disable early stopping.
        lr_head: Learning rate for projection head optimizer.
        lr_loss: Learning rate for ArcFace loss optimizer.
        warmup_epochs: Linear warmup epochs (ramps LR from 0 to target).
        samples_per_class: Samples per class per batch for MPerClassSampler.
        val_fraction: Fraction of groups to hold out for validation.
        device: PyTorch device string.
        model_name: Backbone model name for FishEmbedder.
        batch_size: Batch size for backbone embedding during cache build.
        crop_size: Crop size for backbone embedding during cache build.
    """

    reid_crops_dir: Path = field(default_factory=lambda: Path("."))
    output_dir: Path = field(default_factory=lambda: Path("."))
    embedding_dim: int = 128
    hidden_dim: int = 256
    backbone_dim: int = 768
    num_classes: int = 9
    sub_centers: int = 3
    arcface_margin: float = 28.6
    arcface_scale: float = 64
    epochs: int = 50
    patience: int = 0
    lr_head: float = 3e-4
    lr_loss: float = 0.01
    warmup_epochs: int = 5
    samples_per_class: int = 16
    val_fraction: float = 0.2
    device: str = "cuda"
    model_name: str = "hf-hub:BVRA/MegaDescriptor-T-224"
    batch_size: int = 32
    crop_size: int = 224


class ProjectionHead(nn.Module):
    """Two-layer projection head mapping backbone features to L2-normalized embeddings.

    Architecture: Linear -> BatchNorm1d -> ReLU -> Linear -> L2-normalize.

    Args:
        in_dim: Input feature dimension (default 768 for MegaDescriptor-T).
        hidden_dim: Hidden layer dimension.
        out_dim: Output embedding dimension.
    """

    def __init__(
        self, in_dim: int = 768, hidden_dim: int = 256, out_dim: int = 128
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing L2-normalized embeddings.

        Args:
            x: Input tensor of shape ``(N, in_dim)``.

        Returns:
            L2-normalized tensor of shape ``(N, out_dim)``.
        """
        return F.normalize(self.net(x), p=2, dim=1)


class CachedFeatureDataset(Dataset):  # type: ignore[type-arg]
    """Dataset wrapping cached backbone features for metric learning.

    Args:
        features: Backbone feature array of shape ``(N, backbone_dim)``.
        labels: Integer label array of shape ``(N,)``.
        group_ids: Group ID array of shape ``(N,)``.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        group_ids: np.ndarray,
    ) -> None:
        self._features = torch.from_numpy(features).float()
        self._labels = torch.from_numpy(labels).long()
        self._group_ids = group_ids

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (feature_tensor, label_int) for the given index."""
        return self._features[idx], self._labels[idx]

    def get_labels(self) -> list[int]:
        """Return list of integer labels for MPerClassSampler."""
        return self._labels.tolist()


def build_feature_cache(
    reid_crops_dir: Path, config: ReidTrainingConfig
) -> dict[str, Any]:
    """Build or load a feature cache from reid_crops group manifests.

    Walks all ``group_NNN/manifest.json`` files, loads crops, and embeds
    them through FishEmbedder. Results are cached to disk as NPZ.

    Args:
        reid_crops_dir: Path to the reid_crops directory with group subdirs.
        config: Training configuration (used for embedder settings and output_dir).

    Returns:
        Dict with keys ``features`` (N, 768), ``labels`` (N,),
        ``group_ids`` (N,), and ``paths`` (list of str).
    """
    cache_path = config.output_dir / "backbone_cache.npz"
    if cache_path.exists():
        logger.info("Loading existing feature cache from %s", cache_path)
        data = np.load(cache_path, allow_pickle=True)
        return {
            "features": data["features"],
            "labels": data["labels"],
            "group_ids": data["group_ids"],
            "paths": list(data["paths"]),
        }

    # Discover groups sorted by group_id (matches temporal order from miner).
    group_dirs = sorted(reid_crops_dir.glob("group_*"))
    if not group_dirs:
        msg = f"No group directories found in {reid_crops_dir}"
        raise FileNotFoundError(msg)

    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    all_group_ids: list[int] = []
    all_paths: list[str] = []

    # Create embedder for backbone feature extraction.
    embedder_config = _make_embedder_config(config)
    embedder = FishEmbedder(embedder_config)

    for group_dir in group_dirs:
        manifest_path = group_dir / "manifest.json"
        if not manifest_path.exists():
            logger.warning("No manifest.json in %s, skipping", group_dir)
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        group_id = manifest["group_id"]
        crop_entries = manifest["crops"]

        # Collect crop images and metadata.
        crop_images: list[np.ndarray] = []
        crop_labels: list[int] = []
        crop_paths: list[str] = []

        for entry in crop_entries:
            img_path = group_dir / entry["filename"]
            if not img_path.exists():
                logger.warning("Missing crop image: %s", img_path)
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("Failed to read image: %s", img_path)
                continue
            crop_images.append(img)
            crop_labels.append(entry["fish_id"])
            crop_paths.append(str(img_path.relative_to(reid_crops_dir)))

        if not crop_images:
            continue

        # Embed in batches.
        features = embedder.embed_batch(crop_images)
        all_features.append(features)
        all_labels.extend(crop_labels)
        all_group_ids.extend([group_id] * len(crop_labels))
        all_paths.extend(crop_paths)

    result: dict[str, Any] = {
        "features": np.concatenate(all_features, axis=0).astype(np.float32),
        "labels": np.array(all_labels, dtype=np.int32),
        "group_ids": np.array(all_group_ids, dtype=np.int32),
        "paths": all_paths,
    }

    # Save cache to disk.
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        features=result["features"],
        labels=result["labels"],
        group_ids=result["group_ids"],
        paths=np.array(result["paths"], dtype=object),
    )
    logger.info(
        "Saved feature cache: %d crops, %d groups to %s",
        len(result["labels"]),
        len(set(result["group_ids"].tolist())),
        cache_path,
    )

    return result


def split_by_group(
    cache_dict: dict[str, Any], val_fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Split cache indices by temporal group for train/val holdout.

    Groups are sorted numerically (matching temporal order from the miner
    which slides forward through the video). The last ``val_fraction`` of
    groups form the validation set.

    Args:
        cache_dict: Feature cache dict with ``group_ids`` array.
        val_fraction: Fraction of groups to hold out for validation.

    Returns:
        Tuple of (train_indices, val_indices) as numpy int arrays.
    """
    group_ids = cache_dict["group_ids"]
    unique_groups = np.sort(np.unique(group_ids))
    n_val = max(1, int(len(unique_groups) * val_fraction))
    val_groups = set(unique_groups[-n_val:].tolist())

    train_mask = np.array([g not in val_groups for g in group_ids])
    val_mask = ~train_mask

    return np.where(train_mask)[0], np.where(val_mask)[0]


def compute_female_auc(
    embeddings: np.ndarray,
    fish_ids: np.ndarray,
    female_ids: frozenset[int] | set[int] = DEFAULT_FEMALE_IDS,
) -> float:
    """Compute pairwise AUC over all female fish embeddings.

    Filters to female-only embeddings, computes pairwise cosine similarity
    (dot product since L2-normalized), and returns sklearn roc_auc_score.

    Args:
        embeddings: L2-normalized embeddings of shape ``(N, D)``.
        fish_ids: Fish ID array of shape ``(N,)``.
        female_ids: Set of fish IDs considered female.

    Returns:
        AUC score (0.0 to 1.0). Returns 0.0 with warning if fewer than
        2 female fish are present.
    """
    mask = np.array([int(fid) in female_ids for fid in fish_ids])
    emb_f = embeddings[mask]
    ids_f = fish_ids[mask]

    unique_female = set(int(x) for x in ids_f)
    if len(unique_female) < 2:
        logger.warning(
            "Fewer than 2 female fish in data (%d found), returning AUC=0.0",
            len(unique_female),
        )
        return 0.0

    # Cosine similarity (embeddings are L2-normalized).
    sim = emb_f @ emb_f.T
    n = len(ids_f)
    scores: list[float] = []
    pair_labels: list[int] = []

    for i in range(n):
        for j in range(i + 1, n):
            scores.append(float(sim[i, j]))
            pair_labels.append(int(ids_f[i] == ids_f[j]))

    if sum(pair_labels) == 0 or sum(pair_labels) == len(pair_labels):
        logger.warning("Degenerate pair labels (all same or all different)")
        return 0.0

    return float(roc_auc_score(pair_labels, scores))


def compute_per_pair_auc(
    embeddings: np.ndarray,
    fish_ids: np.ndarray,
    female_ids: frozenset[int] | set[int] = DEFAULT_FEMALE_IDS,
) -> dict[tuple[int, int], float]:
    """Compute AUC for each unique female fish pair.

    For each pair ``(fish_i, fish_j)`` of distinct female fish, compute
    AUC restricted to embeddings of those two identities only.

    Args:
        embeddings: L2-normalized embeddings of shape ``(N, D)``.
        fish_ids: Fish ID array of shape ``(N,)``.
        female_ids: Set of fish IDs considered female.

    Returns:
        Dict keyed by ``(fish_i, fish_j)`` tuples (i < j) with AUC values.
    """
    mask = np.array([int(fid) in female_ids for fid in fish_ids])
    emb_f = embeddings[mask]
    ids_f = fish_ids[mask]

    unique_female = sorted(set(int(x) for x in ids_f))
    result: dict[tuple[int, int], float] = {}

    for idx_i, fi in enumerate(unique_female):
        for fj in unique_female[idx_i + 1 :]:
            pair_mask = np.array([int(fid) in {fi, fj} for fid in ids_f])
            pair_emb = emb_f[pair_mask]
            pair_ids = ids_f[pair_mask]

            if len(set(int(x) for x in pair_ids)) < 2:
                result[(fi, fj)] = 0.0
                continue

            sim = pair_emb @ pair_emb.T
            n = len(pair_ids)
            scores: list[float] = []
            labels: list[int] = []

            for a in range(n):
                for b in range(a + 1, n):
                    scores.append(float(sim[a, b]))
                    labels.append(int(pair_ids[a] == pair_ids[b]))

            if sum(labels) == 0 or sum(labels) == len(labels):
                result[(fi, fj)] = 0.0
            else:
                result[(fi, fj)] = float(roc_auc_score(labels, scores))

    return result


def train_reid_head(
    cache_dict_or_path: dict[str, Any] | Path | str,
    config: ReidTrainingConfig,
) -> dict[str, Any]:
    """Train projection head with SubCenterArcFace + BatchHardMiner.

    Accepts either a pre-built cache dict or path to an NPZ file. Splits
    data by group temporally, trains with early stopping on female AUC,
    and saves the best projection head weights.

    Args:
        cache_dict_or_path: Feature cache dict or path to NPZ file.
        config: Training configuration.

    Returns:
        Dict with ``best_auc``, ``best_epoch``, ``epochs_run``,
        ``per_pair_auc``, and ``val_auc_history``.
    """
    # Load cache.
    if isinstance(cache_dict_or_path, (str, Path)):
        data = np.load(str(cache_dict_or_path), allow_pickle=True)
        cache: dict[str, Any] = {
            "features": data["features"],
            "labels": data["labels"],
            "group_ids": data["group_ids"],
            "paths": list(data["paths"]),
        }
    else:
        cache = cache_dict_or_path

    device = torch.device(config.device)

    # Remap labels to contiguous 0..N-1 (ArcFace indexes a weight matrix by label).
    unique_labels = np.unique(cache["labels"])
    label_map = {int(old): new for new, old in enumerate(unique_labels)}
    remapped_labels = np.array([label_map[int(lbl)] for lbl in cache["labels"]])
    logger.info(
        "Label remap (%d classes): %s",
        len(unique_labels),
        {int(k): v for k, v in label_map.items()},
    )
    cache["labels_original"] = cache["labels"]
    cache["labels"] = remapped_labels

    # Split by group.
    train_idx, val_idx = split_by_group(cache, config.val_fraction)
    logger.info(
        "Split: %d train samples, %d val samples",
        len(train_idx),
        len(val_idx),
    )

    # Create datasets.
    train_ds = CachedFeatureDataset(
        cache["features"][train_idx],
        cache["labels"][train_idx],
        cache["group_ids"][train_idx],
    )
    # Create sampler and dataloader.
    train_labels = train_ds.get_labels()

    # Derive num_classes from actual data (some fish may be absent from mined crops).
    actual_num_classes = len(set(train_labels))
    batch_size = config.samples_per_class * actual_num_classes
    sampler = MPerClassSampler(
        labels=train_labels,
        m=config.samples_per_class,
        batch_size=batch_size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, drop_last=True
    )

    # Initialize model and loss.
    head = ProjectionHead(
        in_dim=config.backbone_dim,
        hidden_dim=config.hidden_dim,
        out_dim=config.embedding_dim,
    ).to(device)

    loss_func = losses.MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5)
    miner_func = miners.MultiSimilarityMiner(epsilon=0.1)

    # Single optimizer — MultiSimilarityLoss has no learnable parameters.
    head_optimizer = torch.optim.Adam(head.parameters(), lr=config.lr_head)

    # Cosine annealing scheduler (applied after warmup).
    cosine_epochs = max(1, config.epochs - config.warmup_epochs)
    head_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        head_optimizer, T_max=cosine_epochs, eta_min=config.lr_head * 0.01
    )

    # Training loop.
    best_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    val_auc_history: list[float] = []
    best_state_dict: dict[str, Any] | None = None

    for epoch in range(config.epochs):
        # Linear warmup: scale LR from 0 to target over warmup_epochs.
        if config.warmup_epochs > 0 and epoch < config.warmup_epochs:
            warmup_factor = (epoch + 1) / config.warmup_epochs
            for pg in head_optimizer.param_groups:
                pg["lr"] = config.lr_head * warmup_factor

        # Train.
        head.train()
        epoch_loss = 0.0
        n_batches = 0

        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)

            head_optimizer.zero_grad()

            embeddings = head(features_batch)
            mined_tuples = miner_func(embeddings, labels_batch)
            loss = loss_func(embeddings, labels_batch, mined_tuples)

            loss.backward()
            head_optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Step scheduler after warmup completes.
        if epoch >= config.warmup_epochs:
            head_scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)

        # Evaluate on val set.
        head.eval()
        with torch.no_grad():
            val_features = (
                torch.from_numpy(cache["features"][val_idx]).float().to(device)
            )
            val_embeddings = head(val_features).cpu().numpy()

        val_fish_ids = cache["labels_original"][val_idx]
        val_auc = compute_female_auc(val_embeddings, val_fish_ids)
        val_auc_history.append(val_auc)

        logger.info(
            "Epoch %d/%d  loss=%.4f  val_auc=%.4f",
            epoch + 1,
            config.epochs,
            avg_loss,
            val_auc,
        )

        # Track best.
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state_dict = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if config.patience > 0 and epochs_without_improvement >= config.patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs)",
                epoch + 1,
                config.patience,
            )
            break

    # Save best model.
    if best_state_dict is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        save_path = config.output_dir / "best_reid_model.pt"
        torch.save(best_state_dict, save_path)
        logger.info(
            "Saved best model (epoch %d, AUC %.4f) to %s",
            best_epoch,
            best_auc,
            save_path,
        )

    # Compute per-pair AUC on best model.
    per_pair_auc: dict[tuple[int, int], float] = {}
    if best_state_dict is not None:
        head.load_state_dict(best_state_dict)
        head.eval()
        with torch.no_grad():
            val_features_t = (
                torch.from_numpy(cache["features"][val_idx]).float().to(device)
            )
            val_emb_best = head(val_features_t).cpu().numpy()
        per_pair_auc = compute_per_pair_auc(val_emb_best, val_fish_ids)

    return {
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1 if config.epochs > 0 else 0,
        "per_pair_auc": per_pair_auc,
        "val_auc_history": val_auc_history,
    }


@dataclass(frozen=True)
class _EmbedderConfig:
    """Lightweight config satisfying ReidConfigLike protocol for FishEmbedder."""

    model_name: str
    batch_size: int
    crop_size: int
    device: str
    embedding_dim: int


def _make_embedder_config(config: ReidTrainingConfig) -> _EmbedderConfig:
    """Create a lightweight config object for FishEmbedder.

    Args:
        config: Training config with model_name, batch_size, crop_size, device.

    Returns:
        Object satisfying the ReidConfigLike protocol.
    """
    return _EmbedderConfig(
        model_name=config.model_name,
        batch_size=config.batch_size,
        crop_size=config.crop_size,
        device=config.device,
        embedding_dim=config.backbone_dim,
    )
