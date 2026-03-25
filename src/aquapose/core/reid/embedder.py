"""MegaDescriptor-T backbone wrapper for fish re-identification embeddings."""

from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ReidConfigLike(Protocol):
    """Structural protocol matching ReidConfig from engine.config."""

    model_name: str
    batch_size: int
    crop_size: int
    device: str
    embedding_dim: int


class FishEmbedder:
    """Stateful GPU wrapper around MegaDescriptor-T for fish re-identification.

    Loads the backbone once at construction time, holds it on GPU.
    Accepts BGR uint8 crops of any size, preprocesses internally
    (resize to crop_size x crop_size, normalize to [-1, 1]),
    and returns L2-normalized embedding vectors.

    Args:
        config: Object with model_name, batch_size, crop_size, device, and
            embedding_dim attributes (e.g. ``ReidConfig``).
    """

    def __init__(self, config: ReidConfigLike) -> None:
        self._device = torch.device(config.device)
        self._batch_size = config.batch_size
        self._crop_size = config.crop_size
        self._embedding_dim = config.embedding_dim

        self._model = timm.create_model(
            config.model_name, num_classes=0, pretrained=True
        )
        self._model.to(self._device)
        self._model.eval()

        # Verify output dimension with a dummy forward pass.
        with torch.no_grad():
            dummy = torch.randn(1, 3, self._crop_size, self._crop_size).to(self._device)
            out = self._model(dummy)
        if out.shape != (1, self._embedding_dim):
            raise ValueError(
                f"Model output shape {out.shape} does not match expected "
                f"(1, {self._embedding_dim}). Check model_name and embedding_dim."
            )

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embed a list of BGR uint8 crops into L2-normalized vectors.

        Each crop is resized to ``(crop_size, crop_size)``, converted to
        RGB, normalized to [-1, 1], and passed through the backbone.
        Results are L2-normalized and returned as a numpy array.

        Args:
            crops: List of BGR uint8 numpy arrays of any spatial size.
                Each array has shape ``(H, W, 3)``.

        Returns:
            Float32 numpy array of shape ``(N, embedding_dim)`` with
            unit-norm rows, where ``N = len(crops)``.
        """
        if len(crops) == 0:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        # Process in sub-batches to avoid materializing all crops at once.
        all_feats = []
        with torch.no_grad():
            for i in range(0, len(crops), self._batch_size):
                sub_crops = crops[i : i + self._batch_size]

                # Preprocess this sub-batch only.
                tensors = []
                for crop in sub_crops:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(
                        rgb,
                        (self._crop_size, self._crop_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    arr = resized.astype(np.float32) / 255.0
                    arr = (arr - 0.5) / 0.5
                    tensors.append(torch.from_numpy(arr.transpose(2, 0, 1)))

                batch_tensor = torch.stack(tensors).to(self._device)
                feats = self._model(batch_tensor)
                feats = F.normalize(feats, p=2, dim=1)
                all_feats.append(feats.cpu().numpy())
                del batch_tensor, feats  # free GPU memory promptly

        return np.concatenate(all_feats, axis=0).astype(np.float32)
