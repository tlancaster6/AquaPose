"""Training infrastructure for AquaPose models."""

from __future__ import annotations

from .common import EarlyStopping, MetricsLogger, make_loader, save_best_and_last
from .datasets import (
    BinaryMaskDataset,
    CropDataset,
    apply_augmentation,
    stratified_split,
)
from .pose import KeypointDataset, train_pose
from .unet import train_unet
from .yolo_obb import train_yolo_obb

__all__ = [
    "BinaryMaskDataset",
    "CropDataset",
    "EarlyStopping",
    "KeypointDataset",
    "MetricsLogger",
    "apply_augmentation",
    "make_loader",
    "save_best_and_last",
    "stratified_split",
    "train_pose",
    "train_unet",
    "train_yolo_obb",
]
