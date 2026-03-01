"""Training infrastructure for AquaPose models."""

from __future__ import annotations

from .common import (
    EarlyStopping,
    MetricsLogger,
    make_loader,
    save_best_and_last,
)
from .datasets import CropDataset, apply_augmentation, stratified_split
from .prep import prep_group
from .yolo_obb import train_yolo_obb
from .yolo_pose import train_yolo_pose
from .yolo_seg import train_yolo_seg

__all__ = [
    "CropDataset",
    "EarlyStopping",
    "MetricsLogger",
    "apply_augmentation",
    "make_loader",
    "prep_group",
    "save_best_and_last",
    "stratified_split",
    "train_yolo_obb",
    "train_yolo_pose",
    "train_yolo_seg",
]
