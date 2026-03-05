"""Training infrastructure for AquaPose models."""

from __future__ import annotations

from .common import (
    EarlyStopping,
    MetricsLogger,
    make_loader,
    save_best_and_last,
)
from .datasets import CropDataset, apply_augmentation, stratified_split
from .geometry import (
    extrapolate_edge_keypoints,
    format_obb_annotation,
    format_pose_annotation,
    pca_obb,
)
from .prep import prep_group
from .pseudo_labels import (
    compute_confidence_score,
    detect_gaps,
    generate_fish_labels,
    generate_gap_fish_labels,
    reproject_spline_keypoints,
)
from .yolo_obb import train_yolo_obb
from .yolo_pose import train_yolo_pose
from .yolo_seg import train_yolo_seg

__all__ = [
    "CropDataset",
    "EarlyStopping",
    "MetricsLogger",
    "apply_augmentation",
    "compute_confidence_score",
    "detect_gaps",
    "extrapolate_edge_keypoints",
    "format_obb_annotation",
    "format_pose_annotation",
    "generate_fish_labels",
    "generate_gap_fish_labels",
    "make_loader",
    "pca_obb",
    "prep_group",
    "reproject_spline_keypoints",
    "save_best_and_last",
    "stratified_split",
    "train_yolo_obb",
    "train_yolo_pose",
    "train_yolo_seg",
]
