"""Training infrastructure for AquaPose models."""

from __future__ import annotations

from .coco_convert import (
    compute_median_arc_length,
    generate_obb_dataset,
    generate_pose_dataset,
    load_coco,
    parse_frame_index,
    parse_keypoints,
    temporal_split,
)
from .coco_interchange import (
    coco_to_yolo_pose,
    write_coco_keypoints,
    yolo_pose_to_coco,
)
from .common import (
    EarlyStopping,
    MetricsLogger,
    make_loader,
    save_best_and_last,
)
from .compare import (
    discover_runs,
    format_comparison_table,
    load_run_summaries,
    write_comparison_csv,
)
from .data_cli import data_group
from .datasets import CropDataset, apply_augmentation, stratified_split
from .elastic_deform import (
    deform_keypoints_c_curve,
    deform_keypoints_s_curve,
    generate_deformed_labels,
    generate_variants,
    parse_pose_label,
    tps_warp_image,
)
from .geometry import (
    affine_warp_crop,
    compute_arc_length,
    extrapolate_edge_keypoints,
    format_obb_annotation,
    format_pose_annotation,
    pca_obb,
    transform_keypoints,
)
from .labelstudio_export import export_labelstudio_json
from .labelstudio_import import import_labelstudio_json
from .prep import prep_group
from .pseudo_labels import (
    compute_confidence_score,
    compute_curvature,
    detect_gaps,
    generate_fish_labels,
    generate_gap_fish_labels,
    reproject_spline_keypoints,
)
from .reid_training import (
    CachedFeatureDataset,
    ImageCropDataset,
    ProjectionHead,
    ReidTrainingConfig,
    build_feature_cache,
    compute_female_auc,
    compute_per_pair_auc,
    split_by_group,
    train_reid_end_to_end,
    train_reid_head,
    unfreeze_last_n_blocks,
)
from .run_manager import (
    create_run_dir,
    extract_dataset_provenance,
    parse_best_metrics,
    print_next_steps,
    register_trained_model,
    snapshot_config,
    update_config_weights,
    write_summary,
)
from .select_diverse_subset import select_obb_subset, select_pose_subset
from .store import SampleStore
from .yolo_training import train_yolo, train_yolo_obb, train_yolo_pose, train_yolo_seg

__all__ = [
    "CachedFeatureDataset",
    "CropDataset",
    "EarlyStopping",
    "ImageCropDataset",
    "MetricsLogger",
    "ProjectionHead",
    "ReidTrainingConfig",
    "SampleStore",
    "affine_warp_crop",
    "apply_augmentation",
    "build_feature_cache",
    "coco_to_yolo_pose",
    "compute_arc_length",
    "compute_confidence_score",
    "compute_curvature",
    "compute_female_auc",
    "compute_median_arc_length",
    "compute_per_pair_auc",
    "create_run_dir",
    "data_group",
    "deform_keypoints_c_curve",
    "deform_keypoints_s_curve",
    "detect_gaps",
    "discover_runs",
    "export_labelstudio_json",
    "extract_dataset_provenance",
    "extrapolate_edge_keypoints",
    "format_comparison_table",
    "format_obb_annotation",
    "format_pose_annotation",
    "generate_deformed_labels",
    "generate_fish_labels",
    "generate_gap_fish_labels",
    "generate_obb_dataset",
    "generate_pose_dataset",
    "generate_variants",
    "import_labelstudio_json",
    "load_coco",
    "load_run_summaries",
    "make_loader",
    "parse_best_metrics",
    "parse_frame_index",
    "parse_keypoints",
    "parse_pose_label",
    "pca_obb",
    "prep_group",
    "print_next_steps",
    "register_trained_model",
    "reproject_spline_keypoints",
    "save_best_and_last",
    "select_obb_subset",
    "select_pose_subset",
    "snapshot_config",
    "split_by_group",
    "stratified_split",
    "temporal_split",
    "tps_warp_image",
    "train_reid_end_to_end",
    "train_reid_head",
    "train_yolo",
    "train_yolo_obb",
    "train_yolo_pose",
    "train_yolo_seg",
    "transform_keypoints",
    "unfreeze_last_n_blocks",
    "update_config_weights",
    "write_coco_keypoints",
    "write_comparison_csv",
    "write_summary",
    "yolo_pose_to_coco",
]
