"""Training infrastructure for AquaPose models."""

from __future__ import annotations

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
from .dataset_assembly import assemble_dataset
from .datasets import CropDataset, apply_augmentation, stratified_split
from .elastic_deform import (
    deform_keypoints_c_curve,
    deform_keypoints_s_curve,
    generate_deformed_labels,
    generate_preview_grid,
    generate_variants,
    parse_pose_label,
    tps_warp_image,
    write_yolo_dataset,
)
from .frame_selection import (
    DiversitySampleResult,
    compute_curvature,
    diversity_sample,
    filter_empty_frames,
    temporal_subsample,
)
from .geometry import (
    affine_warp_crop,
    extrapolate_edge_keypoints,
    format_obb_annotation,
    format_pose_annotation,
    pca_obb,
    transform_keypoints,
)
from .prep import prep_group
from .pseudo_labels import (
    compute_confidence_score,
    detect_gaps,
    generate_fish_labels,
    generate_gap_fish_labels,
    reproject_spline_keypoints,
)
from .run_manager import (
    create_run_dir,
    extract_dataset_provenance,
    parse_best_metrics,
    print_next_steps,
    snapshot_config,
    write_summary,
)
from .yolo_obb import train_yolo_obb
from .yolo_pose import train_yolo_pose
from .yolo_seg import train_yolo_seg

__all__ = [
    "CropDataset",
    "DiversitySampleResult",
    "EarlyStopping",
    "MetricsLogger",
    "affine_warp_crop",
    "apply_augmentation",
    "assemble_dataset",
    "coco_to_yolo_pose",
    "compute_confidence_score",
    "compute_curvature",
    "create_run_dir",
    "deform_keypoints_c_curve",
    "deform_keypoints_s_curve",
    "detect_gaps",
    "discover_runs",
    "diversity_sample",
    "extract_dataset_provenance",
    "extrapolate_edge_keypoints",
    "filter_empty_frames",
    "format_comparison_table",
    "format_obb_annotation",
    "format_pose_annotation",
    "generate_deformed_labels",
    "generate_fish_labels",
    "generate_gap_fish_labels",
    "generate_preview_grid",
    "generate_variants",
    "load_run_summaries",
    "make_loader",
    "parse_best_metrics",
    "parse_pose_label",
    "pca_obb",
    "prep_group",
    "print_next_steps",
    "reproject_spline_keypoints",
    "save_best_and_last",
    "snapshot_config",
    "stratified_split",
    "temporal_subsample",
    "tps_warp_image",
    "train_yolo_obb",
    "train_yolo_pose",
    "train_yolo_seg",
    "transform_keypoints",
    "write_coco_keypoints",
    "write_comparison_csv",
    "write_summary",
    "write_yolo_dataset",
    "yolo_pose_to_coco",
]
