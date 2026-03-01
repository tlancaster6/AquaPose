"""YOLO-pose training wrapper around the ultralytics library."""

from __future__ import annotations

from pathlib import Path

from .common import convert_ndjson_to_txt, rewrite_data_yaml, train_yolo_ndjson

_POSE_YAML_KEYS = ("kpt_shape", "kpt_names", "flip_idx")


def _format_pose_label_line(ann: dict) -> str:
    """Format one pose annotation as a YOLO label line.

    Output: ``class_id cx cy w h x1 y1 v1 x2 y2 v2 ... xN yN vN``.
    """
    class_id = int(ann["class_id"])
    bbox = ann["bbox"]
    cx, cy, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    kp_parts: list[str] = []
    for kp in ann["keypoints"]:
        kp_parts.append(f"{float(kp[0]):.6f}")
        kp_parts.append(f"{float(kp[1]):.6f}")
        kp_parts.append(str(int(kp[2])))
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} " + " ".join(kp_parts)


def _convert_pose_ndjson_to_txt(ndjson_path: Path, labels_dir: Path) -> None:
    """Convert NDJSON pose records to YOLO .txt label files."""
    convert_ndjson_to_txt(ndjson_path, labels_dir, _format_pose_label_line)


def _rewrite_data_yaml_pose(data_dir: Path, original_yaml_path: Path) -> Path:
    """Rewrite data.yaml for Ultralytics pose training."""
    return rewrite_data_yaml(
        data_dir, original_yaml_path, preserve_keys=_POSE_YAML_KEYS
    )


def train_yolo_pose(
    data_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 100,
    batch_size: int = 16,
    device: str | None = None,
    val_split: float = 0.2,
    imgsz: int = 640,
    model: str = "yolo26n-pose",
    weights: Path | None = None,
) -> Path:
    """Train a YOLO-pose keypoint estimation model on a project NDJSON dataset.

    Converts the project's NDJSON pose labels to YOLO .txt format, rewrites
    ``data.yaml`` with absolute paths and ``images/`` directories, then invokes
    ``model.train()``. After training, copies ``best.pt`` and ``last.pt`` to
    ``output_dir`` under consistent names. Keypoint definition (names, count,
    skeleton) is read from ``data.yaml`` and preserved — nothing is hardcoded.

    Args:
        data_dir: Directory containing ``data.yaml`` and the NDJSON dataset
            (with ``images/train/`` and ``images/val/`` subdirectories).
        output_dir: Directory for model weights and metrics CSV.
        epochs: Number of training epochs.
        batch_size: Images per batch.
        device: Torch device string (e.g. ``"cuda:0"``, ``"cpu"``). Auto-
            detected if None.
        val_split: Validation split fraction (informational — recorded in
            metrics; actual split is determined by ``data.yaml``).
        imgsz: Training image size (square).
        model: YOLO model variant name (e.g. ``"yolo26n-pose"``,
            ``"yolo26s-pose"``). Used to download pretrained weights when
            ``weights`` is None.
        weights: Path to pretrained weights for transfer learning. When
            provided, the model is initialised from these weights instead of
            downloading.

    Returns:
        Path to the best model weights file (``output_dir/best_model.pt``).

    Raises:
        FileNotFoundError: If ``data.yaml`` is not found in ``data_dir``.
    """
    return train_yolo_ndjson(
        data_dir,
        output_dir,
        format_label_line=_format_pose_label_line,
        yaml_preserve_keys=_POSE_YAML_KEYS,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        val_split=val_split,
        imgsz=imgsz,
        model=model,
        weights=weights,
    )
