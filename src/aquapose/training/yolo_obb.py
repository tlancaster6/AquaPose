"""YOLO-OBB training wrapper around the ultralytics library."""

from __future__ import annotations

from pathlib import Path

from .common import convert_ndjson_to_txt, rewrite_data_yaml, train_yolo_ndjson


def _format_obb_label_line(ann: dict) -> str:
    """Format one OBB annotation as a YOLO-OBB label line.

    Output format: ``class_id x1 y1 x2 y2 x3 y3 x4 y4`` with normalized coords.
    """
    class_id = int(ann["class_id"])
    parts: list[str] = []
    for corner in ann["corners"]:
        parts.append(f"{float(corner[0]):.6f}")
        parts.append(f"{float(corner[1]):.6f}")
    return f"{class_id} " + " ".join(parts)


def _convert_obb_ndjson_to_txt(ndjson_path: Path, labels_dir: Path) -> None:
    """Convert NDJSON OBB records to YOLO .txt label files."""
    convert_ndjson_to_txt(ndjson_path, labels_dir, _format_obb_label_line)


def _rewrite_data_yaml_obb(data_dir: Path, original_yaml_path: Path) -> Path:
    """Rewrite data.yaml for Ultralytics OBB training."""
    return rewrite_data_yaml(data_dir, original_yaml_path)


def train_yolo_obb(
    data_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 100,
    batch_size: int = 16,
    device: str | None = None,
    val_split: float = 0.2,
    imgsz: int = 640,
    model: str = "yolov8s-obb",
    weights: Path | None = None,
) -> Path:
    """Train a YOLO-OBB model on a project NDJSON oriented bounding-box dataset.

    Converts the project's NDJSON OBB labels to YOLO .txt format, rewrites
    ``data.yaml`` with absolute paths and ``images/`` directories, then invokes
    ``model.train()``. After training, copies ``best.pt`` and ``last.pt`` to
    ``output_dir`` under consistent names.

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
        model: YOLO model variant name (e.g. ``"yolov8s-obb"``,
            ``"yolov8n-obb"``). Used to download pretrained weights when
            ``weights`` is None.
        weights: Path to pretrained weights for transfer learning. When
            provided, the model is initialised from these weights instead of
            downloading.

    Returns:
        Path to the best model weights file (``output_dir/best_model.pt``).

    Raises:
        FileNotFoundError: If ``data.yaml`` is not found in ``data_dir``.
        ImportError: If the ``ultralytics`` package is not installed.
    """
    return train_yolo_ndjson(
        data_dir,
        output_dir,
        format_label_line=_format_obb_label_line,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        val_split=val_split,
        imgsz=imgsz,
        model=model,
        weights=weights,
    )
