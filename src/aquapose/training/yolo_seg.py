"""YOLO-seg training wrapper around the ultralytics library."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from .common import MetricsLogger


def _convert_seg_ndjson_to_txt(ndjson_path: Path, labels_dir: Path) -> None:
    """Convert NDJSON segmentation records to YOLO .txt label files.

    Each NDJSON record corresponds to one image. For each annotation in a
    record, writes one line in YOLO seg format::

        class_id x1 y1 x2 y2 ... xN yN

    Coordinates are normalized [0, 1] and sourced directly from the polygon
    field in the NDJSON record.

    Args:
        ndjson_path: Path to the NDJSON file (one JSON object per line).
        labels_dir: Output directory for .txt label files. Created if absent.
    """
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(ndjson_path, encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            annotations = record.get("annotations", [])
            if not annotations:
                continue

            image_stem = Path(record["image"]).stem
            label_path = labels_dir / f"{image_stem}.txt"

            lines: list[str] = []
            for ann in annotations:
                class_id = int(ann["class_id"])
                polygon = ann["polygon"]
                coords: list[str] = []
                for pt in polygon:
                    coords.append(f"{float(pt[0]):.6f}")
                    coords.append(f"{float(pt[1]):.6f}")
                lines.append(f"{class_id} " + " ".join(coords))

            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rewrite_data_yaml_seg(
    data_dir: Path,
    original_yaml_path: Path,
) -> Path:
    """Rewrite data.yaml for Ultralytics seg training.

    Reads the original data.yaml (which references NDJSON files), then writes
    ``data_ultralytics.yaml`` with ``train: images/train``, ``val: images/val``
    and an absolute ``path:`` field. Preserves ``nc`` and ``names`` from the
    original.

    Args:
        data_dir: Dataset root directory (absolute path used for ``path:``).
        original_yaml_path: Path to the original ``data.yaml`` file.

    Returns:
        Path to the newly written ``data_ultralytics.yaml``.
    """
    import yaml  # stdlib-compatible; pyyaml is already a dep via ultralytics

    with open(original_yaml_path, encoding="utf-8") as f:
        original = yaml.safe_load(f)

    nc = original.get("nc", 1)
    names = original.get("names", [])
    abs_path = str(data_dir.resolve())

    lines = [
        f"path: {abs_path}",
        "train: images/train",
        "val: images/val",
        f"nc: {nc}",
    ]
    if isinstance(names, list):
        names_str = "[" + ", ".join(repr(n) for n in names) + "]"
        lines.append(f"names: {names_str}")
    elif isinstance(names, dict):
        lines.append("names:")
        for k, v in sorted(names.items()):
            lines.append(f"  {k}: {v!r}")

    output_path = data_dir / "data_ultralytics.yaml"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def train_yolo_seg(
    data_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 100,
    batch_size: int = 16,
    device: str | None = None,
    val_split: float = 0.2,
    imgsz: int = 640,
    model: str = "yolo26n-seg",
    weights: Path | None = None,
) -> Path:
    """Train a YOLO-seg instance segmentation model on a project NDJSON dataset.

    Converts the project's NDJSON segmentation labels to YOLO .txt format,
    rewrites ``data.yaml`` with absolute paths and ``images/`` directories, then
    invokes ``model.train()``. After training, copies ``best.pt`` and ``last.pt``
    to ``output_dir`` under consistent names.

    The dataset must contain a ``data.yaml`` in ``data_dir`` with ``train`` and
    ``val`` keys pointing to NDJSON files, plus ``nc`` and ``names`` fields.

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
        model: YOLO model variant name (e.g. ``"yolo26n-seg"``,
            ``"yolo26s-seg"``). Used to download pretrained weights when
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
    import yaml  # used for reading NDJSON split paths from data.yaml
    from ultralytics import YOLO  # type: ignore[import-untyped]

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = data_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found in {data_dir}")

    # Convert NDJSON labels to YOLO .txt format for each split
    with open(data_yaml, encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f)

    for split in ("train", "val"):
        ndjson_ref = yaml_cfg.get(split)
        if not ndjson_ref:
            continue
        ndjson_path = data_dir / ndjson_ref
        if not ndjson_path.exists():
            continue
        labels_dir = data_dir / "labels" / split
        _convert_seg_ndjson_to_txt(ndjson_path, labels_dir)

    # Rewrite data.yaml with absolute path and images/ directories
    rewritten_yaml = _rewrite_data_yaml_seg(data_dir, data_yaml)

    # Initialise metrics logger for a final summary line
    logger = MetricsLogger(output_dir, fields=["epochs", "val_split"])
    logger.log(0, epochs=float(epochs), val_split=val_split)

    # Initialize model
    yolo_model = YOLO(str(weights)) if weights is not None else YOLO(f"{model}.pt")

    # Run ultralytics training
    results = yolo_model.train(
        data=str(rewritten_yaml),
        epochs=epochs,
        batch=batch_size,
        device=device,
        project=str(output_dir / "_ultralytics"),
        name="train",
        imgsz=imgsz,
    )

    # Locate ultralytics output directory
    save_dir = (
        results.save_dir
        if results is not None and results.save_dir is not None
        else output_dir / "_ultralytics" / "train"
    )
    weights_dir = Path(str(save_dir)) / "weights"

    best_src = weights_dir / "best.pt"
    last_src = weights_dir / "last.pt"

    best_dst = output_dir / "best_model.pt"
    last_dst = output_dir / "last_model.pt"

    if best_src.exists():
        shutil.copy2(best_src, best_dst)
    if last_src.exists():
        shutil.copy2(last_src, last_dst)

    # Fall back to last if best was never saved
    if not best_dst.exists() and last_dst.exists():
        shutil.copy2(last_dst, best_dst)

    return best_dst
