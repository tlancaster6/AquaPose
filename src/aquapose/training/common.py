"""Shared training utilities for AquaPose model training."""

from __future__ import annotations

import csv
import json
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class EarlyStopping:
    """Tracks metric history and signals when training should stop.

    Args:
        patience: Number of epochs without improvement before stopping.
        mode: "min" if lower metric is better, "max" if higher is better.
    """

    def __init__(self, patience: int, mode: str = "max") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self._patience = patience
        self._mode = mode
        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._epochs_without_improvement: int = 0

    @property
    def best(self) -> float:
        """Return the best metric value seen so far."""
        return self._best

    def step(self, metric: float) -> bool:
        """Update with the latest metric value.

        Args:
            metric: The current epoch metric value.

        Returns:
            True if training should stop (patience exceeded), False otherwise.
        """
        improved: bool = (
            metric < self._best if self._mode == "min" else metric > self._best
        )

        if improved:
            self._best = metric
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        return self._patience > 0 and self._epochs_without_improvement >= self._patience


class MetricsLogger:
    """Writes per-epoch metrics to console and a CSV file.

    Args:
        output_dir: Directory where ``metrics.csv`` will be written.
        fields: List of metric field names (excluding epoch).
    """

    def __init__(self, output_dir: Path, fields: list[str]) -> None:
        self._output_dir = Path(output_dir)
        self._fields = fields
        self._csv_path = self._output_dir / "metrics.csv"
        self._start_time = time.monotonic()
        self._total_epochs: int | None = None

        # Write CSV header
        self._output_dir.mkdir(parents=True, exist_ok=True)
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", *fields])
            writer.writeheader()

    def set_total_epochs(self, total: int) -> None:
        """Set the total number of epochs for progress display.

        Args:
            total: Total epoch count.
        """
        self._total_epochs = total

    def log(self, epoch: int, **metrics: float) -> None:
        """Log metrics for one epoch to console and CSV.

        Args:
            epoch: Current epoch number (1-indexed).
            **metrics: Metric values keyed by field name.
        """
        elapsed = time.monotonic() - self._start_time
        total_str = f"/{self._total_epochs}" if self._total_epochs is not None else ""
        parts = [f"Epoch {epoch}{total_str}"]
        for field in self._fields:
            val = metrics.get(field, float("nan"))
            parts.append(f"{field}: {val:.4f}")
        parts.append(f"time: {elapsed:.1f}s")
        print(" - ".join(parts))

        # Append to CSV
        row: dict[str, object] = {"epoch": epoch}
        for field in self._fields:
            row[field] = metrics.get(field, float("nan"))
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", *self._fields])
            writer.writerow(row)


def save_best_and_last(
    model: nn.Module,
    output_dir: Path,
    metric: float,
    best_metric: float,
    metric_name: str = "",
    *,
    mode: str | None = None,
) -> tuple[Path, float]:
    """Save the model checkpoint, updating best if metric improved.

    Saves ``last_model.pth`` unconditionally. Saves ``best_model.pth`` when
    the metric improves.

    Args:
        model: The PyTorch model to save.
        output_dir: Directory for checkpoints.
        metric: Current epoch metric value.
        best_metric: Best metric seen so far.
        metric_name: Deprecated — use ``mode`` instead.
        mode: ``"min"`` if lower is better, ``"max"`` if higher is better.
            Defaults to ``"min"`` when ``metric_name`` contains ``"loss"``
            or ``"error"``, ``"max"`` otherwise.

    Returns:
        Tuple of (best_model_path, updated_best_metric).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    last_path = output_dir / "last_model.pth"
    best_path = output_dir / "best_model.pth"

    torch.save(model.state_dict(), last_path)

    if mode is None:
        name_lower = metric_name.lower()
        minimize = "loss" in name_lower or "error" in name_lower
    else:
        minimize = mode == "min"
    improved = metric < best_metric if minimize else metric > best_metric

    if improved:
        torch.save(model.state_dict(), best_path)
        return best_path, metric

    return best_path, best_metric


def convert_ndjson_to_txt(
    ndjson_path: Path,
    labels_dir: Path,
    format_line: Callable[[dict], str],
) -> None:
    """Convert NDJSON records to YOLO ``.txt`` label files.

    Reads one JSON object per line. For each record with annotations, writes
    a ``.txt`` file named after the image stem, with one label line per
    annotation formatted by ``format_line``.

    Args:
        ndjson_path: Path to the NDJSON file.
        labels_dir: Output directory for ``.txt`` label files.
        format_line: Converts a single annotation dict to a YOLO label line.
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
            lines = [format_line(ann) for ann in annotations]
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rewrite_data_yaml(
    data_dir: Path,
    original_yaml_path: Path,
    preserve_keys: Sequence[str] = (),
) -> Path:
    """Rewrite ``data.yaml`` for Ultralytics training.

    Reads the original ``data.yaml`` (which may reference NDJSON files) and
    writes ``data_ultralytics.yaml`` with an absolute ``path:`` field and
    ``train: images/train``, ``val: images/val``.  Preserves ``nc`` and
    ``names`` from the original, plus any keys listed in ``preserve_keys``.

    Args:
        data_dir: Dataset root directory (absolute path used for ``path:``).
        original_yaml_path: Path to the original ``data.yaml``.
        preserve_keys: Additional YAML keys to copy from the original
            (e.g. ``("kpt_shape", "kpt_names", "flip_idx")`` for pose).

    Returns:
        Path to the newly written ``data_ultralytics.yaml``.
    """
    import yaml  # pyyaml — already a dep via ultralytics

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

    for key in preserve_keys:
        if key not in original:
            continue
        val = original[key]
        if isinstance(val, list):
            items = ", ".join(repr(x) if isinstance(x, str) else str(x) for x in val)
            lines.append(f"{key}: [{items}]")
        else:
            lines.append(f"{key}: {val}")

    output_path = data_dir / "data_ultralytics.yaml"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def train_yolo_ndjson(
    data_dir: Path,
    output_dir: Path,
    *,
    format_label_line: Callable[[dict], str],
    yaml_preserve_keys: Sequence[str] = (),
    epochs: int = 100,
    batch_size: int = 16,
    device: str | None = None,
    val_split: float = 0.2,
    imgsz: int = 640,
    model: str,
    weights: Path | None = None,
) -> Path:
    """Train a YOLO model from an NDJSON-labeled dataset.

    Handles the complete pipeline: NDJSON-to-YOLO ``.txt`` conversion,
    ``data.yaml`` rewriting with absolute paths, Ultralytics model
    initialisation and training, and weight file copying.

    Args:
        data_dir: Directory containing ``data.yaml`` and the NDJSON dataset
            (with ``images/train/`` and ``images/val/`` subdirectories).
        output_dir: Directory for model weights and metrics CSV.
        format_label_line: Converts one annotation dict to a YOLO label line.
        yaml_preserve_keys: Extra ``data.yaml`` keys to preserve (e.g. pose
            keypoint fields).
        epochs: Number of training epochs.
        batch_size: Images per batch.
        device: Torch device string. Auto-detected if None.
        val_split: Validation split fraction (informational — recorded in
            metrics; actual split is determined by ``data.yaml``).
        imgsz: Training image size (square).
        model: YOLO model variant name (e.g. ``"yolo26n-seg"``). Used to
            download pretrained weights when ``weights`` is None.
        weights: Pretrained weights path for transfer learning.

    Returns:
        Path to the best model weights file (``output_dir/best_model.pt``).

    Raises:
        FileNotFoundError: If ``data.yaml`` is not found in ``data_dir``.
    """
    import yaml
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
        convert_ndjson_to_txt(
            ndjson_path, data_dir / "labels" / split, format_label_line
        )

    # Rewrite data.yaml with absolute path and images/ directories
    rewritten_yaml = rewrite_data_yaml(data_dir, data_yaml, yaml_preserve_keys)

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

    # Copy weights to output_dir
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
    if not best_dst.exists() and last_dst.exists():
        shutil.copy2(last_dst, best_dst)

    return best_dst


def make_loader(
    dataset: Dataset,  # type: ignore[type-arg]
    batch_size: int,
    shuffle: bool,
    device: str,
    num_workers: int = 4,
) -> DataLoader:  # type: ignore[type-arg]
    """Create a DataLoader with sensible defaults for the target device.

    Args:
        dataset: The dataset to wrap.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle samples each epoch.
        device: Target device string (e.g. "cuda", "cpu").
        num_workers: Number of parallel data loading workers.

    Returns:
        Configured DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=(num_workers > 0),
    )
