"""Shared training utilities for AquaPose model training."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class _LutConfigFromDict:
    """Minimal LUT config satisfying the ``LutConfigLike`` protocol.

    Built from a plain dict (YAML ``lut`` section) to avoid importing
    ``aquapose.engine.config`` (which would violate the training->engine
    import boundary).
    """

    def __init__(self, d: dict) -> None:
        self.tank_diameter: float = float(d.get("tank_diameter", 2.0))
        self.tank_height: float = float(d.get("tank_height", 1.0))
        self.voxel_resolution_m: float = float(d.get("voxel_resolution_m", 0.02))
        self.margin_fraction: float = float(d.get("margin_fraction", 0.1))
        self.forward_grid_step: int = int(d.get("forward_grid_step", 1))


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
