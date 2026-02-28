"""YOLO-OBB training wrapper around the ultralytics library."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from .common import MetricsLogger


def train_yolo_obb(
    data_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 100,
    batch_size: int = 16,
    device: str | None = None,
    val_split: float = 0.2,
    imgsz: int = 640,
    model_size: str = "s",
) -> Path:
    """Train a YOLO-OBB model on a YOLO-format oriented bounding-box dataset.

    Wraps the ultralytics training API. After training, copies best.pt and
    last.pt from the ultralytics output directory to ``output_dir`` under
    consistent names (``best_model.pt`` and ``last_model.pt``).

    The dataset must contain a ``data.yaml`` file in ``data_dir`` following
    the ultralytics YOLO format. ``val_split`` is recorded in the metrics log
    but the actual train/val split is controlled by ``data.yaml``.

    Args:
        data_dir: Directory containing ``data.yaml`` and the YOLO-format
            dataset (images/ and labels/ subdirectories).
        output_dir: Directory for model weights and metrics CSV.
        epochs: Number of training epochs.
        batch_size: Images per batch.
        device: Torch device string (e.g. ``"cuda:0"``, ``"cpu"``). Auto-
            detected if None.
        val_split: Validation split fraction (informational — recorded in
            metrics; actual split is determined by ``data.yaml``).
        imgsz: Training image size (square).
        model_size: YOLO model size suffix. One of ``n``, ``s``, ``m``,
            ``l``, ``x``. Selects ``yolov8{model_size}-obb.pt``.

    Returns:
        Path to the best model weights file (``output_dir/best_model.pt``).

    Raises:
        FileNotFoundError: If ``data.yaml`` is not found in ``data_dir``.
        ImportError: If the ``ultralytics`` package is not installed.
    """
    from ultralytics import YOLO  # type: ignore[import-untyped]

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = data_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found in {data_dir}")

    # Initialise metrics logger for a final summary line
    logger = MetricsLogger(output_dir, fields=["epochs", "val_split"])
    logger.log(0, epochs=float(epochs), val_split=val_split)

    # Run ultralytics training — it handles its own epoch-level logging
    model = YOLO(f"yolov8{model_size}-obb.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        device=device,
        project=str(output_dir / "_ultralytics"),
        name="train",
        imgsz=imgsz,
    )

    # Locate ultralytics output directory
    # ultralytics saves to project/name/weights/
    weights_dir = Path(str(results.save_dir)) / "weights"

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
