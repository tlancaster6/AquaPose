"""YOLO-seg training wrapper around the ultralytics library."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch


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
    """Train a YOLO-seg instance segmentation model on a standard Ultralytics txt+yaml dataset.

    Passes the ``dataset.yaml`` file path directly to ``model.train(data=...)``.
    After training, copies ``best.pt`` and ``last.pt`` to ``output_dir`` under
    consistent names.

    Args:
        data_dir: Directory containing ``dataset.yaml`` (with
            ``images/train/``, ``images/val/``, ``labels/train/``, and
            ``labels/val/`` subdirectories).
        output_dir: Directory for model weights and metrics CSV.
        epochs: Number of training epochs.
        batch_size: Images per batch.
        device: Torch device string (e.g. ``"cuda:0"``, ``"cpu"``). Auto-
            detected if None.
        val_split: Validation split fraction (informational only).
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
        FileNotFoundError: If ``dataset.yaml`` is not found in ``data_dir``.
        ImportError: If the ``ultralytics`` package is not installed.
    """
    from ultralytics import YOLO  # type: ignore[import-untyped]

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = data_dir / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found in {data_dir}")

    # Initialize model
    yolo_model = YOLO(str(weights)) if weights is not None else YOLO(f"{model}.pt")

    # Run ultralytics training
    results = yolo_model.train(
        data=str(yaml_path),
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
