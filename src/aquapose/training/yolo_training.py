"""Consolidated YOLO training wrappers for OBB, segmentation, and pose models."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

_MODEL_DEFAULTS: dict[str, dict] = {
    "obb": {"model": "yolo26n-obb"},
    "seg": {"model": "yolo26n-seg"},
    "pose": {"model": "yolo26n-pose", "rect": True},
}


def train_yolo(
    data_dir: Path,
    output_dir: Path,
    model_type: str,
    *,
    epochs: int = 100,
    batch_size: int = 16,
    device: str | None = None,
    val_split: float = 0.2,
    imgsz: int = 640,
    model: str | None = None,
    weights: Path | None = None,
    patience: int = 100,
    mosaic: float = 1.0,
    rect: bool | None = None,
) -> Path:
    """Train a YOLO model on a standard Ultralytics txt+yaml dataset.

    Unified entry point for OBB, segmentation, and pose training. Model-type-
    specific defaults (model name, rect mode) are applied from
    ``_MODEL_DEFAULTS`` when not explicitly provided.

    Args:
        data_dir: Directory containing ``dataset.yaml`` (with
            ``images/train/``, ``images/val/``, ``labels/train/``, and
            ``labels/val/`` subdirectories).
        output_dir: Directory for model weights and metrics CSV.
        model_type: One of ``"obb"``, ``"seg"``, or ``"pose"``.
        epochs: Number of training epochs.
        batch_size: Images per batch.
        device: Torch device string (e.g. ``"cuda:0"``, ``"cpu"``). Auto-
            detected if None.
        val_split: Validation split fraction (informational only).
        imgsz: Training image size (square).
        model: YOLO model variant name. When None, uses the default for the
            given ``model_type``.
        weights: Path to pretrained weights for transfer learning. When
            provided, the model is initialised from these weights instead of
            downloading.
        patience: Early-stopping patience in epochs.
        mosaic: Mosaic augmentation probability (0.0 to 1.0). Set to 0.0
            to disable mosaic, which can help when targets are small.
        rect: If True, use rectangular training batches. When None, uses
            the model-type default (True for pose, not set for others).

    Returns:
        Path to the best model weights file (``output_dir/best_model.pt``).

    Raises:
        FileNotFoundError: If ``dataset.yaml`` is not found in ``data_dir``.
        ValueError: If ``model_type`` is not one of ``"obb"``, ``"seg"``,
            ``"pose"``.
    """
    from ultralytics import YOLO  # type: ignore[import-untyped]

    if model_type not in _MODEL_DEFAULTS:
        msg = f"Unknown model_type {model_type!r}; expected one of {list(_MODEL_DEFAULTS)}"
        raise ValueError(msg)

    defaults = _MODEL_DEFAULTS[model_type]

    if model is None:
        model = defaults["model"]
    if rect is None:
        rect = defaults.get("rect")

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

    # Build training kwargs
    train_kwargs: dict = {
        "data": str(yaml_path),
        "epochs": epochs,
        "batch": batch_size,
        "device": device,
        "project": str(output_dir / "_ultralytics"),
        "name": "train",
        "imgsz": imgsz,
        "patience": patience,
        "mosaic": mosaic,
    }
    if rect:
        train_kwargs["rect"] = rect

    # Run ultralytics training
    results = yolo_model.train(**train_kwargs)

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


def train_yolo_obb(
    data_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 100,
    batch_size: int = 16,
    device: str | None = None,
    val_split: float = 0.2,
    imgsz: int = 640,
    model: str = "yolo26n-obb",
    weights: Path | None = None,
    patience: int = 100,
    mosaic: float = 1.0,
) -> Path:
    """Train a YOLO-OBB oriented bounding-box detection model.

    Convenience wrapper around :func:`train_yolo` with OBB defaults.
    See :func:`train_yolo` for full parameter documentation.
    """
    return train_yolo(
        data_dir,
        output_dir,
        "obb",
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        val_split=val_split,
        imgsz=imgsz,
        model=model,
        weights=weights,
        patience=patience,
        mosaic=mosaic,
    )


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
    patience: int = 100,
    mosaic: float = 1.0,
) -> Path:
    """Train a YOLO-seg instance segmentation model.

    Convenience wrapper around :func:`train_yolo` with segmentation defaults.
    See :func:`train_yolo` for full parameter documentation.
    """
    return train_yolo(
        data_dir,
        output_dir,
        "seg",
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        val_split=val_split,
        imgsz=imgsz,
        model=model,
        weights=weights,
        patience=patience,
        mosaic=mosaic,
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
    patience: int = 100,
    mosaic: float = 1.0,
    rect: bool = True,
) -> Path:
    """Train a YOLO-pose keypoint estimation model.

    Convenience wrapper around :func:`train_yolo` with pose defaults.
    See :func:`train_yolo` for full parameter documentation.
    """
    return train_yolo(
        data_dir,
        output_dir,
        "pose",
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        val_split=val_split,
        imgsz=imgsz,
        model=model,
        weights=weights,
        patience=patience,
        mosaic=mosaic,
        rect=rect,
    )
