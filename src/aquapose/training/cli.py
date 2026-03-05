"""CLI group and subcommands for training AquaPose models."""

from __future__ import annotations

from pathlib import Path

import click


@click.group("train")
def train_group() -> None:
    """Train AquaPose models."""


@train_group.command("yolo-obb")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing dataset.yaml file.",
)
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Project config YAML path.",
)
@click.option(
    "--tag",
    default=None,
    type=str,
    help="Human-readable tag for this run (e.g. 'round2-high-conf').",
)
@click.option("--epochs", default=100, type=int, help="Number of training epochs.")
@click.option("--batch-size", default=16, type=int, help="Batch size.")
@click.option(
    "--device", default=None, type=str, help="Torch device (auto-detect if omitted)."
)
@click.option("--val-split", default=0.2, type=float, help="Validation split fraction.")
@click.option("--imgsz", default=640, type=int, help="Training image size (square).")
@click.option(
    "--model",
    default="yolo26n-obb",
    type=str,
    help="YOLO model variant (e.g. yolo26n-obb, yolo26s-obb).",
)
@click.option(
    "--weights",
    default=None,
    type=click.Path(exists=True),
    help="Pretrained weights for transfer learning.",
)
@click.option(
    "--patience", default=100, type=int, help="Early-stopping patience in epochs."
)
@click.option(
    "--mosaic",
    default=1.0,
    type=float,
    help="Mosaic augmentation probability (0.0 to disable).",
)
def yolo_obb(
    data_dir: str,
    config: str,
    tag: str | None,
    epochs: int,
    batch_size: int,
    device: str | None,
    val_split: float,
    imgsz: int,
    model: str,
    weights: str | None,
    patience: int,
    mosaic: float,
) -> None:
    """Train YOLO-OBB oriented bounding-box detection model."""
    from aquapose.logging import setup_file_logging

    from .run_manager import (
        create_run_dir,
        print_next_steps,
        snapshot_config,
        write_summary,
    )

    model_type = "obb"
    run_dir = create_run_dir(Path(config), model_type)
    setup_file_logging(run_dir, "train-yolo-obb")

    cli_args = {
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "val_split": val_split,
        "imgsz": imgsz,
        "model": model,
        "weights": str(weights) if weights else None,
        "mosaic": mosaic,
        "patience": patience,
        "data_dir": data_dir,
        "tag": tag,
    }
    snapshot_config(run_dir, cli_args, dataset_dir=Path(data_dir))

    from .yolo_obb import train_yolo_obb

    best_path = train_yolo_obb(
        data_dir=Path(data_dir),
        output_dir=run_dir,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        val_split=val_split,
        imgsz=imgsz,
        model=model,
        weights=Path(weights) if weights is not None else None,
        patience=patience,
        mosaic=mosaic,
    )

    results_csv = run_dir / "_ultralytics" / "train" / "results.csv"
    write_summary(
        run_dir,
        results_csv,
        training_args=cli_args,
        model_type=model_type,
        tag=tag,
        dataset_dir=Path(data_dir),
    )
    print_next_steps(run_dir, model_type, best_path)


@train_group.command("seg")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing dataset.yaml file.",
)
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Project config YAML path.",
)
@click.option(
    "--tag",
    default=None,
    type=str,
    help="Human-readable tag for this run (e.g. 'round2-high-conf').",
)
@click.option("--epochs", default=100, type=int, help="Number of training epochs.")
@click.option("--batch-size", default=16, type=int, help="Batch size.")
@click.option(
    "--device", default=None, type=str, help="Torch device (auto-detect if omitted)."
)
@click.option("--val-split", default=0.2, type=float, help="Validation split fraction.")
@click.option("--imgsz", default=640, type=int, help="Training image size (square).")
@click.option(
    "--model",
    default="yolo26n-seg",
    type=str,
    help="YOLO model variant (e.g. yolo26n-seg, yolo26s-seg).",
)
@click.option(
    "--weights",
    default=None,
    type=click.Path(exists=True),
    help="Pretrained weights for transfer learning.",
)
@click.option(
    "--patience", default=100, type=int, help="Early-stopping patience in epochs."
)
@click.option(
    "--mosaic",
    default=1.0,
    type=float,
    help="Mosaic augmentation probability (0.0 to disable).",
)
def seg(
    data_dir: str,
    config: str,
    tag: str | None,
    epochs: int,
    batch_size: int,
    device: str | None,
    val_split: float,
    imgsz: int,
    model: str,
    weights: str | None,
    patience: int,
    mosaic: float,
) -> None:
    """Train YOLO-seg instance segmentation model."""
    from aquapose.logging import setup_file_logging

    from .run_manager import (
        create_run_dir,
        print_next_steps,
        snapshot_config,
        write_summary,
    )

    model_type = "seg"
    run_dir = create_run_dir(Path(config), model_type)
    setup_file_logging(run_dir, "train-seg")

    cli_args = {
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "val_split": val_split,
        "imgsz": imgsz,
        "model": model,
        "weights": str(weights) if weights else None,
        "mosaic": mosaic,
        "patience": patience,
        "data_dir": data_dir,
        "tag": tag,
    }
    snapshot_config(run_dir, cli_args, dataset_dir=Path(data_dir))

    from .yolo_seg import train_yolo_seg

    best_path = train_yolo_seg(
        data_dir=Path(data_dir),
        output_dir=run_dir,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        val_split=val_split,
        imgsz=imgsz,
        model=model,
        weights=Path(weights) if weights is not None else None,
        patience=patience,
        mosaic=mosaic,
    )

    results_csv = run_dir / "_ultralytics" / "train" / "results.csv"
    write_summary(
        run_dir,
        results_csv,
        training_args=cli_args,
        model_type=model_type,
        tag=tag,
        dataset_dir=Path(data_dir),
    )
    print_next_steps(run_dir, model_type, best_path)


@train_group.command("pose")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing dataset.yaml file.",
)
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Project config YAML path.",
)
@click.option(
    "--tag",
    default=None,
    type=str,
    help="Human-readable tag for this run (e.g. 'round2-high-conf').",
)
@click.option("--epochs", default=100, type=int, help="Number of training epochs.")
@click.option("--batch-size", default=16, type=int, help="Batch size.")
@click.option(
    "--device", default=None, type=str, help="Torch device (auto-detect if omitted)."
)
@click.option("--val-split", default=0.2, type=float, help="Validation split fraction.")
@click.option("--imgsz", default=640, type=int, help="Training image size (square).")
@click.option(
    "--model",
    default="yolo26n-pose",
    type=str,
    help="YOLO model variant (e.g. yolo26n-pose, yolo26s-pose).",
)
@click.option(
    "--weights",
    default=None,
    type=click.Path(exists=True),
    help="Pretrained weights for transfer learning.",
)
@click.option(
    "--patience", default=100, type=int, help="Early-stopping patience in epochs."
)
@click.option(
    "--mosaic",
    default=1.0,
    type=float,
    help="Mosaic augmentation probability (0.0 to disable).",
)
def pose(
    data_dir: str,
    config: str,
    tag: str | None,
    epochs: int,
    batch_size: int,
    device: str | None,
    val_split: float,
    imgsz: int,
    model: str,
    weights: str | None,
    patience: int,
    mosaic: float,
) -> None:
    """Train YOLO-pose keypoint estimation model."""
    from aquapose.logging import setup_file_logging

    from .run_manager import (
        create_run_dir,
        print_next_steps,
        snapshot_config,
        write_summary,
    )

    model_type = "pose"
    run_dir = create_run_dir(Path(config), model_type)
    setup_file_logging(run_dir, "train-pose")

    cli_args = {
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "val_split": val_split,
        "imgsz": imgsz,
        "model": model,
        "weights": str(weights) if weights else None,
        "mosaic": mosaic,
        "patience": patience,
        "data_dir": data_dir,
        "tag": tag,
    }
    snapshot_config(run_dir, cli_args, dataset_dir=Path(data_dir))

    from .yolo_pose import train_yolo_pose

    best_path = train_yolo_pose(
        data_dir=Path(data_dir),
        output_dir=run_dir,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        val_split=val_split,
        imgsz=imgsz,
        model=model,
        weights=Path(weights) if weights is not None else None,
        patience=patience,
        mosaic=mosaic,
    )

    results_csv = run_dir / "_ultralytics" / "train" / "results.csv"
    write_summary(
        run_dir,
        results_csv,
        training_args=cli_args,
        model_type=model_type,
        tag=tag,
        dataset_dir=Path(data_dir),
    )
    print_next_steps(run_dir, model_type, best_path)
