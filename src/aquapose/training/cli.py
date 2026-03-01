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
    help="Directory with data.yaml and NDJSON OBB dataset.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory for model weights and metrics.",
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
    default="yolov8s-obb",
    type=str,
    help="YOLO model variant (e.g. yolov8s-obb, yolov8n-obb).",
)
@click.option(
    "--weights",
    default=None,
    type=click.Path(exists=True),
    help="Pretrained weights for transfer learning.",
)
def yolo_obb(
    data_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    device: str | None,
    val_split: float,
    imgsz: int,
    model: str,
    weights: str | None,
) -> None:
    """Train YOLO-OBB oriented bounding-box detection model."""
    from .yolo_obb import train_yolo_obb

    best_path = train_yolo_obb(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        val_split=val_split,
        imgsz=imgsz,
        model=model,
        weights=Path(weights) if weights is not None else None,
    )
    click.echo(f"Training complete. Best model: {best_path}")


@train_group.command("seg")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory with data.yaml and NDJSON seg dataset.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory for model weights and metrics.",
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
def seg(
    data_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    device: str | None,
    val_split: float,
    imgsz: int,
    model: str,
    weights: str | None,
) -> None:
    """Train YOLO-seg instance segmentation model."""
    from .yolo_seg import train_yolo_seg

    best_path = train_yolo_seg(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        val_split=val_split,
        imgsz=imgsz,
        model=model,
        weights=Path(weights) if weights is not None else None,
    )
    click.echo(f"Training complete. Best model: {best_path}")


@train_group.command("pose")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory with data.yaml and NDJSON pose dataset.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory for model weights and metrics.",
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
def pose(
    data_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    device: str | None,
    val_split: float,
    imgsz: int,
    model: str,
    weights: str | None,
) -> None:
    """Train YOLO-pose keypoint estimation model."""
    from .yolo_pose import train_yolo_pose

    best_path = train_yolo_pose(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        val_split=val_split,
        imgsz=imgsz,
        model=model,
        weights=Path(weights) if weights is not None else None,
    )
    click.echo(f"Training complete. Best model: {best_path}")
