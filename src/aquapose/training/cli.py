"""CLI group and subcommands for training AquaPose models."""

from __future__ import annotations

from pathlib import Path

import click


def _parse_input_size(value: str) -> tuple[int, int]:
    """Parse a ``WxH`` string into a ``(width, height)`` integer tuple.

    Args:
        value: Input size string, e.g. ``"128x64"``.

    Returns:
        ``(width, height)`` tuple.

    Raises:
        click.BadParameter: If the string cannot be parsed.
    """
    try:
        w_str, h_str = value.lower().split("x")
        return (int(w_str), int(h_str))
    except (ValueError, AttributeError) as err:
        raise click.BadParameter(
            f"Invalid input size {value!r}. Expected format: WxH (e.g. 128x64)"
        ) from err


@click.group("train")
def train_group() -> None:
    """Train AquaPose models."""


@train_group.command("unet")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory with annotations.json and images.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory for model weights and metrics.",
)
@click.option("--epochs", default=100, type=int, help="Number of training epochs.")
@click.option("--batch-size", default=8, type=int, help="Batch size.")
@click.option("--lr", default=1e-4, type=float, help="Learning rate.")
@click.option("--val-split", default=0.2, type=float, help="Validation split fraction.")
@click.option(
    "--patience",
    default=20,
    type=int,
    help="Early stopping patience (0=disabled).",
)
@click.option(
    "--device", default=None, type=str, help="Torch device (auto-detect if omitted)."
)
@click.option("--num-workers", default=4, type=int, help="DataLoader workers.")
@click.option(
    "--input-size",
    default="128x64",
    type=str,
    help="Input size as WxH (e.g. 128x64).",
)
def unet(
    data_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_split: float,
    patience: int,
    device: str | None,
    num_workers: int,
    input_size: str,
) -> None:
    """Train U-Net binary segmentation model."""
    from .unet import train_unet

    parsed_size = _parse_input_size(input_size)
    best_path = train_unet(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_split=val_split,
        patience=patience,
        num_workers=num_workers,
        device=device,
        input_size=parsed_size,
    )
    click.echo(f"Training complete. Best model: {best_path}")


@train_group.command("yolo-obb")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory with data.yaml and YOLO-format dataset.",
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
    "--model-size",
    default="s",
    type=click.Choice(["n", "s", "m", "l", "x"]),
    help="YOLO model size (yolov8{size}-obb.pt).",
)
def yolo_obb(
    data_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    device: str | None,
    val_split: float,
    imgsz: int,
    model_size: str,
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
        model_size=model_size,
    )
    click.echo(f"Training complete. Best model: {best_path}")


@train_group.command("pose")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory with annotations.json and images.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory for model weights and metrics.",
)
@click.option("--epochs", default=100, type=int, help="Number of training epochs.")
@click.option("--batch-size", default=8, type=int, help="Batch size.")
@click.option(
    "--lr", default=1e-4, type=float, help="Learning rate for regression head."
)
@click.option("--val-split", default=0.2, type=float, help="Validation split fraction.")
@click.option(
    "--patience",
    default=20,
    type=int,
    help="Early stopping patience (0=disabled).",
)
@click.option(
    "--device", default=None, type=str, help="Torch device (auto-detect if omitted)."
)
@click.option("--num-workers", default=4, type=int, help="DataLoader workers.")
@click.option(
    "--backbone-weights",
    default=None,
    type=click.Path(exists=True),
    help="Path to U-Net weights for encoder transfer learning.",
)
@click.option(
    "--unfreeze",
    is_flag=True,
    default=False,
    help="Unfreeze encoder for end-to-end fine-tuning (requires --backbone-weights).",
)
@click.option(
    "--n-keypoints", default=6, type=int, help="Number of anatomical keypoints."
)
@click.option(
    "--input-size",
    default="128x64",
    type=str,
    help="Input size as WxH (e.g. 128x64).",
)
def pose(
    data_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_split: float,
    patience: int,
    device: str | None,
    num_workers: int,
    backbone_weights: str | None,
    unfreeze: bool,
    n_keypoints: int,
    input_size: str,
) -> None:
    """Train pose/keypoint regression model."""
    from .pose import train_pose

    parsed_size = _parse_input_size(input_size)
    best_path = train_pose(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_split=val_split,
        patience=patience,
        num_workers=num_workers,
        device=device,
        backbone_weights=Path(backbone_weights) if backbone_weights else None,
        unfreeze=unfreeze,
        n_keypoints=n_keypoints,
        input_size=parsed_size,
    )
    click.echo(f"Training complete. Best model: {best_path}")
