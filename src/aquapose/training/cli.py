"""CLI group and subcommands for training AquaPose models."""

from __future__ import annotations

from pathlib import Path

import click


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
) -> None:
    """Train U-Net binary segmentation model."""
    from .unet import train_unet

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
    )
    click.echo(f"Training complete. Best model: {best_path}")
