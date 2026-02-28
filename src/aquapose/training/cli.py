"""CLI group and subcommands for training AquaPose models."""

from __future__ import annotations

import click


@click.group("train")
def train_group() -> None:
    """Train AquaPose models."""
