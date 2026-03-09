"""CLI group and subcommands for training AquaPose models."""

from __future__ import annotations

from pathlib import Path

import click

from aquapose.cli_utils import get_config_path, get_project_dir


@click.group("train")
def train_group() -> None:
    """Train AquaPose models."""


def _run_training(
    ctx: click.Context,
    model_type: str,
    data_dir: str | None,
    tag: str | None,
    cli_args: dict,
    *,
    train_kwargs: dict | None = None,
) -> None:
    """Shared training orchestrator for all YOLO model types.

    Handles the full lifecycle: run directory creation, config snapshot,
    training, summary writing, model registration, and next-steps output.

    Args:
        ctx: Click context with project/config resolution.
        model_type: One of ``"obb"``, ``"seg"``, ``"pose"``.
        data_dir: Dataset directory path, or None to use project default.
        tag: Human-readable run tag, or None.
        cli_args: Dict of CLI option values for config snapshot.
        train_kwargs: Additional keyword arguments forwarded to
            :func:`~aquapose.training.yolo_training.train_yolo` beyond
            those in ``cli_args``.
    """
    from aquapose.logging import setup_file_logging

    from .run_manager import (
        create_run_dir,
        print_next_steps,
        register_trained_model,
        snapshot_config,
        write_summary,
    )
    from .yolo_training import train_yolo

    config_path = get_config_path(ctx)
    project_dir = get_project_dir(ctx)

    if data_dir is None:
        data_dir = str(project_dir / "training_data" / model_type)

    run_dir = create_run_dir(config_path, model_type)
    setup_file_logging(run_dir, f"train-{model_type}")

    cli_args["data_dir"] = data_dir
    snapshot_config(run_dir, cli_args, dataset_dir=Path(data_dir))

    # Build train_yolo kwargs from cli_args (exclude non-training keys)
    excluded_keys = {"data_dir", "tag"}
    kwargs = {k: v for k, v in cli_args.items() if k not in excluded_keys}
    # Convert weights string to Path if present
    if kwargs.get("weights") is not None:
        kwargs["weights"] = Path(kwargs["weights"])
    if train_kwargs:
        kwargs.update(train_kwargs)

    best_path = train_yolo(
        data_dir=Path(data_dir),
        output_dir=run_dir,
        model_type=model_type,
        **kwargs,
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

    try:
        register_trained_model(
            config_path=config_path,
            run_dir=run_dir,
            model_type=model_type,
            best_weights=best_path,
            dataset_dir=Path(data_dir),
            tag=tag,
        )
    except Exception as exc:
        click.echo(
            click.style(
                f"Warning: Model registration failed: {exc}",
                fg="yellow",
            )
        )

    print_next_steps(run_dir, model_type, best_path)


@train_group.command("obb")
@click.option(
    "--data-dir",
    default=None,
    type=click.Path(exists=True),
    help="Directory containing dataset.yaml file. Defaults to project training_data/obb/.",
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
    default=0.3,
    type=float,
    help="Mosaic augmentation probability (0.0 to disable).",
)
@click.pass_context
def yolo_obb(
    ctx: click.Context,
    data_dir: str | None,
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
        "tag": tag,
    }
    _run_training(ctx, "obb", data_dir, tag, cli_args)


@train_group.command("compare")
@click.option(
    "--model-type",
    required=True,
    type=click.Choice(["obb", "seg", "pose"]),
    help="Model type to compare runs for.",
)
@click.option(
    "--csv",
    "csv_path",
    default=None,
    type=click.Path(),
    help="Export comparison to CSV file.",
)
@click.argument("run_paths", nargs=-1, type=click.Path(exists=True))
@click.pass_context
def compare(
    ctx: click.Context,
    model_type: str,
    csv_path: str | None,
    run_paths: tuple[str, ...],
) -> None:
    """Compare training runs side-by-side."""
    from .compare import (
        discover_runs,
        format_comparison_table,
        load_run_summaries,
        write_comparison_csv,
    )

    if run_paths:
        run_dirs = [Path(p) for p in run_paths]
    else:
        project_dir = get_project_dir(ctx)
        run_dirs = discover_runs(project_dir / "training" / model_type)

    summaries = load_run_summaries(run_dirs)
    table = format_comparison_table(summaries)
    click.echo(table)

    if csv_path is not None:
        write_comparison_csv(summaries, Path(csv_path))
        click.echo(f"\nCSV exported to {csv_path}")


@train_group.command("seg")
@click.option(
    "--data-dir",
    default=None,
    type=click.Path(exists=True),
    help="Directory containing dataset.yaml file. Defaults to project training_data/seg/.",
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
@click.pass_context
def seg(
    ctx: click.Context,
    data_dir: str | None,
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
        "tag": tag,
    }
    _run_training(ctx, "seg", data_dir, tag, cli_args)


@train_group.command("pose")
@click.option(
    "--data-dir",
    default=None,
    type=click.Path(exists=True),
    help="Directory containing dataset.yaml file. Defaults to project training_data/pose/.",
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
@click.option("--imgsz", default=320, type=int, help="Training image size (square).")
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
    default=0.1,
    type=float,
    help="Mosaic augmentation probability (0.0 to disable).",
)
@click.option(
    "--rect/--no-rect",
    default=True,
    help="Use rectangular training batches.",
)
@click.pass_context
def pose(
    ctx: click.Context,
    data_dir: str | None,
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
    rect: bool,
) -> None:
    """Train YOLO-pose keypoint estimation model."""
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
        "tag": tag,
        "rect": rect,
    }
    _run_training(ctx, "pose", data_dir, tag, cli_args)
