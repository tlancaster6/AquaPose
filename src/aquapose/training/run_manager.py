"""Training run directory management and provenance tracking."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import click
import yaml


def resolve_project_dir(config_path: Path) -> Path:
    """Extract project_dir from a project config YAML.

    Args:
        config_path: Path to the project config YAML file.

    Returns:
        Resolved absolute path to the project directory.

    Raises:
        ValueError: If ``project_dir`` is not set in the config.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    project_dir = config.get("project_dir", "")
    if not project_dir:
        raise ValueError(f"project_dir not set in {config_path}")
    return Path(project_dir).expanduser().resolve()


def create_run_dir(project_config_path: Path, model_type: str) -> Path:
    """Create a timestamped run directory under the project training tree.

    Creates ``{project_dir}/training/{model_type}/run_{timestamp}/``.

    Args:
        project_config_path: Path to the project config YAML.
        model_type: Model type string (``"obb"``, ``"seg"``, or ``"pose"``).

    Returns:
        Path to the newly created run directory.

    Raises:
        ValueError: If ``project_dir`` is not set in the config.
    """
    project_dir = resolve_project_dir(project_config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_dir / "training" / model_type / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def snapshot_config(
    run_dir: Path,
    cli_args: dict,
    dataset_dir: Path | None = None,
) -> None:
    """Write frozen training config and copy dataset sidecars to run directory.

    Writes ``cli_args`` as ``config.yaml`` via ``yaml.dump``. If ``dataset_dir``
    is provided, copies ``confidence.json`` and ``dataset.yaml`` into the run
    directory as ``dataset_confidence.json`` and ``dataset_dataset.yaml``
    (skipping any that do not exist).

    Args:
        run_dir: Run directory to write config into.
        cli_args: Dictionary of CLI arguments to snapshot.
        dataset_dir: Optional path to the assembled dataset directory.
    """
    config_path = run_dir / "config.yaml"
    config_path.write_text(
        yaml.dump(cli_args, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    if dataset_dir is not None:
        for sidecar in ("confidence.json", "dataset.yaml"):
            src = dataset_dir / sidecar
            if src.exists():
                shutil.copy2(src, run_dir / f"dataset_{sidecar}")


def parse_best_metrics(results_csv: Path) -> dict:
    """Extract best-epoch metrics from an Ultralytics results.csv.

    Finds the epoch with the highest ``metrics/mAP50-95(B)`` value and returns
    its key metrics. Strips whitespace from CSV headers to handle the
    Ultralytics formatting quirk.

    Args:
        results_csv: Path to the Ultralytics ``results.csv`` file.

    Returns:
        Dictionary with ``best_epoch``, ``mAP50``, ``mAP50-95``,
        ``precision``, ``recall``, and ``total_time``. Returns an empty
        dict if the CSV has no data rows.
    """
    rows: list[dict[str, str]] = []
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            rows.append(cleaned)

    if not rows:
        return {}

    best_row = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0)))

    return {
        "best_epoch": int(float(best_row.get("epoch", 0))),
        "mAP50": float(best_row.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(best_row.get("metrics/mAP50-95(B)", 0)),
        "precision": float(best_row.get("metrics/precision(B)", 0)),
        "recall": float(best_row.get("metrics/recall(B)", 0)),
        "total_time": sum(float(r.get("time", 0)) for r in rows),
    }


def extract_dataset_provenance(dataset_dir: Path) -> dict:
    """Extract source breakdown from assembled dataset metadata.

    Reads ``confidence.json`` and ``pseudo_val_metadata.json`` from the
    dataset directory to determine the pseudo-label round and thresholds used.

    Args:
        dataset_dir: Path to the assembled dataset directory.

    Returns:
        Dictionary with provenance information including thresholds,
        pipeline run, image counts, and source breakdown.
    """
    provenance: dict = {}

    # Read confidence.json for threshold info
    confidence_path = dataset_dir / "confidence.json"
    if confidence_path.exists():
        with open(confidence_path) as f:
            confidence_data = json.load(f)
        provenance.update(confidence_data)

    # Count training images
    train_images_dir = dataset_dir / "images" / "train"
    if train_images_dir.exists():
        provenance["n_train_images"] = len(list(train_images_dir.glob("*.jpg")))

    # Read pseudo_val_metadata.json for source breakdown
    meta_path = dataset_dir / "pseudo_val_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        sources = [entry.get("source", "unknown") for entry in meta.values()]
        provenance["n_consensus"] = sources.count("consensus")
        provenance["n_gap"] = sources.count("gap")

    return provenance


def write_summary(
    run_dir: Path,
    results_csv_path: Path,
    training_args: dict,
    model_type: str,
    tag: str | None = None,
    dataset_dir: Path | None = None,
) -> None:
    """Parse training results and write a pretty-printed summary.json.

    Creates ``summary.json`` in the run directory with run metadata,
    best-epoch metrics, training configuration, and dataset provenance.

    Args:
        run_dir: Run directory to write summary into.
        results_csv_path: Path to the Ultralytics ``results.csv``.
        training_args: Dictionary of CLI arguments used for training.
        model_type: Model type string (``"obb"``, ``"seg"``, or ``"pose"``).
        tag: Optional human-readable tag for this run.
        dataset_dir: Optional dataset directory for provenance extraction.
    """
    best_metrics = parse_best_metrics(results_csv_path)

    dataset_sources: dict = {}
    if dataset_dir is not None:
        dataset_sources = extract_dataset_provenance(dataset_dir)

    summary = {
        "run_id": run_dir.name,
        "tag": tag,
        "model_type": model_type,
        "model_variant": training_args.get("model", ""),
        "parent_weights": str(training_args.get("weights", "")) or None,
        "dataset_path": str(training_args.get("data_dir", "")),
        "dataset_sources": dataset_sources,
        "training_config": {
            k: v
            for k, v in training_args.items()
            if k in ("epochs", "batch_size", "imgsz", "patience", "mosaic", "val_split")
        },
        "metrics": {
            "best_epoch": best_metrics.get("best_epoch", -1),
            "mAP50": best_metrics.get("mAP50", 0),
            "mAP50-95": best_metrics.get("mAP50-95", 0),
            "precision": best_metrics.get("precision", 0),
            "recall": best_metrics.get("recall", 0),
        },
        "training_duration_seconds": best_metrics.get("total_time", 0),
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def update_config_weights(
    config_path: Path,
    model_type: str,
    weights_path: Path,
) -> None:
    """Update the project config YAML with new model weights path.

    Maps model_type to config section: ``"obb"`` -> ``"detection"``,
    ``"pose"`` -> ``"midline"``, ``"seg"`` -> ``"segmentation"``.

    Args:
        config_path: Path to the project config YAML file.
        model_type: Model type string (``"obb"``, ``"pose"``, ``"seg"``).
        weights_path: Path to the new best weights file.
    """
    section_map = {
        "obb": "detection",
        "pose": "midline",
        "seg": "segmentation",
    }
    section = section_map.get(model_type, model_type)

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    if section not in config:
        config[section] = {}
    config[section]["weights_path"] = str(weights_path)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    click.echo(
        click.style(
            f"Updated {config_path}: {section}.weights_path = {weights_path}",
            bold=True,
            fg="green",
        )
    )


def register_trained_model(
    config_path: Path,
    run_dir: Path,
    model_type: str,
    best_weights: Path,
    dataset_dir: Path | None = None,
    tag: str | None = None,
) -> None:
    """Register a trained model in the store and update config.

    Opens (or auto-creates) the SampleStore for the model type,
    reads metrics from ``summary.json`` if available, registers the
    model, and updates the project config with the new weights path.

    Args:
        config_path: Path to the project config YAML file.
        run_dir: Path to the completed training run directory.
        model_type: Model type string (``"obb"``, ``"pose"``, ``"seg"``).
        best_weights: Path to the best model weights file.
        dataset_dir: Optional dataset directory for lineage tracking.
        tag: Optional human-readable tag for this model.
    """
    from .store import SampleStore

    project_dir = resolve_project_dir(config_path)
    store_path = project_dir / "training_data" / model_type / "store.db"

    # Read metrics from summary.json if it exists
    metrics: dict | None = None
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary_data = json.load(f)
        metrics = summary_data.get("metrics")

    # Determine dataset_name from dataset_dir (use basename regardless of
    # whether it's a store-managed dataset or an external directory)
    dataset_name: str | None = None
    if dataset_dir is not None:
        dataset_name = dataset_dir.name

    # Register in store
    with SampleStore(store_path) as store:
        store.register_model(
            run_id=run_dir.name,
            weights_path=str(best_weights),
            model_type=model_type,
            metrics=metrics,
            dataset_name=dataset_name,
            tag=tag,
        )

    # Update config
    update_config_weights(config_path, model_type, best_weights)


def print_next_steps(run_dir: Path, model_type: str, best_weights: Path) -> None:
    """Print suggested next steps after training completes.

    Args:
        run_dir: Path to the completed run directory.
        model_type: Model type string (``"obb"``, ``"seg"``, or ``"pose"``).
        best_weights: Path to the best model weights file.
    """
    click.echo("")
    click.echo(click.style("Training complete!", bold=True))
    click.echo(f"  Run directory: {run_dir}")
    click.echo(f"  Best weights:  {best_weights}")
    click.echo(f"  Model registered as: {run_dir.name}")
    click.echo("")
    click.echo("Next steps:")
    config_path = run_dir.parent.parent.parent / "config.yaml"
    click.echo(
        f"  Compare runs:  aquapose train compare --config {config_path}"
        f" --model-type {model_type}"
    )
    click.echo(
        f"  Retrain:       aquapose train {model_type}"
        f" --config ... --weights {best_weights}"
    )
