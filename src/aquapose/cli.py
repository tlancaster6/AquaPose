"""AquaPose CLI -- thin wrapper over PosePipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click
import yaml

from aquapose.engine import (
    PosePipeline,
    build_observers,
    load_config,
)
from aquapose.engine.pipeline import build_stages
from aquapose.training.cli import train_group
from aquapose.training.prep import prep_group

# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """AquaPose -- 3D fish pose estimation via refractive multi-view triangulation."""


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to pipeline config YAML.",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(
        ["production", "diagnostic", "benchmark", "synthetic"], case_sensitive=False
    ),
    default=None,
    help="Execution mode preset (default: from config, or production).",
)
@click.option(
    "--set",
    "overrides",
    multiple=True,
    help="Config override as key=val (e.g. --set detection.detector_kind=mog2).",
)
@click.option(
    "--add-observer",
    "extra_observers",
    multiple=True,
    type=click.Choice(
        ["timing", "hdf5", "overlay2d", "animation3d", "diagnostic", "console"],
        case_sensitive=False,
    ),
    help="Add observer by name (additive).",
)
@click.option(
    "--stop-after",
    "stop_after",
    type=click.Choice(
        ["detection", "tracking", "association", "midline"], case_sensitive=False
    ),
    default=None,
    help="Stop pipeline after the named stage (skip later stages).",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose output.")
def run(
    config: str,
    mode: str | None,
    overrides: tuple[str, ...],
    extra_observers: tuple[str, ...],
    stop_after: str | None,
    verbose: bool,
) -> None:
    """Run the AquaPose pipeline."""
    # 1. Parse --set overrides into dict
    cli_overrides: dict[str, Any] = {}
    for item in overrides:
        key, _, value = item.partition("=")
        if not key:
            continue
        # Coerce numeric/boolean strings to native types
        if value.lower() in ("true", "false"):
            cli_overrides[key] = value.lower() == "true"
        else:
            try:
                cli_overrides[key] = int(value)
            except ValueError:
                try:
                    cli_overrides[key] = float(value)
                except ValueError:
                    cli_overrides[key] = value

    # 2. Inject mode and stop_after into overrides only if explicitly provided
    if mode is not None:
        cli_overrides["mode"] = mode
    if stop_after is not None:
        cli_overrides["stop_after"] = stop_after

    # 3. Load config
    pipeline_config = load_config(yaml_path=config, cli_overrides=cli_overrides)

    # 4. Resolve effective mode (CLI > YAML > default)
    effective_mode = pipeline_config.mode

    # 5. Build stages
    stages = build_stages(pipeline_config)

    # 6. Assemble observers
    observers = build_observers(
        config=pipeline_config,
        mode=effective_mode,
        verbose=verbose,
        total_stages=len(stages),
        extra_observers=extra_observers,
    )

    # 7. Create and run pipeline
    pipeline = PosePipeline(
        stages=stages,
        config=pipeline_config,
        observers=observers,
    )

    try:
        pipeline.run()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


@cli.command("init-config")
@click.argument("name")
@click.option(
    "--synthetic",
    is_flag=True,
    default=False,
    help="Include a synthetic section in the generated config.",
)
def init_config(name: str, synthetic: bool) -> None:
    """Create a new AquaPose project directory scaffold with a starter config.

    NAME is the project name. Creates ~/aquapose/projects/<name>/ with
    config.yaml and subdirectories: runs/, models/, geometry/, videos/.
    """
    project_dir = Path("~/aquapose/projects").expanduser() / name
    if project_dir.exists():
        raise click.ClickException(f"'{project_dir}' already exists.")

    # Create directory structure
    project_dir.mkdir(parents=True, exist_ok=False)
    for subdir in ("runs", "models", "geometry", "videos"):
        (project_dir / subdir).mkdir()

    # Build ordered config dict (user-relevant order, not alphabetical)
    data: dict[str, Any] = {}
    # --- Paths (edited first by users) ---
    data["project_dir"] = str(project_dir)
    data["video_dir"] = "videos/"
    data["calibration_path"] = "geometry/calibration.json"
    data["output_dir"] = "runs/"
    # --- Core parameters ---
    data["n_animals"] = "SET_ME"  # required -- must be an integer
    # --- Detection ---
    data["detection"] = {"detector_kind": "yolo", "model_path": "models/best.pt"}
    # --- Midline ---
    data["midline"] = {"weights_path": "models/unet_best.pth"}
    # --- Synthetic (only with --synthetic) ---
    if synthetic:
        data["synthetic"] = {"frame_count": 30, "noise_std": 0.0, "seed": 42}

    # Write config.yaml with brief comment header
    header = "# AquaPose pipeline config\n# See documentation for advanced options\n\n"
    yaml_content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    (project_dir / "config.yaml").write_text(header + yaml_content)

    click.echo(f"Project created at {project_dir}")


cli.add_command(train_group)
cli.add_command(prep_group)


def main() -> None:
    """Entry point for the ``aquapose`` console script."""
    cli()
