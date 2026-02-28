"""AquaPose CLI -- thin wrapper over PosePipeline."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import click
import yaml

from aquapose.engine import (
    PipelineConfig,
    PosePipeline,
    build_observers,
    load_config,
)
from aquapose.engine.pipeline import build_stages

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
@click.option(
    "--output",
    "-o",
    default="aquapose.yaml",
    type=click.Path(),
    help="Output file path (default: aquapose.yaml).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing file.",
)
def init_config(output: str, force: bool) -> None:
    """Generate a default template YAML config file with all pipeline defaults."""
    output_path = Path(output)
    if output_path.exists() and not force:
        raise click.ClickException(
            f"'{output}' already exists. Use --force to overwrite."
        )
    config = PipelineConfig()
    data = dataclasses.asdict(config)
    for key in ("run_id", "output_dir"):
        data.pop(key, None)
    yaml_content = yaml.dump(data, default_flow_style=False, sort_keys=True)
    output_path.write_text(yaml_content)
    click.echo(f"Config written to {output}")


def main() -> None:
    """Entry point for the ``aquapose`` console script."""
    cli()
