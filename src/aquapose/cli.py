"""AquaPose CLI -- thin wrapper over PosePipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click

from aquapose.engine import (
    Animation3DObserver,
    ConsoleObserver,
    DiagnosticObserver,
    HDF5ExportObserver,
    Overlay2DObserver,
    PipelineConfig,
    PosePipeline,
    TimingObserver,
    load_config,
    serialize_config,
)
from aquapose.engine.observers import Observer
from aquapose.engine.pipeline import build_stages

# ---------------------------------------------------------------------------
# Observer name -> factory mapping
# ---------------------------------------------------------------------------

_OBSERVER_MAP: dict[str, type] = {
    "timing": TimingObserver,
    "hdf5": HDF5ExportObserver,
    "overlay2d": Overlay2DObserver,
    "animation3d": Animation3DObserver,
    "diagnostic": DiagnosticObserver,
    "console": ConsoleObserver,
}


def _build_observers(
    config: PipelineConfig,
    mode: str,
    verbose: bool,
    total_stages: int,
    extra_observers: tuple[str, ...],
) -> list[Observer]:
    """Assemble the observer list based on execution mode and additive flags.

    Args:
        config: Pipeline configuration (used for output paths).
        mode: Execution mode (production, diagnostic, benchmark, synthetic).
        verbose: Whether to enable verbose console output.
        total_stages: Number of stages in the pipeline.
        extra_observers: Additional observer names from --add-observer flags.

    Returns:
        List of Observer instances for the pipeline.
    """
    observers: list[Observer] = [
        ConsoleObserver(verbose=verbose, total_stages=total_stages),
    ]

    output_dir = Path(config.output_dir)

    if mode in ("production", "synthetic"):
        observers.append(TimingObserver(output_path=output_dir / "timing.txt"))
        observers.append(HDF5ExportObserver(output_dir=config.output_dir))

    elif mode == "diagnostic":
        observers.append(TimingObserver(output_path=output_dir / "timing.txt"))
        observers.append(HDF5ExportObserver(output_dir=config.output_dir))
        observers.append(
            Overlay2DObserver(
                output_dir=config.output_dir,
                video_dir=config.video_dir,
                calibration_path=config.calibration_path,
            )
        )
        observers.append(Animation3DObserver(output_dir=config.output_dir))
        observers.append(DiagnosticObserver())

    elif mode == "benchmark":
        observers.append(TimingObserver(output_path=output_dir / "timing.txt"))

    # Additive --add-observer flags
    for name in extra_observers:
        cls = _OBSERVER_MAP.get(name)
        if cls is None:
            continue
        # Instantiate with sensible defaults based on observer type
        if cls is TimingObserver:
            observers.append(TimingObserver(output_path=output_dir / "timing.txt"))
        elif cls is HDF5ExportObserver:
            observers.append(HDF5ExportObserver(output_dir=config.output_dir))
        elif cls is Overlay2DObserver:
            observers.append(
                Overlay2DObserver(
                    output_dir=config.output_dir,
                    video_dir=config.video_dir,
                    calibration_path=config.calibration_path,
                )
            )
        elif cls is Animation3DObserver:
            observers.append(Animation3DObserver(output_dir=config.output_dir))
        elif cls is DiagnosticObserver:
            observers.append(DiagnosticObserver())
        elif cls is ConsoleObserver:
            observers.append(
                ConsoleObserver(verbose=verbose, total_stages=total_stages)
            )

    return observers


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
    default="production",
    help="Execution mode preset.",
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
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose output.")
def run(
    config: str,
    mode: str,
    overrides: tuple[str, ...],
    extra_observers: tuple[str, ...],
    verbose: bool,
) -> None:
    """Run the AquaPose pipeline."""
    # 1. Parse --set overrides into dict
    cli_overrides: dict[str, Any] = {}
    for item in overrides:
        key, _, value = item.partition("=")
        if not key:
            continue
        cli_overrides[key] = value

    # 2. Inject mode into overrides
    cli_overrides["mode"] = mode

    # 3. Load config
    pipeline_config = load_config(yaml_path=config, cli_overrides=cli_overrides)

    # 4. Build stages
    stages = build_stages(pipeline_config)

    # 5. Assemble observers
    observers = _build_observers(
        config=pipeline_config,
        mode=mode,
        verbose=verbose,
        total_stages=len(stages),
        extra_observers=extra_observers,
    )

    # 6. Create and run pipeline
    pipeline = PosePipeline(
        stages=stages,
        config=pipeline_config,
        observers=observers,
    )

    try:
        pipeline.run()
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
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
    yaml_content = serialize_config(config)
    output_path.write_text(yaml_content)
    click.echo(f"Config written to {output}")


def main() -> None:
    """Entry point for the ``aquapose`` console script."""
    cli()
