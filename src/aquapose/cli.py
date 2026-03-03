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
@click.option(
    "--resume-from",
    "resume_from",
    type=click.Path(exists=True),
    default=None,
    help="Path to a stage cache pickle file. Skips stages whose outputs are already populated.",
)
def run(
    config: str,
    mode: str | None,
    overrides: tuple[str, ...],
    extra_observers: tuple[str, ...],
    stop_after: str | None,
    verbose: bool,
    resume_from: str | None,
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

    # 5.5. Load initial context from cache if --resume-from provided.
    # Each resumed run gets its own run_id (not inherited from cache).
    initial_context = None
    if resume_from is not None:
        from aquapose.core.context import StaleCacheError, load_stage_cache

        try:
            initial_context = load_stage_cache(resume_from)
            click.echo(f"Loaded stage cache from {resume_from}")
        except StaleCacheError as exc:
            raise click.ClickException(str(exc)) from exc
        except FileNotFoundError:
            raise click.ClickException(f"Cache file not found: {resume_from}") from None

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
        pipeline.run(initial_context=initial_context)
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
    data["video_dir"] = "videos"
    data["calibration_path"] = "geometry/calibration.json"
    data["output_dir"] = "runs"
    # --- Core parameters ---
    data["n_animals"] = "SET_ME"  # required -- must be an integer
    # --- Detection ---
    data["detection"] = {
        "detector_kind": "yolo_obb",  # oriented bounding box detection
        "weights_path": "models/yolo_obb.pt",
    }
    # --- Midline ---
    data["midline"] = {
        "backend": "pose_estimation",  # segmentation or pose_estimation
        "weights_path": "models/yolo_pose.pt",
    }
    # --- Synthetic (only with --synthetic) ---
    if synthetic:
        data["synthetic"] = {"frame_count": 30, "noise_std": 0.0, "seed": 42}

    # Write config.yaml with brief comment header
    header = "# AquaPose pipeline config\n# See documentation for advanced options\n\n"
    yaml_content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    (project_dir / "config.yaml").write_text(header + yaml_content)

    click.echo(f"Project created at {project_dir}")


@cli.command("eval")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--report",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format (default: text).",
)
@click.option(
    "--n-frames",
    "n_frames",
    type=int,
    default=None,
    help="Number of frames to evaluate (default: all frames).",
)
def eval_cmd(run_dir: str, report: str, n_frames: int | None) -> None:
    """Evaluate a diagnostic run directory and print a quality report."""
    import json as _json

    from aquapose.core.context import StaleCacheError
    from aquapose.evaluation.output import (
        _NumpySafeEncoder,
        format_eval_json,
        format_eval_report,
    )
    from aquapose.evaluation.runner import EvalRunner

    runner = EvalRunner(Path(run_dir))
    try:
        result = runner.run(n_frames=n_frames)
    except StaleCacheError as exc:
        raise click.ClickException(str(exc)) from exc

    if report == "json":
        output = format_eval_json(result)
    else:
        output = format_eval_report(result)

    click.echo(output)

    # Write eval_results.json to run directory on every invocation
    results_path = Path(run_dir) / "eval_results.json"
    with results_path.open("w") as f:
        _json.dump(result.to_dict(), f, cls=_NumpySafeEncoder, indent=2)


cli.add_command(train_group)
cli.add_command(prep_group)


def main() -> None:
    """Entry point for the ``aquapose`` console script."""
    cli()
