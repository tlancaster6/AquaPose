"""AquaPose CLI -- thin wrapper over ChunkOrchestrator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click
import yaml

from aquapose.engine import load_config
from aquapose.engine.orchestrator import ChunkOrchestrator
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
        ["timing", "diagnostic", "console"],
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
    "--max-chunks",
    "max_chunks",
    type=int,
    default=None,
    help="Process at most N chunks then stop (e.g. --max-chunks 1 for single-chunk).",
)
def run(
    config: str,
    mode: str | None,
    overrides: tuple[str, ...],
    extra_observers: tuple[str, ...],
    stop_after: str | None,
    verbose: bool,
    max_chunks: int | None,
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

    # 4. Delegate to ChunkOrchestrator (config-only handoff).
    # Orchestrator constructs and manages VideoFrameSource internally so that
    # __enter__ is called (discovers videos).  The frame_source parameter is
    # only used when injecting a pre-opened source (e.g. synthetic mode).
    orchestrator = ChunkOrchestrator(
        config=pipeline_config,
        verbose=verbose,
        max_chunks=max_chunks,
        stop_after=stop_after,
        extra_observers=extra_observers,
    )

    try:
        orchestrator.run()
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
    from aquapose.logging import setup_file_logging

    setup_file_logging(Path(run_dir), "eval")

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


@cli.command("tune")
@click.option(
    "--stage",
    "-s",
    required=True,
    type=click.Choice(["association", "reconstruction"], case_sensitive=False),
    help="Stage to tune.",
)
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to the run-generated exhaustive config YAML.",
)
@click.option(
    "--n-frames",
    "n_frames",
    type=int,
    default=30,
    help="Frame count for fast sweep (reconstruction only, default: 30).",
)
@click.option(
    "--n-frames-validate",
    "n_frames_validate",
    type=int,
    default=100,
    help="Frame count for top-N validation (reconstruction only, default: 100).",
)
@click.option(
    "--top-n",
    "top_n",
    type=int,
    default=3,
    help="Number of top candidates for validation (reconstruction only, default: 3).",
)
def tune_cmd(
    stage: str,
    config: str,
    n_frames: int,
    n_frames_validate: int,
    top_n: int,
) -> None:
    """Sweep stage parameters and output a recommended config diff."""
    from pathlib import Path as _Path

    from aquapose.core.context import StaleCacheError
    from aquapose.evaluation.tuning import (
        TuningOrchestrator,
        format_comparison_table,
        format_config_diff,
        format_yield_matrix,
    )
    from aquapose.logging import setup_file_logging

    config_path = _Path(config)
    setup_file_logging(config_path.parent, "tune")

    try:
        orchestrator = TuningOrchestrator(config_path)
    except StaleCacheError as exc:
        raise click.ClickException(str(exc)) from exc
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        if stage == "association":
            result = orchestrator.sweep_association()
        else:
            result = orchestrator.sweep_reconstruction(
                n_frames=n_frames,
                n_frames_validate=n_frames_validate,
                top_n=top_n,
            )
    except StaleCacheError as exc:
        raise click.ClickException(str(exc)) from exc

    # Print 2D yield matrix for association sweeps
    if result.joint_grid_results is not None:
        from aquapose.evaluation.stages.association import DEFAULT_GRID as ASSOC_GRID

        click.echo(
            format_yield_matrix(
                result.joint_grid_results,
                "ray_distance_threshold",
                ASSOC_GRID["ray_distance_threshold"],
                "score_min",
                ASSOC_GRID["score_min"],
            )
        )
        click.echo()

    # Print comparison table
    click.echo(format_comparison_table(result.baseline_metrics, result.winner_metrics))
    click.echo()

    # Print config diff
    from aquapose.engine.config import load_config as _load_config

    baseline_config = _load_config(config_path)
    stage_config = getattr(baseline_config, stage)
    click.echo(format_config_diff(stage, result.winner_params, stage_config))


@cli.group()
def viz() -> None:
    """Generate visualizations from diagnostic run caches."""


@viz.command("overlay")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--output-dir",
    "-o",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Custom output directory (default: {run_dir}/viz/).",
)
def viz_overlay(run_dir: str, output_dir: str | None) -> None:
    """Generate a 2D reprojection overlay mosaic video."""
    from aquapose.evaluation.viz import generate_overlay

    out_dir = Path(output_dir) if output_dir is not None else None
    try:
        result = generate_overlay(Path(run_dir), out_dir)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Overlay written to: {result}")


@viz.command("animation")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--output-dir",
    "-o",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Custom output directory (default: {run_dir}/viz/).",
)
def viz_animation(run_dir: str, output_dir: str | None) -> None:
    """Generate an interactive 3D midline animation HTML."""
    from aquapose.evaluation.viz import generate_animation

    out_dir = Path(output_dir) if output_dir is not None else None
    try:
        result = generate_animation(Path(run_dir), out_dir)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Animation written to: {result}")


@viz.command("trails")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--output-dir",
    "-o",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Custom output directory (default: {run_dir}/viz/).",
)
def viz_trails(run_dir: str, output_dir: str | None) -> None:
    """Generate per-camera trail videos and association mosaic."""
    from aquapose.evaluation.viz import generate_trails

    out_dir = Path(output_dir) if output_dir is not None else None
    try:
        result = generate_trails(Path(run_dir), out_dir)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Trail videos written to: {result}")


@viz.command("all")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--output-dir",
    "-o",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Custom output directory (default: {run_dir}/viz/).",
)
def viz_all(run_dir: str, output_dir: str | None) -> None:
    """Attempt every visualization; skip gracefully on failure."""
    from aquapose.evaluation.viz import generate_all

    out_dir = Path(output_dir) if output_dir is not None else None
    results = generate_all(Path(run_dir), out_dir)

    succeeded: list[str] = []
    skipped: list[tuple[str, str]] = []
    for name, outcome in results.items():
        if isinstance(outcome, Exception):
            skipped.append((name, str(outcome)))
        else:
            succeeded.append(f"  {name}: {outcome}")

    if succeeded:
        click.echo("Succeeded:")
        for line in succeeded:
            click.echo(line)

    if skipped:
        click.echo("Skipped (failures):")
        for name, reason in skipped:
            click.echo(f"  {name}: {reason}")


cli.add_command(train_group)
cli.add_command(prep_group)


def main() -> None:
    """Entry point for the ``aquapose`` console script."""
    cli()
