"""AquaPose CLI -- thin wrapper over ChunkOrchestrator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click
import yaml

from aquapose.cli_utils import get_config_path, get_project_dir, resolve_run
from aquapose.engine import load_config
from aquapose.engine.orchestrator import ChunkOrchestrator
from aquapose.training.cli import train_group
from aquapose.training.data_cli import data_group
from aquapose.training.prep import prep_group
from aquapose.training.pseudo_label_cli import pseudo_label_group

# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--project",
    "-p",
    "project_name",
    default=None,
    help="Project name (looked up under ~/aquapose/projects/).",
)
@click.pass_context
def cli(ctx: click.Context, project_name: str | None) -> None:
    """AquaPose -- 3D fish pose estimation via refractive multi-view triangulation."""
    ctx.ensure_object(dict)
    ctx.obj["project_name"] = project_name


@cli.command()
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
        ["detection", "pose", "tracking", "association"], case_sensitive=False
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
@click.pass_context
def run(
    ctx: click.Context,
    mode: str | None,
    overrides: tuple[str, ...],
    extra_observers: tuple[str, ...],
    stop_after: str | None,
    verbose: bool,
    max_chunks: int | None,
) -> None:
    """Run the AquaPose pipeline."""
    config_path = get_config_path(ctx)

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
    pipeline_config = load_config(
        yaml_path=str(config_path), cli_overrides=cli_overrides
    )

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


@cli.command("init")
@click.argument("name")
@click.option(
    "--synthetic",
    is_flag=True,
    default=False,
    help="Include a synthetic section in the generated config.",
)
def init_cmd(name: str, synthetic: bool) -> None:
    """Create a new AquaPose project directory scaffold with a starter config.

    NAME is the project name. Creates ~/aquapose/projects/<name>/ with
    config.yaml and subdirectories: runs/, models/, geometry/, videos/,
    training_data/obb/, training_data/pose/.
    """
    project_dir = Path("~/aquapose/projects").expanduser() / name
    if project_dir.exists():
        raise click.ClickException(f"'{project_dir}' already exists.")

    # Create directory structure
    project_dir.mkdir(parents=True, exist_ok=False)
    for subdir in (
        "runs",
        "models",
        "geometry",
        "videos",
        "training_data/obb",
        "training_data/pose",
    ):
        (project_dir / subdir).mkdir(parents=True)

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
    # --- Pose ---
    data["pose"] = {
        "weights_path": "models/yolo_pose.pt",
    }
    # --- Synthetic (only with --synthetic) ---
    if synthetic:
        data["synthetic"] = {"frame_count": 30, "noise_std": 0.0, "seed": 42}

    # Write config.yaml with brief comment header
    header = "# AquaPose pipeline config\n# See documentation for advanced options\n\n"
    yaml_content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    # Inject reminder comment before the pose section
    yaml_content = yaml_content.replace(
        "pose:",
        "# Run 'aquapose prep calibrate-keypoints' to set keypoint_t_values\npose:",
    )
    (project_dir / "config.yaml").write_text(header + yaml_content)

    click.echo(f"Project created at {project_dir}")
    click.echo("")
    click.echo("Next steps:")
    click.echo("  1. Place calibration JSON in geometry/calibration.json")
    click.echo(f"  2. Run: aquapose --project {name} prep generate-luts")
    click.echo(
        f"  3. Run: aquapose --project {name} prep calibrate-keypoints"
        " --annotations <json>"
    )


@cli.command("eval")
@click.argument("run", default=None, required=False)
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
@click.pass_context
def eval_cmd(
    ctx: click.Context, run: str | None, report: str, n_frames: int | None
) -> None:
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

    run_dir = resolve_run(run, get_project_dir(ctx))

    setup_file_logging(run_dir, "eval")

    runner = EvalRunner(run_dir)
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
    results_path = run_dir / "eval_results.json"
    with results_path.open("w") as f:
        _json.dump(result.to_dict(), f, cls=_NumpySafeEncoder, indent=2)


@cli.command("eval-compare")
@click.argument("run_a")
@click.argument("run_b")
@click.pass_context
def eval_compare_cmd(ctx: click.Context, run_a: str, run_b: str) -> None:
    """Compare pipeline evaluation metrics between two runs."""
    from aquapose.evaluation.compare import (
        format_comparison_table,
        load_eval_results,
        write_comparison_json,
    )

    project_dir = get_project_dir(ctx)
    run_a_dir = resolve_run(run_a, project_dir)
    run_b_dir = resolve_run(run_b, project_dir)

    try:
        results_a = load_eval_results(run_a_dir)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        results_b = load_eval_results(run_b_dir)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    table = format_comparison_table(
        results_a, results_b, run_a_dir.name, run_b_dir.name
    )
    click.echo(table)

    out_path = write_comparison_json(
        results_a, results_b, run_a_dir, run_b_dir, run_b_dir
    )
    click.echo(f"\nComparison written to {out_path}")


@cli.command("tune")
@click.argument("run", default=None, required=False)
@click.option(
    "--stage",
    "-s",
    required=True,
    type=click.Choice(["association", "reconstruction"], case_sensitive=False),
    help="Stage to tune.",
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
@click.pass_context
def tune_cmd(
    ctx: click.Context,
    run: str | None,
    stage: str,
    n_frames: int,
    n_frames_validate: int,
    top_n: int,
) -> None:
    """Sweep stage parameters and output a recommended config diff."""
    from aquapose.core.context import StaleCacheError
    from aquapose.evaluation.tuning import (
        TuningOrchestrator,
        format_comparison_table,
        format_config_diff,
        format_yield_matrix,
    )
    from aquapose.logging import setup_file_logging

    run_dir = resolve_run(run, get_project_dir(ctx))
    config_path = run_dir / "config_exhaustive.yaml"
    if not config_path.exists():
        config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise click.ClickException(
            f"No config found in {run_dir}. "
            "Run the pipeline first to generate a run directory."
        )

    setup_file_logging(run_dir, "tune")

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


@cli.command()
@click.argument("run", default=None, required=False)
@click.option(
    "--output-dir",
    "-o",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Custom output directory (default: {run_dir}/viz/).",
)
@click.option(
    "--overlay", is_flag=True, help="Generate 2D reprojection overlay mosaic video."
)
@click.option(
    "--animation", is_flag=True, help="Generate interactive 3D midline animation HTML."
)
@click.option(
    "--trails",
    is_flag=True,
    help="Generate per-camera trail videos and association mosaic.",
)
@click.option(
    "--fade-trails",
    "fade_trails",
    is_flag=True,
    default=False,
    help="Enable per-segment alpha fading on trail lines (significantly slower).",
)
@click.option(
    "--detections",
    is_flag=True,
    help="Generate detection overlay mosaic PNG (OBB boxes colored by confidence).",
)
@click.option(
    "--stride",
    type=int,
    default=1,
    show_default=True,
    help="Animation frame stride: keep every Nth frame (reduces HTML size for long videos).",
)
@click.pass_context
def viz(
    ctx: click.Context,
    run: str | None,
    output_dir: str | None,
    overlay: bool,
    animation: bool,
    trails: bool,
    fade_trails: bool,
    detections: bool,
    stride: int,
) -> None:
    """Generate visualizations from diagnostic run caches.

    With no flags, generates all visualizations. Pass one or more flags to
    select specific outputs.
    """
    from aquapose.evaluation.viz import (
        generate_animation,
        generate_detection_overlay,
        generate_overlay,
        generate_trails,
    )

    out_dir = Path(output_dir) if output_dir is not None else None
    run_path = resolve_run(run, get_project_dir(ctx))

    # No flags -> all visualizations
    selected = {
        "overlay": overlay,
        "animation": animation,
        "trails": trails,
        "detections": detections,
    }
    if not any(selected.values()):
        selected = {k: True for k in selected}

    generators: dict[str, Any] = {
        "overlay": lambda: generate_overlay(run_path, out_dir),
        "animation": lambda: generate_animation(run_path, out_dir, stride=stride),
        "detections": lambda: generate_detection_overlay(run_path, out_dir),
        "trails": lambda: generate_trails(run_path, out_dir, fade_trails=fade_trails),
    }

    succeeded: list[str] = []
    skipped: list[tuple[str, str]] = []
    for name, enabled in selected.items():
        if not enabled:
            continue
        try:
            result = generators[name]()
            succeeded.append(f"  {name}: {result}")
        except Exception as exc:
            skipped.append((name, str(exc)))

    if succeeded:
        click.echo("Succeeded:")
        for line in succeeded:
            click.echo(line)

    if skipped:
        click.echo("Skipped (failures):")
        for name, reason in skipped:
            click.echo(f"  {name}: {reason}")


@cli.command("smooth-z")
@click.argument("run", default=None, required=False)
@click.option(
    "--sigma-frames",
    default=3,
    type=int,
    show_default=True,
    help="Gaussian filter sigma in frames for centroid z smoothing.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report metrics without modifying the file.",
)
@click.pass_context
def smooth_z_cmd(
    ctx: click.Context, run: str | None, sigma_frames: int, dry_run: bool
) -> None:
    """Temporally smooth centroid z and shift control points in a midlines HDF5.

    Reads centroid_z per fish from a pipeline run's midlines.h5, applies
    Gaussian smoothing per-fish within continuous track segments, and shifts
    all control point z-coordinates by the smoothing delta.
    """
    import h5py
    import numpy as np

    from aquapose.core.reconstruction.temporal_smoothing import smooth_centroid_z
    from aquapose.io.midline_writer import read_midline3d_results

    run_dir = resolve_run(run, get_project_dir(ctx))
    input_path = run_dir / "midlines.h5"
    if not input_path.exists():
        raise click.ClickException(
            f"midlines.h5 not found in {run_dir}. "
            "Run the pipeline first to generate midline data."
        )

    data = read_midline3d_results(str(input_path))

    if data["centroid_z"] is None:
        raise click.ClickException(
            "No centroid_z dataset found. "
            "Run the pipeline with z_denoising.enabled=true first."
        )

    frame_indices = data["frame_index"]  # (N,)
    fish_ids_all = data["fish_id"]  # (N, max_fish)
    centroid_z_all = data["centroid_z"]  # (N, max_fish)
    control_points = data["control_points"]  # (N, max_fish, 7, 3)

    N, max_fish = fish_ids_all.shape
    smoothed_cz = centroid_z_all.copy()
    shifted_cp = control_points.copy()

    # Collect unique fish IDs (excluding fill value -1)
    unique_fish = set()
    for f in range(N):
        for s in range(max_fish):
            fid = int(fish_ids_all[f, s])
            if fid >= 0:
                unique_fish.add(fid)

    total_frames = 0
    total_jitter_before = 0.0
    total_jitter_after = 0.0
    n_jitter = 0

    for fid in sorted(unique_fish):
        ts_frames: list[int] = []
        ts_frame_idx: list[int] = []
        ts_slot: list[int] = []

        for f in range(N):
            for s in range(max_fish):
                if int(fish_ids_all[f, s]) == fid:
                    ts_frames.append(f)
                    ts_frame_idx.append(int(frame_indices[f]))
                    ts_slot.append(s)
                    break

        if len(ts_frames) < 2:
            continue

        raw_cz = np.array(
            [centroid_z_all[ts_frames[i], ts_slot[i]] for i in range(len(ts_frames))]
        )
        f_frame_idx = np.array(ts_frame_idx, dtype=np.int64)

        if np.all(np.isnan(raw_cz)):
            continue

        sm_cz = smooth_centroid_z(raw_cz, f_frame_idx, sigma_frames=sigma_frames)

        for i in range(len(ts_frames)):
            fi = ts_frames[i]
            si = ts_slot[i]
            dz = float(sm_cz[i] - raw_cz[i])
            smoothed_cz[fi, si] = float(sm_cz[i])
            shifted_cp[fi, si, :, 2] += dz
            total_frames += 1

        # Track jitter reduction (skip pairs where either value is NaN)
        for i in range(1, len(ts_frames)):
            if np.isnan(raw_cz[i]) or np.isnan(raw_cz[i - 1]):
                continue
            total_jitter_before += abs(float(raw_cz[i] - raw_cz[i - 1]))
            total_jitter_after += abs(float(sm_cz[i] - sm_cz[i - 1]))
            n_jitter += 1

    mean_jitter_before = (total_jitter_before / n_jitter * 100) if n_jitter else 0
    mean_jitter_after = (total_jitter_after / n_jitter * 100) if n_jitter else 0

    click.echo(f"Fish processed: {len(unique_fish)}")
    click.echo(f"Frames processed: {total_frames}")
    click.echo(f"Sigma frames: {sigma_frames}")
    click.echo(
        f"Mean F2F centroid z jitter: {mean_jitter_before:.3f} cm -> "
        f"{mean_jitter_after:.3f} cm"
    )

    if dry_run:
        click.echo("Dry run -- no changes written.")
        return

    with h5py.File(str(input_path), "r+") as f:
        grp = f["midlines"]
        assert isinstance(grp, h5py.Group)

        if "smoothed_centroid_z" in grp:
            del grp["smoothed_centroid_z"]
        grp.create_dataset(
            "smoothed_centroid_z",
            data=smoothed_cz,
            compression="gzip",
            compression_opts=4,
        )

        cp_ds = grp["control_points"]
        assert isinstance(cp_ds, h5py.Dataset)
        cp_ds[...] = shifted_cp

    click.echo(f"Written to {input_path}")


cli.add_command(data_group)
cli.add_command(train_group)
cli.add_command(prep_group)
cli.add_command(pseudo_label_group)


def main() -> None:
    """Entry point for the ``aquapose`` console script."""
    cli()
