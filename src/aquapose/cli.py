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
from aquapose.training.pseudo_label_cli import pseudo_label_group

# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
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
    # Inject reminder comment before the midline section
    yaml_content = yaml_content.replace(
        "midline:",
        "# Run 'aquapose prep calibrate-keypoints' to set keypoint_t_values\nmidline:",
    )
    (project_dir / "config.yaml").write_text(header + yaml_content)

    click.echo(f"Project created at {project_dir}")
    click.echo("")
    click.echo("Next steps:")
    click.echo("  1. Place calibration JSON in geometry/calibration.json")
    click.echo("  2. Run: aquapose prep generate-luts --config config.yaml")
    click.echo(
        "  3. Run: aquapose prep calibrate-keypoints"
        " --annotations <json> --config config.yaml"
    )


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


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
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
def viz(
    run_dir: str,
    output_dir: str | None,
    overlay: bool,
    animation: bool,
    trails: bool,
    fade_trails: bool,
) -> None:
    """Generate visualizations from diagnostic run caches.

    With no flags, generates all visualizations. Pass one or more flags to
    select specific outputs.
    """
    from aquapose.evaluation.viz import (
        generate_animation,
        generate_overlay,
        generate_trails,
    )

    out_dir = Path(output_dir) if output_dir is not None else None
    run_path = Path(run_dir)

    # No flags → all visualizations
    selected = {
        "overlay": overlay,
        "animation": animation,
        "trails": trails,
    }
    if not any(selected.values()):
        selected = {k: True for k in selected}

    generators: dict[str, Any] = {
        "overlay": lambda: generate_overlay(run_path, out_dir),
        "animation": lambda: generate_animation(run_path, out_dir),
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


@cli.command("smooth-planes")
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to midlines.h5 file (from pipeline run).",
)
@click.option(
    "--sigma-frames",
    default=3,
    type=int,
    show_default=True,
    help="Gaussian filter sigma in frames for normal smoothing.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report metrics without modifying the file.",
)
def smooth_planes(input_path: str, sigma_frames: int, dry_run: bool) -> None:
    """Smooth plane normals and rotate control points in a midlines HDF5 file.

    Reads plane metadata from a pipeline run's midlines.h5, applies temporal
    Gaussian smoothing to plane normals per-fish within continuous track
    segments, rotates control points to match the smoothed orientation, and
    writes the results back in-place.
    """
    import h5py
    import numpy as np

    from aquapose.core.reconstruction.temporal_smoothing import (
        rotate_control_points_to_plane,
        smooth_plane_normals,
    )
    from aquapose.io.midline_writer import read_midline3d_results

    data = read_midline3d_results(input_path)

    if data["plane_normal"] is None:
        raise click.ClickException(
            "No plane_normal dataset found in the HDF5 file. "
            "Run the pipeline with plane_projection.enabled=True first."
        )

    frame_indices = data["frame_index"]  # (N,)
    fish_ids_all = data["fish_id"]  # (N, max_fish)
    plane_normals = data["plane_normal"]  # (N, max_fish, 3)
    plane_centroids = data["plane_centroid"]  # (N, max_fish, 3)
    control_points = data["control_points"]  # (N, max_fish, 7, 3)
    is_degenerate = data["is_degenerate_plane"]  # (N, max_fish)

    N, max_fish = fish_ids_all.shape
    smoothed_normals = plane_normals.copy()
    rotated_cp = control_points.copy()

    # Collect unique fish IDs (excluding fill value -1)
    unique_fish = set()
    for f in range(N):
        for s in range(max_fish):
            fid = int(fish_ids_all[f, s])
            if fid >= 0:
                unique_fish.add(fid)

    total_frames_processed = 0
    total_angular_change = 0.0
    n_angular = 0

    for fid in sorted(unique_fish):
        # Extract time series for this fish
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

        # Build arrays for this fish
        f_normals = np.array(
            [plane_normals[ts_frames[i], ts_slot[i]] for i in range(len(ts_frames))]
        )
        f_degen = np.array(
            [is_degenerate[ts_frames[i], ts_slot[i]] for i in range(len(ts_frames))]
        )
        f_fish_ids = np.full(len(ts_frames), fid, dtype=int)
        f_frame_idx = np.array(ts_frame_idx, dtype=np.int64)

        # Skip if all normals are NaN (no plane data)
        if np.all(np.isnan(f_normals)):
            continue

        # Sign-correct raw normals before smoothing so the rotation
        # reference matches the smoother's internal sign convention.
        for t in range(1, len(f_normals)):
            if np.dot(f_normals[t], f_normals[t - 1]) < 0:
                f_normals[t] = -f_normals[t]

        # Smooth
        f_smoothed = smooth_plane_normals(
            f_normals, f_degen, f_fish_ids, f_frame_idx, sigma_frames=sigma_frames
        )

        # Rotate control points and store results
        for i in range(len(ts_frames)):
            fi = ts_frames[i]
            si = ts_slot[i]
            raw_n = f_normals[i]  # now sign-corrected
            sm_n = f_smoothed[i]
            cent = plane_centroids[fi, si]
            cp = control_points[fi, si]

            if np.any(np.isnan(raw_n)) or np.any(np.isnan(sm_n)):
                continue

            rotated = rotate_control_points_to_plane(cp, cent, raw_n, sm_n)
            smoothed_normals[fi, si] = sm_n.astype(np.float32)
            rotated_cp[fi, si] = rotated.astype(np.float32)

            # Track angular change
            dot_val = np.clip(np.dot(raw_n, sm_n), -1.0, 1.0)
            total_angular_change += np.arccos(dot_val)
            n_angular += 1
            total_frames_processed += 1

    mean_angular_deg = (
        np.degrees(total_angular_change / n_angular) if n_angular > 0 else 0.0
    )

    click.echo(f"Fish processed: {len(unique_fish)}")
    click.echo(f"Frames processed: {total_frames_processed}")
    click.echo(f"Mean angular change: {mean_angular_deg:.2f} degrees")
    click.echo(f"Sigma frames: {sigma_frames}")

    if dry_run:
        click.echo("Dry run -- no changes written.")
        return

    # Write back in-place
    with h5py.File(input_path, "r+") as f:
        grp = f["midlines"]

        # Add smoothed_plane_normal dataset if it doesn't exist
        if "smoothed_plane_normal" in grp:
            del grp["smoothed_plane_normal"]
        grp.create_dataset(
            "smoothed_plane_normal",
            data=smoothed_normals,
            compression="gzip",
            compression_opts=4,
        )

        # Overwrite control_points with rotated version
        grp["control_points"][...] = rotated_cp

    click.echo(f"Written to {input_path}")


cli.add_command(train_group)
cli.add_command(prep_group)
cli.add_command(pseudo_label_group)


def main() -> None:
    """Entry point for the ``aquapose`` console script."""
    cli()
