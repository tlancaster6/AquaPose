"""Calibration data preparation CLI for AquaPose training.

Provides the ``prep`` CLI group with subcommands for computing derived
configuration values from annotation data. Supports:

- ``calibrate-keypoints``: Compute keypoint t-values from COCO annotations.
- ``generate-luts``: Pre-generate forward and inverse lookup tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import yaml

__all__ = ["prep_group"]


class _LutConfigFromDict:
    """Minimal LUT config satisfying the ``LutConfigLike`` protocol.

    Built from a plain dict (YAML ``lut`` section) to avoid importing
    ``aquapose.engine.config`` (which would violate the training→engine
    import boundary).
    """

    def __init__(self, d: dict) -> None:
        self.tank_diameter: float = float(d.get("tank_diameter", 1.0))
        self.tank_height: float = float(d.get("tank_height", 0.5))
        self.voxel_resolution_m: float = float(d.get("voxel_resolution_m", 0.01))
        self.margin_fraction: float = float(d.get("margin_fraction", 0.1))
        self.forward_grid_step: int = int(d.get("forward_grid_step", 4))


@click.group("prep")
def prep_group() -> None:
    """Prepare calibration data and derived configuration."""


def _parse_keypoints_coco(
    annotations_path: Path, n_keypoints: int
) -> list[list[tuple[float, float]]]:
    """Parse keypoint instances from a COCO JSON file.

    Returns a list of instances, each a list of (x, y) tuples with NaN for
    invisible keypoints.
    """
    with annotations_path.open() as fh:
        coco = json.load(fh)

    instances: list[list[tuple[float, float]]] = []
    for ann in coco.get("annotations", []):
        raw_kps = ann.get("keypoints")
        if not raw_kps:
            continue
        # COCO keypoints: [x, y, v, x, y, v, ...]
        points: list[tuple[float, float]] = []
        for k in range(n_keypoints):
            base = k * 3
            if base + 2 >= len(raw_kps):
                break
            x, y, v = raw_kps[base], raw_kps[base + 1], raw_kps[base + 2]
            if v > 0:
                points.append((float(x), float(y)))
            else:
                points.append((float("nan"), float("nan")))
        instances.append(points)
    return instances


def _parse_keypoints_yolo(
    labels_dir: Path, n_keypoints: int
) -> list[list[tuple[float, float]]]:
    """Parse keypoint instances from a directory of YOLO label txt files.

    Recurses through subdirectories (e.g. train/, val/) to find all .txt
    files. YOLO pose format per line:
    ``class cx cy w h x1 y1 v1 x2 y2 v2 ...``
    where coordinates are normalized [0, 1].
    """
    txt_files = sorted(labels_dir.rglob("*.txt"))
    instances: list[list[tuple[float, float]]] = []
    for txt_path in txt_files:
        for line in txt_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5 + n_keypoints * 3:
                continue
            # Skip class + bbox (5 values), then parse keypoints
            kp_values = parts[5:]
            points: list[tuple[float, float]] = []
            for k in range(n_keypoints):
                base = k * 3
                x, y, v = (
                    float(kp_values[base]),
                    float(kp_values[base + 1]),
                    float(kp_values[base + 2]),
                )
                if v > 0:
                    points.append((x, y))
                else:
                    points.append((float("nan"), float("nan")))
            instances.append(points)
    return instances


def _compute_t_values(
    instances: list[list[tuple[float, float]]], n_keypoints: int
) -> tuple[list[float], int]:
    """Compute mean arc-length t-values from keypoint instances.

    Args:
        instances: List of keypoint instances (each a list of (x, y) tuples).
        n_keypoints: Expected number of keypoints per instance.

    Returns:
        Tuple of (t_values_list, n_processed).
    """
    all_t_values: list[list[float]] = []
    n_processed = 0

    for points in instances:
        if len(points) < 2:
            continue

        visible_indices = [i for i, (x, _) in enumerate(points) if not np.isnan(x)]
        if len(visible_indices) < 2:
            continue

        visible_pts = np.array([points[i] for i in visible_indices], dtype=np.float64)
        diffs = np.diff(visible_pts, axis=0)
        dists = np.sqrt((diffs**2).sum(axis=1))
        cumulative = np.concatenate([[0.0], np.cumsum(dists)])
        total_length = cumulative[-1]

        if total_length < 1e-6:
            continue

        t_per_annotation = [float("nan")] * n_keypoints
        for j, idx in enumerate(visible_indices):
            t_per_annotation[idx] = cumulative[j] / total_length

        all_t_values.append(t_per_annotation)
        n_processed += 1

    if n_processed == 0:
        return [], 0

    t_array = np.array(all_t_values, dtype=np.float64)
    mean_t = np.nanmean(t_array, axis=0)
    mean_t = np.clip(mean_t, 0.0, 1.0)
    return [round(float(t), 4) for t in mean_t], n_processed


@prep_group.command("calibrate-keypoints")
@click.option(
    "--annotations",
    required=True,
    type=click.Path(exists=True),
    help="Path to COCO JSON file or directory of YOLO label txt files.",
)
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Path to pipeline config YAML. Updates keypoint_t_values in place.",
)
@click.option(
    "--n-keypoints",
    default=6,
    type=int,
    help="Number of anatomical keypoints.",
)
def calibrate_keypoints(annotations: str, config: str, n_keypoints: int) -> None:
    """Compute keypoint t-values from keypoint annotations.

    Auto-detects annotation format: if ``--annotations`` points to a JSON
    file, reads COCO format. If it points to a directory, reads YOLO-format
    txt labels (recursing through subdirectories like train/ and val/).

    Computes the arc-length fraction (t-value) for each anatomical keypoint,
    averaged across all annotated instances. The resulting t-values represent
    each keypoint's position along the fish body curve (0.0 = nose, 1.0 = tail).

    Updates the pipeline config YAML in place, setting
    ``midline.keypoint_t_values`` to the computed values.

    Args:
        annotations: Path to COCO JSON file or directory of YOLO label txts.
        config: Path to pipeline config YAML to update in place.
        n_keypoints: Number of anatomical keypoints expected per annotation.
    """
    annotations_path = Path(annotations)
    config_path = Path(config)

    if annotations_path.is_dir():
        click.echo(f"Detected YOLO label directory: {annotations_path}")
        instances = _parse_keypoints_yolo(annotations_path, n_keypoints)
    elif annotations_path.suffix.lower() == ".json":
        click.echo(f"Detected COCO JSON file: {annotations_path}")
        instances = _parse_keypoints_coco(annotations_path, n_keypoints)
    else:
        raise click.ClickException(
            f"Cannot detect annotation format for {annotations_path}. "
            "Provide a .json file (COCO) or a directory (YOLO labels)."
        )

    t_values_list, n_processed = _compute_t_values(instances, n_keypoints)

    if n_processed == 0:
        raise click.ClickException(
            f"No valid keypoint annotations found in {annotations_path}. "
            "Ensure annotations have keypoints with visibility > 0."
        )

    # Update pipeline config YAML in place
    with config_path.open() as fh:
        config_data = yaml.safe_load(fh) or {}

    if "midline" not in config_data:
        config_data["midline"] = {}
    config_data["midline"]["keypoint_t_values"] = t_values_list

    with config_path.open("w") as fh:
        yaml.dump(config_data, fh, default_flow_style=False, sort_keys=False)

    click.echo(f"Processed {n_processed} annotations.")
    click.echo(f"Computed t-values: {t_values_list}")
    click.echo(f"Updated config: {config_path}")


@prep_group.command("generate-luts")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Path to pipeline config YAML.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Regenerate even if LUTs already exist.",
)
def generate_luts_cmd(config: str, force: bool) -> None:
    """Pre-generate forward and inverse lookup tables for association.

    Reads the pipeline config to determine calibration path and LUT
    parameters, then generates and saves forward and inverse LUTs to
    the luts/ directory next to the calibration file.

    Args:
        config: Path to pipeline config YAML.
        force: If True, regenerate even when cached LUTs already exist.
    """
    from aquapose.calibration.loader import (
        compute_undistortion_maps,
        load_calibration_data,
    )
    from aquapose.calibration.luts import (
        generate_forward_luts,
        generate_inverse_lut,
        load_forward_luts,
        load_inverse_luts,
        save_forward_luts,
        save_inverse_luts,
    )

    config_path = Path(config)

    with config_path.open() as fh:
        raw = yaml.safe_load(fh) or {}

    calibration_path = raw.get("calibration_path")
    if not calibration_path:
        raise click.ClickException(
            "calibration_path is not set in the pipeline config."
        )

    # Resolve calibration_path relative to config file's parent
    cal_path = Path(calibration_path)
    if not cal_path.is_absolute():
        cal_path = config_path.parent / cal_path
    calibration_path_str = str(cal_path)

    # Build LUT config from YAML (satisfies LutConfigLike protocol)
    lut_raw = raw.get("lut", {})
    lut_config = _LutConfigFromDict(lut_raw)

    # Check for existing LUTs (skip unless --force)
    if not force:
        fwd = load_forward_luts(calibration_path_str, lut_config)
        inv = load_inverse_luts(calibration_path_str, lut_config)
        if fwd is not None and inv is not None:
            lut_dir = cal_path.parent / "luts"
            click.echo(f"LUTs already exist at {lut_dir}. Use --force to regenerate.")
            return

    if not cal_path.exists():
        raise click.ClickException(f"Calibration file not found: {cal_path}")

    # Load calibration and compute undistortion maps
    calibration = load_calibration_data(calibration_path_str)
    undistortion_maps = {
        cam_id: compute_undistortion_maps(calibration.cameras[cam_id])
        for cam_id in calibration.ring_cameras
    }

    # Generate forward LUTs
    click.echo("Generating forward LUTs...")
    forward_luts = generate_forward_luts(
        calibration, lut_config, undistortion_maps=undistortion_maps
    )
    save_forward_luts(forward_luts, calibration_path_str, lut_config)
    click.echo(f"Saved forward LUTs for {len(forward_luts)} cameras.")

    # Generate inverse LUT
    click.echo("Generating inverse LUT...")
    inverse_lut = generate_inverse_lut(
        calibration, lut_config, undistortion_maps=undistortion_maps
    )
    save_inverse_luts(inverse_lut, calibration_path_str, lut_config)

    lut_dir = cal_path.parent / "luts"
    click.echo(f"LUT generation complete. Saved to: {lut_dir}")
