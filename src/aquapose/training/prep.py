"""Calibration data preparation CLI for AquaPose training.

Provides the ``prep`` CLI group with subcommands for computing derived
configuration values from annotation data. Currently supports:

- ``calibrate-keypoints``: Compute keypoint t-values from COCO annotations.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import yaml

__all__ = ["prep_group"]


@click.group("prep")
def prep_group() -> None:
    """Prepare calibration data and derived configuration."""


@prep_group.command("calibrate-keypoints")
@click.option(
    "--annotations",
    required=True,
    type=click.Path(exists=True),
    help="Path to COCO keypoint annotations JSON.",
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
    """Compute keypoint t-values from COCO keypoint annotations.

    Reads a COCO-format JSON with keypoint annotations and computes the
    arc-length fraction (t-value) for each anatomical keypoint, averaged
    across all annotated instances. The resulting t-values represent each
    keypoint's position along the fish body curve (0.0 = nose, 1.0 = tail).

    Updates the pipeline config YAML in place, setting
    ``midline.keypoint_t_values`` to the computed values.

    Args:
        annotations: Path to COCO keypoint annotations JSON.
        config: Path to pipeline config YAML to update in place.
        n_keypoints: Number of anatomical keypoints expected per annotation.
    """
    annotations_path = Path(annotations)
    config_path = Path(config)

    with annotations_path.open() as fh:
        coco = json.load(fh)

    # Collect per-keypoint t-values across all annotations
    all_t_values: list[list[float]] = []
    n_processed = 0

    for ann in coco.get("annotations", []):
        raw_kps = ann.get("keypoints")
        if not raw_kps:
            continue

        # COCO keypoints: [x, y, v, x, y, v, ...]
        # Extract visible keypoints (v > 0)
        points: list[tuple[float, float]] = []
        for k in range(n_keypoints):
            base = k * 3
            if base + 2 >= len(raw_kps):
                break
            x, y, v = raw_kps[base], raw_kps[base + 1], raw_kps[base + 2]
            if v > 0:
                points.append((float(x), float(y)))
            else:
                # Use NaN to mark invisible keypoints
                points.append((float("nan"), float("nan")))

        if len(points) < 2:
            continue

        # Compute arc-length fractions for visible keypoints only
        # Build sequence of positions for visible keypoints
        visible_indices = [i for i, (x, _) in enumerate(points) if not np.isnan(x)]

        if len(visible_indices) < 2:
            continue

        visible_pts = np.array([points[i] for i in visible_indices], dtype=np.float64)

        # Compute cumulative arc length between consecutive visible keypoints
        diffs = np.diff(visible_pts, axis=0)
        dists = np.sqrt((diffs**2).sum(axis=1))
        cumulative = np.concatenate([[0.0], np.cumsum(dists)])
        total_length = cumulative[-1]

        if total_length < 1e-6:
            continue

        # Assign t-values by visible keypoint index in [0, 1]
        # Fill full t-value array with NaN for invisible keypoints
        t_per_annotation = [float("nan")] * n_keypoints
        for j, idx in enumerate(visible_indices):
            t_per_annotation[idx] = cumulative[j] / total_length

        all_t_values.append(t_per_annotation)
        n_processed += 1

    if n_processed == 0:
        raise click.ClickException(
            f"No valid keypoint annotations found in {annotations_path}. "
            "Ensure annotations have keypoints with visibility > 0."
        )

    # Average t-values across all annotations (ignoring NaN)
    t_array = np.array(all_t_values, dtype=np.float64)
    mean_t = np.nanmean(t_array, axis=0)

    # Clip to [0, 1] and round for readability
    mean_t = np.clip(mean_t, 0.0, 1.0)
    t_values_list = [round(float(t), 4) for t in mean_t]

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
