"""CLI command for pseudo-label generation from diagnostic caches."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import click
import cv2
import numpy as np

from aquapose.training.pseudo_labels import generate_fish_labels

logger = logging.getLogger(__name__)


@dataclass
class PseudoLabelConfig:
    """Configuration for pseudo-label generation."""

    lateral_pad: float = 40.0
    max_camera_residual_px: float = 15.0
    edge_factor: float = 2.0


@click.group("pseudo-label")
def pseudo_label_group() -> None:
    """Pseudo-label generation from pipeline diagnostic caches."""


@pseudo_label_group.command("generate")
@click.option(
    "--config",
    "-c",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to frozen run config.yaml (stored in the run directory).",
)
@click.option(
    "--lateral-pad",
    type=float,
    default=40.0,
    show_default=True,
    help="OBB lateral padding in pixels.",
)
@click.option(
    "--max-camera-residual",
    type=float,
    default=15.0,
    show_default=True,
    help="Per-camera residual threshold in pixels.",
)
def generate(config_path: str, lateral_pad: float, max_camera_residual: float) -> None:
    """Generate YOLO OBB and pose pseudo-labels from diagnostic caches.

    Reads a pipeline run's frozen config, loads diagnostic caches containing
    3D midline reconstructions, reprojects them into camera views, extracts
    video frames, and writes YOLO-standard OBB and pose datasets.
    """
    # Dynamic imports to avoid training->engine import boundary violation
    # (AST-level boundary check prohibits even lazy `from aquapose.engine` imports)
    import importlib

    _engine_config = importlib.import_module("aquapose.engine.config")
    load_config = _engine_config.load_config

    from aquapose.calibration.loader import (
        compute_undistortion_maps,
        load_calibration_data,
    )
    from aquapose.calibration.projection import RefractiveProjectionModel
    from aquapose.core.types.frame_source import VideoFrameSource
    from aquapose.evaluation.runner import load_run_context

    # 1. Load frozen config
    pipeline_config = load_config(yaml_path=config_path)

    # 2. Fail fast if keypoint_t_values not configured
    keypoint_t_values = pipeline_config.midline.keypoint_t_values
    if keypoint_t_values is None:
        raise click.ClickException(
            "keypoint_t_values is None in config. "
            "Run 'aquapose prep calibrate-keypoints' first to set t-values."
        )

    # 3. Resolve run_dir (frozen config is stored IN the run_dir)
    run_dir = Path(config_path).parent

    # 4. Load diagnostic caches
    context, _metadata = load_run_context(run_dir)
    if context is None:
        raise click.ClickException(
            f"No diagnostic caches found in {run_dir}/diagnostics/. "
            "Run the pipeline with --add-observer diagnostic first."
        )

    if context.midlines_3d is None:
        raise click.ClickException(
            "Diagnostic caches contain no 3D midline reconstructions."
        )

    # 5. Load calibration
    calibration = load_calibration_data(pipeline_config.calibration_path)

    # 6. Build projection models per camera (using undistorted K_new)
    proj_models: dict[str, RefractiveProjectionModel] = {}
    for cam_id in calibration.ring_cameras:
        cam = calibration.cameras[cam_id]
        undist_maps = compute_undistortion_maps(cam)
        proj_models[cam_id] = RefractiveProjectionModel(
            K=undist_maps.K_new,
            R=cam.R,
            t=cam.t,
            water_z=calibration.water_z,
            normal=calibration.interface_normal,
            n_air=calibration.n_air,
            n_water=calibration.n_water,
        )

    # 7. Create output directories
    pseudo_dir = run_dir / "pseudo_labels"
    obb_images_dir = pseudo_dir / "obb" / "images" / "train"
    obb_labels_dir = pseudo_dir / "obb" / "labels" / "train"
    pose_images_dir = pseudo_dir / "pose" / "images" / "train"
    pose_labels_dir = pseudo_dir / "pose" / "labels" / "train"

    for d in [obb_images_dir, obb_labels_dir, pose_images_dir, pose_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 8. Iterate frames and generate labels
    confidence_sidecar: dict[str, dict] = {}
    total_images = 0
    total_labels = 0
    confidence_values: list[float] = []

    n_keypoints = len(keypoint_t_values)

    with VideoFrameSource(
        pipeline_config.video_dir,
        pipeline_config.calibration_path,
    ) as frame_source:
        n_frames = len(context.midlines_3d)
        click.echo(f"Processing {n_frames} frames across {len(proj_models)} cameras...")

        for frame_idx, fish_dict in enumerate(context.midlines_3d):
            if not fish_dict:
                continue

            # Collect labels per (frame_idx, cam_id)
            image_labels: dict[
                str, dict
            ] = {}  # cam_id -> {obb_lines, pose_lines, conf_entries}

            for fish_id, midline in fish_dict.items():
                if midline.per_camera_residuals is None:
                    continue

                for cam_id in midline.per_camera_residuals:
                    if cam_id not in proj_models:
                        continue

                    cam_data = calibration.cameras[cam_id]
                    img_w, img_h = cam_data.image_size

                    result = generate_fish_labels(
                        midline=midline,
                        projection_model=proj_models[cam_id],
                        img_w=img_w,
                        img_h=img_h,
                        keypoint_t_values=keypoint_t_values,
                        lateral_pad=lateral_pad,
                        max_camera_residual_px=max_camera_residual,
                        camera_id=cam_id,
                    )

                    if result is None:
                        continue

                    if cam_id not in image_labels:
                        image_labels[cam_id] = {
                            "obb_lines": [],
                            "pose_lines": [],
                            "conf_entries": [],
                        }

                    image_labels[cam_id]["obb_lines"].append(result["obb_line"])
                    image_labels[cam_id]["pose_lines"].append(result["pose_line"])
                    image_labels[cam_id]["conf_entries"].append(
                        {
                            "fish_id": int(fish_id),
                            "confidence": result["confidence"],
                            "raw_metrics": result["raw_metrics"],
                        }
                    )

            if not image_labels:
                continue

            # Read frame images (only if we have labels for this frame)
            try:
                frames = frame_source.read_frame(frame_idx)
            except Exception:
                logger.warning("Failed to read frame %d, skipping", frame_idx)
                continue

            # Write images and labels
            for cam_id, labels in image_labels.items():
                if cam_id not in frames:
                    continue

                image_name = f"{frame_idx:06d}_{cam_id}"

                # Save image
                img_path = obb_images_dir / f"{image_name}.jpg"
                cv2.imwrite(str(img_path), frames[cam_id])
                # Symlink for pose (same image, avoid duplication on disk)
                pose_img_path = pose_images_dir / f"{image_name}.jpg"
                if not pose_img_path.exists():
                    cv2.imwrite(str(pose_img_path), frames[cam_id])

                # Write OBB label file
                obb_label_path = obb_labels_dir / f"{image_name}.txt"
                obb_label_path.write_text("\n".join(labels["obb_lines"]) + "\n")

                # Write pose label file
                pose_label_path = pose_labels_dir / f"{image_name}.txt"
                pose_label_path.write_text("\n".join(labels["pose_lines"]) + "\n")

                # Record confidence
                confidence_sidecar[image_name] = {"labels": labels["conf_entries"]}

                total_images += 1
                total_labels += len(labels["obb_lines"])
                for entry in labels["conf_entries"]:
                    confidence_values.append(entry["confidence"])

            if (frame_idx + 1) % 100 == 0:
                click.echo(f"  Frame {frame_idx + 1}/{n_frames}...")

    # 9. Write dataset.yaml for OBB
    obb_dataset = {
        "path": str(pseudo_dir / "obb"),
        "train": "images/train",
        "nc": 1,
        "names": {0: "fish"},
    }
    (pseudo_dir / "obb" / "dataset.yaml").write_text(
        _yaml_dump(obb_dataset), encoding="utf-8"
    )

    # 10. Write dataset.yaml for pose
    # flip_idx: symmetric keypoint pairs (none for fish midline)
    flip_idx = list(range(n_keypoints))  # identity = no symmetry
    pose_dataset = {
        "path": str(pseudo_dir / "pose"),
        "train": "images/train",
        "nc": 1,
        "names": {0: "fish"},
        "kpt_shape": [n_keypoints, 3],
        "flip_idx": flip_idx,
    }
    (pseudo_dir / "pose" / "dataset.yaml").write_text(
        _yaml_dump(pose_dataset), encoding="utf-8"
    )

    # 11. Write confidence sidecar
    confidence_path = pseudo_dir / "confidence.json"
    confidence_path.write_text(
        json.dumps(confidence_sidecar, indent=2), encoding="utf-8"
    )

    # 12. Print summary
    mean_conf = float(np.mean(confidence_values)) if confidence_values else 0.0
    click.echo("\nPseudo-label generation complete.")
    click.echo(f"  Images: {total_images}")
    click.echo(f"  Labels: {total_labels}")
    click.echo(f"  Mean confidence: {mean_conf:.3f}")
    click.echo(f"  Output: {pseudo_dir}")


def _yaml_dump(data: dict) -> str:
    """Dump dict to YAML string."""
    import yaml

    return yaml.dump(data, default_flow_style=False, sort_keys=False)
