"""CLI command for pseudo-label generation from diagnostic caches."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import click
import cv2
import numpy as np

from aquapose.training.geometry import (
    affine_warp_crop,
    format_pose_annotation,
    pca_obb,
    transform_keypoints,
)
from aquapose.training.pseudo_labels import (
    detect_gaps,
    generate_fish_labels,
    generate_gap_fish_labels,
)

logger = logging.getLogger(__name__)


@dataclass
class PseudoLabelConfig:
    """Configuration for pseudo-label generation."""

    lateral_pad: float = 40.0
    max_camera_residual_px: float = 15.0
    edge_factor: float = 2.0


class _LutConfigFromDict:
    """Minimal LUT config satisfying the ``LutConfigLike`` protocol.

    Built from a plain dict (YAML ``lut`` section) to avoid importing
    ``aquapose.engine.config`` (which would violate the training->engine
    import boundary). Pattern established in ``training/prep.py``.
    """

    def __init__(self, d: dict) -> None:
        self.tank_diameter: float = float(d.get("tank_diameter", 1.0))
        self.tank_height: float = float(d.get("tank_height", 0.5))
        self.voxel_resolution_m: float = float(d.get("voxel_resolution_m", 0.01))
        self.margin_fraction: float = float(d.get("margin_fraction", 0.1))
        self.forward_grid_step: int = int(d.get("forward_grid_step", 4))


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
    help="Per-camera residual threshold in pixels (consensus only).",
)
@click.option(
    "--consensus",
    is_flag=True,
    default=False,
    help="Generate Source A consensus labels.",
)
@click.option(
    "--gaps",
    is_flag=True,
    default=False,
    help="Generate Source B gap-fill labels.",
)
@click.option(
    "--min-cameras",
    type=int,
    default=3,
    show_default=True,
    help="Minimum contributing cameras before gap detection activates.",
)
@click.option(
    "--crop-width",
    type=int,
    default=128,
    show_default=True,
    help="Pose crop width in pixels.",
)
@click.option(
    "--crop-height",
    type=int,
    default=64,
    show_default=True,
    help="Pose crop height in pixels.",
)
@click.option(
    "--temporal-step",
    type=int,
    default=1,
    show_default=True,
    help="Process every Nth frame (1 = all frames).",
)
def generate(
    config_path: str,
    lateral_pad: float,
    max_camera_residual: float,
    consensus: bool,
    gaps: bool,
    min_cameras: int,
    crop_width: int,
    crop_height: int,
    temporal_step: int,
) -> None:
    """Generate YOLO OBB and pose pseudo-labels from diagnostic caches.

    Reads a pipeline run's frozen config, loads diagnostic caches containing
    3D midline reconstructions, reprojects them into camera views, extracts
    video frames, and writes YOLO-standard OBB and pose datasets.

    At least one of ``--consensus`` or ``--gaps`` must be specified.
    """
    if not consensus and not gaps:
        raise click.ClickException("At least one of --consensus or --gaps is required.")

    # Dynamic imports to avoid training->engine import boundary violation
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

    # Fail fast for gaps: need detections and tracks_2d
    if gaps:
        if context.detections is None:
            raise click.ClickException(
                "Diagnostic caches contain no detections. "
                "Gap detection requires detection data."
            )
        if context.tracks_2d is None:
            raise click.ClickException(
                "Diagnostic caches contain no 2D tracks. "
                "Gap detection requires tracking data."
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

    # 7. Load InverseLUT if gaps mode is active
    inverse_lut = None
    if gaps:
        _luts_mod = importlib.import_module("aquapose.calibration.luts")
        load_inverse_luts = _luts_mod.load_inverse_luts

        # Build LUT config from frozen config's lut section
        lut_raw = pipeline_config.lut.__dict__
        lut_config = _LutConfigFromDict(lut_raw)
        inverse_lut = load_inverse_luts(pipeline_config.calibration_path, lut_config)
        if inverse_lut is None:
            raise click.ClickException(
                "Failed to load InverseLUT. Run 'aquapose prep generate-luts' first."
            )

        # Pre-build frame-to-tracklet index per camera for O(1) lookup
        frame_tracklet_index: dict[str, dict[int, list]] = {}
        assert context.tracks_2d is not None  # validated above
        for cam_id, tracklets in context.tracks_2d.items():
            cam_index: dict[int, list] = {}
            for tracklet in tracklets:
                for frame_i in tracklet.frames:
                    cam_index.setdefault(frame_i, []).append(tracklet)
            frame_tracklet_index[cam_id] = cam_index

    # 8. Create output directories
    pseudo_dir = run_dir / "pseudo_labels"

    # Consensus directories
    cons_obb_images = pseudo_dir / "consensus" / "obb" / "images" / "train"
    cons_obb_labels = pseudo_dir / "consensus" / "obb" / "labels" / "train"
    cons_pose_images = pseudo_dir / "consensus" / "pose" / "images" / "train"
    cons_pose_labels = pseudo_dir / "consensus" / "pose" / "labels" / "train"
    if consensus:
        for d in [cons_obb_images, cons_obb_labels, cons_pose_images, cons_pose_labels]:
            d.mkdir(parents=True, exist_ok=True)

    # Gap directories
    gap_obb_images = pseudo_dir / "gap" / "obb" / "images" / "train"
    gap_obb_labels = pseudo_dir / "gap" / "obb" / "labels" / "train"
    gap_pose_images = pseudo_dir / "gap" / "pose" / "images" / "train"
    gap_pose_labels = pseudo_dir / "gap" / "pose" / "labels" / "train"
    if gaps:
        for d in [gap_obb_images, gap_obb_labels, gap_pose_images, gap_pose_labels]:
            d.mkdir(parents=True, exist_ok=True)

    # 9. Iterate frames and generate labels
    cons_confidence: dict[str, dict] = {}
    gap_confidence: dict[str, dict] = {}
    cons_images_count = 0
    cons_labels_count = 0
    gap_images_count = 0
    gap_labels_count = 0
    cons_confidence_values: list[float] = []
    gap_confidence_values: list[float] = []

    n_keypoints = len(keypoint_t_values)

    with VideoFrameSource(
        pipeline_config.video_dir,
        pipeline_config.calibration_path,
    ) as frame_source:
        n_frames = len(context.midlines_3d)
        n_selected = (n_frames + temporal_step - 1) // temporal_step if temporal_step > 1 else n_frames
        step_msg = f" (every {temporal_step}th)" if temporal_step > 1 else ""
        click.echo(f"Processing {n_selected}/{n_frames} frames{step_msg} across {len(proj_models)} cameras...")
        if consensus:
            click.echo("  Generating consensus (Source A) labels")
        if gaps:
            click.echo("  Generating gap (Source B) labels")

        for frame_idx, fish_dict in enumerate(context.midlines_3d):
            if not fish_dict:
                continue

            if temporal_step > 1 and frame_idx % temporal_step != 0:
                continue

            # Collect labels per (frame_idx, cam_id) for consensus
            cons_image_labels: dict[str, dict] = {}
            # Collect labels per (frame_idx, cam_id) for gap
            gap_image_labels: dict[str, dict] = {}

            for fish_id, midline in fish_dict.items():
                # --- Consensus path ---
                if consensus and midline.per_camera_residuals is not None:
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

                        if cam_id not in cons_image_labels:
                            cons_image_labels[cam_id] = {
                                "obb_lines": [],
                                "pose_lines": [],
                                "conf_entries": [],
                                "fish_data": [],
                            }

                        cons_image_labels[cam_id]["obb_lines"].append(
                            result["obb_line"]
                        )
                        cons_image_labels[cam_id]["pose_lines"].append(
                            result["pose_line"]
                        )
                        cons_image_labels[cam_id]["conf_entries"].append(
                            {
                                "fish_id": int(fish_id),
                                "confidence": result["confidence"],
                                "raw_metrics": result["raw_metrics"],
                            }
                        )
                        cons_image_labels[cam_id]["fish_data"].append(
                            {
                                "keypoints_2d": result["keypoints_2d"],
                                "visibility": result["visibility"],
                                "fish_id": int(fish_id),
                            }
                        )

                # --- Gap path ---
                if gaps and inverse_lut is not None:
                    # Build frame-specific tracklet lookup for this frame
                    frame_tracks_for_frame: dict[str, list] = {}
                    for cam_id_t, cam_index in frame_tracklet_index.items():
                        tracklets_at_frame = cam_index.get(frame_idx, [])
                        if tracklets_at_frame:
                            frame_tracks_for_frame[cam_id_t] = tracklets_at_frame

                    frame_dets = (
                        context.detections[frame_idx]
                        if context.detections is not None
                        else {}
                    )

                    gap_list = detect_gaps(
                        midline,
                        inverse_lut,
                        proj_models,
                        frame_dets,
                        frame_tracks_for_frame,
                        min_cameras=min_cameras,
                    )

                    for cam_id, reason in gap_list:
                        if cam_id not in proj_models:
                            continue

                        cam_data = calibration.cameras[cam_id]
                        img_w, img_h = cam_data.image_size

                        gap_result = generate_gap_fish_labels(
                            midline=midline,
                            projection_model=proj_models[cam_id],
                            img_w=img_w,
                            img_h=img_h,
                            keypoint_t_values=keypoint_t_values,
                            lateral_pad=lateral_pad,
                        )

                        if gap_result is None:
                            continue

                        if cam_id not in gap_image_labels:
                            gap_image_labels[cam_id] = {
                                "obb_lines": [],
                                "pose_lines": [],
                                "conf_entries": [],
                                "fish_data": [],
                            }

                        gap_image_labels[cam_id]["obb_lines"].append(
                            gap_result["obb_line"]
                        )
                        gap_image_labels[cam_id]["pose_lines"].append(
                            gap_result["pose_line"]
                        )
                        gap_image_labels[cam_id]["conf_entries"].append(
                            {
                                "fish_id": int(fish_id),
                                "confidence": gap_result["confidence"],
                                "raw_metrics": gap_result["raw_metrics"],
                                "gap_reason": reason,
                                "n_source_cameras": midline.n_cameras,
                            }
                        )
                        gap_image_labels[cam_id]["fish_data"].append(
                            {
                                "keypoints_2d": gap_result["keypoints_2d"],
                                "visibility": gap_result["visibility"],
                                "fish_id": int(fish_id),
                            }
                        )

            # Determine which cameras need frame images
            cams_needing_frames = set(cons_image_labels.keys()) | set(
                gap_image_labels.keys()
            )
            if not cams_needing_frames:
                continue

            # Read frame images once (shared between consensus and gap)
            try:
                frames = frame_source.read_frame(frame_idx)
            except Exception:
                logger.warning("Failed to read frame %d, skipping", frame_idx)
                continue

            # Write consensus images and labels
            for cam_id, labels in cons_image_labels.items():
                if cam_id not in frames:
                    continue

                image_name = f"{frame_idx:06d}_{cam_id}"

                # OBB: full-frame image + full-frame labels (unchanged)
                cv2.imwrite(str(cons_obb_images / f"{image_name}.jpg"), frames[cam_id])
                (cons_obb_labels / f"{image_name}.txt").write_text(
                    "\n".join(labels["obb_lines"]) + "\n"
                )

                # Pose: per-fish OBB crops with crop-space keypoints
                _write_pose_crops(
                    frame=frames[cam_id],
                    fish_data_list=labels["fish_data"],
                    frame_idx=frame_idx,
                    cam_id=cam_id,
                    crop_w=crop_width,
                    crop_h=crop_height,
                    lateral_pad=lateral_pad,
                    pose_images_dir=cons_pose_images,
                    pose_labels_dir=cons_pose_labels,
                )

                cons_confidence[image_name] = {"labels": labels["conf_entries"]}
                cons_images_count += 1
                cons_labels_count += len(labels["obb_lines"])
                for entry in labels["conf_entries"]:
                    cons_confidence_values.append(entry["confidence"])

            # Write gap images and labels
            for cam_id, labels in gap_image_labels.items():
                if cam_id not in frames:
                    continue

                image_name = f"{frame_idx:06d}_{cam_id}"

                # OBB: full-frame image + full-frame labels (unchanged)
                cv2.imwrite(str(gap_obb_images / f"{image_name}.jpg"), frames[cam_id])
                (gap_obb_labels / f"{image_name}.txt").write_text(
                    "\n".join(labels["obb_lines"]) + "\n"
                )

                # Pose: per-fish OBB crops with crop-space keypoints
                _write_pose_crops(
                    frame=frames[cam_id],
                    fish_data_list=labels["fish_data"],
                    frame_idx=frame_idx,
                    cam_id=cam_id,
                    crop_w=crop_width,
                    crop_h=crop_height,
                    lateral_pad=lateral_pad,
                    pose_images_dir=gap_pose_images,
                    pose_labels_dir=gap_pose_labels,
                )

                gap_confidence[image_name] = {"labels": labels["conf_entries"]}
                gap_images_count += 1
                gap_labels_count += len(labels["obb_lines"])
                for entry in labels["conf_entries"]:
                    gap_confidence_values.append(entry["confidence"])

            if (frame_idx + 1) % 100 == 0:
                click.echo(f"  Frame {frame_idx + 1}/{n_frames}...")

    # 10. Write dataset.yaml and confidence.json for each subset
    if consensus:
        _write_dataset_yamls(
            pseudo_dir / "consensus",
            n_keypoints,
            cons_confidence,
            cons_confidence_values,
        )
        _write_confidence_json(pseudo_dir / "consensus", cons_confidence)

    if gaps:
        _write_dataset_yamls(
            pseudo_dir / "gap", n_keypoints, gap_confidence, gap_confidence_values
        )
        _write_confidence_json(pseudo_dir / "gap", gap_confidence)

    # 11. Print summary
    click.echo("\nPseudo-label generation complete.")
    if consensus:
        mean_conf = (
            float(np.mean(cons_confidence_values)) if cons_confidence_values else 0.0
        )
        click.echo(
            f"  Consensus: {cons_images_count} images, {cons_labels_count} labels, mean confidence {mean_conf:.3f}"
        )
    if gaps:
        mean_conf = (
            float(np.mean(gap_confidence_values)) if gap_confidence_values else 0.0
        )
        click.echo(
            f"  Gap: {gap_images_count} images, {gap_labels_count} labels, mean confidence {mean_conf:.3f}"
        )
    click.echo(f"  Output: {pseudo_dir}")


def _write_pose_crops(
    frame: np.ndarray,
    fish_data_list: list[dict],
    frame_idx: int,
    cam_id: str,
    crop_w: int,
    crop_h: int,
    lateral_pad: float,
    pose_images_dir: Path,
    pose_labels_dir: Path,
    min_visible: int = 2,
) -> None:
    """Write per-fish OBB crop images and crop-space pose labels.

    For each fish (the "primary"), computes the OBB from its keypoints,
    warps the frame to crop space, then collects pose annotations for ALL
    fish visible in that crop (multi-fish-per-crop logic matching
    ``scripts/build_yolo_training_data.py``).

    Args:
        frame: Full-frame BGR image.
        fish_data_list: List of dicts with ``keypoints_2d``, ``visibility``,
            ``fish_id`` for each fish in this camera view.
        frame_idx: Frame index (for filename).
        cam_id: Camera identifier (for filename).
        crop_w: Output crop width in pixels.
        crop_h: Output crop height in pixels.
        lateral_pad: OBB lateral padding in pixels.
        pose_images_dir: Output directory for crop images.
        pose_labels_dir: Output directory for crop label files.
        min_visible: Minimum visible keypoints for a fish to appear in crop.
    """
    for fish_idx, primary in enumerate(fish_data_list):
        kp_primary = primary["keypoints_2d"]
        vis_primary = primary["visibility"]

        # Compute OBB for the primary fish in image space
        obb_corners = pca_obb(kp_primary, vis_primary, lateral_pad)

        # Warp frame to crop
        warped, affine_mat = affine_warp_crop(frame, obb_corners, crop_w, crop_h)

        # Collect pose annotations for all fish visible in this crop
        pose_rows: list[str] = []
        for other in fish_data_list:
            kp_crop, vis_crop = transform_keypoints(
                other["keypoints_2d"],
                other["visibility"],
                affine_mat,
                crop_w,
                crop_h,
            )
            if int(vis_crop.sum()) < min_visible:
                continue

            # Compute PCA OBB on crop-space keypoints for the bbox
            crop_obb = pca_obb(kp_crop, vis_crop, lateral_pad)
            crop_obb[:, 0] = np.clip(crop_obb[:, 0], 0, crop_w - 1)
            crop_obb[:, 1] = np.clip(crop_obb[:, 1], 0, crop_h - 1)
            x_min, y_min = crop_obb.min(axis=0)
            x_max, y_max = crop_obb.max(axis=0)
            cx = (x_min + x_max) / 2.0 / crop_w
            cy = (y_min + y_max) / 2.0 / crop_h
            bw = (x_max - x_min) / crop_w
            bh = (y_max - y_min) / crop_h

            row = format_pose_annotation(
                cx, cy, bw, bh, kp_crop, vis_crop, crop_w, crop_h
            )
            pose_rows.append(" ".join(str(v) for v in row))

        if not pose_rows:
            continue

        crop_stem = f"{frame_idx:06d}_{cam_id}_{fish_idx:03d}"
        cv2.imwrite(str(pose_images_dir / f"{crop_stem}.jpg"), warped)
        (pose_labels_dir / f"{crop_stem}.txt").write_text("\n".join(pose_rows) + "\n")


def _write_dataset_yamls(
    subset_dir: Path,
    n_keypoints: int,
    confidence_sidecar: dict[str, dict],
    confidence_values: list[float],
) -> None:
    """Write OBB and pose dataset.yaml files for a subset directory."""
    # OBB dataset.yaml
    obb_dataset = {
        "path": str(subset_dir / "obb"),
        "train": "images/train",
        "nc": 1,
        "names": {0: "fish"},
    }
    (subset_dir / "obb" / "dataset.yaml").write_text(
        _yaml_dump(obb_dataset), encoding="utf-8"
    )

    # Pose dataset.yaml
    flip_idx = list(range(n_keypoints))
    pose_dataset = {
        "path": str(subset_dir / "pose"),
        "train": "images/train",
        "nc": 1,
        "names": {0: "fish"},
        "kpt_shape": [n_keypoints, 3],
        "flip_idx": flip_idx,
    }
    (subset_dir / "pose" / "dataset.yaml").write_text(
        _yaml_dump(pose_dataset), encoding="utf-8"
    )


def _write_confidence_json(
    subset_dir: Path,
    confidence_sidecar: dict[str, dict],
) -> None:
    """Write confidence.json sidecar for a subset directory."""
    confidence_path = subset_dir / "confidence.json"
    confidence_path.write_text(
        json.dumps(confidence_sidecar, indent=2), encoding="utf-8"
    )


def _yaml_dump(data: dict) -> str:
    """Dump dict to YAML string."""
    import yaml

    return yaml.dump(data, default_flow_style=False, sort_keys=False)


@pseudo_label_group.command("inspect")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="YOLO-format directory containing images/ and labels/ (e.g. pseudo_labels/consensus/obb).",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(),
    help="Output directory for annotated PNGs. Defaults to {data-dir}/inspect/.",
)
@click.option(
    "-n",
    "n_samples",
    default=None,
    type=int,
    help="Number of images to sample randomly. Omit to inspect all.",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    show_default=True,
    help="Random seed for sampling.",
)
def inspect(
    data_dir: str,
    output_dir: str | None,
    n_samples: int | None,
    seed: int,
) -> None:
    """Visualize pseudo-labels by drawing annotations on images.

    Reads YOLO-format images and labels, draws OBB polygons or pose
    keypoints on the images, and writes annotated PNGs. If a
    confidence.json sidecar is found, overlays confidence scores,
    source type, and gap reasons as text.
    """
    import random

    data_path = Path(data_dir)
    out_path = Path(output_dir) if output_dir else data_path / "inspect"
    out_path.mkdir(parents=True, exist_ok=True)

    # Find image/label directories (support both flat and train/ subdirectory)
    images_dir = data_path / "images" / "train"
    labels_dir = data_path / "labels" / "train"
    if not images_dir.exists():
        images_dir = data_path / "images"
    if not labels_dir.exists():
        labels_dir = data_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise click.ClickException(
            f"Expected images/ and labels/ subdirectories in {data_path}"
        )

    # Collect image files that have matching label files
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and (labels_dir / f"{p.stem}.txt").exists()
    )

    if not image_files:
        raise click.ClickException(f"No image/label pairs found in {data_path}")

    # Sample if requested
    if n_samples is not None and n_samples < len(image_files):
        rng = random.Random(seed)
        image_files = rng.sample(image_files, n_samples)

    # Load confidence sidecar if available
    confidence_data: dict = {}
    for candidate in [data_path / "confidence.json", data_path.parent / "confidence.json"]:
        if candidate.exists():
            confidence_data = json.loads(candidate.read_text())
            break

    # Detect label type from first label file
    first_label = (labels_dir / f"{image_files[0].stem}.txt").read_text().strip()
    first_tokens = first_label.split("\n")[0].split()
    is_obb = len(first_tokens) == 9  # cls + 4 corners x 2

    click.echo(
        f"Inspecting {len(image_files)} images "
        f"({'OBB' if is_obb else 'pose'} labels)"
    )

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Failed to read %s, skipping", img_path)
            continue

        img_h, img_w = image.shape[:2]
        label_path = labels_dir / f"{img_path.stem}.txt"
        label_text = label_path.read_text().strip()

        if not label_text:
            continue

        # Look up confidence metadata
        conf_entry = confidence_data.get(img_path.stem, {})
        fish_labels = conf_entry.get("labels", [])

        for line_idx, line in enumerate(label_text.split("\n")):
            tokens = line.split()
            if not tokens:
                continue

            # Get per-fish metadata if available
            fish_meta = fish_labels[line_idx] if line_idx < len(fish_labels) else {}
            confidence = fish_meta.get("confidence")
            gap_reason = fish_meta.get("gap_reason")

            if is_obb:
                _draw_obb_inspection(image, tokens, img_w, img_h)
            else:
                _draw_pose_inspection(image, tokens, img_w, img_h)

            # Build text overlay
            text_parts: list[str] = []
            if confidence is not None:
                text_parts.append(f"conf={confidence:.2f}")
            if gap_reason is not None:
                text_parts.append(f"gap:{gap_reason}")
            elif fish_meta:
                text_parts.append("consensus")

            if text_parts:
                text = " | ".join(text_parts)
                # Position text near the annotation
                if is_obb:
                    tx = int(float(tokens[1]) * img_w)
                    ty = max(15, int(float(tokens[2]) * img_h) - 10)
                else:
                    tx = max(5, int(float(tokens[1]) * img_w - float(tokens[3]) * img_w / 2))
                    ty = max(15, int(float(tokens[2]) * img_h - float(tokens[4]) * img_h / 2) - 10)

                cv2.putText(
                    image, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2,
                )
                cv2.putText(
                    image, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1,
                )

        cv2.imwrite(str(out_path / f"{img_path.stem}.png"), image)

    click.echo(f"Wrote {len(image_files)} annotated images to {out_path}")


def _draw_obb_inspection(
    image: np.ndarray,
    tokens: list[str],
    img_w: int,
    img_h: int,
) -> None:
    """Draw an OBB polygon from a YOLO-OBB label line."""
    # tokens: cls x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
    corners = np.array(
        [
            [float(tokens[1]) * img_w, float(tokens[2]) * img_h],
            [float(tokens[3]) * img_w, float(tokens[4]) * img_h],
            [float(tokens[5]) * img_w, float(tokens[6]) * img_h],
            [float(tokens[7]) * img_w, float(tokens[8]) * img_h],
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)


def _draw_pose_inspection(
    image: np.ndarray,
    tokens: list[str],
    img_w: int,
    img_h: int,
) -> None:
    """Draw a pose bbox and keypoints from a YOLO-Pose label line."""
    # tokens: cls cx cy w h x1 y1 v1 x2 y2 v2 ...
    cx = float(tokens[1]) * img_w
    cy = float(tokens[2]) * img_h
    bw = float(tokens[3]) * img_w
    bh = float(tokens[4]) * img_h
    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw keypoints
    kp_tokens = tokens[5:]
    for i in range(0, len(kp_tokens), 3):
        if i + 2 >= len(kp_tokens):
            break
        kx = float(kp_tokens[i]) * img_w
        ky = float(kp_tokens[i + 1]) * img_h
        vis = int(float(kp_tokens[i + 2]))
        if vis > 0:
            color = (0, 0, 255) if i == 0 else (0, 255, 0)
            cv2.circle(image, (int(kx), int(ky)), 4, color, -1)


@pseudo_label_group.command("assemble")
@click.option(
    "--run-dir",
    "run_dirs",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Pipeline run directory containing pseudo_labels/. Can be repeated.",
)
@click.option(
    "--manual-dir",
    default=None,
    type=click.Path(exists=True),
    help="YOLO-format manual annotation directory.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Output directory for assembled dataset.",
)
@click.option(
    "--model-type",
    required=True,
    type=click.Choice(["obb", "pose"]),
    help="Which label type to assemble.",
)
@click.option(
    "--consensus-threshold",
    default=0.5,
    type=float,
    show_default=True,
    help="Min confidence for consensus (Source A) labels.",
)
@click.option(
    "--gap-threshold",
    default=0.3,
    type=float,
    show_default=True,
    help="Min confidence for gap (Source B) labels.",
)
@click.option(
    "--exclude-gap-reason",
    "exclude_gap_reasons",
    multiple=True,
    type=str,
    help="Gap reasons to exclude (e.g. 'no-tracklet'). Can be repeated.",
)
@click.option(
    "--manual-val-fraction",
    default=0.2,
    type=float,
    show_default=True,
    help="Fraction of manual data for validation.",
)
@click.option(
    "--pseudo-val-fraction",
    default=0.1,
    type=float,
    show_default=True,
    help="Fraction of pseudo-labels held out for evaluation.",
)
@click.option(
    "--temporal-step",
    default=1,
    type=int,
    show_default=True,
    help="Temporal subsampling step (1 = no subsampling).",
)
@click.option(
    "--diversity-bins",
    default=5,
    type=int,
    show_default=True,
    help="Number of curvature bins for diversity sampling.",
)
@click.option(
    "--diversity-max-per-bin",
    default=None,
    type=int,
    help="Max frames per curvature bin (None = no limit).",
)
@click.option(
    "--max-frames",
    default=None,
    type=int,
    help="Hard cap on total pseudo-label images after all filtering. Uniform random subsample.",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    show_default=True,
    help="Random seed.",
)
def assemble(
    run_dirs: tuple[str, ...],
    manual_dir: str | None,
    output_dir: str,
    model_type: str,
    consensus_threshold: float,
    gap_threshold: float,
    exclude_gap_reasons: tuple[str, ...],
    manual_val_fraction: float,
    pseudo_val_fraction: float,
    temporal_step: int,
    diversity_bins: int,
    diversity_max_per_bin: int | None,
    max_frames: int | None,
    seed: int,
) -> None:
    """Assemble a YOLO training dataset from manual annotations and pseudo-labels.

    Pools labels from multiple pipeline runs with independent confidence
    thresholds for consensus (Source A) and gap (Source B) labels. Creates
    a ready-to-train YOLO dataset with manual validation split.
    """
    from aquapose.training.dataset_assembly import assemble_dataset

    run_dir_paths = [Path(d) for d in run_dirs]
    manual_dir_path = Path(manual_dir) if manual_dir is not None else None
    output_path = Path(output_dir)

    # Frame selection (optional): filter pseudo-labels by selected frames
    # Only activate if temporal_step > 1 or diversity_max_per_bin is set
    need_frame_selection = temporal_step > 1 or diversity_max_per_bin is not None

    selected_frames: dict[str, set[int]] | None = None

    if need_frame_selection:
        import importlib

        from aquapose.training.frame_selection import (
            diversity_sample,
            filter_empty_frames,
            temporal_subsample,
        )

        _eval_mod = importlib.import_module("aquapose.evaluation.runner")
        load_run_context = _eval_mod.load_run_context

        selected_frames = {}

        # Collect all midlines_3d across runs to build frame selection
        click.echo("Loading diagnostic caches for frame selection...")
        for rd in run_dir_paths:
            context, _meta = load_run_context(rd)
            if context is None or context.midlines_3d is None:
                click.echo(
                    f"  Warning: No midlines_3d in {rd}, skipping frame selection"
                )
                continue

            midlines_3d = context.midlines_3d
            all_frames = list(range(len(midlines_3d)))

            # Step 1: filter empty
            all_frames = filter_empty_frames(all_frames, midlines_3d)
            # Step 2: temporal subsample
            all_frames = temporal_subsample(all_frames, temporal_step)
            # Step 3: diversity sample
            all_frames = diversity_sample(
                midlines_3d,
                all_frames,
                n_bins=diversity_bins,
                max_per_bin=diversity_max_per_bin,
                seed=seed,
            )

            click.echo(f"  {rd.name}: {len(all_frames)} frames selected")
            selected_frames[rd.name] = set(all_frames)

    result = assemble_dataset(
        output_dir=output_path,
        manual_dir=manual_dir_path,
        run_dirs=run_dir_paths,
        model_type=model_type,
        consensus_threshold=consensus_threshold,
        gap_threshold=gap_threshold,
        exclude_gap_reasons=list(exclude_gap_reasons),
        manual_val_fraction=manual_val_fraction,
        pseudo_val_fraction=pseudo_val_fraction,
        seed=seed,
        selected_frames=selected_frames,
        max_frames=max_frames,
    )

    # Print summary
    click.echo("\nDataset assembly complete.")
    click.echo(f"  Output: {output_path}")
    click.echo(f"  Manual train: {result['manual_train']}")
    click.echo(f"  Manual val: {result['manual_val']}")
    click.echo(f"  Consensus train: {result['consensus_train']}")
    click.echo(f"  Consensus val (pseudo): {result['consensus_val']}")
    click.echo(f"  Gap train: {result['gap_train']}")
    click.echo(f"  Gap val (pseudo): {result['gap_val']}")
    total_train = (
        result["manual_train"] + result["consensus_train"] + result["gap_train"]
    )
    total_val = result["manual_val"]
    click.echo(f"  Total train: {total_train}")
    click.echo(f"  Total val (manual): {total_val}")
