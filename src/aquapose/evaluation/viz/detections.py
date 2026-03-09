"""Detection overlay visualization: OBB bounding boxes colored by confidence."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from aquapose.evaluation.viz._loader import load_all_chunk_caches, read_config_yaml

logger = logging.getLogger(__name__)


def _confidence_color(conf: float) -> tuple[int, int, int]:
    """Map confidence [0, 1] to a BGR color from red (low) to green (high).

    Args:
        conf: Detection confidence in [0, 1].

    Returns:
        BGR color tuple.
    """
    # Red (0,0,255) -> Yellow (0,255,255) -> Green (0,255,0)
    conf = max(0.0, min(1.0, conf))
    if conf < 0.5:
        t = conf / 0.5
        return (0, int(255 * t), 255)
    t = (conf - 0.5) / 0.5
    return (0, 255, int(255 * (1 - t)))


def _render_detection_mosaic(
    frame_dets: dict[str, list],
    frames: dict[str, np.ndarray],
    camera_ids: list[str],
    global_frame_idx: int,
) -> np.ndarray:
    """Render a mosaic image with OBB detections drawn on camera frames.

    Args:
        frame_dets: Per-camera detection lists for this frame.
        frames: Per-camera BGR images.
        camera_ids: Ordered camera IDs.
        global_frame_idx: Global frame index (for the title label).

    Returns:
        BGR mosaic image as a numpy array.
    """
    annotated: list[tuple[str, np.ndarray]] = []
    for cam_id in camera_ids:
        frame = frames.get(cam_id)
        if frame is None:
            continue
        canvas = frame.copy()
        dets = frame_dets.get(cam_id, [])

        for det in dets:
            color = _confidence_color(det.confidence)

            if det.obb_points is not None:
                pts = det.obb_points.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
            else:
                x, y, w, h = det.bbox
                cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

            # Confidence label.
            label = f"{det.confidence:.2f}"
            if det.obb_points is not None:
                lx = int(det.obb_points[:, 0].min())
                ly = int(det.obb_points[:, 1].min()) - 5
            else:
                lx, ly = det.bbox[0], det.bbox[1] - 5
            cv2.putText(
                canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        # Camera label in top-left.
        cv2.putText(
            canvas, cam_id, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )
        annotated.append((cam_id, canvas))

    if not annotated:
        raise RuntimeError("No frames could be annotated")

    # Build mosaic.
    n = len(annotated)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    cell_w = 640
    cell_h = int(cell_w * annotated[0][1].shape[0] / annotated[0][1].shape[1])

    mosaic = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for i, (_cam_id, img) in enumerate(annotated):
        r, c = divmod(i, cols)
        resized = cv2.resize(img, (cell_w, cell_h))
        mosaic[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = resized

    return mosaic


def _resolve_video_source(run_dir: Path) -> tuple[Path, Path]:
    """Resolve video_dir and calibration_path from a run's config.yaml.

    Args:
        run_dir: Path to the pipeline run directory.

    Returns:
        Tuple of (video_path, calibration_path).

    Raises:
        RuntimeError: If config.yaml is missing required fields.
    """
    config_yaml = read_config_yaml(run_dir)
    video_dir_str = config_yaml.get("video_dir", "")
    calibration_path_str = config_yaml.get("calibration_path", "")
    if not video_dir_str or not calibration_path_str:
        raise RuntimeError("config.yaml must have video_dir and calibration_path")

    project_dir = config_yaml.get("project_dir", "")
    video_path = Path(video_dir_str)
    calibration_path = Path(calibration_path_str)
    if not video_path.is_absolute() and project_dir:
        candidate = Path(project_dir) / video_path
        if candidate.exists():
            video_path = candidate
    if not calibration_path.is_absolute() and project_dir:
        candidate = Path(project_dir) / calibration_path
        if candidate.exists():
            calibration_path = candidate

    return video_path, calibration_path


def generate_detection_overlay(
    run_dir: Path,
    output_dir: Path | None = None,
    n_samples: int = 3,
) -> list[Path]:
    """Generate mosaic images showing OBB detections at evenly-spaced frames.

    Loads all chunk caches, picks *n_samples* evenly-spaced global frame
    indices, opens the video to grab the actual frame pixels, and draws
    oriented bounding boxes colored by confidence.

    Args:
        run_dir: Path to the pipeline run directory.
        output_dir: Directory for output. Defaults to ``{run_dir}/viz/``.
        n_samples: Number of evenly-spaced frames to render. Default 3.

    Returns:
        List of paths to the output PNG files.

    Raises:
        RuntimeError: If no chunk caches or video frames are available.
    """
    from aquapose.core.types.frame_source import VideoFrameSource

    out_dir = output_dir or run_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    caches = load_all_chunk_caches(run_dir)
    if not caches:
        raise RuntimeError(f"No chunk caches found in {run_dir}")

    # Build a flat list of (global_frame_idx, detections_dict) across all chunks.
    # Each chunk cache has chunk_start in the manifest; read it to compute global indices.
    import json

    manifest_path = run_dir / "diagnostics" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_entries = sorted(manifest.get("chunks", []), key=lambda c: c.get("index", 0))

    all_frames: list[tuple[int, dict]] = []  # (global_idx, per_camera_dets)
    for chunk_entry, ctx in zip(chunk_entries, caches, strict=True):
        chunk_start = chunk_entry.get("start_frame", 0)
        if ctx.detections is None:
            continue
        for local_idx, frame_dets in enumerate(ctx.detections):
            all_frames.append((chunk_start + local_idx, frame_dets))

    total = len(all_frames)
    if total == 0:
        raise RuntimeError("No detection frames found across all chunks")

    # Pick n_samples evenly-spaced indices.
    if n_samples >= total:
        sample_indices = list(range(total))
    else:
        sample_indices = [
            round(i * (total - 1) / (n_samples - 1)) for i in range(n_samples)
        ]

    camera_ids = caches[0].camera_ids or sorted(all_frames[0][1].keys())

    # Open video source once, render all samples.
    video_path, calibration_path = _resolve_video_source(run_dir)
    frame_source = VideoFrameSource(
        video_dir=video_path, calibration_path=calibration_path
    )

    output_paths: list[Path] = []
    with frame_source:
        for si in sample_indices:
            global_idx, frame_dets = all_frames[si]
            frames = frame_source.read_frame(global_idx)
            mosaic = _render_detection_mosaic(
                frame_dets, frames, camera_ids, global_idx
            )
            out_path = out_dir / f"detections_frame{global_idx:04d}.png"
            cv2.imwrite(str(out_path), mosaic)
            sys.stderr.write(f"Detection overlay written: {out_path}\n")
            output_paths.append(out_path)

    return output_paths
