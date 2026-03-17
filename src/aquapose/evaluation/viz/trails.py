"""Trail visualization: association mosaic from 3D midline reprojections."""

from __future__ import annotations

import contextlib
import logging
import math
import sys
from collections import deque
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from aquapose.evaluation.viz._frames import synthetic_frame_iter
from aquapose.evaluation.viz._loader import read_config_yaml

if TYPE_CHECKING:
    from aquapose.calibration.projection import RefractiveProjectionModel

logger = logging.getLogger(__name__)

# Paul Tol 22-color palette in BGR for OpenCV.
FISH_COLORS_BGR: list[tuple[int, int, int]] = [
    (112, 48, 0),
    (76, 211, 234),
    (153, 170, 68),
    (238, 204, 102),
    (51, 136, 34),
    (51, 153, 153),
    (119, 102, 238),
    (170, 153, 238),
    (119, 51, 170),
    (136, 34, 51),
    (51, 119, 238),
    (17, 51, 204),
    (85, 34, 136),
    (119, 51, 238),
    (153, 68, 170),
    (170, 119, 68),
    (221, 170, 119),
    (119, 204, 221),
    (51, 204, 187),
    (102, 136, 238),
    (187, 187, 187),
    (119, 119, 119),
]

_GRAY_BGR: tuple[int, int, int] = (119, 119, 119)
_TRAIL_LENGTH: int = 30
_TILE_SCALE: float = 0.35


def _fish_color(fish_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR color for a global fish ID.

    Args:
        fish_id: Global fish identifier.

    Returns:
        BGR color tuple from the palette.
    """
    return FISH_COLORS_BGR[fish_id % len(FISH_COLORS_BGR)]


def _dim_color(base_color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Blend a BGR color toward gray (128) at 50% to indicate dimmed segments.

    Args:
        base_color: BGR color tuple.

    Returns:
        Lighter BGR color tuple.
    """
    return tuple(int(c * 0.5 + 128 * 0.5) for c in base_color)  # type: ignore[return-value]


def _load_midline_positions(h5_path: Path) -> dict[int, dict[int, np.ndarray]]:
    """Load per-fish per-frame 3D centroids from midlines.h5.

    Reads ``frame_index``, ``fish_id``, and ``points`` datasets. For each
    valid slot, the centroid is computed as the mean of non-NaN keypoints.

    Args:
        h5_path: Path to the midlines HDF5 file.

    Returns:
        Mapping ``{frame_idx: {fish_id: centroid_xyz}}``.
    """
    from aquapose.io.midline_writer import read_midline3d_results

    data = read_midline3d_results(h5_path)
    frame_indices = data["frame_index"]  # (N,)
    fish_ids = data["fish_id"]  # (N, max_fish)
    points = data["points"]  # (N, max_fish, n_kpts, 3) or None

    N, max_fish = fish_ids.shape

    frame_data: dict[int, dict[int, np.ndarray]] = {}
    for i in range(N):
        fidx = int(frame_indices[i])
        fish_dict: dict[int, np.ndarray] = {}
        for s in range(max_fish):
            fid = int(fish_ids[i, s])
            if fid < 0:
                continue
            if points is not None:
                pts = points[i, s]  # (n_kpts, 3)
                valid_mask = ~np.isnan(pts).any(axis=1)
                if valid_mask.any():
                    centroid = pts[valid_mask].mean(axis=0)
                    fish_dict[fid] = centroid
        if fish_dict:
            frame_data[fidx] = fish_dict

    return frame_data


def _build_projected_frame_lookup(
    frame_data: dict[int, dict[int, np.ndarray]],
    models: dict[str, RefractiveProjectionModel],
) -> dict[str, dict[int, list[tuple[int, float, float]]]]:
    """Build per-camera per-frame index of projected centroids.

    Batch-projects all fish centroids per frame per camera for efficiency.

    Args:
        frame_data: ``{frame_idx: {fish_id: centroid_xyz}}``.
        models: Per-camera ``RefractiveProjectionModel``.

    Returns:
        ``{cam_id: {frame_idx: [(fish_id, u, v), ...]}}``.
    """
    import torch

    lookup: dict[str, dict[int, list[tuple[int, float, float]]]] = {
        cam_id: {} for cam_id in models
    }

    for frame_idx, fish_dict in frame_data.items():
        if not fish_dict:
            continue
        fids = list(fish_dict.keys())
        centroids = np.stack([fish_dict[fid] for fid in fids], axis=0)  # (K, 3)

        for cam_id, model in models.items():
            pts_tensor = torch.tensor(
                centroids, dtype=torch.float32, device=model.C.device
            )
            try:
                pixels, valid = model.project(pts_tensor)
                pixels_np = (
                    pixels.cpu().numpy()
                    if hasattr(pixels, "cpu")
                    else np.asarray(pixels)
                )
                valid_np = (
                    valid.cpu().numpy() if hasattr(valid, "cpu") else np.asarray(valid)
                )
            except Exception:
                continue

            entries: list[tuple[int, float, float]] = []
            for k, fid in enumerate(fids):
                if valid_np[k]:
                    entries.append(
                        (fid, float(pixels_np[k, 0]), float(pixels_np[k, 1]))
                    )
            if entries:
                lookup[cam_id][frame_idx] = entries

    return lookup


def _draw_trail_scaled(
    tile: np.ndarray,
    trail_pts: list[tuple[float, float]],
    color: tuple[int, int, int],
    fish_id: int,
    scale_x: float,
    scale_y: float,
    fade: bool = False,
) -> None:
    """Draw a trail on a downsampled tile with pre-applied scale factors.

    Args:
        tile: BGR tile image to draw on (modified in-place).
        trail_pts: List of ``(u, v)`` positions from oldest to newest.
        color: BGR color for the trail.
        fish_id: Fish ID for the label.
        scale_x: Horizontal scaling factor from original to tile.
        scale_y: Vertical scaling factor from original to tile.
        fade: If True, alpha-blend each segment individually (slow).
    """
    n_pts = len(trail_pts)
    if n_pts < 1:
        return

    if fade:
        for seg_i in range(n_pts - 1):
            alpha = 0.3 + 0.7 * (seg_i / max(n_pts - 1, 1))
            overlay = tile.copy()
            pt1 = (
                int(trail_pts[seg_i][0] * scale_x),
                int(trail_pts[seg_i][1] * scale_y),
            )
            pt2 = (
                int(trail_pts[seg_i + 1][0] * scale_x),
                int(trail_pts[seg_i + 1][1] * scale_y),
            )
            cv2.line(overlay, pt1, pt2, color, 1)
            cv2.addWeighted(overlay, alpha, tile, 1 - alpha, 0, tile)
    else:
        for seg_i in range(n_pts - 1):
            pt1 = (
                int(trail_pts[seg_i][0] * scale_x),
                int(trail_pts[seg_i][1] * scale_y),
            )
            pt2 = (
                int(trail_pts[seg_i + 1][0] * scale_x),
                int(trail_pts[seg_i + 1][1] * scale_y),
            )
            cv2.line(tile, pt1, pt2, color, 1)

    head_u = int(trail_pts[-1][0] * scale_x)
    head_v = int(trail_pts[-1][1] * scale_y)
    cv2.circle(tile, (head_u, head_v), 2, color, -1)
    label = str(fish_id)
    cv2.putText(
        tile,
        label,
        (head_u + 3, head_v - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        color,
        1,
    )


def _build_mosaic(
    tiles: dict[str, np.ndarray],
    camera_ids: list[str],
    tile_w: int,
    tile_h: int,
) -> np.ndarray:
    """Composite camera tiles into a grid mosaic.

    Args:
        tiles: camera_id -> tile image (all same size tile_w x tile_h).
        camera_ids: Ordered list for grid placement.
        tile_w: Tile width in pixels.
        tile_h: Tile height in pixels.

    Returns:
        Composite mosaic image.
    """
    n_cams = len(camera_ids)
    if n_cams == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    n_cols = math.ceil(math.sqrt(n_cams))
    n_rows = math.ceil(n_cams / n_cols)
    mosaic = np.zeros((n_rows * tile_h, n_cols * tile_w, 3), dtype=np.uint8)

    for idx, cam_id in enumerate(camera_ids):
        row = idx // n_cols
        col = idx % n_cols
        tile = tiles.get(cam_id)
        if tile is not None:
            y0, y1 = row * tile_h, (row + 1) * tile_h
            x0, x1 = col * tile_w, (col + 1) * tile_w
            if tile.shape[:2] != (tile_h, tile_w):
                tile = cv2.resize(tile, (tile_w, tile_h))
            mosaic[y0:y1, x0:x1] = tile

    return mosaic


def _write_association_mosaic(
    camera_ids: list[str],
    projected_lookup: dict[str, dict[int, list[tuple[int, float, float]]]],
    out_dir: Path,
    frame_source: object | None,
    frame_sizes: dict[str, tuple[int, int]] | None,
    total_frames: int,
    fps: float,
    tile_scale: float,
    trail_length: int,
    fade: bool = False,
) -> None:
    """Write the association mosaic MP4 from projected 3D centroids.

    Args:
        camera_ids: Ordered list of all camera IDs.
        projected_lookup: Per-camera per-frame projected positions.
        out_dir: Output directory for mosaic video.
        frame_source: Optional VideoFrameSource.
        frame_sizes: camera_id -> (width, height) for synthetic fallback.
        total_frames: Total frame count.
        fps: Output video frame rate.
        tile_scale: Downsampling factor applied to each camera tile.
        trail_length: Trail length in frames.
        fade: If True, use per-segment alpha blending.
    """
    use_synthetic = frame_source is None and bool(frame_sizes)

    if frame_source is None and not use_synthetic:
        return

    ctx_mgr: AbstractContextManager  # type: ignore[type-arg]
    if use_synthetic:
        assert frame_sizes is not None
        ctx_mgr = contextlib.nullcontext(
            synthetic_frame_iter(camera_ids, frame_sizes, total_frames)
        )
    else:
        assert frame_source is not None
        ctx_mgr = frame_source  # type: ignore[assignment]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer: cv2.VideoWriter | None = None
    tile_w: int = 0
    tile_h: int = 0

    # Per-camera per-fish trail buffer: cam_id -> fish_id -> deque of (u, v)
    trail_buffers: dict[str, dict[int, deque[tuple[float, float]]]] = {
        cam_id: {} for cam_id in camera_ids
    }

    # Collect all unique fish IDs to assign colors
    all_fish_ids: set[int] = set()
    for cam_lookup in projected_lookup.values():
        for entries in cam_lookup.values():
            for fid, _u, _v in entries:
                all_fish_ids.add(fid)

    fish_color_map: dict[int, tuple[int, int, int]] = {
        fid: _fish_color(fid) for fid in all_fish_ids
    }

    try:
        with ctx_mgr as frame_iter:
            for frame_idx, frames in frame_iter:
                if frame_idx >= total_frames:
                    break
                # Determine tile dimensions from first real frame.
                if tile_w == 0:
                    for cam_id in camera_ids:
                        if cam_id in frames:
                            raw_h, raw_w = frames[cam_id].shape[:2]
                            tile_w = max(1, int(raw_w * tile_scale))
                            tile_h = max(1, int(raw_h * tile_scale))
                            break
                if tile_w == 0:
                    break

                tiles: dict[str, np.ndarray] = {}
                for cam_id in camera_ids:
                    raw = frames.get(cam_id)
                    if raw is None:
                        tiles[cam_id] = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                        continue

                    tile = cv2.resize(raw, (tile_w, tile_h))
                    scale_x = tile_w / raw.shape[1]
                    scale_y = tile_h / raw.shape[0]

                    # Update trail buffers for this camera
                    cam_entries = projected_lookup.get(cam_id, {}).get(frame_idx, [])
                    cam_buf = trail_buffers[cam_id]

                    # Track which fish are present this frame
                    present_fids = set()
                    for fid, u, v in cam_entries:
                        present_fids.add(fid)
                        if fid not in cam_buf:
                            cam_buf[fid] = deque(maxlen=trail_length)
                        cam_buf[fid].append((u, v))

                    # Draw trails for all fish with buffered positions
                    for fid, buf in cam_buf.items():
                        if len(buf) == 0:
                            continue
                        color = fish_color_map.get(fid, _GRAY_BGR)
                        _draw_trail_scaled(
                            tile,
                            list(buf),
                            color,
                            fid,
                            scale_x,
                            scale_y,
                            fade=fade,
                        )

                    # Camera label.
                    cv2.putText(
                        tile,
                        cam_id,
                        (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 0),
                        2,
                    )
                    cv2.putText(
                        tile,
                        cam_id,
                        (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    tiles[cam_id] = tile

                mosaic = _build_mosaic(tiles, camera_ids, tile_w, tile_h)

                if writer is None:
                    mh, mw = mosaic.shape[:2]
                    out_path = out_dir / "association_mosaic.mp4"
                    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (mw, mh))

                writer.write(mosaic)
    finally:
        if writer is not None:
            writer.release()


def generate_trails(
    run_dir: Path,
    output_dir: Path | None = None,
    *,
    fps: float = 30.0,
    trail_length: int = _TRAIL_LENGTH,
    tile_scale: float = _TILE_SCALE,
    fade_trails: bool = False,
    unstitched: bool = False,
) -> Path:
    """Generate an association mosaic video from 3D midline reprojections.

    Reads 3D midline data from the run's HDF5 file, reprojects centroids to
    each camera, and renders a mosaic video with colored trails per fish.

    By default, prefers ``midlines_stitched.h5`` if it exists (post-stitching
    IDs). Pass ``unstitched=True`` to force reading ``midlines.h5``.

    Args:
        run_dir: Path to the pipeline run directory.
        output_dir: Directory for output. Defaults to ``{run_dir}/viz/``.
        fps: Output video frame rate.
        trail_length: Number of past frames to include in each trail.
        tile_scale: Downsampling factor for mosaic tiles.
        fade_trails: If True, alpha-blend each trail segment for a fade
            effect (significantly slower due to per-segment frame copies).
        unstitched: If True, always use ``midlines.h5`` even when
            ``midlines_stitched.h5`` exists.

    Returns:
        Path to the output directory (``{run_dir}/viz/`` or output_dir).

    Raises:
        RuntimeError: If no midlines HDF5 file is found.
    """
    # Select H5 file.
    if not unstitched:
        h5_path = run_dir / "midlines_stitched.h5"
        if not h5_path.exists():
            h5_path = run_dir / "midlines.h5"
    else:
        h5_path = run_dir / "midlines.h5"

    if not h5_path.exists():
        raise RuntimeError(f"No midlines HDF5 found in {run_dir}")

    out_dir = output_dir or run_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load 3D centroids from H5.
    frame_data = _load_midline_positions(h5_path)
    if not frame_data:
        raise RuntimeError(f"No valid midline data in {h5_path}")

    total_frames = max(frame_data.keys()) + 1

    # Load calibration and build projection models.
    config_yaml = read_config_yaml(run_dir)
    calibration_path_str = config_yaml.get("calibration_path", "")
    calib_path = Path(calibration_path_str)
    if not calib_path.is_absolute():
        for base in (run_dir, run_dir.parent):
            candidate = base / calib_path
            if candidate.exists():
                calib_path = candidate
                break

    if not calib_path.exists():
        raise RuntimeError(f"Calibration file not found: {calib_path}")

    from aquapose.calibration.loader import (
        compute_undistortion_maps,
        load_calibration_data,
    )
    from aquapose.calibration.projection import RefractiveProjectionModel

    calib_data = load_calibration_data(str(calib_path))
    camera_ids = sorted(calib_data.cameras.keys())

    models: dict[str, RefractiveProjectionModel] = {}
    frame_sizes: dict[str, tuple[int, int]] = {}
    for cam_id in camera_ids:
        cam = calib_data.cameras[cam_id]
        undist_maps = compute_undistortion_maps(cam)
        models[cam_id] = RefractiveProjectionModel(
            K=undist_maps.K_new,
            R=cam.R,
            t=cam.t,
            water_z=calib_data.water_z,
            normal=calib_data.interface_normal,
            n_air=calib_data.n_air,
            n_water=calib_data.n_water,
        )
        frame_sizes[cam_id] = cam.image_size

    # Build projected lookup.
    projected_lookup = _build_projected_frame_lookup(frame_data, models)

    # Resolve frame source.
    frame_source = None
    video_dir_str = config_yaml.get("video_dir", "")
    if video_dir_str:
        video_path = Path(video_dir_str)
        if not video_path.is_absolute():
            for base in (run_dir, run_dir.parent):
                candidate = base / video_path
                if candidate.exists():
                    video_path = candidate
                    break
        if video_path.exists():
            try:
                from aquapose.core.types.frame_source import VideoFrameSource

                frame_source = VideoFrameSource(
                    video_dir=video_path,
                    calibration_path=calib_path,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to open VideoFrameSource for trails; using black frames: %s",
                    exc,
                )
                frame_source = None

    # Narrow camera list to those the frame source actually provides.
    if frame_source is not None:
        source_cams = set(frame_source.camera_ids)
        camera_ids = [c for c in camera_ids if c in source_cams]
        models = {c: m for c, m in models.items() if c in source_cams}
        frame_sizes = {c: s for c, s in frame_sizes.items() if c in source_cams}
        # Rebuild projected lookup with narrowed camera set.
        projected_lookup = {
            c: v for c, v in projected_lookup.items() if c in source_cams
        }

    use_synthetic = frame_source is None and bool(frame_sizes)
    if not use_synthetic and frame_source is None:
        logger.warning(
            "No frame source or calibration data available; skipping trail video generation"
        )
        return out_dir

    sys.stderr.write("Generating trail mosaic video...\n")
    sys.stderr.flush()

    _write_association_mosaic(
        camera_ids=camera_ids,
        projected_lookup=projected_lookup,
        out_dir=out_dir,
        frame_source=frame_source,
        frame_sizes=frame_sizes,
        total_frames=total_frames,
        fps=fps,
        tile_scale=tile_scale,
        trail_length=trail_length,
        fade=fade_trails,
    )

    if frame_source is not None:
        with contextlib.suppress(Exception):
            frame_source.__exit__(None, None, None)

    logger.info("Trail mosaic written to %s", out_dir)
    return out_dir


__all__ = ["FISH_COLORS_BGR", "generate_trails"]
