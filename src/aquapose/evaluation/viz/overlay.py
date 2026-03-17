"""Overlay visualization: reprojected 3D midlines on camera frames."""

from __future__ import annotations

import contextlib
import logging
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import scipy.interpolate

from aquapose.evaluation.viz._frames import synthetic_frame_iter
from aquapose.evaluation.viz._loader import (
    load_midlines_from_h5,
    read_config_yaml,
    resolve_h5_path,
)

if TYPE_CHECKING:
    from aquapose.calibration.projection import RefractiveProjectionModel

logger = logging.getLogger(__name__)

# Deterministic color palette (BGR for OpenCV), cycled by global fish_id % len(palette).
_PALETTE_BGR: list[tuple[int, int, int]] = [
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
]


def _fish_color(fish_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR color for a global fish ID.

    Args:
        fish_id: Global fish identifier.

    Returns:
        BGR color tuple from the palette.
    """
    return _PALETTE_BGR[fish_id % len(_PALETTE_BGR)]


_N_SPLINE_EVAL: int = 50
_T_VALS: np.ndarray = np.linspace(0.0, 1.0, _N_SPLINE_EVAL)


def _eval_spline_pts(spline: object) -> np.ndarray | None:
    """Evaluate a B-spline or return raw keypoints as 3D points.

    Args:
        spline: Midline3D (or similar) with optional control_points, knots,
            degree, and points attributes.

    Returns:
        (N, 3) float64 array, or None on failure.
    """
    cp = getattr(spline, "control_points", None)
    knots = getattr(spline, "knots", None)
    degree = getattr(spline, "degree", None)
    if cp is not None and knots is not None and degree is not None:
        cp_arr = np.asarray(cp, dtype=np.float64)
        if not np.all(np.isnan(cp_arr)):
            try:
                bspl = scipy.interpolate.BSpline(
                    np.asarray(knots, dtype=np.float64),
                    cp_arr,
                    degree,
                )
                return bspl(_T_VALS)  # (N, 3)
            except Exception:
                pass

    # Fall back to raw keypoints.
    pts = getattr(spline, "points", None)
    if pts is not None:
        pts_arr = np.asarray(pts, dtype=np.float64)
        if (
            pts_arr.ndim == 2
            and pts_arr.shape[1] == 3
            and not np.all(np.isnan(pts_arr))
        ):
            return pts_arr
    return None


def _batch_reproject(
    splines_3d: list[np.ndarray],
    model: RefractiveProjectionModel,
) -> list[np.ndarray | None]:
    """Project multiple spline point arrays through a camera model in one call.

    Concatenates all 3D points into a single tensor, projects once, then
    splits results back per-spline. Much faster than one project() call per
    spline due to reduced GPU round-trips.

    Args:
        splines_3d: List of (N_i, 3) float64 arrays (one per spline).
        model: Per-camera RefractiveProjectionModel.

    Returns:
        List of (M_i, 2) float32 arrays (valid pixels per spline), or None
        entries for splines with fewer than 2 valid projections.
    """
    import torch

    if not splines_3d:
        return []

    lengths = [len(pts) for pts in splines_3d]
    combined = np.concatenate(splines_3d, axis=0)  # (sum(N_i), 3)
    pts_tensor = torch.tensor(combined, dtype=torch.float32, device=model.C.device)

    try:
        pixels, valid = model.project(pts_tensor)
        pixels_np = (
            pixels.cpu().numpy() if hasattr(pixels, "cpu") else np.asarray(pixels)
        )
        valid_np = valid.cpu().numpy() if hasattr(valid, "cpu") else np.asarray(valid)
    except Exception:
        return [None] * len(splines_3d)

    results: list[np.ndarray | None] = []
    offset = 0
    for length in lengths:
        seg_pixels = pixels_np[offset : offset + length]
        seg_valid = valid_np[offset : offset + length]
        seg_pixels = seg_pixels[seg_valid]
        if len(seg_pixels) < 2:
            results.append(None)
        else:
            results.append(seg_pixels.astype(np.float32))
        offset += length
    return results


def _draw_midline(
    frame: np.ndarray,
    points_2d: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a polyline on a frame in-place.

    Args:
        frame: BGR image to draw on (modified in-place).
        points_2d: (N, 2) array of pixel coordinates.
        color: BGR color tuple.
        thickness: Line thickness in pixels.
    """
    pts = np.asarray(points_2d, dtype=np.int32)
    if len(pts) < 2:
        return
    cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=thickness)


def _draw_midline_points(
    frame: np.ndarray,
    points_2d: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 3,
) -> None:
    """Draw individual midline points (circles) with head highlighted in red.

    Args:
        frame: BGR image to draw on (modified in-place).
        points_2d: (N, 2) array of pixel coordinates, head-to-tail order.
        color: BGR color tuple for body points.
        radius: Circle radius in pixels.
    """
    pts = np.asarray(points_2d, dtype=np.int32)
    if len(pts) < 2:
        return
    cv2.circle(frame, (pts[0][0], pts[0][1]), radius, (0, 0, 255), -1)
    for pt in pts[1:]:
        cv2.circle(frame, (pt[0], pt[1]), radius, color, -1)


def _build_mosaic(
    frames: dict[str, np.ndarray],
    camera_ids: list[str],
) -> np.ndarray:
    """Tile camera frames into a grid mosaic.

    Args:
        frames: Dict mapping camera_id to BGR image arrays.
        camera_ids: Ordered list of camera IDs for grid layout.

    Returns:
        Single mosaic image with all cameras tiled.
    """
    n_cams = len(camera_ids)
    if n_cams == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    n_cols = math.ceil(math.sqrt(n_cams))
    n_rows = math.ceil(n_cams / n_cols)

    sample_frame = None
    for cam_id in camera_ids:
        if cam_id in frames:
            sample_frame = frames[cam_id]
            break
    if sample_frame is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    cell_h, cell_w = sample_frame.shape[:2]
    mosaic = np.zeros((n_rows * cell_h, n_cols * cell_w, 3), dtype=np.uint8)

    for idx, cam_id in enumerate(camera_ids):
        row = idx // n_cols
        col = idx % n_cols
        if cam_id in frames:
            frame = frames[cam_id]
            if frame.shape[:2] != (cell_h, cell_w):
                frame = cv2.resize(frame, (cell_w, cell_h))
            y0, y1 = row * cell_h, (row + 1) * cell_h
            x0, x1 = col * cell_w, (col + 1) * cell_w
            mosaic[y0:y1, x0:x1] = frame

    return mosaic


def generate_overlay(
    run_dir: Path,
    output_dir: Path | None = None,
    *,
    fps: float = 30.0,
    scale: float = 0.5,
    show_bbox: bool = False,
    show_fish_id: bool = False,
    unstitched: bool = False,
) -> Path:
    """Generate a continuous mosaic overlay video from 3D midline reprojections.

    Reads midline data from the run's HDF5 file, builds projection models,
    and renders reprojected 3D midlines on top of camera frames. If video
    frames are not available, falls back to synthetic black frames.

    By default, prefers ``midlines_stitched.h5`` if it exists (post-stitching
    IDs). Pass ``unstitched=True`` to force reading ``midlines.h5``.

    Args:
        run_dir: Path to the pipeline run directory.
        output_dir: Directory for output. Defaults to ``{run_dir}/viz/``.
        fps: Output video frame rate.
        scale: Downscale factor applied to each frame before mosaic assembly.
        show_bbox: If True, draw detection bounding boxes (requires diagnostic
            chunk caches; ignored when not available).
        show_fish_id: If True, annotate fish IDs on bounding boxes.
        unstitched: If True, always use ``midlines.h5`` even when
            ``midlines_stitched.h5`` exists.

    Returns:
        Path to the written ``overlay_mosaic.mp4`` file.

    Raises:
        RuntimeError: If no midlines HDF5 file is found.
    """
    h5_path = resolve_h5_path(run_dir, unstitched=unstitched)
    if h5_path is None:
        raise RuntimeError(f"No midlines HDF5 found in {run_dir}")

    out_dir = output_dir or run_dir / "viz"
    _ = show_fish_id  # reserved for future use

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "overlay_mosaic.mp4"

    # Load midlines from H5.
    sys.stderr.write(f"Loading midlines from {h5_path.name}...\n")
    sys.stderr.flush()
    all_midlines_3d = load_midlines_from_h5(h5_path)
    total_frames = len(all_midlines_3d)
    if total_frames == 0:
        raise RuntimeError(f"No midline data in {h5_path}")

    # Load calibration.
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

    models: dict[str, RefractiveProjectionModel] = {}
    frame_w: int = 0
    frame_h: int = 0

    from aquapose.calibration.loader import (
        compute_undistortion_maps,
        load_calibration_data,
    )
    from aquapose.calibration.projection import RefractiveProjectionModel

    calib_data = load_calibration_data(str(calib_path))
    camera_ids = sorted(calib_data.cameras.keys())

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

    # Get frame dimensions from calibration.
    first_cam = next((cid for cid in camera_ids if cid in calib_data.cameras), None)
    if first_cam is not None:
        frame_w, frame_h = calib_data.cameras[first_cam].image_size

    # Try to open a VideoFrameSource from config.
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
                    "Failed to open VideoFrameSource; falling back to black frames: %s",
                    exc,
                )
                frame_source = None

    # Narrow camera list to those the frame source actually provides.
    if frame_source is not None:
        source_cams = set(frame_source.camera_ids)
        camera_ids = [c for c in camera_ids if c in source_cams]
        models = {c: m for c, m in models.items() if c in source_cams}

    # Build synthetic fallback frame sizes.
    frame_sizes: dict[str, tuple[int, int]] = {}
    if frame_w > 0 and frame_h > 0:
        frame_sizes = {cam_id: (frame_w, frame_h) for cam_id in camera_ids}

    # Compute scaled output dimensions.
    out_w = max(1, int(frame_w * scale)) if frame_w > 0 else 0
    out_h = max(1, int(frame_h * scale)) if frame_h > 0 else 0

    if frame_source is not None:
        ctx_mgr = frame_source
    elif frame_sizes:
        ctx_mgr = contextlib.nullcontext(
            synthetic_frame_iter(camera_ids, frame_sizes, total_frames)
        )
    else:
        raise RuntimeError("No frame source or calibration data available for overlay")

    # Load detection data from chunk caches if show_bbox requested.
    all_detections: list[dict | None] = []
    if show_bbox:
        try:
            from aquapose.evaluation.viz._loader import load_all_chunk_caches

            contexts = load_all_chunk_caches(run_dir)
            for ctx in contexts:
                det = getattr(ctx, "detections", None) or [None] * getattr(
                    ctx, "frame_count", 0
                )
                all_detections.extend(det)
        except Exception as exc:
            logger.warning("Could not load chunk caches for bbox overlay: %s", exc)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer: cv2.VideoWriter | None = None

    sys.stderr.write("Generating overlay mosaic video...\n")
    sys.stderr.flush()

    try:
        with ctx_mgr as frame_iter:
            for frame_idx, frames in frame_iter:
                if frame_idx >= total_frames:
                    break

                # Draw 2D keypoints from detections (optional).
                if show_bbox and frame_idx < len(all_detections):
                    frame_dets = all_detections[frame_idx]
                    if isinstance(frame_dets, dict):
                        for cam_id, dets in frame_dets.items():
                            if cam_id not in frames:
                                continue
                            for det in dets:
                                kpts = getattr(det, "keypoints", None)
                                if kpts is not None:
                                    pts_arr = np.asarray(kpts, dtype=np.float32)
                                    _draw_midline_points(
                                        frames[cam_id], pts_arr, (255, 0, 0)
                                    )

                # Draw reprojected 3D midlines (batched per camera).
                if frame_idx < len(all_midlines_3d):
                    frame_midlines = all_midlines_3d[frame_idx]
                    if isinstance(frame_midlines, dict):
                        # Evaluate all splines once (CPU).
                        fish_ids: list[int] = []
                        spline_pts: list[np.ndarray] = []
                        for fish_id, spline in frame_midlines.items():
                            pts = _eval_spline_pts(spline)
                            if pts is not None:
                                fish_ids.append(fish_id)
                                spline_pts.append(pts)

                        if spline_pts:
                            # One batched GPU call per camera.
                            for cam_id in camera_ids:
                                if cam_id not in frames or cam_id not in models:
                                    continue
                                batch_results = _batch_reproject(
                                    spline_pts, models[cam_id]
                                )
                                for fid, pts_2d in zip(
                                    fish_ids, batch_results, strict=True
                                ):
                                    if pts_2d is not None:
                                        _draw_midline(
                                            frames[cam_id], pts_2d, _fish_color(fid)
                                        )

                # Scale frames.
                if scale != 1.0 and out_w > 0 and out_h > 0:
                    frames = {
                        cid: cv2.resize(f, (out_w, out_h)) for cid, f in frames.items()
                    }

                mosaic = _build_mosaic(frames, camera_ids)

                # Lazy-init writer.
                if writer is None:
                    mh, mw = mosaic.shape[:2]
                    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (mw, mh))

                writer.write(mosaic)
    finally:
        if writer is not None:
            writer.release()
        if frame_source is not None:
            with contextlib.suppress(Exception):
                frame_source.__exit__(None, None, None)

    logger.info("Overlay mosaic written to %s", output_path)
    return output_path


__all__ = ["generate_overlay"]
