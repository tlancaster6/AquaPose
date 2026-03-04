"""Tracklet trail visualization: per-camera trail videos and association mosaic across all chunks."""

from __future__ import annotations

import contextlib
import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from aquapose.evaluation.viz._loader import load_all_chunk_caches, read_config_yaml
from aquapose.visualization.frames import synthetic_frame_iter

logger = logging.getLogger(__name__)

# Paul Tol 22-color palette in BGR for OpenCV (same as TrackletTrailObserver).
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


def _coasted_color(base_color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Blend a BGR color toward gray (128) at 50% to indicate coasted frames.

    Args:
        base_color: BGR color tuple for a detected frame.

    Returns:
        Lighter BGR color tuple for coasted frames.
    """
    return tuple(int(c * 0.5 + 128 * 0.5) for c in base_color)  # type: ignore[return-value]


def _build_color_maps(
    tracklet_groups: list,
) -> tuple[dict[int, tuple[int, int, int]], dict[tuple[str, int], int]]:
    """Build fish_id -> BGR color and (camera_id, track_id) -> fish_id maps.

    Singletons (1 tracklet) share _GRAY_BGR so multi-camera associations
    visually stand out. Non-singleton groups are assigned deterministic
    colors by fish_id % palette_length.

    Args:
        tracklet_groups: List of TrackletGroup objects.

    Returns:
        Tuple of (fish_color_map, track_to_fish_map).
    """
    fish_color_map: dict[int, tuple[int, int, int]] = {}
    track_to_fish: dict[tuple[str, int], int] = {}

    for group in tracklet_groups:
        fish_id: int = group.fish_id
        is_singleton = len(group.tracklets) <= 1
        if is_singleton:
            fish_color_map[fish_id] = _GRAY_BGR
        else:
            fish_color_map[fish_id] = _fish_color(fish_id)
        for tracklet in group.tracklets:
            key = (tracklet.camera_id, tracklet.track_id)
            track_to_fish[key] = fish_id

    return fish_color_map, track_to_fish


def _build_frame_lookup(
    tracks_2d: dict,
    track_to_fish: dict[tuple[str, int], int],
) -> dict[str, dict[int, list[tuple[object, int, int]]]]:
    """Build per-camera per-frame index for trail rendering.

    Args:
        tracks_2d: dict[str, list[Tracklet2D]] from PipelineContext.
        track_to_fish: Mapping (camera_id, track_id) -> fish_id.

    Returns:
        Per-camera per-frame lookup: camera_id -> frame_idx -> [(tracklet, idx, fish_id)].
    """
    lookup: dict[str, dict[int, list[tuple[object, int, int]]]] = {}
    for cam_id, tracklet_list in tracks_2d.items():
        cam_lookup: dict[int, list[tuple[object, int, int]]] = {}
        for tracklet in tracklet_list:
            fish_id = track_to_fish.get((cam_id, tracklet.track_id), -1)
            for idx, frame_idx in enumerate(tracklet.frames):
                if frame_idx not in cam_lookup:
                    cam_lookup[frame_idx] = []
                cam_lookup[frame_idx].append((tracklet, idx, fish_id))
        lookup[cam_id] = cam_lookup
    return lookup


def _draw_trail(
    frame: np.ndarray,
    tracklet: object,
    current_idx: int,
    fish_id: int,
    fish_color_map: dict[int, tuple[int, int, int]],
    trail_length: int = _TRAIL_LENGTH,
) -> None:
    """Draw a fading polyline trail and fish ID label on a frame.

    Args:
        frame: BGR image to draw on (modified in-place).
        tracklet: Tracklet2D with frames, centroids, frame_status.
        current_idx: Current index within tracklet.frames.
        fish_id: Global fish identity (or -1 for ungrouped).
        fish_color_map: Mapping from fish_id to BGR color.
        trail_length: Number of past frames to include in the trail.
    """
    base_color = fish_color_map.get(fish_id, _GRAY_BGR)
    start_idx = max(0, current_idx - trail_length)
    trail_pts = list(tracklet.centroids[start_idx : current_idx + 1])  # type: ignore[index]
    trail_statuses = list(tracklet.frame_status[start_idx : current_idx + 1])  # type: ignore[index]
    n_pts = len(trail_pts)

    if n_pts < 1:
        return

    for seg_i in range(n_pts - 1):
        alpha = 0.3 + 0.7 * (seg_i / max(n_pts - 1, 1))
        status = trail_statuses[seg_i]
        seg_color = base_color if status == "detected" else _coasted_color(base_color)
        overlay = frame.copy()
        pt1 = (int(trail_pts[seg_i][0]), int(trail_pts[seg_i][1]))
        pt2 = (int(trail_pts[seg_i + 1][0]), int(trail_pts[seg_i + 1][1]))
        cv2.line(overlay, pt1, pt2, seg_color, 2)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    head_u = int(trail_pts[-1][0])
    head_v = int(trail_pts[-1][1])
    head_status = trail_statuses[-1]
    head_color = base_color if head_status == "detected" else _coasted_color(base_color)
    cv2.circle(frame, (head_u, head_v), 4, head_color, -1)

    label = str(fish_id) if fish_id >= 0 else "?"
    cv2.putText(
        frame,
        label,
        (head_u + 6, head_v - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        head_color,
        2,
    )


def _draw_trail_scaled(
    tile: np.ndarray,
    tracklet: object,
    current_idx: int,
    fish_id: int,
    fish_color_map: dict[int, tuple[int, int, int]],
    scale_x: float,
    scale_y: float,
    trail_length: int = _TRAIL_LENGTH,
) -> None:
    """Draw fading trail on a downsampled tile with pre-applied scale factors.

    Args:
        tile: BGR tile image to draw on (modified in-place).
        tracklet: Tracklet2D with frames, centroids, frame_status.
        current_idx: Current index within tracklet.frames.
        fish_id: Global fish identity (or -1 for ungrouped).
        fish_color_map: fish_id -> BGR color.
        scale_x: Horizontal scaling factor from original to tile.
        scale_y: Vertical scaling factor from original to tile.
        trail_length: Number of past frames to include in the trail.
    """
    base_color = fish_color_map.get(fish_id, _GRAY_BGR)
    start_idx = max(0, current_idx - trail_length)
    trail_pts = list(tracklet.centroids[start_idx : current_idx + 1])  # type: ignore[index]
    trail_statuses = list(tracklet.frame_status[start_idx : current_idx + 1])  # type: ignore[index]
    n_pts = len(trail_pts)

    if n_pts < 1:
        return

    for seg_i in range(n_pts - 1):
        alpha = 0.3 + 0.7 * (seg_i / max(n_pts - 1, 1))
        status = trail_statuses[seg_i]
        seg_color = base_color if status == "detected" else _coasted_color(base_color)
        overlay = tile.copy()
        pt1 = (int(trail_pts[seg_i][0] * scale_x), int(trail_pts[seg_i][1] * scale_y))
        pt2 = (
            int(trail_pts[seg_i + 1][0] * scale_x),
            int(trail_pts[seg_i + 1][1] * scale_y),
        )
        cv2.line(overlay, pt1, pt2, seg_color, 1)
        cv2.addWeighted(overlay, alpha, tile, 1 - alpha, 0, tile)

    head_u = int(trail_pts[-1][0] * scale_x)
    head_v = int(trail_pts[-1][1] * scale_y)
    head_status = trail_statuses[-1]
    head_color = base_color if head_status == "detected" else _coasted_color(base_color)
    cv2.circle(tile, (head_u, head_v), 2, head_color, -1)
    label = str(fish_id) if fish_id >= 0 else "?"
    cv2.putText(
        tile,
        label,
        (head_u + 3, head_v - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        head_color,
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


def generate_trails(
    run_dir: Path,
    output_dir: Path | None = None,
    *,
    fps: float = 30.0,
    trail_length: int = _TRAIL_LENGTH,
    tile_scale: float = _TILE_SCALE,
) -> Path:
    """Generate per-camera trail videos and an association mosaic across all chunks.

    Loads all chunk caches, merges track data, and renders continuous trail
    videos spanning the full recording. All outputs go to ``{run_dir}/viz/``
    (or output_dir if provided).

    Fish colors are deterministic: palette[fish_id % palette_length]. Singletons
    (only seen by one camera) use gray to highlight multi-camera associations.

    Args:
        run_dir: Path to the pipeline run directory.
        output_dir: Directory for output. Defaults to ``{run_dir}/viz/``.
        fps: Output video frame rate.
        trail_length: Number of past frames to include in each trail.
        tile_scale: Downsampling factor for mosaic tiles.

    Returns:
        Path to the output directory (``{run_dir}/viz/`` or output_dir).

    Raises:
        RuntimeError: If no chunk caches are found in run_dir.
    """
    contexts = load_all_chunk_caches(run_dir)
    if not contexts:
        raise RuntimeError(f"No chunk caches found in {run_dir}")

    out_dir = output_dir or run_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather camera_ids and merge track data across all chunks.
    camera_ids: list[str] | None = None
    all_tracks_2d: dict[str, list] = {}
    all_tracklet_groups: list = []
    all_detections: list = []
    total_frames: int = 0

    # Chunk frame offsets for remapping frame indices to global.
    chunk_offsets: list[int] = []
    offset = 0
    for ctx in contexts:
        chunk_offsets.append(offset)
        fc = getattr(ctx, "frame_count", None) or 0
        offset += fc

    total_frames = offset

    for chunk_idx, ctx in enumerate(contexts):
        frame_offset = chunk_offsets[chunk_idx]

        if camera_ids is None:
            cam = getattr(ctx, "camera_ids", None)
            if cam:
                camera_ids = cam

        # Merge tracks_2d with global frame index offsets.
        tracks_2d = getattr(ctx, "tracks_2d", None) or {}
        for cam_id, tracklet_list in tracks_2d.items():
            if cam_id not in all_tracks_2d:
                all_tracks_2d[cam_id] = []
            for tracklet in tracklet_list:
                # Rebase frame indices to global space.
                rebased = _rebase_tracklet(tracklet, frame_offset)
                all_tracks_2d[cam_id].append(rebased)

        # Merge tracklet_groups (fish_ids are already global).
        groups = getattr(ctx, "tracklet_groups", None) or []
        for group in groups:
            rebased_group = _rebase_group(group, frame_offset)
            all_tracklet_groups.append(rebased_group)

        # Merge detections.
        dets = getattr(ctx, "detections", None) or []
        all_detections.extend(dets)

    if not camera_ids:
        camera_ids = list(all_tracks_2d.keys())
    if not camera_ids:
        raise RuntimeError("No camera_ids found in any chunk cache")

    # Build color maps and frame lookup.
    fish_color_map, track_to_fish = _build_color_maps(all_tracklet_groups)
    frame_lookup = _build_frame_lookup(all_tracks_2d, track_to_fish)

    # Resolve frame source.
    config_yaml = read_config_yaml(run_dir)
    frame_source = None
    frame_sizes: dict[str, tuple[int, int]] | None = None

    calibration_path_str = config_yaml.get("calibration_path", "")
    calib_path = Path(calibration_path_str)
    if not calib_path.is_absolute():
        for base in (run_dir, run_dir.parent):
            candidate = base / calib_path
            if candidate.exists():
                calib_path = candidate
                break

    if calib_path.exists():
        try:
            from aquapose.calibration.loader import load_calibration_data

            calib_data = load_calibration_data(str(calib_path))
            frame_sizes = {
                cam_id: calib_data.cameras[cam_id].image_size
                for cam_id in camera_ids
                if cam_id in calib_data.cameras
            }
        except Exception as exc:
            logger.warning("Failed to load calibration for trail sizes: %s", exc)

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
                    camera_ids=camera_ids,
                    calibration_path=calib_path if calib_path.exists() else None,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to open VideoFrameSource for trails; using black frames: %s",
                    exc,
                )
                frame_source = None

    use_synthetic = frame_source is None and bool(frame_sizes)
    if not use_synthetic and frame_source is None:
        logger.warning(
            "No frame source or calibration data available; skipping trail video generation"
        )
        return out_dir

    sys.stderr.write("Generating tracklet trail videos...\n")
    sys.stderr.flush()

    # Generate per-camera trail videos.
    _write_per_camera_trails(
        camera_ids=camera_ids,
        tracks_2d=all_tracks_2d,
        frame_lookup=frame_lookup,
        fish_color_map=fish_color_map,
        out_dir=out_dir,
        frame_source=frame_source,
        frame_sizes=frame_sizes,
        total_frames=total_frames,
        fps=fps,
        trail_length=trail_length,
        detections=all_detections if all_detections else None,
    )

    # Generate association mosaic.
    _write_association_mosaic(
        camera_ids=camera_ids,
        frame_lookup=frame_lookup,
        fish_color_map=fish_color_map,
        out_dir=out_dir,
        frame_source=frame_source,
        frame_sizes=frame_sizes,
        total_frames=total_frames,
        fps=fps,
        tile_scale=tile_scale,
        trail_length=trail_length,
        detections=all_detections if all_detections else None,
    )

    if frame_source is not None:
        import contextlib

        with contextlib.suppress(Exception):
            frame_source.__exit__(None, None, None)

    logger.info("Trail videos written to %s", out_dir)
    return out_dir


def _rebase_tracklet(tracklet: object, frame_offset: int) -> object:
    """Return a lightweight proxy that offsets frame indices by frame_offset.

    Creates a simple namespace object mirroring the tracklet's interface but
    with frames offset to global frame space. Uses the same centroids and
    frame_status arrays (no copies).

    Args:
        tracklet: Tracklet2D-like object with frames, centroids, frame_status,
            camera_id, track_id attributes.
        frame_offset: Number of frames to add to each frame index.

    Returns:
        Proxy object with rebased frames.
    """
    if frame_offset == 0:
        return tracklet

    class _RebasedTracklet:
        def __init__(self, t: object, offset: int) -> None:
            self._t = t
            self.frames = [f + offset for f in t.frames]  # type: ignore[union-attr]
            self.centroids = t.centroids  # type: ignore[union-attr]
            self.frame_status = t.frame_status  # type: ignore[union-attr]
            self.camera_id = t.camera_id  # type: ignore[union-attr]
            self.track_id = t.track_id  # type: ignore[union-attr]

    return _RebasedTracklet(tracklet, frame_offset)


def _rebase_group(group: object, frame_offset: int) -> object:
    """Return a proxy TrackletGroup with rebased tracklet frame indices.

    Args:
        group: TrackletGroup-like object with fish_id and tracklets attributes.
        frame_offset: Number of frames to add to each tracklet's frame indices.

    Returns:
        Proxy group with rebased tracklets.
    """
    if frame_offset == 0:
        return group

    class _RebasedGroup:
        def __init__(self, g: object, offset: int) -> None:
            self.fish_id = g.fish_id  # type: ignore[union-attr]
            self.tracklets = [_rebase_tracklet(t, offset) for t in g.tracklets]  # type: ignore[union-attr]

    return _RebasedGroup(group, frame_offset)


def _write_per_camera_trails(
    camera_ids: list[str],
    tracks_2d: dict,
    frame_lookup: dict[str, dict[int, list]],
    fish_color_map: dict[int, tuple[int, int, int]],
    out_dir: Path,
    frame_source: object | None,
    frame_sizes: dict[str, tuple[int, int]] | None,
    total_frames: int,
    fps: float,
    trail_length: int,
    detections: list | None = None,
) -> None:
    """Write per-camera trail MP4 files.

    Args:
        camera_ids: Ordered list of camera IDs.
        tracks_2d: Per-camera tracklet lists.
        frame_lookup: Pre-built frame lookup structure.
        fish_color_map: fish_id -> BGR color.
        out_dir: Output directory for trail videos.
        frame_source: Optional VideoFrameSource.
        frame_sizes: camera_id -> (width, height) for synthetic fallback.
        total_frames: Total frame count for synthetic fallback.
        fps: Output video frame rate.
        trail_length: Trail length in frames.
        detections: Per-frame per-camera detection lists (optional).
    """
    use_synthetic = frame_source is None and bool(frame_sizes)

    for cam_id in camera_ids:
        cam_lookup = frame_lookup.get(cam_id, {})
        if not cam_lookup:
            continue

        if use_synthetic:
            assert frame_sizes is not None
            ctx_mgr = contextlib.nullcontext(
                synthetic_frame_iter([cam_id], frame_sizes, total_frames)
            )
        elif frame_source is not None:
            ctx_mgr = frame_source  # type: ignore[assignment]
        else:
            continue

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer: cv2.VideoWriter | None = None

        try:
            with ctx_mgr as frame_iter:
                for frame_idx, frames in frame_iter:
                    frame = frames.get(cam_id)
                    if frame is None:
                        continue
                    if writer is None:
                        h, w = frame.shape[:2]
                        out_path = out_dir / f"tracklet_trails_{cam_id}.mp4"
                        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                    active = cam_lookup.get(frame_idx, [])
                    for tracklet, idx_in_tracklet, fish_id in active:
                        _draw_trail(
                            frame,
                            tracklet,
                            idx_in_tracklet,
                            fish_id,
                            fish_color_map,
                            trail_length,
                        )
                    writer.write(frame)
        finally:
            if writer is not None:
                writer.release()


def _write_association_mosaic(
    camera_ids: list[str],
    frame_lookup: dict[str, dict[int, list]],
    fish_color_map: dict[int, tuple[int, int, int]],
    out_dir: Path,
    frame_source: object | None,
    frame_sizes: dict[str, tuple[int, int]] | None,
    total_frames: int,
    fps: float,
    tile_scale: float,
    trail_length: int,
    detections: list | None = None,
) -> None:
    """Write the association mosaic MP4.

    Args:
        camera_ids: Ordered list of all camera IDs.
        frame_lookup: Pre-built frame lookup structure.
        fish_color_map: fish_id -> BGR color.
        out_dir: Output directory for mosaic video.
        frame_source: Optional VideoFrameSource.
        frame_sizes: camera_id -> (width, height) for synthetic fallback.
        total_frames: Total frame count for synthetic fallback.
        fps: Output video frame rate.
        tile_scale: Downsampling factor applied to each camera tile.
        trail_length: Trail length in frames.
        detections: Per-frame per-camera detection lists (optional).
    """
    use_synthetic = frame_source is None and bool(frame_sizes)

    if frame_source is None and not use_synthetic:
        return

    if use_synthetic:
        assert frame_sizes is not None
        ctx_mgr = contextlib.nullcontext(
            synthetic_frame_iter(camera_ids, frame_sizes, total_frames)
        )
    else:
        assert frame_source is not None
        ctx_mgr = frame_source  # type: ignore[assignment]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: cv2.VideoWriter | None = None
    tile_w: int = 0
    tile_h: int = 0

    try:
        with ctx_mgr as frame_iter:
            for frame_idx, frames in frame_iter:
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

                    cam_lookup = frame_lookup.get(cam_id, {})
                    active = cam_lookup.get(frame_idx, [])
                    for tracklet, idx_in_tracklet, fish_id in active:
                        _draw_trail_scaled(
                            tile,
                            tracklet,
                            idx_in_tracklet,
                            fish_id,
                            fish_color_map,
                            scale_x,
                            scale_y,
                            trail_length,
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


__all__ = ["FISH_COLORS_BGR", "generate_trails"]
