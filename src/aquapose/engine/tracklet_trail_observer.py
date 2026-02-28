"""Tracklet trail visualization observer for 2D tracking and cross-camera association diagnostics."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np

from aquapose.engine.events import Event, PipelineComplete

logger = logging.getLogger(__name__)

# Paul Tol 22-color palette in BGR for OpenCV.
# Each tuple is (B, G, R).
FISH_COLORS_BGR: list[tuple[int, int, int]] = [
    (112, 48, 0),  # blue    #003070
    (76, 211, 234),  # gold    #EAD34C
    (153, 170, 68),  # teal    #44AA99
    (238, 204, 102),  # cyan    #66CCEE
    (51, 136, 34),  # green   #228833
    (51, 153, 153),  # olive   #999933
    (119, 102, 238),  # coral   #EE6677
    (170, 153, 238),  # rose    #EE99AA
    (119, 51, 170),  # purple  #AA3377
    (136, 34, 51),  # indigo  #332288
    (51, 119, 238),  # orange  #EE7733
    (17, 51, 204),  # red     #CC3311
    (85, 34, 136),  # wine    #882255
    (119, 51, 238),  # magenta #EE3377
    (153, 68, 170),  # plum    #AA4499
    (170, 119, 68),  # steel   #4477AA
    (221, 170, 119),  # sky     #77AADD
    (119, 204, 221),  # sand    #DDCC77
    (51, 204, 187),  # lime    #BBCC33
    (102, 136, 238),  # peach   #EE8866
    (187, 187, 187),  # gray    #BBBBBB
    (119, 119, 119),  # dark gray #777777
]

_GRAY_BGR: tuple[int, int, int] = (119, 119, 119)


def _coasted_color(base_color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Blend a BGR color toward gray (128) at 50%, producing a washed-out version.

    Args:
        base_color: BGR color tuple for a detected frame.

    Returns:
        Lighter BGR color tuple for coasted frames.
    """
    return tuple(int(c * 0.5 + 128 * 0.5) for c in base_color)  # type: ignore[return-value]


class TrackletTrailObserver:
    """Generates per-camera centroid trail videos and a cross-camera association mosaic.

    Subscribes to PipelineComplete and produces two diagnostic video outputs:

    1. Per-camera trail videos — fading centroid polylines color-coded by global
       fish ID, with detected vs coasted frame distinction.
    2. Association mosaic — all cameras tiled in a grid, using the same color
       coding to reveal cross-camera association quality.

    This observer is passive (no pipeline state mutation) and fault-tolerant
    (exceptions are logged as warnings, not raised).

    Args:
        output_dir: Root directory for output. Videos are written to
            ``{output_dir}/observers/diagnostics/``.
        video_dir: Directory containing source camera MP4 files.
        calibration_path: Path to the calibration JSON (used by VideoSet for
            undistortion).
        trail_length: Number of past frames to include in the fading tail.
        fps: Output video frame rate.
        tile_scale: Downsampling factor applied to each camera tile in the
            mosaic (e.g., 0.35 for 35% of original size).
        stop_frame: If set, stop rendering after this frame index. When
            ``None`` (default), inherits ``config.detection.stop_frame``
            from the pipeline context at render time.

    Example::

        observer = TrackletTrailObserver(
            output_dir="/tmp/output",
            video_dir="/data/videos",
            calibration_path="/data/calibration.json",
        )
        pipeline = PosePipeline(stages=stages, config=config, observers=[observer])
        context = pipeline.run()
    """

    def __init__(
        self,
        output_dir: str | Path,
        video_dir: str | Path,
        calibration_path: str | Path,
        *,
        trail_length: int = 30,
        fps: float = 30.0,
        tile_scale: float = 0.35,
        stop_frame: int | None = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._video_dir = Path(video_dir)
        self._calibration_path = Path(calibration_path)
        self._trail_length = trail_length
        self._fps = fps
        self._tile_scale = tile_scale
        self._stop_frame = stop_frame

    # ------------------------------------------------------------------
    # Observer protocol
    # ------------------------------------------------------------------

    def on_event(self, event: Event) -> None:
        """Receive a dispatched event and trigger trail video generation.

        Only responds to PipelineComplete events. Requires both
        ``tracks_2d`` and ``tracklet_groups`` to be non-None on the context.

        Args:
            event: The event instance from the pipeline event bus.
        """
        if not isinstance(event, PipelineComplete):
            return

        context = event.context
        if context is None:
            return

        tracks_2d = getattr(context, "tracks_2d", None)
        tracklet_groups = getattr(context, "tracklet_groups", None)
        if tracks_2d is None or tracklet_groups is None:
            logger.warning(
                "TrackletTrailObserver: missing tracks_2d or tracklet_groups on context; skipping"
            )
            return

        try:
            self._generate_trail_videos(context)
        except Exception:
            logger.warning(
                "TrackletTrailObserver: trail generation failed", exc_info=True
            )

    # ------------------------------------------------------------------
    # Color helpers
    # ------------------------------------------------------------------

    def _build_color_map(
        self,
        tracklet_groups: list,
    ) -> tuple[dict[int, tuple[int, int, int]], dict[tuple[str, int], int]]:
        """Build fish_id -> BGR color and (camera_id, track_id) -> fish_id maps.

        Args:
            tracklet_groups: List of TrackletGroup objects.

        Returns:
            Tuple of (fish_color_map, track_to_fish_map) where fish_color_map
            maps fish_id to a BGR color tuple and track_to_fish_map maps
            (camera_id, track_id) to fish_id.
        """
        fish_color_map: dict[int, tuple[int, int, int]] = {}
        track_to_fish: dict[tuple[str, int], int] = {}

        for group in tracklet_groups:
            fish_id: int = group.fish_id
            color = FISH_COLORS_BGR[fish_id % len(FISH_COLORS_BGR)]
            fish_color_map[fish_id] = color
            for tracklet in group.tracklets:
                key = (tracklet.camera_id, tracklet.track_id)
                track_to_fish[key] = fish_id

        return fish_color_map, track_to_fish

    # ------------------------------------------------------------------
    # Frame-lookup structure
    # ------------------------------------------------------------------

    def _build_frame_lookup(
        self,
        tracks_2d: dict,
        track_to_fish: dict[tuple[str, int], int],
    ) -> dict[str, dict[int, list[tuple[object, int, int]]]]:
        """Build per-camera per-frame index for fast trail rendering.

        Returns a nested dict:
            camera_id -> frame_idx -> [(tracklet, idx_in_tracklet, fish_id)]

        Args:
            tracks_2d: dict[str, list[Tracklet2D]] from PipelineContext.
            track_to_fish: Mapping (camera_id, track_id) -> fish_id.

        Returns:
            Per-camera per-frame lookup structure.
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

    # ------------------------------------------------------------------
    # Trail drawing helpers
    # ------------------------------------------------------------------

    def _draw_trail(
        self,
        frame: np.ndarray,
        tracklet: object,
        current_idx: int,
        fish_id: int,
        fish_color_map: dict[int, tuple[int, int, int]],
    ) -> None:
        """Draw a fading polyline trail and fish ID label on a frame.

        Args:
            frame: BGR image to draw on (modified in-place).
            tracklet: Tracklet2D with frames, centroids, frame_status.
            current_idx: Current index within tracklet.frames.
            fish_id: Global fish identity (or -1 for ungrouped).
            fish_color_map: Mapping from fish_id to BGR color.
        """
        base_color = fish_color_map.get(fish_id, _GRAY_BGR)

        # Collect trail points from max(0, current_idx - trail_length) to current_idx.
        start_idx = max(0, current_idx - self._trail_length)
        trail_pts = list(tracklet.centroids[start_idx : current_idx + 1])  # type: ignore[index]
        trail_statuses = list(tracklet.frame_status[start_idx : current_idx + 1])  # type: ignore[index]
        n_pts = len(trail_pts)

        if n_pts < 1:
            return

        # Draw fading segments (oldest alpha 0.3, newest 1.0).
        for seg_i in range(n_pts - 1):
            alpha = 0.3 + 0.7 * (seg_i / max(n_pts - 1, 1))
            status = trail_statuses[seg_i]
            seg_color = (
                base_color if status == "detected" else _coasted_color(base_color)
            )

            # Alpha-blend a line overlay.
            overlay = frame.copy()
            pt1 = (int(trail_pts[seg_i][0]), int(trail_pts[seg_i][1]))
            pt2 = (int(trail_pts[seg_i + 1][0]), int(trail_pts[seg_i + 1][1]))
            cv2.line(overlay, pt1, pt2, seg_color, 2)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw trail head dot.
        head_u = int(trail_pts[-1][0])
        head_v = int(trail_pts[-1][1])
        head_status = trail_statuses[-1]
        head_color = (
            base_color if head_status == "detected" else _coasted_color(base_color)
        )
        cv2.circle(frame, (head_u, head_v), 4, head_color, -1)

        # Draw global fish ID label at trail head.
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

    # ------------------------------------------------------------------
    # Per-camera trail video generation
    # ------------------------------------------------------------------

    def _generate_per_camera_trails(
        self,
        camera_ids: list[str],
        tracks_2d: dict,
        frame_lookup: dict[str, dict[int, list[tuple[object, int, int]]]],
        fish_color_map: dict[int, tuple[int, int, int]],
        camera_map: dict[str, Path],
        calib_data: object,
        diag_dir: Path,
    ) -> None:
        """Write per-camera trail MP4 files.

        Args:
            camera_ids: Ordered list of camera IDs.
            tracks_2d: Per-camera tracklet lists.
            frame_lookup: Pre-built frame lookup structure.
            fish_color_map: fish_id -> BGR color.
            camera_map: camera_id -> video Path.
            calib_data: CalibrationData for VideoSet undistortion.
            diag_dir: Output directory for diagnostic videos.
        """
        from aquapose.io.video import VideoSet

        for cam_id in camera_ids:
            if cam_id not in camera_map:
                continue
            cam_lookup = frame_lookup.get(cam_id, {})
            if not cam_lookup:
                # No tracklets for this camera — skip.
                continue

            cam_map_single = {cam_id: camera_map[cam_id]}
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer: cv2.VideoWriter | None = None

            try:
                with VideoSet(cam_map_single, undistortion=calib_data) as vs:  # type: ignore[arg-type]
                    for frame_idx, frames in vs:
                        if (
                            self._stop_frame is not None
                            and frame_idx >= self._stop_frame
                        ):
                            break
                        frame = frames.get(cam_id)
                        if frame is None:
                            continue

                        # Lazy-init writer with actual frame size.
                        if writer is None:
                            h, w = frame.shape[:2]
                            out_path = diag_dir / f"tracklet_trails_{cam_id}.mp4"
                            writer = cv2.VideoWriter(
                                str(out_path), fourcc, self._fps, (w, h)
                            )

                        active = cam_lookup.get(frame_idx, [])
                        for tracklet, idx_in_tracklet, fish_id in active:
                            self._draw_trail(
                                frame,
                                tracklet,
                                idx_in_tracklet,
                                fish_id,
                                fish_color_map,
                            )

                        writer.write(frame)
            finally:
                if writer is not None:
                    writer.release()

    # ------------------------------------------------------------------
    # Association mosaic generation
    # ------------------------------------------------------------------

    @staticmethod
    def _mosaic_dims(n_cameras: int, tile_w: int, tile_h: int) -> tuple[int, int]:
        """Compute mosaic canvas dimensions for the given tile layout.

        Args:
            n_cameras: Total number of camera tiles.
            tile_w: Width of one camera tile in pixels.
            tile_h: Height of one camera tile in pixels.

        Returns:
            (mosaic_height, mosaic_width).
        """
        n_cols = math.ceil(math.sqrt(n_cameras))
        n_rows = math.ceil(n_cameras / n_cols)
        return n_rows * tile_h, n_cols * tile_w

    @staticmethod
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

    def _generate_association_mosaic(
        self,
        camera_ids: list[str],
        frame_lookup: dict[str, dict[int, list[tuple[object, int, int]]]],
        fish_color_map: dict[int, tuple[int, int, int]],
        camera_map: dict[str, Path],
        calib_data: object,
        diag_dir: Path,
    ) -> None:
        """Write the association mosaic MP4.

        Each frame is a tiled grid of all camera views with centroid trails
        drawn in consistent global fish ID colors, enabling quick visual
        inspection of cross-camera association quality.

        Args:
            camera_ids: Ordered list of all camera IDs.
            frame_lookup: Pre-built frame lookup structure.
            fish_color_map: fish_id -> BGR color.
            camera_map: camera_id -> video Path.
            calib_data: CalibrationData for VideoSet undistortion.
            diag_dir: Output directory for the mosaic video.
        """
        from aquapose.io.video import VideoSet

        if not camera_map:
            return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer: cv2.VideoWriter | None = None
        tile_w: int = 0
        tile_h: int = 0

        try:
            with VideoSet(camera_map, undistortion=calib_data) as vs:  # type: ignore[arg-type]
                for frame_idx, frames in vs:
                    if self._stop_frame is not None and frame_idx >= self._stop_frame:
                        break
                    # Determine tile dimensions from first real frame.
                    if tile_w == 0:
                        for cam_id in camera_ids:
                            if cam_id in frames:
                                raw_h, raw_w = frames[cam_id].shape[:2]
                                tile_w = max(1, int(raw_w * self._tile_scale))
                                tile_h = max(1, int(raw_h * self._tile_scale))
                                break

                    if tile_w == 0:
                        break

                    # Build downsampled tiles with trails drawn.
                    tiles: dict[str, np.ndarray] = {}
                    for cam_id in camera_ids:
                        raw = frames.get(cam_id)
                        if raw is None:
                            tiles[cam_id] = np.zeros(
                                (tile_h, tile_w, 3), dtype=np.uint8
                            )
                            continue

                        tile = cv2.resize(raw, (tile_w, tile_h))
                        scale_x = tile_w / raw.shape[1]
                        scale_y = tile_h / raw.shape[0]

                        cam_lookup = frame_lookup.get(cam_id, {})
                        active = cam_lookup.get(frame_idx, [])
                        for tracklet, idx_in_tracklet, fish_id in active:
                            # Build a scaled version of the tracklet for the tile.
                            # We draw directly on the tile using scaled centroids.
                            self._draw_trail_scaled(
                                tile,
                                tracklet,
                                idx_in_tracklet,
                                fish_id,
                                fish_color_map,
                                scale_x,
                                scale_y,
                            )

                        # Add camera label.
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

                    mosaic = self._build_mosaic(tiles, camera_ids, tile_w, tile_h)

                    if writer is None:
                        mh, mw = mosaic.shape[:2]
                        out_path = diag_dir / "association_mosaic.mp4"
                        writer = cv2.VideoWriter(
                            str(out_path), fourcc, self._fps, (mw, mh)
                        )

                    writer.write(mosaic)
        finally:
            if writer is not None:
                writer.release()

    def _draw_trail_scaled(
        self,
        tile: np.ndarray,
        tracklet: object,
        current_idx: int,
        fish_id: int,
        fish_color_map: dict[int, tuple[int, int, int]],
        scale_x: float,
        scale_y: float,
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
        """
        base_color = fish_color_map.get(fish_id, _GRAY_BGR)
        start_idx = max(0, current_idx - self._trail_length)
        trail_pts = list(tracklet.centroids[start_idx : current_idx + 1])  # type: ignore[index]
        trail_statuses = list(tracklet.frame_status[start_idx : current_idx + 1])  # type: ignore[index]
        n_pts = len(trail_pts)

        if n_pts < 1:
            return

        for seg_i in range(n_pts - 1):
            alpha = 0.3 + 0.7 * (seg_i / max(n_pts - 1, 1))
            status = trail_statuses[seg_i]
            seg_color = (
                base_color if status == "detected" else _coasted_color(base_color)
            )
            overlay = tile.copy()
            pt1 = (
                int(trail_pts[seg_i][0] * scale_x),
                int(trail_pts[seg_i][1] * scale_y),
            )
            pt2 = (
                int(trail_pts[seg_i + 1][0] * scale_x),
                int(trail_pts[seg_i + 1][1] * scale_y),
            )
            cv2.line(overlay, pt1, pt2, seg_color, 1)
            cv2.addWeighted(overlay, alpha, tile, 1 - alpha, 0, tile)

        head_u = int(trail_pts[-1][0] * scale_x)
        head_v = int(trail_pts[-1][1] * scale_y)
        head_status = trail_statuses[-1]
        head_color = (
            base_color if head_status == "detected" else _coasted_color(base_color)
        )
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

    # ------------------------------------------------------------------
    # Top-level generation orchestrator
    # ------------------------------------------------------------------

    def _generate_trail_videos(self, context: object) -> None:
        """Orchestrate per-camera trail and mosaic video generation.

        Args:
            context: PipelineContext with tracks_2d, tracklet_groups, camera_ids.
        """
        from aquapose.calibration.loader import load_calibration_data

        tracks_2d: dict = getattr(context, "tracks_2d", {}) or {}
        tracklet_groups: list = getattr(context, "tracklet_groups", []) or []
        camera_ids: list[str] = getattr(context, "camera_ids", None) or list(
            tracks_2d.keys()
        )

        # Inherit stop_frame from pipeline config if not explicitly set
        if self._stop_frame is None:
            config = getattr(context, "config", None)
            detection = getattr(config, "detection", None)
            sf = getattr(detection, "stop_frame", None)
            if sf is not None:
                self._stop_frame = sf

        # Build color maps.
        fish_color_map, track_to_fish = self._build_color_map(tracklet_groups)

        # Build frame lookup.
        frame_lookup = self._build_frame_lookup(tracks_2d, track_to_fish)

        # Load calibration for undistortion.
        calib_data = load_calibration_data(str(self._calibration_path))

        # Resolve video paths.
        camera_map: dict[str, Path] = {}
        for cam_id in camera_ids:
            video_path = self._video_dir / f"{cam_id}.mp4"
            if not video_path.exists():
                matches = sorted(self._video_dir.glob(f"{cam_id}-*.mp4"))
                if matches:
                    video_path = matches[0]
                else:
                    continue
            camera_map[cam_id] = video_path

        if not camera_map:
            logger.warning(
                "TrackletTrailObserver: no video files found in %s", self._video_dir
            )
            return

        # Create output directory.
        diag_dir = self._output_dir / "observers" / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        # Generate per-camera trail videos.
        self._generate_per_camera_trails(
            camera_ids,
            tracks_2d,
            frame_lookup,
            fish_color_map,
            camera_map,
            calib_data,
            diag_dir,
        )

        # Generate association mosaic.
        self._generate_association_mosaic(
            camera_ids, frame_lookup, fish_color_map, camera_map, calib_data, diag_dir
        )

        logger.info("TrackletTrailObserver: diagnostic videos written to %s", diag_dir)


__all__ = ["FISH_COLORS_BGR", "TrackletTrailObserver"]
