"""2D reprojection overlay visualization observer for pipeline output assessment."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.interpolate

from aquapose.engine.events import Event, PipelineComplete

logger = logging.getLogger(__name__)

# Default colors (BGR for OpenCV).
_DEFAULT_REPROJECTED_COLOR = (0, 255, 0)  # green
_DEFAULT_MIDLINE_2D_COLOR = (255, 0, 0)  # blue


class Overlay2DObserver:
    """Generates MP4 video(s) with reprojected 3D midlines overlaid on camera frames.

    Subscribes to PipelineComplete to trigger video generation. Supports mosaic
    grid (all cameras tiled into one video) and per-camera individual videos.

    Args:
        output_dir: Directory for output video files.
        video_dir: Directory containing source camera MP4 files.
        calibration_path: Path to calibration JSON file for 3D-to-2D reprojection.
        mosaic: If True (default), generate a single mosaic video with all cameras.
        per_camera: If True, generate individual videos per camera.
        show_bbox: If True, draw bounding boxes around detections.
        show_fish_id: If True, draw fish ID text near detection centroids.
        fps: Output video frame rate.
        scale: Output scale factor (e.g. 0.5 for half resolution).
        reprojected_color: BGR color for reprojected 3D midlines.
        midline_2d_color: BGR color for original 2D midlines.

    Example::

        observer = Overlay2DObserver(
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
        mosaic: bool = True,
        per_camera: bool = False,
        show_bbox: bool = False,
        show_fish_id: bool = False,
        fps: float = 30.0,
        scale: float = 0.5,
        reprojected_color: tuple[int, int, int] = _DEFAULT_REPROJECTED_COLOR,
        midline_2d_color: tuple[int, int, int] = _DEFAULT_MIDLINE_2D_COLOR,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._video_dir = Path(video_dir)
        self._calibration_path = Path(calibration_path)
        self._mosaic = mosaic
        self._per_camera = per_camera
        self._show_bbox = show_bbox
        self._show_fish_id = show_fish_id
        self._fps = fps
        self._scale = scale
        self._reprojected_color = reprojected_color
        self._midline_2d_color = midline_2d_color

    def on_event(self, event: Event) -> None:
        """Receive a dispatched event and trigger overlay generation.

        Args:
            event: The event instance from the pipeline event bus.
        """
        if isinstance(event, PipelineComplete):
            context = event.context
            if context is None:
                return
            try:
                sys.stderr.write("Generating reprojection overlay videos...\n")
                sys.stderr.flush()
                self._generate_overlays(context)
            except Exception:
                logger.warning("Overlay generation failed", exc_info=True)

    def _generate_overlays(self, context: object) -> None:
        """Generate overlay videos from PipelineContext data.

        Args:
            context: PipelineContext with midlines_3d, annotated_detections,
                tracks, and camera_ids.
        """
        midlines_3d = getattr(context, "midlines_3d", None)
        camera_ids = getattr(context, "camera_ids", None)
        annotated_detections = getattr(context, "annotated_detections", None)

        if midlines_3d is None or camera_ids is None:
            logger.warning("Missing required context fields for overlay generation")
            return

        # Load calibration and build per-camera projection models.
        from aquapose.calibration.loader import load_calibration_data
        from aquapose.calibration.projection import RefractiveProjectionModel
        from aquapose.io.video import VideoSet

        calib_data = load_calibration_data(str(self._calibration_path))
        models: dict[str, RefractiveProjectionModel] = {}
        for cam_id in camera_ids:
            if cam_id in calib_data.cameras:
                cam = calib_data.cameras[cam_id]
                models[cam_id] = RefractiveProjectionModel(
                    K=cam.K,
                    R=cam.R,
                    t=cam.t,
                    water_z=calib_data.water_z,
                    normal=calib_data.interface_normal,
                    n_air=calib_data.n_air,
                    n_water=calib_data.n_water,
                )

        # Resolve video paths for each camera.
        # Supports both exact "{cam_id}.mp4" and prefix-match "{cam_id}-*.mp4"
        # (e.g. timestamped filenames like "e3v829d-20260218T145915-150429.mp4").
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
            logger.warning("No video files found in %s", self._video_dir)
            return

        n_frames = len(midlines_3d)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Use VideoSet for synchronized, undistorted frame reading.
        with VideoSet(camera_map, undistortion=calib_data) as video_set:
            # Frame dimensions from calibration (undistortion preserves size).
            first_cam = calib_data.cameras[next(iter(camera_map))]
            frame_w, frame_h = first_cam.image_size

            # Set up video writers.
            mosaic_writer = None
            per_cam_writers: dict[str, cv2.VideoWriter] = {}
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            # Compute scaled output dimensions.
            out_w = int(frame_w * self._scale)
            out_h = int(frame_h * self._scale)

            if self._mosaic:
                mosaic_h, mosaic_w = self._mosaic_dims(len(camera_ids), out_w, out_h)
                mosaic_path = self._output_dir / "overlay_mosaic.mp4"
                mosaic_writer = cv2.VideoWriter(
                    str(mosaic_path), fourcc, self._fps, (mosaic_w, mosaic_h)
                )

            if self._per_camera:
                for cam_id in camera_map:
                    cam_path = self._output_dir / f"overlay_{cam_id}.mp4"
                    per_cam_writers[cam_id] = cv2.VideoWriter(
                        str(cam_path), fourcc, self._fps, (out_w, out_h)
                    )

            try:
                for frame_idx, frames in video_set:
                    if frame_idx >= n_frames:
                        break

                    # Draw reprojected 3D midlines.
                    frame_midlines = midlines_3d[frame_idx]
                    if isinstance(frame_midlines, dict):
                        for _fish_id, spline in frame_midlines.items():
                            for cam_id in camera_ids:
                                if cam_id not in frames or cam_id not in models:
                                    continue
                                pts_2d = self._reproject_3d_midline(
                                    spline, cam_id, models[cam_id]
                                )
                                if pts_2d is not None:
                                    self._draw_midline(
                                        frames[cam_id],
                                        pts_2d,
                                        self._reprojected_color,
                                    )

                    # Draw original 2D midlines.
                    if annotated_detections is not None and frame_idx < len(
                        annotated_detections
                    ):
                        frame_dets = annotated_detections[frame_idx]
                        if isinstance(frame_dets, dict):
                            for cam_id, dets in frame_dets.items():
                                if cam_id not in frames:
                                    continue
                                for det in dets:
                                    midline = getattr(det, "midline", None)
                                    if midline is not None:
                                        pts = getattr(midline, "points", None)
                                        if pts is not None:
                                            pts_arr = np.asarray(pts, dtype=np.float32)
                                            self._draw_midline(
                                                frames[cam_id],
                                                pts_arr,
                                                self._midline_2d_color,
                                            )
                                    if self._show_bbox:
                                        bbox = getattr(det, "bbox", None)
                                        if bbox is not None:
                                            fish_id_label = (
                                                getattr(det, "fish_id", None)
                                                if self._show_fish_id
                                                else None
                                            )
                                            self._draw_bbox(
                                                frames[cam_id],
                                                bbox,
                                                self._midline_2d_color,
                                                fish_id=fish_id_label,
                                            )

                    # Scale frames down for output.
                    if self._scale != 1.0:
                        frames = {
                            cid: cv2.resize(f, (out_w, out_h))
                            for cid, f in frames.items()
                        }

                    # Write mosaic.
                    if mosaic_writer is not None:
                        mosaic_frame = self._build_mosaic(frames, camera_ids)
                        mosaic_writer.write(mosaic_frame)

                    # Write per-camera.
                    for cam_id, writer in per_cam_writers.items():
                        if cam_id in frames:
                            writer.write(frames[cam_id])
            finally:
                if mosaic_writer is not None:
                    mosaic_writer.release()
                for writer in per_cam_writers.values():
                    writer.release()

        logger.info("Overlay videos written to %s", self._output_dir)

    @staticmethod
    def _reproject_3d_midline(
        spline: object,
        cam_id: str,
        model: object,
    ) -> np.ndarray | None:
        """Evaluate spline at N points and project each to 2D.

        Args:
            spline: Object with ``control_points`` array (7, 3).
            cam_id: Camera identifier (unused, model is already per-camera).
            model: Per-camera RefractiveProjectionModel with ``project()`` method.

        Returns:
            2D pixel coordinates as (N, 2) array, or None on failure.
        """
        control_points = getattr(spline, "control_points", None)
        if control_points is None:
            return None

        cp = np.asarray(control_points, dtype=np.float64)
        knots = spline.knots
        degree = spline.degree
        try:
            bspl = scipy.interpolate.BSpline(
                np.asarray(knots, dtype=np.float64), cp, degree
            )
            t_vals = np.linspace(0.0, 1.0, 50)
            pts_3d = bspl(t_vals)  # shape (50, 3)
        except Exception:
            return None

        try:
            import torch

            pts_tensor = torch.tensor(pts_3d, dtype=torch.float32)
            pixels, valid = model.project(pts_tensor)  # type: ignore[union-attr]
            pixels_np = (
                pixels.cpu().numpy() if hasattr(pixels, "cpu") else np.asarray(pixels)
            )
            valid_np = (
                valid.cpu().numpy() if hasattr(valid, "cpu") else np.asarray(valid)
            )
            # Filter to valid projections only.
            pixels_np = pixels_np[valid_np]
            if len(pixels_np) < 2:
                return None
            return pixels_np.astype(np.float32)
        except Exception:
            return None

    @staticmethod
    def _draw_midline(
        frame: np.ndarray,
        points_2d: np.ndarray,
        color: tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        """Draw a polyline with a head-direction arrowhead on the frame.

        Points are assumed head-to-tail order (index 0 = head). A small
        filled arrowhead is drawn at the head end pointing forwards.

        Args:
            frame: BGR image to draw on (modified in-place).
            points_2d: (N, 2) array of pixel coordinates, head-to-tail.
            color: BGR color tuple.
            thickness: Line thickness in pixels.
        """
        pts = np.asarray(points_2d, dtype=np.int32)
        if len(pts) < 2:
            return
        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=thickness)

        # Draw arrowhead at the head end (pts[0]), pointing forwards.
        head = pts[0].astype(np.float64)
        neck = pts[1].astype(np.float64)
        direction = head - neck
        length = np.linalg.norm(direction)
        if length < 1e-3:
            return
        direction /= length

        # Arrow size scales with line thickness.
        arrow_len = max(thickness * 4.0, 8.0)
        arrow_half_w = max(thickness * 2.0, 4.0)

        # Perpendicular vector.
        perp = np.array([-direction[1], direction[0]])

        tip = head + direction * arrow_len
        left = head - perp * arrow_half_w
        right = head + perp * arrow_half_w
        triangle = np.array([tip, left, right], dtype=np.int32)
        cv2.fillPoly(frame, [triangle], color)

    @staticmethod
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

        # Compute grid dimensions.
        n_cols = math.ceil(math.sqrt(n_cams))
        n_rows = math.ceil(n_cams / n_cols)

        # Get cell size from the first available frame.
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
                # Resize if dimensions differ.
                if frame.shape[:2] != (cell_h, cell_w):
                    frame = cv2.resize(frame, (cell_w, cell_h))
                y0, y1 = row * cell_h, (row + 1) * cell_h
                x0, x1 = col * cell_w, (col + 1) * cell_w
                mosaic[y0:y1, x0:x1] = frame

        return mosaic

    @staticmethod
    def _draw_bbox(
        frame: np.ndarray,
        bbox: object,
        color: tuple[int, int, int],
        fish_id: int | None = None,
    ) -> None:
        """Draw a bounding box rectangle and optional fish ID text.

        Args:
            frame: BGR image to draw on (modified in-place).
            bbox: Bounding box with x1, y1, x2, y2 attributes or (x1, y1, x2, y2) tuple.
            color: BGR color tuple.
            fish_id: If provided, draw fish ID text near the top-left corner.
        """
        if hasattr(bbox, "x1"):
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        else:
            return

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if fish_id is not None:
            cv2.putText(
                frame,
                str(fish_id),
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    @staticmethod
    def _mosaic_dims(n_cameras: int, frame_w: int, frame_h: int) -> tuple[int, int]:
        """Compute mosaic output dimensions.

        Args:
            n_cameras: Number of cameras.
            frame_w: Width of each camera frame.
            frame_h: Height of each camera frame.

        Returns:
            Tuple of (mosaic_height, mosaic_width).
        """
        n_cols = math.ceil(math.sqrt(n_cameras))
        n_rows = math.ceil(n_cameras / n_cols)
        return n_rows * frame_h, n_cols * frame_w
