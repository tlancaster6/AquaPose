"""2D reprojection overlay renderer for fish midlines on camera frames.

Projects 3D B-spline midlines through the refractive camera model and draws
polyline overlays with optional width indicators on video frames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import scipy.interpolate
import torch

if TYPE_CHECKING:
    from aquapose.calibration.projection import RefractiveProjectionModel
    from aquapose.reconstruction.triangulation import Midline3D

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared color palette (BGR for OpenCV)
# 10 visually distinct colors cycling by fish_id % 10
# ---------------------------------------------------------------------------

FISH_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),  # 0: green
    (255, 0, 0),  # 1: blue
    (0, 0, 255),  # 2: red
    (0, 255, 255),  # 3: yellow
    (255, 0, 255),  # 4: magenta
    (255, 165, 0),  # 5: orange
    (128, 0, 128),  # 6: purple
    (0, 165, 255),  # 7: amber
    (0, 128, 255),  # 8: orange-red
    (255, 255, 0),  # 9: cyan
    (0, 128, 0),  # 10: dark green
    (128, 0, 0),  # 11: dark blue (navy)
    (0, 0, 128),  # 12: dark red (maroon)
    (0, 200, 200),  # 13: teal
    (200, 0, 200),  # 14: dark magenta
    (128, 128, 0),  # 15: olive
    (0, 200, 128),  # 16: spring green
    (200, 128, 0),  # 17: steel blue
    (64, 64, 255),  # 18: salmon
    (128, 255, 128),  # 19: light green
]


def draw_midline_overlay(
    frame: np.ndarray,
    midline: Midline3D,
    model: RefractiveProjectionModel,
    *,
    color: tuple[int, int, int] | None = None,
    thickness: int = 2,
    n_eval: int = 30,
    draw_widths: bool = True,
) -> np.ndarray:
    """Draw a 3D midline reprojection overlay onto a camera frame.

    Evaluates the B-spline at ``n_eval`` points, projects each through the
    refractive camera model, and draws a polyline. Width circles are drawn at
    every 5th point using the pinhole half-width approximation.

    Args:
        frame: BGR image as uint8 ndarray, shape (H, W, 3). Modified in-place.
        midline: 3D midline with B-spline control points.
        model: Refractive projection model for the target camera.
        color: BGR color tuple. If None, uses ``FISH_COLORS[fish_id % 10]``.
        thickness: Polyline and circle stroke width in pixels.
        n_eval: Number of evaluation points along the spline.
        draw_widths: If True, draw width circles at every 5th evaluation point.

    Returns:
        The annotated frame (same array, modified in-place).
    """
    if color is None:
        color = FISH_COLORS[midline.fish_id % len(FISH_COLORS)]

    # Evaluate B-spline at n_eval uniform parameter values
    u_vals = np.linspace(0.0, 1.0, n_eval)
    spline = scipy.interpolate.BSpline(
        midline.knots.astype(np.float64),
        midline.control_points.astype(np.float64),
        midline.degree,
    )
    pts_3d = spline(u_vals).astype(np.float32)  # shape (n_eval, 3)

    # Project all points to 2D
    pts_tensor = torch.from_numpy(pts_3d).to(model.C.device)  # (n_eval, 3)
    pixels, valid = model.project(pts_tensor)  # (n_eval, 2), (n_eval,)
    pixels_np = pixels.detach().cpu().numpy()  # (n_eval, 2)
    valid_np = valid.detach().cpu().numpy()  # (n_eval,)

    # Build list of valid pixel coordinates for polyline
    polyline_pts: list[tuple[int, int]] = []
    for i in range(n_eval):
        if valid_np[i] and not np.any(np.isnan(pixels_np[i])):
            u_px = round(float(pixels_np[i, 0]))
            v_px = round(float(pixels_np[i, 1]))
            polyline_pts.append((u_px, v_px))

    if len(polyline_pts) >= 2:
        pts_array = np.array(polyline_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            frame, [pts_array], isClosed=False, color=color, thickness=thickness
        )

        # Draw arrowhead at the head end (index 0), pointing forwards.
        head = np.array(polyline_pts[0], dtype=np.float64)
        neck = np.array(polyline_pts[1], dtype=np.float64)
        direction = head - neck
        length = np.linalg.norm(direction)
        if length >= 1e-3:
            direction /= length
            arrow_len = max(thickness * 4.0, 8.0)
            arrow_half_w = max(thickness * 2.0, 4.0)
            perp = np.array([-direction[1], direction[0]])
            tip = head + direction * arrow_len
            left = head - perp * arrow_half_w
            right = head + perp * arrow_half_w
            triangle = np.array([tip, left, right], dtype=np.int32)
            cv2.fillPoly(frame, [triangle], (255, 255, 255))

    # Draw width circles at every 5th valid point
    if draw_widths and len(polyline_pts) > 0:
        # Interpolate half-widths at n_eval positions from the stored 15 samples
        n_hw = len(midline.half_widths)
        u_hw = np.linspace(0.0, 1.0, n_hw)
        hw_interp = np.interp(u_vals, u_hw, midline.half_widths.astype(np.float64))

        # Get focal length from intrinsics (mean of fx, fy)
        focal_px = float((model.K[0, 0].item() + model.K[1, 1].item()) / 2.0)

        for i in range(0, n_eval, 5):
            if not valid_np[i] or np.any(np.isnan(pixels_np[i])):
                continue
            half_width_m = float(hw_interp[i])
            if half_width_m <= 0.0:
                continue

            # Depth of the 3D point below the water surface
            depth_m = max(0.01, float(pts_3d[i, 2]) - model.water_z)

            # Pinhole approximation: radius_px = half_width_m * focal_px / depth_m
            radius_px = max(1, round(half_width_m * focal_px / depth_m))

            cx = round(float(pixels_np[i, 0]))
            cy = round(float(pixels_np[i, 1]))
            cv2.circle(frame, (cx, cy), radius_px, color, thickness)

    return frame


def render_overlay_video(
    video_path: Path,
    output_path: Path,
    midlines_per_frame: list[dict[int, Midline3D]],
    model: RefractiveProjectionModel,
    *,
    fps: float = 30.0,
    codec: str = "mp4v",
) -> None:
    """Render a per-camera overlay video with 3D midline reprojections.

    Opens the input video, draws midline overlays for all fish in each frame,
    and writes the annotated frames to an output video file.

    Args:
        video_path: Path to the input camera video file.
        output_path: Path for the output overlay video (MP4).
        midlines_per_frame: Per-frame fish midlines, indexed by frame index.
            Each entry maps fish_id to Midline3D.
        model: Refractive projection model for this camera.
        fps: Output video frame rate.
        codec: FourCC codec string (default ``"mp4v"``).
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            logger.error("Failed to open video: %s", video_path)
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter.fourcc(*codec)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        try:
            n_frames = min(
                len(midlines_per_frame), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            )

            for frame_idx in range(n_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                frame_midlines = midlines_per_frame[frame_idx]
                for _fish_id, midline in frame_midlines.items():
                    draw_midline_overlay(frame, midline, model)

                writer.write(frame)

            logger.info(
                "Overlay video written to %s (%d frames)", output_path, n_frames
            )
        finally:
            writer.release()
    finally:
        cap.release()
