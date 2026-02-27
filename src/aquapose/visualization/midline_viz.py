"""Per-stage diagnostic visualizations for detection, tracking, and midline extraction.

Provides visualization functions for pipeline stages 1 (detection), 3 (tracking),
and 4 (midline extraction): detection grids, confidence histograms, claiming overlays,
midline extraction montages, and skip-reason pie charts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from matplotlib import pyplot as plt

from aquapose.visualization.overlay import FISH_COLORS

if TYPE_CHECKING:
    from aquapose.calibration.projection import RefractiveProjectionModel
    from aquapose.core.tracking import FishTrack, TrackState
    from aquapose.segmentation.crop import CropRegion
    from aquapose.segmentation.detector import Detection

logger = logging.getLogger(__name__)


__all__ = [
    "TrackSnapshot",
    "vis_claiming_overlay",
    "vis_confidence_histogram",
    "vis_detection_grid",
    "vis_midline_extraction_montage",
    "vis_skip_reason_pie",
]


# ---------------------------------------------------------------------------
# TrackSnapshot dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrackSnapshot:
    """Lightweight snapshot of a track's state at a single frame.

    Attributes:
        fish_id: Globally unique fish identifier.
        position: 3D centroid position, shape (3,).
        state: Track lifecycle state (PROBATIONARY, CONFIRMED, COASTING).
        camera_detections: Camera-to-detection-index mapping for this frame.
    """

    fish_id: int
    position: np.ndarray
    state: TrackState
    camera_detections: dict[str, int]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _orientation_invariant_ctrl_error(
    recon_ctrl: np.ndarray,
    gt_ctrl: np.ndarray,
) -> np.ndarray:
    """Compute per-control-point errors invariant to head-tail flip.

    PCA orientation is unsigned, so the optimizer may converge with the
    spline reversed relative to GT. Since the knot vector is symmetric,
    reversing control points correctly reverses the spline direction.
    This function computes errors for both orderings and returns whichever
    has lower mean error.

    Args:
        recon_ctrl: Reconstructed control points, shape (K, 3).
        gt_ctrl: Ground-truth control points, shape (K, 3).

    Returns:
        Per-control-point error array, shape (K,), in metres.
    """
    fwd_errors = np.linalg.norm(recon_ctrl - gt_ctrl, axis=1)
    rev_errors = np.linalg.norm(recon_ctrl[::-1] - gt_ctrl, axis=1)
    if rev_errors.mean() < fwd_errors.mean():
        return rev_errors
    return fwd_errors


# ---------------------------------------------------------------------------
# Stage 1: Detection
# ---------------------------------------------------------------------------


def vis_detection_grid(
    detections_per_frame: list[dict[str, list[Detection]]],
    video_set: object,
    output_path: Path,
) -> None:
    """Save a 3x3 grid of frames with detection bounding boxes.

    Selects 9 evenly spaced frame indices and for each picks the camera with
    the most detections. Draws bounding boxes with confidence labels.

    Args:
        detections_per_frame: Per-frame detection dicts from Stage 1.
        video_set: Opened VideoSet providing undistorted frames.
        output_path: Output PNG path.
    """
    n_frames = len(detections_per_frame)
    if n_frames == 0:
        logger.warning("vis_detection_grid: no frames")
        return

    camera_ids = set(video_set.camera_ids)  # type: ignore[union-attr]
    frame_indices = np.linspace(0, n_frames - 1, 9, dtype=int)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes_flat = axes.flatten()

    for ax_idx, fi in enumerate(frame_indices):
        ax = axes_flat[ax_idx]
        frame_dets = detections_per_frame[fi]

        # Pick camera with most detections
        best_cam = max(frame_dets, key=lambda c: len(frame_dets[c]), default=None)
        if best_cam is None or best_cam not in camera_ids:
            ax.set_title(f"Frame {fi}: no data")
            ax.axis("off")
            continue

        try:
            frame = video_set.read_frame(fi)[best_cam]  # type: ignore[union-attr]
        except (IndexError, RuntimeError):
            ax.set_title(f"Frame {fi}: read failed")
            ax.axis("off")
            continue

        # Draw bounding boxes
        dets = frame_dets[best_cam]
        for det in dets:
            x, y, w, h = det.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{det.confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        ax.set_title(f"Frame {fi} | {best_cam} ({len(dets)} det)")
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Detection grid saved to %s", output_path)


def vis_confidence_histogram(
    detections_per_frame: list[dict[str, list[Detection]]],
    output_path: Path,
) -> None:
    """Save a histogram of detection confidence values.

    Args:
        detections_per_frame: Per-frame detection dicts from Stage 1.
        output_path: Output PNG path.
    """
    confidences = [
        det.confidence
        for frame_dets in detections_per_frame
        for dets in frame_dets.values()
        for det in dets
    ]
    if not confidences:
        logger.warning("vis_confidence_histogram: no detections")
        return

    arr = np.array(confidences)
    mean_val = float(arr.mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.3f}")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title(f"Detection Confidence Distribution (n={len(confidences)})")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confidence histogram saved to %s", output_path)


# ---------------------------------------------------------------------------
# Stage 3: Tracking
# ---------------------------------------------------------------------------


def vis_claiming_overlay(
    snapshots_per_frame: list[list[TrackSnapshot]],
    detections_per_frame: list[dict[str, list[Detection]]],
    video_set: object,
    models: dict[str, RefractiveProjectionModel],
    output_path: Path,
    *,
    cameras: list[str] | None = None,
    fps: float = 15.0,
) -> None:
    """Save a 2x2 tiled video showing track claiming overlaid on camera frames.

    For each frame, shows detection boxes (gray=unclaimed, colored=claimed)
    and reprojected track positions with lines to claimed detections.

    Args:
        snapshots_per_frame: Per-frame lists of TrackSnapshot objects.
        detections_per_frame: Per-frame detection dicts from Stage 1.
        video_set: Opened VideoSet providing undistorted frames.
        models: Per-camera refractive projection models.
        output_path: Output MP4 path.
        cameras: Optional list of 4 camera IDs. If None, picks top 4 by
            total detection count.
        fps: Output video frame rate.
    """
    import torch

    n_frames = len(detections_per_frame)
    if n_frames == 0:
        logger.warning("vis_claiming_overlay: no frames")
        return

    available_cams = set(video_set.camera_ids)  # type: ignore[union-attr]

    # Select top cameras by detection count (up to 6 for 2x3 grid)
    if cameras is None:
        cam_counts: dict[str, int] = {}
        for frame_dets in detections_per_frame:
            for cam, dets in frame_dets.items():
                cam_counts[cam] = cam_counts.get(cam, 0) + len(dets)
        sorted_cams = sorted(cam_counts, key=lambda c: cam_counts[c], reverse=True)
        cameras = sorted_cams[:6]

    cameras = [c for c in cameras if c in available_cams]
    if not cameras:
        logger.warning("vis_claiming_overlay: no valid cameras")
        return

    # Determine tile size from first frame (2x3 grid: 2 rows, 3 cols)
    n_cols = 3
    n_rows = 2
    sample_frame = video_set.read_frame(0)[cameras[0]]  # type: ignore[union-attr]
    full_h, full_w = sample_frame.shape[:2]
    tile_w = full_w // n_cols
    tile_h = full_h // n_rows

    out_w = tile_w * n_cols
    out_h = tile_h * n_rows

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    try:
        for fi in range(min(n_frames, len(snapshots_per_frame))):
            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            frame_snaps = snapshots_per_frame[fi]
            frame_dets = detections_per_frame[fi]

            # Read all camera frames for this index
            all_frames = video_set.read_frame(fi)  # type: ignore[union-attr]

            # Build claim lookup: cam -> det_idx -> fish_id
            claim_map: dict[str, dict[int, int]] = {}
            for snap in frame_snaps:
                for cam, di in snap.camera_detections.items():
                    claim_map.setdefault(cam, {})[di] = snap.fish_id

            for cam_idx, cam in enumerate(cameras[: n_cols * n_rows]):
                if cam not in all_frames:
                    continue
                frame = all_frames[cam]

                tile = cv2.resize(frame, (tile_w, tile_h))
                scale_x = tile_w / frame.shape[1]
                scale_y = tile_h / frame.shape[0]

                dets = frame_dets.get(cam, [])
                cam_claims = claim_map.get(cam, {})

                # Draw all detection boxes
                for di, det in enumerate(dets):
                    bx, by, bw, bh = det.bbox
                    sx = int(bx * scale_x)
                    sy = int(by * scale_y)
                    sw = int(bw * scale_x)
                    sh = int(bh * scale_y)

                    if di in cam_claims:
                        fish_id = cam_claims[di]
                        color = FISH_COLORS[fish_id % len(FISH_COLORS)]
                    else:
                        color = (128, 128, 128)

                    cv2.rectangle(tile, (sx, sy), (sx + sw, sy + sh), color, 2)

                # Draw reprojected track positions
                if cam in models:
                    model = models[cam]
                    for snap in frame_snaps:
                        color = FISH_COLORS[snap.fish_id % len(FISH_COLORS)]
                        pos_tensor = torch.tensor(
                            snap.position,
                            dtype=torch.float32,
                            device=model.K.device,
                        ).unsqueeze(0)
                        pixels, valid = model.project(pos_tensor)
                        if valid[0]:
                            px = int(float(pixels[0, 0]) * scale_x)
                            py = int(float(pixels[0, 1]) * scale_y)
                            cv2.circle(tile, (px, py), 5, color, -1)

                            # Draw line to claimed bbox center
                            if cam in snap.camera_detections:
                                di = snap.camera_detections[cam]
                                if di < len(dets):
                                    bx, by, bw, bh = dets[di].bbox
                                    cx = int((bx + bw / 2) * scale_x)
                                    cy = int((by + bh / 2) * scale_y)
                                    cv2.line(tile, (px, py), (cx, cy), color, 1)

                # Place tile in canvas
                row = cam_idx // n_cols
                col = cam_idx % n_cols
                y0, y1 = row * tile_h, (row + 1) * tile_h
                x0, x1 = col * tile_w, (col + 1) * tile_w
                canvas[y0:y1, x0:x1] = tile

                # Camera label
                cv2.putText(
                    canvas,
                    cam,
                    (x0 + 5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            writer.write(canvas)
    finally:
        writer.release()

    logger.info("Claiming overlay video saved to %s", output_path)


# ---------------------------------------------------------------------------
# Stage 4: Midline Extraction
# ---------------------------------------------------------------------------


def vis_midline_extraction_montage(
    tracks_per_frame: list[list[FishTrack]],
    masks_per_frame: list[dict[str, list[tuple[np.ndarray, CropRegion]]]],
    detections_per_frame: list[dict[str, list[Detection]]],
    video_set: object,
    output_path: Path,
) -> None:
    """Save a 5x5 montage showing crop and midline extraction sub-pipeline stages.

    Each cell shows 5 panels: crop | mask overlay | smoothed | skeleton+path |
    15-pt midline on crop. All panels are auto-rotated to landscape orientation.
    Uniformly samples 25 valid (track, camera, mask) triples.

    Args:
        tracks_per_frame: Per-frame confirmed track lists from Stage 3.
        masks_per_frame: Per-frame mask dicts from Stage 2.
        detections_per_frame: Per-frame detection dicts from Stage 1.
        video_set: Opened VideoSet providing undistorted frames.
        output_path: Output PNG path.
    """
    from aquapose.reconstruction.midline import (
        _adaptive_smooth,
        _check_skip_mask,
        _longest_path_bfs,
        _resample_arc_length,
        _skeleton_and_widths,
    )

    # Build flat list of valid (frame_idx, cam, det_idx, mask, crop_region) tuples
    camera_ids = set(video_set.camera_ids)  # type: ignore[union-attr]
    valid_tuples: list[tuple[int, str, int, np.ndarray, CropRegion]] = []
    for fi, tracks in enumerate(tracks_per_frame):
        if fi >= len(masks_per_frame):
            break
        frame_masks = masks_per_frame[fi]
        for track in tracks:
            for cam, di in track.camera_detections.items():
                if cam not in frame_masks or cam not in camera_ids:
                    continue
                mask_list = frame_masks[cam]
                if di >= len(mask_list):
                    continue
                mask, crop_region = mask_list[di]
                skip = _check_skip_mask(mask, crop_region)
                if skip is None and np.count_nonzero(mask) > 0:
                    valid_tuples.append((fi, cam, di, mask, crop_region))

    if not valid_tuples:
        logger.warning("vis_midline_extraction_montage: no valid triples")
        return

    n_samples = min(30, len(valid_tuples))
    sample_indices = np.linspace(0, len(valid_tuples) - 1, n_samples, dtype=int)
    samples = [valid_tuples[i] for i in sample_indices]

    n_cols = 3
    n_rows = 10
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(27, 15))
    axes_flat = axes.flatten()

    for ax_idx, (fi, cam, _di, mask, crop_region) in enumerate(samples):
        ax = axes_flat[ax_idx]

        # Read the frame (undistorted via VideoSet) and extract the crop
        try:
            frame = video_set.read_frame(fi)[cam]  # type: ignore[union-attr]
        except (IndexError, RuntimeError, KeyError):
            ax.set_title(f"F{fi} {cam}")
            ax.axis("off")
            continue

        crop = frame[crop_region.y1 : crop_region.y2, crop_region.x1 : crop_region.x2]
        h, w = mask.shape[:2]
        crop_h, crop_w = crop.shape[:2]

        # Resize mask to match crop dimensions
        if mask.shape[:2] != (crop_h, crop_w):
            mask_resized = cv2.resize(
                mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_resized = mask

        # Run sub-pipeline
        smooth = _adaptive_smooth(mask)
        skel_bool, dt = _skeleton_and_widths(smooth)
        path_yx = _longest_path_bfs(skel_bool)

        # Pad a panel to square with black bars (letterbox/pillarbox)
        def _pad_square(img: np.ndarray) -> np.ndarray:
            ih, iw = img.shape[:2]
            side = max(ih, iw)
            out = np.zeros((side, side, 3), dtype=np.uint8)
            y_off = (side - ih) // 2
            x_off = (side - iw) // 2
            out[y_off : y_off + ih, x_off : x_off + iw] = img
            return out

        # Panel 1: raw crop
        p1 = crop.copy()

        # Panel 2: mask overlay on crop (green)
        p2 = crop.copy()
        green = np.zeros_like(crop)
        green[:, :, 1] = 255
        mask_bool = mask_resized > 0
        p2[mask_bool] = cv2.addWeighted(crop, 0.5, green, 0.5, 0)[mask_bool]

        # Panels 3-5 are mask-resolution; build at mask size then resize

        # Panel 3: smoothed mask
        p3_raw = cv2.cvtColor(
            np.where(smooth > 0, np.uint8(255), np.uint8(0)),
            cv2.COLOR_GRAY2RGB,
        )
        p3 = cv2.resize(p3_raw, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

        # Panel 4: skeleton (gray) + longest path (green) on black
        p4_raw = np.zeros((h, w, 3), dtype=np.uint8)
        p4_raw[skel_bool] = (128, 128, 128)
        if path_yx:
            for r, c in path_yx:
                if 0 <= r < h and 0 <= c < w:
                    p4_raw[r, c] = (0, 255, 0)
        p4 = cv2.resize(p4_raw, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

        # Panel 5: 15-pt midline on crop
        p5 = crop.copy()
        # Panel 6: half-width circles on crop
        p6 = crop.copy()
        midline_data: tuple[np.ndarray, np.ndarray] | None = None
        if path_yx and len(path_yx) >= 2:
            xy_pts, hw_pts = _resample_arc_length(path_yx, dt, 15)
            midline_data = (xy_pts, hw_pts)
            # Scale midline points from mask coords to crop coords
            sx = crop_w / w if w > 0 else 1.0
            sy = crop_h / h if h > 0 else 1.0
            for pt in xy_pts:
                px, py = round(pt[0] * sx), round(pt[1] * sy)
                if 0 <= px < crop_w and 0 <= py < crop_h:
                    cv2.circle(p5, (px, py), 3, (0, 0, 255), -1)

        # Draw half-width circles on p6: transparent fill, solid outline
        if midline_data is not None:
            xy_pts, hw_pts = midline_data
            sx = crop_w / w if w > 0 else 1.0
            sy = crop_h / h if h > 0 else 1.0
            # Semi-transparent filled circles on a separate layer
            fill_layer = p6.copy()
            for pt, hw in zip(xy_pts, hw_pts, strict=True):
                px, py = round(pt[0] * sx), round(pt[1] * sy)
                radius = max(1, round(float(hw) * max(sx, sy)))
                if 0 <= px < crop_w and 0 <= py < crop_h:
                    cv2.circle(fill_layer, (px, py), radius, (0, 200, 255), -1)
            cv2.addWeighted(fill_layer, 0.25, p6, 0.75, 0, dst=p6)
            # Solid outlines drawn on top
            for pt, hw in zip(xy_pts, hw_pts, strict=True):
                px, py = round(pt[0] * sx), round(pt[1] * sy)
                radius = max(1, round(float(hw) * max(sx, sy)))
                if 0 <= px < crop_w and 0 <= py < crop_h:
                    cv2.circle(p6, (px, py), radius, (0, 200, 255), 1)
                    cv2.circle(p6, (px, py), 2, (0, 0, 255), -1)

        # Rotate landscape panels to portrait before padding
        def _to_portrait(img: np.ndarray) -> np.ndarray:
            ih, iw = img.shape[:2]
            if iw > ih:
                return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return img

        combined = np.concatenate(
            [_pad_square(_to_portrait(p)) for p in [p1, p2, p3, p4, p5, p6]],
            axis=1,
        )

        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        ax.imshow(combined_rgb)
        ax.set_title(f"F{fi} {cam}", fontsize=7)
        ax.axis("off")

    for ax_idx in range(len(samples), n_rows * n_cols):
        axes_flat[ax_idx].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Midline extraction montage saved to %s", output_path)


def vis_skip_reason_pie(
    tracks_per_frame: list[list[FishTrack]],
    masks_per_frame: list[dict[str, list[tuple[np.ndarray, CropRegion]]]],
    output_path: Path,
) -> None:
    """Save a pie chart of midline skip reasons.

    Categorizes all (track, camera, mask) combinations by their skip reason.

    Args:
        tracks_per_frame: Per-frame confirmed track lists from Stage 3.
        masks_per_frame: Per-frame mask dicts from Stage 2.
        output_path: Output PNG path.
    """
    from aquapose.reconstruction.midline import (
        _adaptive_smooth,
        _check_skip_mask,
        _longest_path_bfs,
        _skeleton_and_widths,
    )

    counts: dict[str, int] = {}

    for fi, tracks in enumerate(tracks_per_frame):
        if fi >= len(masks_per_frame):
            break
        frame_masks = masks_per_frame[fi]
        for track in tracks:
            for cam, di in track.camera_detections.items():
                if cam not in frame_masks:
                    continue
                mask_list = frame_masks[cam]
                if di >= len(mask_list):
                    continue
                mask, crop_region = mask_list[di]

                skip = _check_skip_mask(mask, crop_region)
                if skip is not None:
                    # Categorize
                    if "too small" in skip:
                        key = "too small"
                    elif "boundary-clipped" in skip:
                        key = "boundary-clipped"
                    else:
                        key = skip
                    counts[key] = counts.get(key, 0) + 1
                    continue

                # Passed skip check — try skeleton
                smooth = _adaptive_smooth(mask)
                skel_bool, _dt = _skeleton_and_widths(smooth)
                n_skel = int(np.sum(skel_bool))
                if n_skel < 15:
                    counts["skeleton too short"] = (
                        counts.get("skeleton too short", 0) + 1
                    )
                    continue

                path_yx = _longest_path_bfs(skel_bool)
                if not path_yx:
                    counts["empty path"] = counts.get("empty path", 0) + 1
                    continue

                counts["valid"] = counts.get("valid", 0) + 1

    if not counts:
        logger.warning("vis_skip_reason_pie: no data")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list(counts.keys())
    values = list(counts.values())
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Midline Extraction Skip Reasons")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Skip reason pie chart saved to %s", output_path)
