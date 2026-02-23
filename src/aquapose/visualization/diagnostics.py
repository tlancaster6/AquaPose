"""Per-stage diagnostic visualizations for the AquaPose reconstruction pipeline.

Provides visualization functions covering all 5 pipeline stages:
detection grids, confidence histograms, mask montages, trajectory videos,
claiming overlays, midline extraction montages, skip-reason charts,
residual heatmaps, arc-length histograms, spline camera overlays, and
synthetic-mode GT comparison, camera overlay, error distribution, and report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from matplotlib import pyplot as plt

from aquapose.io.video import VideoSet
from aquapose.visualization.overlay import FISH_COLORS, draw_midline_overlay

if TYPE_CHECKING:
    from aquapose.calibration.projection import RefractiveProjectionModel
    from aquapose.reconstruction.triangulation import Midline3D, MidlineSet
    from aquapose.segmentation.crop import CropRegion
    from aquapose.segmentation.detector import Detection
    from aquapose.synthetic.fish import FishConfig
    from aquapose.tracking.tracker import FishTrack, TrackState

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Stage 1: Detection
# ---------------------------------------------------------------------------


def vis_detection_grid(
    detections_per_frame: list[dict[str, list[Detection]]],
    video_set: VideoSet,
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

    camera_ids = set(video_set.camera_ids)
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
            frame = video_set.read_frame(fi)[best_cam]
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
        for cam_dets in frame_dets.values()
        for det in cam_dets
    ]
    if not confidences:
        logger.warning("vis_confidence_histogram: no detections")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidences, bins=50, range=(0, 1), edgecolor="black", alpha=0.7)
    mean_conf = float(np.mean(confidences))
    ax.axvline(mean_conf, color="red", linestyle="--", label=f"Mean: {mean_conf:.3f}")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title(f"Detection Confidence Distribution (n={len(confidences)})")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confidence histogram saved to %s", output_path)


# ---------------------------------------------------------------------------
# Stage 2: Segmentation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage 3: Tracking
# ---------------------------------------------------------------------------


def vis_claiming_overlay(
    snapshots_per_frame: list[list[TrackSnapshot]],
    detections_per_frame: list[dict[str, list[Detection]]],
    video_set: VideoSet,
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

    available_cams = set(video_set.camera_ids)

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
    sample_frame = video_set.read_frame(0)[cameras[0]]
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
            all_frames = video_set.read_frame(fi)

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
    video_set: VideoSet,
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
    camera_ids = set(video_set.camera_ids)
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
            frame = video_set.read_frame(fi)[cam]
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


# ---------------------------------------------------------------------------
# Stage 5: Triangulation
# ---------------------------------------------------------------------------


def vis_residual_heatmap(
    midlines_3d_per_frame: list[dict[int, Midline3D]],
    output_path: Path,
) -> None:
    """Save a heatmap of mean reprojection residuals (fish x frame).

    Args:
        midlines_3d_per_frame: Per-frame triangulation results from Stage 5.
        output_path: Output PNG path.
    """
    if not midlines_3d_per_frame:
        logger.warning("vis_residual_heatmap: no frames")
        return

    # Collect all fish IDs
    all_fish_ids: set[int] = set()
    for frame_midlines in midlines_3d_per_frame:
        all_fish_ids.update(frame_midlines.keys())

    if not all_fish_ids:
        logger.warning("vis_residual_heatmap: no midlines")
        return

    fish_ids = sorted(all_fish_ids)
    fish_id_to_row = {fid: i for i, fid in enumerate(fish_ids)}
    n_fish = len(fish_ids)
    n_frames = len(midlines_3d_per_frame)

    data = np.full((n_fish, n_frames), np.nan)
    for fi, frame_midlines in enumerate(midlines_3d_per_frame):
        for fid, ml in frame_midlines.items():
            data[fish_id_to_row[fid], fi] = ml.mean_residual

    fig, ax = plt.subplots(figsize=(max(10, n_frames / 5), max(4, n_fish)))
    im = ax.imshow(data, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Fish ID")
    ax.set_yticks(range(n_fish))
    ax.set_yticklabels([str(fid) for fid in fish_ids])
    ax.set_title("Mean Reprojection Residual (pixels)")
    plt.colorbar(im, ax=ax, label="Residual (px)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Residual heatmap saved to %s", output_path)


def vis_arclength_histogram(
    midlines_3d_per_frame: list[dict[int, Midline3D]],
    output_path: Path,
) -> None:
    """Save a histogram of 3D midline arc lengths.

    Args:
        midlines_3d_per_frame: Per-frame triangulation results from Stage 5.
        output_path: Output PNG path.
    """
    arc_lengths = [
        ml.arc_length
        for frame_midlines in midlines_3d_per_frame
        for ml in frame_midlines.values()
    ]
    if not arc_lengths:
        logger.warning("vis_arclength_histogram: no midlines")
        return

    arr = np.array(arc_lengths)
    mean_val = float(arr.mean())
    median_val = float(np.median(arr))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.4f} m")
    ax.axvline(
        median_val, color="blue", linestyle=":", label=f"Median: {median_val:.4f} m"
    )
    ax.set_xlabel("Arc Length (m)")
    ax.set_ylabel("Count")
    ax.set_title(f"3D Midline Arc Length Distribution (n={len(arc_lengths)})")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Arc-length histogram saved to %s", output_path)


def vis_per_camera_spline_overlays(
    frame_index: int,
    frame_3d: dict[int, Midline3D],
    frame_2d: MidlineSet,
    models: dict[str, RefractiveProjectionModel],
    video_set: VideoSet,
    output_dir: Path,
) -> None:
    """Save per-camera overlay images with 3D spline reprojections and 2D midline dots.

    Generates one PNG per camera for the given frame showing:
      - 3D spline reprojections as colored polylines with width indicators
      - 2D midline sample points as colored dots with white outlines
      - Per-fish annotation: fish_id, residual, arc_length

    Comparing the 2D dots (from midline extraction) against the 3D lines
    (from reconstruction) reveals whether reconstruction or calibration is
    at fault when overlays don't match the fish.

    Args:
        frame_index: Frame index to render overlays for.
        frame_3d: 3D reconstruction results for the target frame.
        frame_2d: 2D midline sets for the target frame.
        models: Per-camera refractive projection models.
        video_set: Opened VideoSet providing undistorted frames.
        output_dir: Directory for per-camera PNG outputs.
    """
    import scipy.interpolate
    import torch

    if not frame_3d:
        logger.warning(
            "vis_per_camera_spline_overlays: no 3D midlines at frame %d", frame_index
        )
        return

    logger.info(
        "Per-camera overlays: frame %d, %d 3D midlines, %d 2D midline sets",
        frame_index,
        len(frame_3d),
        len(frame_2d),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    camera_ids = set(video_set.camera_ids)

    try:
        all_frames = video_set.read_frame(frame_index)
    except (IndexError, RuntimeError):
        logger.warning(
            "vis_per_camera_spline_overlays: could not read frame %d", frame_index
        )
        return

    n_saved = 0
    for cam_id in sorted(models):
        if cam_id not in camera_ids or cam_id not in all_frames:
            continue

        frame = all_frames[cam_id].copy()
        model = models[cam_id]

        # Draw 3D spline reprojections (thick colored lines)
        for fish_id, m3d in sorted(frame_3d.items()):
            color = FISH_COLORS[fish_id % len(FISH_COLORS)]
            draw_midline_overlay(
                frame,
                m3d,
                model,
                color=color,
                thickness=3,
                n_eval=40,
                draw_widths=True,
            )

            # Annotate near the spline head
            spline = scipy.interpolate.BSpline(
                m3d.knots.astype(np.float64),
                m3d.control_points.astype(np.float64),
                m3d.degree,
            )
            head_3d = spline(0.0).astype(np.float32)
            head_px, head_valid = model.project(torch.from_numpy(head_3d.reshape(1, 3)))
            if head_valid[0]:
                hx = round(float(head_px[0, 0]))
                hy = round(float(head_px[0, 1]))
                label = (
                    f"F{fish_id} res={m3d.mean_residual:.0f}px "
                    f"arc={m3d.arc_length:.3f}m"
                )
                _annotate_label(frame, label, (hx + 5, hy - 10), color)

        # Draw 2D midline dots (filled circles with white outline)
        for fish_id, cam_midlines in frame_2d.items():
            if cam_id not in cam_midlines:
                continue
            m2d = cam_midlines[cam_id]
            color = FISH_COLORS[fish_id % len(FISH_COLORS)]
            for pt in m2d.points:
                px, py = round(float(pt[0])), round(float(pt[1]))
                cv2.circle(frame, (px, py), 4, color, -1)
                cv2.circle(frame, (px, py), 4, (255, 255, 255), 1)

        # Camera / frame label
        cv2.putText(
            frame,
            f"Camera: {cam_id}  Frame: {frame_index}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Lines=3D spline  Dots=2D midline",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        out_path = output_dir / f"spline_overlay_{cam_id}.png"
        cv2.imwrite(str(out_path), frame)
        n_saved += 1

    logger.info("%d per-camera spline overlays saved to %s", n_saved, output_dir)


def _annotate_label(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw a text label with a dark background for readability.

    Args:
        frame: BGR image, modified in-place.
        text: Label string.
        position: (x, y) pixel position for text origin.
        color: BGR text color.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    x = max(0, min(x, frame.shape[1] - tw - 4))
    y = max(th + 4, min(y, frame.shape[0] - 4))
    cv2.rectangle(
        frame,
        (x - 2, y - th - 4),
        (x + tw + 2, y + baseline + 2),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Quantitative Markdown Report
# ---------------------------------------------------------------------------


def write_diagnostic_report(
    output_path: Path,
    stage_timing: dict[str, float],
    detections_per_frame: list[dict[str, list[Detection]]],
    masks_per_frame: list[dict[str, list[tuple[np.ndarray, CropRegion]]]],
    snapshots_per_frame: list[list[TrackSnapshot]],
    tracks_per_frame: list[list[FishTrack]],
    midlines_3d_per_frame: list[dict[int, Midline3D]],
    n_cameras: int,
) -> None:
    """Write a quantitative Markdown diagnostic report covering all 5 pipeline stages.

    Args:
        output_path: Path for the output .md file.
        stage_timing: Stage name to wall-clock seconds mapping.
        detections_per_frame: Per-frame detection dicts from Stage 1.
        masks_per_frame: Per-frame mask dicts from Stage 2.
        snapshots_per_frame: Per-frame lists of TrackSnapshot objects.
        tracks_per_frame: Per-frame confirmed track lists from Stage 3.
        midlines_3d_per_frame: Per-frame triangulation results from Stage 5.
        n_cameras: Number of cameras in the rig.
    """
    from datetime import UTC, datetime

    from aquapose.reconstruction.midline import (
        _adaptive_smooth,
        _check_skip_mask,
        _longest_path_bfs,
        _skeleton_and_widths,
    )
    from aquapose.tracking.tracker import TrackState

    lines: list[str] = []
    n_frames = len(detections_per_frame)

    # ------------------------------------------------------------------
    # Section 1: Header
    # ------------------------------------------------------------------
    lines.append("# AquaPose Diagnostic Report")
    lines.append("")
    lines.append(f"- **Date**: {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"- **Frames processed**: {n_frames}")
    lines.append(f"- **Cameras**: {n_cameras}")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 2: Stage Timing
    # ------------------------------------------------------------------
    lines.append("## Stage Timing")
    lines.append("")
    total_time = sum(stage_timing.values())
    lines.append("| Stage | Wall Time (s) | % of Total |")
    lines.append("|-------|--------------|-----------|")
    for stage_name, elapsed in stage_timing.items():
        pct = 100.0 * elapsed / total_time if total_time > 0 else 0.0
        lines.append(f"| {stage_name} | {elapsed:.2f} | {pct:.1f}% |")
    lines.append(f"| **TOTAL** | **{total_time:.2f}** | **100.0%** |")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 3: Detection (Stage 1)
    # ------------------------------------------------------------------
    lines.append("## Stage 1: Detection")
    lines.append("")

    # Per-camera stats
    cam_det_counts: dict[str, list[int]] = {}
    all_confidences: list[float] = []
    for frame_dets in detections_per_frame:
        for cam, dets in frame_dets.items():
            cam_det_counts.setdefault(cam, []).append(len(dets))
            for det in dets:
                all_confidences.append(det.confidence)

    lines.append("| Camera | Total Detections | Mean/Frame | Min/Frame | Max/Frame |")
    lines.append("|--------|-----------------|-----------|----------|----------|")
    total_dets = 0
    for cam in sorted(cam_det_counts):
        counts = cam_det_counts[cam]
        total = sum(counts)
        total_dets += total
        mean_c = float(np.mean(counts))
        min_c = min(counts)
        max_c = max(counts)
        lines.append(f"| {cam} | {total} | {mean_c:.1f} | {min_c} | {max_c} |")

    conf_arr = np.array(all_confidences) if all_confidences else np.array([0.0])
    lines.append("")
    lines.append(
        f"**Summary**: {total_dets} total detections, "
        f"mean confidence {float(conf_arr.mean()):.3f}, "
        f"std {float(conf_arr.std()):.3f}"
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Section 4: Segmentation (Stage 2)
    # ------------------------------------------------------------------
    lines.append("## Stage 2: Segmentation")
    lines.append("")

    total_masks = 0
    non_empty_masks = 0
    mask_areas: list[int] = []
    for frame_masks in masks_per_frame:
        for mask_list in frame_masks.values():
            for mask, _region in mask_list:
                total_masks += 1
                area = int(np.count_nonzero(mask))
                if area > 0:
                    non_empty_masks += 1
                    mask_areas.append(area)

    lines.append(f"- **Total masks**: {total_masks}")
    lines.append(f"- **Non-empty masks**: {non_empty_masks}")
    if mask_areas:
        area_arr = np.array(mask_areas)
        lines.append(f"- **Mean mask area**: {float(area_arr.mean()):.0f} px")
        lines.append(f"- **Median mask area**: {float(np.median(area_arr)):.0f} px")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 5: Tracking (Stage 3)
    # ------------------------------------------------------------------
    lines.append("## Stage 3: Tracking")
    lines.append("")

    # Aggregate per-fish state counts
    fish_state_counts: dict[int, dict[str, int]] = {}
    for frame_snaps in snapshots_per_frame:
        for snap in frame_snaps:
            fid = snap.fish_id
            if fid not in fish_state_counts:
                fish_state_counts[fid] = {
                    "confirmed": 0,
                    "probationary": 0,
                    "coasting": 0,
                }
            state_name = snap.state.value
            if state_name in fish_state_counts[fid]:
                fish_state_counts[fid][state_name] += 1

    if fish_state_counts:
        lines.append(
            "| Fish ID | Frames Seen | % Confirmed | % Probationary | % Coasting |"
        )
        lines.append(
            "|---------|------------|------------|---------------|-----------|"
        )
        for fid in sorted(fish_state_counts):
            sc = fish_state_counts[fid]
            total_seen = sc["confirmed"] + sc["probationary"] + sc["coasting"]
            if total_seen == 0:
                continue
            pct_c = 100.0 * sc["confirmed"] / total_seen
            pct_p = 100.0 * sc["probationary"] / total_seen
            pct_co = 100.0 * sc["coasting"] / total_seen
            lines.append(
                f"| {fid} | {total_seen} | {pct_c:.1f}% | {pct_p:.1f}% | {pct_co:.1f}% |"
            )
        lines.append("")

        # Max concurrent confirmed
        max_concurrent = 0
        for frame_snaps in snapshots_per_frame:
            n_conf = sum(1 for s in frame_snaps if s.state == TrackState.CONFIRMED)
            max_concurrent = max(max_concurrent, n_conf)

        lines.append(
            f"**Summary**: {len(fish_state_counts)} unique fish IDs, "
            f"max concurrent confirmed tracks: {max_concurrent}"
        )
    else:
        lines.append("No tracks recorded.")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 6: Midline Extraction (Stage 4)
    # ------------------------------------------------------------------
    lines.append("## Stage 4: Midline Extraction")
    lines.append("")

    skip_counts: dict[str, int] = {}
    total_pairs = 0
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
                total_pairs += 1

                skip = _check_skip_mask(mask, crop_region)
                if skip is not None:
                    if "too small" in skip:
                        key = "too small"
                    elif "boundary-clipped" in skip:
                        key = "boundary-clipped"
                    else:
                        key = skip
                    skip_counts[key] = skip_counts.get(key, 0) + 1
                    continue

                smooth = _adaptive_smooth(mask)
                skel_bool, _dt = _skeleton_and_widths(smooth)
                n_skel = int(np.sum(skel_bool))
                if n_skel < 15:
                    skip_counts["skeleton too short"] = (
                        skip_counts.get("skeleton too short", 0) + 1
                    )
                    continue

                path_yx = _longest_path_bfs(skel_bool)
                if not path_yx:
                    skip_counts["empty path"] = skip_counts.get("empty path", 0) + 1
                    continue

                skip_counts["valid"] = skip_counts.get("valid", 0) + 1

    n_valid = skip_counts.get("valid", 0)
    lines.append(f"- **Total (track, camera) pairs**: {total_pairs}")
    lines.append(f"- **Successful extractions**: {n_valid}")
    if total_pairs > 0:
        lines.append(f"- **Yield**: {100.0 * n_valid / total_pairs:.1f}%")
    lines.append("")

    if skip_counts:
        lines.append("| Reason | Count | % |")
        lines.append("|--------|-------|---|")
        for reason in sorted(skip_counts, key=lambda r: skip_counts[r], reverse=True):
            count = skip_counts[reason]
            pct = 100.0 * count / total_pairs if total_pairs > 0 else 0.0
            lines.append(f"| {reason} | {count} | {pct:.1f}% |")
        lines.append("")

    # ------------------------------------------------------------------
    # Section 7: Triangulation (Stage 5)
    # ------------------------------------------------------------------
    lines.append("## Stage 5: Triangulation")
    lines.append("")

    # Per-fish aggregated stats
    fish_tri_stats: dict[int, dict[str, list[float]]] = {}
    total_low_conf = 0
    total_midlines = 0
    for frame_midlines in midlines_3d_per_frame:
        for fid, ml in frame_midlines.items():
            total_midlines += 1
            if fid not in fish_tri_stats:
                fish_tri_stats[fid] = {
                    "mean_residuals": [],
                    "max_residuals": [],
                    "arc_lengths": [],
                    "low_conf": [],
                }
            fish_tri_stats[fid]["mean_residuals"].append(ml.mean_residual)
            fish_tri_stats[fid]["max_residuals"].append(ml.max_residual)
            fish_tri_stats[fid]["arc_lengths"].append(ml.arc_length)
            fish_tri_stats[fid]["low_conf"].append(float(ml.is_low_confidence))
            if ml.is_low_confidence:
                total_low_conf += 1

    if fish_tri_stats:
        lines.append(
            "| Fish ID | Frames with 3D | Mean Residual (px) | Max Residual (px) "
            "| Mean Arc Length (m) | % Low-Confidence |"
        )
        lines.append(
            "|---------|---------------|-------------------|------------------|"
            "--------------------|-----------------|"
        )
        for fid in sorted(fish_tri_stats):
            s = fish_tri_stats[fid]
            n_f = len(s["mean_residuals"])
            mean_res = float(np.mean(s["mean_residuals"]))
            max_res = float(np.max(s["max_residuals"]))
            mean_arc = float(np.mean(s["arc_lengths"]))
            pct_low = 100.0 * float(np.mean(s["low_conf"]))
            lines.append(
                f"| {fid} | {n_f} | {mean_res:.2f} | {max_res:.2f} "
                f"| {mean_arc:.4f} | {pct_low:.1f}% |"
            )
        lines.append("")

        # Per-camera residual breakdown (averaged over all fish and frames)
        cam_res_accum: dict[str, list[float]] = {}
        fish_cam_res: dict[int, dict[str, list[float]]] = {}
        for frame_midlines in midlines_3d_per_frame:
            for fid, ml in frame_midlines.items():
                if ml.per_camera_residuals:
                    if fid not in fish_cam_res:
                        fish_cam_res[fid] = {}
                    for cid, res in ml.per_camera_residuals.items():
                        cam_res_accum.setdefault(cid, []).append(res)
                        fish_cam_res[fid].setdefault(cid, []).append(res)

        if cam_res_accum:
            # Fish x Camera residual matrix
            all_cam_ids = sorted(cam_res_accum)
            all_fish_ids = sorted(fish_cam_res)
            lines.append("### Per-Camera Residual Breakdown (mean px)")
            lines.append("")
            header = "| Fish \\ Cam | " + " | ".join(all_cam_ids) + " |"
            sep = "|-----------|" + "|".join("---:" for _ in all_cam_ids) + "|"
            lines.append(header)
            lines.append(sep)
            for fid in all_fish_ids:
                cells = []
                for cid in all_cam_ids:
                    vals = fish_cam_res.get(fid, {}).get(cid, [])
                    if vals:
                        mean_val = float(np.mean(vals))
                        # Flag outliers with bold
                        if mean_val > 50.0:
                            cells.append(f"**{mean_val:.0f}**")
                        else:
                            cells.append(f"{mean_val:.1f}")
                    else:
                        cells.append("-")
                lines.append(f"| {fid} | " + " | ".join(cells) + " |")

            # Camera summary row
            cells = []
            for cid in all_cam_ids:
                vals = cam_res_accum[cid]
                cells.append(f"{float(np.mean(vals)):.1f}")
            lines.append("| **All** | " + " | ".join(cells) + " |")
            lines.append("")

        # Arc length distribution
        all_arc_for_dist = [
            ml.arc_length for fm in midlines_3d_per_frame for ml in fm.values()
        ]
        if all_arc_for_dist:
            arc_arr = np.array(all_arc_for_dist)
            lines.append("### Arc Length Distribution")
            lines.append("")
            lines.append(
                f"| Stat | Value (m) |\n|------|----------|\n"
                f"| Min | {float(arc_arr.min()):.4f} |\n"
                f"| P5 | {float(np.percentile(arc_arr, 5)):.4f} |\n"
                f"| P25 | {float(np.percentile(arc_arr, 25)):.4f} |\n"
                f"| Median | {float(np.median(arc_arr)):.4f} |\n"
                f"| P75 | {float(np.percentile(arc_arr, 75)):.4f} |\n"
                f"| P95 | {float(np.percentile(arc_arr, 95)):.4f} |\n"
                f"| Max | {float(arc_arr.max()):.4f} |"
            )
            lines.append("")

        # Per-fish temporal stability: arc length std over time
        # High std → identity swaps or reconstruction instability
        lines.append("### Temporal Stability (per-fish arc length)")
        lines.append("")
        lines.append(
            "| Fish ID | Mean Arc (m) | Std Arc (m) | CoV | Mean Residual Std (px) |"
        )
        lines.append(
            "|---------|-------------|------------|-----|----------------------|"
        )
        for fid in sorted(fish_tri_stats):
            s = fish_tri_stats[fid]
            arcs = np.array(s["arc_lengths"])
            resids = np.array(s["mean_residuals"])
            mean_a = float(arcs.mean())
            std_a = float(arcs.std())
            cov = std_a / mean_a if mean_a > 0 else 0.0
            std_r = float(resids.std())
            cov_str = f"**{cov:.2f}**" if cov > 0.3 else f"{cov:.2f}"
            lines.append(
                f"| {fid} | {mean_a:.4f} | {std_a:.4f} | {cov_str} | {std_r:.1f} |"
            )
        lines.append("")

        # Overall summary
        all_mean_res = [
            ml.mean_residual for fm in midlines_3d_per_frame for ml in fm.values()
        ]
        all_max_res = [
            ml.max_residual for fm in midlines_3d_per_frame for ml in fm.values()
        ]
        all_arc = [ml.arc_length for fm in midlines_3d_per_frame for ml in fm.values()]
        pct_low_total = (
            100.0 * total_low_conf / total_midlines if total_midlines > 0 else 0.0
        )
        lines.append(
            f"**Summary**: overall mean residual {float(np.mean(all_mean_res)):.2f} px, "
            f"max residual {float(np.max(all_max_res)):.2f} px, "
            f"mean arc length {float(np.mean(all_arc)):.4f} m, "
            f"low-confidence {pct_low_total:.1f}%"
        )
    else:
        lines.append("No 3D midlines produced.")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 8: Diagnostic Outputs
    # ------------------------------------------------------------------
    lines.append("## Diagnostic Outputs")
    lines.append("")

    diag_dir = output_path.parent
    generated_files = sorted(
        p
        for p in diag_dir.iterdir()
        if p.is_file() and p.suffix in (".png", ".mp4", ".gif") and p != output_path
    )
    if generated_files:
        for f in generated_files:
            lines.append(f"- `{f.name}`")
    else:
        lines.append("No visualization files found.")
    lines.append("")

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Diagnostic report written to %s", output_path)

    # Write per-frame CSV for time-series analysis
    csv_path = output_path.with_name("per_frame_metrics.csv")
    csv_lines = [
        "frame,fish_id,mean_residual_px,max_residual_px,arc_length_m,"
        "n_cameras,is_low_confidence,"
        + ",".join(
            f"res_{cid}"
            for cid in sorted(
                {
                    cid
                    for fm in midlines_3d_per_frame
                    for ml in fm.values()
                    if ml.per_camera_residuals
                    for cid in ml.per_camera_residuals
                }
            )
        )
    ]
    all_cam_ids_csv = sorted(
        {
            cid
            for fm in midlines_3d_per_frame
            for ml in fm.values()
            if ml.per_camera_residuals
            for cid in ml.per_camera_residuals
        }
    )
    # Rewrite header with known camera list
    csv_lines = [
        "frame,fish_id,mean_residual_px,max_residual_px,arc_length_m,"
        "n_cameras,is_low_confidence"
        + (
            "," + ",".join(f"res_{c}" for c in all_cam_ids_csv)
            if all_cam_ids_csv
            else ""
        )
    ]
    for fi, fm in enumerate(midlines_3d_per_frame):
        for fid, ml in sorted(fm.items()):
            cam_vals = []
            for cid in all_cam_ids_csv:
                if ml.per_camera_residuals and cid in ml.per_camera_residuals:
                    cam_vals.append(f"{ml.per_camera_residuals[cid]:.2f}")
                else:
                    cam_vals.append("")
            row = (
                f"{fi},{fid},{ml.mean_residual:.2f},{ml.max_residual:.2f},"
                f"{ml.arc_length:.6f},{ml.n_cameras},"
                f"{int(ml.is_low_confidence)}"
            )
            if cam_vals:
                row += "," + ",".join(cam_vals)
            csv_lines.append(row)
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")
    logger.info("Per-frame metrics CSV written to %s", csv_path)


# ---------------------------------------------------------------------------
# Synthetic Mode Diagnostics
# ---------------------------------------------------------------------------


def vis_synthetic_3d_comparison(
    midlines_3d: list[dict[int, Midline3D]],
    ground_truths: list[dict[int, Midline3D]],
    output_path: Path,
    *,
    n_eval: int = 30,
) -> None:
    """Save a 3D plot comparing GT and reconstructed midlines for all fish.

    Evaluates both ground truth and reconstructed B-splines at ``n_eval``
    points and overlays them in a single 3D axes.  Mean control-point error
    (in mm) is annotated for each fish.

    Args:
        midlines_3d: Reconstructed Midline3D dicts, one per frame.
        ground_truths: Ground truth Midline3D dicts, one per frame.
        output_path: Output PNG path.
        n_eval: Number of evaluation points along each spline.
    """
    import scipy.interpolate

    from aquapose.visualization.plot3d import _robust_bounds

    # Pick the frame with the most GT fish (prefer frame 0)
    best_frame_idx = 0
    for fi, gt_frame in enumerate(ground_truths):
        if len(gt_frame) >= len(ground_truths[best_frame_idx]):
            best_frame_idx = fi

    gt_frame = ground_truths[best_frame_idx]
    recon_frame = (
        midlines_3d[best_frame_idx] if best_frame_idx < len(midlines_3d) else {}
    )

    if not gt_frame:
        logger.warning("vis_synthetic_3d_comparison: no ground truth data")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    t_eval = np.linspace(0.0, 1.0, n_eval)
    all_pts: list[np.ndarray] = []

    for fish_id, gt_midline in sorted(gt_frame.items()):
        bgr = FISH_COLORS[fish_id % len(FISH_COLORS)]
        rgb = (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)

        # Evaluate GT spline
        gt_spline = scipy.interpolate.BSpline(
            gt_midline.knots.astype(np.float64),
            gt_midline.control_points.astype(np.float64),
            gt_midline.degree,
        )
        gt_pts = gt_spline(t_eval)  # (n_eval, 3)
        all_pts.append(gt_pts)

        ax.plot(
            gt_pts[:, 0],
            gt_pts[:, 1],
            gt_pts[:, 2],
            linestyle="--",
            color=rgb,
            linewidth=1.5,
            label=f"GT Fish {fish_id}",
        )

        if fish_id in recon_frame:
            recon_midline = recon_frame[fish_id]
            recon_spline = scipy.interpolate.BSpline(
                recon_midline.knots.astype(np.float64),
                recon_midline.control_points.astype(np.float64),
                recon_midline.degree,
            )
            recon_pts = recon_spline(t_eval)  # (n_eval, 3)
            all_pts.append(recon_pts)

            ax.plot(
                recon_pts[:, 0],
                recon_pts[:, 1],
                recon_pts[:, 2],
                linestyle="-",
                color=rgb,
                linewidth=2.0,
                label=f"Recon Fish {fish_id}",
            )

            # Compute mean control-point error in mm
            mean_err_mm = float(
                np.linalg.norm(
                    recon_midline.control_points - gt_midline.control_points,
                    axis=1,
                ).mean()
                * 1000.0
            )
            # Annotate at GT head position
            ax.text(
                gt_pts[0, 0],
                gt_pts[0, 1],
                gt_pts[0, 2],
                f"F{fish_id}: {mean_err_mm:.1f}mm",
                fontsize=8,
                color=rgb,
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")  # type: ignore[attr-defined]
    ax.set_title("GT vs Reconstructed 3D Midlines")
    ax.legend(fontsize=7)

    if all_pts:
        combined = np.concatenate(all_pts, axis=0)
        bounds = _robust_bounds(combined)
        if bounds is not None:
            centers, half_range = bounds
            ax.set_xlim(centers[0] - half_range, centers[0] + half_range)
            ax.set_ylim(centers[1] - half_range, centers[1] + half_range)
            ax.set_zlim(centers[2] - half_range, centers[2] + half_range)  # type: ignore[attr-defined]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Synthetic 3D comparison saved to %s", output_path)


def vis_synthetic_camera_overlays(
    midlines_3d: list[dict[int, Midline3D]],
    ground_truths: list[dict[int, Midline3D]],
    models: dict[str, RefractiveProjectionModel],
    output_dir: Path,
    *,
    canvas_size: tuple[int, int] = (720, 1280),
    n_eval: int = 40,
) -> None:
    """Save per-camera overlay images comparing GT and reconstructed midline projections.

    For each camera, generates a PNG showing ground truth midlines as dashed
    green polylines and reconstructed midlines as solid colored polylines.
    Per-fish reprojection residual (mean pixel distance) is annotated.

    Args:
        midlines_3d: Reconstructed Midline3D dicts, one per frame.
        ground_truths: Ground truth Midline3D dicts, one per frame.
        models: Per-camera refractive projection models.
        output_dir: Directory for per-camera PNG outputs.
        canvas_size: (height, width) of the output canvas.
        n_eval: Number of points to evaluate along each spline.
    """
    import scipy.interpolate
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pick best frame (most GT fish)
    best_frame_idx = 0
    for fi, gt_frame in enumerate(ground_truths):
        if len(gt_frame) >= len(ground_truths[best_frame_idx]):
            best_frame_idx = fi

    gt_frame = ground_truths[best_frame_idx]
    recon_frame = (
        midlines_3d[best_frame_idx] if best_frame_idx < len(midlines_3d) else {}
    )

    if not gt_frame:
        logger.warning("vis_synthetic_camera_overlays: no ground truth data")
        return

    t_eval = np.linspace(0.0, 1.0, n_eval)
    n_saved = 0

    for cam_id in sorted(models):
        model = models[cam_id]
        canvas = np.full((canvas_size[0], canvas_size[1], 3), 64, dtype=np.uint8)

        for fish_id, gt_midline in sorted(gt_frame.items()):
            bgr = FISH_COLORS[fish_id % len(FISH_COLORS)]

            # Evaluate and project GT spline
            gt_spline = scipy.interpolate.BSpline(
                gt_midline.knots.astype(np.float64),
                gt_midline.control_points.astype(np.float64),
                gt_midline.degree,
            )
            gt_pts_3d = gt_spline(t_eval).astype(np.float32)
            gt_px, gt_valid = model.project(torch.from_numpy(gt_pts_3d))
            gt_px_np = gt_px.numpy()
            gt_valid_np = gt_valid.numpy()

            # Draw GT as dashed green (alternate segments)
            gt_color = (0, 200, 0)
            gt_screen = [
                (round(float(gt_px_np[i, 0])), round(float(gt_px_np[i, 1])))
                for i in range(n_eval)
                if gt_valid_np[i]
            ]
            for seg_idx in range(0, len(gt_screen) - 1, 2):
                p0 = gt_screen[seg_idx]
                p1 = gt_screen[seg_idx + 1]
                cv2.line(canvas, p0, p1, gt_color, 1)

            # Evaluate and project reconstructed spline (if available)
            if fish_id in recon_frame:
                recon_midline = recon_frame[fish_id]
                recon_spline = scipy.interpolate.BSpline(
                    recon_midline.knots.astype(np.float64),
                    recon_midline.control_points.astype(np.float64),
                    recon_midline.degree,
                )
                recon_pts_3d = recon_spline(t_eval).astype(np.float32)
                recon_px, recon_valid = model.project(torch.from_numpy(recon_pts_3d))
                recon_px_np = recon_px.numpy()
                recon_valid_np = recon_valid.numpy()

                # Draw solid colored polyline for reconstructed
                recon_screen = np.array(
                    [
                        [
                            round(float(recon_px_np[i, 0])),
                            round(float(recon_px_np[i, 1])),
                        ]
                        for i in range(n_eval)
                        if recon_valid_np[i]
                    ],
                    dtype=np.int32,
                )
                if len(recon_screen) >= 2:
                    cv2.polylines(
                        canvas,
                        [recon_screen.reshape(-1, 1, 2)],
                        isClosed=False,
                        color=bgr,
                        thickness=2,
                    )

                # Compute reprojection residual (mean pixel distance between
                # GT projected and recon projected at matching valid points)
                both_valid = gt_valid_np & recon_valid_np
                if both_valid.any():
                    gt_valid_px = gt_px_np[both_valid]
                    recon_valid_px = recon_px_np[both_valid]
                    residual = float(
                        np.linalg.norm(gt_valid_px - recon_valid_px, axis=1).mean()
                    )
                    # Annotate near the first projected GT point
                    if len(gt_screen) > 0:
                        label = f"F{fish_id} res={residual:.1f}px"
                        _annotate_label(
                            canvas,
                            label,
                            (gt_screen[0][0] + 5, gt_screen[0][1] - 10),
                            bgr,
                        )

        # Camera label and legend
        cv2.putText(
            canvas,
            f"Camera: {cam_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Dashed=GT, Solid=Reconstructed",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        out_path = output_dir / f"synthetic_overlay_{cam_id}.png"
        cv2.imwrite(str(out_path), canvas)
        n_saved += 1

    logger.info("%d synthetic camera overlays saved to %s", n_saved, output_dir)


def vis_synthetic_error_distribution(
    midlines_3d: list[dict[int, Midline3D]],
    ground_truths: list[dict[int, Midline3D]],
    output_path: Path,
) -> None:
    """Save a 3-panel figure showing the distribution of 3D control-point errors.

    Panels: (a) histogram of all per-control-point errors in mm,
    (b) box plot grouped by fish ID, (c) scatter of error vs body position.

    Args:
        midlines_3d: Reconstructed Midline3D dicts, one per frame.
        ground_truths: Ground truth Midline3D dicts, one per frame.
        output_path: Output PNG path.
    """
    # Collect per-control-point errors
    all_errors_mm: list[float] = []
    fish_errors: dict[int, list[float]] = {}
    cp_errors: list[tuple[int, int, float]] = []  # (fish_id, cp_idx, error_mm)

    for _fi, (recon_frame, gt_frame) in enumerate(
        zip(midlines_3d, ground_truths, strict=False)
    ):
        for fish_id, gt_midline in gt_frame.items():
            if fish_id not in recon_frame:
                continue
            recon_midline = recon_frame[fish_id]
            errors = (
                np.linalg.norm(
                    recon_midline.control_points - gt_midline.control_points,
                    axis=1,
                )
                * 1000.0
            )  # mm, shape (7,)
            for cp_idx, err in enumerate(errors.tolist()):
                all_errors_mm.append(err)
                fish_errors.setdefault(fish_id, []).append(err)
                cp_errors.append((fish_id, cp_idx, err))

    if not all_errors_mm:
        logger.warning("vis_synthetic_error_distribution: no matching fish data")
        return

    all_errors_arr = np.array(all_errors_mm)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel (a): histogram
    ax = axes[0]
    ax.hist(all_errors_arr, bins=30, edgecolor="black", alpha=0.7)
    mean_err = float(all_errors_arr.mean())
    median_err = float(np.median(all_errors_arr))
    ax.axvline(mean_err, color="red", linestyle="--", label=f"Mean: {mean_err:.2f} mm")
    ax.axvline(
        median_err, color="blue", linestyle=":", label=f"Median: {median_err:.2f} mm"
    )
    ax.set_title("Control-Point Error Distribution")
    ax.set_xlabel("Error (mm)")
    ax.set_ylabel("Count")
    ax.legend()

    # Panel (b): box plot per fish
    ax = axes[1]
    fish_ids_sorted = sorted(fish_errors.keys())
    box_data = [fish_errors[fid] for fid in fish_ids_sorted]
    ax.boxplot(box_data, labels=[str(fid) for fid in fish_ids_sorted])
    ax.set_title("Per-Fish Error Distribution")
    ax.set_xlabel("Fish ID")
    ax.set_ylabel("Error (mm)")

    # Panel (c): scatter error vs control-point index
    ax = axes[2]
    for fish_id, cp_idx, err_mm in cp_errors:
        bgr = FISH_COLORS[fish_id % len(FISH_COLORS)]
        rgb = (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)
        ax.scatter(cp_idx, err_mm, color=rgb, alpha=0.5, s=20)
    ax.set_title("Error vs Body Position")
    ax.set_xlabel("Control Point Index (head->tail)")
    ax.set_ylabel("Error (mm)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Synthetic error distribution saved to %s", output_path)


def write_synthetic_report(
    output_path: Path,
    stage_timing: dict[str, float],
    midlines_3d: list[dict[int, Midline3D]],
    ground_truths: list[dict[int, Midline3D]],
    models: dict[str, RefractiveProjectionModel],
    fish_configs: list[FishConfig],
    method: str,
    diag_dir: Path,
) -> None:
    """Write a structured Markdown report for a synthetic pipeline run.

    Includes configuration summary, per-fish GT comparison table, per-camera
    mean reprojection residuals, error statistics, stage timing, and a list
    of all diagnostic files produced.

    Args:
        output_path: Path for the output .md file.
        stage_timing: Stage name to wall-clock seconds mapping.
        midlines_3d: Reconstructed Midline3D dicts, one per frame.
        ground_truths: Ground truth Midline3D dicts, one per frame.
        models: Per-camera refractive projection models.
        fish_configs: List of FishConfig objects used to generate synthetic fish.
        method: Reconstruction method name (``"triangulation"`` or ``"curve"``).
        diag_dir: Directory containing diagnostic output files.
    """
    from datetime import UTC, datetime

    lines: list[str] = []
    n_frames = len(midlines_3d)
    n_fish = len(fish_configs)
    n_cameras = len(models)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    lines.append("# Synthetic Diagnostic Report")
    lines.append("")
    lines.append(f"- **Date**: {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"- **Method**: {method}")
    lines.append("")

    # ------------------------------------------------------------------
    # Config Summary
    # ------------------------------------------------------------------
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| n_fish | {n_fish} |")
    lines.append(f"| n_cameras | {n_cameras} |")
    lines.append(f"| n_frames | {n_frames} |")
    lines.append(f"| method | {method} |")
    lines.append("")

    lines.append("### Fish Configurations")
    lines.append("")
    lines.append("| Fish | Position | Heading (rad) | Curvature (m⁻¹) | Scale (m) |")
    lines.append("|------|----------|---------------|-----------------|-----------|")
    for fi, fc in enumerate(fish_configs):
        pos_str = f"({fc.position[0]:.3f}, {fc.position[1]:.3f}, {fc.position[2]:.3f})"
        lines.append(
            f"| {fi} | {pos_str} | {fc.heading_rad:.3f} | {fc.curvature:.3f} | {fc.scale:.4f} |"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Per-Fish GT Comparison
    # ------------------------------------------------------------------
    lines.append("## Per-Fish GT Comparison")
    lines.append("")

    # Collect per-fish stats across all frames
    fish_ctrl_errors: dict[int, list[float]] = {}
    fish_arc_errors: dict[int, list[float]] = {}

    for _fi, (recon_frame, gt_frame) in enumerate(
        zip(midlines_3d, ground_truths, strict=False)
    ):
        for fish_id, gt_midline in gt_frame.items():
            if fish_id not in recon_frame:
                continue
            recon_midline = recon_frame[fish_id]
            cp_errors_m = np.linalg.norm(
                recon_midline.control_points - gt_midline.control_points, axis=1
            )
            for err in cp_errors_m.tolist():
                fish_ctrl_errors.setdefault(fish_id, []).append(err * 1000.0)
            arc_err_mm = abs(recon_midline.arc_length - gt_midline.arc_length) * 1000.0
            fish_arc_errors.setdefault(fish_id, []).append(arc_err_mm)

    # Per-camera residuals from reconstructed midlines
    fish_cam_residuals: dict[int, list[float]] = {}
    for recon_frame in midlines_3d:
        for fish_id, ml in recon_frame.items():
            if ml.per_camera_residuals:
                for res in ml.per_camera_residuals.values():
                    fish_cam_residuals.setdefault(fish_id, []).append(res)

    if fish_ctrl_errors:
        lines.append(
            "| Fish ID | Mean Error (mm) | Max Error (mm) | Std Error (mm) "
            "| Arc Length Error (mm) | Mean Residual (px) |"
        )
        lines.append(
            "|---------|-----------------|----------------|----------------|"
            "-----------------------|--------------------|"
        )
        for fish_id in sorted(fish_ctrl_errors):
            errs = np.array(fish_ctrl_errors[fish_id])
            arc_errs = np.array(fish_arc_errors.get(fish_id, [0.0]))
            cam_res = fish_cam_residuals.get(fish_id, [])
            mean_res = float(np.mean(cam_res)) if cam_res else float("nan")
            lines.append(
                f"| {fish_id} | {float(errs.mean()):.2f} | {float(errs.max()):.2f} "
                f"| {float(errs.std()):.2f} | {float(arc_errs.mean()):.2f} "
                f"| {mean_res:.2f} |"
            )
    else:
        lines.append("No matched fish found between reconstruction and ground truth.")
    lines.append("")

    # ------------------------------------------------------------------
    # Per-Camera Mean Reprojection Residual
    # ------------------------------------------------------------------
    cam_res_accum: dict[str, list[float]] = {}
    for recon_frame in midlines_3d:
        for ml in recon_frame.values():
            if ml.per_camera_residuals:
                for cam_id, res in ml.per_camera_residuals.items():
                    cam_res_accum.setdefault(cam_id, []).append(res)

    if cam_res_accum:
        lines.append("## Per-Camera Mean Reprojection Residual")
        lines.append("")
        lines.append("| Camera ID | Mean Residual (px) |")
        lines.append("|-----------|-------------------|")
        for cam_id in sorted(cam_res_accum):
            mean_res = float(np.mean(cam_res_accum[cam_id]))
            lines.append(f"| {cam_id} | {mean_res:.2f} |")
        lines.append("")

    # ------------------------------------------------------------------
    # Error Statistics
    # ------------------------------------------------------------------
    lines.append("## Error Statistics")
    lines.append("")

    all_errs_mm: list[float] = [e for errs in fish_ctrl_errors.values() for e in errs]
    if all_errs_mm:
        arr = np.array(all_errs_mm)
        lines.append("| Percentile | Error (mm) |")
        lines.append("|------------|-----------|")
        for pct, label in [
            (5, "p5"),
            (25, "p25"),
            (50, "p50"),
            (75, "p75"),
            (95, "p95"),
        ]:
            lines.append(f"| {label} | {float(np.percentile(arr, pct)):.2f} |")
        lines.append(f"| max | {float(arr.max()):.2f} |")
    else:
        lines.append("No error data available.")
    lines.append("")

    # ------------------------------------------------------------------
    # Stage Timing
    # ------------------------------------------------------------------
    lines.append("## Stage Timing")
    lines.append("")
    total_time = sum(stage_timing.values())
    lines.append("| Stage | Wall Time (s) | % of Total |")
    lines.append("|-------|--------------|-----------|")
    for stage_name, elapsed in stage_timing.items():
        pct = 100.0 * elapsed / total_time if total_time > 0 else 0.0
        lines.append(f"| {stage_name} | {elapsed:.2f} | {pct:.1f}% |")
    lines.append(f"| **TOTAL** | **{total_time:.2f}** | **100.0%** |")
    lines.append("")

    # ------------------------------------------------------------------
    # Diagnostic Files
    # ------------------------------------------------------------------
    lines.append("## Diagnostic Files")
    lines.append("")
    generated_files: list[Path] = []
    if diag_dir.exists():
        for p in sorted(diag_dir.iterdir()):
            if p.is_file() and p.suffix in (".png", ".mp4", ".gif", ".md"):
                generated_files.append(p)
            elif p.is_dir():
                # Include subdirectory contents
                for sub_p in sorted(p.iterdir()):
                    if sub_p.is_file() and sub_p.suffix in (".png", ".mp4", ".gif"):
                        generated_files.append(sub_p)

    if generated_files:
        for f in generated_files:
            rel = f.relative_to(diag_dir) if f.is_relative_to(diag_dir) else f
            lines.append(f"- `{rel}`")
    else:
        lines.append("No visualization files found.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Synthetic diagnostic report written to %s", output_path)
