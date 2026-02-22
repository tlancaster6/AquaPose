"""Per-stage diagnostic visualizations for the AquaPose reconstruction pipeline.

Provides 10 visualization functions covering all 5 pipeline stages:
detection grids, confidence histograms, mask montages, trajectory videos,
claiming overlays, midline extraction montages, skip-reason charts,
residual heatmaps, arc-length histograms, and spline camera overlays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from matplotlib import pyplot as plt

from aquapose.visualization.overlay import FISH_COLORS, draw_midline_overlay

if TYPE_CHECKING:
    from aquapose.calibration.projection import RefractiveProjectionModel
    from aquapose.reconstruction.triangulation import Midline3D
    from aquapose.segmentation.crop import CropRegion
    from aquapose.segmentation.detector import Detection
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


def _read_video_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    """Read a single frame from a video file by index.

    Args:
        video_path: Path to the video file.
        frame_idx: Zero-based frame index to read.

    Returns:
        BGR frame as uint8 ndarray, or None if the frame could not be read.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            return frame
        return None
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Stage 1: Detection
# ---------------------------------------------------------------------------


def vis_detection_grid(
    detections_per_frame: list[dict[str, list[Detection]]],
    video_paths: dict[str, Path],
    output_path: Path,
) -> None:
    """Save a 3x3 grid of frames with detection bounding boxes.

    Selects 9 evenly spaced frame indices and for each picks the camera with
    the most detections. Draws bounding boxes with confidence labels.

    Args:
        detections_per_frame: Per-frame detection dicts from Stage 1.
        video_paths: Camera ID to video path mapping.
        output_path: Output PNG path.
    """
    n_frames = len(detections_per_frame)
    if n_frames == 0:
        logger.warning("vis_detection_grid: no frames")
        return

    frame_indices = np.linspace(0, n_frames - 1, 9, dtype=int)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes_flat = axes.flatten()

    for ax_idx, fi in enumerate(frame_indices):
        ax = axes_flat[ax_idx]
        frame_dets = detections_per_frame[fi]

        # Pick camera with most detections
        best_cam = max(frame_dets, key=lambda c: len(frame_dets[c]), default=None)
        if best_cam is None or best_cam not in video_paths:
            ax.set_title(f"Frame {fi}: no data")
            ax.axis("off")
            continue

        frame = _read_video_frame(video_paths[best_cam], fi)
        if frame is None:
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
    video_paths: dict[str, Path],
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
        video_paths: Camera ID to video path mapping.
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

    # Select top cameras by detection count (up to 6 for 2x3 grid)
    if cameras is None:
        cam_counts: dict[str, int] = {}
        for frame_dets in detections_per_frame:
            for cam, dets in frame_dets.items():
                cam_counts[cam] = cam_counts.get(cam, 0) + len(dets)
        sorted_cams = sorted(cam_counts, key=lambda c: cam_counts[c], reverse=True)
        cameras = sorted_cams[:6]

    cameras = [c for c in cameras if c in video_paths]
    if not cameras:
        logger.warning("vis_claiming_overlay: no valid cameras")
        return

    # Open video captures
    captures: dict[str, cv2.VideoCapture] = {}
    try:
        for cam in cameras:
            cap = cv2.VideoCapture(str(video_paths[cam]))
            if not cap.isOpened():
                logger.warning("Cannot open video for camera %s", cam)
                continue
            captures[cam] = cap

        if not captures:
            return

        # Determine tile size from first frame (2x3 grid: 2 rows, 3 cols)
        n_cols = 3
        n_rows = 2
        sample_cap = next(iter(captures.values()))
        tile_w = int(sample_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // n_cols
        tile_h = int(sample_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // n_rows

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

                # Build claim lookup: cam -> det_idx -> fish_id
                claim_map: dict[str, dict[int, int]] = {}
                for snap in frame_snaps:
                    for cam, di in snap.camera_detections.items():
                        claim_map.setdefault(cam, {})[di] = snap.fish_id

                for cam_idx, cam in enumerate(cameras[: n_cols * n_rows]):
                    # Read frame
                    if cam not in captures:
                        continue
                    ret, frame = captures[cam].read()
                    if not ret:
                        continue

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
                                snap.position, dtype=torch.float32
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
    finally:
        for cap in captures.values():
            cap.release()

    logger.info("Claiming overlay video saved to %s", output_path)


# ---------------------------------------------------------------------------
# Stage 4: Midline Extraction
# ---------------------------------------------------------------------------


def vis_midline_extraction_montage(
    tracks_per_frame: list[list[FishTrack]],
    masks_per_frame: list[dict[str, list[tuple[np.ndarray, CropRegion]]]],
    detections_per_frame: list[dict[str, list[Detection]]],
    video_paths: dict[str, Path],
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
        video_paths: Camera ID to video path mapping.
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
    valid_tuples: list[tuple[int, str, int, np.ndarray, CropRegion]] = []
    for fi, tracks in enumerate(tracks_per_frame):
        if fi >= len(masks_per_frame):
            break
        frame_masks = masks_per_frame[fi]
        for track in tracks:
            for cam, di in track.camera_detections.items():
                if cam not in frame_masks or cam not in video_paths:
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

        # Read the raw frame and extract the crop
        frame = _read_video_frame(video_paths[cam], fi)
        if frame is None:
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


def vis_spline_camera_overlay(
    midlines_3d_per_frame: list[dict[int, Midline3D]],
    models: dict[str, RefractiveProjectionModel],
    video_paths: dict[str, Path],
    output_path: Path,
) -> None:
    """Save a cropped camera frame with all 3D splines reprojected onto it.

    Finds the frame with the most fish, picks the camera where most splines
    project within bounds, draws overlays, and crops to a tight bounding box.

    Args:
        midlines_3d_per_frame: Per-frame triangulation results from Stage 5.
        models: Per-camera refractive projection models.
        video_paths: Camera ID to video path mapping.
        output_path: Output PNG path.
    """
    import scipy.interpolate
    import torch

    if not midlines_3d_per_frame:
        logger.warning("vis_spline_camera_overlay: no frames")
        return

    # Find frame with most fish (limit to first 9)
    best_fi = 0
    best_count = 0
    for fi, frame_midlines in enumerate(midlines_3d_per_frame):
        count = min(len(frame_midlines), 9)
        if count > best_count:
            best_count = count
            best_fi = fi

    if best_count == 0:
        logger.warning("vis_spline_camera_overlay: no midlines in any frame")
        return

    frame_midlines = midlines_3d_per_frame[best_fi]

    # For each camera, count how many splines project within image bounds
    best_cam = None
    best_cam_count = 0

    for cam_id in models:
        if cam_id not in video_paths:
            continue
        model = models[cam_id]
        in_bounds = 0

        for ml in frame_midlines.values():
            spline = scipy.interpolate.BSpline(
                ml.knots.astype(np.float64),
                ml.control_points.astype(np.float64),
                ml.degree,
            )
            pts_3d = spline(np.linspace(0, 1, 15)).astype(np.float32)
            pixels, valid = model.project(torch.from_numpy(pts_3d))
            pixels_np = pixels.detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy()

            # Check if majority of points are within reasonable bounds
            valid_pts = pixels_np[valid_np]
            if len(valid_pts) >= 5:
                in_bounds += 1

        if in_bounds > best_cam_count:
            best_cam_count = in_bounds
            best_cam = cam_id

    if best_cam is None:
        logger.warning("vis_spline_camera_overlay: no camera with valid projections")
        return

    # Read the frame
    frame = _read_video_frame(video_paths[best_cam], best_fi)
    if frame is None:
        logger.warning("vis_spline_camera_overlay: could not read frame")
        return

    # Draw all midline overlays on the full frame
    for ml in frame_midlines.values():
        draw_midline_overlay(frame, ml, models[best_cam])

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100, h / 100))
    ax.imshow(frame_rgb)
    ax.set_title(f"Spline Overlay — Frame {best_fi}, Camera {best_cam}")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Spline camera overlay saved to %s", output_path)


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
