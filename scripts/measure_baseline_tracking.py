#!/usr/bin/env python3
"""Standalone baseline tracking measurement script for AquaPose.

Runs OC-SORT on a configurable camera/frame range, computes tracking and
fragmentation metrics, produces an annotated video with track IDs, and saves
results as JSON. Designed to establish the quantitative baseline for Phase 84
comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Palette (BGR) — same as investigate_occlusion.py
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Config loading (same pattern as investigate_occlusion.py)
# ---------------------------------------------------------------------------


def _load_config(config_path: str) -> dict:
    """Load project config YAML and resolve paths relative to project_dir.

    Args:
        config_path: Path to the project config YAML file.

    Returns:
        Parsed config dict with resolved paths.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    project_dir = Path(cfg["project_dir"]).expanduser()

    # Resolve video_dir
    vd = Path(cfg["video_dir"])
    if not vd.is_absolute():
        vd = project_dir / vd
    cfg["_video_dir"] = vd

    # Resolve detection weights
    dw = Path(cfg["detection"]["weights_path"])
    if not dw.is_absolute():
        dw = project_dir / dw
    cfg["_obb_weights"] = str(dw)

    return cfg


def _find_video(video_dir: Path, camera_id: str) -> Path:
    """Find the video file for a given camera ID.

    Args:
        video_dir: Directory containing video files.
        camera_id: Camera identifier (e.g., 'e3v83eb').

    Returns:
        Path to the matching video file.

    Raises:
        FileNotFoundError: If no video matches the camera ID.
    """
    candidates = list(video_dir.glob(f"{camera_id}*"))
    if not candidates:
        raise FileNotFoundError(
            f"No video found for camera '{camera_id}' in {video_dir}"
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# OBB detection helpers
# ---------------------------------------------------------------------------


def _parse_obb_results(results: list) -> list[dict]:
    """Parse ultralytics OBB results into detection dicts.

    Args:
        results: Ultralytics Results list from model.predict().

    Returns:
        List of detection dicts with bbox, confidence, angle, obb_points, area.
    """
    detections = []
    if not results:
        return detections

    r = results[0]
    if r.obb is None:
        return detections

    xywhr = r.obb.xywhr.cpu().numpy()
    corners_all = r.obb.xyxyxyxy.cpu().numpy()
    confs = r.obb.conf.cpu().numpy()

    for i in range(len(xywhr)):
        _cx, _cy, w, h, angle_cw_rad = xywhr[i]
        corners = corners_all[i]  # (4, 2)

        x_min = int(corners[:, 0].min())
        y_min = int(corners[:, 1].min())
        x_max = int(corners[:, 0].max())
        y_max = int(corners[:, 1].max())

        detections.append(
            {
                "bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
                "confidence": float(confs[i]),
                "angle": -float(angle_cw_rad),
                "obb_points": corners.copy(),
                "area": int(w * h),
            }
        )
    return detections


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _draw_obb(
    img: np.ndarray,
    corners: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw an oriented bounding box on an image.

    Args:
        img: Image to draw on (modified in-place).
        corners: OBB corners, shape (4, 2).
        color: BGR color tuple.
        thickness: Line thickness.
    """
    pts = corners.astype(np.int32)
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % 4]), color, thickness)


def _obb_center(corners: np.ndarray) -> tuple[int, int]:
    """Compute the center pixel of an OBB.

    Args:
        corners: OBB corners, shape (4, 2).

    Returns:
        (x, y) integer center coordinate.
    """
    cx = int(corners[:, 0].mean())
    cy = int(corners[:, 1].mean())
    return cx, cy


# ---------------------------------------------------------------------------
# Track assignment: map detection index -> track_id for this frame
# ---------------------------------------------------------------------------


def _get_frame_track_assignments(
    tracker: object, dets: list[dict], frame_idx: int
) -> dict[int, int | None]:
    """Map detection indices to track IDs using tracker's internal builder state.

    Args:
        tracker: OcSortTracker instance.
        dets: List of detection dicts for this frame.
        frame_idx: Current frame index.

    Returns:
        Dict mapping detection index to track_id (or None if untracked).
    """
    assignments: dict[int, int | None] = {i: None for i in range(len(dets))}
    builders = getattr(tracker, "_builders", {})

    for local_id, builder in builders.items():
        if not builder.frames or builder.frames[-1] != frame_idx:
            continue
        if builder.frame_status[-1] != "detected":
            continue

        trk_bbox = builder.bboxes[-1]  # (x1, y1, w, h)
        tx, ty, tw, th = trk_bbox
        best_iou = 0.3
        best_di = -1

        for di, det in enumerate(dets):
            if assignments[di] is not None:
                continue
            dx, dy, dw, dh = det["bbox"]
            iou = _bbox_iou_xywh((tx, ty, tw, th), (dx, dy, dw, dh))
            if iou > best_iou:
                best_iou = iou
                best_di = di

        if best_di >= 0:
            assignments[best_di] = local_id

    return assignments


def _bbox_iou_xywh(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two (x, y, w, h) bounding boxes.

    Args:
        a: First bbox as (x, y, w, h).
        b: Second bbox as (x, y, w, h).

    Returns:
        Intersection over union value.
    """
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_baseline(
    cfg: dict,
    camera_id: str,
    start_frame: int,
    end_frame: int,
    n_animals: int,
    output_dir: Path,
) -> None:
    """Run OC-SORT on the specified frame range, compute metrics, save outputs.

    Args:
        cfg: Parsed project config with resolved paths.
        camera_id: Camera ID to analyze.
        start_frame: First frame index (inclusive).
        end_frame: Last frame index (exclusive).
        n_animals: Expected number of fish (used for fragmentation analysis).
        output_dir: Directory to write video and metrics JSON.
    """
    from ultralytics import YOLO

    from aquapose.core.detection.backends.yolo_obb import polygon_nms
    from aquapose.core.tracking.ocsort_wrapper import OcSortTracker
    from aquapose.core.types.detection import Detection
    from aquapose.evaluation.stages.fragmentation import evaluate_fragmentation_2d
    from aquapose.evaluation.stages.tracking import evaluate_tracking

    video_path = _find_video(cfg["_video_dir"], camera_id)
    obb_model = YOLO(cfg["_obb_weights"])

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if end_frame <= 0 or end_frame > total_frames:
        end_frame = total_frames

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_frames = end_frame - start_frame
    print(f"Video: {video_path} ({total_frames} frames)")
    print(f"Frame range: {start_frame}-{end_frame} ({n_frames} frames)")

    # Initialize tracker with min_hits=1 for honest baseline (no warm-up penalty)
    tracker = OcSortTracker(camera_id=camera_id, min_hits=1, det_thresh=0.1)

    # --- Pass 1: Detection + Tracking ---
    frame_data: list[dict] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for fidx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        # OBB detection with polygon NMS (matches production pipeline)
        obb_results = obb_model.predict(frame, conf=0.1, iou=0.95, verbose=False)
        dets = _parse_obb_results(obb_results)

        det_objects = [
            Detection(
                bbox=d["bbox"],
                mask=None,
                area=d["area"],
                confidence=d["confidence"],
                angle=d["angle"],
                obb_points=d["obb_points"],
            )
            for d in dets
        ]
        det_objects = polygon_nms(det_objects, iou_threshold=0.45)

        # Rebuild dets to match filtered det_objects
        dets = [
            {
                "bbox": d.bbox,
                "confidence": d.confidence,
                "angle": d.angle,
                "obb_points": d.obb_points,
                "area": d.area,
            }
            for d in det_objects
        ]

        # Feed to tracker using ABSOLUTE frame index
        tracker.update(fidx, det_objects)

        track_assignments = _get_frame_track_assignments(tracker, dets, fidx)

        frame_data.append(
            {
                "frame_idx": fidx,
                "detections": dets,
                "track_assignments": track_assignments,
                "frame_image": frame,
            }
        )

        if (fidx - start_frame) % 100 == 0:
            print(f"  Processed frame {fidx} ({len(dets)} detections)")

    cap.release()
    print(f"Pass 1 complete: {len(frame_data)} frames processed")

    # --- Compute metrics ---
    tracklets = tracker.get_tracklets()
    print(f"Tracklets: {len(tracklets)} confirmed tracks")

    tracking_metrics = evaluate_tracking(tracklets)
    frag_metrics = evaluate_fragmentation_2d(tracklets, n_animals)

    # --- Print metrics summary ---
    _print_summary(
        tracking_metrics, frag_metrics, camera_id, start_frame, end_frame, n_animals
    )

    # --- Save metrics JSON ---
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = {
        "config": {
            "camera": camera_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "n_frames": n_frames,
            "n_animals": n_animals,
            "min_hits": 1,
            "det_thresh": 0.1,
            "conf_threshold": 0.1,
            "nms_threshold": 0.45,
        },
        "tracking": tracking_metrics.to_dict(),
        "fragmentation": frag_metrics.to_dict(),
    }
    metrics_path = output_dir / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # --- Pass 2: Render annotated video ---
    output_video = output_dir / "baseline_tracking.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (frame_w, frame_h))

    for fd in frame_data:
        frame_img = fd["frame_image"].copy()

        for di, det in enumerate(fd["detections"]):
            track_id = fd["track_assignments"].get(di)

            if track_id is not None:
                color = _PALETTE_BGR[track_id % len(_PALETTE_BGR)]
            else:
                color = (128, 128, 128)  # gray for untracked

            if det["obb_points"] is not None:
                _draw_obb(frame_img, det["obb_points"], color, thickness=2)
                cx, cy = _obb_center(det["obb_points"])

                if track_id is not None:
                    label = str(track_id)
                    cv2.putText(
                        frame_img,
                        label,
                        (cx - 8, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

        # Frame number annotation (top-left, small)
        cv2.putText(
            frame_img,
            f"F{fd['frame_idx']}",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        writer.write(frame_img)

    writer.release()
    print(f"Video saved: {output_video}")


def _print_summary(
    tracking_metrics: object,
    frag_metrics: object,
    camera_id: str,
    start_frame: int,
    end_frame: int,
    n_animals: int,
) -> None:
    """Print a human-readable metrics summary to stdout.

    Args:
        tracking_metrics: TrackingMetrics instance.
        frag_metrics: FragmentationMetrics instance.
        camera_id: Camera ID used.
        start_frame: First frame index.
        end_frame: Last frame index.
        n_animals: Expected fish count.
    """
    n_frames = end_frame - start_frame
    tm = tracking_metrics
    fm = frag_metrics

    print()
    print("=== OC-SORT Baseline Tracking Metrics ===")
    print(
        f"Camera: {camera_id} | Frames: {start_frame}-{end_frame} ({n_frames} frames)"
    )
    print()
    print(f"Track count:        {tm.track_count}")
    print(f"Length min:          {tm.length_min} frames")
    print(f"Length max:          {tm.length_max} frames")
    print(f"Length median:       {tm.length_median} frames")
    print(f"Length std:          {_tracklet_std(tm):.1f} frames")
    print(f"Coast frequency:    {tm.coast_frequency:.3f}")
    print(f"Detection coverage: {tm.detection_coverage:.3f}")
    print()
    print("Fragmentation:")
    print(f"  Total gaps:       {fm.total_gaps}")
    mean_gap = fm.mean_gap_duration
    print(f"  Mean gap length:  {mean_gap:.1f}")
    print(f"  Continuity ratio: {fm.mean_continuity_ratio:.3f}")
    print(f"  Births:           {fm.track_births}")
    print(f"  Deaths:           {fm.track_deaths}")
    print()
    print(f"Target: {n_animals} tracks, zero fragmentation")

    # Gap-to-target analysis
    track_delta = tm.track_count - n_animals
    track_sign = "+" if track_delta >= 0 else ""
    analysis_parts = [
        f"track count delta: {track_sign}{track_delta} vs target {n_animals}",
        f"gaps: {fm.total_gaps}",
        f"continuity: {fm.mean_continuity_ratio:.3f}",
    ]
    print(f"Gap to target: {'; '.join(analysis_parts)}")


def _tracklet_std(tracking_metrics: object) -> float:
    """Compute a placeholder std from min/max/median (exact not stored).

    This approximates std as (max - min) / 4 when full lengths are unavailable.
    For the actual value, use the raw tracklets.

    Args:
        tracking_metrics: TrackingMetrics instance.

    Returns:
        Approximate standard deviation.
    """
    # TrackingMetrics does not store per-track lengths; use range/4 approximation
    tm = tracking_metrics
    return (tm.length_max - tm.length_min) / 4.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for baseline tracking measurement."""
    parser = argparse.ArgumentParser(
        description=(
            "Run OC-SORT on a camera clip, compute tracking + fragmentation metrics, "
            "produce an annotated video with track IDs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to project config YAML (e.g., ~/aquapose/projects/YH/config.yaml)",
    )
    parser.add_argument(
        "--camera",
        default="e3v83eb",
        help="Camera ID to analyze (default: e3v83eb)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=3300,
        help="First frame index, inclusive (default: 3300)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=4500,
        help="Last frame index, exclusive (default: 4500)",
    )
    parser.add_argument(
        "--n-animals",
        type=int,
        default=9,
        help="Expected number of fish (default: 9)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./baseline_tracking_output",
        help="Directory to save video and metrics JSON (default: ./baseline_tracking_output)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    cfg = _load_config(args.config)

    run_baseline(
        cfg=cfg,
        camera_id=args.camera,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        n_animals=args.n_animals,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
