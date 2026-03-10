#!/usr/bin/env python3
"""Standalone occlusion investigation script for AquaPose.

Runs OBB detection, OC-SORT tracking, and multi-instance pose estimation on a
configurable camera/frame range, producing an annotated crop video, per-frame
statistics JSON, and optional confidence sweep table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Palette (reused from evaluation/viz/overlay.py)
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

# Keypoint skeleton: nose(0)->head(1)->spine1(2)->spine2(3)->spine3(4)->tail(5)
_SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
_KP_NAMES = ["nose", "head", "spine1", "spine2", "spine3", "tail"]


# ---------------------------------------------------------------------------
# Config loading
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

    # Resolve pose weights
    pw = Path(cfg["midline"]["weights_path"])
    if not pw.is_absolute():
        pw = project_dir / pw
    cfg["_pose_weights"] = str(pw)

    return cfg


def _find_video(video_dir: Path, camera_id: str) -> Path:
    """Find the video file for a given camera ID.

    Args:
        video_dir: Directory containing video files.
        camera_id: Camera identifier (e.g., 'e3v831e').

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
    """Parse ultralytics OBB results into dicts with bbox, confidence, angle, obb_points.

    Args:
        results: Ultralytics Results list from model.predict().

    Returns:
        List of detection dicts for this frame.
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

        # AABB bbox from OBB corners
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
# Affine crop extraction (standalone, from PoseEstimationBackend._extract_crop)
# ---------------------------------------------------------------------------

_CROP_SIZE = (128, 64)  # (width, height) matching pose model training


def _extract_crop(det: dict, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract an OBB-aligned affine crop for a detection.

    Implements the same 3-point affine logic as PoseEstimationBackend._extract_crop(),
    keeping this script standalone.

    Args:
        det: Detection dict with 'obb_points' key.
        frame: Full-frame BGR image.

    Returns:
        Tuple of (crop_image, affine_matrix_M).
    """
    crop_w, crop_h = _CROP_SIZE
    pts = det["obb_points"].astype(np.float32)

    # Ultralytics xyxyxyxy corner order:
    #   pts[0] = right-bottom
    #   pts[1] = right-top
    #   pts[2] = left-top  (TL)
    #   pts[3] = left-bottom (LB)
    lt, rt, lb = pts[2], pts[1], pts[3]
    side_w = float(np.linalg.norm(rt - lt))
    side_h = float(np.linalg.norm(lb - lt))

    if side_h > side_w:
        rb = pts[0]
        src = np.array([lb, lt, rb], dtype=np.float32)
    else:
        src = np.array([lt, rt, lb], dtype=np.float32)

    dst = np.array([[0, 0], [crop_w - 1, 0], [0, crop_h - 1]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop_image = cv2.warpAffine(
        frame,
        M,
        (crop_w, crop_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return crop_image, M


def _invert_affine_points(crop_points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Back-project crop-space points to frame coordinates.

    Args:
        crop_points: Points in crop space, shape (N, 2).
        M: Affine transform matrix, shape (2, 3).

    Returns:
        Points in frame coordinates, shape (N, 2).
    """
    M_inv = cv2.invertAffineTransform(M)
    pts = crop_points.reshape(1, -1, 2).astype(np.float64)
    result = cv2.transform(pts, M_inv)
    return result.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Multi-instance pose extraction
# ---------------------------------------------------------------------------


def _extract_all_pose_instances(results: list) -> list[dict]:
    """Extract ALL pose instances from YOLO-pose results (not just [0]).

    Args:
        results: Ultralytics Results list from pose model.predict().

    Returns:
        List of dicts, each with 'kpts_xy' (K,2) and 'kpts_conf' (K,) arrays.
    """
    instances = []
    if not results:
        return instances
    res = results[0]
    if res.keypoints is None:
        return instances

    kp = res.keypoints
    n_instances = len(kp.xy)

    for i in range(n_instances):
        kpts_xy = kp.xy[i].cpu().numpy().astype(np.float32)
        if kp.conf is not None and len(kp.conf) > i:
            kpts_conf = kp.conf[i].cpu().numpy().astype(np.float32)
        else:
            kpts_conf = np.ones(len(kpts_xy), dtype=np.float32)
        instances.append({"kpts_xy": kpts_xy, "kpts_conf": kpts_conf})

    return instances


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _draw_obb(
    img: np.ndarray,
    corners: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    offset: tuple[int, int] = (0, 0),
) -> None:
    """Draw an oriented bounding box on an image.

    Args:
        img: Image to draw on (modified in-place).
        corners: OBB corners, shape (4, 2).
        color: BGR color tuple.
        thickness: Line thickness.
        offset: (x, y) offset to subtract from corner coordinates (for cropping).
    """
    ox, oy = offset
    pts = (corners - np.array([ox, oy])).astype(np.int32)
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % 4]), color, thickness)


def _draw_keypoints(
    img: np.ndarray,
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    color: tuple[int, int, int],
    offset: tuple[int, int] = (0, 0),
    dashed: bool = False,
) -> None:
    """Draw keypoints with connections on an image.

    Circle radius is proportional to per-keypoint confidence.

    Args:
        img: Image to draw on (modified in-place).
        kpts_xy: Keypoint coordinates in frame space, shape (K, 2).
        kpts_conf: Per-keypoint confidence, shape (K,).
        color: BGR color tuple.
        offset: (x, y) offset for crop region.
        dashed: If True, draw connections as dotted lines (secondary instances).
    """
    ox, oy = offset

    # Desaturate color for secondary instances
    if dashed:
        color = tuple(min(255, c + 80) for c in color)  # type: ignore[assignment]

    # Draw connections
    for i, j in _SKELETON:
        if i < len(kpts_xy) and j < len(kpts_xy):
            pt1 = (int(kpts_xy[i][0] - ox), int(kpts_xy[i][1] - oy))
            pt2 = (int(kpts_xy[j][0] - ox), int(kpts_xy[j][1] - oy))
            if dashed:
                _draw_dashed_line(img, pt1, pt2, color, thickness=1)
            else:
                cv2.line(img, pt1, pt2, color, 2)

    # Draw circles with confidence-encoded radius
    for k in range(len(kpts_xy)):
        pt = (int(kpts_xy[k][0] - ox), int(kpts_xy[k][1] - oy))
        conf = float(kpts_conf[k]) if k < len(kpts_conf) else 0.5
        radius = max(2, int(conf * 10))
        cv2.circle(img, pt, radius, color, -1)


def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 5,
) -> None:
    """Draw a dashed line between two points.

    Args:
        img: Image to draw on.
        pt1: Start point (x, y).
        pt2: End point (x, y).
        color: BGR color.
        thickness: Line thickness.
        dash_length: Length of each dash segment in pixels.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if dist < 1:
        return
    n_dashes = max(1, int(dist / dash_length))
    for i in range(0, n_dashes, 2):
        t1 = i / n_dashes
        t2 = min((i + 1) / n_dashes, 1.0)
        sx = int(x1 + t1 * (x2 - x1))
        sy = int(y1 + t1 * (y2 - y1))
        ex = int(x1 + t2 * (x2 - x1))
        ey = int(y1 + t2 * (y2 - y1))
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_single_threshold(
    cfg: dict,
    camera_id: str,
    start_frame: int,
    end_frame: int,
    crop_region: tuple[int, int, int, int],
    output_dir: Path,
    conf_threshold: float,
) -> None:
    """Run OBB detection + tracking + pose estimation, produce annotated video and stats.

    Args:
        cfg: Parsed project config.
        camera_id: Camera to analyze.
        start_frame: First frame index.
        end_frame: Last frame index (exclusive).
        crop_region: Pixel crop region as (x1, y1, x2, y2).
        output_dir: Directory for output files.
        conf_threshold: Detection confidence threshold.
    """
    from ultralytics import YOLO

    video_path = _find_video(cfg["_video_dir"], camera_id)
    obb_model = YOLO(cfg["_obb_weights"])
    pose_model = YOLO(cfg["_pose_weights"])

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame <= 0 or end_frame > total_frames:
        end_frame = total_frames

    print(f"Video: {video_path} ({total_frames} frames)")
    print(f"Frame range: {start_frame}-{end_frame}")
    print(f"Crop region: {crop_region}")
    print(f"Confidence threshold: {conf_threshold}")

    # --- Pass 1: Detection + Tracking + Pose ---
    from aquapose.core.tracking.ocsort_wrapper import OcSortTracker

    tracker = OcSortTracker(camera_id=camera_id, min_hits=1, det_thresh=0.1)

    frame_data: list[dict] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for fidx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        # OBB detection
        obb_results = obb_model.predict(
            frame, conf=conf_threshold, iou=0.95, verbose=False
        )
        dets = _parse_obb_results(obb_results)

        # Build Detection objects for tracker
        from aquapose.core.types.detection import Detection

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
        tracker.update(fidx, det_objects)

        # Get per-frame track assignments from tracker output
        # We check active_tracks to map detections to track IDs
        # Build a mapping from bbox overlap to track assignment
        track_assignments = _get_frame_track_assignments(tracker, dets, fidx)

        # Pose estimation for each detection
        all_poses: list[list[dict]] = []
        all_poses_frame: list[list[dict]] = []  # poses in frame coords
        affine_Ms: list[np.ndarray | None] = []

        for d in dets:
            if d["obb_points"] is not None:
                try:
                    crop_img, M = _extract_crop(d, frame)
                    pose_results = pose_model.predict(crop_img, conf=0.1, verbose=False)
                    instances = _extract_all_pose_instances(pose_results)
                    all_poses.append(instances)

                    # Transform keypoints to frame coords
                    frame_instances = []
                    for inst in instances:
                        frame_kpts = _invert_affine_points(inst["kpts_xy"], M)
                        frame_instances.append(
                            {"kpts_xy": frame_kpts, "kpts_conf": inst["kpts_conf"]}
                        )
                    all_poses_frame.append(frame_instances)
                    affine_Ms.append(M)
                except Exception as e:
                    print(f"  Frame {fidx}: crop/pose error: {e}")
                    all_poses.append([])
                    all_poses_frame.append([])
                    affine_Ms.append(None)
            else:
                all_poses.append([])
                all_poses_frame.append([])
                affine_Ms.append(None)

        frame_data.append(
            {
                "frame_idx": fidx,
                "detections": dets,
                "track_assignments": track_assignments,
                "poses_frame": all_poses_frame,
                "poses_crop": all_poses,
                "frame_image": frame,
            }
        )

        if (fidx - start_frame) % 50 == 0:
            print(f"  Processed frame {fidx} ({len(dets)} detections)")

    cap.release()
    print(f"Pass 1 complete: {len(frame_data)} frames processed")

    # Get final tracklets for reference
    tracklets = tracker.get_tracklets()
    print(f"Tracklets: {len(tracklets)} confirmed tracks")

    # --- Pass 2: Render annotated crop video ---
    cx1, cy1, cx2, cy2 = crop_region
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    output_video = output_dir / "occlusion_investigation.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, 30, (crop_w, crop_h))

    for fd in frame_data:
        frame = fd["frame_image"]
        crop = frame[cy1:cy2, cx1:cx2].copy()
        offset = (cx1, cy1)

        for di, det in enumerate(fd["detections"]):
            track_id = fd["track_assignments"].get(di)
            conf = det["confidence"]

            # Color logic: tracked = palette color, untracked high conf = red,
            # untracked low conf = gray
            if track_id is not None:
                color = _PALETTE_BGR[track_id % len(_PALETTE_BGR)]
            elif conf >= 0.5:
                color = (0, 0, 255)  # red
            else:
                color = (128, 128, 128)  # gray

            # Draw OBB
            _draw_obb(crop, det["obb_points"], color, thickness=2, offset=offset)

            # Draw keypoints
            poses = fd["poses_frame"][di] if di < len(fd["poses_frame"]) else []
            for pi, pose in enumerate(poses):
                is_secondary = pi > 0
                _draw_keypoints(
                    crop,
                    pose["kpts_xy"],
                    pose["kpts_conf"],
                    color,
                    offset=offset,
                    dashed=is_secondary,
                )

        # Frame number annotation (small, top-left)
        cv2.putText(
            crop,
            f"F{fd['frame_idx']}",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        writer.write(crop)

    writer.release()
    print(f"Video saved: {output_video}")

    # --- Save per-frame statistics ---
    stats = []
    for fd in frame_data:
        n_dets = len(fd["detections"])
        n_tracked = sum(1 for v in fd["track_assignments"].values() if v is not None)
        multi_instance_flags = []
        per_det_instances = []
        per_det_kp_confs = []

        for di, _det in enumerate(fd["detections"]):
            poses = fd["poses_frame"][di] if di < len(fd["poses_frame"]) else []
            n_instances = len(poses)
            per_det_instances.append(n_instances)
            multi_instance_flags.append(n_instances > 1)

            # Per-keypoint confidences for primary instance
            if poses:
                per_det_kp_confs.append(poses[0]["kpts_conf"].tolist())
            else:
                per_det_kp_confs.append([])

        stats.append(
            {
                "frame_idx": fd["frame_idx"],
                "n_detections": n_dets,
                "n_tracked": n_tracked,
                "per_det_n_instances": per_det_instances,
                "has_multi_instance": any(multi_instance_flags),
                "per_det_kp_confidences": per_det_kp_confs,
            }
        )

    stats_path = output_dir / "frame_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved: {stats_path}")

    # Clean up frame images from memory
    for fd in frame_data:
        del fd["frame_image"]


def _get_frame_track_assignments(
    tracker: object, dets: list[dict], frame_idx: int
) -> dict[int, int | None]:
    """Map detection indices to track IDs using tracker's internal state.

    Uses IoU matching between detection bboxes and tracker's active track positions
    to determine which detection corresponds to which track.

    Args:
        tracker: OcSortTracker instance.
        dets: List of detection dicts for this frame.
        frame_idx: Current frame index.

    Returns:
        Dict mapping detection index to track_id (or None if untracked).
    """
    assignments: dict[int, int | None] = {i: None for i in range(len(dets))}

    # Access tracker's internal builders to find tracks active in this frame
    builders = getattr(tracker, "_builders", {})

    # Get the last frame's track positions from builders
    for local_id, builder in builders.items():
        if not builder.frames or builder.frames[-1] != frame_idx:
            continue
        if builder.frame_status[-1] != "detected":
            continue

        # Match by bbox IoU
        trk_bbox = builder.bboxes[-1]  # (x1, y1, w, h)
        tx, ty, tw, th = trk_bbox
        best_iou = 0.3  # minimum IoU threshold
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
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
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
# Confidence sweep mode
# ---------------------------------------------------------------------------


def run_confidence_sweep(
    cfg: dict,
    camera_id: str,
    start_frame: int,
    end_frame: int,
    output_dir: Path,
) -> None:
    """Run detection at low threshold, then filter at multiple thresholds.

    Args:
        cfg: Parsed project config.
        camera_id: Camera to analyze.
        start_frame: First frame index.
        end_frame: Last frame index (exclusive).
        output_dir: Directory for output files.
    """
    from ultralytics import YOLO

    video_path = _find_video(cfg["_video_dir"], camera_id)
    obb_model = YOLO(cfg["_obb_weights"])

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame <= 0 or end_frame > total_frames:
        end_frame = total_frames

    print(f"Confidence sweep: {video_path}")
    print(f"Frame range: {start_frame}-{end_frame}")

    # Detect at very low threshold to capture everything
    all_confs: list[list[float]] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for fidx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        results = obb_model.predict(frame, conf=0.05, iou=0.95, verbose=False)
        dets = _parse_obb_results(results)
        all_confs.append([d["confidence"] for d in dets])

        if (fidx - start_frame) % 100 == 0:
            print(f"  Frame {fidx}")

    cap.release()

    # Sweep thresholds
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    rows = []

    for thresh in thresholds:
        per_frame_counts = []
        total = 0
        for frame_confs in all_confs:
            n = sum(1 for c in frame_confs if c >= thresh)
            per_frame_counts.append(n)
            total += n

        arr = np.array(per_frame_counts)
        rows.append(
            {
                "threshold": thresh,
                "total_dets": total,
                "mean_per_frame": float(arr.mean()),
                "median_per_frame": float(np.median(arr)),
                "min_per_frame": int(arr.min()),
                "max_per_frame": int(arr.max()),
            }
        )

    # Format as markdown table
    header = (
        "| Threshold | Total Dets | Mean/Frame | Median/Frame | Min/Frame | Max/Frame |"
    )
    sep = (
        "|-----------|-----------|------------|--------------|-----------|-----------|"
    )
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['threshold']:.2f}      | {r['total_dets']:>9} | {r['mean_per_frame']:>10.1f} | "
            f"{r['median_per_frame']:>12.1f} | {r['min_per_frame']:>9} | {r['max_per_frame']:>9} |"
        )

    table = "\n".join(lines)
    print("\n## Confidence Sweep Results\n")
    print(table)

    sweep_path = output_dir / "confidence_sweep.md"
    with open(sweep_path, "w") as f:
        f.write("# Confidence Sweep Results\n\n")
        f.write(f"Camera: {camera_id}\n")
        f.write(f"Frames: {start_frame}-{end_frame} ({len(all_confs)} frames)\n\n")
        f.write(table)
        f.write("\n")
    print(f"\nSaved: {sweep_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_crop_region(s: str) -> tuple[int, int, int, int]:
    """Parse a crop region string 'x1,y1,x2,y2' into a tuple.

    Args:
        s: Comma-separated crop region string.

    Returns:
        Tuple of (x1, y1, x2, y2).
    """
    parts = s.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"Expected x1,y1,x2,y2 but got '{s}'")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def main() -> None:
    """CLI entry point for occlusion investigation."""
    parser = argparse.ArgumentParser(
        description="Investigate OBB detection and pose estimation behavior during fish occlusion events.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project-config",
        required=True,
        help="Path to project config YAML (e.g., ~/aquapose/projects/YH/config.yaml)",
    )
    parser.add_argument(
        "--camera",
        required=True,
        help="Camera ID to analyze (e.g., e3v831e)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index (default: 0)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=-1,
        help="Last frame index, exclusive (default: full video)",
    )
    parser.add_argument(
        "--crop-region",
        type=_parse_crop_region,
        default="263,225,613,525",
        help="Pixel crop region as x1,y1,x2,y2 (default: 263,225,613,525)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for video and stats (default: current dir)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.2,
        help="Detection confidence threshold (default: 0.2)",
    )
    parser.add_argument(
        "--conf-sweep",
        action="store_true",
        help="Run confidence sweep mode instead of single-threshold video",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(args.project_config)

    if args.conf_sweep:
        run_confidence_sweep(
            cfg=cfg,
            camera_id=args.camera,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            output_dir=output_dir,
        )
    else:
        run_single_threshold(
            cfg=cfg,
            camera_id=args.camera,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            crop_region=args.crop_region,
            output_dir=output_dir,
            conf_threshold=args.conf_threshold,
        )


if __name__ == "__main__":
    main()
