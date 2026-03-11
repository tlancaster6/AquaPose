#!/usr/bin/env python3
"""Evaluation script comparing KeypointTracker (keypoint_bidi) against OC-SORT baseline.

Runs OBB detection + pose estimation once and caches all Detection objects per
frame.  The same cached detections (including keypoints) are fed to both an
OcSortTracker and a KeypointTracker.  Computes evaluate_tracking() and
evaluate_fragmentation_2d() for both, prints a side-by-side comparison, and
saves an annotated video for the keypoint_bidi tracker.

Phase 80 baseline (OC-SORT, e3v83eb, frames 3300-4500):
  - 27 tracks, 93.1% coverage, 0 gaps, 1.000 continuity, 18 births / 17 deaths

Usage::

    hatch run python scripts/evaluate_custom_tracker.py \\
        --config ~/aquapose/projects/YH/config.yaml

Tuning knobs (adjust if track count does not improve over 27-track baseline)::

    --max-age INT               Max coast frames before dropping a track (default 15)
    --det-thresh FLOAT          Minimum detection confidence forwarded to tracker (default 0.1)
    --n-init INT                Minimum hits before track is confirmed (default 1)
    --base-r FLOAT              KF base measurement noise variance (default 10.0)
    --lambda-ocm FLOAT          OCM weight in cost matrix (default 0.2)
    --match-cost-threshold FLOAT  Max cost for Hungarian match acceptance (default 1.2)
    --ocr-threshold FLOAT       Min OKS for observation-centric recovery (default 0.5)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Colour palette (BGR) — same as measure_baseline_tracking.py
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

    # Resolve pose weights — config.yaml uses "midline" key (backward compat)
    pose_section = cfg.get("pose", cfg.get("midline", {}))
    pw = pose_section.get("weights_path", "")
    if pw:
        pw_path = Path(pw)
        if not pw_path.is_absolute():
            pw_path = project_dir / pw_path
        cfg["_pose_weights"] = str(pw_path)
    else:
        cfg["_pose_weights"] = None

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
# Pose inference helpers
# ---------------------------------------------------------------------------


def _run_pose_on_detections(
    pose_backend: object,
    det_objects: list,
    frame: np.ndarray,
    frame_idx: int,
) -> None:
    """Run pose estimation on all detections for one frame, writing results in-place.

    Args:
        pose_backend: PoseEstimationBackend instance.
        det_objects: List of Detection objects to enrich with keypoints.
        frame: Full-frame BGR image for this camera.
        frame_idx: Absolute frame index (for metadata).
    """
    if not det_objects:
        return

    # Extract crops
    crops = []
    metadata = []
    det_refs = []
    for det in det_objects:
        try:
            crop = pose_backend._extract_crop(det, frame)  # type: ignore[union-attr]
        except Exception:
            continue
        crops.append(crop)
        metadata.append((det, "eval_cam", frame_idx))
        det_refs.append(det)

    if not crops:
        return

    # Batched inference
    results = pose_backend.process_batch(crops, metadata)  # type: ignore[union-attr]

    # Write keypoints onto Detection objects in-place
    for det, (kpts_xy, kpts_conf) in zip(det_refs, results, strict=True):
        if kpts_xy is not None and kpts_conf is not None:
            det.keypoints = kpts_xy
            det.keypoint_conf = kpts_conf


# ---------------------------------------------------------------------------
# Track assignment for video rendering (OcSort internal _builders)
# ---------------------------------------------------------------------------


def _get_frame_track_assignments_ocsort(
    tracker: object, frame_idx: int
) -> dict[int, int]:
    """Map builder local_ids to their latest confirmed frame for OcSortTracker.

    Args:
        tracker: OcSortTracker instance with _builders attribute.
        frame_idx: Current absolute frame index.

    Returns:
        Dict mapping local_id to track_id for builders active in this frame.
    """
    assignments: dict[int, int] = {}
    builders = getattr(tracker, "_builders", {})
    for local_id, builder in builders.items():
        if builder.frames and builder.frames[-1] == frame_idx:
            assignments[local_id] = local_id
    return assignments


def _get_frame_centroid_assignments_keypoint(
    tracker: object, frame_idx: int
) -> dict[tuple[float, float], int]:
    """Map frame centroids to track IDs for KeypointTracker at current frame.

    Uses the tracker's internal builders to find which track owns each
    centroid in this frame.

    Args:
        tracker: KeypointTracker instance with _builders attribute.
        frame_idx: Current absolute frame index.

    Returns:
        Dict mapping (cx, cy) centroid to track_id.
    """
    centroid_to_id: dict[tuple[float, float], int] = {}
    builders = getattr(tracker, "_builders", {})
    for local_id, builder in builders.items():
        frames = getattr(builder, "frames", [])
        centroids = getattr(builder, "centroids", [])
        if not frames or frames[-1] != frame_idx:
            continue
        if centroids:
            cx, cy = centroids[-1]
            centroid_to_id[(float(cx), float(cy))] = local_id
    return centroid_to_id


def _nearest_centroid_track_id(
    det_cx: float,
    det_cy: float,
    centroid_map: dict[tuple[float, float], int],
    max_dist: float = 50.0,
) -> int | None:
    """Find the track ID whose centroid is nearest to a detection centroid.

    Args:
        det_cx: Detection centroid x.
        det_cy: Detection centroid y.
        centroid_map: Mapping from (cx, cy) to track_id.
        max_dist: Maximum pixel distance to consider a match.

    Returns:
        Track ID of nearest centroid, or None if none within max_dist.
    """
    best_id = None
    best_dist = max_dist
    for (cx, cy), tid in centroid_map.items():
        dist = float(np.hypot(det_cx - cx, det_cy - cy))
        if dist < best_dist:
            best_dist = dist
            best_id = tid
    return best_id


# ---------------------------------------------------------------------------
# Metrics summary printing
# ---------------------------------------------------------------------------


def _print_metrics(
    label: str,
    tracking_metrics: object,
    frag_metrics: object,
    camera_id: str,
    start_frame: int,
    end_frame: int,
    n_animals: int,
) -> None:
    """Print a human-readable metrics summary to stdout.

    Args:
        label: Tracker label for display.
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
    print(f"=== {label} Metrics ===")
    print(
        f"Camera: {camera_id} | Frames: {start_frame}-{end_frame} ({n_frames} frames)"
    )
    print()
    print(f"Track count:        {tm.track_count}")
    print(f"Length min:          {tm.length_min} frames")
    print(f"Length max:          {tm.length_max} frames")
    print(f"Length median:       {tm.length_median} frames")
    print(f"Coast frequency:    {tm.coast_frequency:.3f}")
    print(f"Detection coverage: {tm.detection_coverage:.3f}")
    print()
    print("Fragmentation:")
    print(f"  Total gaps:       {fm.total_gaps}")
    print(f"  Mean gap length:  {fm.mean_gap_duration:.1f}")
    print(f"  Continuity ratio: {fm.mean_continuity_ratio:.3f}")
    print(f"  Births:           {fm.track_births}")
    print(f"  Deaths:           {fm.track_deaths}")
    print()
    print(f"Target: {n_animals} tracks, zero fragmentation")
    track_delta = tm.track_count - n_animals
    track_sign = "+" if track_delta >= 0 else ""
    print(
        f"Gap to target: track count delta {track_sign}{track_delta} vs {n_animals}; "
        f"gaps {fm.total_gaps}; continuity {fm.mean_continuity_ratio:.3f}"
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(
    cfg: dict,
    camera_id: str,
    start_frame: int,
    end_frame: int,
    n_animals: int,
    output_dir: Path,
    max_age: int,
    det_thresh: float,
    n_init: int,
    base_r: float = 10.0,
    lambda_ocm: float = 0.2,
    match_cost_threshold: float = 1.2,
    ocr_threshold: float = 0.5,
) -> None:
    """Run dual-tracker evaluation on the specified frame range.

    Caches detections (with pose keypoints) once; feeds to both OC-SORT and
    KeypointTracker.  Saves an annotated video for the keypoint_bidi run and
    writes comparison JSON to output_dir.

    Args:
        cfg: Parsed project config with resolved paths.
        camera_id: Camera ID to analyze.
        start_frame: First frame index (inclusive).
        end_frame: Last frame index (exclusive).
        n_animals: Expected number of fish.
        output_dir: Directory to write video and metrics JSON.
        max_age: Maximum coast frames before dropping a track.
        det_thresh: Minimum detection confidence for trackers.
        n_init: Minimum confirmed hits before track appears in output.
        base_r: KF base measurement noise variance.
        lambda_ocm: OCM weight in cost matrix.
        match_cost_threshold: Max cost for Hungarian assignment match acceptance.
        ocr_threshold: Min OKS for observation-centric recovery.
    """
    from ultralytics import YOLO

    from aquapose.core.detection.backends.yolo_obb import polygon_nms
    from aquapose.core.pose.backends.pose_estimation import PoseEstimationBackend
    from aquapose.core.tracking.keypoint_tracker import KeypointTracker
    from aquapose.core.tracking.ocsort_wrapper import OcSortTracker
    from aquapose.core.types.detection import Detection
    from aquapose.evaluation.stages.fragmentation import evaluate_fragmentation_2d
    from aquapose.evaluation.stages.tracking import evaluate_tracking

    video_path = _find_video(cfg["_video_dir"], camera_id)
    obb_model = YOLO(cfg["_obb_weights"])

    # Build pose backend
    pose_weights = cfg.get("_pose_weights")
    pose_backend: PoseEstimationBackend | None = None
    if pose_weights:
        print(f"Loading pose model: {pose_weights}")
        pose_backend = PoseEstimationBackend(
            weights_path=pose_weights,
            device="cuda",
            n_keypoints=6,
            confidence_floor=0.3,
            min_observed_keypoints=1,
            crop_size=(128, 64),
            conf=0.5,
        )
    else:
        print(
            "WARNING: No pose weights found. KeypointTracker will use OBB centroids only."
        )

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
    print(
        f"Tracker params: max_age={max_age}, det_thresh={det_thresh}, n_init={n_init}"
    )

    # --- Pass 1: Cache all Detection objects with keypoints ---
    print("\nPass 1: OBB detection + pose estimation (caching detections)...")
    all_frame_dets: list[list[Detection]] = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for fidx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        obb_results = obb_model.predict(frame, conf=0.1, iou=0.95, verbose=False)
        raw_dets = _parse_obb_results(obb_results)

        det_objects = [
            Detection(
                bbox=d["bbox"],
                mask=None,
                area=d["area"],
                confidence=d["confidence"],
                angle=d["angle"],
                obb_points=d["obb_points"],
            )
            for d in raw_dets
        ]
        det_objects = polygon_nms(det_objects, iou_threshold=0.45)

        # Populate keypoints for KeypointTracker's OKS cost
        if pose_backend is not None:
            _run_pose_on_detections(pose_backend, det_objects, frame, fidx)

        all_frame_dets.append(det_objects)

        if (fidx - start_frame) % 100 == 0:
            n_with_kpts = sum(1 for d in det_objects if d.keypoints is not None)
            print(
                f"  Frame {fidx}: {len(det_objects)} detections, "
                f"{n_with_kpts} with keypoints"
            )

    cap.release()
    print(f"Detection cache complete: {len(all_frame_dets)} frames")

    # --- Pass 2a: OC-SORT ---
    print("\nPass 2a: Running OC-SORT (matching Phase 80 baseline settings)...")
    ocsort_tracker = OcSortTracker(
        camera_id=camera_id,
        max_age=max_age,
        min_hits=n_init,
        iou_threshold=0.3,
        det_thresh=det_thresh,
    )
    for frame_offset, det_objects in enumerate(all_frame_dets):
        fidx = start_frame + frame_offset
        ocsort_tracker.update(fidx, det_objects)

    ocsort_tracklets = ocsort_tracker.get_tracklets()
    ocsort_tracking = evaluate_tracking(ocsort_tracklets)
    ocsort_frag = evaluate_fragmentation_2d(ocsort_tracklets, n_animals)
    print(f"  OC-SORT: {ocsort_tracking.track_count} tracks")

    # --- Pass 2b: KeypointTracker ---
    print("\nPass 2b: Running KeypointTracker (keypoint_bidi)...")
    kp_tracker = KeypointTracker(
        camera_id=camera_id,
        max_age=max_age,
        n_init=n_init,
        det_thresh=det_thresh,
        base_r=base_r,
        lambda_ocm=lambda_ocm,
        max_gap_frames=5,
        match_cost_threshold=match_cost_threshold,
        ocr_threshold=ocr_threshold,
    )
    for frame_offset, det_objects in enumerate(all_frame_dets):
        fidx = start_frame + frame_offset
        kp_tracker.update(fidx, det_objects)

    kp_tracklets = kp_tracker.get_tracklets()
    kp_tracking = evaluate_tracking(kp_tracklets)
    kp_frag = evaluate_fragmentation_2d(kp_tracklets, n_animals)
    print(f"  KeypointTracker: {kp_tracking.track_count} tracks")

    # --- Coverage trigger check ---
    byte_needed = kp_tracking.detection_coverage < 0.90
    if byte_needed:
        print(
            f"\nCoverage below 90% ({kp_tracking.detection_coverage:.3f}) — "
            "BYTE-style secondary pass recommended. See TRACK-10."
        )
    else:
        print(
            f"\nCoverage {kp_tracking.detection_coverage:.3f} >= 90% threshold — "
            "BYTE-style secondary pass not triggered."
        )

    # --- Print summaries ---
    _print_metrics(
        "OC-SORT (current run)",
        ocsort_tracking,
        ocsort_frag,
        camera_id,
        start_frame,
        end_frame,
        n_animals,
    )
    _print_metrics(
        "KeypointTracker (keypoint_bidi)",
        kp_tracking,
        kp_frag,
        camera_id,
        start_frame,
        end_frame,
        n_animals,
    )

    # --- Side-by-side comparison ---
    _print_comparison(
        ocsort_tracking,
        ocsort_frag,
        kp_tracking,
        kp_frag,
    )

    # --- Save annotated video for keypoint_bidi tracker ---
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "keypoint_bidi_tracking.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (frame_w, frame_h))

    # Build frame->track lookup from tracklets
    frame_to_tracks: dict[int, list[tuple[int, np.ndarray | None]]] = {}
    for t in kp_tracklets:
        for i, f in enumerate(t.frames):
            if f not in frame_to_tracks:
                frame_to_tracks[f] = []
            centroid = t.centroids[i] if i < len(t.centroids) else None
            frame_to_tracks[f].append((t.track_id, centroid))

    print(f"\nRendering annotated video ({len(all_frame_dets)} frames)...")
    cap2 = cv2.VideoCapture(str(video_path))
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_offset, det_objects in enumerate(all_frame_dets):
        ret, frame = cap2.read()
        if not ret:
            break
        fidx = start_frame + frame_offset

        active_tracks = frame_to_tracks.get(fidx, [])
        # Build OBB->track_id assignment via exclusive centroid matching.
        # Use linear_sum_assignment so each detection gets at most one track ID
        # and each track ID is used at most once per frame (no duplicate IDs).
        valid_dets = [
            (i, det, _obb_center(det.obb_points))
            for i, det in enumerate(det_objects)
            if det.obb_points is not None
        ]
        valid_tracks = [
            (j, tid, centroid)
            for j, (tid, centroid) in enumerate(active_tracks)
            if centroid is not None
        ]

        det_to_track_id: dict[int, int] = {}
        if valid_dets and valid_tracks:
            from scipy.optimize import linear_sum_assignment

            dist_thresh = 60.0  # pixel tolerance
            cost_mat = np.full((len(valid_dets), len(valid_tracks)), dist_thresh + 1.0)
            for di, (_, _, (dcx, dcy)) in enumerate(valid_dets):
                for tj, (_, _, (tcx, tcy)) in enumerate(valid_tracks):
                    d = float(np.hypot(dcx - tcx, dcy - tcy))
                    if d <= dist_thresh:
                        cost_mat[di, tj] = d
            row_idx, col_idx = linear_sum_assignment(cost_mat)
            for r, c in zip(row_idx, col_idx, strict=False):
                if cost_mat[r, c] <= dist_thresh:
                    det_i = valid_dets[r][0]
                    tid = valid_tracks[c][1]
                    det_to_track_id[det_i] = tid

        for det_i, det, (cx, cy) in valid_dets:
            best_id = det_to_track_id.get(det_i)
            color = (
                _PALETTE_BGR[best_id % len(_PALETTE_BGR)]
                if best_id is not None
                else (128, 128, 128)
            )
            _draw_obb(frame, det.obb_points, color, thickness=2)
            if best_id is not None:
                cv2.putText(
                    frame,
                    str(best_id),
                    (int(cx) - 8, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        cv2.putText(
            frame,
            f"F{fidx}",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        writer.write(frame)

    cap2.release()
    writer.release()
    print(f"Video saved: {output_video}")

    # --- Save comparison JSON ---
    comparison_json = {
        "config": {
            "camera": camera_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "n_frames": n_frames,
            "n_animals": n_animals,
            "max_age": max_age,
            "det_thresh": det_thresh,
            "n_init": n_init,
            "base_r": base_r,
            "lambda_ocm": lambda_ocm,
            "match_cost_threshold": match_cost_threshold,
            "ocr_threshold": ocr_threshold,
        },
        "ocsort": {
            "tracking": ocsort_tracking.to_dict(),
            "fragmentation": ocsort_frag.to_dict(),
        },
        "keypoint_bidi": {
            "tracking": kp_tracking.to_dict(),
            "fragmentation": kp_frag.to_dict(),
        },
        "byte_trigger": byte_needed,
    }
    metrics_path = output_dir / "comparison_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(comparison_json, f, indent=2)
    print(f"Metrics saved: {metrics_path}")


def _print_comparison(
    ocsort_tracking: object,
    ocsort_frag: object,
    kp_tracking: object,
    kp_frag: object,
) -> None:
    """Print a side-by-side comparison table.

    Args:
        ocsort_tracking: TrackingMetrics for OC-SORT.
        ocsort_frag: FragmentationMetrics for OC-SORT.
        kp_tracking: TrackingMetrics for KeypointTracker.
        kp_frag: FragmentationMetrics for KeypointTracker.
    """
    # Phase 80 baseline reference
    baseline = {
        "track_count": 27,
        "detection_coverage": 0.931,
        "total_gaps": 0,
        "continuity": 1.000,
        "track_births": 18,
        "track_deaths": 17,
    }

    print()
    print("=" * 70)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 70)
    print(
        f"{'Metric':<28} {'Ph80 Baseline':>15} {'OC-SORT (now)':>15} {'Kpt Bidi':>15}"
    )
    print("-" * 70)

    rows = [
        (
            "Track count (target=9)",
            baseline["track_count"],
            ocsort_tracking.track_count,
            kp_tracking.track_count,
        ),
        (
            "Detection coverage",
            f"{baseline['detection_coverage']:.3f}",
            f"{ocsort_tracking.detection_coverage:.3f}",
            f"{kp_tracking.detection_coverage:.3f}",
        ),
        (
            "Total gaps",
            baseline["total_gaps"],
            ocsort_frag.total_gaps,
            kp_frag.total_gaps,
        ),
        (
            "Continuity ratio",
            f"{baseline['continuity']:.3f}",
            f"{ocsort_frag.mean_continuity_ratio:.3f}",
            f"{kp_frag.mean_continuity_ratio:.3f}",
        ),
        (
            "Track births",
            baseline["track_births"],
            ocsort_frag.track_births,
            kp_frag.track_births,
        ),
        (
            "Track deaths",
            baseline["track_deaths"],
            ocsort_frag.track_deaths,
            kp_frag.track_deaths,
        ),
        (
            "Length median (frames)",
            "-",
            f"{ocsort_tracking.length_median:.0f}",
            f"{kp_tracking.length_median:.0f}",
        ),
        (
            "Length max (frames)",
            "-",
            ocsort_tracking.length_max,
            kp_tracking.length_max,
        ),
        (
            "Coast frequency",
            "-",
            f"{ocsort_tracking.coast_frequency:.3f}",
            f"{kp_tracking.coast_frequency:.3f}",
        ),
    ]

    for name, baseline_val, ocsort_val, kp_val in rows:
        print(f"{name:<28} {baseline_val!s:>15} {ocsort_val!s:>15} {kp_val!s:>15}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for custom tracker evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Run OBB detection + pose estimation once, then feed the same "
            "detections to both OC-SORT and KeypointTracker. "
            "Produces a side-by-side metrics comparison and annotated video."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        default="./eval_tracker_output",
        help="Directory to save video and metrics JSON (default: ./eval_tracker_output)",
    )
    # Tuning knobs
    parser.add_argument(
        "--max-age",
        type=int,
        default=15,
        help="Max coast frames before dropping a track (default: 15)",
    )
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=0.1,
        help="Minimum detection confidence forwarded to trackers (default: 0.1)",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=1,
        help="Minimum hits before track is confirmed (default: 1, matching Phase 80 baseline)",
    )
    parser.add_argument(
        "--base-r",
        type=float,
        default=None,
        help="KF base measurement noise variance for keypoint_bidi (default: 10.0)",
    )
    parser.add_argument(
        "--lambda-ocm",
        type=float,
        default=None,
        help="OCM weight in cost matrix for keypoint_bidi (default: 0.2)",
    )
    parser.add_argument(
        "--match-cost-threshold",
        type=float,
        default=None,
        help="Max cost for Hungarian assignment match acceptance (default: 1.2)",
    )
    parser.add_argument(
        "--ocr-threshold",
        type=float,
        default=None,
        help="Min OKS for observation-centric recovery (default: 0.5)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser()

    cfg = _load_config(args.config)

    run_evaluation(
        cfg=cfg,
        camera_id=args.camera,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        n_animals=args.n_animals,
        output_dir=output_dir,
        max_age=args.max_age,
        det_thresh=args.det_thresh,
        n_init=args.n_init,
        base_r=args.base_r if args.base_r is not None else 10.0,
        lambda_ocm=args.lambda_ocm if args.lambda_ocm is not None else 0.2,
        match_cost_threshold=args.match_cost_threshold
        if args.match_cost_threshold is not None
        else 1.2,
        ocr_threshold=args.ocr_threshold if args.ocr_threshold is not None else 0.5,
    )


if __name__ == "__main__":
    main()
