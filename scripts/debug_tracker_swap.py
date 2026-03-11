#!/usr/bin/env python3
"""Debug script to investigate ID swap between tracks 0 and 1 around frame 3390.

Runs the KeypointTracker with match_cost_threshold=2.0 and dumps per-frame
association details (cost matrix, OKS, OCM, assignments) for frames around
the occlusion event.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_dir = Path(cfg["project_dir"]).expanduser()
    vd = Path(cfg["video_dir"])
    if not vd.is_absolute():
        vd = project_dir / vd
    cfg["_video_dir"] = vd
    dw = Path(cfg["detection"]["weights_path"])
    if not dw.is_absolute():
        dw = project_dir / dw
    cfg["_obb_weights"] = str(dw)
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
    candidates = list(video_dir.glob(f"{camera_id}*"))
    if not candidates:
        raise FileNotFoundError(
            f"No video found for camera '{camera_id}' in {video_dir}"
        )
    return candidates[0]


def _parse_obb_results(results: list) -> list[dict]:
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
        corners = corners_all[i]
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


def run_debug(
    cfg: dict,
    camera_id: str,
    start_frame: int,
    end_frame: int,
    debug_start: int,
    debug_end: int,
) -> None:
    from ultralytics import YOLO

    from aquapose.core.detection.backends.yolo_obb import polygon_nms
    from aquapose.core.pose.backends.pose_estimation import PoseEstimationBackend
    from aquapose.core.tracking.keypoint_sigmas import DEFAULT_SIGMAS
    from aquapose.core.tracking.keypoint_tracker import (
        KeypointTracker,
        _SinglePassTracker,
        build_cost_matrix,
        compute_heading,
        compute_ocm_matrix,
        compute_oks_matrix,
    )
    from aquapose.core.types.detection import Detection

    video_path = _find_video(cfg["_video_dir"], camera_id)
    obb_model = YOLO(cfg["_obb_weights"])

    pose_weights = cfg.get("_pose_weights")
    pose_backend = None
    if pose_weights:
        pose_backend = PoseEstimationBackend(
            weights_path=pose_weights,
            device="cuda",
            n_keypoints=6,
            confidence_floor=0.3,
            min_observed_keypoints=1,
            crop_size=(128, 64),
            conf=0.5,
        )

    # Cache detections
    print(f"Caching detections for frames {start_frame}-{end_frame}...")
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    all_frame_dets: list[list[Detection]] = []
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
        if pose_backend is not None and det_objects:
            crops, metadata, det_refs = [], [], []
            for det in det_objects:
                try:
                    crop = pose_backend._extract_crop(det, frame)
                except Exception:
                    continue
                crops.append(crop)
                metadata.append((det, "eval_cam", fidx))
                det_refs.append(det)
            if crops:
                results = pose_backend.process_batch(crops, metadata)
                for det, (kpts_xy, kpts_conf) in zip(det_refs, results, strict=True):
                    if kpts_xy is not None and kpts_conf is not None:
                        det.keypoints = kpts_xy
                        det.keypoint_conf = kpts_conf
        all_frame_dets.append(det_objects)
    cap.release()
    print(f"Cached {len(all_frame_dets)} frames")

    # Now run tracker with instrumentation
    sigmas = DEFAULT_SIGMAS
    match_cost_threshold = 2.0
    lambda_ocm = 0.2

    tracker = KeypointTracker(
        camera_id=camera_id,
        max_age=15,
        n_init=1,
        det_thresh=0.1,
        base_r=10.0,
        lambda_ocm=lambda_ocm,
        max_gap_frames=5,
        match_cost_threshold=match_cost_threshold,
        ocr_threshold=0.5,
    )

    # Access internal tracker
    sp: _SinglePassTracker = tracker._fwd_tracker

    for frame_offset, det_objects in enumerate(all_frame_dets):
        fidx = start_frame + frame_offset
        in_debug_range = debug_start <= fidx <= debug_end

        if in_debug_range:
            # Pre-update: capture state before update
            valid_dets = []
            for det in det_objects:
                if float(det.confidence) < 0.1:
                    continue
                kpts = getattr(det, "keypoints", None)
                kconf = getattr(det, "keypoint_conf", None)
                if kpts is None or kconf is None:
                    continue
                kpts_arr = np.asarray(kpts, dtype=np.float64)
                kconf_arr = np.asarray(kconf, dtype=np.float64)
                if kpts_arr.shape[0] < 5:
                    continue
                valid_dets.append((det, kpts_arr, kconf_arr))

            track_ids = list(sp._active_tracks.keys())

            print(f"\n{'=' * 70}")
            print(f"FRAME {fidx} (t={((fidx - start_frame) / 30):.2f}s)")
            print(f"{'=' * 70}")
            print(f"Active tracks: {track_ids}")
            print("  Track states: ", end="")
            for tid in track_ids:
                trk = sp._active_tracks[tid]
                print(f"  T{tid}({trk.state}, missed={trk.time_since_update})", end="")
            print()
            print(f"Valid detections: {len(valid_dets)}")

            # Show detection centroids and headings
            for di, (det, kpts, kconf) in enumerate(valid_dets):
                cx = kpts[:, 0].mean()
                cy = kpts[:, 1].mean()
                heading = compute_heading(kpts)
                angle_deg = np.degrees(np.arctan2(heading[1], heading[0]))
                print(
                    f"  Det[{di}]: centroid=({cx:.0f},{cy:.0f}) heading={angle_deg:.0f}deg conf={det.confidence:.2f} kconf_mean={kconf.mean():.2f}"
                )

            if track_ids and valid_dets:
                # Compute cost matrix manually
                pred_kpts_list = []
                for tid in track_ids:
                    trk = sp._active_tracks[tid]
                    pred = trk.kf.predict()
                    pred_kpts_list.append(pred)

                pred_kpts = np.stack(pred_kpts_list, axis=0)
                det_kpts_arr = np.stack([d[1] for d in valid_dets], axis=0)
                det_confs_arr = np.stack([d[2] for d in valid_dets], axis=0)
                det_scales_arr = np.array(
                    [
                        max(float(np.sqrt(max(float(d[0].area or 1), 1.0))), 1.0)
                        for d in valid_dets
                    ],
                    dtype=np.float64,
                )

                oks = compute_oks_matrix(
                    pred_kpts, det_kpts_arr, det_confs_arr, det_scales_arr, sigmas
                )
                pred_headings = np.stack(
                    [compute_heading(p) for p in pred_kpts_list], axis=0
                )
                det_headings = np.stack(
                    [compute_heading(d[1]) for d in valid_dets], axis=0
                )
                ocm = compute_ocm_matrix(pred_headings, det_headings)
                cost = build_cost_matrix(oks, ocm, lambda_ocm=lambda_ocm)

                # Show predicted track positions
                for ti, tid in enumerate(track_ids):
                    pred = pred_kpts_list[ti]
                    cx = pred[:, 0].mean()
                    cy = pred[:, 1].mean()
                    heading = compute_heading(pred)
                    angle_deg = np.degrees(np.arctan2(heading[1], heading[0]))
                    print(
                        f"  Track[{tid}] predicted: centroid=({cx:.0f},{cy:.0f}) heading={angle_deg:.0f}deg"
                    )

                print("\n  Cost matrix (tracks x dets):")
                print(f"  {'':>8}", end="")
                for di in range(len(valid_dets)):
                    print(f"  Det[{di}]:>8", end="")
                print()

                # Print header
                header = f"  {'':>8}"
                for di in range(len(valid_dets)):
                    header += f"  {'D' + str(di):>8}"
                print(header)

                for ti, tid in enumerate(track_ids):
                    row = f"  T{tid:>6}"
                    for di in range(len(valid_dets)):
                        row += f"  {cost[ti, di]:>8.3f}"
                    print(row)

                print("\n  OKS matrix:")
                print(header)
                for ti, tid in enumerate(track_ids):
                    row = f"  T{tid:>6}"
                    for di in range(len(valid_dets)):
                        row += f"  {oks[ti, di]:>8.3f}"
                    print(row)

                print("\n  OCM matrix:")
                print(header)
                for ti, tid in enumerate(track_ids):
                    row = f"  T{tid:>6}"
                    for di in range(len(valid_dets)):
                        row += f"  {ocm[ti, di]:>8.3f}"
                    print(row)

                # Show what Hungarian would pick
                from scipy.optimize import linear_sum_assignment

                row_idx, col_idx = linear_sum_assignment(cost)
                print("\n  Hungarian assignments:")
                for r, c in zip(row_idx, col_idx, strict=True):
                    tid = track_ids[r]
                    accepted = cost[r, c] < match_cost_threshold
                    print(
                        f"    T{tid} -> Det[{c}]  cost={cost[r, c]:.3f}  OKS={oks[r, c]:.3f}  OCM={ocm[r, c]:.3f}  {'ACCEPT' if accepted else f'REJECT (>{match_cost_threshold:.1f})'}"
                    )

        # Actually run the tracker update
        tracker.update(fidx, det_objects)

        if in_debug_range:
            # Post-update: show what happened
            print(f"\n  Post-update active tracks: {list(sp._active_tracks.keys())}")
            for tid in sp._active_tracks:
                trk = sp._active_tracks[tid]
                print(
                    f"    T{tid}: state={trk.state}, hits={trk.hit_streak}, missed={trk.time_since_update}"
                )


if __name__ == "__main__":
    cfg = _load_config(str(Path("~/aquapose/projects/YH/config.yaml").expanduser()))
    run_debug(
        cfg,
        camera_id="e3v83eb",
        start_frame=3300,
        end_frame=3460,
        debug_start=3395,
        debug_end=3450,
    )
