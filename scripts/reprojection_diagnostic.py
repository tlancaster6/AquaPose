"""Diagnostic: all camera views at one frame with reprojected centroids.

For each camera with detections, draws:
- Green bbox for each YOLO detection
- Cyan X at bbox center
- Red dot at reprojected 3D centroid
- Yellow dashed line connecting matched pairs
- Pixel distance annotation
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from aquapose.calibration.loader import (
    compute_undistortion_maps,
    load_calibration_data,
    undistort_image,
)
from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.segmentation.detector import make_detector
from aquapose.tracking import FishTracker

DATA_ROOT = Path("C:/Users/tucke/Desktop/Aqua/AquaPose")
CALIB_PATH = DATA_ROOT / "calibration.json"
VIDEO_DIR = DATA_ROOT / "videos" / "core_videos"
YOLO_WEIGHTS = DATA_ROOT / "yolo" / "train_v2" / "weights" / "best.pt"
OUTPUT_DIR = Path("output")
SKIP_CAMERA = "e3v8250"
TARGET_FRAME = 120


def build_models(calib):
    models = {}
    for name in calib.ring_cameras:
        if name == SKIP_CAMERA:
            continue
        cam = calib.cameras[name]
        undist = compute_undistortion_maps(cam)
        models[name] = RefractiveProjectionModel(
            K=undist.K_new,
            R=cam.R,
            t=cam.t,
            water_z=calib.water_z,
            normal=calib.interface_normal,
            n_air=calib.n_air,
            n_water=calib.n_water,
        )
    return models


def main() -> None:
    print("Loading calibration and models...")
    calib = load_calibration_data(CALIB_PATH)
    models = build_models(calib)
    detector = make_detector("yolo", model_path=str(YOLO_WEIGHTS))

    # Open videos, seek, read one frame per camera
    frames = {}
    undist_maps = {}
    for name in calib.ring_cameras:
        if name == SKIP_CAMERA:
            continue
        matches = sorted(VIDEO_DIR.glob(f"{name}-*.mp4"))
        if not matches:
            continue
        cap = cv2.VideoCapture(str(matches[0]))
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME)
        ret, frame = cap.read()
        cap.release()
        if ret:
            undist_maps[name] = compute_undistortion_maps(calib.cameras[name])
            frames[name] = undistort_image(frame, undist_maps[name])

    print(f"Read frame {TARGET_FRAME} from {len(frames)} cameras")

    # Detect
    dets = {name: detector.detect(frame) for name, frame in frames.items()}
    for name, d in sorted(dets.items()):
        print(f"  {name}: {len(d)} detections")

    # Run tracker for a few frames to get confirmed tracks
    warmup_start = max(0, TARGET_FRAME - 10)
    caps = {}
    for name in frames:
        matches = sorted(VIDEO_DIR.glob(f"{name}-*.mp4"))
        cap = cv2.VideoCapture(str(matches[0]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
        caps[name] = cap

    tracker = FishTracker(min_hits=2, max_age=7, expected_count=9)
    confirmed = []
    for f in range(TARGET_FRAME - warmup_start + 1):
        fr = {}
        for name, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                fr[name] = undistort_image(frame, undist_maps[name])
        if not fr:
            break
        d = {name: detector.detect(frame) for name, frame in fr.items()}
        confirmed = tracker.update(d, models, frame_index=f)

    for cap in caps.values():
        cap.release()

    print(f"\nConfirmed tracks: {len(confirmed)}")

    # Use confirmed tracks from the last frame for visualization
    print(f"Confirmed tracks at target frame: {len(confirmed)}")

    # Build a color map for fish
    colors_rgb = [
        (255, 50, 50),
        (50, 255, 50),
        (50, 50, 255),
        (255, 255, 50),
        (255, 50, 255),
        (50, 255, 255),
        (255, 150, 50),
        (150, 50, 255),
        (50, 255, 150),
        (200, 200, 200),
        (255, 100, 100),
        (100, 255, 100),
    ]

    # For each camera, annotate the frame
    cam_names = sorted(frames.keys())
    annotated = {}

    for cam_id in cam_names:
        img = frames[cam_id].copy()

        # Draw all YOLO bboxes in green
        for det in dets.get(cam_id, []):
            bx, by, bw, bh = det.bbox
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
            # Bbox center
            cx, cy = int(bx + bw / 2), int(by + bh / 2)
            cv2.drawMarker(img, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 20, 3)

        # For each confirmed track, draw the reprojection
        for i, track in enumerate(confirmed):
            color = colors_rgb[i % len(colors_rgb)]
            det_idx = track.camera_detections.get(cam_id)

            # Reproject 3D centroid into this camera regardless
            centroid = list(track.positions)[-1]
            pt = torch.tensor(centroid, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pixels, valid = models[cam_id].project(pt)

            if not valid[0]:
                continue
            pu, pv = int(pixels[0, 0].item()), int(pixels[0, 1].item())

            if not (0 <= pu < 1600 and 0 <= pv < 1200):
                continue

            # Draw reprojected centroid
            cv2.circle(img, (pu, pv), 12, color, -1)
            cv2.circle(img, (pu, pv), 12, (255, 255, 255), 3)

            # Label with fish_id
            label = f"ID{track.fish_id}"
            cv2.putText(
                img,
                label,
                (pu + 15, pv - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                3,
            )
            cv2.putText(
                img, label, (pu + 15, pv - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

            # If this camera was in the association, draw line to matched bbox center
            if det_idx is not None and det_idx < len(dets.get(cam_id, [])):
                det = dets[cam_id][det_idx]
                bx, by, bw, bh = det.bbox
                cx, cy = int(bx + bw / 2), int(by + bh / 2)
                cv2.line(img, (pu, pv), (cx, cy), color, 3, cv2.LINE_AA)
                dist = ((pu - cx) ** 2 + (pv - cy) ** 2) ** 0.5
                mx, my = (pu + cx) // 2, (pv + cy) // 2
                cv2.putText(
                    img,
                    f"{dist:.0f}px",
                    (mx + 8, my - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    3,
                )
                cv2.putText(
                    img,
                    f"{dist:.0f}px",
                    (mx + 8, my - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

        annotated[cam_id] = img

    # Only show cameras that have detections
    active_cams = [c for c in cam_names if len(dets.get(c, [])) > 0]
    print(f"\nCameras with detections: {len(active_cams)}: {active_cams}")

    n_active = len(active_cams)
    cols = min(3, n_active)
    rows = (n_active + cols - 1) // cols
    # Keep full resolution
    thumb_w, thumb_h = 1600, 1200

    grid = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    for idx, cam_id in enumerate(active_cams):
        r, c = divmod(idx, cols)
        img = annotated[cam_id]
        # Add camera label (large)
        cv2.putText(
            img, cam_id, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5
        )
        cv2.putText(img, cam_id, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 2)
        n_det = len(dets.get(cam_id, []))
        cv2.putText(
            img,
            f"{n_det} dets",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
        )
        grid[r * thumb_h : (r + 1) * thumb_h, c * thumb_w : (c + 1) * thumb_w] = img

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "reprojection_diagnostic.png"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"\nSaved: {out_path}")
    print(f"Grid: {rows}x{cols} = {grid.shape[1]}x{grid.shape[0]} pixels")


if __name__ == "__main__":
    main()
