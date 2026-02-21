"""Tiled tracking video: all 12 ring cameras with live tracking overlays.

Produces an MP4 where each frame is a 4x3 tiled grid of all ring cameras
(skipping e3v8250), annotated with YOLO bboxes, reprojected 3D centroids,
and connecting lines between matched detections and track centroids.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from aquapose.calibration.loader import (
    UndistortionMaps,
    compute_undistortion_maps,
    load_calibration_data,
    undistort_image,
)
from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.segmentation.detector import make_detector
from aquapose.tracking import FishTracker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT = Path("C:/Users/tucke/Desktop/Aqua/AquaPose")
CALIB_PATH = DATA_ROOT / "calibration.json"
VIDEO_DIR = DATA_ROOT / "videos" / "core_videos"
YOLO_WEIGHTS = DATA_ROOT / "yolo" / "train_v2" / "weights" / "best.pt"
OUTPUT_DIR = Path("output")

FRAME_START = 0
N_FRAMES = 420
SKIP_CAMERA = "e3v8250"

# Tile: 1600/3 ≈ 533, 1200/3 = 400
TILE_W = 533
TILE_H = 400
GRID_COLS = 4
GRID_ROWS = 3

# Fish color palette (BGR for OpenCV)
COLORS_BGR = [
    (50, 50, 255),
    (50, 255, 50),
    (255, 50, 50),
    (50, 255, 255),
    (255, 50, 255),
    (255, 255, 50),
    (50, 150, 255),
    (255, 50, 150),
    (150, 255, 50),
    (200, 200, 200),
    (100, 100, 255),
    (100, 255, 100),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_models(calib):
    """Build RefractiveProjectionModel per ring camera using undistorted K."""
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


def precompute_undistortion(calib) -> dict[str, UndistortionMaps]:
    """Compute undistortion maps once for all ring cameras."""
    maps = {}
    for name in calib.ring_cameras:
        if name == SKIP_CAMERA:
            continue
        maps[name] = compute_undistortion_maps(calib.cameras[name])
    return maps


def annotate_camera(
    img: np.ndarray,
    cam_id: str,
    dets: list,
    confirmed: list,
    model: RefractiveProjectionModel,
) -> np.ndarray:
    """Draw tracking overlays on a single camera image."""
    out = img.copy()

    # Green bbox + cyan cross for each detection
    for det in dets:
        bx, by, bw, bh = det.bbox
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cx, cy = int(bx + bw / 2), int(by + bh / 2)
        cv2.drawMarker(out, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 15, 2)

    # Per confirmed track: colored dot, ID label, optional line
    for track in confirmed:
        color = COLORS_BGR[track.fish_id % len(COLORS_BGR)]
        centroid = list(track.positions)[-1]
        pt = torch.tensor(centroid, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pixels, valid = model.project(pt)

        if not valid[0]:
            continue
        pu, pv = int(pixels[0, 0].item()), int(pixels[0, 1].item())
        if not (0 <= pu < 1600 and 0 <= pv < 1200):
            continue

        # Reprojected centroid dot
        cv2.circle(out, (pu, pv), 8, color, -1)
        cv2.circle(out, (pu, pv), 8, (255, 255, 255), 2)

        # ID label
        label = f"ID{track.fish_id}"
        cv2.putText(
            out,
            label,
            (pu + 10, pv - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            out,
            label,
            (pu + 10, pv - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
        )

        # Line to matched bbox center if this camera claimed
        det_idx = track.camera_detections.get(cam_id)
        if det_idx is not None and det_idx < len(dets):
            det = dets[det_idx]
            bx, by, bw, bh = det.bbox
            cx, cy = int(bx + bw / 2), int(by + bh / 2)
            cv2.line(out, (pu, pv), (cx, cy), color, 2, cv2.LINE_AA)
            dist = ((pu - cx) ** 2 + (pv - cy) ** 2) ** 0.5
            mx, my = (pu + cx) // 2, (pv + cy) // 2
            cv2.putText(
                out,
                f"{dist:.0f}px",
                (mx + 5, my - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                out,
                f"{dist:.0f}px",
                (mx + 5, my - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    # Camera name + detection count
    n_det = len(dets)
    cv2.putText(
        out,
        cam_id,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
    )
    cv2.putText(
        out,
        cam_id,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        1,
    )
    cv2.putText(
        out,
        f"{n_det} dets",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading calibration and models...")
    calib = load_calibration_data(CALIB_PATH)
    models = build_models(calib)
    detector = make_detector("yolo", model_path=str(YOLO_WEIGHTS))

    print("Precomputing undistortion maps...")
    undist_maps = precompute_undistortion(calib)

    # Open videos
    print("Opening videos...")
    caps: dict[str, cv2.VideoCapture] = {}
    fps = None
    for name in calib.ring_cameras:
        if name == SKIP_CAMERA:
            continue
        matches = sorted(VIDEO_DIR.glob(f"{name}-*.mp4"))
        if not matches:
            print(f"  Warning: no video for {name}")
            continue
        cap = cv2.VideoCapture(str(matches[0]))
        if not cap.isOpened():
            print(f"  Warning: could not open {matches[0]}")
            continue
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
        caps[name] = cap
    if fps is None or fps <= 0:
        fps = 30.0

    cam_names = sorted(caps.keys())
    n_cams = len(cam_names)
    print(f"  Opened {n_cams} cameras, source FPS={fps:.1f}")

    # Skip to start frame
    print(f"Seeking to frame {FRAME_START}...")
    for cap in caps.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_START)

    # Initialize tracker
    tracker = FishTracker(min_hits=2, max_age=7, expected_count=9)

    # Grid dimensions
    grid_w = GRID_COLS * TILE_W
    grid_h = GRID_ROWS * TILE_H

    # Open video writer
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "tracking_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (grid_w, grid_h))
    if not writer.isOpened():
        print(f"ERROR: Could not open VideoWriter for {out_path}")
        return

    print(
        f"Writing {N_FRAMES} frames to {out_path} ({grid_w}x{grid_h} @ {fps:.1f} fps)"
    )

    # Frame loop
    for f in range(N_FRAMES):
        # Read and undistort
        frames: dict[str, np.ndarray] = {}
        for name in cam_names:
            ret, frame = caps[name].read()
            if ret:
                frames[name] = undistort_image(frame, undist_maps[name])

        if not frames:
            print(f"  End of video at frame {f}")
            break

        # Detect
        dets = {name: detector.detect(frame) for name, frame in frames.items()}

        # Track
        confirmed = tracker.update(dets, models, frame_index=f)

        # Annotate and tile
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for idx, cam_id in enumerate(cam_names):
            row, col = divmod(idx, GRID_COLS)
            if cam_id in frames:
                annotated = annotate_camera(
                    frames[cam_id],
                    cam_id,
                    dets.get(cam_id, []),
                    confirmed,
                    models[cam_id],
                )
                tile = cv2.resize(annotated, (TILE_W, TILE_H))
            else:
                tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
                cv2.putText(
                    tile,
                    f"{cam_id} (no frame)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
            grid[
                row * TILE_H : (row + 1) * TILE_H, col * TILE_W : (col + 1) * TILE_W
            ] = tile

        # Frame counter overlay
        abs_frame = FRAME_START + f
        text = f"Frame {abs_frame} ({f + 1}/{N_FRAMES})  Tracks: {len(confirmed)}"
        cv2.putText(
            grid,
            text,
            (10, grid_h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        writer.write(grid)

        if (f + 1) % 30 == 0:
            total_dets = sum(len(d) for d in dets.values())
            print(
                f"  Frame {f + 1}/{N_FRAMES}: {len(confirmed)} tracks, "
                f"{total_dets} detections"
            )

    # Cleanup
    writer.release()
    for cap in caps.values():
        cap.release()

    print(f"\nDone. Saved {out_path}")
    print(f"  Grid: {GRID_COLS}x{GRID_ROWS} = {grid_w}x{grid_h}")


if __name__ == "__main__":
    main()
