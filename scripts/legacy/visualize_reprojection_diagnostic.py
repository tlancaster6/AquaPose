"""Diagnostic: reprojected 3D centroids vs detection bbox centers across cameras.

For each tracked fish at the sample frame, shows a grid of camera crops
with the YOLO bbox, bbox center, and reprojected 3D centroid overlaid.
Annotates the pixel distance between them.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

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
TARGET_FRAME = 120  # absolute frame index to examine


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

    # Open videos and seek to target frame
    caps = {}
    undist_maps = {}
    for name in calib.ring_cameras:
        if name == SKIP_CAMERA:
            continue
        matches = sorted(VIDEO_DIR.glob(f"{name}-*.mp4"))
        if not matches:
            continue
        cap = cv2.VideoCapture(str(matches[0]))
        if cap.isOpened():
            caps[name] = cap
            undist_maps[name] = compute_undistortion_maps(calib.cameras[name])

    # Seek to target frame
    for cap in caps.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME)

    # Read and undistort
    frames = {}
    for name, cap in caps.items():
        ret, frame = cap.read()
        if ret:
            frames[name] = undistort_image(frame, undist_maps[name])
    for cap in caps.values():
        cap.release()

    print(f"Read frame {TARGET_FRAME} from {len(frames)} cameras")

    # Run detection
    dets = {name: detector.detect(frame) for name, frame in frames.items()}
    total = sum(len(d) for d in dets.values())
    print(f"Total detections: {total}")

    # Run tracking on a few frames up to target to get confirmed tracks
    # Re-open and run from a few frames before target
    warmup_start = max(0, TARGET_FRAME - 10)
    caps2 = {}
    for name in frames:
        matches = sorted(VIDEO_DIR.glob(f"{name}-*.mp4"))
        cap = cv2.VideoCapture(str(matches[0]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
        caps2[name] = cap

    tracker = FishTracker(min_hits=2, max_age=7, expected_count=9)
    confirmed = []
    for f in range(TARGET_FRAME - warmup_start + 1):
        fr = {}
        for name, cap in caps2.items():
            ret, frame = cap.read()
            if ret:
                fr[name] = undistort_image(frame, undist_maps[name])
        if not fr:
            break
        d = {name: detector.detect(frame) for name, frame in fr.items()}
        confirmed = tracker.update(d, models, frame_index=f)

    for cap in caps2.values():
        cap.release()

    print(f"Confirmed tracks at frame {TARGET_FRAME}: {len(confirmed)}")

    # Pick top fish by n_cameras (most visible)
    confirmed_sorted = sorted(confirmed, key=lambda t: t.n_cameras, reverse=True)
    fish_to_show = confirmed_sorted[: min(4, len(confirmed_sorted))]

    # For each fish, show crops from cameras that see it
    cam_names = sorted(frames.keys())

    for fish in fish_to_show:
        pos_3d = list(fish.positions)[-1]
        pt = torch.tensor(pos_3d, dtype=torch.float32).unsqueeze(0)

        # Project into all cameras
        projections = {}
        for cam_id in cam_names:
            model = models[cam_id]
            with torch.no_grad():
                pixels, valid = model.project(pt)
            if valid[0]:
                u, v = float(pixels[0, 0]), float(pixels[0, 1])
                if 0 <= u < 1600 and 0 <= v < 1200:
                    projections[cam_id] = (u, v)

        # Find cameras with detections near the projection
        cam_with_det = []
        for cam_id, (pu, pv) in projections.items():
            for det in dets.get(cam_id, []):
                bx, by, bw, bh = det.bbox
                cx, cy = bx + bw / 2, by + bh / 2
                dist = ((pu - cx) ** 2 + (pv - cy) ** 2) ** 0.5
                if dist < 100:  # generous for visualization
                    cam_with_det.append((cam_id, det, pu, pv, cx, cy, dist))
                    break

        if not cam_with_det:
            continue

        n_show = min(6, len(cam_with_det))
        fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
        if n_show == 1:
            axes = [axes]
        fig.suptitle(
            f"Fish ID {fish.fish_id} — 3D centroid: ({pos_3d[0]:.3f}, {pos_3d[1]:.3f}, {pos_3d[2]:.3f})m — "
            f"{fish.n_cameras} cameras",
            fontsize=12,
            fontweight="bold",
        )

        for ax, (cam_id, det, pu, pv, cx, cy, dist) in zip(
            axes, cam_with_det[:n_show], strict=False
        ):
            bx, by, bw, bh = det.bbox
            # Crop region with padding
            pad = 80
            x0 = max(0, int(min(pu, cx) - bw / 2 - pad))
            y0 = max(0, int(min(pv, cy) - bh / 2 - pad))
            x1 = min(1600, int(max(pu, cx) + bw / 2 + pad))
            y1 = min(1200, int(max(pv, cy) + bh / 2 + pad))

            crop = frames[cam_id][y0:y1, x0:x1]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ax.imshow(crop_rgb)

            # Draw bbox (shifted to crop coords)
            rect = Rectangle(
                (bx - x0, by - y0),
                bw,
                bh,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)

            # Bbox center (blue cross)
            ax.plot(
                cx - x0,
                cy - y0,
                "x",
                color="cyan",
                markersize=12,
                markeredgewidth=2,
                label="bbox center",
            )

            # Reprojected 3D centroid (red dot)
            ax.plot(
                pu - x0,
                pv - y0,
                "o",
                color="red",
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label="reprojected 3D",
            )

            # Draw line between them
            ax.plot(
                [cx - x0, pu - x0],
                [cy - y0, pv - y0],
                "--",
                color="yellow",
                linewidth=1.5,
                alpha=0.8,
            )

            ax.set_title(f"{cam_id}\n{dist:.1f}px offset", fontsize=10)
            ax.axis("off")

        axes[0].legend(loc="lower left", fontsize=8, framealpha=0.8)
        plt.tight_layout()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / f"reprojection_fish_{fish.fish_id}.png"
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")
        plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
