"""Visualize cross-view tracking on real multi-camera video data.

Loads calibration, YOLO detector, and synchronized video from the 12 ring
cameras. Runs RANSAC association + Hungarian tracking over N frames and
produces a 4-panel visualization:
  1. 3D trajectories (XY bird's-eye)
  2. Sample frame with reprojected centroids overlaid
  3. Fish ID stability over time
  4. Reprojection residuals and camera counts
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from aquapose.calibration.loader import (
    CalibrationData,
    compute_undistortion_maps,
    load_calibration_data,
    undistort_image,
)
from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.segmentation.detector import Detection, make_detector
from aquapose.tracking import FishTracker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT = Path("C:/Users/tucke/Desktop/Aqua/AquaPose")
CALIB_PATH = DATA_ROOT / "calibration.json"
VIDEO_DIR = DATA_ROOT / "videos" / "core_videos"
YOLO_WEIGHTS = DATA_ROOT / "yolo" / "train_v2" / "weights" / "best.pt"
OUTPUT_DIR = Path("output")

N_FRAMES = 60  # frames to process
FRAME_START = 90  # skip first N frames (let YOLO warm up on clear water)
SKIP_CAMERA = "e3v8250"  # center top-down camera (poor quality)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def build_projection_models(
    calib: CalibrationData,
) -> dict[str, RefractiveProjectionModel]:
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


def open_videos(
    calib: CalibrationData,
) -> dict[str, cv2.VideoCapture]:
    """Open one VideoCapture per ring camera."""
    caps = {}
    for name in calib.ring_cameras:
        if name == SKIP_CAMERA:
            continue
        pattern = f"{name}-*.mp4"
        matches = sorted(VIDEO_DIR.glob(pattern))
        if not matches:
            print(f"  Warning: no video for {name}, skipping")
            continue
        cap = cv2.VideoCapture(str(matches[0]))
        if cap.isOpened():
            caps[name] = cap
        else:
            print(f"  Warning: could not open {matches[0]}")
    return caps


def read_synced_frames(
    caps: dict[str, cv2.VideoCapture],
    calib: CalibrationData,
) -> dict[str, np.ndarray] | None:
    """Read one frame from each camera and undistort."""
    frames = {}
    for name, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            return None
        cam = calib.cameras[name]
        undist = compute_undistortion_maps(cam)
        frames[name] = undistort_image(frame, undist)
    return frames


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_all_cameras(
    frames: dict[str, np.ndarray],
    detector,
) -> dict[str, list[Detection]]:
    """Run YOLO detection on each camera frame."""
    dets = {}
    for name, frame in frames.items():
        dets[name] = detector.detect(frame)
    return dets


# ---------------------------------------------------------------------------
# Run tracking
# ---------------------------------------------------------------------------


def run_pipeline(
    calib: CalibrationData,
    models: dict[str, RefractiveProjectionModel],
    caps: dict[str, cv2.VideoCapture],
    detector,
    n_frames: int,
    frame_start: int,
) -> dict:
    """Run full detection + association + tracking pipeline."""
    tracker = FishTracker(min_hits=2, max_age=7, expected_count=9)

    # Storage
    all_positions: list[dict[int, np.ndarray]] = []  # frame -> {fish_id: pos}
    all_residuals: list[dict[int, float]] = []
    all_n_cameras: list[dict[int, int]] = []
    all_det_counts: list[dict[str, int]] = []
    sample_frame_data = None

    # Skip to start frame
    for _ in range(frame_start):
        for cap in caps.values():
            cap.read()

    for f in range(n_frames):
        frames = read_synced_frames(caps, calib)
        if frames is None:
            print(f"  End of video at frame {f}")
            break

        dets = detect_all_cameras(frames, detector)
        det_counts = {cam: len(d) for cam, d in dets.items()}
        all_det_counts.append(det_counts)

        confirmed = tracker.update(dets, models, frame_index=f)

        positions = {}
        residuals = {}
        n_cams = {}
        for track in confirmed:
            pos = list(track.positions)[-1]
            positions[track.fish_id] = pos
            residuals[track.fish_id] = track.reprojection_residual
            n_cams[track.fish_id] = track.n_cameras

        all_positions.append(positions)
        all_residuals.append(residuals)
        all_n_cameras.append(n_cams)

        # Save a sample frame + associations for overlay visualization
        # Snapshot positions now — tracks are mutable and will change later
        if f == n_frames // 2:
            sample_frame_data = {
                "frames": {k: v.copy() for k, v in frames.items()},
                "dets": dets,
                "confirmed_snapshot": [
                    {"fish_id": t.fish_id, "pos": list(t.positions)[-1].copy()}
                    for t in confirmed
                ],
            }

        if (f + 1) % 10 == 0:
            print(
                f"  Frame {f + 1}/{n_frames}: {len(confirmed)} confirmed tracks, "
                f"{sum(det_counts.values())} total detections"
            )

    return {
        "positions": all_positions,
        "residuals": all_residuals,
        "n_cameras": all_n_cameras,
        "det_counts": all_det_counts,
        "sample": sample_frame_data,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_results(result: dict, models: dict[str, RefractiveProjectionModel]) -> None:
    n_frames = len(result["positions"])

    # Collect all fish IDs
    all_ids = set()
    for frame_pos in result["positions"]:
        all_ids.update(frame_pos.keys())
    all_ids = sorted(all_ids)
    n_ids = max(len(all_ids), 1)
    id_colors = {fid: plt.cm.tab20(i / n_ids) for i, fid in enumerate(all_ids)}

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Cross-View Tracking — Real Data — {n_frames} frames, "
        f"{len(models)} cameras, {len(all_ids)} fish IDs",
        fontsize=14,
        fontweight="bold",
    )

    # --- Panel 1: 3D trajectories (XY bird's-eye) ---
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("3D Centroids (XY bird's-eye)")
    for fid in all_ids:
        xs, ys = [], []
        for frame_pos in result["positions"]:
            if fid in frame_pos:
                pos = frame_pos[fid]
                xs.append(pos[0])
                ys.append(pos[1])
            else:
                # gap — plot segment so far, start new
                if xs:
                    ax1.plot(xs, ys, color=id_colors[fid], alpha=0.7, lw=1.2)
                    ax1.scatter(
                        xs[0],
                        ys[0],
                        color=id_colors[fid],
                        s=30,
                        zorder=5,
                        edgecolors="k",
                        linewidths=0.5,
                    )
                xs, ys = [], []
        if xs:
            ax1.plot(xs, ys, color=id_colors[fid], alpha=0.7, lw=1.2)
            ax1.scatter(
                xs[0],
                ys[0],
                color=id_colors[fid],
                s=30,
                zorder=5,
                edgecolors="k",
                linewidths=0.5,
            )
            ax1.scatter(
                xs[-1],
                ys[-1],
                color=id_colors[fid],
                s=30,
                marker="s",
                zorder=5,
                edgecolors="k",
                linewidths=0.5,
            )
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Sample frame with reprojected centroids ---
    ax2 = fig.add_subplot(2, 2, 2)
    sample = result.get("sample")
    if sample is not None:
        # Pick the camera with most detections for display
        cam_name = max(sample["dets"], key=lambda c: len(sample["dets"][c]))
        frame_bgr = sample["frames"][cam_name]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ax2.imshow(frame_rgb)
        ax2.set_title(f"Camera {cam_name} — mid-sequence")

        # Draw YOLO detection boxes
        for det in sample["dets"][cam_name]:
            x, y, w, h = det.bbox
            rect = Rectangle(
                (x, y), w, h, linewidth=1, edgecolor="lime", facecolor="none", alpha=0.5
            )
            ax2.add_patch(rect)

        # Reproject snapshotted 3D centroids into this camera
        model = models[cam_name]
        for snap in sample["confirmed_snapshot"]:
            pt = torch.tensor(snap["pos"], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pixels, valid = model.project(pt)
            if valid[0]:
                u, v = float(pixels[0, 0]), float(pixels[0, 1])
                c = id_colors.get(snap["fish_id"], "white")
                ax2.plot(
                    u,
                    v,
                    "o",
                    color=c,
                    markersize=8,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    zorder=10,
                )
                ax2.annotate(
                    f"ID {snap['fish_id']}",
                    (u, v),
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                    xytext=(5, -10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
                )
        ax2.axis("off")
    else:
        ax2.text(
            0.5,
            0.5,
            "No sample frame",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    # --- Panel 3: ID stability over time ---
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("Fish ID Assignment Over Time")
    for f, frame_pos in enumerate(result["positions"]):
        for fid in frame_pos:
            c = id_colors.get(fid, "gray")
            ax3.scatter(f, fid, color=c, s=8, alpha=0.7)
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Fish ID")
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Residuals and camera count ---
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("Reprojection Residual & Camera Count")

    mean_res = []
    mean_cams = []
    for f in range(n_frames):
        res_vals = list(result["residuals"][f].values())
        cam_vals = list(result["n_cameras"][f].values())
        mean_res.append(np.mean(res_vals) if res_vals else np.nan)
        mean_cams.append(np.mean(cam_vals) if cam_vals else np.nan)

    ax4.plot(mean_res, color="tab:red", lw=1.5, label="Mean residual (px)")
    ax4.set_xlabel("Frame")
    ax4.set_ylabel("Residual (px)", color="tab:red")
    ax4.tick_params(axis="y", labelcolor="tab:red")

    ax4b = ax4.twinx()
    ax4b.plot(mean_cams, color="tab:blue", lw=1.5, alpha=0.7, label="Mean cameras")
    ax4b.set_ylabel("Cameras per fish", color="tab:blue")
    ax4b.tick_params(axis="y", labelcolor="tab:blue")

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "tracking_real_data.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def print_metrics(result: dict) -> None:
    n_frames = len(result["positions"])
    all_ids = set()
    for fp in result["positions"]:
        all_ids.update(fp.keys())

    total_dets = sum(sum(dc.values()) for dc in result["det_counts"])
    all_res = [r for fr in result["residuals"] for r in fr.values()]
    all_cams = [c for fc in result["n_cameras"] for c in fc.values()]

    print("\n--- Real-Data Tracking Metrics ---")
    print(f"Frames processed: {n_frames}")
    print(f"Unique fish IDs:  {len(all_ids)}")
    print(f"Total detections: {total_dets}")
    print(
        f"Mean detections/cam/frame: {total_dets / n_frames / len(result['det_counts'][0]):.1f}"
    )
    if all_res:
        print(f"Reprojection residual mean: {np.mean(all_res):.2f} px")
        print(f"Reprojection residual p95:  {np.percentile(all_res, 95):.2f} px")
    if all_cams:
        print(f"Mean cameras per fish: {np.mean(all_cams):.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading calibration...")
    calib = load_calibration_data(CALIB_PATH)
    ring_cams = [c for c in calib.ring_cameras if c != SKIP_CAMERA]
    print(f"  {len(ring_cams)} ring cameras: {ring_cams}")

    print("Building refractive projection models...")
    models = build_projection_models(calib)

    print("Opening videos...")
    caps = open_videos(calib)
    print(f"  Opened {len(caps)} video streams")

    print(f"Loading YOLO detector from {YOLO_WEIGHTS}...")
    detector = make_detector("yolo", model_path=str(YOLO_WEIGHTS))

    print(
        f"\nRunning tracking pipeline ({N_FRAMES} frames, starting at {FRAME_START})..."
    )
    result = run_pipeline(calib, models, caps, detector, N_FRAMES, FRAME_START)

    # Cleanup
    for cap in caps.values():
        cap.release()

    print_metrics(result)
    plot_results(result, models)


if __name__ == "__main__":
    main()
