"""Visualize a single per-camera tracklet across all chunks.

Renders sampled frames from a specific camera showing the target track's
keypoints and bbox, plus any other tracks in that camera for context.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    run_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("~/aquapose/projects/YH/runs/run_20260318_114822").expanduser()
    )
    target_cam = sys.argv[2] if len(sys.argv) > 2 else "e3v8334"
    target_tid = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    n_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 30

    # Load chunk caches
    from aquapose.core.context import load_chunk_cache
    from aquapose.core.types.frame_source import VideoFrameSource
    from aquapose.evaluation.viz._loader import read_config_yaml

    diag_dir = run_dir / "diagnostics"
    manifest = json.loads((diag_dir / "manifest.json").read_text())
    chunk_entries = sorted(manifest["chunks"], key=lambda c: c["index"])

    # Collect tracklet data for target camera across all chunks
    # frames -> (kpts, kconf, bbox, status, track_id)
    all_track_frames: dict[
        int, list[tuple[np.ndarray, np.ndarray, tuple, str, int]]
    ] = {}

    for chunk_entry in chunk_entries:
        idx = chunk_entry["index"]
        chunk_start = chunk_entry["start_frame"]
        cache_path = diag_dir / f"chunk_{idx:03d}" / "cache.pkl"
        if not cache_path.exists():
            continue
        ctx = load_chunk_cache(cache_path)
        tracks_2d = ctx.tracks_2d
        if tracks_2d is None:
            continue
        cam_tracklets = tracks_2d.get(target_cam, [])
        for t in cam_tracklets:
            for i, frame_idx in enumerate(t.frames):
                global_frame = chunk_start + frame_idx
                kpts = t.keypoints[i] if t.keypoints is not None else None
                kconf = t.keypoint_conf[i] if t.keypoint_conf is not None else None
                bbox = t.bboxes[i]
                status = t.frame_status[i]
                if global_frame not in all_track_frames:
                    all_track_frames[global_frame] = []
                all_track_frames[global_frame].append(
                    (kpts, kconf, bbox, status, t.track_id)
                )

    # Find frames where target track appears
    target_frames = sorted(
        f
        for f, entries in all_track_frames.items()
        if any(e[4] == target_tid for e in entries)
    )
    print(
        f"Track {target_tid} in {target_cam}: {len(target_frames)} frames, "
        f"range [{target_frames[0]}-{target_frames[-1]}]"
    )

    # Also show frame_status breakdown
    statuses = {}
    for f in target_frames:
        for e in all_track_frames[f]:
            if e[4] == target_tid:
                s = e[3]
                statuses[s] = statuses.get(s, 0) + 1
    print(f"  Status breakdown: {statuses}")

    # Sample frames evenly across the target track's lifetime
    if n_samples >= len(target_frames):
        sample_frames = target_frames
    else:
        sample_frames = [
            target_frames[round(i * (len(target_frames) - 1) / (n_samples - 1))]
            for i in range(n_samples)
        ]

    # Open video source
    config_yaml = read_config_yaml(run_dir)
    video_dir_str = config_yaml["video_dir"]
    calibration_path_str = config_yaml["calibration_path"]
    project_dir = config_yaml.get("project_dir", "")
    video_path = Path(video_dir_str)
    calibration_path = Path(calibration_path_str)
    if not video_path.is_absolute() and project_dir:
        video_path = Path(project_dir) / video_path
    if not calibration_path.is_absolute() and project_dir:
        calibration_path = Path(project_dir) / calibration_path

    frame_source = VideoFrameSource(
        video_dir=video_path, calibration_path=calibration_path
    )

    out_dir = run_dir / "viz" / f"track_{target_cam}_{target_tid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Color scheme
    TARGET_COLOR = (0, 255, 0)  # Green for target track
    OTHER_COLOR = (128, 128, 128)  # Gray for other tracks
    COASTED_COLOR = (0, 165, 255)  # Orange for coasted frames

    with frame_source:
        for frame_idx in sample_frames:
            frames = frame_source.read_frame(frame_idx)
            img = frames.get(target_cam)
            if img is None:
                continue
            canvas = img.copy()

            entries = all_track_frames.get(frame_idx, [])
            for kpts, _kconf, bbox, status, tid in entries:
                is_target = tid == target_tid
                if is_target and status == "coasted":
                    color = COASTED_COLOR
                elif is_target:
                    color = TARGET_COLOR
                else:
                    color = OTHER_COLOR
                thickness = 2 if is_target else 1

                # Draw bbox
                x, y, w, h = bbox
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)

                # Draw keypoints
                if kpts is not None:
                    for ki in range(kpts.shape[0]):
                        px, py = int(kpts[ki, 0]), int(kpts[ki, 1])
                        r = 4 if is_target else 2
                        cv2.circle(canvas, (px, py), r, color, -1)
                    # Draw spine line
                    pts = kpts.astype(np.int32)
                    for ki in range(len(pts) - 1):
                        cv2.line(
                            canvas, tuple(pts[ki]), tuple(pts[ki + 1]), color, thickness
                        )

                # Label
                label = f"T{tid}" + (" [coast]" if status == "coasted" else "")
                cv2.putText(
                    canvas,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            # Frame label
            cv2.putText(
                canvas,
                f"{target_cam} frame={frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            out_path = out_dir / f"frame_{frame_idx:04d}.png"
            cv2.imwrite(str(out_path), canvas)

    print(f"Wrote {len(sample_frames)} frames to {out_dir}")

    # Also generate a mosaic summary
    images = sorted(out_dir.glob("frame_*.png"))
    if images:
        cols = min(6, len(images))
        rows = math.ceil(len(images) / cols)
        cell_w, cell_h = 480, 360
        mosaic = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        for i, img_path in enumerate(images):
            r, c = divmod(i, cols)
            img = cv2.imread(str(img_path))
            resized = cv2.resize(img, (cell_w, cell_h))
            mosaic[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = (
                resized
            )

        mosaic_path = out_dir / "mosaic.png"
        cv2.imwrite(str(mosaic_path), mosaic)
        print(f"Mosaic: {mosaic_path}")


if __name__ == "__main__":
    main()
