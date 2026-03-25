"""Training data miner for ReID contrastive fine-tuning.

Extracts high-confidence OBB-aligned crops from completed pipeline runs,
organized into temporal groupings where fish identity labels are locally
consistent.  Output structure: ``reid_crops/group_NNN/fish_N/*.jpg`` with
per-grouping manifest JSON files.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import h5py
import numpy as np
import yaml

from aquapose.core.context import StaleCacheError, load_chunk_cache
from aquapose.core.pose.crop import extract_affine_crop

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MinerConfig:
    """Configuration for :class:`TrainingDataMiner`.

    All fields have sensible defaults; override via CLI options.

    Attributes:
        window_size: Frames per temporal window (~10 s at 30 fps).
        window_stride: Stride between windows (50 % overlap by default).
        min_cooccurring: Minimum fish per accepted grouping.
        min_cameras: Minimum cameras for the quality gate.
        max_residual: Maximum mean residual (px) for the quality gate.
        min_duration: Minimum contiguous frames per segment.
        crops_per_fish: Target crops per fish per grouping.
        crop_size: Crop side length (px); matches MegaDescriptor-T input.
    """

    window_size: int = 300
    window_stride: int = 150
    min_cooccurring: int = 3
    min_cameras: int = 3
    max_residual: float = 5.0
    min_duration: int = 10
    crops_per_fish: int = 8
    crop_size: int = 224


# ---------------------------------------------------------------------------
# Pure helper functions (tested independently)
# ---------------------------------------------------------------------------


def _frame_passes_quality(
    n_cameras: int,
    mean_residual: float,
    is_low_confidence: bool,
    min_cameras: int,
    max_residual: float,
) -> bool:
    """Check whether a single frame passes the quality gate.

    Args:
        n_cameras: Number of cameras that observed the fish.
        mean_residual: Mean reprojection residual in pixels.
            A value of -1.0 indicates unknown (H5 fillvalue) and passes.
        is_low_confidence: Whether the detection was low-confidence.
        min_cameras: Minimum camera count threshold.
        max_residual: Maximum allowed mean residual.

    Returns:
        ``True`` if the frame passes all quality gates.
    """
    if n_cameras < min_cameras:
        return False
    if mean_residual >= 0 and mean_residual > max_residual:
        return False
    return not is_low_confidence


def _find_contiguous_segments(frames: list[int]) -> list[list[int]]:
    """Split a list of frame indices into contiguous runs.

    A gap of more than 1 between consecutive (sorted) frames starts a new
    segment.

    Args:
        frames: Frame indices (need not be sorted).

    Returns:
        List of segments, each a sorted list of contiguous frame indices.
    """
    if not frames:
        return []
    arr = np.array(sorted(frames))
    gaps = np.where(np.diff(arr) > 1)[0]
    segments: list[list[int]] = []
    prev = 0
    for g in gaps:
        segments.append(arr[prev : g + 1].tolist())
        prev = g + 1
    segments.append(arr[prev:].tolist())
    return segments


def _camera_aware_sample(
    detections: list[tuple[int, str, Any]],
    crops_per_fish: int,
) -> list[tuple[int, str, Any]]:
    """Sample detections with camera diversity via round-robin.

    Interleaves detections across cameras, then uniformly sub-samples to
    the target count.

    Args:
        detections: List of ``(frame, camera_id, detection)`` tuples.
        crops_per_fish: Target number of samples.

    Returns:
        Sampled subset with balanced camera representation.
    """
    if not detections:
        return []

    by_camera: dict[str, list[tuple[int, str, Any]]] = defaultdict(list)
    for frame, cam, det in detections:
        by_camera[cam].append((frame, cam, det))

    # Round-robin interleave across cameras
    cameras = sorted(by_camera.keys())
    max_len = max(len(v) for v in by_camera.values())
    interleaved: list[tuple[int, str, Any]] = []
    for i in range(max_len):
        for cam in cameras:
            if i < len(by_camera[cam]):
                interleaved.append(by_camera[cam][i])

    if len(interleaved) <= crops_per_fish:
        return interleaved

    indices = np.linspace(0, len(interleaved) - 1, crops_per_fish, dtype=int)
    return [interleaved[i] for i in indices]


# ---------------------------------------------------------------------------
# Main miner class
# ---------------------------------------------------------------------------


class TrainingDataMiner:
    """Mine high-confidence trajectory segments for ReID fine-tuning.

    Reads ``midlines_stitched.h5`` (or ``midlines.h5``) and per-chunk
    ``cache.pkl`` files from a completed pipeline run.  Slides temporal
    windows across the video, applies quality gates, and extracts
    OBB-aligned crops organized into groupings suitable for contrastive
    learning.

    Args:
        run_dir: Path to a completed pipeline run directory.
        config: Mining configuration.  Defaults to :class:`MinerConfig`
            with default parameters.
    """

    def __init__(
        self,
        run_dir: Path,
        config: MinerConfig | None = None,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._config = config or MinerConfig()

        # Validate run directory has required inputs
        diag_dir = self._run_dir / "diagnostics"
        if not diag_dir.exists():
            raise FileNotFoundError(
                "Mining requires a diagnostic-mode run. "
                "Re-run the pipeline with --mode diagnostic."
            )

        self._h5_path = self._run_dir / "midlines_stitched.h5"
        if not self._h5_path.exists():
            self._h5_path = self._run_dir / "midlines.h5"
            if self._h5_path.exists():
                logger.warning(
                    "midlines_stitched.h5 not found; falling back to midlines.h5"
                )
        if not self._h5_path.exists():
            raise FileNotFoundError(
                f"No midlines H5 file found in {self._run_dir}. "
                "Run the full pipeline first."
            )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _load_h5_quality(self) -> dict[int, dict[int, dict[str, Any]]]:
        """Load quality fields from H5, indexed by fish_id -> frame -> metrics.

        Returns:
            Nested dict: ``{fish_id: {frame: {n_cameras, mean_residual,
            is_low_confidence}}}``.
        """
        with h5py.File(self._h5_path, "r") as f:
            grp = cast(h5py.Group, f["midlines"])
            frame_index = cast(h5py.Dataset, grp["frame_index"])[()]
            fish_id_arr = cast(h5py.Dataset, grp["fish_id"])[()]
            n_cameras_arr = cast(h5py.Dataset, grp["n_cameras"])[()]
            mean_residual_arr = cast(h5py.Dataset, grp["mean_residual"])[()]
            is_low_conf_arr = cast(h5py.Dataset, grp["is_low_confidence"])[()]

        quality: dict[int, dict[int, dict[str, Any]]] = {}
        n_rows = frame_index.shape[0]

        if fish_id_arr.ndim == 1:
            for row in range(n_rows):
                fid = int(fish_id_arr[row])
                if fid < 0:
                    continue
                frame = int(frame_index[row])
                quality.setdefault(fid, {})[frame] = {
                    "n_cameras": int(n_cameras_arr[row]),
                    "mean_residual": float(mean_residual_arr[row]),
                    "is_low_confidence": bool(is_low_conf_arr[row]),
                }
        else:
            max_fish = fish_id_arr.shape[1]
            for row in range(n_rows):
                frame = int(frame_index[row])
                for slot in range(max_fish):
                    fid = int(fish_id_arr[row, slot])
                    if fid < 0:
                        continue
                    quality.setdefault(fid, {})[frame] = {
                        "n_cameras": int(n_cameras_arr[row, slot]),
                        "mean_residual": float(mean_residual_arr[row, slot]),
                        "is_low_confidence": bool(is_low_conf_arr[row, slot]),
                    }

        return quality

    def _build_prestitch_map(self) -> dict[int, int]:
        """Map pre-stitch (chunk-local) fish IDs to stitched global IDs.

        Compares 3D centroids between ``midlines.h5`` (pre-stitch) and
        ``midlines_stitched.h5`` (post-stitch) to match fragment IDs.

        Returns:
            Dict mapping pre-stitch fish_id to stitched fish_id.
        """
        pre_path = self._run_dir / "midlines.h5"
        post_path = self._run_dir / "midlines_stitched.h5"

        if not pre_path.exists() or not post_path.exists():
            logger.warning(
                "Cannot build ID mapping — midlines.h5 or "
                "midlines_stitched.h5 missing. Using raw cache IDs."
            )
            return {}

        with h5py.File(pre_path, "r") as pre, h5py.File(post_path, "r") as post:
            fid_pre = cast(h5py.Dataset, pre["midlines/fish_id"])[:]
            fid_post = cast(h5py.Dataset, post["midlines/fish_id"])[:]
            pts_pre = cast(h5py.Dataset, pre["midlines/points"])[:]
            pts_post = cast(h5py.Dataset, post["midlines/points"])[:]

        mapping: dict[int, int] = {}
        n_frames = fid_pre.shape[0]
        mid_kpt = pts_pre.shape[2] // 2

        for fr in range(n_frames):
            for slot in range(fid_pre.shape[1]):
                pid = int(fid_pre[fr, slot])
                if pid < 0 or pid in mapping:
                    continue

                centroid_pre = pts_pre[fr, slot, mid_kpt, :]
                if np.any(np.isnan(centroid_pre)):
                    continue

                best_dist = np.inf
                best_post_id = -1
                for s2 in range(fid_post.shape[1]):
                    if fid_post[fr, s2] < 0:
                        continue
                    centroid_post = pts_post[fr, s2, mid_kpt, :]
                    if np.any(np.isnan(centroid_post)):
                        continue
                    d = float(np.linalg.norm(centroid_pre - centroid_post))
                    if d < best_dist:
                        best_dist = d
                        best_post_id = int(fid_post[fr, s2])

                if best_dist < 0.01:
                    mapping[pid] = best_post_id

        logger.info(
            "Pre->post stitch mapping: %d fragment IDs -> %d stitched IDs",
            len(mapping),
            len(set(mapping.values())),
        )
        return mapping

    def _load_run_config(self) -> dict[str, Any]:
        """Load the run's ``config.yaml``."""
        config_path = self._run_dir / "config.yaml"
        if config_path.exists():
            with config_path.open() as fh:
                return yaml.safe_load(fh) or {}
        return {}

    def _discover_chunk_caches(self) -> list[Path]:
        """Find and sort chunk cache files."""
        diag_dir = self._run_dir / "diagnostics"
        return sorted(diag_dir.glob("chunk_*/cache.pkl"))

    @staticmethod
    def _parse_chunk_number(cache_path: Path) -> int:
        """Extract chunk number from ``chunk_NNN/cache.pkl`` path."""
        match = re.search(r"chunk_(\d+)", cache_path.parent.name)
        if match:
            return int(match.group(1))
        raise ValueError(f"Cannot parse chunk number from {cache_path}")

    def _build_detection_map(
        self,
        ctx: Any,
        chunk_start: int,
        valid_frames: set[int],
        prestitch_map: dict[int, int] | None = None,
    ) -> dict[tuple[int, str], tuple[int, Any]]:
        """Build ``(global_frame, camera_id) -> (fish_id, Detection)`` map.

        Reuses the same closest-centroid matching pattern as
        :class:`~aquapose.core.reid.runner.EmbedRunner`.

        Args:
            ctx: Loaded pipeline context from ``cache.pkl``.
            chunk_start: Global frame offset for this chunk.
            valid_frames: Set of global frames with H5 quality data.
            prestitch_map: Maps pre-stitch (chunk-local) fish IDs to
                post-stitch (global) fish IDs.  When provided, cache
                fish IDs are remapped before use.

        Returns:
            Detection map keyed by ``(global_frame, camera_id)``.
        """
        detection_map: dict[tuple[int, str], tuple[int, Any]] = {}

        if ctx.tracklet_groups is None or ctx.detections is None:
            return detection_map

        for group in ctx.tracklet_groups:
            raw_id = group.fish_id
            if prestitch_map:
                fish_id = prestitch_map.get(raw_id, -1)
                if fish_id < 0:
                    continue
            else:
                fish_id = raw_id
            for tracklet in group.tracklets:
                cam_id = tracklet.camera_id
                for i, local_frame in enumerate(tracklet.frames):
                    if tracklet.frame_status[i] == "coasted":
                        continue

                    global_frame = chunk_start + local_frame
                    if global_frame not in valid_frames:
                        continue

                    t_centroid = tracklet.centroids[i]
                    frame_dets = ctx.detections[local_frame].get(cam_id, [])
                    if not frame_dets:
                        continue

                    best_det = None
                    best_dist = float("inf")
                    for det in frame_dets:
                        dx = (det.bbox[0] + det.bbox[2] / 2) - t_centroid[0]
                        dy = (det.bbox[1] + det.bbox[3] / 2) - t_centroid[1]
                        dist = dx * dx + dy * dy
                        if dist < best_dist:
                            best_dist = dist
                            best_det = det

                    if best_det is not None:
                        detection_map[(global_frame, cam_id)] = (
                            fish_id,
                            best_det,
                        )

        return detection_map

    def _select_grouping_windows(
        self,
        quality_data: dict[int, dict[int, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Slide temporal windows and select those with enough cooccurring fish.

        Returns:
            List of accepted grouping dicts, each containing ``window_start``,
            ``window_end``, and ``fish_segments`` (fish_id -> list of valid
            contiguous segments).
        """
        cfg = self._config
        all_frames: set[int] = set()
        for frame_data in quality_data.values():
            all_frames.update(frame_data.keys())

        if not all_frames:
            raise RuntimeError(
                "No frames found in H5 quality data. "
                "Ensure the pipeline produced midline reconstructions."
            )

        video_start = min(all_frames)
        video_end = max(all_frames) + 1

        accepted: list[dict[str, Any]] = []
        n_evaluated = 0

        for window_start in range(video_start, video_end, cfg.window_stride):
            window_end = window_start + cfg.window_size
            n_evaluated += 1

            fish_segments: dict[int, list[list[int]]] = {}
            fish_with_valid = 0

            for fid, frame_data in quality_data.items():
                valid_frames_in_window: list[int] = []
                for frame, qdata in frame_data.items():
                    if window_start <= frame < window_end and _frame_passes_quality(
                        qdata["n_cameras"],
                        qdata["mean_residual"],
                        qdata["is_low_confidence"],
                        cfg.min_cameras,
                        cfg.max_residual,
                    ):
                        valid_frames_in_window.append(frame)

                segments = _find_contiguous_segments(valid_frames_in_window)
                long_segments = [s for s in segments if len(s) >= cfg.min_duration]

                if long_segments:
                    fish_segments[fid] = long_segments
                    fish_with_valid += 1

            if fish_with_valid >= cfg.min_cooccurring:
                accepted.append(
                    {
                        "window_start": window_start,
                        "window_end": window_end,
                        "fish_segments": fish_segments,
                    }
                )

        logger.info(
            "Window selection: %d accepted / %d evaluated",
            len(accepted),
            n_evaluated,
        )

        return accepted

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the full mining pipeline.

        Returns:
            Stats dict with ``n_groups``, ``n_crops``, and
            ``per_fish_counts``.

        Raises:
            RuntimeError: If ALL fish have zero valid segments.
        """
        cfg = self._config
        output_dir = self._run_dir / "reid_crops"

        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"{output_dir} already exists and is non-empty. "
                "Delete it manually or re-run with --overwrite."
            )

        # Step 1: Load H5 quality data
        logger.info("Loading H5 quality data from %s", self._h5_path)
        quality_data = self._load_h5_quality()
        logger.info("Found %d fish in H5 data", len(quality_data))

        # Log per-fish total frames
        for fid in sorted(quality_data):
            logger.info("  Fish %d: %d total frames", fid, len(quality_data[fid]))

        # Check for fish with zero valid frames and log funnel
        fish_valid_counts: dict[int, int] = {}
        for fid, frame_data in quality_data.items():
            count = sum(
                1
                for qdata in frame_data.values()
                if _frame_passes_quality(
                    qdata["n_cameras"],
                    qdata["mean_residual"],
                    qdata["is_low_confidence"],
                    cfg.min_cameras,
                    cfg.max_residual,
                )
            )
            fish_valid_counts[fid] = count
            logger.info(
                "  Fish %d: %d / %d frames passed quality gates",
                fid,
                count,
                len(frame_data),
            )

        fish_with_zero = [fid for fid, count in fish_valid_counts.items() if count == 0]
        if len(fish_with_zero) == len(quality_data):
            raise RuntimeError(
                f"ALL {len(quality_data)} fish have zero valid frames after "
                f"quality filtering (min_cameras={cfg.min_cameras}, "
                f"max_residual={cfg.max_residual}). "
                "Check your quality gate thresholds."
            )
        if fish_with_zero:
            logger.warning(
                "Fish with zero valid frames (will be skipped): %s",
                fish_with_zero,
            )

        # Step 2: Select grouping windows
        groupings = self._select_grouping_windows(quality_data)
        if not groupings:
            raise RuntimeError(
                "No windows accepted as groupings. "
                f"Need {cfg.min_cooccurring}+ fish with segments >= "
                f"{cfg.min_duration} frames in a {cfg.window_size}-frame window."
            )

        # Step 3: Load run config
        run_config = self._load_run_config()
        chunk_size = run_config.get("chunk_size", 300) or 300
        video_dir = run_config.get("video_dir", "")
        calibration_path = run_config.get("calibration_path", "")

        if not video_dir or not calibration_path:
            raise ValueError(
                "Run config.yaml must contain video_dir and calibration_path"
            )

        # Step 4: Build pre-stitch -> stitched ID mapping
        prestitch_map = self._build_prestitch_map()

        # Step 5: Discover chunk caches and build global detection map
        chunk_caches = self._discover_chunk_caches()
        if not chunk_caches:
            raise FileNotFoundError(
                f"No chunk caches found in {self._run_dir / 'diagnostics'}"
            )

        # Collect all valid frames across all groupings
        all_valid_frames: set[int] = set()
        for grp in groupings:
            for segments in grp["fish_segments"].values():
                for seg in segments:
                    all_valid_frames.update(seg)

        # Build global detection map from all chunk caches
        global_det_map: dict[tuple[int, str], tuple[int, Any]] = {}
        for cache_path in chunk_caches:
            chunk_number = self._parse_chunk_number(cache_path)
            chunk_start = chunk_number * chunk_size

            try:
                ctx = load_chunk_cache(cache_path)
            except StaleCacheError as exc:
                logger.warning(
                    "Skipping chunk %d: stale cache -- %s", chunk_number, exc
                )
                continue

            chunk_map = self._build_detection_map(
                ctx, chunk_start, all_valid_frames, prestitch_map or None
            )
            global_det_map.update(chunk_map)

        logger.info(
            "Detection map: %d (frame, camera) entries from %d chunk caches",
            len(global_det_map),
            len(chunk_caches),
        )

        # Step 5: Plan crops for each grouping
        # Structure: list of (group_idx, fish_id, global_frame, cam_id, det)
        crop_plan: list[tuple[int, int, int, str, Any]] = []

        for group_idx, grp in enumerate(groupings):
            fish_ids_in_group = sorted(grp["fish_segments"].keys())
            logger.info(
                "Group %d [%d-%d): fish %s",
                group_idx,
                grp["window_start"],
                grp["window_end"],
                fish_ids_in_group,
            )

            for fid in fish_ids_in_group:
                # Collect all (frame, camera, det) for this fish in this window
                fish_detections: list[tuple[int, str, Any]] = []
                for seg in grp["fish_segments"][fid]:
                    for frame in seg:
                        # Find all cameras with detections for this frame+fish
                        for (gf, cam), (det_fid, det) in global_det_map.items():
                            if gf == frame and det_fid == fid:
                                fish_detections.append((frame, cam, det))

                sampled = _camera_aware_sample(fish_detections, cfg.crops_per_fish)
                logger.info(
                    "  Fish %d: %d detections -> %d sampled",
                    fid,
                    len(fish_detections),
                    len(sampled),
                )

                for frame, cam, det in sampled:
                    crop_plan.append((group_idx, fid, frame, cam, det))

        if not crop_plan:
            raise RuntimeError(
                "No crops to extract. Detection map may not overlap with "
                "grouping windows."
            )

        # Step 6: Group by frame for efficient video I/O
        frames_needed: dict[int, list[tuple[int, int, str, Any]]] = defaultdict(list)
        for group_idx, fid, frame, cam, det in crop_plan:
            frames_needed[frame].append((group_idx, fid, cam, det))

        # Step 7: Extract crops
        from aquapose.core.types.frame_source import VideoFrameSource

        output_dir.mkdir(parents=True, exist_ok=True)
        n_crops = 0
        per_fish_counts: dict[int, int] = defaultdict(int)
        manifests: dict[int, dict[str, Any]] = {}  # group_idx -> manifest

        with VideoFrameSource(video_dir, calibration_path) as frame_source:
            for frame_idx, frame in enumerate(sorted(frames_needed.keys())):
                if frame_idx % 50 == 0:
                    logger.info(
                        "Extracting crops: frame %d (%d/%d unique frames)",
                        frame,
                        frame_idx + 1,
                        len(frames_needed),
                    )

                try:
                    cam_frames = frame_source.read_frame(frame)
                except Exception:
                    logger.warning("Failed to read frame %d, skipping", frame)
                    continue

                for group_idx, fid, cam, det in frames_needed[frame]:
                    if cam not in cam_frames:
                        continue

                    bgr_frame = cam_frames[cam]

                    # Extract OBB-aligned crop
                    cx = det.bbox[0] + det.bbox[2] / 2
                    cy = det.bbox[1] + det.bbox[3] / 2
                    angle = det.angle if det.angle is not None else 0.0
                    obb_w = float(det.bbox[2])
                    obb_h = float(det.bbox[3])

                    affine_crop = extract_affine_crop(
                        bgr_frame,
                        (cx, cy),
                        angle,
                        obb_w,
                        obb_h,
                        crop_size=(cfg.crop_size, cfg.crop_size),
                    )

                    # Save JPEG
                    group_dir = output_dir / f"group_{group_idx:03d}" / f"fish_{fid}"
                    group_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"f{frame:06d}_{cam}.jpg"
                    cv2.imwrite(str(group_dir / filename), affine_crop.image)

                    n_crops += 1
                    per_fish_counts[fid] += 1

                    # Track manifest entry
                    if group_idx not in manifests:
                        grp_data = groupings[group_idx]
                        manifests[group_idx] = {
                            "group_id": group_idx,
                            "window_start": grp_data["window_start"],
                            "window_end": grp_data["window_end"],
                            "fish_ids": sorted(grp_data["fish_segments"].keys()),
                            "crops": [],
                        }

                    manifests[group_idx]["crops"].append(
                        {
                            "fish_id": fid,
                            "frame": frame,
                            "camera": cam,
                            "n_cameras": quality_data.get(fid, {})
                            .get(frame, {})
                            .get("n_cameras", 0),
                            "mean_residual": quality_data.get(fid, {})
                            .get(frame, {})
                            .get("mean_residual", -1.0),
                            "detection_confidence": (
                                det.confidence if hasattr(det, "confidence") else None
                            ),
                            "filename": f"fish_{fid}/{filename}",
                        }
                    )

        # Step 8: Write manifests
        for group_idx, manifest in manifests.items():
            manifest_path = output_dir / f"group_{group_idx:03d}" / "manifest.json"
            with manifest_path.open("w") as fh:
                json.dump(manifest, fh, indent=2)

        # Write summary manifest
        summary = {
            "n_groups": len(manifests),
            "n_crops": n_crops,
            "per_fish_counts": dict(per_fish_counts),
            "config": {
                "window_size": cfg.window_size,
                "window_stride": cfg.window_stride,
                "min_cooccurring": cfg.min_cooccurring,
                "min_cameras": cfg.min_cameras,
                "max_residual": cfg.max_residual,
                "min_duration": cfg.min_duration,
                "crops_per_fish": cfg.crops_per_fish,
                "crop_size": cfg.crop_size,
            },
        }
        summary_path = output_dir / "manifest_summary.json"
        with summary_path.open("w") as fh:
            json.dump(summary, fh, indent=2)

        logger.info(
            "Mining complete: %d groups, %d crops total", len(manifests), n_crops
        )
        for fid in sorted(per_fish_counts):
            logger.info("  Fish %d: %d crops", fid, per_fish_counts[fid])

        return {
            "n_groups": len(manifests),
            "n_crops": n_crops,
            "per_fish_counts": dict(per_fish_counts),
        }
