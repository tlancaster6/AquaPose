"""Batch embedding runner for completed pipeline runs."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, cast

import cv2
import h5py
import numpy as np
import yaml

from aquapose.core.context import StaleCacheError, load_chunk_cache
from aquapose.core.pose.crop import extract_affine_crop
from aquapose.core.reid.embedder import FishEmbedder

logger = logging.getLogger(__name__)


class EmbedRunner:
    """Iterates a completed run's chunk caches, extracts OBB-aligned crops,
    embeds them via FishEmbedder, and writes ``reid/embeddings.npz``.

    Args:
        run_dir: Path to a completed pipeline run directory.
        config: ReidConfig (or any object with model_name, batch_size,
            crop_size, device, embedding_dim attributes).
        save_crops: If True, save individual crops to ``reid/crops/``.
        frame_stride: Embed every Nth frame (default 1 = all frames).
    """

    def __init__(
        self,
        run_dir: Path,
        config: Any,
        save_crops: bool = False,
        frame_stride: int = 1,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._config = config
        self._save_crops = save_crops
        self._frame_stride = frame_stride

        diag_dir = self._run_dir / "diagnostics"
        if not diag_dir.exists():
            raise FileNotFoundError(
                "Embedding requires a diagnostic-mode run. "
                "Re-run the pipeline with --mode diagnostic."
            )

        # Accept either midlines_stitched.h5 or midlines.h5
        self._h5_path = self._run_dir / "midlines_stitched.h5"
        if not self._h5_path.exists():
            self._h5_path = self._run_dir / "midlines.h5"
        if not self._h5_path.exists():
            raise FileNotFoundError(
                f"No midlines H5 file found in {self._run_dir}. "
                "Run the full pipeline first."
            )

    def _load_h5_frame_mapping(
        self,
    ) -> tuple[set[int], dict[int, set[int]]]:
        """Load stitched/raw H5 to get valid frames and frame->fish_id mapping.

        Returns:
            Tuple of (valid_frames set, frame_to_fish_ids dict).
        """
        frame_to_fish_ids: dict[int, set[int]] = {}
        with h5py.File(self._h5_path, "r") as f:
            grp = cast(h5py.Group, f["midlines"])
            frame_index = cast(h5py.Dataset, grp["frame_index"])[()]
            fish_id_arr = cast(h5py.Dataset, grp["fish_id"])[()]

        n_rows = frame_index.shape[0]
        if fish_id_arr.ndim == 1:
            # Single-column layout
            for row_idx in range(n_rows):
                frame = int(frame_index[row_idx])
                fid = int(fish_id_arr[row_idx])
                if fid >= 0:
                    frame_to_fish_ids.setdefault(frame, set()).add(fid)
        else:
            # Multi-column layout: (n_frames, max_fish)
            max_fish = fish_id_arr.shape[1]
            for row_idx in range(n_rows):
                frame = int(frame_index[row_idx])
                for slot in range(max_fish):
                    fid = int(fish_id_arr[row_idx, slot])
                    if fid >= 0:
                        frame_to_fish_ids.setdefault(frame, set()).add(fid)

        valid_frames = set(frame_to_fish_ids.keys())
        return valid_frames, frame_to_fish_ids

    def _build_prestitch_to_stitched_map(self) -> dict[int, int]:
        """Map pre-stitch fish IDs (used in chunk caches) to stitched IDs.

        Compares 3D centroids between ``midlines.h5`` and
        ``midlines_stitched.h5`` to match fragment IDs to their
        post-stitch equivalents.  Cache fish IDs that don't appear in
        ``midlines.h5`` (singletons filtered during H5 write) map to -1.

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
            fid_pre = pre["midlines/fish_id"][:]
            fid_post = post["midlines/fish_id"][:]
            pts_pre = pre["midlines/points"][:]
            pts_post = post["midlines/points"][:]

        mapping: dict[int, int] = {}
        n_frames = fid_pre.shape[0]
        mid_kpt = pts_pre.shape[2] // 2  # spine midpoint index

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

                if best_dist < 0.01:  # 1 cm — true matches are ~0
                    mapping[pid] = best_post_id

        logger.info(
            "Pre->post stitch mapping: %d fragment IDs -> %d stitched IDs",
            len(mapping),
            len(set(mapping.values())),
        )
        return mapping

    def _load_run_config(self) -> dict[str, Any]:
        """Load the run's config.yaml."""
        config_path = self._run_dir / "config.yaml"
        if config_path.exists():
            with config_path.open() as fh:
                return yaml.safe_load(fh) or {}
        return {}

    def _discover_chunk_caches(self) -> list[Path]:
        """Find and sort chunk cache files."""
        diag_dir = self._run_dir / "diagnostics"
        caches = sorted(diag_dir.glob("chunk_*/cache.pkl"))
        return caches

    def _parse_chunk_number(self, cache_path: Path) -> int:
        """Extract chunk number from cache path like .../chunk_003/cache.pkl."""
        chunk_dir = cache_path.parent.name
        match = re.search(r"chunk_(\d+)", chunk_dir)
        if match:
            return int(match.group(1))
        raise ValueError(f"Cannot parse chunk number from {cache_path}")

    def run(self) -> Path:
        """Execute the full embedding pipeline.

        Returns:
            Path to the written ``embeddings.npz`` file.
        """
        from aquapose.core.reid.eval import compute_reid_metrics, print_reid_report

        # Step 1: Load H5 frame mapping and ID remapping
        valid_frames, frame_to_fish_ids = self._load_h5_frame_mapping()
        prestitch_map = self._build_prestitch_to_stitched_map()
        stitched_ids = set(fid for fids in frame_to_fish_ids.values() for fid in fids)
        logger.info(
            "H5 mapping: %d valid frames, %d unique stitched fish, "
            "%d pre-stitch fragments mapped",
            len(valid_frames),
            len(stitched_ids),
            len(prestitch_map),
        )

        # Step 2: Load run config for chunk_size and video/calib paths
        run_config = self._load_run_config()
        chunk_size = run_config.get("chunk_size", 300) or 300
        video_dir = run_config.get("video_dir", "")
        calibration_path = run_config.get("calibration_path", "")

        if not video_dir or not calibration_path:
            raise ValueError(
                "Run config.yaml must contain video_dir and calibration_path"
            )

        # Step 3: Discover chunk caches
        chunk_caches = self._discover_chunk_caches()
        if not chunk_caches:
            raise FileNotFoundError(
                f"No chunk caches found in {self._run_dir / 'diagnostics'}"
            )

        # Step 4: Open video source
        from aquapose.core.types.frame_source import VideoFrameSource

        frame_source = VideoFrameSource(video_dir, calibration_path)

        # Accumulators
        all_embeddings: list[np.ndarray] = []
        all_frame_indices: list[int] = []
        all_fish_ids: list[int] = []
        all_camera_ids: list[str] = []
        all_confidences: list[float] = []

        # Create embedder
        embedder = FishEmbedder(self._config)

        # Create crops directory if saving
        if self._save_crops:
            crops_dir = self._run_dir / "reid" / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)

        n_chunks = len(chunk_caches)

        with frame_source:
            for cache_idx, cache_path in enumerate(chunk_caches):
                chunk_number = self._parse_chunk_number(cache_path)
                chunk_start = chunk_number * chunk_size

                # Load chunk cache
                try:
                    ctx = load_chunk_cache(cache_path)
                except StaleCacheError as exc:
                    logger.warning(
                        "Skipping chunk %d: stale cache — %s", chunk_number, exc
                    )
                    continue

                if ctx.tracklet_groups is None or ctx.detections is None:
                    logger.warning(
                        "Skipping chunk %d: missing tracklet_groups or detections",
                        chunk_number,
                    )
                    continue

                # Build detection-to-fish_id mapping for this chunk.
                # Key includes fish_id so multiple fish per (frame, camera)
                # are all retained.
                detection_map: dict[tuple[int, str, int], tuple[int, Any, float]] = {}

                for group in ctx.tracklet_groups:
                    # Map cache fish_id to stitched fish_id
                    raw_id = group.fish_id
                    fish_id = prestitch_map.get(raw_id, -1)
                    if fish_id < 0:
                        continue  # singleton or unmapped fragment
                    for tracklet in group.tracklets:
                        cam_id = tracklet.camera_id
                        for i, local_frame in enumerate(tracklet.frames):
                            if tracklet.frame_status[i] == "coasted":
                                continue

                            global_frame = chunk_start + local_frame
                            if global_frame not in valid_frames:
                                continue

                            # Find matching detection by closest centroid
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
                                detection_map[(local_frame, cam_id, fish_id)] = (
                                    fish_id,
                                    best_det,
                                    best_det.confidence,
                                )

                # Extract crops for this chunk, sorted by frame for sequential reads
                chunk_crops: list[np.ndarray] = []
                chunk_meta: list[tuple[int, int, str, float]] = []

                sorted_entries = sorted(detection_map.keys(), key=lambda x: x[0])

                # Group by local_frame for efficient frame reads
                frames_needed: dict[int, list[tuple[int, str, int]]] = {}
                for local_frame, cam_id, fid in sorted_entries:
                    frames_needed.setdefault(local_frame, []).append(
                        (local_frame, cam_id, fid)
                    )

                for local_frame in sorted(frames_needed.keys()):
                    global_frame = chunk_start + local_frame
                    if (
                        self._frame_stride > 1
                        and global_frame % self._frame_stride != 0
                    ):
                        continue
                    try:
                        cam_frames = frame_source.read_frame(global_frame)
                    except Exception:
                        logger.warning(
                            "Failed to read frame %d, skipping", global_frame
                        )
                        continue

                    for lf, cam_id, fid in frames_needed[local_frame]:
                        if cam_id not in cam_frames:
                            continue

                        fish_id, det, conf = detection_map[(lf, cam_id, fid)]
                        bgr_frame = cam_frames[cam_id]

                        # Extract OBB-aligned crop
                        cx = det.bbox[0] + det.bbox[2] / 2
                        cy = det.bbox[1] + det.bbox[3] / 2
                        angle = det.angle if det.angle is not None else 0.0
                        obb_w = float(det.bbox[2])
                        obb_h = float(det.bbox[3])

                        crop_sz = self._config.crop_size
                        affine_crop = extract_affine_crop(
                            bgr_frame,
                            (cx, cy),
                            angle,
                            obb_w,
                            obb_h,
                            crop_size=(crop_sz, crop_sz),
                        )

                        chunk_crops.append(affine_crop.image)
                        chunk_meta.append((global_frame, fish_id, cam_id, conf))

                        if self._save_crops:
                            fname = (
                                f"frame{global_frame:06d}_{cam_id}_fish{fish_id}.jpg"
                            )
                            cv2.imwrite(
                                str(crops_dir / fname),  # type: ignore[possibly-undefined]
                                affine_crop.image,
                            )

                # Embed this chunk's crops
                n_crops = len(chunk_crops)
                print(
                    f"Embedding chunk {cache_idx + 1}/{n_chunks} ({n_crops} crops)..."
                )

                if n_crops > 0:
                    embeddings = embedder.embed_batch(chunk_crops)
                    all_embeddings.append(embeddings)

                    for gf, fid, cid, conf in chunk_meta:
                        all_frame_indices.append(gf)
                        all_fish_ids.append(fid)
                        all_camera_ids.append(cid)
                        all_confidences.append(conf)

        # Step 6: Write NPZ
        reid_dir = self._run_dir / "reid"
        reid_dir.mkdir(parents=True, exist_ok=True)
        npz_path = reid_dir / "embeddings.npz"

        if all_embeddings:
            stacked_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            stacked_embeddings = np.empty(
                (0, self._config.embedding_dim), dtype=np.float32
            )

        n_total = stacked_embeddings.shape[0]
        print(f"Total embeddings: {n_total}")

        np.savez(
            npz_path,
            embeddings=stacked_embeddings,
            frame_index=np.array(all_frame_indices, dtype=np.int64),
            fish_id=np.array(all_fish_ids, dtype=np.int32),
            camera_id=np.array(all_camera_ids, dtype=object),
            detection_confidence=np.array(all_confidences, dtype=np.float32),
        )

        logger.info("Wrote %d embeddings to %s", n_total, npz_path)

        # Step 7: Run zero-shot evaluation
        if n_total > 0:
            metrics = compute_reid_metrics(npz_path)
            print_reid_report(metrics)

        return npz_path
