"""Appearance-based swap detection and repair for fish identity correction.

Detects identity swaps via cross-pattern cosine similarity of temporal
embedding windows, and repairs them by relabeling fish_id in a corrected
H5 output file. Two detection modes: body-length seeded (fast) and
independent embedding scan (thorough).

Repair only relabels fish_id values; all 3D point data is mathematically
identical before and after repair. Reprojection error is invariant by
construction.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import h5py
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ReidEvent:
    """A detected or analyzed identity swap event with provenance.

    Args:
        frame: Frame index where the swap is detected.
        fish_a: First fish ID involved in the swap.
        fish_b: Second fish ID involved in the swap.
        cosine_margin: Minimum cross-pattern cosine margin (positive = swapped).
        detection_mode: How the event was detected ("seeded", "scan_proximity",
            "scan_window").
        action: Result of analysis ("confirmed", "rejected", "repaired",
            "skipped_insufficient_data").
    """

    frame: int
    fish_a: int
    fish_b: int
    cosine_margin: float
    detection_mode: str
    action: str


@dataclass
class SwapDetectorConfig:
    """Configuration for swap detection thresholds.

    Args:
        cosine_margin_threshold: Minimum cross-pattern margin to confirm swap.
        window_frames: Number of frames before/after event for embedding window.
        proximity_threshold_m: 3D centroid distance (meters) for proximity scan.
        scan_frame_stride: Frame stride for scan mode embedding.
        scan_dense_window: Frames to embed densely around proximity events.
        scan_gap_stride: Coarse stride for gap-filling in scan mode.
        min_window_frames: Minimum embedding frames required per window.
    """

    cosine_margin_threshold: float = 0.15
    window_frames: int = 10
    proximity_threshold_m: float = 0.15
    scan_frame_stride: int = 1
    scan_dense_window: int = 20
    scan_gap_stride: int = 10
    min_window_frames: int = 3


def _confirm_swap(
    emb_a_pre: NDArray[np.float32],
    emb_a_post: NDArray[np.float32],
    emb_b_pre: NDArray[np.float32],
    emb_b_post: NDArray[np.float32],
    threshold: float,
) -> tuple[bool, float]:
    """Cross-pattern swap confirmation via cosine similarity margins.

    A swap is confirmed when fish A's post-event embedding is closer to
    fish B's pre-event embedding than to its own pre-event embedding,
    AND vice versa, with both margins exceeding the threshold.

    All inputs must be L2-normalized.

    Args:
        emb_a_pre: Mean pre-event embedding for fish A.
        emb_a_post: Mean post-event embedding for fish A.
        emb_b_pre: Mean pre-event embedding for fish B.
        emb_b_post: Mean post-event embedding for fish B.
        threshold: Minimum margin for confirmation.

    Returns:
        Tuple of (confirmed, cosine_margin) where margin is
        min(cross_a - self_a, cross_b - self_b).
    """
    # Cross similarities: does A-post match B-pre, and B-post match A-pre?
    cross_a = float(np.dot(emb_a_post, emb_b_pre))
    cross_b = float(np.dot(emb_b_post, emb_a_pre))

    # Self similarities: does A-post still match A-pre?
    self_a = float(np.dot(emb_a_post, emb_a_pre))
    self_b = float(np.dot(emb_b_post, emb_b_pre))

    margin_a = cross_a - self_a
    margin_b = cross_b - self_b
    cosine_margin = min(margin_a, margin_b)

    return cosine_margin > threshold, cosine_margin


def _compute_mean_embedding(
    embeddings: NDArray[np.float32],
) -> NDArray[np.float32] | None:
    """Compute L2-normalized mean of embedding vectors.

    Args:
        embeddings: Array of shape (N, D) with embedding rows.

    Returns:
        Unit-norm mean vector of shape (D,), or None if input is empty.
    """
    if embeddings.shape[0] == 0:
        return None

    mean = embeddings.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-10:
        return None
    return (mean / norm).astype(np.float32)


def _find_proximity_events(
    frame_index: NDArray[np.int64],
    fish_id: NDArray[np.int32],
    points: NDArray[np.float32],
    threshold_m: float,
) -> list[tuple[int, int, int]]:
    """Find frames where two fish 3D centroids are within distance threshold.

    Applies Z_WEIGHT to the z-component to account for reconstruction
    anisotropy. Consecutive frames with the same pair are collapsed into
    a single event at the midpoint frame.

    Args:
        frame_index: Shape (N,) frame indices.
        fish_id: Shape (N, max_fish) fish ID assignments (-1 = empty slot).
        points: Shape (N, max_fish, n_kpts, 3) 3D keypoints.
        threshold_m: Distance threshold in meters.

    Returns:
        List of (frame, fish_a, fish_b) tuples, deduplicated.
    """
    from aquapose.core.stitching import Z_WEIGHT

    n_frames, max_fish = fish_id.shape
    # Collect raw per-frame proximity hits
    raw_events: list[tuple[int, int, int]] = []

    for row in range(n_frames):
        fish_slots = [
            (int(fish_id[row, s]), s) for s in range(max_fish) if fish_id[row, s] >= 0
        ]
        for i, (fid_a, slot_a) in enumerate(fish_slots):
            for fid_b, slot_b in fish_slots[i + 1 :]:
                pts_a = points[row, slot_a]
                pts_b = points[row, slot_b]
                valid_a = ~np.isnan(pts_a).any(axis=1)
                valid_b = ~np.isnan(pts_b).any(axis=1)
                if not valid_a.any() or not valid_b.any():
                    continue
                c_a = pts_a[valid_a].mean(axis=0)
                c_b = pts_b[valid_b].mean(axis=0)
                diff = c_a - c_b
                diff[2] *= Z_WEIGHT
                dist = float(np.linalg.norm(diff))
                if dist < threshold_m:
                    raw_events.append((int(frame_index[row]), int(fid_a), int(fid_b)))

    # Collapse consecutive frames of same pair into single event at midpoint
    if not raw_events:
        return []

    # Group by fish pair
    pair_frames: dict[tuple[int, int], list[int]] = {}
    for frame, fa, fb in raw_events:
        pair_key = (min(fa, fb), max(fa, fb))
        pair_frames.setdefault(pair_key, []).append(frame)

    result: list[tuple[int, int, int]] = []
    for (fa, fb), frames in pair_frames.items():
        frames_sorted = sorted(frames)
        # Split into consecutive runs
        runs: list[list[int]] = [[frames_sorted[0]]]
        for f in frames_sorted[1:]:
            if f - runs[-1][-1] <= 1:
                runs[-1].append(f)
            else:
                runs.append([f])
        for run in runs:
            midpoint = run[len(run) // 2]
            result.append((midpoint, fa, fb))

    return sorted(result)


def _write_reid_events(h5_path: Path, events: list[ReidEvent]) -> None:
    """Write reid events provenance to H5 file under /reid_events/ group.

    Uses separate datasets per field to avoid structured dtype string issues.

    Args:
        h5_path: Path to the H5 file to write to.
        events: List of ReidEvent records.
    """
    with h5py.File(h5_path, "r+") as f:
        if "reid_events" in f:
            del f["reid_events"]
        if not events:
            return
        grp = f.create_group("reid_events")
        grp.create_dataset(
            "frame", data=np.array([e.frame for e in events], dtype=np.int32)
        )
        grp.create_dataset(
            "fish_a", data=np.array([e.fish_a for e in events], dtype=np.int32)
        )
        grp.create_dataset(
            "fish_b", data=np.array([e.fish_b for e in events], dtype=np.int32)
        )
        grp.create_dataset(
            "cosine_margin",
            data=np.array([e.cosine_margin for e in events], dtype=np.float32),
        )
        grp.create_dataset(
            "detection_mode",
            data=np.array(
                [e.detection_mode for e in events], dtype=h5py.string_dtype()
            ),
        )
        grp.create_dataset(
            "action",
            data=np.array([e.action for e in events], dtype=h5py.string_dtype()),
        )


class SwapDetector:
    """Detects and repairs fish identity swaps using appearance embeddings.

    Supports two detection modes:
    - **seeded**: Reads body-length swap events from ``/midlines/swap_events``
      in the stitched H5, confirms or rejects each using cross-pattern cosine
      similarity of temporal embedding windows.
    - **scan**: Finds proximity events from 3D centroids, embeds densely
      around them, and applies cross-pattern check. Also scans remaining
      gaps at coarse stride with a sliding window.

    Args:
        run_dir: Path to a completed pipeline run directory.
        config: Detection thresholds. Uses defaults if None.
        projection_head_path: Optional path to a trained projection head
            (``torch.nn.Linear`` or ``nn.Sequential``) to transform raw
            768-dim backbone embeddings before comparison.
    """

    def __init__(
        self,
        run_dir: Path,
        config: SwapDetectorConfig | None = None,
        projection_head_path: Path | None = None,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._config = config or SwapDetectorConfig()
        self._projection_head_path = projection_head_path

        # Find H5 file
        self._h5_path = self._run_dir / "midlines_stitched.h5"
        if not self._h5_path.exists():
            self._h5_path = self._run_dir / "midlines.h5"
        if not self._h5_path.exists():
            raise FileNotFoundError(f"No midlines H5 file found in {self._run_dir}")

        # Load H5 data
        with h5py.File(self._h5_path, "r") as f:
            grp = cast(h5py.Group, f["midlines"])
            self._frame_index = cast(h5py.Dataset, grp["frame_index"])[()]
            self._fish_id = cast(h5py.Dataset, grp["fish_id"])[()]
            self._points = cast(h5py.Dataset, grp["points"])[()]
            # Load swap events if present
            self._swap_events_raw = None
            if "swap_events" in grp:
                self._swap_events_raw = grp["swap_events"][()]

        # Opportunistically load existing embeddings
        self._embeddings: NDArray[np.float32] | None = None
        self._emb_frames: NDArray[np.int64] | None = None
        self._emb_fish_ids: NDArray[np.int32] | None = None
        npz_path = self._run_dir / "reid" / "embeddings.npz"
        if npz_path.exists():
            npz = np.load(npz_path, allow_pickle=True)
            self._embeddings = npz["embeddings"].astype(np.float32)
            self._emb_frames = npz["frame_index"].astype(np.int64)
            self._emb_fish_ids = npz["fish_id"].astype(np.int32)
            logger.info(
                "Loaded %d existing embeddings from %s",
                self._embeddings.shape[0],
                npz_path,
            )

        # Lazy-init for on-the-fly embedding
        self._embedder = None
        self._frame_source = None
        self._projection_head = None

    def _ensure_embedder(self) -> None:
        """Lazy-initialize FishEmbedder and VideoFrameSource if needed."""
        if self._embedder is not None:
            return

        import yaml

        from aquapose.core.reid.embedder import FishEmbedder

        # Load run config for video paths
        config_path = self._run_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Run config not found at {config_path}. "
                "Cannot do on-the-fly embedding."
            )
        with config_path.open() as fh:
            run_config = yaml.safe_load(fh) or {}

        # Create a simple config-like object for FishEmbedder
        class _EmbedConfig:
            model_name = "hf-hub:BVRA/MegaDescriptor-T-224"
            batch_size = 32
            crop_size = 224
            device = "cuda"
            embedding_dim = 768

        self._embedder = FishEmbedder(_EmbedConfig())

        # Load projection head if provided
        if self._projection_head_path and self._projection_head_path.exists():
            import torch

            self._projection_head = torch.load(
                self._projection_head_path,
                map_location="cuda",
                weights_only=True,
            )
            self._projection_head.eval()
            logger.info("Loaded projection head from %s", self._projection_head_path)

        # Open video frame source
        from aquapose.core.types.frame_source import VideoFrameSource

        video_dir = run_config.get("video_dir", "")
        calibration_path = run_config.get("calibration_path", "")
        if video_dir and calibration_path:
            self._frame_source = VideoFrameSource(video_dir, calibration_path)
            self._frame_source.__enter__()

    def _get_embeddings(
        self, fish_id: int, frame_start: int, frame_end: int
    ) -> NDArray[np.float32] | None:
        """Get embeddings for a fish in a frame range.

        Looks up cached embeddings.npz first, falls back to on-the-fly
        if needed.

        Args:
            fish_id: The stitched fish ID.
            frame_start: Start of frame range (inclusive).
            frame_end: End of frame range (inclusive).

        Returns:
            Array of shape (M, 768) or None if no valid embeddings found.
        """
        if self._embeddings is not None:
            assert self._emb_frames is not None
            assert self._emb_fish_ids is not None
            mask = (
                (self._emb_fish_ids == fish_id)
                & (self._emb_frames >= frame_start)
                & (self._emb_frames <= frame_end)
            )
            if mask.any():
                embs = self._embeddings[mask]
                if self._projection_head is not None:
                    import torch

                    with torch.no_grad():
                        t = torch.from_numpy(embs).cuda()
                        t = self._projection_head(t)
                        import torch.nn.functional as F

                        t = F.normalize(t, p=2, dim=1)
                        embs = t.cpu().numpy()
                return embs

        # On-the-fly embedding would go here but requires video frame source
        # and chunk cache infrastructure. For now, return None if no cached
        # embeddings are available.
        logger.debug(
            "No cached embeddings for fish %d in frames [%d, %d]",
            fish_id,
            frame_start,
            frame_end,
        )
        return None

    def run(self, mode: Literal["seeded", "scan"]) -> list[ReidEvent]:
        """Run swap detection in the specified mode.

        Args:
            mode: Detection mode — "seeded" uses body-length swap events,
                "scan" uses proximity + sliding window.

        Returns:
            List of ReidEvent records with detection results.
        """
        if mode == "seeded":
            return self._run_seeded()
        elif mode == "scan":
            return self._run_scan()
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def _run_seeded(self) -> list[ReidEvent]:
        """Confirm/reject body-length swap events using embeddings.

        Reads ``/midlines/swap_events`` from H5, gathers embeddings in
        pre/post windows, and applies cross-pattern confirmation.

        Returns:
            List of ReidEvent records.
        """
        if self._swap_events_raw is None or len(self._swap_events_raw) == 0:
            logger.info("No body-length swap events found in H5")
            return []

        events: list[ReidEvent] = []
        cfg = self._config

        for row in self._swap_events_raw:
            frame = int(row["frame"])
            fish_a = int(row["fish_a"])
            fish_b = int(row["fish_b"])

            # Gather pre/post window embeddings
            pre_start = max(0, frame - cfg.window_frames)
            pre_end = frame - 1
            post_start = frame
            post_end = frame + cfg.window_frames

            emb_a_pre = self._get_embeddings(fish_a, pre_start, pre_end)
            emb_a_post = self._get_embeddings(fish_a, post_start, post_end)
            emb_b_pre = self._get_embeddings(fish_b, pre_start, pre_end)
            emb_b_post = self._get_embeddings(fish_b, post_start, post_end)

            # Check for insufficient data
            min_frames = cfg.min_window_frames
            insufficient = False
            for emb, label in [
                (emb_a_pre, f"fish {fish_a} pre"),
                (emb_a_post, f"fish {fish_a} post"),
                (emb_b_pre, f"fish {fish_b} pre"),
                (emb_b_post, f"fish {fish_b} post"),
            ]:
                if emb is None or emb.shape[0] < min_frames:
                    logger.warning(
                        "Skipping swap event at frame %d: insufficient "
                        "embedding data for %s (%d frames, need %d)",
                        frame,
                        label,
                        0 if emb is None else emb.shape[0],
                        min_frames,
                    )
                    insufficient = True
                    break

            if insufficient:
                events.append(
                    ReidEvent(
                        frame=frame,
                        fish_a=fish_a,
                        fish_b=fish_b,
                        cosine_margin=0.0,
                        detection_mode="seeded",
                        action="skipped_insufficient_data",
                    )
                )
                continue

            # Compute mean embeddings
            assert emb_a_pre is not None
            assert emb_a_post is not None
            assert emb_b_pre is not None
            assert emb_b_post is not None
            mean_a_pre = _compute_mean_embedding(emb_a_pre)
            mean_a_post = _compute_mean_embedding(emb_a_post)
            mean_b_pre = _compute_mean_embedding(emb_b_pre)
            mean_b_post = _compute_mean_embedding(emb_b_post)

            if any(
                m is None for m in [mean_a_pre, mean_a_post, mean_b_pre, mean_b_post]
            ):
                events.append(
                    ReidEvent(
                        frame=frame,
                        fish_a=fish_a,
                        fish_b=fish_b,
                        cosine_margin=0.0,
                        detection_mode="seeded",
                        action="skipped_insufficient_data",
                    )
                )
                continue

            assert mean_a_pre is not None
            assert mean_a_post is not None
            assert mean_b_pre is not None
            assert mean_b_post is not None
            confirmed, margin = _confirm_swap(
                mean_a_pre,
                mean_a_post,
                mean_b_pre,
                mean_b_post,
                cfg.cosine_margin_threshold,
            )

            events.append(
                ReidEvent(
                    frame=frame,
                    fish_a=fish_a,
                    fish_b=fish_b,
                    cosine_margin=margin,
                    detection_mode="seeded",
                    action="confirmed" if confirmed else "rejected",
                )
            )
            logger.info(
                "Seeded event frame %d: fish %d <-> %d, margin=%.3f, %s",
                frame,
                fish_a,
                fish_b,
                margin,
                "CONFIRMED" if confirmed else "rejected",
            )

        return events

    def _run_scan(self) -> list[ReidEvent]:
        """Independent scan mode: proximity + sliding window detection.

        1. Finds proximity events from 3D centroids.
        2. Applies cross-pattern check to each proximity candidate.
        3. Scans remaining gaps with sliding window at coarse stride.

        Returns:
            List of ReidEvent records.
        """
        cfg = self._config
        events: list[ReidEvent] = []

        # Step 1: Find proximity events
        proximity_events = _find_proximity_events(
            self._frame_index,
            self._fish_id,
            self._points,
            cfg.proximity_threshold_m,
        )
        logger.info("Scan mode: found %d proximity events", len(proximity_events))

        # Step 2: Check each proximity event
        for frame, fish_a, fish_b in proximity_events:
            pre_start = max(0, frame - cfg.window_frames)
            pre_end = frame - 1
            post_start = frame
            post_end = frame + cfg.window_frames

            emb_a_pre = self._get_embeddings(fish_a, pre_start, pre_end)
            emb_a_post = self._get_embeddings(fish_a, post_start, post_end)
            emb_b_pre = self._get_embeddings(fish_b, pre_start, pre_end)
            emb_b_post = self._get_embeddings(fish_b, post_start, post_end)

            min_frames = cfg.min_window_frames
            insufficient = False
            for emb, _label in [
                (emb_a_pre, f"fish {fish_a} pre"),
                (emb_a_post, f"fish {fish_a} post"),
                (emb_b_pre, f"fish {fish_b} pre"),
                (emb_b_post, f"fish {fish_b} post"),
            ]:
                if emb is None or emb.shape[0] < min_frames:
                    insufficient = True
                    break

            if insufficient:
                events.append(
                    ReidEvent(
                        frame=frame,
                        fish_a=fish_a,
                        fish_b=fish_b,
                        cosine_margin=0.0,
                        detection_mode="scan_proximity",
                        action="skipped_insufficient_data",
                    )
                )
                continue

            assert emb_a_pre is not None
            assert emb_a_post is not None
            assert emb_b_pre is not None
            assert emb_b_post is not None
            mean_a_pre = _compute_mean_embedding(emb_a_pre)
            mean_a_post = _compute_mean_embedding(emb_a_post)
            mean_b_pre = _compute_mean_embedding(emb_b_pre)
            mean_b_post = _compute_mean_embedding(emb_b_post)

            if any(
                m is None for m in [mean_a_pre, mean_a_post, mean_b_pre, mean_b_post]
            ):
                events.append(
                    ReidEvent(
                        frame=frame,
                        fish_a=fish_a,
                        fish_b=fish_b,
                        cosine_margin=0.0,
                        detection_mode="scan_proximity",
                        action="skipped_insufficient_data",
                    )
                )
                continue

            assert mean_a_pre is not None
            assert mean_a_post is not None
            assert mean_b_pre is not None
            assert mean_b_post is not None
            confirmed, margin = _confirm_swap(
                mean_a_pre,
                mean_a_post,
                mean_b_pre,
                mean_b_post,
                cfg.cosine_margin_threshold,
            )

            events.append(
                ReidEvent(
                    frame=frame,
                    fish_a=fish_a,
                    fish_b=fish_b,
                    cosine_margin=margin,
                    detection_mode="scan_proximity",
                    action="confirmed" if confirmed else "rejected",
                )
            )
            logger.info(
                "Scan proximity event frame %d: fish %d <-> %d, margin=%.3f, %s",
                frame,
                fish_a,
                fish_b,
                margin,
                "CONFIRMED" if confirmed else "rejected",
            )

        # Step 3: Sliding window gap scan
        # Get all unique fish IDs and frame range
        unique_fish = sorted(set(int(fid) for fid in self._fish_id.flat if fid >= 0))
        frame_min = int(self._frame_index.min())
        frame_max = int(self._frame_index.max())

        # Build set of covered frames (proximity windows)
        covered_frames: set[int] = set()
        for frame, _fa, _fb in proximity_events:
            for f in range(
                frame - cfg.scan_dense_window, frame + cfg.scan_dense_window + 1
            ):
                covered_frames.add(f)

        # Scan uncovered gaps at coarse stride
        scan_frame = frame_min
        while scan_frame <= frame_max:
            if scan_frame in covered_frames:
                scan_frame += 1
                continue

            # Check all fish pairs at this frame
            for i, fa in enumerate(unique_fish):
                for fb in unique_fish[i + 1 :]:
                    pre_start = max(frame_min, scan_frame - cfg.window_frames)
                    pre_end = scan_frame - 1
                    post_start = scan_frame
                    post_end = min(frame_max, scan_frame + cfg.window_frames)

                    emb_a_pre = self._get_embeddings(fa, pre_start, pre_end)
                    emb_a_post = self._get_embeddings(fa, post_start, post_end)
                    emb_b_pre = self._get_embeddings(fb, pre_start, pre_end)
                    emb_b_post = self._get_embeddings(fb, post_start, post_end)

                    min_fr = cfg.min_window_frames
                    if any(
                        e is None or e.shape[0] < min_fr
                        for e in [emb_a_pre, emb_a_post, emb_b_pre, emb_b_post]
                    ):
                        continue

                    assert emb_a_pre is not None
                    assert emb_a_post is not None
                    assert emb_b_pre is not None
                    assert emb_b_post is not None
                    ma_pre = _compute_mean_embedding(emb_a_pre)
                    ma_post = _compute_mean_embedding(emb_a_post)
                    mb_pre = _compute_mean_embedding(emb_b_pre)
                    mb_post = _compute_mean_embedding(emb_b_post)

                    if any(m is None for m in [ma_pre, ma_post, mb_pre, mb_post]):
                        continue

                    assert ma_pre is not None
                    assert ma_post is not None
                    assert mb_pre is not None
                    assert mb_post is not None
                    confirmed, margin = _confirm_swap(
                        ma_pre, ma_post, mb_pre, mb_post, cfg.cosine_margin_threshold
                    )

                    if confirmed:
                        events.append(
                            ReidEvent(
                                frame=scan_frame,
                                fish_a=fa,
                                fish_b=fb,
                                cosine_margin=margin,
                                detection_mode="scan_window",
                                action="confirmed",
                            )
                        )
                        logger.info(
                            "Scan window event frame %d: fish %d <-> %d, "
                            "margin=%.3f, CONFIRMED",
                            scan_frame,
                            fa,
                            fb,
                            margin,
                        )

            scan_frame += cfg.scan_gap_stride

        return events

    def repair(self, events: list[ReidEvent]) -> Path:
        """Apply confirmed swap repairs and write corrected H5 output.

        Copies ``midlines_stitched.h5`` to ``midlines_reid.h5``, applies
        fish_id swaps for events with action="repaired", and writes
        ``/reid_events/`` provenance group.

        Repair only relabels fish_id; 3D point data is mathematically
        identical before and after. Reprojection error is invariant by
        construction.

        Args:
            events: List of ReidEvent records. Only events with
                action="repaired" trigger fish_id swaps.

        Returns:
            Path to the output ``midlines_reid.h5`` file.
        """
        src = self._h5_path
        dst = self._run_dir / "midlines_reid.h5"

        # Copy source to destination
        shutil.copy2(src, dst)

        # Apply fish_id swaps for repaired events in chronological order
        repaired = sorted(
            [e for e in events if e.action == "repaired"],
            key=lambda e: e.frame,
        )

        if repaired:
            with h5py.File(dst, "r+") as f:
                grp = cast(h5py.Group, f["midlines"])
                fish_id_ds = cast(h5py.Dataset, grp["fish_id"])
                data = fish_id_ds[()]
                frame_index = grp["frame_index"][:]

                for event in repaired:
                    start_row = int(np.searchsorted(frame_index, event.frame))
                    n_swapped = 0
                    for row in range(start_row, data.shape[0]):
                        slots_a = np.where(data[row] == event.fish_a)[0]
                        slots_b = np.where(data[row] == event.fish_b)[0]
                        for sa in slots_a:
                            data[row, sa] = event.fish_b
                            n_swapped += 1
                        for sb in slots_b:
                            data[row, sb] = event.fish_a
                            n_swapped += 1

                    logger.info(
                        "Repair: fish %d <-> %d from frame %d (%d slot updates)",
                        event.fish_a,
                        event.fish_b,
                        event.frame,
                        n_swapped,
                    )

                fish_id_ds[...] = data

        # Write provenance
        _write_reid_events(dst, events)

        logger.info("Wrote corrected H5 to %s (%d repairs)", dst, len(repaired))
        return dst
