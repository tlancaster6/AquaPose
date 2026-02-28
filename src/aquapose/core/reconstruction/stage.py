"""ReconstructionStage -- Stage 5 of the 5-stage AquaPose pipeline.

Reads tracklet_groups (Stage 3) and annotated_detections (Stage 4), assembles
per-frame MidlineSets using known camera membership from TrackletGroup, and
produces 3D B-spline midlines via the configured backend. Populates
PipelineContext.midlines_3d.

Camera membership is determined by the tracklet_groups produced by the
Association stage -- no RANSAC cross-view matching is required. Frames where
a fish is observed by fewer than min_cameras cameras are dropped. Short gaps
(up to max_interp_gap consecutive frames) are filled by linear interpolation
of control points with confidence=0.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from aquapose.core.context import PipelineContext
from aquapose.core.reconstruction.backends import get_backend
from aquapose.reconstruction.midline import Midline2D
from aquapose.reconstruction.triangulation import DEFAULT_INLIER_THRESHOLD, Midline3D

__all__ = ["ReconstructionStage"]

logger = logging.getLogger(__name__)

# Default config values (match ReconstructionConfig defaults)
_DEFAULT_MIN_CAMERAS: int = 3
_DEFAULT_MAX_INTERP_GAP: int = 5
_DEFAULT_N_CONTROL_POINTS: int = 7


class ReconstructionStage:
    """Stage 5: Triangulates tracked fish 2D midlines into 3D B-spline midlines.

    Runs after MidlineStage (Stage 4). For each TrackletGroup (fish), determines
    per-frame camera membership from the group's constituent Tracklet2D objects,
    looks up the corresponding AnnotatedDetection midlines, assembles a MidlineSet,
    and delegates triangulation to the configured backend.

    Frames with fewer than ``min_cameras`` cameras are dropped. Consecutive
    dropped frames up to ``max_interp_gap`` are filled by linear interpolation
    of B-spline control points; interpolated frames carry ``confidence=0`` and
    ``is_low_confidence=True``.

    Args:
        calibration_path: Path to the AquaCal calibration JSON file.
        backend: Backend kind -- ``"triangulation"`` (default) or
            ``"curve_optimizer"``.
        inlier_threshold: Maximum reprojection error (pixels) for RANSAC inliers.
        snap_threshold: Maximum pixel distance from epipolar curve for
            correspondence refinement.
        max_depth: Maximum allowed fish depth below the water surface (metres).
            None disables the upper depth bound.
        min_cameras: Minimum cameras to attempt triangulation per fish per frame.
        max_interp_gap: Maximum consecutive dropped frames to interpolate.
        n_control_points: Fixed B-spline control point count per fish per frame.

    Raises:
        FileNotFoundError: If *calibration_path* does not exist.
        ValueError: If *backend* is not a recognized backend identifier.

    """

    def __init__(
        self,
        calibration_path: str | Path,
        backend: str = "triangulation",
        inlier_threshold: float = DEFAULT_INLIER_THRESHOLD,
        snap_threshold: float = 20.0,
        max_depth: float | None = None,
        min_cameras: int = _DEFAULT_MIN_CAMERAS,
        max_interp_gap: int = _DEFAULT_MAX_INTERP_GAP,
        n_control_points: int = _DEFAULT_N_CONTROL_POINTS,
    ) -> None:
        self._calibration_path = Path(calibration_path)
        self._min_cameras = min_cameras
        self._max_interp_gap = max_interp_gap
        self._n_control_points = n_control_points

        # Build kwargs for the backend constructor
        combined_kwargs: dict[str, object] = {
            "calibration_path": calibration_path,
        }
        if backend == "triangulation":
            combined_kwargs["inlier_threshold"] = inlier_threshold
            combined_kwargs["snap_threshold"] = snap_threshold
            combined_kwargs["max_depth"] = max_depth
            combined_kwargs["n_control_points"] = n_control_points
        self._backend = get_backend(backend, **combined_kwargs)

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run 3D reconstruction across all frames.

        When ``tracklet_groups`` is non-empty, uses known camera membership from
        each TrackletGroup to build per-frame MidlineSets and triangulate without
        RANSAC. Frames below ``min_cameras`` are dropped; short gaps are
        interpolated.

        When ``tracklet_groups`` is empty (stub from early phases), produces
        empty midlines_3d without attempting reconstruction.

        Populates ``context.midlines_3d`` as a list (one entry per frame) of
        dicts mapping fish_id to Midline3D.

        Args:
            context: Accumulated pipeline state from prior stages.

        Returns:
            The same *context* object with ``midlines_3d`` populated.

        Raises:
            ValueError: If ``context.tracklet_groups`` is None and
                ``context.annotated_detections`` is also None.

        """
        tracklet_groups = context.tracklet_groups

        # Empty tracklet_groups: produce empty midlines for each frame.
        if tracklet_groups is not None and len(tracklet_groups) == 0:
            frame_count = context.frame_count or 0
            context.midlines_3d = [{} for _ in range(frame_count)]
            logger.info(
                "ReconstructionStage.run: tracklet_groups empty (stub) -- "
                "producing empty midlines_3d for %d frames",
                frame_count,
            )
            return context

        # Non-empty tracklet_groups: use known camera membership.
        if tracklet_groups is not None and len(tracklet_groups) > 0:
            return self._run_with_tracklet_groups(context, tracklet_groups)

        # Fallback: no tracklet_groups, use annotated_detections directly.
        if context.annotated_detections is None:
            raise ValueError(
                "ReconstructionStage requires context.annotated_detections -- "
                "it is not populated. Ensure Stage 4 (MidlineStage) has run.",
            )
        return self._run_legacy(context)

    def _run_with_tracklet_groups(
        self,
        context: PipelineContext,
        tracklet_groups: list,
    ) -> PipelineContext:
        """Reconstruct using known camera membership from tracklet groups.

        For each fish (TrackletGroup), determines which cameras observe the
        fish in each frame, looks up the corresponding midline from
        annotated_detections, builds a MidlineSet, and triangulates. Short
        gaps are interpolated.

        Args:
            context: Pipeline context with annotated_detections populated.
            tracklet_groups: Non-empty list of TrackletGroup objects.

        Returns:
            Context with midlines_3d populated.
        """
        if context.annotated_detections is None:
            raise ValueError(
                "ReconstructionStage requires context.annotated_detections -- "
                "it is not populated. Ensure Stage 4 (MidlineStage) has run.",
            )

        t0 = time.perf_counter()
        annotated = context.annotated_detections
        frame_count = context.frame_count or len(annotated)

        # Per-fish reconstruction results: fish_id -> {frame_idx -> Midline3D}
        per_fish_results: dict[int, dict[int, Midline3D]] = {}
        # Track dropped frames per fish for logging
        dropped_frames: dict[int, dict[int, str]] = {}

        for group in tracklet_groups:
            fish_id = group.fish_id
            per_fish_results[fish_id] = {}
            dropped_frames[fish_id] = {}

            # Build per-frame camera membership: frame_idx -> list of
            # (camera_id, tracklet, tracklet_frame_index)
            frame_cameras: dict[int, list[tuple]] = {}
            for tracklet in group.tracklets:
                frame_map = {f: i for i, f in enumerate(tracklet.frames)}
                for frame_idx, tidx in frame_map.items():
                    if frame_idx >= frame_count:
                        continue
                    # Only use detected frames (not coasted)
                    if tracklet.frame_status[tidx] != "detected":
                        continue
                    if frame_idx not in frame_cameras:
                        frame_cameras[frame_idx] = []
                    frame_cameras[frame_idx].append(
                        (tracklet.camera_id, tracklet, tidx)
                    )

            # Process each frame for this fish
            for frame_idx in range(frame_count):
                cameras = frame_cameras.get(frame_idx, [])

                if len(cameras) < self._min_cameras:
                    dropped_frames[fish_id][frame_idx] = "insufficient_views"
                    continue

                # Build MidlineSet for this fish+frame
                cam_midlines: dict[str, Midline2D] = {}
                for cam_id, tracklet, tidx in cameras:
                    if frame_idx >= len(annotated):
                        continue
                    frame_ann = annotated[frame_idx]
                    if cam_id not in frame_ann:
                        continue

                    centroid = tracklet.centroids[tidx]
                    ad = _find_matching_annotated(frame_ann[cam_id], centroid)
                    if ad is not None:
                        midline = getattr(ad, "midline", None)
                        if midline is not None:
                            cam_midlines[cam_id] = midline

                # Re-check after lookup (some cameras may not have midlines)
                if len(cam_midlines) < self._min_cameras:
                    dropped_frames[fish_id][frame_idx] = "insufficient_views"
                    continue

                midline_set = {fish_id: cam_midlines}
                frame_results = self._backend.reconstruct_frame(
                    frame_idx=frame_idx,
                    midline_set=midline_set,
                )
                if fish_id in frame_results:
                    per_fish_results[fish_id][frame_idx] = frame_results[fish_id]
                else:
                    dropped_frames[fish_id][frame_idx] = "reconstruction_failed"

            # Gap interpolation for this fish
            self._interpolate_gaps(
                per_fish_results[fish_id],
                frame_count,
                group,
            )

        # Assemble frame-major output
        midlines_3d: list[dict[int, Midline3D]] = []
        for frame_idx in range(frame_count):
            frame_dict: dict[int, Midline3D] = {}
            for fish_id, fish_results in per_fish_results.items():
                if frame_idx in fish_results:
                    frame_dict[fish_id] = fish_results[frame_idx]
            midlines_3d.append(frame_dict)

        elapsed = time.perf_counter() - t0
        n_fish = len(per_fish_results)
        n_total = sum(len(r) for r in per_fish_results.values())
        n_dropped = sum(len(d) for d in dropped_frames.values())
        logger.info(
            "ReconstructionStage.run: %d frames, %d fish, "
            "%d reconstructions, %d dropped, %.2fs",
            frame_count,
            n_fish,
            n_total,
            n_dropped,
            elapsed,
        )

        context.midlines_3d = midlines_3d
        return context

    def _interpolate_gaps(
        self,
        fish_results: dict[int, Midline3D],
        frame_count: int,
        group: object,
    ) -> None:
        """Fill short gaps in a fish's results by linear interpolation.

        Scans for consecutive missing frames bounded by valid frames on both
        sides. If the gap length is <= max_interp_gap, linearly interpolates
        control points and sets is_low_confidence=True.

        Args:
            fish_results: Mutable dict mapping frame_idx to Midline3D for one fish.
            frame_count: Total number of frames.
            group: TrackletGroup (used for per_frame_confidence if available).
        """
        valid_frames = sorted(fish_results.keys())
        if len(valid_frames) < 2:
            return

        for i in range(len(valid_frames) - 1):
            start_f = valid_frames[i]
            end_f = valid_frames[i + 1]
            gap = end_f - start_f - 1

            if gap < 1 or gap > self._max_interp_gap:
                continue

            start_m = fish_results[start_f]
            end_m = fish_results[end_f]

            start_cp = np.asarray(start_m.control_points, dtype=np.float32)
            end_cp = np.asarray(end_m.control_points, dtype=np.float32)

            for g in range(1, gap + 1):
                t = g / (gap + 1)
                interp_cp = (1.0 - t) * start_cp + t * end_cp

                interp_midline = Midline3D(
                    fish_id=start_m.fish_id,
                    frame_index=start_f + g,
                    control_points=interp_cp,
                    knots=start_m.knots.copy(),
                    degree=start_m.degree,
                    arc_length=float(
                        (1.0 - t) * start_m.arc_length + t * end_m.arc_length
                    ),
                    half_widths=start_m.half_widths.copy(),
                    n_cameras=0,
                    mean_residual=0.0,
                    max_residual=0.0,
                    is_low_confidence=True,
                )
                fish_results[start_f + g] = interp_midline

    def _run_legacy(self, context: PipelineContext) -> PipelineContext:
        """Legacy path using annotated_detections without tracklet_groups.

        Args:
            context: Pipeline context with annotated_detections.

        Returns:
            Context with midlines_3d populated.
        """
        t0 = time.perf_counter()
        midlines_3d_per_frame: list[dict] = []

        for frame_idx, frame_annotated in enumerate(
            context.annotated_detections,  # type: ignore[arg-type]
        ):
            midline_set = self._assemble_midline_set_legacy(
                frame_idx=frame_idx,
                frame_annotated=frame_annotated,
            )

            if midline_set:
                frame_results = self._backend.reconstruct_frame(
                    frame_idx=frame_idx,
                    midline_set=midline_set,
                )
            else:
                frame_results = {}

            midlines_3d_per_frame.append(frame_results)

        elapsed = time.perf_counter() - t0
        logger.info(
            "ReconstructionStage.run (legacy): %d frames, %.2fs",
            len(midlines_3d_per_frame),
            elapsed,
        )

        context.midlines_3d = midlines_3d_per_frame
        return context

    @staticmethod
    def _assemble_midline_set_legacy(
        frame_idx: int,
        frame_annotated: dict[str, list],
    ) -> dict[int, dict[str, Midline2D]]:
        """Assemble a MidlineSet from annotated detections (legacy mode).

        Assigns fish IDs sequentially from detections across cameras.

        Args:
            frame_idx: Current frame index (for logging).
            frame_annotated: Dict mapping camera_id to list of AnnotatedDetection.

        Returns:
            MidlineSet: dict[fish_id, dict[cam_id, Midline2D]].
        """
        midline_set: dict[int, dict[str, Midline2D]] = {}
        fish_counter = 0

        for cam_id, det_list in frame_annotated.items():
            for det in det_list:
                midline = getattr(det, "midline", None)
                if midline is not None:
                    if fish_counter not in midline_set:
                        midline_set[fish_counter] = {}
                    midline_set[fish_counter][cam_id] = midline
                    fish_counter += 1

        return midline_set


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_matching_annotated(
    annotated_list: list,
    centroid: tuple[float, float],
    tolerance: float = 10.0,
) -> object | None:
    """Find the AnnotatedDetection closest to a tracklet centroid.

    Args:
        annotated_list: List of AnnotatedDetection objects.
        centroid: (u, v) centroid from the tracklet.
        tolerance: Maximum pixel distance for matching.

    Returns:
        Closest AnnotatedDetection, or None if none within tolerance.
    """
    best_ad = None
    best_dist = tolerance

    for ad in annotated_list:
        if hasattr(ad, "detection") and hasattr(ad.detection, "centroid"):
            c = ad.detection.centroid
            dist = ((c[0] - centroid[0]) ** 2 + (c[1] - centroid[1]) ** 2) ** 0.5
        elif hasattr(ad, "detection") and hasattr(ad.detection, "bbox"):
            bx, by, bw, bh = ad.detection.bbox
            cx = bx + bw / 2
            cy = by + bh / 2
            dist = ((cx - centroid[0]) ** 2 + (cy - centroid[1]) ** 2) ** 0.5
        elif hasattr(ad, "centroid"):
            c = ad.centroid
            dist = ((c[0] - centroid[0]) ** 2 + (c[1] - centroid[1]) ** 2) ** 0.5
        elif hasattr(ad, "bbox"):
            bx, by, bw, bh = ad.bbox
            cx = bx + bw / 2
            cy = by + bh / 2
            dist = ((cx - centroid[0]) ** 2 + (cy - centroid[1]) ** 2) ** 0.5
        else:
            continue

        if dist < best_dist:
            best_dist = dist
            best_ad = ad

    return best_ad
