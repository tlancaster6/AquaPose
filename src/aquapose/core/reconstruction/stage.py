"""ReconstructionStage -- Stage 5 of the 5-stage AquaPose pipeline.

Reads tracklet_groups (Stage 4) and their per-frame keypoints, assembles
per-frame MidlineSets by interpolating raw 6-keypoint poses to dense
n_sample_points Midline2D objects, and produces 3D B-spline midlines via the
configured backend. Populates PipelineContext.midlines_3d.

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
import scipy.interpolate

from aquapose.core.context import PipelineContext
from aquapose.core.reconstruction.backends import get_backend
from aquapose.core.types.midline import Midline2D
from aquapose.core.types.reconstruction import Midline3D

__all__ = ["ReconstructionStage"]

logger = logging.getLogger(__name__)

# Default config values (match ReconstructionConfig defaults)
_DEFAULT_MIN_CAMERAS: int = 3
_DEFAULT_MAX_INTERP_GAP: int = 5
_DEFAULT_N_CONTROL_POINTS: int = 7
_DEFAULT_N_SAMPLE_POINTS: int = 6

# Default arc-length parameter values for the 6 anatomical keypoints.
# Used only when keypoint_t_values is not provided via config.
# Order: nose(0.0), head(0.1), spine1(0.3), spine2(0.5), spine3(0.7), tail(1.0)
_DEFAULT_KEYPOINT_T_VALUES: np.ndarray = np.array(
    [0.0, 0.1, 0.3, 0.5, 0.7, 1.0], dtype=np.float64
)


def _keypoints_to_midline(
    kpts_xy: np.ndarray,
    t_values: np.ndarray,
    confidences: np.ndarray,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate visible keypoints to a dense midline via linear spline.

    Fits a linear interpolating spline through the visible (high-confidence)
    keypoints parameterised by arc-length ``t_values`` and evaluates it at
    ``n_points`` uniformly-spaced positions in [0, 1].

    Args:
        kpts_xy: Shape ``(K, 2)`` float32 — full-frame keypoint coordinates.
        t_values: Shape ``(K,)`` float64 — arc-length parameter for each keypoint
            (values in [0, 1]).
        confidences: Shape ``(K,)`` float32 — per-keypoint confidence.
        n_points: Number of output midline points.

    Returns:
        Tuple of:
        - ``points``: ``(n_points, 2)`` float32 interpolated midline points.
        - ``point_confidence``: ``(n_points,)`` float32 interpolated confidence.
    """
    # Only interpolate through visible (confident) keypoints
    visible = confidences > 0.0
    n_visible = int(visible.sum())

    if n_visible < 2:
        # Fallback: repeat first visible point or use all keypoints
        if n_visible == 1:
            pt = kpts_xy[visible][0]
            points = np.tile(pt, (n_points, 1)).astype(np.float32)
            conf = np.zeros(n_points, dtype=np.float32)
            return points, conf
        # No visible keypoints — use mean of all
        pt = kpts_xy.mean(axis=0)
        points = np.tile(pt, (n_points, 1)).astype(np.float32)
        return points, np.zeros(n_points, dtype=np.float32)

    vis_t = t_values[visible]
    vis_xy = kpts_xy[visible].astype(np.float64)
    vis_conf = confidences[visible].astype(np.float64)

    # Remove duplicates in t_values to avoid scipy error
    _, unique_idx = np.unique(vis_t, return_index=True)
    vis_t = vis_t[unique_idx]
    vis_xy = vis_xy[unique_idx]
    vis_conf = vis_conf[unique_idx]

    t_out = np.linspace(vis_t[0], vis_t[-1], n_points)

    interp_x = scipy.interpolate.interp1d(vis_t, vis_xy[:, 0], kind="linear")
    interp_y = scipy.interpolate.interp1d(vis_t, vis_xy[:, 1], kind="linear")
    interp_c = scipy.interpolate.interp1d(vis_t, vis_conf, kind="linear")

    x_out = interp_x(t_out).astype(np.float32)
    y_out = interp_y(t_out).astype(np.float32)
    c_out = interp_c(t_out).astype(np.float32)

    points = np.stack([x_out, y_out], axis=1)
    return points, c_out


class ReconstructionStage:
    """Stage 5: Triangulates tracked fish 2D midlines into 3D B-spline midlines.

    Runs after PoseStage (Stage 2), TrackingStage (Stage 3), and AssociationStage
    (Stage 4). For each TrackletGroup (fish), determines per-frame camera membership
    from the group's constituent Tracklet2D objects, reads keypoints directly
    from each tracklet, interpolates them to a dense Midline2D, assembles a
    MidlineSet, and delegates triangulation to the configured backend.

    Frames with fewer than ``min_cameras`` cameras are dropped. Consecutive
    dropped frames up to ``max_interp_gap`` are filled by linear interpolation
    of B-spline control points; interpolated frames carry ``confidence=0`` and
    ``is_low_confidence=True``.

    Args:
        calibration_path: Path to the AquaCal calibration JSON file.
        backend: Backend kind -- ``"dlt"`` (default). Only DLT is supported.
        min_cameras: Minimum cameras to attempt triangulation per fish per frame.
        max_interp_gap: Maximum consecutive dropped frames to interpolate.
        n_control_points: Fixed B-spline control point count per fish per frame.
        n_sample_points: Number of dense midline points interpolated from
            keypoints before triangulation. Default 6 (identity mapping).
        keypoint_t_values: Per-keypoint arc-fraction values in [0, 1]. If
            ``None``, uses default uniform spacing ``[0, 0.1, 0.3, 0.5, 0.7, 1]``.
        **backend_kwargs: Additional keyword arguments forwarded to the backend
            constructor (e.g. ``outlier_threshold``).

    Raises:
        FileNotFoundError: If *calibration_path* does not exist.
        ValueError: If *backend* is not a recognized backend identifier.

    """

    def __init__(
        self,
        calibration_path: str | Path,
        backend: str = "dlt",
        min_cameras: int = _DEFAULT_MIN_CAMERAS,
        max_interp_gap: int = _DEFAULT_MAX_INTERP_GAP,
        n_control_points: int = _DEFAULT_N_CONTROL_POINTS,
        n_sample_points: int = _DEFAULT_N_SAMPLE_POINTS,
        keypoint_t_values: list[float] | None = None,
        **backend_kwargs: object,
    ) -> None:
        self._calibration_path = Path(calibration_path)
        self._min_cameras = min_cameras
        self._max_interp_gap = max_interp_gap
        self._n_control_points = n_control_points
        self._n_sample_points = n_sample_points
        self._keypoint_t_values: np.ndarray = (
            np.array(keypoint_t_values, dtype=np.float64)
            if keypoint_t_values is not None
            else _DEFAULT_KEYPOINT_T_VALUES
        )

        # Build kwargs for the backend constructor
        combined_kwargs: dict[str, object] = {
            "calibration_path": calibration_path,
            **backend_kwargs,
        }
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
            ValueError: If ``context.tracklet_groups`` is None.

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

        # Fallback: no tracklet_groups — cannot reconstruct.
        raise ValueError(
            "ReconstructionStage requires context.tracklet_groups -- "
            "it is not populated. Ensure Stage 4 (AssociationStage) has run.",
        )

    def _run_with_tracklet_groups(
        self,
        context: PipelineContext,
        tracklet_groups: list,
    ) -> PipelineContext:
        """Reconstruct using known camera membership from tracklet groups.

        For each fish (TrackletGroup), determines which cameras observe the
        fish in each frame, reads keypoints directly from tracklet data,
        interpolates them to a dense Midline2D, builds a MidlineSet, and
        triangulates. Short gaps are interpolated.

        Args:
            context: Pipeline context (tracklet_groups must be populated).
            tracklet_groups: Non-empty list of TrackletGroup objects.

        Returns:
            Context with midlines_3d populated.
        """
        t0 = time.perf_counter()
        frame_count = context.frame_count or 0

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
                    if tracklet.keypoints is None:
                        continue
                    kpts_xy = tracklet.keypoints[tidx]  # (K, 2)
                    kpts_conf = (
                        tracklet.keypoint_conf[tidx]
                        if tracklet.keypoint_conf is not None
                        else np.ones(kpts_xy.shape[0], dtype=np.float32)
                    )
                    points, point_conf = _keypoints_to_midline(
                        kpts_xy,
                        self._keypoint_t_values,
                        kpts_conf,
                        n_points=self._n_sample_points,
                    )
                    midline = Midline2D(
                        points=points,
                        half_widths=np.zeros(len(points), dtype=np.float32),
                        fish_id=fish_id,
                        camera_id=cam_id,
                        frame_index=frame_idx,
                        point_confidence=point_conf,
                    )
                    cam_midlines[cam_id] = midline

                # Re-check after lookup (some cameras may not have keypoints)
                if len(cam_midlines) < self._min_cameras:
                    dropped_frames[fish_id][frame_idx] = "insufficient_views"
                    continue

                midline_set = {fish_id: cam_midlines}
                frame_results = self._backend.reconstruct_frame(  # type: ignore[union-attr]
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
