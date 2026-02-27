"""Hungarian 3D tracking backend for the Tracking stage (Stage 4).

Wraps the v1.0 FishTracker to maintain persistent fish identities across frames.
The backend is stateful — the FishTracker instance persists across frames,
maintaining track identities and lifecycle state.

Consumes ``AssociationBundle`` objects from Stage 3 (AssociationStage) directly.
FishTracker.update_from_bundles() performs greedy nearest-3D-centroid assignment
between predicted track positions and bundle centroids — no internal RANSAC
re-association is performed. Stage 3 has already computed cross-camera bundles,
so the pipeline's data flow is honest: Stage 4 consumes Stage 3 output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from aquapose.core.tracking.types import FishTrack
from aquapose.tracking.tracker import (
    DEFAULT_BIRTH_INTERVAL,
    DEFAULT_DEDUP_COVISIBILITY_THRESHOLD,
    DEFAULT_DEDUP_DISTANCE,
    DEFAULT_DEDUP_WINDOW,
    DEFAULT_MAX_AGE,
    DEFAULT_MIN_CAMERAS_BIRTH,
    DEFAULT_MIN_HITS,
    DEFAULT_REPROJECTION_THRESHOLD,
    DEFAULT_VELOCITY_DAMPING,
    DEFAULT_VELOCITY_WINDOW,
    FishTracker,
)

__all__ = ["HungarianBackend"]

logger = logging.getLogger(__name__)


class HungarianBackend:
    """Temporal tracking backend wrapping FishTracker, consuming Stage 3 bundles.

    Maintains a FishTracker instance at construction time. The tracker is
    stateful and persists across frames, providing continuous fish identity
    assignment (TRACK-04: population constraint — dead IDs are recycled for
    new fish).

    Calibration is loaded eagerly at construction to enable lifecycle
    parameters that refer to the refractive scale of the scene.

    Bundles from Stage 3 (AssociationStage) are the primary input to
    ``track_frame()``. Each bundle carries a pre-computed 3D centroid and
    camera detection mapping, so FishTracker.update_from_bundles() assigns
    tracks via nearest-3D-centroid matching — no internal RANSAC re-association
    is performed.

    Args:
        calibration_path: Path to the AquaCal calibration JSON file.
        expected_count: Expected number of fish (population constraint).
        min_hits: Consecutive frames before a new track is confirmed.
        max_age: Grace period (frames) before a confirmed track is deleted.
        reprojection_threshold: Maximum pixel distance for track claiming.
        birth_interval: Frames between periodic birth RANSAC attempts.
        min_cameras_birth: Minimum cameras required to birth a new track.
        velocity_damping: Per-frame velocity damping during coasting.
        velocity_window: Number of recent frame-to-frame deltas to average.

    Raises:
        FileNotFoundError: If *calibration_path* does not exist.
    """

    def __init__(
        self,
        calibration_path: str | Path,
        expected_count: int = 9,
        min_hits: int = DEFAULT_MIN_HITS,
        max_age: int = DEFAULT_MAX_AGE,
        reprojection_threshold: float = DEFAULT_REPROJECTION_THRESHOLD,
        birth_interval: int = DEFAULT_BIRTH_INTERVAL,
        min_cameras_birth: int = DEFAULT_MIN_CAMERAS_BIRTH,
        velocity_damping: float = DEFAULT_VELOCITY_DAMPING,
        velocity_window: int = DEFAULT_VELOCITY_WINDOW,
        dedup_distance: float = DEFAULT_DEDUP_DISTANCE,
        dedup_window: int = DEFAULT_DEDUP_WINDOW,
        dedup_covisibility: float = DEFAULT_DEDUP_COVISIBILITY_THRESHOLD,
        **kwargs: Any,
    ) -> None:
        calib_path = Path(calibration_path)
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")

        self._tracker = FishTracker(
            min_hits=min_hits,
            max_age=max_age,
            expected_count=expected_count,
            min_cameras_birth=min_cameras_birth,
            reprojection_threshold=reprojection_threshold,
            birth_interval=birth_interval,
            velocity_damping=velocity_damping,
            velocity_window=velocity_window,
            dedup_distance=dedup_distance,
            dedup_window=dedup_window,
            dedup_covisibility=dedup_covisibility,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track_frame(
        self,
        frame_idx: int,
        bundles: list[Any],
    ) -> list[FishTrack]:
        """Process one frame and return confirmed fish tracks.

        Delegates to FishTracker.update_from_bundles(), which performs greedy
        nearest-3D-centroid matching between predicted track positions and the
        pre-associated bundle centroids from Stage 3. No internal RANSAC
        re-association is performed.

        Args:
            frame_idx: Frame index for bookkeeping.
            bundles: List of ``AssociationBundle`` objects from Stage 3.
                Each bundle has ``centroid_3d``, ``camera_detections``,
                ``n_cameras``, and ``reprojection_residual``.

        Returns:
            List of confirmed FishTrack objects after processing this frame.
        """
        return self._tracker.update_from_bundles(
            bundles=bundles,
            frame_index=frame_idx,
        )
