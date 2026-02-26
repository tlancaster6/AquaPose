"""Hungarian 3D tracking backend for the Tracking stage (Stage 4).

Wraps the v1.0 FishTracker to provide exact behavioral equivalence. The
backend is stateful — the FishTracker instance persists across frames,
maintaining track identities and lifecycle state.

Design note (v1.0 debt): FishTracker.update() expects raw detections_per_camera
and performs its own cross-camera association internally (claim_detections_for_tracks,
discover_births). Stage 3 (AssociationStage) already computed cross-camera bundles,
but to preserve exact v1.0 numerical equivalence, this backend passes the raw
detections to FishTracker.update() directly. The Stage 3 associated_bundles are
a data product for future backends and observers but are NOT consumed here in
v1.0-equivalent mode. This is intentional design debt — documented in the bug
ledger as: "Stage 3 output not consumed by Stage 4 — Stage 4 re-derives
association internally via FishTracker.update() to preserve v1.0 equivalence."
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
    """Temporal tracking backend wrapping v1.0 FishTracker.

    Maintains a FishTracker instance at construction time. The tracker is
    stateful and persists across frames, providing continuous fish identity
    assignment (TRACK-04: population constraint — dead IDs are recycled for
    new fish).

    Calibration is loaded eagerly at construction to build the per-camera
    RefractiveProjectionModel dict required by FishTracker.update().

    Design note (v1.0 debt): FishTracker.update() receives raw
    detections_per_camera and internally re-derives cross-camera association.
    The AssociationBundle list from Stage 3 is available in track_frame() but
    is NOT forwarded to FishTracker.update() in this backend — exact v1.0
    numerical equivalence is preserved by passing raw detections directly.
    See module docstring for full rationale.

    Args:
        calibration_path: Path to the AquaCal calibration JSON file.
        expected_count: Expected number of fish (population constraint).
        skip_camera_id: Camera ID to exclude from tracking.
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
        skip_camera_id: str = "e3v8250",
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

        self._skip_camera_id = skip_camera_id
        self._models = self._load_models(calib_path, skip_camera_id)

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
        detections_per_camera: dict[str, list[Any]],
    ) -> list[FishTrack]:
        """Process one frame and return confirmed fish tracks.

        Passes raw detections_per_camera to FishTracker.update() to preserve
        exact v1.0 behavioral equivalence. The bundles parameter (Stage 3 output)
        is available for future backends but is NOT consumed here.

        Args:
            frame_idx: Frame index for bookkeeping.
            bundles: AssociationBundle list from Stage 3 — available for future
                backends but not consumed in v1.0-equivalent mode.
            detections_per_camera: Raw per-camera detection lists (from Stage 1).
                Passed directly to FishTracker.update() for track claiming and
                birth discovery, replicating exact v1.0 behavior.

        Returns:
            List of confirmed FishTrack objects after processing this frame.
        """
        # Filter out the skip camera before passing to tracker.
        filtered_detections = {
            cam_id: dets
            for cam_id, dets in detections_per_camera.items()
            if cam_id != self._skip_camera_id
        }

        # Filter models to cameras with detections (avoids projection overhead
        # for cameras not contributing to this frame).
        active_models = {
            cam_id: model
            for cam_id, model in self._models.items()
            if cam_id in filtered_detections
        }

        return self._tracker.update(
            detections_per_camera=filtered_detections,
            models=active_models,
            frame_index=frame_idx,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_models(
        calibration_path: Path,
        skip_camera_id: str,
    ) -> dict[str, Any]:
        """Load calibration and build per-camera RefractiveProjectionModel dict.

        Args:
            calibration_path: Path to AquaCal calibration JSON.
            skip_camera_id: Camera ID to exclude from the returned dict.

        Returns:
            Dict mapping camera_id to RefractiveProjectionModel.
        """
        from aquapose.calibration.loader import load_calibration_data
        from aquapose.calibration.projection import RefractiveProjectionModel

        calib = load_calibration_data(str(calibration_path))
        models: dict[str, Any] = {}
        for cam_id, cam_data in calib.cameras.items():
            if cam_id == skip_camera_id:
                continue
            models[cam_id] = RefractiveProjectionModel(
                camera=cam_data,
                water_z=calib.water_z,
                interface_normal=calib.interface_normal,
                n_air=calib.n_air,
                n_water=calib.n_water,
            )
        return models
