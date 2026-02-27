"""Domain types for the 2D Tracking stage (Stage 2) and downstream consumers.

Defines Tracklet2D — the per-camera temporal tracklet produced by Stage 2 — and
re-exports FishTrack, TrackState, and TrackHealth as canonical frozen domain types
for reconstruction compatibility. FishTrack and TrackState are defined here in
core/ (not in the legacy tracking/ package) as of v2.1.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field

import numpy as np

__all__ = ["FishTrack", "TrackHealth", "TrackState", "Tracklet2D"]

# ---------------------------------------------------------------------------
# v2.1 domain type: Tracklet2D
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tracklet2D:
    """Per-camera temporal tracklet from Stage 2 (2D Tracking).

    Represents a single fish's track within one camera over a batch of frames.
    Tracklets are the outputs of the per-camera 2D tracker (e.g., OC-SORT) and
    the inputs to Stage 3 (Association), which groups them across cameras into
    TrackletGroup objects.

    All sequence fields use ``tuple`` (not ``list``) for immutability — the
    dataclass is frozen and these fields must not be mutated after creation.

    Attributes:
        camera_id: Which camera produced this tracklet.
        track_id: Unique tracklet identifier within this camera (local, not global).
            Track IDs are NOT comparable across cameras.
        frames: Ordered frame indices where the tracklet is active.
            Type: ``tuple[int, ...]``
        centroids: Per-frame (u, v) pixel centroids, one per entry in ``frames``.
            Type: ``tuple[tuple[float, float], ...]``
        bboxes: Per-frame bounding boxes as (x, y, w, h), one per entry in ``frames``.
            Type: ``tuple[tuple[float, float, float, float], ...]``
        frame_status: Per-frame detection status, one per entry in ``frames``.
            Each value is ``"detected"`` (directly observed) or ``"coasted"``
            (position interpolated during a missed detection).
            Type: ``tuple[str, ...]``
    """

    camera_id: str
    track_id: int
    frames: tuple
    centroids: tuple
    bboxes: tuple
    frame_status: tuple


# ---------------------------------------------------------------------------
# Legacy compatibility types: FishTrack, TrackState, TrackHealth
#
# These types are kept for reconstruction compatibility (used by Stage 5 and
# visualization). They are no longer populated by TrackingStage (which is
# deleted in v2.1). Reconstruction stage will be updated in Phase 26 to
# consume TrackletGroup instead. Until then, these types remain importable
# from aquapose.core.tracking.
# ---------------------------------------------------------------------------

# Default constants for FishTrack (kept from v1.0 tracker)
_DEFAULT_MIN_HITS: int = 5
_DEFAULT_MAX_AGE: int = 7
_DEFAULT_VELOCITY_DAMPING: float = 0.8
_DEFAULT_DEDUP_WINDOW: int = 10
_DEFAULT_VELOCITY_WINDOW: int = 5


class TrackState(enum.Enum):
    """Lifecycle state of a fish track."""

    PROBATIONARY = "probationary"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    DEAD = "dead"


@dataclass
class TrackHealth:
    """Running health statistics for a fish track.

    Attributes:
        residual_history: Ring buffer of recent reprojection residuals.
        cameras_per_frame: Ring buffer of recent camera counts.
        degraded_frames: Consecutive single-view updates.
        consecutive_hits: Consecutive successful matches.
    """

    residual_history: deque = field(default_factory=lambda: deque(maxlen=20))
    cameras_per_frame: deque = field(default_factory=lambda: deque(maxlen=20))
    degraded_frames: int = 0
    consecutive_hits: int = 0

    @property
    def mean_residual(self) -> float:
        """Mean of recent reprojection residuals, or 0.0 if empty.

        Returns:
            Mean residual value.
        """
        if not self.residual_history:
            return 0.0
        return float(np.mean(list(self.residual_history)))

    @property
    def mean_cameras(self) -> float:
        """Mean of recent camera counts, or 0.0 if empty.

        Returns:
            Mean camera count.
        """
        if not self.cameras_per_frame:
            return 0.0
        return float(np.mean(list(self.cameras_per_frame)))


@dataclass
class FishTrack:
    """Persistent track for a single fish across frames.

    Legacy compatibility type from v1.0. In v2.1, Stage 2 produces Tracklet2D
    objects (not FishTrack objects). FishTrack is retained here so that
    reconstruction/stage.py and visualization modules can continue importing it
    without modification until Phase 26 updates them to use TrackletGroup.

    Attributes:
        fish_id: Globally unique identifier for this fish track.
        positions: Ring buffer of the 2 most recent 3D centroids, shape (3,) each.
        velocity: Explicit velocity vector, shape (3,).
        velocity_history: Ring buffer of recent frame-to-frame position deltas.
        velocity_window: Number of recent deltas to average for velocity smoothing.
        age: Total number of frames since this track was created.
        frames_since_update: Frames elapsed since the last successful match.
        state: Current lifecycle state.
        health: Running health statistics.
        min_hits: Confirmation threshold.
        max_age: Death threshold.
        velocity_damping: Per-frame velocity damping during coasting.
        camera_detections: Latest frame's camera_id to detection_index map.
        camera_history: Ring buffer of recent camera sets for deduplication.
        bboxes: Latest frame's per-camera bounding boxes.
        reprojection_residual: Mean reprojection residual from the latest association.
        confidence: Confidence from the latest association.
        n_cameras: Number of cameras observing this fish in the latest frame.
        _coasting_prediction: Running predicted position during coasting.
    """

    fish_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity_window: int = _DEFAULT_VELOCITY_WINDOW
    velocity_history: deque = field(init=False, repr=False)
    age: int = 0
    frames_since_update: int = 0
    state: TrackState = TrackState.PROBATIONARY
    health: TrackHealth = field(default_factory=TrackHealth)
    min_hits: int = _DEFAULT_MIN_HITS
    max_age: int = _DEFAULT_MAX_AGE
    velocity_damping: float = _DEFAULT_VELOCITY_DAMPING
    camera_detections: dict = field(default_factory=dict)
    camera_history: deque = field(
        default_factory=lambda: deque(maxlen=_DEFAULT_DEDUP_WINDOW)
    )
    bboxes: dict = field(default_factory=dict)
    reprojection_residual: float = 0.0
    confidence: float = 1.0
    n_cameras: int = 0
    _coasting_prediction: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize fields that depend on other field values."""
        self.velocity_history = deque(maxlen=self.velocity_window)

    @property
    def is_confirmed(self) -> bool:
        """True when the track is confirmed or coasting.

        Returns:
            True if state is CONFIRMED or COASTING.
        """
        return self.state in (TrackState.CONFIRMED, TrackState.COASTING)

    def predict(self) -> np.ndarray:
        """Predict the next 3D position using constant-velocity extrapolation.

        Returns:
            Predicted 3D centroid, shape (3,).
        """
        if len(self.positions) == 0:
            return np.zeros(3, dtype=np.float32)

        if self._coasting_prediction is not None:
            return self._coasting_prediction.copy()

        last_pos = list(self.positions)[-1]
        return last_pos + self.velocity

    def update_from_claim(
        self,
        centroid_3d: np.ndarray,
        camera_detections: dict,
        reprojection_residual: float,
        n_cameras: int,
    ) -> None:
        """Update the track with a claimed detection result.

        Args:
            centroid_3d: New 3D centroid, shape (3,).
            camera_detections: Camera-to-detection-index mapping.
            reprojection_residual: Mean reprojection residual (pixels).
            n_cameras: Number of cameras contributing.
        """
        if len(self.positions) > 0:
            prev = list(self.positions)[-1]
            delta = (centroid_3d - prev).astype(np.float32)
            self.velocity_history.append(delta)
            self.velocity = np.mean(list(self.velocity_history), axis=0).astype(
                np.float32
            )
        else:
            self.velocity = np.zeros(3, dtype=np.float32)
            self.velocity_history.clear()

        self.positions.append(centroid_3d.copy())
        self.frames_since_update = 0
        self._coasting_prediction = None
        self.age += 1
        self.camera_detections = dict(camera_detections)
        self.camera_history.append(frozenset(camera_detections.keys()))
        self.bboxes = {}
        self.reprojection_residual = reprojection_residual
        self.confidence = 1.0
        self.n_cameras = n_cameras

        self.health.residual_history.append(reprojection_residual)
        self.health.cameras_per_frame.append(n_cameras)
        self.health.consecutive_hits += 1
        if n_cameras > 1:
            self.health.degraded_frames = 0

        if self.state == TrackState.PROBATIONARY:
            if self.health.consecutive_hits >= self.min_hits:
                self.state = TrackState.CONFIRMED
        elif self.state == TrackState.COASTING:
            self.state = TrackState.CONFIRMED

    def update_position_only(
        self,
        centroid_3d: np.ndarray,
        camera_detections: dict,
        reprojection_residual: float,
        n_cameras: int,
    ) -> None:
        """Update position but freeze velocity (single-view penalty).

        Args:
            centroid_3d: New 3D centroid, shape (3,).
            camera_detections: Camera-to-detection-index mapping.
            reprojection_residual: Mean reprojection residual (pixels).
            n_cameras: Number of cameras contributing.
        """
        self.positions.append(centroid_3d.copy())
        self.frames_since_update = 0
        self._coasting_prediction = None
        self.age += 1
        self.camera_detections = dict(camera_detections)
        self.camera_history.append(frozenset(camera_detections.keys()))
        self.bboxes = {}
        self.reprojection_residual = reprojection_residual
        self.confidence = 1.0
        self.n_cameras = n_cameras

        self.health.residual_history.append(reprojection_residual)
        self.health.cameras_per_frame.append(n_cameras)
        self.health.consecutive_hits += 1
        self.health.degraded_frames += 1

        if self.state == TrackState.PROBATIONARY:
            if self.health.consecutive_hits >= self.min_hits:
                self.state = TrackState.CONFIRMED
        elif self.state == TrackState.COASTING:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        """Record that this track had no match in the current frame.

        CONFIRMED transitions to COASTING. PROBATIONARY remains PROBATIONARY.
        Advances the coasting prediction by one damped velocity step.
        """
        self.frames_since_update += 1
        self.health.consecutive_hits = 0
        self.age += 1

        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.COASTING
            if len(self.positions) > 0:
                last_pos = list(self.positions)[-1]
                self._coasting_prediction = (
                    last_pos + self.velocity * self.velocity_damping
                )
            else:
                self._coasting_prediction = None
        elif self.state == TrackState.COASTING:
            if self._coasting_prediction is not None:
                self._coasting_prediction = (
                    self._coasting_prediction + self.velocity * self.velocity_damping
                )
            elif len(self.positions) > 0:
                last_pos = list(self.positions)[-1]
                self._coasting_prediction = (
                    last_pos + self.velocity * self.velocity_damping
                )

    @property
    def is_dead(self) -> bool:
        """True when the track should be pruned.

        Returns:
            True if the track has exceeded its grace period.
        """
        if self.state == TrackState.PROBATIONARY:
            return self.frames_since_update > 2
        return self.frames_since_update > self.max_age
