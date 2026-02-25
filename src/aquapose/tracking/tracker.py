"""Temporal fish tracker with track-driven association and lifecycle management."""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.segmentation.detector import Detection

from .associate import (
    AssociationResult,
    UnclaimedInfo,
    claim_detections_for_tracks,
    discover_births,
)

# --- Module-level defaults ---

DEFAULT_MIN_HITS: int = 5
"""Frames before a track is confirmed (birth confirmation)."""

DEFAULT_MAX_AGE: int = 7
"""Grace period frames before a track is declared dead."""

DEFAULT_REPROJECTION_THRESHOLD: float = 15.0
"""Maximum pixel distance for a track to claim a detection."""

DEFAULT_MIN_CAMERAS_BIRTH: int = 3
"""Minimum cameras in an association to birth a new track."""

DEFAULT_BIRTH_INTERVAL: int = 30
"""Frames between birth RANSAC attempts (when not triggered by deficit)."""

DEFAULT_RESIDUAL_REJECT_FACTOR: float = 3.0
"""Reject a claim if residual exceeds mean_residual * this factor."""

DEFAULT_VELOCITY_DAMPING: float = 0.8
"""Per-frame velocity damping factor during coasting."""

DEFAULT_DEDUP_DISTANCE: float = 0.04
"""3D proximity threshold (metres) for confirmed-track deduplication."""

DEFAULT_BIRTH_PROXIMITY_DISTANCE: float = 0.08
"""3D proximity threshold (metres) for pre-birth ghost rejection.

Larger than dedup_distance (4 cm) because cross-fish phantoms (triangulated
from one detection of each fish) land at the midpoint between fish.  The
synthetic trajectory generator enforces a minimum fish separation of 17 cm
(2 x body length), so the nearest a real phantom can appear to a legitimate
fish is ~8.5 cm.  An 8 cm threshold therefore catches the majority of
cross-fish phantoms (those from fish pairs > 16 cm apart, i.e. almost all
real-scenario pairs) while not blocking legitimate births from newly arriving
fish that are physically separated by the collision-avoidance minimum.
"""

DEFAULT_DEDUP_WINDOW: int = 10
"""Consecutive frames two tracks must be close before dedup fires."""

DEFAULT_DEDUP_COVISIBILITY_THRESHOLD: float = 0.2
"""Maximum camera co-visibility ratio to consider tracks duplicates."""

DEFAULT_VELOCITY_WINDOW: int = 5
"""Number of recent frame-to-frame deltas to average for velocity smoothing."""


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

    residual_history: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    cameras_per_frame: deque[int] = field(default_factory=lambda: deque(maxlen=20))
    degraded_frames: int = 0
    consecutive_hits: int = 0

    @property
    def mean_residual(self) -> float:
        """Mean of recent reprojection residuals, or 0.0 if empty."""
        if not self.residual_history:
            return 0.0
        return float(np.mean(list(self.residual_history)))

    @property
    def mean_cameras(self) -> float:
        """Mean of recent camera counts, or 0.0 if empty."""
        if not self.cameras_per_frame:
            return 0.0
        return float(np.mean(list(self.cameras_per_frame)))


@dataclass
class FishTrack:
    """Persistent track for a single fish across frames.

    Attributes:
        fish_id: Globally unique identifier for this fish track.
        positions: Ring buffer of the 2 most recent 3D centroids (shape (3,)
            each). Used for constant-velocity prediction.
        velocity: Explicit velocity vector, shape (3,). Set to the windowed
            mean of recent frame-to-frame deltas by ``update_from_claim``.
        velocity_history: Ring buffer of recent frame-to-frame position deltas
            (shape (3,) each). Length capped at ``velocity_window``. Used to
            compute a smoothed velocity estimate that is more robust to noisy
            3D triangulated positions.
        velocity_window: Number of recent deltas to average for velocity
            smoothing (default ``DEFAULT_VELOCITY_WINDOW``). Setting this to 1
            reproduces the legacy single-frame delta behaviour.
        age: Total number of frames since this track was created.
        frames_since_update: Frames elapsed since the last successful match.
        state: Current lifecycle state.
        health: Running health statistics.
        min_hits: Confirmation threshold (copied from FishTracker at creation).
        max_age: Death threshold (copied from FishTracker at creation).
        velocity_damping: Per-frame velocity damping during coasting.
        camera_detections: Latest frame's ``camera_id → detection_index`` map.
        bboxes: Latest frame's per-camera bounding boxes.
        reprojection_residual: Mean reprojection residual from the latest
            association (pixels).
        confidence: Confidence from the latest association.
        n_cameras: Number of cameras observing this fish in the latest frame.
        _coasting_prediction: Running predicted position advanced each coasting
            frame. None when not coasting (predict() uses positions directly).
    """

    fish_id: int
    positions: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity_window: int = DEFAULT_VELOCITY_WINDOW
    velocity_history: deque[np.ndarray] = field(init=False, repr=False)
    age: int = 0
    frames_since_update: int = 0
    state: TrackState = TrackState.PROBATIONARY
    health: TrackHealth = field(default_factory=TrackHealth)
    min_hits: int = DEFAULT_MIN_HITS
    max_age: int = DEFAULT_MAX_AGE
    velocity_damping: float = DEFAULT_VELOCITY_DAMPING
    camera_detections: dict[str, int] = field(default_factory=dict)
    camera_history: deque[frozenset[str]] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_DEDUP_WINDOW)
    )
    bboxes: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    reprojection_residual: float = 0.0
    confidence: float = 1.0
    n_cameras: int = 0
    _coasting_prediction: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize fields that depend on other field values."""
        self.velocity_history = deque(maxlen=self.velocity_window)

    @property
    def is_confirmed(self) -> bool:
        """True when the track is confirmed or coasting (writer compat).

        Returns:
            True if state is CONFIRMED or COASTING.
        """
        return self.state in (TrackState.CONFIRMED, TrackState.COASTING)

    def predict(self) -> np.ndarray:
        """Predict the next 3D position using constant-velocity extrapolation.

        When coasting, returns the running ``_coasting_prediction`` that is
        advanced by ``mark_missed()`` each frame.  This ensures the prediction
        keeps pace with the fish rather than stalling at the last observed
        position with a rapidly-decaying velocity offset.

        When active (not coasting), adds one step of velocity to the last
        observed position.

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
        camera_detections: dict[str, int],
        reprojection_residual: float,
        n_cameras: int,
    ) -> None:
        """Update the track with a claimed detection result.

        Computes velocity from previous position delta, updates health,
        and handles state transitions.

        Args:
            centroid_3d: New 3D centroid, shape (3,).
            camera_detections: Camera-to-detection-index mapping.
            reprojection_residual: Mean reprojection residual (pixels).
            n_cameras: Number of cameras contributing.
        """
        # Compute velocity from position delta using windowed smoothing.
        # Append the raw frame delta to the ring buffer and set velocity to
        # the mean of all deltas in the buffer (up to velocity_window frames).
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
        self._coasting_prediction = None  # reset on re-acquisition
        self.age += 1
        self.camera_detections = dict(camera_detections)
        self.camera_history.append(frozenset(camera_detections.keys()))
        self.bboxes = {}
        self.reprojection_residual = reprojection_residual
        self.confidence = 1.0
        self.n_cameras = n_cameras

        # Update health
        self.health.residual_history.append(reprojection_residual)
        self.health.cameras_per_frame.append(n_cameras)
        self.health.consecutive_hits += 1
        if n_cameras > 1:
            self.health.degraded_frames = 0

        # State transitions
        if self.state == TrackState.PROBATIONARY:
            if self.health.consecutive_hits >= self.min_hits:
                self.state = TrackState.CONFIRMED
        elif self.state == TrackState.COASTING:
            self.state = TrackState.CONFIRMED

    def update_position_only(
        self,
        centroid_3d: np.ndarray,
        camera_detections: dict[str, int],
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
        self._coasting_prediction = None  # reset on re-acquisition
        self.age += 1
        self.camera_detections = dict(camera_detections)
        self.camera_history.append(frozenset(camera_detections.keys()))
        self.bboxes = {}
        self.reprojection_residual = reprojection_residual
        self.confidence = 1.0
        self.n_cameras = n_cameras

        # Health: push stats but increment degraded_frames
        self.health.residual_history.append(reprojection_residual)
        self.health.cameras_per_frame.append(n_cameras)
        self.health.consecutive_hits += 1
        self.health.degraded_frames += 1
        # Velocity frozen — no change

        # State transitions (same as full update)
        if self.state == TrackState.PROBATIONARY:
            if self.health.consecutive_hits >= self.min_hits:
                self.state = TrackState.CONFIRMED
        elif self.state == TrackState.COASTING:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        """Record that this track had no match in the current frame.

        CONFIRMED → COASTING, reset consecutive_hits.
        PROBATIONARY stays PROBATIONARY.

        Advances the coasting prediction by one damped velocity step so that
        subsequent ``predict()`` calls continue tracking the fish trajectory
        rather than stalling at the last observed position.
        """
        self.frames_since_update += 1
        self.health.consecutive_hits = 0
        self.age += 1

        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.COASTING
            # Seed the coasting prediction from the last observed position.
            if len(self.positions) > 0:
                last_pos = list(self.positions)[-1]
                self._coasting_prediction = (
                    last_pos + self.velocity * self.velocity_damping
                )
            else:
                self._coasting_prediction = None
        elif self.state == TrackState.COASTING:
            # Advance the running prediction by one more damped velocity step.
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

        Confirmed/coasting tracks die after max_age missed frames.
        Probationary tracks have a short leash: die after 2 missed frames.

        Returns:
            True if the track has exceeded its grace period.
        """
        if self.state == TrackState.PROBATIONARY:
            return self.frames_since_update > 2
        return self.frames_since_update > self.max_age


class FishTracker:
    """Temporal tracker with track-driven association.

    Architecture:
    1. Track claiming: existing tracks project into cameras, claim nearest
       detections via greedy assignment.
    2. Birth discovery: unclaimed detections go through RANSAC to find new fish.
    3. Lifecycle: probationary → confirmed → coasting → dead.

    Population constraint (TRACK-04): if a track dies in the same frame as a
    new unmatched observation appears, the new observation inherits the dead
    track's fish_id instead of receiving a fresh one.

    Args:
        min_hits: Consecutive frames required to confirm a new track.
        max_age: Grace period (frames) before a confirmed track is deleted.
        expected_count: Expected number of fish (default 9).
        min_cameras_birth: Minimum cameras to birth a new track.
        reprojection_threshold: Max pixel distance for track claiming.
        birth_interval: Frames between periodic birth RANSAC.
        residual_reject_factor: Reject claim if residual > mean * this.
        velocity_damping: Per-frame velocity damping during coasting.
        velocity_window: Number of recent frame-to-frame deltas to average
            for velocity smoothing (default ``DEFAULT_VELOCITY_WINDOW``).
            Setting this to 1 reproduces the legacy single-frame delta
            behaviour.
        dedup_distance: 3D proximity threshold (metres) for confirmed-track
            deduplication (covisibility-based dedup).
        birth_proximity_distance: 3D proximity threshold (metres) for
            pre-birth ghost rejection. Larger than dedup_distance because
            cross-fish phantoms land at the midpoint between two fish
            (typically 15-25 cm away) which is too far from either fish to
            be caught by the smaller dedup_distance threshold.
        dedup_window: Consecutive close frames required before dedup fires.
        dedup_covisibility: Maximum camera co-visibility ratio to consider
            two tracks duplicates (low = they're splitting one fish's
            detections rather than tracking two distinct fish).
    """

    def __init__(
        self,
        min_hits: int = DEFAULT_MIN_HITS,
        max_age: int = DEFAULT_MAX_AGE,
        expected_count: int = 9,
        min_cameras_birth: int = DEFAULT_MIN_CAMERAS_BIRTH,
        reprojection_threshold: float = DEFAULT_REPROJECTION_THRESHOLD,
        birth_interval: int = DEFAULT_BIRTH_INTERVAL,
        residual_reject_factor: float = DEFAULT_RESIDUAL_REJECT_FACTOR,
        velocity_damping: float = DEFAULT_VELOCITY_DAMPING,
        velocity_window: int = DEFAULT_VELOCITY_WINDOW,
        dedup_distance: float = DEFAULT_DEDUP_DISTANCE,
        birth_proximity_distance: float = DEFAULT_BIRTH_PROXIMITY_DISTANCE,
        dedup_window: int = DEFAULT_DEDUP_WINDOW,
        dedup_covisibility: float = DEFAULT_DEDUP_COVISIBILITY_THRESHOLD,
    ) -> None:
        self.min_hits = min_hits
        self.max_age = max_age
        self.expected_count = expected_count
        self.min_cameras_birth = min_cameras_birth
        self.reprojection_threshold = reprojection_threshold
        self.birth_interval = birth_interval
        self.residual_reject_factor = residual_reject_factor
        self.velocity_damping = velocity_damping
        self.velocity_window = velocity_window
        self.dedup_distance = dedup_distance
        self.birth_proximity_distance = birth_proximity_distance
        self.dedup_window = dedup_window
        self.dedup_covisibility = dedup_covisibility

        self.tracks: list[FishTrack] = []
        self._next_id: int = 0
        self.frame_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections_per_camera: dict[str, list[Detection]],
        models: dict[str, RefractiveProjectionModel],
        frame_index: int = 0,
    ) -> list[FishTrack]:
        """Process one frame of detections and return confirmed tracks.

        Three phases:
        1. Track claiming — existing tracks claim detections via reprojection.
        2. Birth discovery — unclaimed detections go through RANSAC.
        3. Lifecycle — prune dead tracks, recycle IDs.

        Args:
            detections_per_camera: Per-camera detection lists.
            models: Per-camera refractive projection models.
            frame_index: Frame index for bookkeeping.

        Returns:
            List of confirmed tracks after processing this frame.
        """
        # ----------------------------------------------------------
        # Phase 1: Track Claiming
        # ----------------------------------------------------------
        non_dead = [t for t in self.tracks if not t.is_dead]
        claimed_track_ids: set[int] = set()
        unclaimed_info = UnclaimedInfo(indices={}, min_claim_distance={})

        if non_dead and any(len(dets) > 0 for dets in detections_per_camera.values()):
            predicted_positions: dict[int, np.ndarray] = {}
            track_priorities: dict[int, int] = {}
            for t in non_dead:
                predicted_positions[t.fish_id] = t.predict()
                # Confirmed/coasting get priority 0, probationary gets 1
                track_priorities[t.fish_id] = (
                    0 if t.state in (TrackState.CONFIRMED, TrackState.COASTING) else 1
                )

            claims, unclaimed_info = claim_detections_for_tracks(
                predicted_positions=predicted_positions,
                track_priorities=track_priorities,
                detections_per_camera=detections_per_camera,
                models=models,
                reprojection_threshold=self.reprojection_threshold,
            )

            # Apply claims to tracks
            track_map = {t.fish_id: t for t in non_dead}
            for claim in claims:
                track = track_map[claim.track_id]
                claimed_track_ids.add(claim.track_id)

                # Residual validation: reject if residual is anomalously high.
                # Require at least 3 history entries for a statistically
                # meaningful baseline; use absolute floor to avoid false
                # rejection when baseline residual is near zero.
                mean_res = track.health.mean_residual
                residual_floor = self.reprojection_threshold * 0.5
                if len(
                    track.health.residual_history
                ) >= 3 and claim.reprojection_residual > max(
                    mean_res * self.residual_reject_factor,
                    residual_floor,
                ):
                    # Reject — coast on prediction
                    track.mark_missed()
                    continue

                # Single-view penalty: update position but freeze velocity
                if claim.n_cameras == 1:
                    track.update_position_only(
                        centroid_3d=claim.centroid_3d,
                        camera_detections=claim.camera_detections,
                        reprojection_residual=claim.reprojection_residual,
                        n_cameras=claim.n_cameras,
                    )
                else:
                    track.update_from_claim(
                        centroid_3d=claim.centroid_3d,
                        camera_detections=claim.camera_detections,
                        reprojection_residual=claim.reprojection_residual,
                        n_cameras=claim.n_cameras,
                    )

            # Mark unclaimed tracks as missed
            for t in non_dead:
                if t.fish_id not in claimed_track_ids:
                    t.mark_missed()
        else:
            # No tracks or no detections — all tracks missed
            for t in non_dead:
                t.mark_missed()
            # All detections are unclaimed (no tracks → infinite distance)
            unclaimed_idx = {
                cam: list(range(len(dets)))
                for cam, dets in detections_per_camera.items()
                if len(dets) > 0
            }
            unclaimed_min_dist = {
                cam: {i: float("inf") for i in indices}
                for cam, indices in unclaimed_idx.items()
            }
            unclaimed_info = UnclaimedInfo(
                indices=unclaimed_idx, min_claim_distance=unclaimed_min_dist
            )

        # ----------------------------------------------------------
        # Phase 1.5: Deduplicate confirmed tracks (disabled for testing)
        # ----------------------------------------------------------
        # self._dedup_confirmed_tracks()

        # ----------------------------------------------------------
        # Phase 2: Birth Discovery
        # ----------------------------------------------------------
        # Use total non-dead track count (not just confirmed) to gate births.
        # During the probationary window confirmed_count is always 0, which
        # would trigger birth every frame even though all fish are already
        # being tracked as probationary. non_dead_count correctly reflects
        # how many fish are already accounted for.
        non_dead_count = len([t for t in self.tracks if not t.is_dead])
        total_unclaimed = sum(len(v) for v in unclaimed_info.indices.values())

        should_birth = (
            self.frame_count == 0
            or non_dead_count < self.expected_count
            or self.frame_count % self.birth_interval == 0
        )

        if should_birth and total_unclaimed > 0:
            births = discover_births(
                unclaimed_indices=unclaimed_info.indices,
                detections_per_camera=detections_per_camera,
                models=models,
                expected_count=self.expected_count,
                reprojection_threshold=self.reprojection_threshold,
                min_cameras=self.min_cameras_birth,
                seed_points=self.get_seed_points(),
                near_claim_distances=unclaimed_info.min_claim_distance,
            )

            # TRACK-04: collect dead IDs for recycling
            dead_ids: list[int] = [t.fish_id for t in self.tracks if t.is_dead]
            dead_id_iter = iter(dead_ids)

            # Sort births by X for deterministic ID assignment on first frame
            if self.frame_count == 0:
                births = sorted(births, key=lambda a: float(a.centroid_3d[0]))

            for birth in births:
                # Pre-birth proximity check: reject if too close to an
                # existing non-dead track (prevents ghost duplicates).
                # Uses birth_proximity_distance (default 8 cm) rather than
                # dedup_distance (4 cm) because cross-fish phantoms land at
                # the midpoint between two fish (typically 15-25 cm away)
                # which is too far from either to be caught by dedup_distance.
                too_close = False
                for t in self.tracks:
                    if not t.is_dead and len(t.positions) > 0:
                        dist = float(
                            np.linalg.norm(birth.centroid_3d - list(t.positions)[-1])
                        )
                        if dist < self.birth_proximity_distance:
                            too_close = True
                            break
                if too_close:
                    continue

                dead_id = next(dead_id_iter, None)
                self._create_track(birth, fish_id_override=dead_id)

        # ----------------------------------------------------------
        # Phase 3: Lifecycle
        # ----------------------------------------------------------
        self.tracks = [t for t in self.tracks if not t.is_dead]
        self.frame_count += 1
        return [t for t in self.tracks if t.is_confirmed]

    def get_all_tracks(self) -> list[FishTrack]:
        """Return all tracks, including tentative (unconfirmed) ones.

        Returns:
            All current tracks.
        """
        return list(self.tracks)

    def get_seed_points(self) -> list[np.ndarray] | None:
        """Return predicted 3D positions of confirmed tracks.

        Returns:
            List of shape-(3,) arrays (predicted positions), or None if
            no confirmed tracks exist.
        """
        confirmed = [t for t in self.tracks if t.is_confirmed]
        if not confirmed:
            return None
        seeds: list[np.ndarray] = []
        for t in confirmed:
            seeds.append(t.predict())
        return seeds if seeds else None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dedup_confirmed_tracks(self) -> None:
        """Kill the younger of two confirmed tracks that appear to be duplicates.

        Two confirmed tracks are considered duplicates when:
        1. Their 3D centroids have been within ``dedup_distance`` for the last
           ``dedup_window`` consecutive frames (both must have full history).
        2. Their camera co-visibility ratio is below ``dedup_covisibility`` —
           meaning they rarely both have a detection in the same camera,
           indicating they're splitting one fish's detections rather than
           tracking two distinct fish.

        The younger track (higher fish_id) is killed.
        """
        confirmed = [t for t in self.tracks if t.is_confirmed]
        if len(confirmed) < 2:
            return

        kill_ids: set[int] = set()

        for i in range(len(confirmed)):
            if confirmed[i].fish_id in kill_ids:
                continue
            for j in range(i + 1, len(confirmed)):
                if confirmed[j].fish_id in kill_ids:
                    continue

                a, b = confirmed[i], confirmed[j]

                # Both must have full dedup_window of camera history
                if (
                    len(a.camera_history) < self.dedup_window
                    or len(b.camera_history) < self.dedup_window
                ):
                    continue

                # Check proximity over the window: use current positions
                # (camera_history length already guarantees they've been
                # updated for dedup_window consecutive frames)
                if len(a.positions) == 0 or len(b.positions) == 0:
                    continue
                pos_a = list(a.positions)[-1]
                pos_b = list(b.positions)[-1]
                dist = float(np.linalg.norm(pos_a - pos_b))
                if dist > self.dedup_distance:
                    continue

                # Camera co-visibility: fraction of frames where both tracks
                # had a detection in at least one common camera
                n_covisible = 0
                hist_a = list(a.camera_history)
                hist_b = list(b.camera_history)
                for cams_a, cams_b in zip(hist_a, hist_b, strict=True):
                    if cams_a & cams_b:  # intersection
                        n_covisible += 1
                covis_ratio = n_covisible / self.dedup_window

                if covis_ratio < self.dedup_covisibility:
                    # Kill the younger track (higher fish_id)
                    younger = a if a.fish_id > b.fish_id else b
                    younger.state = TrackState.DEAD
                    kill_ids.add(younger.fish_id)

    def _create_track(
        self,
        association: AssociationResult,
        fish_id_override: int | None = None,
    ) -> FishTrack:
        """Allocate a new FishTrack and add it to self.tracks.

        Args:
            association: Association to seed the new track with.
            fish_id_override: If provided, use this fish_id instead of
                allocating a fresh one (TRACK-04 population constraint).

        Returns:
            The newly created FishTrack.
        """
        if fish_id_override is not None:
            fish_id = fish_id_override
        else:
            fish_id = self._next_id
            self._next_id += 1

        track = FishTrack(
            fish_id=fish_id,
            min_hits=self.min_hits,
            max_age=self.max_age,
            velocity_damping=self.velocity_damping,
            velocity_window=self.velocity_window,
        )
        track.update_from_claim(
            centroid_3d=association.centroid_3d,
            camera_detections=dict(association.camera_detections),
            reprojection_residual=association.reprojection_residual,
            n_cameras=association.n_cameras,
        )
        self.tracks.append(track)
        return track
