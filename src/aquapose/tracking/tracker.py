"""Temporal fish tracker with Hungarian assignment and track lifecycle management."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .associate import AssociationResult, FrameAssociations

# --- Module-level defaults ---

DEFAULT_MIN_HITS: int = 2
"""Frames before a track is confirmed (birth confirmation)."""

DEFAULT_MAX_AGE: int = 7
"""Grace period frames before a track is declared dead."""

DEFAULT_MAX_DISTANCE: float = 0.1
"""Maximum XY distance (metres) for a Hungarian assignment to be accepted."""


@dataclass
class FishTrack:
    """Persistent track for a single fish across frames.

    Attributes:
        fish_id: Globally unique identifier for this fish track.
        positions: Ring buffer of the 2 most recent 3D centroids (shape (3,)
            each). Used for constant-velocity prediction.
        age: Total number of frames since this track was created.
        hit_streak: Consecutive frames in which this track was matched.
        frames_since_update: Frames elapsed since the last successful match.
        is_confirmed: True once ``hit_streak >= min_hits``.
        min_hits: Confirmation threshold (copied from FishTracker at creation).
        max_age: Death threshold (copied from FishTracker at creation).
        camera_detections: Latest frame's ``camera_id → detection_index`` map.
        bboxes: Latest frame's per-camera bounding boxes
            ``camera_id → (x1, y1, x2, y2)``.
        reprojection_residual: Mean reprojection residual from the latest
            association (pixels).
        confidence: Confidence from the latest association.
        n_cameras: Number of cameras observing this fish in the latest frame.
    """

    fish_id: int
    positions: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=2))
    age: int = 0
    hit_streak: int = 0
    frames_since_update: int = 0
    is_confirmed: bool = False
    min_hits: int = DEFAULT_MIN_HITS
    max_age: int = DEFAULT_MAX_AGE
    camera_detections: dict[str, int] = field(default_factory=dict)
    bboxes: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    reprojection_residual: float = 0.0
    confidence: float = 1.0
    n_cameras: int = 0

    def predict(self) -> np.ndarray:
        """Predict the next 3D position using constant-velocity motion.

        Uses the last two stored positions to extrapolate one step forward.
        If only one position is available, returns a copy of that position
        (zero velocity assumption).

        Returns:
            Predicted 3D centroid, shape (3,).
        """
        if len(self.positions) >= 2:
            pos_list = list(self.positions)
            velocity = pos_list[-1] - pos_list[-2]
            return pos_list[-1] + velocity
        elif len(self.positions) == 1:
            return self.positions[0].copy()
        else:
            return np.zeros(3, dtype=np.float32)

    def update(self, association: AssociationResult) -> None:
        """Update the track with a new matched association.

        Appends the 3D centroid to the position deque, resets
        ``frames_since_update``, increments ``hit_streak`` and ``age``, and
        sets ``is_confirmed`` once the streak reaches ``min_hits``.

        Args:
            association: The matched AssociationResult for this frame.
        """
        self.positions.append(association.centroid_3d.copy())
        self.frames_since_update = 0
        self.hit_streak += 1
        self.age += 1
        self.camera_detections = dict(association.camera_detections)
        self.bboxes = {}  # bboxes not carried in AssociationResult; cleared
        self.reprojection_residual = association.reprojection_residual
        self.confidence = association.confidence
        self.n_cameras = association.n_cameras
        if self.hit_streak >= self.min_hits:
            self.is_confirmed = True

    def mark_missed(self) -> None:
        """Record that this track had no match in the current frame.

        Increments ``frames_since_update`` and ``age``, resets ``hit_streak``
        to zero (confirmation requires consecutive hits).
        """
        self.frames_since_update += 1
        self.hit_streak = 0
        self.age += 1

    @property
    def is_dead(self) -> bool:
        """True when the track has exceeded the grace period.

        Returns:
            True if ``frames_since_update > max_age``.
        """
        return self.frames_since_update > self.max_age


class FishTracker:
    """Temporal tracker that maintains persistent fish IDs across frames.

    Uses a SORT-derived lifecycle:
    - Birth: tentative for ``min_hits`` frames, then confirmed.
    - Active: matched and updated every frame.
    - Grace period: unmatched for up to ``max_age`` frames.
    - Death: pruned after exceeding the grace period.

    Population constraint (TRACK-04): if a track dies in the same frame as a
    new unmatched observation appears, the new observation inherits the dead
    track's fish_id instead of receiving a fresh one.

    Args:
        min_hits: Consecutive frames required to confirm a new track.
        max_age: Grace period (frames) before a track is deleted.
        max_distance: Maximum XY Euclidean distance (metres) for a valid
            Hungarian assignment.
        expected_count: Expected number of fish (default 9). Used for
            first-frame batch initialisation ordering.
    """

    def __init__(
        self,
        min_hits: int = DEFAULT_MIN_HITS,
        max_age: int = DEFAULT_MAX_AGE,
        max_distance: float = DEFAULT_MAX_DISTANCE,
        expected_count: int = 9,
    ) -> None:
        self.min_hits = min_hits
        self.max_age = max_age
        self.max_distance = max_distance
        self.expected_count = expected_count

        self.tracks: list[FishTrack] = []
        self._next_id: int = 0
        self.frame_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame_associations: FrameAssociations) -> list[FishTrack]:
        """Process one frame of associations and return confirmed tracks.

        Args:
            frame_associations: Per-frame association results from
                ``ransac_centroid_cluster``.

        Returns:
            List of confirmed tracks (``is_confirmed=True``) after processing
            this frame.
        """
        observations = frame_associations.associations

        # --- Frame 0: batch-initialise, sorted by X for deterministic IDs ---
        if self.frame_count == 0:
            sorted_obs = sorted(observations, key=lambda a: float(a.centroid_3d[0]))
            for assoc in sorted_obs:
                self._create_track(assoc)
            self.frame_count += 1
            return [t for t in self.tracks if t.is_confirmed]

        # --- Predict positions for existing tracks ---
        predicted: list[np.ndarray] = [t.predict() for t in self.tracks]

        # --- Build XY-only cost matrix ---
        n_tracks = len(self.tracks)
        n_obs = len(observations)

        if n_tracks > 0 and n_obs > 0:
            cost_matrix = np.full(
                (n_tracks, n_obs),
                self.max_distance + 1.0,
                dtype=np.float64,
            )
            for r, pred in enumerate(predicted):
                for c, assoc in enumerate(observations):
                    cost_matrix[r, c] = float(
                        np.linalg.norm(pred[:2] - assoc.centroid_3d[:2])
                    )

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_tracks: set[int] = set()
            matched_obs: set[int] = set()

            for r, c in zip(row_ind, col_ind, strict=True):
                if cost_matrix[r, c] <= self.max_distance:
                    self.tracks[r].update(observations[c])
                    matched_tracks.add(r)
                    matched_obs.add(c)

            unmatched_track_indices = [
                i for i in range(n_tracks) if i not in matched_tracks
            ]
            unmatched_obs_indices = [i for i in range(n_obs) if i not in matched_obs]
        else:
            # No tracks or no observations — all unmatched
            unmatched_track_indices = list(range(n_tracks))
            unmatched_obs_indices = list(range(n_obs))

        # --- Mark unmatched tracks as missed ---
        for i in unmatched_track_indices:
            self.tracks[i].mark_missed()

        # --- TRACK-04 population constraint ---
        # Dead tracks this frame get their fish_id recycled to new observations.
        dead_ids: list[int] = []
        for i in unmatched_track_indices:
            if self.tracks[i].is_dead:
                dead_ids.append(self.tracks[i].fish_id)

        recycled_obs: set[int] = set()
        for obs_i, dead_id in zip(unmatched_obs_indices, dead_ids, strict=False):
            new_track = self._create_track(
                observations[obs_i], fish_id_override=dead_id
            )
            _ = new_track  # track added via _create_track
            recycled_obs.add(obs_i)

        # --- Create new tracks for remaining unmatched observations ---
        for obs_i in unmatched_obs_indices:
            if obs_i not in recycled_obs:
                self._create_track(observations[obs_i])

        # --- Prune dead tracks ---
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
        """Return 3D centroids of confirmed tracks for prior-guided RANSAC.

        Returns:
            List of shape-(3,) arrays, one per confirmed track, or None if
            no confirmed tracks exist (e.g. during the first few frames).
        """
        confirmed = [t for t in self.tracks if t.is_confirmed]
        if not confirmed:
            return None
        seeds: list[np.ndarray] = []
        for t in confirmed:
            if len(t.positions) > 0:
                seeds.append(list(t.positions)[-1].copy())
        return seeds if seeds else None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
        )
        track.update(association)
        # A brand-new track has streak=1 after one update; confirm immediately
        # if min_hits==1.
        self.tracks.append(track)
        return track
