"""Custom keypoint-based single-pass Kalman filter tracker.

Implements the single-pass keypoint tracker:

- _KalmanFilter: 24-dim constant-velocity KF tracking 6 keypoints x 2D.
  State dimension note: CONTEXT.md references "60-dim" conceptually (tracking
  all 6 keypoints' full state). The mathematical dimension is 24: 6 kpts x
  2D x 2 (position + velocity).

- compute_oks_matrix: vectorized (N, M) OKS similarity using NumPy broadcasting.
- compute_ocm_matrix: (N, M) cosine similarity of spine heading vectors.
- build_cost_matrix: (1 - OKS) + lambda * (1 - OCM).
- _KFTrack: per-track state with ORU/OCR mechanisms.
- _KptTrackletBuilder: mutable per-track accumulator with keypoint storage.
- _SinglePassTracker: predict/match/update/birth/death per-frame loop.
- interpolate_gaps: Cubic-spline gap filling for small temporal gaps.
- interpolate_low_confidence_keypoints: Per-keypoint cubic spline fill for low-confidence keypoints.
- KeypointTracker: Public wrapper running a single forward pass with gap
  interpolation. The bidirectional merge was removed after Phase 84-01
  investigation showed it produced 44 duplicate-inflated tracks vs 37 for
  the clean forward-only pass (0.942 coverage each).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment

from aquapose.core.tracking.types import Tracklet2D

logger = logging.getLogger(__name__)

__all__ = [
    "KeypointTracker",
    "_KFTrack",
    "_KalmanFilter",
    "_KptTrackletBuilder",
    "_SinglePassTracker",
    "build_cost_matrix",
    "compute_heading",
    "compute_ocm_matrix",
    "compute_oks_matrix",
    "interpolate_gaps",
    "interpolate_low_confidence_keypoints",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keypoint indices for spine heading (spine1 → spine3)
_SPINE1_IDX: int = 2
_SPINE3_IDX: int = 4


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------


class _KalmanFilter:
    """24-dim constant-velocity Kalman filter for 6 2-D keypoints.

    State vector layout:
        x = [kp0_x, kp0_y, kp1_x, kp1_y, ..., kp5_x, kp5_y,         (positions, 12 dims)
              v_kp0_x, v_kp0_y, v_kp1_x, v_kp1_y, ..., v_kp5_x, v_kp5_y]  (velocities, 12 dims)

    F (transition): [[I_12, I_12], [0_12, I_12]]  — constant-velocity
    H (observation): [I_12 | 0_12]                — extract positions
    Q (process noise): small for velocity, larger for position (velocity coupling)
    R (measurement noise): confidence-scaled diagonal — R_k = base_R / max(conf_k, eps)

    Args:
        initial_obs: Initial keypoint positions, shape (6, 2).
        confs: Per-keypoint confidence scores, shape (6,).
        base_r: Base measurement noise variance (default 10.0).
        process_noise_pos: Process noise for position components (default 1.0).
        process_noise_vel: Process noise for velocity components (default 0.01).
    """

    _DIM_X: int = 24  # state dimension
    _DIM_Z: int = 12  # observation dimension (positions only)

    def __init__(
        self,
        initial_obs: np.ndarray,
        confs: np.ndarray,
        base_r: float = 10.0,
        process_noise_pos: float = 1.0,
        process_noise_vel: float = 0.01,
    ) -> None:
        # State vector: positions followed by velocities
        self.x: np.ndarray = np.zeros(self._DIM_X, dtype=np.float64)
        self.x[: self._DIM_Z] = initial_obs.reshape(-1)

        # Covariance: high initial uncertainty on velocities
        self.P: np.ndarray = np.eye(self._DIM_X, dtype=np.float64) * 10.0
        self.P[self._DIM_Z :, self._DIM_Z :] *= 100.0  # velocities more uncertain

        self.base_r = float(base_r)

        # Transition matrix F
        self.F: np.ndarray = np.eye(self._DIM_X, dtype=np.float64)
        self.F[: self._DIM_Z, self._DIM_Z :] = np.eye(self._DIM_Z, dtype=np.float64)

        # Observation matrix H
        self.H: np.ndarray = np.zeros((self._DIM_Z, self._DIM_X), dtype=np.float64)
        self.H[: self._DIM_Z, : self._DIM_Z] = np.eye(self._DIM_Z, dtype=np.float64)

        # Process noise Q
        self.Q: np.ndarray = np.eye(self._DIM_X, dtype=np.float64)
        self.Q[: self._DIM_Z, : self._DIM_Z] *= process_noise_pos
        self.Q[self._DIM_Z :, self._DIM_Z :] *= process_noise_vel

    def _build_R(self, confs: np.ndarray) -> np.ndarray:
        """Build confidence-scaled measurement noise matrix R.

        R[2k, 2k] = R[2k+1, 2k+1] = base_R / max(conf_k, epsilon)

        Args:
            confs: Per-keypoint confidence scores, shape (6,).

        Returns:
            Diagonal R matrix, shape (12, 12).
        """
        epsilon = 1e-6
        r_diag = np.repeat(
            self.base_r / np.maximum(confs.astype(np.float64), epsilon), 2
        )
        return np.diag(r_diag)

    def predict(self) -> np.ndarray:
        """Advance state by one timestep (constant-velocity prediction).

        Returns:
            Predicted keypoint positions, shape (6, 2).
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[: self._DIM_Z].reshape(6, 2).copy()

    def update(self, obs: np.ndarray, confs: np.ndarray) -> None:
        """Standard Kalman update step with confidence-scaled measurement noise.

        Args:
            obs: Observed keypoint positions, shape (6, 2).
            confs: Per-keypoint confidence scores, shape (6,).
        """
        R = self._build_R(confs)
        z = obs.reshape(-1).astype(np.float64)

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self._DIM_X, dtype=np.float64) - K @ self.H
        self.P = I_KH @ self.P

    def get_positions(self) -> np.ndarray:
        """Return current estimated keypoint positions.

        Returns:
            Shape (6, 2).
        """
        return self.x[: self._DIM_Z].reshape(6, 2).copy()

    def get_state(self) -> dict[str, Any]:
        """Serialize KF state to JSON-safe dict (for chunk handoff).

        Returns:
            Dict with ``x`` and ``P`` as nested lists (not numpy arrays).
        """
        return {
            "x": self.x.tolist(),
            "P": self.P.tolist(),
            "base_r": self.base_r,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> _KalmanFilter:
        """Reconstruct a _KalmanFilter from a serialized state dict.

        Args:
            state: Dict previously returned by ``get_state()``.

        Returns:
            _KalmanFilter with restored ``x`` and ``P``.
        """
        # Create a dummy instance without going through __init__
        instance = cls.__new__(cls)
        x = np.array(state["x"], dtype=np.float64)
        n = len(x)

        instance.x = x
        instance.P = np.array(state["P"], dtype=np.float64)
        instance.base_r = float(state.get("base_r", 10.0))

        # Rebuild matrices from dimensions
        dim_x = n
        dim_z = n // 2
        instance.F = np.eye(dim_x, dtype=np.float64)
        instance.F[:dim_z, dim_z:] = np.eye(dim_z, dtype=np.float64)
        instance.H = np.zeros((dim_z, dim_x), dtype=np.float64)
        instance.H[:dim_z, :dim_z] = np.eye(dim_z, dtype=np.float64)
        instance.Q = np.eye(dim_x, dtype=np.float64)
        instance.Q[:dim_z, :dim_z] *= 1.0
        instance.Q[dim_z:, dim_z:] *= 0.01

        return instance


# ---------------------------------------------------------------------------
# OKS cost functions (vectorized)
# ---------------------------------------------------------------------------


def compute_oks_matrix(
    pred_kpts: np.ndarray,
    det_kpts: np.ndarray,
    det_confs: np.ndarray,
    det_scales: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Compute (N, M) OKS similarity matrix using vectorized NumPy broadcasting.

    For each (track, detection) pair:
        OKS = sum(c_k * exp(-d_k^2 / (2 * s^2 * sigma_k^2))) / sum(c_k)
    where d_k = Euclidean distance for keypoint k, s = sqrt(detection scale).

    Args:
        pred_kpts: Predicted keypoint positions, shape ``(N, 6, 2)``.
        det_kpts: Detection keypoint positions, shape ``(M, 6, 2)``.
        det_confs: Detection keypoint confidences, shape ``(M, 6)``.
        det_scales: Scale proxy (e.g. sqrt(OBB area)), shape ``(M,)``. Clamped >= 1.0.
        sigmas: Per-keypoint OKS sigmas, shape ``(6,)``.

    Returns:
        OKS similarity matrix, shape ``(N, M)``, values in [0, 1].
    """
    # Clamp scale to avoid division by zero
    scales = np.maximum(det_scales, 1.0).astype(np.float64)  # (M,)

    # Expand dims for broadcasting:
    # pred_kpts: (N, 1, K, 2) vs det_kpts: (1, M, K, 2)
    pred = pred_kpts[:, np.newaxis, :, :]  # (N, 1, K, 2)
    det = det_kpts[np.newaxis, :, :, :]  # (1, M, K, 2)

    # Squared Euclidean distance: (N, M, K)
    d2 = np.sum((pred - det) ** 2, axis=-1)  # (N, M, K)

    # Denominator per pair: 2 * s^2 * sigma_k^2
    # scales: (M,) → (1, M, 1); sigmas: (K,) → (1, 1, K)
    s2 = scales[np.newaxis, :, np.newaxis] ** 2  # (1, M, 1)
    sigma2 = sigmas[np.newaxis, np.newaxis, :] ** 2  # (1, 1, K)
    denom = 2.0 * s2 * sigma2  # (1, M, K)

    # Exponent term: (N, M, K)
    exp_term = np.exp(-d2 / denom)  # (N, M, K)

    # Confidence weights: (1, M, K)
    confs = det_confs[np.newaxis, :, :]  # (1, M, K)

    # Weighted sum
    numerator = np.sum(confs * exp_term, axis=-1)  # (N, M)
    # confs is (1, M, K); sum over K gives (1, M); that's the denominator broadcast shape
    denominator = det_confs.sum(axis=-1)[np.newaxis, :] + 1e-12  # (1, M)

    oks = numerator / denominator  # (N, M)
    return oks


def compute_heading(kpts: np.ndarray) -> np.ndarray:
    """Compute normalised spine heading vector from spine1 (idx 2) to spine3 (idx 4).

    Args:
        kpts: Keypoint positions, shape ``(6, 2)`` or ``(K, 2)`` with K >= 5.

    Returns:
        Unit heading vector, shape ``(2,)``. If degenerate (zero length),
        returns ``[1.0, 0.0]``.
    """
    vec = kpts[_SPINE3_IDX] - kpts[_SPINE1_IDX]
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return vec / norm


def compute_ocm_matrix(
    pred_headings: np.ndarray,
    det_headings: np.ndarray,
) -> np.ndarray:
    """Compute (N, M) OCM direction-consistency matrix via cosine similarity.

    Args:
        pred_headings: Track heading vectors, shape ``(N, 2)``.
        det_headings: Detection heading vectors, shape ``(M, 2)``.

    Returns:
        Cosine similarity matrix, shape ``(N, M)``, values in [-1, 1].
    """

    # Normalise (should already be unit, but guard anyway)
    def _unit(v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(norms, 1e-9)

    ph = _unit(pred_headings)  # (N, 2)
    dh = _unit(det_headings)  # (M, 2)

    # (N, M) dot product via broadcasting
    ocm = ph @ dh.T  # (N, M)
    return ocm


def build_cost_matrix(
    oks: np.ndarray,
    ocm: np.ndarray,
    lambda_ocm: float = 0.2,
) -> np.ndarray:
    """Combine OKS similarity and OCM direction-consistency into a cost matrix.

    cost = (1 - OKS) + lambda_ocm * (1 - OCM)

    OKS dominates; OCM breaks ties and penalises head-tail flips.

    Args:
        oks: OKS similarity matrix, shape ``(N, M)``.
        ocm: OCM direction-consistency matrix, shape ``(N, M)``.
        lambda_ocm: Weight for OCM term (default 0.2).

    Returns:
        Cost matrix, shape ``(N, M)``.
    """
    return (1.0 - oks) + lambda_ocm * (1.0 - ocm)


# ---------------------------------------------------------------------------
# Per-track accumulator
# ---------------------------------------------------------------------------


@dataclass
class _KptTrackletBuilder:
    """Mutable per-track accumulator that stores keypoints alongside standard fields.

    Builds a Tracklet2D incrementally from per-frame observations.
    Keypoints and confs are stored for gap interpolation and downstream pose use.

    Attributes:
        camera_id: Camera this tracklet belongs to.
        track_id: Local track ID.
        frames: Frame indices.
        centroids: Per-frame (u, v) centroids.
        bboxes: Per-frame (x, y, w, h) bboxes.
        frame_status: Per-frame "detected" or "coasted".
        keypoints: Per-frame keypoint arrays, shape (6, 2) each.
        keypoint_conf: Per-frame keypoint confidence arrays, shape (6,) each.
        detected_count: Number of "detected" frames.
        active: Whether this track is still alive.
    """

    camera_id: str
    track_id: int
    frames: list[int] = field(default_factory=list)
    centroids: list[tuple[float, float]] = field(default_factory=list)
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    frame_status: list[str] = field(default_factory=list)
    keypoints: list[np.ndarray] = field(default_factory=list)
    keypoint_conf: list[np.ndarray] = field(default_factory=list)
    detected_count: int = 0
    active: bool = True

    def add_frame(
        self,
        frame_idx: int,
        kpts: np.ndarray,
        kconf: np.ndarray,
        bbox_xywh: tuple[float, float, float, float],
        status: str,
    ) -> None:
        """Append one frame.

        Centroid is derived from spine1 keypoint (index 2) when confidence >= 0.3,
        otherwise falls back to bbox centre.

        Args:
            frame_idx: Frame index.
            kpts: Keypoint positions, shape (6, 2).
            kconf: Keypoint confidences, shape (6,).
            bbox_xywh: Bounding box (x, y, w, h).
            status: "detected" or "coasted".
        """
        x, y, w, h = bbox_xywh
        cx = x + w / 2.0
        cy = y + h / 2.0
        # Use spine1 (index 2) keypoint as centroid when confident
        if float(kconf[2]) >= 0.3:
            cx = float(kpts[2, 0])
            cy = float(kpts[2, 1])

        self.frames.append(frame_idx)
        self.centroids.append((cx, cy))
        self.bboxes.append(bbox_xywh)
        self.frame_status.append(status)
        self.keypoints.append(kpts.copy())
        self.keypoint_conf.append(kconf.copy())
        if status == "detected":
            self.detected_count += 1

    def to_tracklet2d(self) -> Tracklet2D:
        """Convert to immutable Tracklet2D.

        Returns:
            Frozen Tracklet2D instance.
        """
        return Tracklet2D(
            camera_id=self.camera_id,
            track_id=self.track_id,
            frames=tuple(self.frames),
            centroids=tuple(self.centroids),
            bboxes=tuple(self.bboxes),
            frame_status=tuple(self.frame_status),
            keypoints=np.stack(self.keypoints) if self.keypoints else None,
            keypoint_conf=np.stack(self.keypoint_conf) if self.keypoint_conf else None,
        )


# ---------------------------------------------------------------------------
# _KFTrack: per-track state with ORU/OCR mechanisms
# ---------------------------------------------------------------------------


class _KFTrack:
    """Per-track state container with ORU and OCR mechanisms.

    ORU (Observation-centric Re-Update): After a coasting track is re-matched,
    re-initialize the KF from the stored pre-coast state and replay the update
    with the new observation.

    OCR (Observation-Centric Recovery): During coast, scan obs_history for a
    past observation whose OKS with the predicted position exceeds a recovery
    threshold. If found, trigger a KF update to stabilize the track.

    Args:
        track_id: Unique local track ID.
        kf: Initialized Kalman filter.
        n_init: Confirmation threshold (hit_streak >= n_init → confirmed).
    """

    def __init__(
        self,
        track_id: int,
        kf: _KalmanFilter,
        n_init: int = 3,
    ) -> None:
        self.track_id = track_id
        self.kf = kf
        self.n_init = n_init
        self.time_since_update: int = 0
        self.hit_streak: int = 0
        self.detected_count: int = 0
        self.state: str = "tentative"  # "tentative" or "confirmed"

        # OCR: ring buffer of (obs_array, conf_array) from recent detected frames
        self.obs_history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=5)

        # ORU: pre-coast state snapshot
        self._pre_coast_x: np.ndarray | None = None
        self._pre_coast_P: np.ndarray | None = None

        # Merger state: track coasting due to a nearby matched track absorbing it
        self.merger_partner_id: int | None = None
        self.merger_frames: int = 0

    def mark_missed(self) -> None:
        """Record a missed frame (no detection match)."""
        # Save pre-coast state on first miss
        if self.time_since_update == 0:
            self._pre_coast_x = self.kf.x.copy()
            self._pre_coast_P = self.kf.P.copy()
        self.time_since_update += 1
        self.hit_streak = 0

    def match_update(self, obs: np.ndarray, confs: np.ndarray) -> None:
        """Update KF with a matched detection.

        If recovering from coast (time_since_update > 0), apply ORU:
        restore the pre-coast state and run KF update from that baseline.

        Args:
            obs: Observed keypoint positions, shape (6, 2).
            confs: Per-keypoint confidences, shape (6,).
        """
        if (
            self.time_since_update > 0
            and self._pre_coast_x is not None
            and self._pre_coast_P is not None
        ):
            # ORU: reset to pre-coast state, then apply update
            self.kf.x = self._pre_coast_x.copy()
            self.kf.P = self._pre_coast_P.copy()
            self._pre_coast_x = None
            self._pre_coast_P = None

        self.kf.update(obs=obs, confs=confs)
        self.obs_history.append((obs.copy(), confs.copy()))
        self.time_since_update = 0
        self.hit_streak += 1
        self.detected_count += 1

        # Clear merger state — track re-acquired its own detection
        self.merger_partner_id = None
        self.merger_frames = 0

        if self.state == "tentative" and self.hit_streak >= self.n_init:
            self.state = "confirmed"

    def attempt_ocr(self, sigmas: np.ndarray, ocr_threshold: float = 0.5) -> bool:
        """OCR: scan obs_history for a stable observation to recover with.

        Computes OKS between the KF predicted positions and each stored
        observation. If OKS > *ocr_threshold*, apply that update.

        Args:
            sigmas: Per-keypoint OKS sigmas, shape (6,).
            ocr_threshold: Minimum OKS for recovery acceptance.

        Returns:
            True if recovery was applied, False otherwise.
        """
        if not self.obs_history:
            return False

        pred = self.kf.x[: self.kf._DIM_Z].reshape(6, 2)
        pred_kpts = pred[np.newaxis, :, :]  # (1, 6, 2)

        for obs, confs in reversed(self.obs_history):
            det_kpts = obs[np.newaxis, :, :]  # (1, 6, 2)
            det_confs_2d = confs[np.newaxis, :]  # (1, 6)
            # Scale from position spread
            scale = np.sqrt(np.sum(np.std(obs, axis=0) ** 2) * 6.0 + 1.0)
            det_scales = np.array([scale])
            oks_val = compute_oks_matrix(
                pred_kpts, det_kpts, det_confs_2d, det_scales, sigmas
            )[0, 0]
            if oks_val > ocr_threshold:
                self.kf.update(obs=obs, confs=confs)
                return True
        return False


# ---------------------------------------------------------------------------
# _SinglePassTracker
# ---------------------------------------------------------------------------


class _SinglePassTracker:
    """Single-pass per-camera keypoint tracker.

    Runs predict/match/update/birth/death per frame using OKS + OCM cost matrix
    and Hungarian assignment. Produces Tracklet2D objects for confirmed tracks.

    Args:
        camera_id: Camera identifier.
        config: Namespace or dataclass with fields:
            - max_age (int): Frames to coast before culling.
            - n_init (int): Hit streak threshold for confirmation.
            - det_thresh (float): Minimum detection confidence.
            - base_r (float): KF base measurement noise.
            - lambda_ocm (float): OCM weight in cost matrix.
            - sigmas (np.ndarray): Per-keypoint OKS sigmas, shape (6,).
            - max_match_distance (float): Max spine1 distance for match gating.
    """

    def __init__(
        self,
        camera_id: str,
        config: Any,
    ) -> None:
        self.camera_id = camera_id
        self._max_age: int = int(config.max_age)
        self._n_init: int = int(config.n_init)
        self._det_thresh: float = float(config.det_thresh)
        self._base_r: float = float(config.base_r)
        self._lambda_ocm: float = float(config.lambda_ocm)
        self._sigmas: np.ndarray = np.asarray(config.sigmas, dtype=np.float64)
        self._match_cost_threshold: float = float(
            getattr(config, "match_cost_threshold", 1.2)
        )
        self._ocr_threshold: float = float(getattr(config, "ocr_threshold", 0.5))
        self._max_match_distance: float = float(
            getattr(config, "max_match_distance", 75.0)
        )
        self._merger_distance: float = float(getattr(config, "merger_distance", 30.0))
        self._merger_max_age: int = int(getattr(config, "merger_max_age", 90))
        self._track_thresh: float = float(
            getattr(config, "track_thresh", self._det_thresh)
        )
        self._birth_thresh: float = float(
            getattr(config, "birth_thresh", self._det_thresh)
        )

        self._next_track_id: int = 0
        self._active_tracks: dict[int, _KFTrack] = {}
        self._builders: dict[int, _KptTrackletBuilder] = {}

        # Inferred image bounds from detection bboxes (for out-of-frame culling).
        self._image_bounds: tuple[float, float] | None = None  # (max_x, max_y)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_track(self, kpts: np.ndarray, confs: np.ndarray) -> _KFTrack:
        """Allocate a new tentative track.

        Args:
            kpts: Initial keypoint positions, shape (6, 2).
            confs: Initial keypoint confidences, shape (6,).

        Returns:
            New _KFTrack.
        """
        tid = self._next_track_id
        self._next_track_id += 1
        kf = _KalmanFilter(
            initial_obs=kpts.astype(np.float64),
            confs=confs.astype(np.float64),
            base_r=self._base_r,
        )
        trk = _KFTrack(track_id=tid, kf=kf, n_init=self._n_init)
        self._active_tracks[tid] = trk
        self._builders[tid] = _KptTrackletBuilder(
            camera_id=self.camera_id,
            track_id=tid,
        )
        return trk

    @staticmethod
    def _det_scale(det: Any) -> float:
        """Extract scale proxy from detection (sqrt of OBB area or bbox area).

        Args:
            det: Detection-like object.

        Returns:
            Scale float >= 1.0.
        """
        area = getattr(det, "obb_area", None)
        if area is None:
            _bx, _by, bw, bh = det.bbox
            area = float(bw) * float(bh)
        return max(float(np.sqrt(max(float(area), 1.0))), 1.0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _run_hungarian_match(
        self,
        track_ids: list[int],
        pred_kpts_map: dict[int, np.ndarray],
        valid_dets: list[tuple[Any, np.ndarray, np.ndarray]],
        det_indices: list[int],
    ) -> list[tuple[int, int]]:
        """Run Hungarian matching between a subset of tracks and detections.

        Args:
            track_ids: Track IDs to match against.
            pred_kpts_map: Mapping from track ID to predicted keypoints (6, 2).
            valid_dets: Full list of validated detections (det, kpts, kconf).
            det_indices: Indices into *valid_dets* for the detections to match.

        Returns:
            List of (track_id, det_index) matched pairs.
        """
        if not track_ids or not det_indices:
            return []

        pred_kpts_list = [pred_kpts_map[tid] for tid in track_ids]
        pred_kpts = np.stack(pred_kpts_list, axis=0)  # (N, 6, 2)
        det_kpts_arr = np.stack(
            [valid_dets[i][1] for i in det_indices], axis=0
        )  # (M, 6, 2)
        det_confs_arr = np.stack(
            [valid_dets[i][2] for i in det_indices], axis=0
        )  # (M, 6)
        det_scales_arr = np.array(
            [self._det_scale(valid_dets[i][0]) for i in det_indices],
            dtype=np.float64,
        )  # (M,)

        oks = compute_oks_matrix(
            pred_kpts, det_kpts_arr, det_confs_arr, det_scales_arr, self._sigmas
        )  # (N, M)

        pred_headings = np.stack(
            [compute_heading(p) for p in pred_kpts_list], axis=0
        )  # (N, 2)
        det_headings = np.stack(
            [compute_heading(valid_dets[i][1]) for i in det_indices], axis=0
        )  # (M, 2)
        ocm = compute_ocm_matrix(pred_headings, det_headings)  # (N, M)

        cost = build_cost_matrix(oks, ocm, lambda_ocm=self._lambda_ocm)  # (N, M)

        _GATE_SENTINEL = 1e6

        pred_spine1 = pred_kpts[:, _SPINE1_IDX, :]  # (N, 2)
        det_spine1 = det_kpts_arr[:, _SPINE1_IDX, :]  # (M, 2)
        spine1_dist = np.linalg.norm(
            pred_spine1[:, np.newaxis, :] - det_spine1[np.newaxis, :, :],
            axis=-1,
        )  # (N, M)
        distance_mask = spine1_dist > self._max_match_distance  # (N, M)

        gated_cost = np.where(
            (cost < self._match_cost_threshold) & ~distance_mask,
            cost,
            _GATE_SENTINEL,
        )
        row_idx, col_idx = linear_sum_assignment(gated_cost)

        matches: list[tuple[int, int]] = []
        for r, c in zip(row_idx, col_idx, strict=False):
            if gated_cost[r, c] >= _GATE_SENTINEL:
                continue
            matches.append((track_ids[r], det_indices[c]))
        return matches

    def update(self, frame_idx: int, detections: list[Any]) -> None:
        """Process one frame: predict → two-phase match → birth → death.

        Phase 1 matches high-confidence detections (conf >= track_thresh)
        against all tracks.  Phase 2 matches low-confidence detections
        against tracks that were actively matched on the previous frame
        (time_since_update == 0 before this frame's matching), preventing
        stale coasting tracks from latching onto low-confidence noise.

        New tracks are only birthed from unmatched Phase 1 detections with
        conf >= birth_thresh.

        Args:
            frame_idx: Current frame index.
            detections: List of Detection-like objects with attributes:
                ``confidence``, ``keypoints`` (K, 2) ndarray,
                ``keypoint_conf`` (K,) ndarray, ``bbox`` (x, y, w, h).
                Detections missing keypoints are skipped.
        """
        # 1. Filter detections by det_thresh
        valid_dets: list[tuple[Any, np.ndarray, np.ndarray]] = []
        for det in detections:
            if float(det.confidence) < self._det_thresh:
                continue
            kpts = getattr(det, "keypoints", None)
            kconf = getattr(det, "keypoint_conf", None)
            if kpts is None or kconf is None:
                continue
            kpts_arr = np.asarray(kpts, dtype=np.float64)
            kconf_arr = np.asarray(kconf, dtype=np.float64)
            if kpts_arr.shape[0] < 5:  # need at least up to spine3 index
                continue
            valid_dets.append((det, kpts_arr, kconf_arr))

        # 2. Split into high/low confidence by track_thresh
        high_conf_indices: list[int] = []
        low_conf_indices: list[int] = []
        for i, (det, _kpts, _kconf) in enumerate(valid_dets):
            if float(det.confidence) >= self._track_thresh:
                high_conf_indices.append(i)
            else:
                low_conf_indices.append(i)

        # Update inferred image bounds from detection bboxes.
        for det, _kpts_arr, _kconf_arr in valid_dets:
            bx, by, bw, bh = det.bbox
            right = float(bx) + float(bw)
            bottom = float(by) + float(bh)
            if self._image_bounds is None:
                self._image_bounds = (right, bottom)
            else:
                self._image_bounds = (
                    max(self._image_bounds[0], right),
                    max(self._image_bounds[1], bottom),
                )

        track_ids = list(self._active_tracks.keys())

        # 3. Predict all active tracks
        pred_kpts_map: dict[int, np.ndarray] = {}
        for tid in track_ids:
            trk = self._active_tracks[tid]
            pred = trk.kf.predict()  # (6, 2)
            pred_kpts_map[tid] = pred

        # Snapshot time_since_update BEFORE matching for Phase 2 eligibility.
        # Tracks with tsu == 0 were matched on the previous frame (actively
        # tracked), not coasting.
        pre_match_tsu: dict[int, int] = {
            tid: self._active_tracks[tid].time_since_update for tid in track_ids
        }

        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()

        # 4. Phase 1: match high-conf dets against ALL tracks
        p1_matches = self._run_hungarian_match(
            track_ids, pred_kpts_map, valid_dets, high_conf_indices
        )
        for tid, det_idx in p1_matches:
            det, kpts_arr, kconf_arr = valid_dets[det_idx]
            trk = self._active_tracks[tid]
            trk.match_update(obs=kpts_arr, confs=kconf_arr)
            x, y, w, h = det.bbox
            self._builders[tid].add_frame(
                frame_idx=frame_idx,
                kpts=kpts_arr.astype(np.float32),
                kconf=kconf_arr.astype(np.float32),
                bbox_xywh=(float(x), float(y), float(w), float(h)),
                status="detected",
            )
            matched_track_ids.add(tid)
            matched_det_indices.add(det_idx)

        # 5. Phase 2: match low-conf dets against recently-active unmatched tracks
        # Eligible: not matched in Phase 1 AND was actively tracked (tsu == 0)
        # before this frame's matching began.
        p2_track_ids = [
            tid
            for tid in track_ids
            if tid not in matched_track_ids and pre_match_tsu[tid] == 0
        ]
        unmatched_low = [i for i in low_conf_indices if i not in matched_det_indices]

        p2_matches = self._run_hungarian_match(
            p2_track_ids, pred_kpts_map, valid_dets, unmatched_low
        )
        for tid, det_idx in p2_matches:
            det, kpts_arr, kconf_arr = valid_dets[det_idx]
            trk = self._active_tracks[tid]
            trk.match_update(obs=kpts_arr, confs=kconf_arr)
            x, y, w, h = det.bbox
            self._builders[tid].add_frame(
                frame_idx=frame_idx,
                kpts=kpts_arr.astype(np.float32),
                kconf=kconf_arr.astype(np.float32),
                bbox_xywh=(float(x), float(y), float(w), float(h)),
                status="detected",
            )
            matched_track_ids.add(tid)
            matched_det_indices.add(det_idx)

        # 6. Unmatched tracks: coast or cull (with merger detection)
        matched_spine1_positions: dict[int, np.ndarray] = {}
        for mtid in matched_track_ids:
            mtrk = self._active_tracks[mtid]
            matched_spine1_positions[mtid] = mtrk.kf.get_positions()[_SPINE1_IDX]

        for tid in track_ids:
            if tid in matched_track_ids:
                continue
            trk = self._active_tracks[tid]

            # Merger detection: is a matched track very close?
            pred_spine1 = trk.kf.get_positions()[_SPINE1_IDX]
            is_merger = False
            for mtid, mpos in matched_spine1_positions.items():
                if np.linalg.norm(pred_spine1 - mpos) < self._merger_distance:
                    trk.merger_partner_id = mtid
                    trk.merger_frames += 1
                    is_merger = True
                    break

            if not is_merger:
                trk.merger_partner_id = None
                trk.merger_frames = 0

            trk.mark_missed()

            # OCR attempt for confirmed coasting tracks
            if trk.state == "confirmed":
                trk.attempt_ocr(self._sigmas, self._ocr_threshold)

            # Add coasted frame for confirmed tracks
            if trk.state == "confirmed":
                pred_pos = trk.kf.get_positions()  # (6, 2)
                dummy_kconf = np.zeros(6, dtype=np.float32)
                # Derive bbox from predicted keypoints span
                kx = pred_pos[:, 0]
                ky = pred_pos[:, 1]
                x1, x2 = float(kx.min()), float(kx.max())
                y1, y2 = float(ky.min()), float(ky.max())
                w = max(x2 - x1, 1.0)
                h = max(y2 - y1, 1.0)
                self._builders[tid].add_frame(
                    frame_idx=frame_idx,
                    kpts=pred_pos.astype(np.float32),
                    kconf=dummy_kconf,
                    bbox_xywh=(x1, y1, w, h),
                    status="coasted",
                )

        # 7. Birth: only unmatched Phase 1 dets with conf >= birth_thresh
        for c in high_conf_indices:
            if c in matched_det_indices:
                continue
            det, kpts_arr, kconf_arr = valid_dets[c]
            if float(det.confidence) < self._birth_thresh:
                continue
            trk = self._new_track(kpts=kpts_arr, confs=kconf_arr)
            x, y, w, h = det.bbox
            trk.match_update(obs=kpts_arr, confs=kconf_arr)
            self._builders[trk.track_id].add_frame(
                frame_idx=frame_idx,
                kpts=kpts_arr.astype(np.float32),
                kconf=kconf_arr.astype(np.float32),
                bbox_xywh=(float(x), float(y), float(w), float(h)),
                status="detected",
            )

        # 8. Death: cull tracks exceeding max_age or out of frame
        _OOF_MARGIN = 50.0  # pixels beyond image edge before culling
        dead_ids = []
        for tid, trk in self._active_tracks.items():
            effective_max_age = self._max_age
            if trk.merger_partner_id is not None:
                effective_max_age = self._merger_max_age
            if trk.time_since_update > effective_max_age:
                dead_ids.append(tid)
                continue
            # Out-of-frame culling: if predicted spine1 has left the image,
            # the fish is gone — cull immediately to prevent detection theft.
            if self._image_bounds is not None and trk.time_since_update > 0:
                pred_pos = trk.kf.get_positions()  # (6, 2)
                spine1 = pred_pos[_SPINE1_IDX]
                max_x, max_y = self._image_bounds
                if (
                    spine1[0] < -_OOF_MARGIN
                    or spine1[1] < -_OOF_MARGIN
                    or spine1[0] > max_x + _OOF_MARGIN
                    or spine1[1] > max_y + _OOF_MARGIN
                ):
                    dead_ids.append(tid)
        for tid in dead_ids:
            del self._active_tracks[tid]
            # Keep builder for get_tracklets()

    def get_tracklets(self) -> list[Tracklet2D]:
        """Return confirmed tracklets (detected_count >= n_init).

        Returns:
            List of frozen Tracklet2D instances.
        """
        result = []
        for _tid, builder in self._builders.items():
            if builder.detected_count >= self._n_init and builder.frames:
                result.append(builder.to_tracklet2d())
        return result

    def get_state(self) -> dict[str, Any]:
        """Serialize tracker state for cross-chunk handoff.

        Builder history is intentionally excluded — only KF states, track
        metadata, OCR state, and config are serialized.  On restore,
        ``from_state()`` creates fresh empty builders for active tracks.

        Returns:
            State dict containing track KF states and config (no builders).
        """
        tracks_state = {}
        for tid, trk in self._active_tracks.items():
            tracks_state[tid] = {
                "kf": trk.kf.get_state(),
                "time_since_update": trk.time_since_update,
                "hit_streak": trk.hit_streak,
                "detected_count": trk.detected_count,
                "state": trk.state,
                "n_init": trk.n_init,
                "obs_history": [(o.tolist(), c.tolist()) for o, c in trk.obs_history],
                "pre_coast_x": trk._pre_coast_x.tolist()
                if trk._pre_coast_x is not None
                else None,
                "pre_coast_P": trk._pre_coast_P.tolist()
                if trk._pre_coast_P is not None
                else None,
                "merger_partner_id": trk.merger_partner_id,
                "merger_frames": trk.merger_frames,
            }
        return {
            "tracks": tracks_state,
            "next_track_id": self._next_track_id,
            "max_age": self._max_age,
            "n_init": self._n_init,
            "det_thresh": self._det_thresh,
            "base_r": self._base_r,
            "lambda_ocm": self._lambda_ocm,
            "sigmas": self._sigmas.tolist(),
            "match_cost_threshold": self._match_cost_threshold,
            "ocr_threshold": self._ocr_threshold,
            "max_match_distance": self._max_match_distance,
            "merger_distance": self._merger_distance,
            "merger_max_age": self._merger_max_age,
            "track_thresh": self._track_thresh,
            "birth_thresh": self._birth_thresh,
            "image_bounds": list(self._image_bounds)
            if self._image_bounds is not None
            else None,
        }

    @classmethod
    def from_state(cls, camera_id: str, state: dict[str, Any]) -> _SinglePassTracker:
        """Reconstruct from a saved state blob.

        Args:
            camera_id: Camera ID.
            state: State dict from ``get_state()``.

        Returns:
            Restored _SinglePassTracker.
        """
        from types import SimpleNamespace

        det_thresh = state["det_thresh"]
        config = SimpleNamespace(
            max_age=state["max_age"],
            n_init=state["n_init"],
            det_thresh=det_thresh,
            base_r=state["base_r"],
            lambda_ocm=state["lambda_ocm"],
            sigmas=np.array(state["sigmas"], dtype=np.float64),
            match_cost_threshold=state.get("match_cost_threshold", 1.2),
            ocr_threshold=state.get("ocr_threshold", 0.5),
            max_match_distance=state.get("max_match_distance", 75.0),
            merger_distance=state.get("merger_distance", 30.0),
            merger_max_age=state.get("merger_max_age", 90),
            track_thresh=state.get("track_thresh", det_thresh),
            birth_thresh=state.get("birth_thresh", det_thresh),
        )
        instance = cls(camera_id=camera_id, config=config)
        instance._next_track_id = state["next_track_id"]

        # Restore inferred image bounds from previous chunk.
        saved_bounds = state.get("image_bounds")
        if saved_bounds is not None:
            instance._image_bounds = (float(saved_bounds[0]), float(saved_bounds[1]))

        for tid_str, ts in state["tracks"].items():
            tid = int(tid_str)
            kf = _KalmanFilter.from_state(ts["kf"])
            trk = _KFTrack(track_id=tid, kf=kf, n_init=ts["n_init"])
            trk.time_since_update = ts["time_since_update"]
            trk.hit_streak = ts["hit_streak"]
            trk.detected_count = ts["detected_count"]
            trk.state = ts["state"]
            for o_list, c_list in ts["obs_history"]:
                trk.obs_history.append(
                    (
                        np.array(o_list, dtype=np.float64),
                        np.array(c_list, dtype=np.float64),
                    )
                )
            if ts["pre_coast_x"] is not None:
                trk._pre_coast_x = np.array(ts["pre_coast_x"], dtype=np.float64)
                trk._pre_coast_P = np.array(ts["pre_coast_P"], dtype=np.float64)
            trk.merger_partner_id = ts.get("merger_partner_id")
            trk.merger_frames = ts.get("merger_frames", 0)
            instance._active_tracks[tid] = trk

        # Create empty builders for active tracks so new detections in the
        # next chunk are accumulated correctly (builder history is NOT carried
        # across chunks to avoid duplicate frame indices).
        for tid in instance._active_tracks:
            instance._builders[tid] = _KptTrackletBuilder(
                camera_id=camera_id, track_id=tid
            )

        return instance


# ---------------------------------------------------------------------------
# Gap interpolation
# ---------------------------------------------------------------------------


def interpolate_gaps(
    builder: _KptTrackletBuilder,
    max_gap_frames: int = 5,
) -> _KptTrackletBuilder:
    """Fill small temporal gaps in a tracklet using cubic-spline interpolation.

    For each gap <= max_gap_frames between consecutive frames, interpolates
    keypoint positions using CubicSpline. Interpolated frames get status
    "coasted". Gaps larger than max_gap_frames are left unfilled.

    Centroids for interpolated frames are derived from spine1 (index 2)
    keypoint when available. Bboxes use linear interpolation.

    Args:
        builder: Mutable tracklet builder to fill gaps in.
        max_gap_frames: Maximum gap size (exclusive) to fill. Gaps of this
            size or larger are left as-is.

    Returns:
        Updated builder with gap frames inserted (sorted by frame index).
    """
    if len(builder.frames) < 2:
        return builder

    # Collect all frame data into a sortable list
    frame_data = list(
        zip(
            builder.frames,
            builder.centroids,
            builder.bboxes,
            builder.frame_status,
            builder.keypoints,
            builder.keypoint_conf,
            strict=False,
        )
    )
    frame_data.sort(key=lambda x: x[0])

    new_frames: list[int] = []
    new_centroids: list[tuple[float, float]] = []
    new_bboxes: list[tuple[float, float, float, float]] = []
    new_statuses: list[str] = []
    new_keypoints: list[np.ndarray] = []
    new_kconfs: list[np.ndarray] = []

    for row in frame_data:
        new_frames.append(row[0])
        new_centroids.append(row[1])
        new_bboxes.append(row[2])
        new_statuses.append(row[3])
        new_keypoints.append(row[4])
        new_kconfs.append(row[5])

    # Scan for gaps and interpolate
    i = 0
    while i < len(new_frames) - 1:
        gap = new_frames[i + 1] - new_frames[i] - 1
        if 0 < gap <= max_gap_frames:
            # Build spline on the known frames around the gap
            # For CubicSpline we need at least 2 knot points; use all known frames
            known_f = np.array(new_frames, dtype=np.float64)
            known_kpts = np.stack(new_keypoints, axis=0).astype(np.float64)  # (N, 6, 2)
            known_bboxes_arr = np.array(new_bboxes, dtype=np.float64)  # (N, 4)

            # Use monotone=True via 'not-a-knot' when possible
            cs_kpts: list[list[CubicSpline]] = []
            n_kp = known_kpts.shape[1]
            for k in range(n_kp):
                cs_kpts.append(
                    [
                        CubicSpline(known_f, known_kpts[:, k, 0]),
                        CubicSpline(known_f, known_kpts[:, k, 1]),
                    ]
                )
            cs_bbox = [CubicSpline(known_f, known_bboxes_arr[:, d]) for d in range(4)]

            # Insert interpolated frames
            insert_idx = i + 1
            for missing_f in range(new_frames[i] + 1, new_frames[i + 1]):
                f_float = float(missing_f)

                # Interpolate keypoints
                interp_kpts = np.zeros((n_kp, 2), dtype=np.float32)
                for k in range(n_kp):
                    interp_kpts[k, 0] = float(cs_kpts[k][0](f_float))
                    interp_kpts[k, 1] = float(cs_kpts[k][1](f_float))

                # Interpolate bbox
                interp_bbox = tuple(float(cs_bbox[d](f_float)) for d in range(4))

                # Centroid from spine1 keypoint (index 2) if available
                if n_kp > 2:
                    cx = float(interp_kpts[2, 0])
                    cy = float(interp_kpts[2, 1])
                else:
                    bx, by, bw, bh = interp_bbox
                    cx = bx + bw / 2.0
                    cy = by + bh / 2.0

                # Insert at correct sorted position
                new_frames.insert(insert_idx, missing_f)
                new_centroids.insert(insert_idx, (cx, cy))
                new_bboxes.insert(insert_idx, interp_bbox)  # type: ignore[arg-type]
                new_statuses.insert(insert_idx, "coasted")
                new_keypoints.insert(insert_idx, interp_kpts)
                new_kconfs.insert(insert_idx, np.zeros(n_kp, dtype=np.float32))
                insert_idx += 1

            # Skip past the inserted frames to find the next real gap
            i = insert_idx
        else:
            i += 1

    # Recount detected frames
    detected_count = sum(1 for s in new_statuses if s == "detected")

    # Build updated builder
    out = _KptTrackletBuilder(camera_id=builder.camera_id, track_id=builder.track_id)
    out.frames = new_frames
    out.centroids = new_centroids  # type: ignore[assignment]
    out.bboxes = new_bboxes  # type: ignore[assignment]
    out.frame_status = new_statuses
    out.keypoints = new_keypoints
    out.keypoint_conf = new_kconfs
    out.detected_count = detected_count
    out.active = builder.active

    return out


def interpolate_low_confidence_keypoints(
    builder: _KptTrackletBuilder,
    conf_threshold: float = 0.3,
) -> _KptTrackletBuilder:
    """Fill low-confidence keypoints using per-keypoint cubic spline interpolation.

    For each keypoint index, identifies frames where the keypoint's confidence
    is below ``conf_threshold`` and replaces the position with a cubic spline
    value estimated from the confident frames (those with conf >= conf_threshold).

    This is designed to run *after* :func:`interpolate_gaps`, so that whole-frame
    gap-filled ("coasted") frames can also receive better keypoint estimates when
    they have a low-confidence placeholder. Coasted frames have conf=0.0 for all
    keypoints and are therefore treated as low-confidence targets, not knot sources.

    Args:
        builder: Mutable tracklet builder. Modified in-place and returned.
        conf_threshold: Confidence value below which a keypoint is considered
            low-confidence and eligible for interpolation. Default 0.3.

    Returns:
        The same builder, with low-confidence keypoints replaced where possible.
        Keypoints that were interpolated have their conf entry set to 0.0.
        If a keypoint has fewer than 2 confident frames, it is left unchanged.
    """
    if len(builder.frames) < 2:
        return builder

    # Sort all frame data by frame index
    order = sorted(range(len(builder.frames)), key=lambda i: builder.frames[i])
    frames_arr = np.array([builder.frames[i] for i in order], dtype=np.float64)
    kpts_arr = np.stack([builder.keypoints[i] for i in order], axis=0).astype(
        np.float64
    )  # (T, K, 2)
    conf_arr = np.stack([builder.keypoint_conf[i] for i in order], axis=0).astype(
        np.float64
    )  # (T, K)

    _T, K, _ = kpts_arr.shape

    for k in range(K):
        conf_k = conf_arr[:, k]
        confident_mask = conf_k >= conf_threshold

        # Need at least 2 confident frames to build a cubic spline
        if confident_mask.sum() < 2:
            continue

        low_conf_mask = ~confident_mask

        # No low-confidence frames for this keypoint — nothing to do
        if not low_conf_mask.any():
            continue

        knot_frames = frames_arr[confident_mask]
        knot_x = kpts_arr[confident_mask, k, 0]
        knot_y = kpts_arr[confident_mask, k, 1]

        cs_x = CubicSpline(knot_frames, knot_x)
        cs_y = CubicSpline(knot_frames, knot_y)

        for t in np.where(low_conf_mask)[0]:
            f_val = frames_arr[t]
            kpts_arr[t, k, 0] = cs_x(f_val)
            kpts_arr[t, k, 1] = cs_y(f_val)
            conf_arr[t, k] = 0.0

    # Write updated arrays back into the builder in sorted order
    for new_i, orig_i in enumerate(order):
        builder.keypoints[orig_i] = kpts_arr[new_i].astype(np.float32)
        builder.keypoint_conf[orig_i] = conf_arr[new_i].astype(np.float32)

    return builder


# ---------------------------------------------------------------------------
# KeypointTracker — public single-pass wrapper
# ---------------------------------------------------------------------------


class KeypointTracker:
    """Single-pass keypoint tracker with ORU/OCR occlusion recovery.

    Runs a single forward pass using _SinglePassTracker with ORU/OCR mechanisms
    for occlusion recovery. Small temporal gaps are filled via cubic-spline
    interpolation before returning Tracklet2D objects.

    The bidirectional merge was removed after Phase 84-01 investigation showed
    it produced 44 duplicate-inflated tracks vs 37 for the clean forward-only
    pass (0.942 coverage each). The N:1 mismatch between forward fragments and
    backward single-track caused unmatched duplicates with identical centroids.

    Args:
        camera_id: Camera identifier.
        max_age: Frames to coast before culling a track. Maps to
            TrackingConfig.max_coast_frames.
        n_init: Hit-streak threshold for track confirmation.
        det_thresh: Floor detection confidence — anything below is discarded.
        track_thresh: High/low confidence split for two-phase matching.
            Defaults to det_thresh (single-pass behavior) when not provided.
        birth_thresh: Minimum confidence for unmatched Phase 1 detections
            to birth new tracks. Defaults to det_thresh when not provided.
        base_r: KF base measurement noise variance.
        lambda_ocm: OCM weight in the cost matrix.
        sigmas: Per-keypoint OKS sigmas. Defaults to DEFAULT_SIGMAS.
        max_gap_frames: Maximum gap size to fill via spline interpolation.
        kpt_conf_threshold: Confidence threshold below which individual
            keypoints are considered low-confidence and eligible for
            per-keypoint spline interpolation. Default 0.3.
        centroid_keypoint_index: Keypoint index used as centroid (unused in
            _SinglePassTracker but retained for API compatibility).
        centroid_confidence_floor: Confidence floor for centroid selection
            (retained for API symmetry).
    """

    def __init__(
        self,
        camera_id: str,
        max_age: int = 15,
        n_init: int = 3,
        det_thresh: float = 0.5,
        track_thresh: float | None = None,
        birth_thresh: float | None = None,
        base_r: float = 10.0,
        lambda_ocm: float = 0.2,
        sigmas: np.ndarray | None = None,
        max_gap_frames: int = 5,
        kpt_conf_threshold: float = 0.3,
        match_cost_threshold: float = 1.2,
        ocr_threshold: float = 0.5,
        max_match_distance: float = 75.0,
        centroid_keypoint_index: int = 2,  # API symmetry only
        centroid_confidence_floor: float = 0.3,  # API symmetry only
        merger_distance: float = 30.0,
        merger_max_age: int = 90,
    ) -> None:
        from types import SimpleNamespace

        from aquapose.core.tracking.keypoint_sigmas import DEFAULT_SIGMAS

        self._camera_id = camera_id
        self._max_age = max_age
        self._n_init = n_init
        self._det_thresh = det_thresh
        self._track_thresh = track_thresh if track_thresh is not None else det_thresh
        self._birth_thresh = birth_thresh if birth_thresh is not None else det_thresh
        self._base_r = base_r
        self._lambda_ocm = lambda_ocm
        self._sigmas = sigmas if sigmas is not None else DEFAULT_SIGMAS.copy()
        self._max_gap_frames = max_gap_frames
        self._kpt_conf_threshold = kpt_conf_threshold
        self._match_cost_threshold = match_cost_threshold
        self._ocr_threshold = ocr_threshold
        self._max_match_distance = max_match_distance
        self._merger_distance = merger_distance
        self._merger_max_age = merger_max_age

        config = SimpleNamespace(
            max_age=max_age,
            n_init=n_init,
            det_thresh=det_thresh,
            track_thresh=self._track_thresh,
            birth_thresh=self._birth_thresh,
            base_r=base_r,
            lambda_ocm=lambda_ocm,
            sigmas=self._sigmas,
            match_cost_threshold=match_cost_threshold,
            ocr_threshold=ocr_threshold,
            max_match_distance=max_match_distance,
            merger_distance=merger_distance,
            merger_max_age=merger_max_age,
        )

        self._fwd_tracker = _SinglePassTracker(camera_id=camera_id, config=config)

        # Cached result (set after get_tracklets() is called)
        self._cached_tracklets: list[Tracklet2D] | None = None

    def update(self, frame_idx: int, detections: list[Any]) -> None:
        """Feed one frame of detections to the forward pass tracker.

        Args:
            frame_idx: Current frame index.
            detections: Detection-like objects with bbox, confidence, keypoints.
        """
        self._fwd_tracker.update(frame_idx=frame_idx, detections=detections)
        # Invalidate cached result
        self._cached_tracklets = None

    def get_tracklets(self) -> list[Tracklet2D]:
        """Collect confirmed forward-pass tracklets with gap interpolation.

        Returns:
            List of Tracklet2D objects with original internal tracker IDs.
        """
        if self._cached_tracklets is not None:
            return self._cached_tracklets

        # Collect confirmed builders from forward pass
        fwd_builders = [
            b
            for b in self._fwd_tracker._builders.values()
            if b.detected_count >= self._n_init and b.frames
        ]

        # Apply gap interpolation and per-keypoint confidence interpolation
        gap_filled: list[_KptTrackletBuilder] = []
        for builder in fwd_builders:
            gap_filled.append(
                interpolate_low_confidence_keypoints(
                    interpolate_gaps(builder, max_gap_frames=self._max_gap_frames),
                    conf_threshold=self._kpt_conf_threshold,
                )
            )

        # Preserve internal tracker IDs (globally unique via _next_track_id
        # carry-forward) so that (camera_id, track_id) tuples in
        # fish_tracklet_sets remain unique across chunks.
        final: list[Tracklet2D] = []
        for builder in gap_filled:
            final.append(builder.to_tracklet2d())

        self._cached_tracklets = final
        return self._cached_tracklets

    def get_state(self) -> dict[str, Any]:
        """Serialize tracker state for cross-chunk handoff.

        Returns:
            JSON-safe state dict. All numpy arrays serialized as lists.
        """
        return {
            "kind": "keypoint_oks",
            "camera_id": self._camera_id,
            "max_age": self._max_age,
            "n_init": self._n_init,
            "det_thresh": self._det_thresh,
            "track_thresh": self._track_thresh,
            "birth_thresh": self._birth_thresh,
            "base_r": self._base_r,
            "lambda_ocm": self._lambda_ocm,
            "sigmas": self._sigmas.tolist(),
            "max_gap_frames": self._max_gap_frames,
            "kpt_conf_threshold": self._kpt_conf_threshold,
            "match_cost_threshold": self._match_cost_threshold,
            "ocr_threshold": self._ocr_threshold,
            "max_match_distance": self._max_match_distance,
            "merger_distance": self._merger_distance,
            "merger_max_age": self._merger_max_age,
            "fwd_state": self._fwd_tracker.get_state(),
        }

    @classmethod
    def from_state(cls, camera_id: str, state: dict[str, Any]) -> KeypointTracker:
        """Reconstruct a KeypointTracker from a saved state blob.

        Restores the forward-pass tracker's KF states so that tracks from
        the previous chunk continue seamlessly into the next chunk.

        Args:
            camera_id: Camera identifier.
            state: State dict previously returned by ``get_state()``.

        Returns:
            Restored KeypointTracker ready to receive new frames.
        """
        sigmas = np.array(state["sigmas"], dtype=np.float64)
        det_thresh = state["det_thresh"]
        instance = cls(
            camera_id=camera_id,
            max_age=state["max_age"],
            n_init=state["n_init"],
            det_thresh=det_thresh,
            track_thresh=state.get("track_thresh", det_thresh),
            birth_thresh=state.get("birth_thresh", det_thresh),
            base_r=state["base_r"],
            lambda_ocm=state["lambda_ocm"],
            sigmas=sigmas,
            max_gap_frames=state["max_gap_frames"],
            kpt_conf_threshold=state.get("kpt_conf_threshold", 0.3),
            match_cost_threshold=state.get("match_cost_threshold", 1.2),
            ocr_threshold=state.get("ocr_threshold", 0.5),
            max_match_distance=state.get("max_match_distance", 75.0),
            merger_distance=state.get("merger_distance", 30.0),
            merger_max_age=state.get("merger_max_age", 90),
        )
        # Restore forward tracker from saved state
        instance._fwd_tracker = _SinglePassTracker.from_state(
            camera_id, state["fwd_state"]
        )
        return instance
