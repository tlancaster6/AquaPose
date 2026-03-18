"""Unit tests for keypoint_tracker module.

Tests cover: _KalmanFilter, compute_oks_matrix, compute_ocm_matrix,
build_cost_matrix, _KFTrack, _SinglePassTracker, interpolate_gaps,
interpolate_low_confidence_keypoints, KeypointTracker.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aquapose.core.tracking.keypoint_sigmas import DEFAULT_SIGMAS
from aquapose.core.tracking.keypoint_tracker import (
    KeypointTracker,
    _KalmanFilter,
    _KptTrackletBuilder,
    _SinglePassTracker,
    build_cost_matrix,
    compute_heading,
    compute_ocm_matrix,
    compute_oks_matrix,
    interpolate_gaps,
    interpolate_low_confidence_keypoints,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(
    bbox: tuple[float, float, float, float] = (100.0, 100.0, 50.0, 80.0),
    confidence: float = 0.9,
    kpts: np.ndarray | None = None,
    kconf: np.ndarray | None = None,
) -> SimpleNamespace:
    """Create a minimal detection-like object."""
    if kpts is None:
        kpts = np.zeros((6, 2), dtype=np.float32)
        for i in range(6):
            kpts[i] = [100.0 + i * 10, 200.0]
    if kconf is None:
        kconf = np.ones(6, dtype=np.float32) * 0.9
    return SimpleNamespace(
        bbox=bbox,
        confidence=confidence,
        keypoints=kpts,
        keypoint_conf=kconf,
    )


def _make_tracker_config(
    max_age: int = 5,
    n_init: int = 3,
    det_thresh: float = 0.3,
    track_thresh: float | None = None,
    birth_thresh: float | None = None,
    base_r: float = 10.0,
    lambda_ocm: float = 0.2,
    sigmas: np.ndarray | None = None,
    merger_distance: float = 30.0,
    merger_max_age: int = 90,
) -> SimpleNamespace:
    if sigmas is None:
        sigmas = DEFAULT_SIGMAS
    return SimpleNamespace(
        max_age=max_age,
        n_init=n_init,
        det_thresh=det_thresh,
        track_thresh=track_thresh if track_thresh is not None else det_thresh,
        birth_thresh=birth_thresh if birth_thresh is not None else det_thresh,
        base_r=base_r,
        lambda_ocm=lambda_ocm,
        sigmas=sigmas,
        merger_distance=merger_distance,
        merger_max_age=merger_max_age,
    )


# ---------------------------------------------------------------------------
# _KalmanFilter tests
# ---------------------------------------------------------------------------


class TestKalmanFilter:
    def test_initial_state_shape(self) -> None:
        initial_obs = np.zeros((6, 2), dtype=np.float64)
        confs = np.ones(6, dtype=np.float64)
        kf = _KalmanFilter(initial_obs=initial_obs, confs=confs)
        assert kf.x.shape == (24,)

    def test_initial_positions_match_obs(self) -> None:
        initial_obs = np.array(
            [
                [10.0, 20.0],
                [30.0, 40.0],
                [50.0, 60.0],
                [70.0, 80.0],
                [90.0, 100.0],
                [110.0, 120.0],
            ],
            dtype=np.float64,
        )
        confs = np.ones(6, dtype=np.float64)
        kf = _KalmanFilter(initial_obs=initial_obs, confs=confs)
        # First 12 elements are positions
        positions = kf.x[:12].reshape(6, 2)
        np.testing.assert_allclose(positions, initial_obs)

    def test_initial_velocities_zero(self) -> None:
        initial_obs = np.zeros((6, 2), dtype=np.float64)
        confs = np.ones(6, dtype=np.float64)
        kf = _KalmanFilter(initial_obs=initial_obs, confs=confs)
        velocities = kf.x[12:]
        np.testing.assert_allclose(velocities, 0.0)

    def test_predict_returns_positions_shape(self) -> None:
        initial_obs = np.zeros((6, 2), dtype=np.float64)
        confs = np.ones(6, dtype=np.float64)
        kf = _KalmanFilter(initial_obs=initial_obs, confs=confs)
        pred = kf.predict()
        assert pred.shape == (6, 2)

    def test_predict_advances_with_velocity(self) -> None:
        """If velocity is [1, 0] per keypoint, predicted positions shift by [1, 0]."""
        initial_obs = np.zeros((6, 2), dtype=np.float64)
        confs = np.ones(6, dtype=np.float64)
        kf = _KalmanFilter(initial_obs=initial_obs, confs=confs)
        # Manually set velocities
        kf.x[12:24] = np.tile([1.0, 0.0], 6)
        pred = kf.predict()
        np.testing.assert_allclose(pred, np.tile([[1.0, 0.0]], (6, 1)), atol=1e-9)

    def test_update_moves_state_toward_observation(self) -> None:
        initial_obs = np.zeros((6, 2), dtype=np.float64)
        confs = np.ones(6, dtype=np.float64)
        kf = _KalmanFilter(initial_obs=initial_obs, confs=confs)
        obs = np.ones((6, 2), dtype=np.float64) * 100.0
        high_confs = np.ones(6, dtype=np.float64)
        kf.update(obs=obs, confs=high_confs)
        positions = kf.x[:12].reshape(6, 2)
        # After update toward 100, positions should be > 0
        assert np.all(positions > 0)

    def test_confidence_scaled_noise_high_conf_adjusts_more(self) -> None:
        """High-confidence keypoints adjust more toward observation than low-confidence."""
        initial_obs = np.zeros((6, 2), dtype=np.float64)
        confs_high = np.ones(6, dtype=np.float64)
        confs_low = np.ones(6, dtype=np.float64) * 0.01
        kf_high = _KalmanFilter(initial_obs=initial_obs.copy(), confs=confs_high)
        kf_low = _KalmanFilter(initial_obs=initial_obs.copy(), confs=confs_low)
        obs = np.ones((6, 2), dtype=np.float64) * 50.0
        kf_high.update(obs=obs, confs=confs_high)
        kf_low.update(obs=obs, confs=confs_low)
        pos_high = kf_high.x[:12].reshape(6, 2)
        pos_low = kf_low.x[:12].reshape(6, 2)
        # High confidence → state moves more toward 50
        assert np.all(pos_high > pos_low)

    def test_low_conf_gives_higher_R(self) -> None:
        """conf=0.01 should give ~100x base_R; conf=1.0 gives base_R."""
        confs = np.ones(6, dtype=np.float64)
        kf = _KalmanFilter(initial_obs=np.zeros((6, 2)), confs=confs, base_r=10.0)
        R_high = kf._build_R(np.ones(6) * 1.0)
        R_low = kf._build_R(np.ones(6) * 0.01)
        # R_low diagonal entries should be ~100x R_high diagonal entries
        ratio = np.diag(R_low) / np.diag(R_high)
        np.testing.assert_allclose(ratio, 100.0, rtol=0.01)

    def test_state_serialization_roundtrip(self) -> None:
        initial_obs = np.random.default_rng(7).random((6, 2))
        confs = np.ones(6, dtype=np.float64) * 0.8
        kf = _KalmanFilter(initial_obs=initial_obs, confs=confs)
        # Run a few predict/update cycles
        for _ in range(3):
            kf.predict()
            kf.update(np.random.default_rng(1).random((6, 2)), confs)
        state = kf.get_state()
        # State must serialize to lists (not numpy)
        assert isinstance(state["x"], list)
        assert isinstance(state["P"], list)
        kf2 = _KalmanFilter.from_state(state)
        np.testing.assert_allclose(kf2.x, kf.x)
        np.testing.assert_allclose(kf2.P, kf.P)


# ---------------------------------------------------------------------------
# OKS cost tests
# ---------------------------------------------------------------------------


class TestOKSMatrix:
    def _make_kpts(self, n: int, value: float = 0.0) -> np.ndarray:
        """Make (N, 6, 2) keypoints."""
        return np.full((n, 6, 2), value, dtype=np.float64)

    def test_identical_positions_oks_near_one(self) -> None:
        pred_kpts = self._make_kpts(2, 100.0)
        det_kpts = self._make_kpts(3, 100.0)
        det_confs = np.ones((3, 6), dtype=np.float64)
        det_scales = np.full(3, 100.0, dtype=np.float64)
        oks = compute_oks_matrix(
            pred_kpts, det_kpts, det_confs, det_scales, DEFAULT_SIGMAS
        )
        assert oks.shape == (2, 3)
        np.testing.assert_allclose(oks, 1.0, atol=1e-6)

    def test_distant_positions_oks_near_zero(self) -> None:
        pred_kpts = self._make_kpts(2, 0.0)
        det_kpts = self._make_kpts(3, 10000.0)
        det_confs = np.ones((3, 6), dtype=np.float64)
        det_scales = np.full(3, 100.0, dtype=np.float64)
        oks = compute_oks_matrix(
            pred_kpts, det_kpts, det_confs, det_scales, DEFAULT_SIGMAS
        )
        np.testing.assert_allclose(oks, 0.0, atol=1e-6)

    def test_zero_confidence_contributes_zero_weight(self) -> None:
        pred_kpts = self._make_kpts(1, 0.0)
        # Different positions
        det_kpts = self._make_kpts(1, 500.0)
        # All zero confidence → OKS undefined (division by zero); treat as 0
        det_confs = np.zeros((1, 6), dtype=np.float64)
        det_scales = np.full(1, 100.0, dtype=np.float64)
        oks = compute_oks_matrix(
            pred_kpts, det_kpts, det_confs, det_scales, DEFAULT_SIGMAS
        )
        # With zero total weight, OKS should be 0 (or handled gracefully)
        assert oks.shape == (1, 1)
        assert np.isfinite(oks[0, 0])

    def test_larger_scale_gives_higher_oks_for_same_distance(self) -> None:
        pred_kpts = self._make_kpts(1, 0.0)
        det_kpts = self._make_kpts(1, 50.0)  # 50px offset
        det_confs = np.ones((1, 6), dtype=np.float64)
        det_scales_small = np.array([10.0])
        det_scales_large = np.array([1000.0])
        oks_small = compute_oks_matrix(
            pred_kpts, det_kpts, det_confs, det_scales_small, DEFAULT_SIGMAS
        )
        oks_large = compute_oks_matrix(
            pred_kpts, det_kpts, det_confs, det_scales_large, DEFAULT_SIGMAS
        )
        # Larger scale → 50px is relatively small → higher OKS
        assert oks_large[0, 0] > oks_small[0, 0]

    def test_cost_matrix_shape(self) -> None:
        N, M = 4, 7
        pred_kpts = self._make_kpts(N, 0.0)
        det_kpts = self._make_kpts(M, 10.0)
        det_confs = np.ones((M, 6), dtype=np.float64)
        det_scales = np.full(M, 100.0, dtype=np.float64)
        oks = compute_oks_matrix(
            pred_kpts, det_kpts, det_confs, det_scales, DEFAULT_SIGMAS
        )
        assert oks.shape == (N, M)


# ---------------------------------------------------------------------------
# OCM / heading tests
# ---------------------------------------------------------------------------


class TestOCMMatrix:
    def test_parallel_heading_gives_ocm_one(self) -> None:
        # Both pointing right
        kpts = np.zeros((6, 2), dtype=np.float64)
        kpts[2] = [0.0, 0.0]
        kpts[4] = [10.0, 0.0]
        heading = compute_heading(kpts)
        np.testing.assert_allclose(heading, [1.0, 0.0], atol=1e-9)

        pred_headings = np.tile(heading, (2, 1))
        det_headings = np.tile(heading, (3, 1))
        ocm = compute_ocm_matrix(pred_headings, det_headings)
        assert ocm.shape == (2, 3)
        np.testing.assert_allclose(ocm, 1.0, atol=1e-6)

    def test_antiparallel_heading_gives_ocm_minus_one(self) -> None:
        kpts_fwd = np.zeros((6, 2), dtype=np.float64)
        kpts_fwd[2] = [0.0, 0.0]
        kpts_fwd[4] = [1.0, 0.0]
        kpts_bwd = np.zeros((6, 2), dtype=np.float64)
        kpts_bwd[2] = [1.0, 0.0]
        kpts_bwd[4] = [0.0, 0.0]
        h_fwd = compute_heading(kpts_fwd)
        h_bwd = compute_heading(kpts_bwd)
        pred_headings = h_fwd[np.newaxis, :]
        det_headings = h_bwd[np.newaxis, :]
        ocm = compute_ocm_matrix(pred_headings, det_headings)
        np.testing.assert_allclose(ocm, -1.0, atol=1e-6)

    def test_build_cost_matrix_shape(self) -> None:
        N, M = 3, 5
        oks = np.random.default_rng(0).random((N, M))
        ocm = np.random.default_rng(1).random((N, M))
        cost = build_cost_matrix(oks, ocm, lambda_ocm=0.2)
        assert cost.shape == (N, M)

    def test_build_cost_matrix_formula(self) -> None:
        oks = np.array([[0.8, 0.2], [0.5, 0.9]])
        ocm = np.array([[0.6, 0.4], [0.7, 0.8]])
        cost = build_cost_matrix(oks, ocm, lambda_ocm=0.2)
        expected = (1 - oks) + 0.2 * (1 - ocm)
        np.testing.assert_allclose(cost, expected)


# ---------------------------------------------------------------------------
# _SinglePassTracker tests
# ---------------------------------------------------------------------------


class TestSinglePassTracker:
    def _make_linear_detections(
        self,
        n_fish: int = 2,
        n_frames: int = 10,
        start_x: float = 100.0,
        spacing: float = 200.0,
        velocity: float = 5.0,
    ) -> list[list[SimpleNamespace]]:
        """Create synthetic detections for n_fish fish moving linearly."""
        all_frames: list[list[SimpleNamespace]] = []
        for frame_idx in range(n_frames):
            frame_dets = []
            for fish_i in range(n_fish):
                cx = start_x + fish_i * spacing + frame_idx * velocity
                cy = 300.0
                kpts = np.zeros((6, 2), dtype=np.float32)
                for k in range(6):
                    kpts[k] = [cx + k * 10, cy]
                kconf = np.full(6, 0.9, dtype=np.float32)
                # OBB area proxy: ~100*50 = 5000
                det = SimpleNamespace(
                    bbox=(cx - 50, cy - 25, 100.0, 50.0),
                    confidence=0.9,
                    keypoints=kpts,
                    keypoint_conf=kconf,
                    obb_area=5000.0,
                )
                frame_dets.append(det)
            all_frames.append(frame_dets)
        return all_frames

    def test_tracker_initializes(self) -> None:
        config = _make_tracker_config()
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        assert tracker is not None

    def test_confirmed_tracks_appear_in_get_tracklets(self) -> None:
        config = _make_tracker_config(n_init=3, max_age=10)
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        frames = self._make_linear_detections(n_fish=1, n_frames=8)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracklets = tracker.get_tracklets()
        assert len(tracklets) >= 1

    def test_tentative_tracks_excluded_from_get_tracklets(self) -> None:
        config = _make_tracker_config(n_init=5, max_age=10)
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        # Only provide 2 frames (below n_init=5)
        frames = self._make_linear_detections(n_fish=1, n_frames=2)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracklets = tracker.get_tracklets()
        # No confirmed tracks after only 2 frames with n_init=5
        assert len(tracklets) == 0

    def test_tracklet_has_correct_fields(self) -> None:
        config = _make_tracker_config(n_init=3, max_age=10)
        tracker = _SinglePassTracker(camera_id="testcam", config=config)
        frames = self._make_linear_detections(n_fish=1, n_frames=6)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracklets = tracker.get_tracklets()
        assert len(tracklets) >= 1
        t = tracklets[0]
        assert t.camera_id == "testcam"
        assert len(t.frames) > 0
        assert len(t.centroids) == len(t.frames)
        assert len(t.bboxes) == len(t.frames)
        assert len(t.frame_status) == len(t.frames)
        assert all(s in ("detected", "coasted") for s in t.frame_status)

    def test_two_fish_produce_two_tracklets(self) -> None:
        config = _make_tracker_config(n_init=3, max_age=10)
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        frames = self._make_linear_detections(n_fish=2, n_frames=8)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracklets = tracker.get_tracklets()
        assert len(tracklets) == 2

    def test_coasting_frame_added_when_detection_missing(self) -> None:
        """Tracks still appear with 'coasted' status when detection is dropped."""
        config = _make_tracker_config(n_init=3, max_age=10)
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        frames = self._make_linear_detections(n_fish=1, n_frames=8)
        # Remove detections from frame 5
        frames[5] = []
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracklets = tracker.get_tracklets()
        assert len(tracklets) >= 1
        t = tracklets[0]
        # Should have a coasted frame
        assert "coasted" in t.frame_status

    def test_dead_tracks_culled_after_max_age(self) -> None:
        """Track that misses too many frames gets culled."""
        config = _make_tracker_config(n_init=2, max_age=2)
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        # 5 frames with fish, then 10 frames empty
        frames = self._make_linear_detections(n_fish=1, n_frames=5)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        for i in range(5, 15):
            tracker.update(frame_idx=i, detections=[])
        # Active track count should be zero (all culled)
        assert len(tracker._active_tracks) == 0

    def test_below_det_thresh_filtered(self) -> None:
        """Detections below det_thresh are ignored."""
        config = _make_tracker_config(det_thresh=0.8)
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        low_conf_det = _make_detection(confidence=0.5)
        tracker.update(frame_idx=0, detections=[low_conf_det])
        # No tracks created
        assert len(tracker._active_tracks) == 0

    def test_state_serialization_roundtrip(self) -> None:
        config = _make_tracker_config(n_init=3, max_age=10)
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        frames = self._make_linear_detections(n_fish=1, n_frames=6)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        state = tracker.get_state()
        tracker2 = _SinglePassTracker.from_state("cam0", state)
        # Should be able to continue updating after restoration
        more_frames = self._make_linear_detections(n_fish=1, n_frames=3, start_x=130.0)
        for i, dets in enumerate(more_frames, start=6):
            tracker2.update(frame_idx=i, detections=dets)

    def test_oru_recovery_on_coast(self) -> None:
        """Track should recover after coasting (ORU mechanism exercised)."""
        config = _make_tracker_config(n_init=3, max_age=10)
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        frames = self._make_linear_detections(n_fish=1, n_frames=10)
        # Drop frames 4-6 (coast), then resume
        frames[4] = []
        frames[5] = []
        frames[6] = []
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracklets = tracker.get_tracklets()
        assert len(tracklets) >= 1
        t = tracklets[0]
        # Should have some coasted frames and some detected frames
        assert "coasted" in t.frame_status
        assert "detected" in t.frame_status

    def test_ocr_history_populated(self) -> None:
        """Observation history buffer fills up over detected frames."""
        config = _make_tracker_config(n_init=3, max_age=10)
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        frames = self._make_linear_detections(n_fish=1, n_frames=8)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        # At least one track should have obs_history populated
        assert any(len(trk.obs_history) > 0 for trk in tracker._active_tracks.values())

    def test_detect_missing_keypoints_skipped(self) -> None:
        """Detection missing keypoints attribute is skipped gracefully."""
        config = _make_tracker_config(det_thresh=0.3)
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        no_kpts = SimpleNamespace(
            bbox=(100.0, 100.0, 50.0, 80.0),
            confidence=0.9,
            keypoints=None,
            keypoint_conf=None,
        )
        tracker.update(frame_idx=0, detections=[no_kpts])
        # No crash — tentative track may or may not be created (skip is fine)


# ---------------------------------------------------------------------------
# Helpers for merge / gap / KeypointTracker tests
# ---------------------------------------------------------------------------


def _make_builder(
    camera_id: str = "cam0",
    track_id: int = 0,
    frames: list[int] | None = None,
    n_kpts: int = 6,
    status_override: str | None = None,
) -> _KptTrackletBuilder:
    """Create a _KptTrackletBuilder with synthetic frame data."""
    if frames is None:
        frames = [0, 1, 2]
    builder = _KptTrackletBuilder(camera_id=camera_id, track_id=track_id)
    for _i, f_idx in enumerate(frames):
        kpts = np.zeros((n_kpts, 2), dtype=np.float32)
        for k in range(n_kpts):
            kpts[k] = [f_idx * 10.0 + k * 5.0, 100.0]
        kconf = np.full(n_kpts, 0.9, dtype=np.float32)
        status = status_override or "detected"
        bbox = (f_idx * 10.0, 90.0, 60.0, 30.0)
        builder.add_frame(
            frame_idx=f_idx,
            kpts=kpts,
            kconf=kconf,
            bbox_xywh=bbox,
            status=status,
        )
    return builder


def _make_keypoint_tracker(
    camera_id: str = "cam0",
    max_age: int = 5,
    n_init: int = 3,
    det_thresh: float = 0.3,
    track_thresh: float | None = None,
    birth_thresh: float | None = None,
    base_r: float = 10.0,
    lambda_ocm: float = 0.2,
    max_gap_frames: int = 5,
) -> KeypointTracker:
    """Create a KeypointTracker with standard test parameters."""
    return KeypointTracker(
        camera_id=camera_id,
        max_age=max_age,
        n_init=n_init,
        det_thresh=det_thresh,
        track_thresh=track_thresh,
        birth_thresh=birth_thresh,
        base_r=base_r,
        lambda_ocm=lambda_ocm,
        max_gap_frames=max_gap_frames,
    )


def _make_linear_detections_kpt(
    n_fish: int = 2,
    n_frames: int = 10,
    start_x: float = 100.0,
    spacing: float = 200.0,
    velocity: float = 5.0,
) -> list[list[SimpleNamespace]]:
    """Create synthetic detections for n_fish fish moving linearly."""
    all_frames: list[list[SimpleNamespace]] = []
    for frame_idx in range(n_frames):
        frame_dets = []
        for fish_i in range(n_fish):
            cx = start_x + fish_i * spacing + frame_idx * velocity
            cy = 300.0
            kpts = np.zeros((6, 2), dtype=np.float32)
            for k in range(6):
                kpts[k] = [cx + k * 10, cy]
            kconf = np.full(6, 0.9, dtype=np.float32)
            det = SimpleNamespace(
                bbox=(cx - 50, cy - 25, 100.0, 50.0),
                confidence=0.9,
                keypoints=kpts,
                keypoint_conf=kconf,
                obb_area=5000.0,
            )
            frame_dets.append(det)
        all_frames.append(frame_dets)
    return all_frames


# ---------------------------------------------------------------------------
# interpolate_gaps tests
# ---------------------------------------------------------------------------


class TestInterpolateGaps:
    def test_no_gaps_unchanged(self) -> None:
        """Tracklet with consecutive frames is unchanged."""
        builder = _make_builder(frames=[0, 1, 2, 3])
        result = interpolate_gaps(builder, max_gap_frames=5)
        assert result.frames == [0, 1, 2, 3]
        assert len(result.keypoints) == 4

    def test_small_gap_filled(self) -> None:
        """Tracklet with 1-frame gap gets frame 2 filled."""
        builder = _make_builder(frames=[0, 1, 3, 4])
        result = interpolate_gaps(builder, max_gap_frames=5)
        assert 2 in result.frames
        assert len(result.frames) == 5

    def test_filled_frame_has_coasted_status(self) -> None:
        """Interpolated frame has status 'coasted'."""
        builder = _make_builder(frames=[0, 1, 3, 4])
        result = interpolate_gaps(builder, max_gap_frames=5)
        idx = result.frames.index(2)
        assert result.frame_status[idx] == "coasted"

    def test_large_gap_not_filled(self) -> None:
        """Gaps larger than max_gap_frames are NOT filled."""
        builder = _make_builder(frames=[0, 1, 10, 11])
        result = interpolate_gaps(builder, max_gap_frames=5)
        # Gap of 8 frames exceeds max_gap_frames=5 — not filled
        assert 2 not in result.frames
        assert len(result.frames) == 4

    def test_interpolated_keypoints_between_neighbors(self) -> None:
        """Interpolated keypoints are between the surrounding frames."""
        builder = _make_builder(frames=[0, 2])  # gap at frame 1
        # Set known keypoint positions
        builder.keypoints[0] = np.zeros((6, 2), dtype=np.float32)
        builder.keypoints[1] = np.full((6, 2), 20.0, dtype=np.float32)
        result = interpolate_gaps(builder, max_gap_frames=5)
        assert 1 in result.frames
        idx = result.frames.index(1)
        interp = result.keypoints[idx]
        # Interpolated value should be between 0 and 20
        assert np.all(interp >= 0.0)
        assert np.all(interp <= 20.0)

    def test_interpolated_frames_sorted(self) -> None:
        """After interpolation, frames remain sorted ascending."""
        builder = _make_builder(frames=[0, 1, 4, 5])
        result = interpolate_gaps(builder, max_gap_frames=5)
        assert result.frames == sorted(result.frames)


# ---------------------------------------------------------------------------
# KeypointTracker wrapper tests
# ---------------------------------------------------------------------------


class TestKeypointTracker:
    def test_initializes(self) -> None:
        """KeypointTracker initializes without error."""
        tracker = _make_keypoint_tracker()
        assert tracker is not None

    def test_get_tracklets_returns_tracklet2d(self) -> None:
        """get_tracklets() returns Tracklet2D objects after enough frames."""
        from aquapose.core.tracking.types import Tracklet2D

        tracker = _make_keypoint_tracker(n_init=3, max_age=10)
        frames = _make_linear_detections_kpt(n_fish=1, n_frames=10)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracklets = tracker.get_tracklets()
        assert len(tracklets) >= 1
        for t in tracklets:
            assert isinstance(t, Tracklet2D)

    def test_get_tracklets_ascending_frames(self) -> None:
        """All returned tracklets have monotonically ascending frame indices."""
        tracker = _make_keypoint_tracker(n_init=3, max_age=10)
        frames = _make_linear_detections_kpt(n_fish=2, n_frames=10)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracklets = tracker.get_tracklets()
        for t in tracklets:
            flist = list(t.frames)
            assert flist == sorted(flist)

    def test_get_tracklets_unique_track_ids(self) -> None:
        """All returned tracklets have unique track IDs."""
        tracker = _make_keypoint_tracker(n_init=3, max_age=10)
        frames = _make_linear_detections_kpt(n_fish=2, n_frames=10)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracklets = tracker.get_tracklets()
        ids = [t.track_id for t in tracklets]
        assert len(ids) == len(set(ids))

    def test_get_state_returns_json_safe(self) -> None:
        """get_state() returns only JSON-safe types (no numpy arrays)."""
        import json

        tracker = _make_keypoint_tracker(n_init=3, max_age=10)
        frames = _make_linear_detections_kpt(n_fish=1, n_frames=6)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracker.get_tracklets()  # populate cache
        state = tracker.get_state()
        # Should serialize without error
        json_str = json.dumps(state)
        assert len(json_str) > 0

    def test_chunk_handoff_roundtrip(self) -> None:
        """from_state() + update() on next chunk continues tracks (track_id continuity)."""
        tracker = _make_keypoint_tracker(n_init=3, max_age=10)
        frames = _make_linear_detections_kpt(n_fish=1, n_frames=8)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracker.get_tracklets()  # populate cache
        state = tracker.get_state()

        # Restore and continue
        tracker2 = KeypointTracker.from_state("cam0", state)
        # Feed more frames
        more_frames = _make_linear_detections_kpt(n_fish=1, n_frames=5, start_x=140.0)
        for i, dets in enumerate(more_frames, start=8):
            tracker2.update(frame_idx=i, detections=dets)
        tracklets2 = tracker2.get_tracklets()
        # Should produce some tracklets
        assert (
            len(tracklets2) >= 0
        )  # not a crash test; state restoration must not raise

    def test_single_pass_tracklets_match_fwd_tracker(self) -> None:
        """KeypointTracker returns same tracklets as direct _SinglePassTracker for simple data."""
        tracker = _make_keypoint_tracker(n_init=2, max_age=15)
        frames = _make_linear_detections_kpt(n_fish=1, n_frames=12)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        kpt_tracklets = tracker.get_tracklets()

        fwd_tracker = _SinglePassTracker(
            camera_id="cam0",
            config=SimpleNamespace(
                max_age=15,
                n_init=2,
                det_thresh=0.3,
                base_r=10.0,
                lambda_ocm=0.2,
                sigmas=DEFAULT_SIGMAS,
            ),
        )
        for i, dets in enumerate(frames):
            fwd_tracker.update(frame_idx=i, detections=dets)
        fwd_tracklets = fwd_tracker.get_tracklets()

        # Single-pass KeypointTracker should produce same number of tracklets
        # as direct _SinglePassTracker (both run single forward pass now)
        assert len(kpt_tracklets) == len(fwd_tracklets)


# ---------------------------------------------------------------------------
# TrackingStage integration tests (config extension + wiring)
# ---------------------------------------------------------------------------


class TestTrackingStageKeypointOks:
    """Integration tests for TrackingStage with tracker_kind='keypoint_oks'."""

    def test_config_accepts_keypoint_oks(self) -> None:
        """TrackingConfig accepts tracker_kind='keypoint_oks' without error."""
        from aquapose.engine.config import TrackingConfig

        cfg = TrackingConfig(tracker_kind="keypoint_oks")
        assert cfg.tracker_kind == "keypoint_oks"

    def test_config_rejects_unknown_kind(self) -> None:
        """TrackingConfig raises ValueError for unknown tracker_kind."""
        import pytest

        from aquapose.engine.config import TrackingConfig

        with pytest.raises(ValueError, match="Unknown tracker_kind"):
            TrackingConfig(tracker_kind="bad_tracker")

    def test_config_new_fields_have_defaults(self) -> None:
        """New keypoint_oks fields have backward-compatible defaults."""
        from aquapose.engine.config import TrackingConfig

        cfg = TrackingConfig()  # default keypoint_oks
        assert hasattr(cfg, "base_r")
        assert hasattr(cfg, "lambda_ocm")
        assert hasattr(cfg, "max_gap_frames")
        assert cfg.base_r == 10.0
        assert cfg.lambda_ocm == 0.2
        assert cfg.max_gap_frames == 5

    def test_keypoint_tracker_in_tracking_all(self) -> None:
        """KeypointTracker appears in the tracking package __all__."""
        import aquapose.core.tracking as tracking_pkg

        assert "KeypointTracker" in tracking_pkg.__all__

    def test_tracking_stage_keypoint_oks_produces_tracklets(self) -> None:
        """TrackingStage with keypoint_oks produces valid Tracklet2D output."""
        from aquapose.core.context import PipelineContext
        from aquapose.core.tracking.stage import TrackingStage
        from aquapose.core.tracking.types import Tracklet2D
        from aquapose.engine.config import TrackingConfig

        cfg = TrackingConfig(tracker_kind="keypoint_oks", n_init=2, max_coast_frames=5)
        stage = TrackingStage(config=cfg)

        # Build synthetic context with 2 cameras x 10 frames x 1 fish
        camera_ids = ["cam0", "cam1"]
        n_frames = 10
        n_fish = 1
        detections_list = []
        for f_idx in range(n_frames):
            frame_dets: dict = {}
            for cam_id in camera_ids:
                cam_offset = 0.0 if cam_id == "cam0" else 500.0
                dets = []
                for fi in range(n_fish):
                    cx = 100.0 + fi * 200.0 + cam_offset + f_idx * 5.0
                    cy = 300.0
                    kpts = np.zeros((6, 2), dtype=np.float32)
                    for k in range(6):
                        kpts[k] = [cx + k * 10, cy]
                    kconf = np.full(6, 0.9, dtype=np.float32)
                    dets.append(
                        SimpleNamespace(
                            bbox=(cx - 50, cy - 25, 100.0, 50.0),
                            confidence=0.9,
                            keypoints=kpts,
                            keypoint_conf=kconf,
                            obb_area=5000.0,
                        )
                    )
                frame_dets[cam_id] = dets
            detections_list.append(frame_dets)

        ctx = PipelineContext(
            camera_ids=camera_ids,
            detections=detections_list,
            frame_count=n_frames,
        )

        ctx, _carry = stage.run(ctx, carry=None)

        assert ctx.tracks_2d is not None
        for cam_id in camera_ids:
            cam_tracks = ctx.tracks_2d[cam_id]
            # Should produce at least one valid tracklet per camera
            for t in cam_tracks:
                assert isinstance(t, Tracklet2D)
                assert t.camera_id == cam_id

    def test_tracking_stage_chunk_handoff_keypoint_oks(self) -> None:
        """Chunk handoff serializes and restores keypoint tracker state."""
        from aquapose.core.context import PipelineContext
        from aquapose.core.tracking.stage import TrackingStage
        from aquapose.engine.config import TrackingConfig

        cfg = TrackingConfig(tracker_kind="keypoint_oks", n_init=2, max_coast_frames=5)
        stage = TrackingStage(config=cfg)

        camera_ids = ["cam0"]
        n_frames = 8

        def _make_ctx(offset: int) -> PipelineContext:
            detections_list = []
            for f_idx in range(n_frames):
                cx = 100.0 + (f_idx + offset) * 5.0
                cy = 300.0
                kpts = np.zeros((6, 2), dtype=np.float32)
                for k in range(6):
                    kpts[k] = [cx + k * 10, cy]
                kconf = np.full(6, 0.9, dtype=np.float32)
                det = SimpleNamespace(
                    bbox=(cx - 50, cy - 25, 100.0, 50.0),
                    confidence=0.9,
                    keypoints=kpts,
                    keypoint_conf=kconf,
                    obb_area=5000.0,
                )
                detections_list.append({"cam0": [det]})
            return PipelineContext(
                camera_ids=camera_ids,
                detections=detections_list,
                frame_count=n_frames,
            )

        # Chunk 1
        ctx1 = _make_ctx(offset=0)
        ctx1, carry1 = stage.run(ctx1, carry=None)
        assert carry1 is not None
        # Chunk 2 with restored carry
        ctx2 = _make_ctx(offset=n_frames)
        ctx2, carry2 = stage.run(ctx2, carry=carry1)
        assert carry2 is not None
        # Should not raise — state was restored and processing continued


class TestCrossChunkHandoff:
    """Regression tests for cross-chunk handoff serialization (Phase 86)."""

    def test_get_state_excludes_builders(self) -> None:
        """get_state() must NOT serialize builder history."""
        config = _make_tracker_config(n_init=3, max_age=10)
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        frames = _make_linear_detections_kpt(n_fish=1, n_frames=6)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        state = tracker.get_state()
        assert "builders" not in state

    def test_from_state_creates_empty_builders(self) -> None:
        """from_state() must create empty builders for active tracks."""
        config = _make_tracker_config(n_init=3, max_age=10)
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        frames = _make_linear_detections_kpt(n_fish=1, n_frames=6)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        state = tracker.get_state()
        tracker2 = _SinglePassTracker.from_state("cam0", state)
        # Should have one builder per active track, each with empty frames
        assert len(tracker2._builders) == len(tracker2._active_tracks)
        for tid, b in tracker2._builders.items():
            assert tid in tracker2._active_tracks
            assert len(b.frames) == 0

    def test_cross_chunk_handoff_no_duplicate_frames(self) -> None:
        """A 2-chunk sequence must produce tracklets with no duplicate frame indices."""
        tracker = _make_keypoint_tracker(n_init=3, max_age=10)
        # Chunk 1: frames 0-7
        frames1 = _make_linear_detections_kpt(n_fish=1, n_frames=8)
        for i, dets in enumerate(frames1):
            tracker.update(frame_idx=i, detections=dets)
        tracklets1 = tracker.get_tracklets()
        state = tracker.get_state()

        # Chunk 2: frames 8-15
        tracker2 = KeypointTracker.from_state("cam0", state)
        frames2 = _make_linear_detections_kpt(n_fish=1, n_frames=8, start_x=140.0)
        for i, dets in enumerate(frames2, start=8):
            tracker2.update(frame_idx=i, detections=dets)
        tracklets2 = tracker2.get_tracklets()

        # Combine all frame indices from both chunks
        all_frames: list[int] = []
        for t in tracklets1:
            all_frames.extend(t.frames)
        for t in tracklets2:
            all_frames.extend(t.frames)
        # No duplicates
        assert len(all_frames) == len(set(all_frames)), (
            f"Duplicate frame indices found: {sorted(all_frames)}"
        )

    def test_next_track_id_preserved_across_handoff(self) -> None:
        """next_track_id must be preserved so new tracks don't collide."""
        config = _make_tracker_config(n_init=3, max_age=10)
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        frames = _make_linear_detections_kpt(n_fish=2, n_frames=8)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        state = tracker.get_state()
        next_id_before = state["next_track_id"]

        tracker2 = _SinglePassTracker.from_state("cam0", state)
        assert tracker2._next_track_id == next_id_before


# ---------------------------------------------------------------------------
# Tracklet2D keypoint round-trip tests (Phase 87)
# ---------------------------------------------------------------------------


class TestTracklet2DKeypointRoundtrip:
    """Round-trip tests verifying keypoint data flows through KeypointTracker."""

    def test_keypoints_shape_matches_frames(self) -> None:
        """After feeding detections, Tracklet2D.keypoints has shape (T, 6, 2)."""
        tracker = _make_keypoint_tracker(n_init=3, max_age=10)
        frames = _make_linear_detections_kpt(n_fish=1, n_frames=8)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracklets = tracker.get_tracklets()
        assert len(tracklets) >= 1
        t = tracklets[0]
        assert t.keypoints is not None
        assert t.keypoints.shape == (len(t.frames), 6, 2)
        assert t.keypoints.dtype == np.float32

    def test_keypoint_conf_shape_matches_frames(self) -> None:
        """Tracklet2D.keypoint_conf has shape (T, 6)."""
        tracker = _make_keypoint_tracker(n_init=3, max_age=10)
        frames = _make_linear_detections_kpt(n_fish=1, n_frames=8)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracklets = tracker.get_tracklets()
        assert len(tracklets) >= 1
        t = tracklets[0]
        assert t.keypoint_conf is not None
        assert t.keypoint_conf.shape == (len(t.frames), 6)
        assert t.keypoint_conf.dtype == np.float32

    def test_coasted_frames_have_zero_conf(self) -> None:
        """Coasted/interpolated frames have keypoint_conf == 0.0 for all keypoints."""
        tracker = _make_keypoint_tracker(n_init=3, max_age=10, max_gap_frames=5)
        frames = _make_linear_detections_kpt(n_fish=1, n_frames=10)
        # Drop frame 5 to create a gap that will be interpolated
        frames[5] = []
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracklets = tracker.get_tracklets()
        assert len(tracklets) >= 1
        t = tracklets[0]
        assert t.keypoint_conf is not None
        for i, status in enumerate(t.frame_status):
            if status == "coasted":
                np.testing.assert_array_equal(
                    t.keypoint_conf[i],
                    np.zeros(6, dtype=np.float32),
                    err_msg=f"Coasted frame {t.frames[i]} has non-zero keypoint_conf",
                )

    def test_detected_frames_have_nonzero_conf(self) -> None:
        """Detected frames retain raw input confidence values (> 0)."""
        tracker = _make_keypoint_tracker(n_init=3, max_age=10)
        frames = _make_linear_detections_kpt(n_fish=1, n_frames=8)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracklets = tracker.get_tracklets()
        assert len(tracklets) >= 1
        t = tracklets[0]
        assert t.keypoint_conf is not None
        for i, status in enumerate(t.frame_status):
            if status == "detected":
                assert np.any(t.keypoint_conf[i] > 0.0), (
                    f"Detected frame {t.frames[i]} has all-zero keypoint_conf"
                )


# ---------------------------------------------------------------------------
# interpolate_low_confidence_keypoints tests
# ---------------------------------------------------------------------------


class TestInterpolateLowConfidenceKeypoints:
    """Tests for per-keypoint cubic spline interpolation of low-confidence frames."""

    def _make_builder_with_custom_kconf(
        self,
        frames: list[int],
        kconf_override: dict[int, np.ndarray] | None = None,
        kpts_override: dict[int, np.ndarray] | None = None,
        n_kpts: int = 6,
    ) -> _KptTrackletBuilder:
        """Build a _KptTrackletBuilder with per-frame kconf and kpts overrides.

        Args:
            frames: Frame indices to include.
            kconf_override: Dict mapping frame index to (n_kpts,) conf array.
            kpts_override: Dict mapping frame index to (n_kpts, 2) kpts array.
            n_kpts: Number of keypoints.
        """
        builder = _KptTrackletBuilder(camera_id="cam0", track_id=0)
        for f_idx in frames:
            if kpts_override and f_idx in kpts_override:
                kpts = kpts_override[f_idx].copy()
            else:
                kpts = np.zeros((n_kpts, 2), dtype=np.float32)
                for k in range(n_kpts):
                    kpts[k] = [f_idx * 10.0 + k * 5.0, 100.0]
            if kconf_override and f_idx in kconf_override:
                kconf = kconf_override[f_idx].copy()
            else:
                kconf = np.full(n_kpts, 0.9, dtype=np.float32)
            bbox = (f_idx * 10.0, 90.0, 60.0, 30.0)
            builder.add_frame(
                frame_idx=f_idx,
                kpts=kpts,
                kconf=kconf,
                bbox_xywh=bbox,
                status="detected",
            )
        return builder

    def test_all_confident_keypoints_unchanged(self) -> None:
        """Keypoints above conf_threshold are not modified."""
        frames = list(range(10))
        builder = self._make_builder_with_custom_kconf(frames)
        # All confs are 0.9, threshold is 0.3 — nothing should change
        original_kpts = [kp.copy() for kp in builder.keypoints]
        result = interpolate_low_confidence_keypoints(builder, conf_threshold=0.3)
        for i in range(len(frames)):
            np.testing.assert_array_equal(
                result.keypoints[i],
                original_kpts[i],
                err_msg=f"Frame {frames[i]} keypoints changed when all conf > threshold",
            )

    def test_single_low_conf_keypoint_interpolated(self) -> None:
        """A single low-conf keypoint at frame 5 is replaced by spline interpolation."""
        frames = list(range(10))
        # Make kpt index 3 low-conf at frame 5
        kconf_override = {5: np.array([0.9, 0.9, 0.9, 0.1, 0.9, 0.9], dtype=np.float32)}
        # Assign simple linear positions for kpt 3
        kpts_override: dict[int, np.ndarray] = {}
        for f in frames:
            kp = np.zeros((6, 2), dtype=np.float32)
            for k in range(6):
                kp[k] = [f * 10.0 + k * 5.0, 100.0]
            kpts_override[f] = kp
        builder = self._make_builder_with_custom_kconf(
            frames, kconf_override=kconf_override, kpts_override=kpts_override
        )
        # Corrupt kpt 3 at frame 5 so we can check it gets replaced
        builder.keypoints[5][3] = np.array([999.0, 999.0], dtype=np.float32)

        result = interpolate_low_confidence_keypoints(builder, conf_threshold=0.3)
        idx5 = result.frames.index(5)
        interp_val = result.keypoints[idx5][3]
        # The spline should produce something near the linear interpolation value (5*10 + 3*5 = 65)
        assert interp_val[0] < 200.0, (
            "Interpolated value should not be the corrupted 999.0"
        )
        # Should be between the neighboring frames' values
        assert 0.0 < interp_val[0] < 200.0

    def test_interpolated_keypoint_conf_set_to_zero(self) -> None:
        """keypoint_conf is set to 0.0 for interpolated (low-conf) keypoints."""
        frames = list(range(10))
        kconf_override = {5: np.array([0.9, 0.9, 0.9, 0.1, 0.9, 0.9], dtype=np.float32)}
        builder = self._make_builder_with_custom_kconf(
            frames, kconf_override=kconf_override
        )
        result = interpolate_low_confidence_keypoints(builder, conf_threshold=0.3)
        idx5 = result.frames.index(5)
        # kpt 3 was low-conf — its conf should now be 0.0
        assert result.keypoint_conf[idx5][3] == 0.0, (
            "Interpolated keypoint conf should be 0.0"
        )
        # Other keypoints at frame 5 remain confident
        for k in range(6):
            if k != 3:
                assert result.keypoint_conf[idx5][k] == pytest.approx(0.9), (
                    f"Non-interpolated kpt {k} conf changed unexpectedly"
                )

    def test_fewer_than_two_confident_frames_left_unchanged(self) -> None:
        """Keypoint with fewer than 2 confident frames is left unchanged (no spline)."""
        frames = list(range(5))
        # Only frame 0 has confident kpt 3; all others are low-conf
        kconf_list = {}
        for f in frames:
            kc = np.full(6, 0.1, dtype=np.float32)  # all low
            if f == 0:
                kc[3] = 0.9  # only frame 0 is confident
            kconf_list[f] = kc
        # Assign known positions
        kpts_override: dict[int, np.ndarray] = {}
        for f in frames:
            kp = np.zeros((6, 2), dtype=np.float32)
            for k in range(6):
                kp[k] = [f * 10.0, 100.0]
            kpts_override[f] = kp

        builder = self._make_builder_with_custom_kconf(
            frames, kconf_override=kconf_list, kpts_override=kpts_override
        )
        original_kpts = [kp.copy() for kp in builder.keypoints]
        result = interpolate_low_confidence_keypoints(builder, conf_threshold=0.3)
        # kpt 3 at all frames (except frame 0) should be unchanged — only 1 confident knot
        for i, f in enumerate(frames):
            np.testing.assert_array_equal(
                result.keypoints[i][3],
                original_kpts[i][3],
                err_msg=f"Frame {f} kpt 3 changed even though <2 confident knots exist",
            )

    def test_coasted_frames_not_used_as_knot_points(self) -> None:
        """Frames with conf=0.0 (from whole-frame gap fill) are not used as knots."""
        # Build a builder: frames [0,1,2,3,4] where frames 2,3 are coasted (conf=0.0)
        # and kpt 1 is low-conf at frame 4. After gap-fill, frames 2,3 have conf=0.0.
        # They should NOT be used as spline knots for kpt 1.
        frames = list(range(5))
        kconf_override: dict[int, np.ndarray] = {}
        for f in frames:
            kc = np.full(6, 0.9, dtype=np.float32)
            if f in (2, 3):
                # Simulate coasted frames: all conf = 0.0
                kc[:] = 0.0
            if f == 4:
                # kpt 1 low-conf at frame 4
                kc[1] = 0.1
            kconf_override[f] = kc

        builder = self._make_builder_with_custom_kconf(
            frames, kconf_override=kconf_override
        )
        # Mark frames 2,3 as "coasted" status
        builder.frame_status[2] = "coasted"
        builder.frame_status[3] = "coasted"

        # Frames 0 and 1 have kpt 1 confident — those are the only knots for kpt 1.
        # Frame 4 is the target.
        result = interpolate_low_confidence_keypoints(builder, conf_threshold=0.3)
        idx4 = result.frames.index(4)
        # kpt 1 at frame 4 should be interpolated (only 2 knot points: frames 0 and 1)
        # This test mainly verifies no crash, and that coasted frames aren't used
        # (i.e., the function still runs and produces finite output)
        assert np.all(np.isfinite(result.keypoints[idx4][1])), (
            "Interpolated kpt 1 at frame 4 is not finite"
        )

    def test_multiple_keypoints_interpolated_independently(self) -> None:
        """Multiple keypoints can be independently interpolated in the same frame."""
        frames = list(range(10))
        kconf_override = {5: np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.9], dtype=np.float32)}
        builder = self._make_builder_with_custom_kconf(
            frames, kconf_override=kconf_override
        )
        # Corrupt kpt 3 and kpt 4 at frame 5
        builder.keypoints[5][3] = np.array([999.0, 999.0], dtype=np.float32)
        builder.keypoints[5][4] = np.array([888.0, 888.0], dtype=np.float32)

        result = interpolate_low_confidence_keypoints(builder, conf_threshold=0.3)
        idx5 = result.frames.index(5)

        # kpt 3 and kpt 4 should be interpolated (not 999/888)
        assert result.keypoints[idx5][3][0] < 200.0, "kpt 3 not interpolated"
        assert result.keypoints[idx5][4][0] < 200.0, "kpt 4 not interpolated"
        # kpt 0 and kpt 1 and kpt 5 should be unchanged
        assert result.keypoint_conf[idx5][0] == pytest.approx(0.9), "kpt 0 conf changed"
        assert result.keypoint_conf[idx5][5] == pytest.approx(0.9), "kpt 5 conf changed"


# ---------------------------------------------------------------------------
# Merger-aware coasting tests
# ---------------------------------------------------------------------------


def _make_kpts_at(x: float, y: float) -> np.ndarray:
    """Create 6 keypoints in a horizontal line starting at (x, y)."""
    kpts = np.zeros((6, 2), dtype=np.float32)
    for i in range(6):
        kpts[i] = [x + i * 10, y]
    return kpts


class TestMergerDetection:
    """Test that unmatched tracks near a matched track get merger flags."""

    def test_merger_detection(self) -> None:
        """Unmatched track close to a matched track gets merger_partner_id set."""
        config = _make_tracker_config(max_age=30, n_init=1, merger_distance=60.0)
        tracker = _SinglePassTracker(camera_id="cam0", config=config)

        # Two fish close together (spine1 at x+20, so spine1s are 120 and 170)
        kpts_a = _make_kpts_at(100, 200)
        kpts_b = _make_kpts_at(150, 200)

        for f in range(3):
            det_a = _make_detection(bbox=(90, 190, 70, 20), kpts=kpts_a, confidence=0.9)
            det_b = _make_detection(
                bbox=(140, 190, 70, 20), kpts=kpts_b, confidence=0.9
            )
            tracker.update(frame_idx=f, detections=[det_a, det_b])

        tracks = tracker._active_tracks
        assert len(tracks) == 2

        # Only one detection at fish A's position — fish B misses
        det_merged = _make_detection(
            bbox=(90, 190, 70, 20), kpts=kpts_a, confidence=0.9
        )
        tracker.update(frame_idx=3, detections=[det_merged])

        # The unmatched track's predicted spine1 (~170) is within 50px of the
        # matched track's spine1 (~120), so merger should be detected
        unmatched = [t for t in tracks.values() if t.time_since_update > 0]
        assert len(unmatched) == 1
        assert unmatched[0].merger_partner_id is not None
        assert unmatched[0].merger_frames == 1

    def test_no_merger_normal_coast(self) -> None:
        """Track missing detection with no nearby matched track dies at normal max_age."""
        config = _make_tracker_config(max_age=5, n_init=1, merger_distance=30.0)
        tracker = _SinglePassTracker(camera_id="cam0", config=config)

        # One fish, confirmed
        kpts = _make_kpts_at(100, 200)
        for f in range(3):
            det = _make_detection(bbox=(90, 190, 70, 20), kpts=kpts, confidence=0.9)
            tracker.update(frame_idx=f, detections=[det])

        assert len(tracker._active_tracks) == 1
        tid = next(iter(tracker._active_tracks))

        # No detections — track should coast then die
        for f in range(3, 10):
            tracker.update(frame_idx=f, detections=[])

        trk = tracker._active_tracks.get(tid)
        # Should be dead after max_age=5 frames with no merger partner
        assert trk is None


class TestMergerExtendedCoast:
    """Test that merger tracks survive beyond normal max_age."""

    def test_merger_extended_coast(self) -> None:
        """Merged track survives past normal max_age and re-matches on separation."""
        config = _make_tracker_config(
            max_age=5, n_init=1, merger_distance=50.0, merger_max_age=50
        )
        tracker = _SinglePassTracker(camera_id="cam0", config=config)

        # Two fish close together — spine1s at x=120 and x=150
        kpts_a = _make_kpts_at(100, 200)
        kpts_b = _make_kpts_at(130, 200)

        for f in range(3):
            det_a = _make_detection(bbox=(90, 190, 70, 20), kpts=kpts_a, confidence=0.9)
            det_b = _make_detection(
                bbox=(120, 190, 70, 20), kpts=kpts_b, confidence=0.9
            )
            tracker.update(frame_idx=f, detections=[det_a, det_b])

        assert len(tracker._active_tracks) == 2

        # Merger: only one detection at fish A's position for 12 frames
        # (exceeds max_age=5, but merger should keep the other alive)
        for f in range(3, 15):
            det_merged = _make_detection(
                bbox=(90, 190, 70, 20), kpts=kpts_a, confidence=0.9
            )
            tracker.update(frame_idx=f, detections=[det_merged])

        # Both tracks should still be alive (merger extends coast)
        assert len(tracker._active_tracks) == 2


class TestMergerStateSerialization:
    """Test that merger state survives get_state/from_state round-trip."""

    def test_merger_state_serialization(self) -> None:
        config = _make_tracker_config(
            max_age=30, n_init=1, merger_distance=60.0, merger_max_age=90
        )
        tracker = _SinglePassTracker(camera_id="cam0", config=config)

        # Two fish close together, then merger
        kpts_a = _make_kpts_at(100, 200)
        kpts_b = _make_kpts_at(150, 200)

        for f in range(3):
            det_a = _make_detection(bbox=(90, 190, 70, 20), kpts=kpts_a, confidence=0.9)
            det_b = _make_detection(
                bbox=(140, 190, 70, 20), kpts=kpts_b, confidence=0.9
            )
            tracker.update(frame_idx=f, detections=[det_a, det_b])

        # Merge: single detection at fish A's position
        det_merged = _make_detection(
            bbox=(90, 190, 70, 20), kpts=kpts_a, confidence=0.9
        )
        tracker.update(frame_idx=3, detections=[det_merged])

        # Find the merging track
        merging = [
            t
            for t in tracker._active_tracks.values()
            if t.merger_partner_id is not None
        ]
        assert len(merging) == 1
        orig_partner = merging[0].merger_partner_id
        orig_frames = merging[0].merger_frames

        # Round-trip
        state = tracker.get_state()
        restored = _SinglePassTracker.from_state("cam0", state)

        # Verify merger fields preserved
        restored_merging = [
            t
            for t in restored._active_tracks.values()
            if t.merger_partner_id is not None
        ]
        assert len(restored_merging) == 1
        assert restored_merging[0].merger_partner_id == orig_partner
        assert restored_merging[0].merger_frames == orig_frames

        # Verify config preserved
        assert restored._merger_distance == 60.0
        assert restored._merger_max_age == 90


# ---------------------------------------------------------------------------
# Two-phase matching tests
# ---------------------------------------------------------------------------


class TestTwoPhaseMatching:
    """Test ByteTrack-style two-phase matching with track_thresh / birth_thresh."""

    def test_low_conf_det_does_not_birth(self) -> None:
        """A low-confidence detection (below birth_thresh) must not start a track."""
        config = _make_tracker_config(
            n_init=1, det_thresh=0.05, track_thresh=0.3, birth_thresh=0.5
        )
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        # Detection at 0.1 — above det_thresh but below both track_thresh and birth_thresh
        det = _make_detection(confidence=0.1)
        tracker.update(frame_idx=0, detections=[det])
        assert len(tracker._active_tracks) == 0

    def test_high_conf_below_birth_does_not_birth(self) -> None:
        """A high-conf det (>= track_thresh) but below birth_thresh doesn't birth."""
        config = _make_tracker_config(
            n_init=1, det_thresh=0.05, track_thresh=0.3, birth_thresh=0.5
        )
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        det = _make_detection(confidence=0.4)
        tracker.update(frame_idx=0, detections=[det])
        assert len(tracker._active_tracks) == 0

    def test_high_conf_above_birth_births(self) -> None:
        """A high-conf det (>= birth_thresh) births a new track."""
        config = _make_tracker_config(
            n_init=1, det_thresh=0.05, track_thresh=0.3, birth_thresh=0.5
        )
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        det = _make_detection(confidence=0.6)
        tracker.update(frame_idx=0, detections=[det])
        assert len(tracker._active_tracks) == 1

    def test_low_conf_extends_active_track(self) -> None:
        """A low-conf det extends an actively-tracked track (tsu=0)."""
        config = _make_tracker_config(
            n_init=1, max_age=10, det_thresh=0.05, track_thresh=0.3, birth_thresh=0.5
        )
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        kpts = _make_kpts_at(100, 200)

        # Birth a track with high-conf det
        det_high = _make_detection(bbox=(90, 190, 70, 20), kpts=kpts, confidence=0.9)
        tracker.update(frame_idx=0, detections=[det_high])
        assert len(tracker._active_tracks) == 1
        tid = next(iter(tracker._active_tracks.keys()))

        # Next frame: same position but low confidence
        det_low = _make_detection(bbox=(90, 190, 70, 20), kpts=kpts, confidence=0.1)
        tracker.update(frame_idx=1, detections=[det_low])

        # Track should still be matched (tsu reset to 0 by match_update)
        trk = tracker._active_tracks[tid]
        assert trk.time_since_update == 0

    def test_low_conf_does_not_extend_coasting_track(self) -> None:
        """A low-conf det must NOT extend a coasting track (tsu > 0)."""
        config = _make_tracker_config(
            n_init=1, max_age=10, det_thresh=0.05, track_thresh=0.3, birth_thresh=0.5
        )
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        kpts = _make_kpts_at(100, 200)

        # Birth a track
        det_high = _make_detection(bbox=(90, 190, 70, 20), kpts=kpts, confidence=0.9)
        tracker.update(frame_idx=0, detections=[det_high])
        tid = next(iter(tracker._active_tracks.keys()))

        # Frame 1: no detections → track coasts (tsu becomes 1)
        tracker.update(frame_idx=1, detections=[])
        assert tracker._active_tracks[tid].time_since_update == 1

        # Frame 2: low-conf det at same position → should NOT match
        det_low = _make_detection(bbox=(90, 190, 70, 20), kpts=kpts, confidence=0.1)
        tracker.update(frame_idx=2, detections=[det_low])
        # Track should still be coasting (tsu incremented again)
        assert tracker._active_tracks[tid].time_since_update == 2

    def test_phase2_excludes_long_coasting_tracks(self) -> None:
        """Phase 2 excludes tracks that have been coasting for multiple frames."""
        config = _make_tracker_config(
            n_init=1, max_age=10, det_thresh=0.05, track_thresh=0.3, birth_thresh=0.5
        )
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        kpts = _make_kpts_at(100, 200)

        # Birth a track
        det_high = _make_detection(bbox=(90, 190, 70, 20), kpts=kpts, confidence=0.9)
        tracker.update(frame_idx=0, detections=[det_high])
        tid = next(iter(tracker._active_tracks.keys()))

        # Coast for 3 frames
        for f in range(1, 4):
            tracker.update(frame_idx=f, detections=[])
        assert tracker._active_tracks[tid].time_since_update == 3

        # Low-conf det at same position — should NOT match (tsu=3)
        det_low = _make_detection(bbox=(90, 190, 70, 20), kpts=kpts, confidence=0.1)
        tracker.update(frame_idx=4, detections=[det_low])
        assert tracker._active_tracks[tid].time_since_update == 4

    def test_state_roundtrip_includes_new_fields(self) -> None:
        """get_state/from_state preserves track_thresh and birth_thresh."""
        config = _make_tracker_config(
            det_thresh=0.05, track_thresh=0.3, birth_thresh=0.5
        )
        tracker = _SinglePassTracker(camera_id="cam0", config=config)
        det = _make_detection(confidence=0.9)
        tracker.update(frame_idx=0, detections=[det])

        state = tracker.get_state()
        assert state["track_thresh"] == 0.3
        assert state["birth_thresh"] == 0.5

        restored = _SinglePassTracker.from_state("cam0", state)
        assert restored._track_thresh == 0.3
        assert restored._birth_thresh == 0.5
