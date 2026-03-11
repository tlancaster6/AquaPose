"""Unit tests for keypoint_tracker module.

Tests cover: _KalmanFilter, compute_oks_matrix, compute_ocm_matrix,
build_cost_matrix, _KFTrack, _SinglePassTracker.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from aquapose.core.tracking.keypoint_sigmas import DEFAULT_SIGMAS
from aquapose.core.tracking.keypoint_tracker import (
    _KalmanFilter,
    _SinglePassTracker,
    build_cost_matrix,
    compute_heading,
    compute_ocm_matrix,
    compute_oks_matrix,
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
    base_r: float = 10.0,
    lambda_ocm: float = 0.2,
    sigmas: np.ndarray | None = None,
) -> SimpleNamespace:
    if sigmas is None:
        sigmas = DEFAULT_SIGMAS
    return SimpleNamespace(
        max_age=max_age,
        n_init=n_init,
        det_thresh=det_thresh,
        base_r=base_r,
        lambda_ocm=lambda_ocm,
        sigmas=sigmas,
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
        tracker = _SinglePassTracker(
            camera_id="cam0", direction="forward", config=config
        )
        assert tracker is not None

    def test_confirmed_tracks_appear_in_get_tracklets(self) -> None:
        config = _make_tracker_config(n_init=3, max_age=10)
        tracker = _SinglePassTracker(
            camera_id="cam0", direction="forward", config=config
        )
        frames = self._make_linear_detections(n_fish=1, n_frames=8)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracklets = tracker.get_tracklets()
        assert len(tracklets) >= 1

    def test_tentative_tracks_excluded_from_get_tracklets(self) -> None:
        config = _make_tracker_config(n_init=5, max_age=10)
        tracker = _SinglePassTracker(
            camera_id="cam0", direction="forward", config=config
        )
        # Only provide 2 frames (below n_init=5)
        frames = self._make_linear_detections(n_fish=1, n_frames=2)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracklets = tracker.get_tracklets()
        # No confirmed tracks after only 2 frames with n_init=5
        assert len(tracklets) == 0

    def test_tracklet_has_correct_fields(self) -> None:
        config = _make_tracker_config(n_init=3, max_age=10)
        tracker = _SinglePassTracker(
            camera_id="testcam", direction="forward", config=config
        )
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
        tracker = _SinglePassTracker(
            camera_id="cam0", direction="forward", config=config
        )
        frames = self._make_linear_detections(n_fish=2, n_frames=8)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        tracklets = tracker.get_tracklets()
        assert len(tracklets) == 2

    def test_coasting_frame_added_when_detection_missing(self) -> None:
        """Tracks still appear with 'coasted' status when detection is dropped."""
        config = _make_tracker_config(n_init=3, max_age=10)
        tracker = _SinglePassTracker(
            camera_id="cam0", direction="forward", config=config
        )
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
        tracker = _SinglePassTracker(
            camera_id="cam0", direction="forward", config=config
        )
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
        tracker = _SinglePassTracker(
            camera_id="cam0", direction="forward", config=config
        )
        low_conf_det = _make_detection(confidence=0.5)
        tracker.update(frame_idx=0, detections=[low_conf_det])
        # No tracks created
        assert len(tracker._active_tracks) == 0

    def test_state_serialization_roundtrip(self) -> None:
        config = _make_tracker_config(n_init=3, max_age=10)
        tracker = _SinglePassTracker(
            camera_id="cam0", direction="forward", config=config
        )
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
        tracker = _SinglePassTracker(
            camera_id="cam0", direction="forward", config=config
        )
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
        tracker = _SinglePassTracker(
            camera_id="cam0", direction="forward", config=config
        )
        frames = self._make_linear_detections(n_fish=1, n_frames=8)
        for i, dets in enumerate(frames):
            tracker.update(frame_idx=i, detections=dets)
        # At least one track should have obs_history populated
        assert any(len(trk.obs_history) > 0 for trk in tracker._active_tracks.values())

    def test_detect_missing_keypoints_skipped(self) -> None:
        """Detection missing keypoints attribute is skipped gracefully."""
        config = _make_tracker_config(det_thresh=0.3)
        tracker = _SinglePassTracker(
            camera_id="cam0", direction="forward", config=config
        )
        no_kpts = SimpleNamespace(
            bbox=(100.0, 100.0, 50.0, 80.0),
            confidence=0.9,
            keypoints=None,
            keypoint_conf=None,
        )
        tracker.update(frame_idx=0, detections=[no_kpts])
        # No crash — tentative track may or may not be created (skip is fine)
