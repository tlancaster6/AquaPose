"""Unit tests for swap detector pure functions."""

from __future__ import annotations

import numpy as np
import pytest


class TestConfirmSwap:
    """Tests for the cross-pattern swap confirmation function."""

    def test_clear_swap_positive_margin(self) -> None:
        """Clear swap: A-post matches B-pre and B-post matches A-pre."""
        from aquapose.core.reid.swap_detector import _confirm_swap

        # Fish A was identity [1,0,0,0,...] pre-swap, Fish B was [0,1,0,0,...]
        # After swap: A-post is now B's identity, B-post is now A's identity
        dim = 768
        a_pre = np.zeros(dim, dtype=np.float32)
        a_pre[0] = 1.0
        b_pre = np.zeros(dim, dtype=np.float32)
        b_pre[1] = 1.0

        # Swapped: A-post looks like B-pre, B-post looks like A-pre
        a_post = b_pre.copy()
        b_post = a_pre.copy()

        confirmed, margin = _confirm_swap(a_pre, a_post, b_pre, b_post, threshold=0.15)
        assert confirmed is True
        assert margin > 0.15
        # cross similarities should be 1.0, self similarities 0.0 => margin 1.0
        assert margin == pytest.approx(1.0, abs=1e-5)

    def test_no_swap_negative_margin(self) -> None:
        """No swap: each fish's post matches its own pre."""
        from aquapose.core.reid.swap_detector import _confirm_swap

        dim = 768
        a_pre = np.zeros(dim, dtype=np.float32)
        a_pre[0] = 1.0
        b_pre = np.zeros(dim, dtype=np.float32)
        b_pre[1] = 1.0

        # No swap: A-post still looks like A-pre
        a_post = a_pre.copy()
        b_post = b_pre.copy()

        confirmed, margin = _confirm_swap(a_pre, a_post, b_pre, b_post, threshold=0.15)
        assert confirmed is False
        assert margin < 0.0
        # self similarities 1.0, cross similarities 0.0 => margin -1.0
        assert margin == pytest.approx(-1.0, abs=1e-5)

    def test_marginal_case_below_threshold(self) -> None:
        """Cross > self but margin below threshold => not confirmed."""
        from aquapose.core.reid.swap_detector import _confirm_swap

        dim = 768
        # Create embeddings where cross is slightly better than self
        # but margin is small (e.g., 0.05)
        a_pre = np.zeros(dim, dtype=np.float32)
        a_pre[0] = 1.0
        b_pre = np.zeros(dim, dtype=np.float32)
        b_pre[1] = 1.0

        # Partially swapped: A-post is a mix leaning slightly toward B-pre
        a_post = np.zeros(dim, dtype=np.float32)
        a_post[0] = 0.47  # slight self sim
        a_post[1] = 0.53  # slight cross sim
        a_post /= np.linalg.norm(a_post)

        b_post = np.zeros(dim, dtype=np.float32)
        b_post[0] = 0.53
        b_post[1] = 0.47
        b_post /= np.linalg.norm(b_post)

        confirmed, margin = _confirm_swap(a_pre, a_post, b_pre, b_post, threshold=0.15)
        assert confirmed is False
        # margin should be positive but small (< 0.15)
        assert 0 < margin < 0.15


class TestComputeMeanEmbedding:
    """Tests for mean embedding computation."""

    def test_multiple_vectors_unit_norm(self) -> None:
        """Mean of multiple vectors should be L2-normalized."""
        from aquapose.core.reid.swap_detector import _compute_mean_embedding

        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        embeddings = np.stack([v1, v2])

        result = _compute_mean_embedding(embeddings)
        assert result is not None
        # Should be unit norm
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)
        # Direction should be [0.707, 0.707, 0]
        expected = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_empty_returns_none(self) -> None:
        """Empty input should return None."""
        from aquapose.core.reid.swap_detector import _compute_mean_embedding

        result = _compute_mean_embedding(np.empty((0, 768), dtype=np.float32))
        assert result is None


class TestFindProximityEvents:
    """Tests for 3D proximity-based event detection."""

    def test_two_fish_close(self) -> None:
        """Two fish within threshold produce an event."""
        from aquapose.core.reid.swap_detector import _find_proximity_events

        n_frames = 3
        max_fish = 2
        n_kpts = 6

        frame_index = np.array([100, 101, 102], dtype=np.int64)
        fish_id = np.array([[0, 1], [0, 1], [0, 1]], dtype=np.int32)

        # Fish 0 at origin, fish 1 at 0.05m away (within 0.15m threshold)
        points = np.zeros((n_frames, max_fish, n_kpts, 3), dtype=np.float32)
        for f in range(n_frames):
            points[f, 0, :, :] = 0.0  # fish 0 at origin
            points[f, 1, :, 0] = 0.05  # fish 1 at x=0.05m

        events = _find_proximity_events(frame_index, fish_id, points, threshold_m=0.15)
        assert len(events) >= 1
        # Should collapse consecutive frames into one event
        for _frame, fa, fb in events:
            assert {fa, fb} == {0, 1}

    def test_two_fish_far(self) -> None:
        """Two fish far apart produce no events."""
        from aquapose.core.reid.swap_detector import _find_proximity_events

        n_frames = 3
        max_fish = 2
        n_kpts = 6

        frame_index = np.array([100, 101, 102], dtype=np.int64)
        fish_id = np.array([[0, 1], [0, 1], [0, 1]], dtype=np.int32)

        points = np.zeros((n_frames, max_fish, n_kpts, 3), dtype=np.float32)
        for f in range(n_frames):
            points[f, 0, :, :] = 0.0
            points[f, 1, :, 0] = 1.0  # 1m apart

        events = _find_proximity_events(frame_index, fish_id, points, threshold_m=0.15)
        assert len(events) == 0

    def test_nan_points_skipped(self) -> None:
        """Fish with all-NaN keypoints should be skipped gracefully."""
        from aquapose.core.reid.swap_detector import _find_proximity_events

        n_frames = 1
        max_fish = 2
        n_kpts = 6

        frame_index = np.array([100], dtype=np.int64)
        fish_id = np.array([[0, 1]], dtype=np.int32)

        points = np.full((n_frames, max_fish, n_kpts, 3), np.nan, dtype=np.float32)

        events = _find_proximity_events(frame_index, fish_id, points, threshold_m=0.15)
        assert len(events) == 0

    def test_invalid_fish_id_skipped(self) -> None:
        """Fish IDs < 0 (empty slots) should be skipped."""
        from aquapose.core.reid.swap_detector import _find_proximity_events

        n_frames = 1
        max_fish = 3
        n_kpts = 6

        frame_index = np.array([100], dtype=np.int64)
        fish_id = np.array([[0, -1, 1]], dtype=np.int32)

        points = np.zeros((n_frames, max_fish, n_kpts, 3), dtype=np.float32)
        points[0, 0, :, 0] = 0.0
        points[0, 2, :, 0] = 0.05  # fish 1 close to fish 0

        events = _find_proximity_events(frame_index, fish_id, points, threshold_m=0.15)
        assert len(events) >= 1
        for _, fa, fb in events:
            assert {fa, fb} == {0, 1}


class TestReidEvent:
    """Tests for the ReidEvent dataclass."""

    def test_construction(self) -> None:
        """ReidEvent can be constructed and fields accessed."""
        from aquapose.core.reid.swap_detector import ReidEvent

        event = ReidEvent(
            frame=100,
            fish_a=0,
            fish_b=5,
            cosine_margin=0.25,
            detection_mode="seeded",
            action="confirmed",
        )
        assert event.frame == 100
        assert event.fish_a == 0
        assert event.fish_b == 5
        assert event.cosine_margin == pytest.approx(0.25)
        assert event.detection_mode == "seeded"
        assert event.action == "confirmed"


class TestSwapDetectorConfig:
    """Tests for SwapDetectorConfig defaults."""

    def test_defaults(self) -> None:
        """Config has expected default values."""
        from aquapose.core.reid.swap_detector import SwapDetectorConfig

        cfg = SwapDetectorConfig()
        assert cfg.cosine_margin_threshold == 0.15
        assert cfg.window_frames == 10
        assert cfg.proximity_threshold_m == 0.15
        assert cfg.scan_frame_stride == 1
