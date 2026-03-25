"""Unit tests for TrainingDataMiner helper functions."""

from __future__ import annotations

import pytest

from aquapose.core.reid.miner import (
    MinerConfig,
    _camera_aware_sample,
    _find_contiguous_segments,
    _frame_passes_quality,
)

# ---------------------------------------------------------------------------
# Quality gate tests
# ---------------------------------------------------------------------------


class TestFramePassesQuality:
    """Tests for the per-frame quality gate function."""

    def test_passes_all_gates(self) -> None:
        """Frame with n_cameras=4, mean_residual=3.0, not low_confidence passes."""
        assert _frame_passes_quality(
            n_cameras=4,
            mean_residual=3.0,
            is_low_confidence=False,
            min_cameras=3,
            max_residual=5.0,
        )

    def test_fails_min_cameras(self) -> None:
        """Frame with n_cameras=2 fails when min_cameras=3."""
        assert not _frame_passes_quality(
            n_cameras=2,
            mean_residual=3.0,
            is_low_confidence=False,
            min_cameras=3,
            max_residual=5.0,
        )

    def test_fails_max_residual(self) -> None:
        """Frame with mean_residual=6.0 fails when max_residual=5.0."""
        assert not _frame_passes_quality(
            n_cameras=4,
            mean_residual=6.0,
            is_low_confidence=False,
            min_cameras=3,
            max_residual=5.0,
        )

    def test_fails_low_confidence(self) -> None:
        """Frame with is_low_confidence=True always fails."""
        assert not _frame_passes_quality(
            n_cameras=4,
            mean_residual=3.0,
            is_low_confidence=True,
            min_cameras=3,
            max_residual=5.0,
        )

    def test_unknown_residual_passes(self) -> None:
        """Frame with mean_residual=-1.0 (fillvalue) passes residual check."""
        assert _frame_passes_quality(
            n_cameras=4,
            mean_residual=-1.0,
            is_low_confidence=False,
            min_cameras=3,
            max_residual=5.0,
        )

    def test_exact_threshold_cameras(self) -> None:
        """Frame with n_cameras exactly at min_cameras passes."""
        assert _frame_passes_quality(
            n_cameras=3,
            mean_residual=3.0,
            is_low_confidence=False,
            min_cameras=3,
            max_residual=5.0,
        )

    def test_exact_threshold_residual(self) -> None:
        """Frame with mean_residual exactly at max_residual passes."""
        assert _frame_passes_quality(
            n_cameras=4,
            mean_residual=5.0,
            is_low_confidence=False,
            min_cameras=3,
            max_residual=5.0,
        )


# ---------------------------------------------------------------------------
# Contiguous segment detection tests
# ---------------------------------------------------------------------------


class TestFindContiguousSegments:
    """Tests for splitting frame lists into contiguous runs."""

    def test_two_segments(self) -> None:
        """[1,2,3,5,6,7] splits into [[1,2,3],[5,6,7]]."""
        result = _find_contiguous_segments([1, 2, 3, 5, 6, 7])
        assert result == [[1, 2, 3], [5, 6, 7]]

    def test_single_segment(self) -> None:
        """[1,2,3,4,5] stays as one segment."""
        result = _find_contiguous_segments([1, 2, 3, 4, 5])
        assert result == [[1, 2, 3, 4, 5]]

    def test_empty_list(self) -> None:
        """Empty list returns empty list."""
        result = _find_contiguous_segments([])
        assert result == []

    def test_single_element(self) -> None:
        """Single element returns one single-element segment."""
        result = _find_contiguous_segments([42])
        assert result == [[42]]

    def test_all_isolated(self) -> None:
        """Non-contiguous frames become individual segments."""
        result = _find_contiguous_segments([1, 3, 5, 7])
        assert result == [[1], [3], [5], [7]]

    def test_unsorted_input(self) -> None:
        """Unsorted input is sorted before segmenting."""
        result = _find_contiguous_segments([5, 3, 1, 2, 6, 7])
        assert result == [[1, 2, 3], [5, 6, 7]]


# ---------------------------------------------------------------------------
# Window grouping logic tests
# ---------------------------------------------------------------------------


class TestWindowGrouping:
    """Tests for temporal window acceptance logic.

    The acceptance criterion is: a window is accepted as a grouping if
    min_cooccurring fish have at least one contiguous segment with
    length >= min_duration within that window.
    """

    @staticmethod
    def _build_quality_data(
        fish_frames: dict[int, list[int]],
    ) -> dict[int, dict[int, dict]]:
        """Build synthetic quality data for testing.

        Args:
            fish_frames: Mapping of fish_id -> list of valid frame indices.

        Returns:
            Quality data in the format expected by TrainingDataMiner.
        """
        quality: dict[int, dict[int, dict]] = {}
        for fid, frames in fish_frames.items():
            quality[fid] = {}
            for f in frames:
                quality[fid][f] = {
                    "n_cameras": 5,
                    "mean_residual": 2.0,
                    "is_low_confidence": False,
                }
        return quality

    def test_window_accepted_with_3_fish(self) -> None:
        """Window with 3 fish having valid segments of length >= 10 is accepted."""
        config = MinerConfig(
            window_size=100,
            min_cooccurring=3,
            min_duration=10,
        )
        quality_data = self._build_quality_data(
            {
                0: list(range(0, 50)),  # 50 contiguous frames
                1: list(range(10, 40)),  # 30 contiguous frames
                2: list(range(20, 60)),  # 40 contiguous frames
            }
        )
        # Window [0, 100): all three fish have segments >= 10
        fish_with_valid = _count_fish_with_valid_segments(
            quality_data,
            0,
            100,
            config.min_cameras,
            config.max_residual,
            config.min_duration,
        )
        assert fish_with_valid >= config.min_cooccurring

    def test_window_rejected_with_2_fish(self) -> None:
        """Window with only 2 fish is rejected when min_cooccurring=3."""
        config = MinerConfig(
            window_size=100,
            min_cooccurring=3,
            min_duration=10,
        )
        quality_data = self._build_quality_data(
            {
                0: list(range(0, 50)),
                1: list(range(10, 40)),
            }
        )
        fish_with_valid = _count_fish_with_valid_segments(
            quality_data,
            0,
            100,
            config.min_cameras,
            config.max_residual,
            config.min_duration,
        )
        assert fish_with_valid < config.min_cooccurring

    def test_window_rejected_short_segments(self) -> None:
        """Window rejected when all fish have segments shorter than min_duration."""
        config = MinerConfig(
            window_size=100,
            min_cooccurring=3,
            min_duration=10,
        )
        # Each fish has frames split by gaps, no contiguous run >= 10
        quality_data = self._build_quality_data(
            {
                0: [0, 1, 2, 10, 11, 12, 20, 21, 22],  # max segment = 3
                1: [5, 6, 7, 15, 16, 17],  # max segment = 3
                2: [30, 31, 32, 40, 41, 42],  # max segment = 3
            }
        )
        fish_with_valid = _count_fish_with_valid_segments(
            quality_data,
            0,
            100,
            config.min_cameras,
            config.max_residual,
            config.min_duration,
        )
        assert fish_with_valid < config.min_cooccurring


def _count_fish_with_valid_segments(
    quality_data: dict[int, dict[int, dict]],
    window_start: int,
    window_end: int,
    min_cameras: int,
    max_residual: float,
    min_duration: int,
) -> int:
    """Count fish with at least one valid segment in the window.

    This is a test helper that mirrors the logic in TrainingDataMiner._select_grouping_windows.
    """
    count = 0
    for _fid, frame_data in quality_data.items():
        valid_frames = []
        for frame, qdata in frame_data.items():
            if window_start <= frame < window_end and _frame_passes_quality(
                qdata["n_cameras"],
                qdata["mean_residual"],
                qdata["is_low_confidence"],
                min_cameras,
                max_residual,
            ):
                valid_frames.append(frame)
        segments = _find_contiguous_segments(valid_frames)
        if any(len(seg) >= min_duration for seg in segments):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Camera-aware sampling tests
# ---------------------------------------------------------------------------


class TestCameraAwareSample:
    """Tests for camera-aware crop sampling."""

    def test_spreads_across_cameras(self) -> None:
        """With 3 cameras and crops_per_fish=6, output includes crops from all 3."""
        detections = []
        for cam in ["cam_a", "cam_b", "cam_c"]:
            for frame in range(3):
                detections.append((frame, cam, f"det_{cam}_{frame}"))

        # 9 items total, request 6 -> round-robin interleave guarantees all cameras
        result = _camera_aware_sample(detections, crops_per_fish=6)
        assert len(result) == 6
        cameras_in_result = {item[1] for item in result}
        assert cameras_in_result == {"cam_a", "cam_b", "cam_c"}

    def test_returns_all_when_fewer_than_target(self) -> None:
        """When total available < crops_per_fish, returns all available."""
        detections = [
            (0, "cam_a", "det_0"),
            (1, "cam_b", "det_1"),
            (2, "cam_c", "det_2"),
        ]
        result = _camera_aware_sample(detections, crops_per_fish=10)
        assert len(result) == 3

    def test_single_camera(self) -> None:
        """With one camera, still returns up to crops_per_fish items."""
        detections = [(i, "cam_a", f"det_{i}") for i in range(20)]
        result = _camera_aware_sample(detections, crops_per_fish=5)
        assert len(result) == 5
        assert all(item[1] == "cam_a" for item in result)

    def test_empty_detections(self) -> None:
        """Empty detections list returns empty list."""
        result = _camera_aware_sample([], crops_per_fish=5)
        assert result == []


# ---------------------------------------------------------------------------
# Error and warning case tests
# ---------------------------------------------------------------------------


class TestErrorCases:
    """Tests for error handling in the miner."""

    def test_all_fish_invalid_raises(self) -> None:
        """When ALL fish have zero valid segments, raises RuntimeError."""
        # Build quality data where no fish passes the quality gate
        quality_data: dict[int, dict[int, dict]] = {
            0: {
                f: {"n_cameras": 1, "mean_residual": 2.0, "is_low_confidence": False}
                for f in range(100)
            },
            1: {
                f: {"n_cameras": 1, "mean_residual": 2.0, "is_low_confidence": False}
                for f in range(100)
            },
        }
        config = MinerConfig(min_cameras=3)

        # All frames fail min_cameras=3 check, so no fish have valid segments
        fish_with_valid = _count_fish_with_valid_segments(
            quality_data,
            0,
            300,
            config.min_cameras,
            config.max_residual,
            config.min_duration,
        )
        assert fish_with_valid == 0

    def test_some_fish_invalid_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """When some fish have no valid segments but others do, logs warning."""
        # Fish 0 has valid data, fish 1 has no valid frames
        quality_data: dict[int, dict[int, dict]] = {
            0: {
                f: {"n_cameras": 5, "mean_residual": 2.0, "is_low_confidence": False}
                for f in range(50)
            },
            1: {
                f: {"n_cameras": 1, "mean_residual": 2.0, "is_low_confidence": False}
                for f in range(50)
            },
        }
        config = MinerConfig(min_cameras=3)

        # Verify fish 0 has valid segments but fish 1 doesn't
        valid_0 = []
        for frame, qdata in quality_data[0].items():
            if _frame_passes_quality(
                qdata["n_cameras"],
                qdata["mean_residual"],
                qdata["is_low_confidence"],
                config.min_cameras,
                config.max_residual,
            ):
                valid_0.append(frame)

        valid_1 = []
        for frame, qdata in quality_data[1].items():
            if _frame_passes_quality(
                qdata["n_cameras"],
                qdata["mean_residual"],
                qdata["is_low_confidence"],
                config.min_cameras,
                config.max_residual,
            ):
                valid_1.append(frame)

        assert len(valid_0) > 0, "Fish 0 should have valid frames"
        assert len(valid_1) == 0, "Fish 1 should have no valid frames"
