"""Unit tests for multi-keypoint group validation with changepoint detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from aquapose.core.association.types import TrackletGroup
from aquapose.core.association.validation import validate_groups
from aquapose.core.tracking.types import Tracklet2D

# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockValidationConfig:
    """Mock config satisfying ValidationConfigLike for tests.

    Default threshold is 2.0m for synthetic test geometry where cameras
    are at unit distances and outliers are at ~5m. The converging cameras
    produce near-zero ray-ray distances, but residuals are averaged across
    all others (including outliers), so the threshold must be high enough
    that good tracklets with one bad "other" still pass.
    Production default is 0.025m.
    """

    eviction_reproj_threshold: float = 2.0
    min_cameras_validate: int = 2
    validation_enabled: bool = True
    min_segment_length: int = 3
    keypoint_confidence_floor: float = 0.3


class MockForwardLUT:
    """Mock ForwardLUT returning controlled ray geometry.

    Each camera sits at a different position and emits rays converging
    near the origin for any centroid (0, 0). Optionally accepts a
    pixel-dependent ray function for testing swap scenarios.
    """

    def __init__(
        self,
        camera_id: str,
        origin: np.ndarray,
        direction: np.ndarray,
        *,
        pixel_fn: object | None = None,
    ) -> None:
        self.camera_id = camera_id
        self._origin = origin.astype(np.float32)
        self._direction = (direction / np.linalg.norm(direction)).astype(np.float32)
        self._pixel_fn = pixel_fn

    def cast_ray(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return rays, optionally pixel-dependent."""
        n = pixels.shape[0]
        if self._pixel_fn is not None:
            origins_list = []
            dirs_list = []
            for i in range(n):
                px = pixels[i].numpy()
                o, d = self._pixel_fn(px)
                origins_list.append(o.astype(np.float32))
                dirs_list.append((d / np.linalg.norm(d)).astype(np.float32))
            origins = torch.from_numpy(np.stack(origins_list))
            dirs = torch.from_numpy(np.stack(dirs_list))
            return origins, dirs
        origins = torch.from_numpy(np.tile(self._origin, (n, 1)))
        dirs = torch.from_numpy(np.tile(self._direction, (n, 1)))
        return origins, dirs


def _make_tracklet(
    camera_id: str,
    track_id: int,
    frames: tuple[int, ...],
    centroids: tuple[tuple[float, float], ...] | None = None,
    keypoints: np.ndarray | None = None,
    keypoint_conf: np.ndarray | None = None,
) -> Tracklet2D:
    """Create a Tracklet2D with default centroid (0,0) at each frame."""
    if centroids is None:
        centroids = tuple((0.0, 0.0) for _ in frames)
    bboxes = tuple((0.0, 0.0, 10.0, 10.0) for _ in frames)
    status = tuple("detected" for _ in frames)
    return Tracklet2D(
        camera_id=camera_id,
        track_id=track_id,
        frames=frames,
        centroids=centroids,
        bboxes=bboxes,
        frame_status=status,
        keypoints=keypoints,
        keypoint_conf=keypoint_conf,
    )


def _make_keypoints(n_frames: int, n_kpts: int = 6) -> np.ndarray:
    """Create default keypoints at (0, 0) for all frames."""
    return np.zeros((n_frames, n_kpts, 2), dtype=np.float32)


def _make_keypoint_conf(
    n_frames: int, n_kpts: int = 6, conf: float = 0.9
) -> np.ndarray:
    """Create uniform keypoint confidence."""
    return np.full((n_frames, n_kpts), conf, dtype=np.float32)


def _converging_luts() -> dict[str, MockForwardLUT]:
    """Create 4 mock ForwardLUTs with rays converging near the origin."""
    return {
        "cam_a": MockForwardLUT(
            "cam_a",
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.5]),
        ),
        "cam_b": MockForwardLUT(
            "cam_b",
            np.array([-1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.5]),
        ),
        "cam_c": MockForwardLUT(
            "cam_c",
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.5]),
        ),
        "cam_d": MockForwardLUT(
            "cam_d",
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 1.0, 0.5]),
        ),
    }


def _divergent_lut() -> MockForwardLUT:
    """Create a LUT whose rays miss the convergence point entirely."""
    return MockForwardLUT(
        "cam_outlier",
        np.array([5.0, 5.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )


# ---------------------------------------------------------------------------
# Tests: well-converging groups (keep)
# ---------------------------------------------------------------------------


class TestValidateGroupsKeep:
    """Test that well-converging groups pass validation without changes."""

    def test_all_converging_no_eviction(self) -> None:
        """Group with 4 converging cameras retains all tracklets."""
        frames = tuple(range(20))
        n = len(frames)
        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet(
                    "cam_a",
                    0,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
                _make_tracklet(
                    "cam_b",
                    1,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
                _make_tracklet(
                    "cam_c",
                    2,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
                _make_tracklet(
                    "cam_d",
                    3,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
            ),
        )
        luts = _converging_luts()
        config = MockValidationConfig()
        result = validate_groups([group], luts, config)

        # Should have exactly 1 group with 4 tracklets
        assert len(result) == 1
        assert len(result[0].tracklets) == 4
        assert result[0].fish_id == 0

    def test_confidence_populated(self) -> None:
        """Validated group has confidence and per_frame_confidence set."""
        frames = tuple(range(10))
        n = len(frames)
        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet(
                    "cam_a",
                    0,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
                _make_tracklet(
                    "cam_b",
                    1,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
                _make_tracklet(
                    "cam_c",
                    2,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
            ),
        )
        luts = _converging_luts()
        config = MockValidationConfig()
        result = validate_groups([group], luts, config)

        assert result[0].confidence is not None
        assert result[0].per_frame_confidence is not None
        assert len(result[0].per_frame_confidence) == n
        for c in result[0].per_frame_confidence:
            assert isinstance(c, float)
            assert 0.0 <= c <= 1.0

    def test_consensus_centroids_populated(self) -> None:
        """Validated group has consensus_centroids as (frame, point) pairs."""
        frames = tuple(range(10))
        n = len(frames)
        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet(
                    "cam_a",
                    0,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
                _make_tracklet(
                    "cam_b",
                    1,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
                _make_tracklet(
                    "cam_c",
                    2,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
            ),
        )
        luts = _converging_luts()
        config = MockValidationConfig()
        result = validate_groups([group], luts, config)

        refined = result[0]
        assert refined.consensus_centroids is not None
        assert isinstance(refined.consensus_centroids, tuple)
        assert len(refined.consensus_centroids) == n
        for frame_idx, point_3d in refined.consensus_centroids:
            assert isinstance(frame_idx, int)
            assert frame_idx in frames
            if point_3d is not None:
                assert isinstance(point_3d, np.ndarray)
                assert point_3d.shape == (3,)


# ---------------------------------------------------------------------------
# Tests: eviction (uniformly bad tracklet)
# ---------------------------------------------------------------------------


class TestValidateGroupsEviction:
    """Test that uniformly bad tracklets are evicted as singletons."""

    def test_outlier_evicted(self) -> None:
        """Tracklet with divergent rays gets evicted as singleton."""
        frames = tuple(range(20))
        n = len(frames)
        luts = _converging_luts()
        luts["cam_outlier"] = _divergent_lut()

        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet(
                    "cam_a",
                    0,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
                _make_tracklet(
                    "cam_b",
                    1,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
                _make_tracklet(
                    "cam_c",
                    2,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
                _make_tracklet(
                    "cam_outlier",
                    99,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
            ),
        )
        config = MockValidationConfig()
        result = validate_groups([group], luts, config)

        # Should have 2 groups: cleaned cluster + evicted singleton
        assert len(result) == 2

        # Main cluster keeps 3 tracklets
        main_group = result[0]
        assert main_group.fish_id == 0
        assert len(main_group.tracklets) == 3
        cam_ids = {t.camera_id for t in main_group.tracklets}
        assert "cam_outlier" not in cam_ids

        # Singleton for evicted tracklet
        singleton = result[1]
        assert len(singleton.tracklets) == 1
        assert singleton.tracklets[0].camera_id == "cam_outlier"
        assert singleton.confidence == 0.1


# ---------------------------------------------------------------------------
# Tests: splitting (temporal swap)
# ---------------------------------------------------------------------------


class TestValidateGroupsSplit:
    """Test that a temporal swap is detected and the tracklet is split."""

    def test_swap_detected_and_split(self) -> None:
        """Tracklet consistent for first half, inconsistent for second half is split."""
        n_frames = 20
        frames = tuple(range(n_frames))
        half = n_frames // 2

        # cam_swap: first half converges with group, second half diverges
        def swap_pixel_fn(px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Return different rays for first and second half of the sequence."""
            # Unused - we control via per-frame keypoints instead
            return np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.5])

        # Build keypoints: first half at (0,0), second half at (500, 500)
        # which will produce divergent rays from a LUT that returns fixed rays
        kpts_swap = np.zeros((n_frames, 6, 2), dtype=np.float32)
        kpts_swap[half:, :, :] = 500.0  # second half far from (0,0)

        # Create a LUT for the swap camera that gives divergent rays for far pixels
        def swap_lut_fn(px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if px[0] > 100:  # far pixels -> divergent ray
                return np.array([5.0, 5.0, 0.0]), np.array([0.0, 0.0, 1.0])
            else:  # near pixels -> converging ray
                return np.array([1.0, 1.0, 0.0]), np.array([-1.0, -1.0, 0.5])

        luts = _converging_luts()
        luts["cam_swap"] = MockForwardLUT(
            "cam_swap",
            np.array([1.0, 1.0, 0.0]),
            np.array([-1.0, -1.0, 0.5]),
            pixel_fn=swap_lut_fn,
        )

        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet(
                    "cam_a",
                    0,
                    frames,
                    keypoints=_make_keypoints(n_frames),
                    keypoint_conf=_make_keypoint_conf(n_frames),
                ),
                _make_tracklet(
                    "cam_b",
                    1,
                    frames,
                    keypoints=_make_keypoints(n_frames),
                    keypoint_conf=_make_keypoint_conf(n_frames),
                ),
                _make_tracklet(
                    "cam_c",
                    2,
                    frames,
                    keypoints=_make_keypoints(n_frames),
                    keypoint_conf=_make_keypoint_conf(n_frames),
                ),
                _make_tracklet(
                    "cam_swap",
                    3,
                    frames,
                    keypoints=kpts_swap,
                    keypoint_conf=_make_keypoint_conf(n_frames),
                ),
            ),
        )
        config = MockValidationConfig(min_segment_length=3)
        result = validate_groups([group], luts, config)

        # Should have the main group + at least one singleton from the split
        main_groups = [g for g in result if len(g.tracklets) > 1]
        singletons = [g for g in result if len(g.tracklets) == 1]

        assert len(main_groups) >= 1
        assert len(singletons) >= 1

        # The main group should have the consistent segment of the split tracklet
        main = main_groups[0]
        swap_tracklets = [t for t in main.tracklets if t.camera_id == "cam_swap"]
        if swap_tracklets:
            # Consistent segment should be shorter than the original
            assert len(swap_tracklets[0].frames) < n_frames

        # There should be a singleton with the inconsistent segment
        swap_singletons = [
            g for g in singletons if g.tracklets[0].camera_id == "cam_swap"
        ]
        assert len(swap_singletons) >= 1

    def test_split_tracklets_have_unique_ids(self) -> None:
        """Split tracklets get new unique track_ids that don't collide."""
        n_frames = 20
        frames = tuple(range(n_frames))

        kpts_swap = np.zeros((n_frames, 6, 2), dtype=np.float32)
        kpts_swap[n_frames // 2 :, :, :] = 500.0

        def swap_lut_fn(px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if px[0] > 100:
                return np.array([5.0, 5.0, 0.0]), np.array([0.0, 0.0, 1.0])
            return np.array([1.0, 1.0, 0.0]), np.array([-1.0, -1.0, 0.5])

        luts = _converging_luts()
        luts["cam_swap"] = MockForwardLUT(
            "cam_swap",
            np.array([1.0, 1.0, 0.0]),
            np.array([-1.0, -1.0, 0.5]),
            pixel_fn=swap_lut_fn,
        )

        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet(
                    "cam_a",
                    0,
                    frames,
                    keypoints=_make_keypoints(n_frames),
                    keypoint_conf=_make_keypoint_conf(n_frames),
                ),
                _make_tracklet(
                    "cam_b",
                    1,
                    frames,
                    keypoints=_make_keypoints(n_frames),
                    keypoint_conf=_make_keypoint_conf(n_frames),
                ),
                _make_tracklet(
                    "cam_c",
                    2,
                    frames,
                    keypoints=_make_keypoints(n_frames),
                    keypoint_conf=_make_keypoint_conf(n_frames),
                ),
                _make_tracklet(
                    "cam_swap",
                    3,
                    frames,
                    keypoints=kpts_swap,
                    keypoint_conf=_make_keypoint_conf(n_frames),
                ),
            ),
        )
        config = MockValidationConfig(min_segment_length=3)
        result = validate_groups([group], luts, config)

        # Collect all track_ids
        all_ids = set()
        for g in result:
            for t in g.tracklets:
                key = (t.camera_id, t.track_id)
                assert key not in all_ids, f"Duplicate track key: {key}"
                all_ids.add(key)

    def test_split_tracklets_have_correct_slices(self) -> None:
        """Split tracklets have correctly sliced frames, centroids, etc."""
        n_frames = 20
        frames = tuple(range(n_frames))

        kpts_swap = np.zeros((n_frames, 6, 2), dtype=np.float32)
        kpts_swap[n_frames // 2 :, :, :] = 500.0

        def swap_lut_fn(px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if px[0] > 100:
                return np.array([5.0, 5.0, 0.0]), np.array([0.0, 0.0, 1.0])
            return np.array([1.0, 1.0, 0.0]), np.array([-1.0, -1.0, 0.5])

        luts = _converging_luts()
        luts["cam_swap"] = MockForwardLUT(
            "cam_swap",
            np.array([1.0, 1.0, 0.0]),
            np.array([-1.0, -1.0, 0.5]),
            pixel_fn=swap_lut_fn,
        )

        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet(
                    "cam_a",
                    0,
                    frames,
                    keypoints=_make_keypoints(n_frames),
                    keypoint_conf=_make_keypoint_conf(n_frames),
                ),
                _make_tracklet(
                    "cam_b",
                    1,
                    frames,
                    keypoints=_make_keypoints(n_frames),
                    keypoint_conf=_make_keypoint_conf(n_frames),
                ),
                _make_tracklet(
                    "cam_c",
                    2,
                    frames,
                    keypoints=_make_keypoints(n_frames),
                    keypoint_conf=_make_keypoint_conf(n_frames),
                ),
                _make_tracklet(
                    "cam_swap",
                    3,
                    frames,
                    keypoints=kpts_swap,
                    keypoint_conf=_make_keypoint_conf(n_frames),
                ),
            ),
        )
        config = MockValidationConfig(min_segment_length=3)
        result = validate_groups([group], luts, config)

        # Find split fragments from cam_swap
        swap_tracklets = []
        for g in result:
            for t in g.tracklets:
                if t.camera_id == "cam_swap":
                    swap_tracklets.append(t)

        # Should have 2 fragments if split occurred
        if len(swap_tracklets) == 2:
            # Frames should be disjoint and cover the original
            all_frames = set()
            for t in swap_tracklets:
                for f in t.frames:
                    assert f not in all_frames, f"Duplicate frame {f}"
                    all_frames.add(f)

            # Each fragment should have consistent lengths
            for t in swap_tracklets:
                n = len(t.frames)
                assert len(t.centroids) == n
                assert len(t.bboxes) == n
                assert len(t.frame_status) == n
                if t.keypoints is not None:
                    assert t.keypoints.shape[0] == n
                if t.keypoint_conf is not None:
                    assert t.keypoint_conf.shape[0] == n


# ---------------------------------------------------------------------------
# Tests: majority consistent (>50%) -> keep
# ---------------------------------------------------------------------------


class TestMajorityConsistentKept:
    """Test that tracklets with >50% consistent frames are kept as-is."""

    def test_mostly_consistent_kept(self) -> None:
        """Tracklet with 60% consistent frames is kept, not split or evicted."""
        n_frames = 20
        frames = tuple(range(n_frames))

        # Create a tracklet where only a few frames diverge (40% bad)
        n_bad = 8  # 8 out of 20 = 40% bad, 60% good
        kpts_partial = np.zeros((n_frames, 6, 2), dtype=np.float32)
        # Scatter bad frames throughout
        bad_indices = list(range(0, n_bad * 2, 2))[:n_bad]
        kpts_partial[bad_indices, :, :] = 500.0

        def partial_lut_fn(px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if px[0] > 100:
                return np.array([5.0, 5.0, 0.0]), np.array([0.0, 0.0, 1.0])
            return np.array([1.0, 1.0, 0.0]), np.array([-1.0, -1.0, 0.5])

        luts = _converging_luts()
        luts["cam_partial"] = MockForwardLUT(
            "cam_partial",
            np.array([1.0, 1.0, 0.0]),
            np.array([-1.0, -1.0, 0.5]),
            pixel_fn=partial_lut_fn,
        )

        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet(
                    "cam_a",
                    0,
                    frames,
                    keypoints=_make_keypoints(n_frames),
                    keypoint_conf=_make_keypoint_conf(n_frames),
                ),
                _make_tracklet(
                    "cam_b",
                    1,
                    frames,
                    keypoints=_make_keypoints(n_frames),
                    keypoint_conf=_make_keypoint_conf(n_frames),
                ),
                _make_tracklet(
                    "cam_c",
                    2,
                    frames,
                    keypoints=_make_keypoints(n_frames),
                    keypoint_conf=_make_keypoint_conf(n_frames),
                ),
                _make_tracklet(
                    "cam_partial",
                    3,
                    frames,
                    keypoints=kpts_partial,
                    keypoint_conf=_make_keypoint_conf(n_frames),
                ),
            ),
        )
        config = MockValidationConfig(min_segment_length=3)
        result = validate_groups([group], luts, config)

        # Should have 1 group with all 4 tracklets (cam_partial kept)
        main_groups = [g for g in result if len(g.tracklets) > 1]
        assert len(main_groups) == 1
        cam_ids = {t.camera_id for t in main_groups[0].tracklets}
        assert "cam_partial" in cam_ids


# ---------------------------------------------------------------------------
# Tests: validation disabled
# ---------------------------------------------------------------------------


class TestValidationDisabled:
    """Test validation_enabled=False returns groups unchanged."""

    def test_disabled_returns_unchanged(self) -> None:
        """Groups pass through unchanged when validation is disabled."""
        frames = tuple(range(10))
        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet("cam_a", 0, frames),
                _make_tracklet("cam_b", 1, frames),
                _make_tracklet("cam_c", 2, frames),
            ),
        )
        config = MockValidationConfig(validation_enabled=False)
        result = validate_groups([group], {}, config)

        assert len(result) == 1
        assert result[0] is group


# ---------------------------------------------------------------------------
# Tests: min cameras threshold
# ---------------------------------------------------------------------------


class TestMinCamerasValidate:
    """Test groups below min_cameras_validate are skipped."""

    def test_below_threshold_skipped(self) -> None:
        """Group with 1 camera (below min_cameras_validate=2) is skipped."""
        frames = tuple(range(10))
        group = TrackletGroup(
            fish_id=0,
            tracklets=(_make_tracklet("cam_a", 0, frames),),
            confidence=0.8,
        )
        luts = _converging_luts()
        config = MockValidationConfig(min_cameras_validate=2)
        result = validate_groups([group], luts, config)

        assert len(result) == 1
        assert result[0] is group
        assert result[0].confidence == 0.8


# ---------------------------------------------------------------------------
# Tests: thin group dissolution
# ---------------------------------------------------------------------------


class TestThinGroupDissolution:
    """Test that groups dissolve to singletons when only 1 camera remains."""

    def test_thin_group_dissolved(self) -> None:
        """Group reduced to 1 camera after evictions is dissolved to singletons."""
        frames = tuple(range(20))
        n = len(frames)
        luts = _converging_luts()
        luts["cam_outlier1"] = MockForwardLUT(
            "cam_outlier1",
            np.array([5.0, 5.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        luts["cam_outlier2"] = MockForwardLUT(
            "cam_outlier2",
            np.array([5.0, -5.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )

        # Group with 1 good camera + 2 outlier cameras -> after eviction only 1 remains
        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet(
                    "cam_a",
                    0,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
                _make_tracklet(
                    "cam_outlier1",
                    1,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
                _make_tracklet(
                    "cam_outlier2",
                    2,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
            ),
        )
        config = MockValidationConfig()
        result = validate_groups([group], luts, config)

        # All tracklets should be singletons now
        for g in result:
            assert len(g.tracklets) == 1


# ---------------------------------------------------------------------------
# Tests: keypoints=None fallback
# ---------------------------------------------------------------------------


class TestKeypointsNoneFallback:
    """Test that tracklets with keypoints=None fall back to centroid-only."""

    def test_no_keypoints_no_crash(self) -> None:
        """Validation runs without crash when keypoints are None."""
        frames = tuple(range(20))
        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet("cam_a", 0, frames),
                _make_tracklet("cam_b", 1, frames),
                _make_tracklet("cam_c", 2, frames),
            ),
        )
        luts = _converging_luts()
        config = MockValidationConfig()
        result = validate_groups([group], luts, config)

        # Should not crash and should produce valid output
        assert len(result) >= 1
        main = result[0]
        assert main.per_frame_confidence is not None
        assert main.consensus_centroids is not None

    def test_no_keypoints_outlier_evicted(self) -> None:
        """Centroid-only fallback still evicts divergent tracklets."""
        frames = tuple(range(20))
        luts = _converging_luts()
        luts["cam_outlier"] = _divergent_lut()

        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet("cam_a", 0, frames),
                _make_tracklet("cam_b", 1, frames),
                _make_tracklet("cam_c", 2, frames),
                _make_tracklet("cam_outlier", 99, frames),
            ),
        )
        config = MockValidationConfig()
        result = validate_groups([group], luts, config)

        # Should evict the outlier
        assert len(result) == 2
        main = next(g for g in result if len(g.tracklets) > 1)
        cam_ids = {t.camera_id for t in main.tracklets}
        assert "cam_outlier" not in cam_ids
