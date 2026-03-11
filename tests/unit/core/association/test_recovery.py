"""Unit tests for singleton recovery module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from aquapose.core.association.recovery import recover_singletons
from aquapose.core.association.types import TrackletGroup
from aquapose.core.tracking.types import Tracklet2D

# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockRecoveryConfig:
    """Mock config satisfying RecoveryConfigLike for tests.

    Defaults use large thresholds (1.0m) so that converging rays in unit
    geometry always score below threshold. Production default is 0.025m.
    """

    recovery_enabled: bool = True
    recovery_residual_threshold: float = 1.0
    recovery_min_shared_frames: int = 1
    recovery_min_segment_length: int = 3
    keypoint_confidence_floor: float = 0.3


class MockForwardLUT:
    """Mock ForwardLUT returning controlled ray geometry.

    Each camera sits at a different position and emits rays converging
    near the origin. Optionally accepts a pixel-dependent ray function.
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
    frame_status: tuple[str, ...] | None = None,
) -> Tracklet2D:
    """Create a Tracklet2D with default centroid (0,0) at each frame."""
    if centroids is None:
        centroids = tuple((0.0, 0.0) for _ in frames)
    bboxes = tuple((0.0, 0.0, 10.0, 10.0) for _ in frames)
    if frame_status is None:
        frame_status = tuple("detected" for _ in frames)
    return Tracklet2D(
        camera_id=camera_id,
        track_id=track_id,
        frames=frames,
        centroids=centroids,
        bboxes=bboxes,
        frame_status=frame_status,
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


def _divergent_lut(camera_id: str = "cam_outlier") -> MockForwardLUT:
    """Create a LUT whose rays miss the convergence point entirely."""
    return MockForwardLUT(
        camera_id,
        np.array([5.0, 5.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )


def _make_multi_group(
    frames: tuple[int, ...],
    luts: dict[str, MockForwardLUT],
    fish_id: int = 0,
    n_kpts: int = 6,
) -> TrackletGroup:
    """Create a 3-camera multi-tracklet group with converging rays."""
    n = len(frames)
    return TrackletGroup(
        fish_id=fish_id,
        tracklets=(
            _make_tracklet(
                "cam_a",
                100 + fish_id,
                frames,
                keypoints=_make_keypoints(n, n_kpts),
                keypoint_conf=_make_keypoint_conf(n, n_kpts),
            ),
            _make_tracklet(
                "cam_b",
                200 + fish_id,
                frames,
                keypoints=_make_keypoints(n, n_kpts),
                keypoint_conf=_make_keypoint_conf(n, n_kpts),
            ),
            _make_tracklet(
                "cam_c",
                300 + fish_id,
                frames,
                keypoints=_make_keypoints(n, n_kpts),
                keypoint_conf=_make_keypoint_conf(n, n_kpts),
            ),
        ),
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# Tests: recovery_enabled=False
# ---------------------------------------------------------------------------


class TestRecoveryDisabled:
    """Test that recovery_enabled=False returns groups unchanged."""

    def test_disabled_returns_groups_unchanged(self) -> None:
        """Groups pass through unchanged when recovery is disabled."""
        frames = tuple(range(10))
        group = _make_multi_group(frames, _converging_luts(), fish_id=0)
        singleton_tracklet = _make_tracklet("cam_d", 999, frames)
        singleton = TrackletGroup(
            fish_id=1, tracklets=(singleton_tracklet,), confidence=0.1
        )
        luts = _converging_luts()
        config = MockRecoveryConfig(recovery_enabled=False)
        result = recover_singletons([group, singleton], luts, config)

        # Should return exactly the same groups in same order
        assert len(result) == 2
        assert result[0] is group
        assert result[1] is singleton


# ---------------------------------------------------------------------------
# Tests: min_shared_frames guard
# ---------------------------------------------------------------------------


class TestMinSharedFrames:
    """Test singleton scoring is skipped when overlap is insufficient."""

    def test_no_shared_frames_returns_singleton_unchanged(self) -> None:
        """Singleton with no shared frames with any group is left unchanged."""
        group_frames = tuple(range(0, 10))
        singleton_frames = tuple(range(20, 30))  # no overlap
        luts = _converging_luts()
        group = _make_multi_group(group_frames, luts, fish_id=0)
        singleton_tracklet = _make_tracklet(
            "cam_d",
            999,
            singleton_frames,
            keypoints=_make_keypoints(len(singleton_frames)),
            keypoint_conf=_make_keypoint_conf(len(singleton_frames)),
        )
        singleton = TrackletGroup(
            fish_id=1, tracklets=(singleton_tracklet,), confidence=0.1
        )
        luts["cam_d"] = _divergent_lut("cam_d")
        config = MockRecoveryConfig(recovery_min_shared_frames=3)
        result = recover_singletons([group, singleton], luts, config)

        # Singleton should remain a singleton (no assignment)
        singletons_out = [g for g in result if len(g.tracklets) == 1]
        assert len(singletons_out) == 1

    def test_insufficient_shared_frames_below_min(self) -> None:
        """Singleton with fewer shared frames than min_shared_frames is skipped."""
        group_frames = tuple(range(0, 10))
        singleton_frames = (9, 20, 21, 22)  # only 1 shared frame with group
        luts = _converging_luts()
        luts["cam_d"] = MockForwardLUT(
            "cam_d",
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 1.0, 0.5]),
        )
        group = _make_multi_group(group_frames, luts, fish_id=0)
        singleton_tracklet = _make_tracklet(
            "cam_d",
            999,
            singleton_frames,
            keypoints=_make_keypoints(len(singleton_frames)),
            keypoint_conf=_make_keypoint_conf(len(singleton_frames)),
        )
        singleton = TrackletGroup(
            fish_id=1, tracklets=(singleton_tracklet,), confidence=0.1
        )
        config = MockRecoveryConfig(
            recovery_min_shared_frames=3
        )  # need 3 but only 1 shared
        result = recover_singletons([group, singleton], luts, config)

        singletons_out = [g for g in result if len(g.tracklets) == 1]
        assert len(singletons_out) == 1


# ---------------------------------------------------------------------------
# Tests: whole-group assignment
# ---------------------------------------------------------------------------


class TestWholeAssignment:
    """Test that singletons matching a group are assigned to it."""

    def test_singleton_assigned_to_matching_group(self) -> None:
        """Singleton with converging rays is assigned to the group."""
        frames = tuple(range(20))
        n = len(frames)
        luts = _converging_luts()
        luts["cam_d"] = MockForwardLUT(
            "cam_d",
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 1.0, 0.5]),
        )
        group = _make_multi_group(frames, luts, fish_id=0)
        singleton_tracklet = _make_tracklet(
            "cam_d",
            999,
            frames,
            keypoints=_make_keypoints(n),
            keypoint_conf=_make_keypoint_conf(n),
        )
        singleton = TrackletGroup(
            fish_id=1, tracklets=(singleton_tracklet,), confidence=0.1
        )
        config = MockRecoveryConfig(
            recovery_residual_threshold=5.0,  # very permissive
            recovery_min_shared_frames=1,
        )
        result = recover_singletons([group, singleton], luts, config)

        # Group should now have 4 tracklets
        multi_groups = [g for g in result if len(g.tracklets) > 1]
        assert len(multi_groups) == 1
        assert len(multi_groups[0].tracklets) == 4
        # Singleton group should no longer appear
        singletons_out = [g for g in result if len(g.tracklets) == 1]
        assert len(singletons_out) == 0

    def test_assigned_group_staleness_invalidated(self) -> None:
        """Assigned group has per_frame_confidence=None and consensus_centroids=None."""
        frames = tuple(range(20))
        n = len(frames)
        luts = _converging_luts()
        luts["cam_d"] = MockForwardLUT(
            "cam_d",
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 1.0, 0.5]),
        )
        # Give the group pre-populated confidence/consensus
        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet(
                    "cam_a",
                    100,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
                _make_tracklet(
                    "cam_b",
                    200,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
                _make_tracklet(
                    "cam_c",
                    300,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
            ),
            confidence=0.9,
            per_frame_confidence=tuple(0.9 for _ in frames),
            consensus_centroids=tuple((f, np.zeros(3)) for f in frames),
        )
        singleton_tracklet = _make_tracklet(
            "cam_d",
            999,
            frames,
            keypoints=_make_keypoints(n),
            keypoint_conf=_make_keypoint_conf(n),
        )
        singleton = TrackletGroup(
            fish_id=1, tracklets=(singleton_tracklet,), confidence=0.1
        )
        config = MockRecoveryConfig(
            recovery_residual_threshold=5.0,
            recovery_min_shared_frames=1,
        )
        result = recover_singletons([group, singleton], luts, config)

        multi_groups = [g for g in result if len(g.tracklets) > 1]
        assert len(multi_groups) == 1
        # Staleness: both fields must be None after assignment
        assert multi_groups[0].per_frame_confidence is None
        assert multi_groups[0].consensus_centroids is None

    def test_singleton_not_assigned_when_threshold_too_tight(self) -> None:
        """Singleton with divergent rays is not assigned when threshold is tiny."""
        frames = tuple(range(20))
        n = len(frames)
        luts = _converging_luts()
        luts["cam_outlier"] = _divergent_lut("cam_outlier")
        group = _make_multi_group(frames, luts, fish_id=0)
        singleton_tracklet = _make_tracklet(
            "cam_outlier",
            999,
            frames,
            keypoints=_make_keypoints(n),
            keypoint_conf=_make_keypoint_conf(n),
        )
        singleton = TrackletGroup(
            fish_id=1, tracklets=(singleton_tracklet,), confidence=0.1
        )
        config = MockRecoveryConfig(
            recovery_residual_threshold=0.0001,  # impossibly tight
            recovery_min_shared_frames=1,
        )
        result = recover_singletons([group, singleton], luts, config)

        # Singleton should remain as singleton
        singletons_out = [g for g in result if len(g.tracklets) == 1]
        assert len(singletons_out) == 1


# ---------------------------------------------------------------------------
# Tests: same-camera overlap constraint
# ---------------------------------------------------------------------------


class TestSameCameraOverlap:
    """Test that same-camera detected-frame overlap blocks assignment."""

    def test_same_camera_detected_overlap_blocks_assignment(self) -> None:
        """Singleton from same camera as a group member with overlapping detected frames is blocked."""
        frames = tuple(range(20))
        n = len(frames)
        luts = _converging_luts()
        # Group has cam_a, cam_b, cam_c. Singleton also from cam_a with overlapping frames.
        group = _make_multi_group(frames, luts, fish_id=0)
        # cam_a singleton — same camera, overlapping detected frames
        singleton_tracklet = _make_tracklet(
            "cam_a",  # same camera as group member!
            999,
            frames,  # overlapping detected frames
            keypoints=_make_keypoints(n),
            keypoint_conf=_make_keypoint_conf(n),
        )
        singleton = TrackletGroup(
            fish_id=1, tracklets=(singleton_tracklet,), confidence=0.1
        )
        config = MockRecoveryConfig(
            recovery_residual_threshold=5.0,  # would assign if overlap not checked
            recovery_min_shared_frames=1,
        )
        result = recover_singletons([group, singleton], luts, config)

        # Singleton should NOT be assigned due to same-camera overlap
        singletons_out = [g for g in result if len(g.tracklets) == 1]
        assert len(singletons_out) == 1
        assert singletons_out[0].tracklets[0].camera_id == "cam_a"

    def test_coasted_frame_overlap_does_not_block_assignment(self) -> None:
        """Singleton overlap with coasted frames only does NOT block assignment."""
        frames = tuple(range(20))
        n = len(frames)
        luts = _converging_luts()
        luts["cam_d"] = MockForwardLUT(
            "cam_d",
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 1.0, 0.5]),
        )

        # Group has cam_a tracklet where all frames are COASTED
        coasted_status = tuple("coasted" for _ in frames)
        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet(
                    "cam_a",
                    100,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                    frame_status=coasted_status,
                ),
                _make_tracklet(
                    "cam_b",
                    200,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
                _make_tracklet(
                    "cam_c",
                    300,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
            ),
            confidence=0.9,
        )
        # Singleton from cam_a — same camera, but group's cam_a frames are coasted only
        singleton_tracklet = _make_tracklet(
            "cam_a",  # same camera, but group's frames are coasted
            999,
            frames,
            keypoints=_make_keypoints(n),
            keypoint_conf=_make_keypoint_conf(n),
        )
        singleton = TrackletGroup(
            fish_id=1, tracklets=(singleton_tracklet,), confidence=0.1
        )
        config = MockRecoveryConfig(
            recovery_residual_threshold=5.0,
            recovery_min_shared_frames=1,
        )
        result = recover_singletons([group, singleton], luts, config)

        # Coasted overlap does NOT block — singleton should be assigned
        multi_groups = [g for g in result if len(g.tracklets) > 1]
        assert len(multi_groups) == 1
        assert len(multi_groups[0].tracklets) == 4


# ---------------------------------------------------------------------------
# Tests: greedy assignment ordering
# ---------------------------------------------------------------------------


class TestGreedyAssignment:
    """Test that greedy assignment assigns best-scoring singleton first."""

    def test_greedy_best_first_when_competing_singletons(self) -> None:
        """When two singletons compete for same group, best-scoring wins."""
        frames = tuple(range(20))
        n = len(frames)
        luts = _converging_luts()

        # Converging singleton (good)
        luts["cam_d"] = MockForwardLUT(
            "cam_d",
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 1.0, 0.5]),
        )
        # Divergent singleton (bad)
        luts["cam_e"] = _divergent_lut("cam_e")

        group = _make_multi_group(frames, luts, fish_id=0)

        good_singleton = TrackletGroup(
            fish_id=1,
            tracklets=(
                _make_tracklet(
                    "cam_d",
                    991,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
            ),
            confidence=0.1,
        )
        bad_singleton = TrackletGroup(
            fish_id=2,
            tracklets=(
                _make_tracklet(
                    "cam_e",
                    992,
                    frames,
                    keypoints=_make_keypoints(n),
                    keypoint_conf=_make_keypoint_conf(n),
                ),
            ),
            confidence=0.1,
        )
        config = MockRecoveryConfig(
            recovery_residual_threshold=1.0,  # good singleton passes, bad doesn't
            recovery_min_shared_frames=1,
        )
        result = recover_singletons(
            [group, good_singleton, bad_singleton], luts, config
        )

        # Good singleton should be assigned; bad stays as singleton
        multi_groups = [g for g in result if len(g.tracklets) > 1]
        singletons_out = [g for g in result if len(g.tracklets) == 1]
        assert len(multi_groups) == 1
        assert len(multi_groups[0].tracklets) == 4
        # Bad singleton (divergent) stays unassigned
        assert len(singletons_out) == 1
        assert singletons_out[0].tracklets[0].camera_id == "cam_e"


# ---------------------------------------------------------------------------
# Tests: split-assign
# ---------------------------------------------------------------------------


class TestSplitAssign:
    """Test that split-assign recovers temporally-split singletons."""

    def test_split_assign_finds_correct_split_point(self) -> None:
        """Singleton with bad whole score but good per-segment scores is split.

        Geometry:
        - Group A: cam_a, cam_b — frames 0-29 — converge near point_a = (0, 0, ~0.5)
        - Group B: cam_e, cam_f — frames 0-29 — converge near point_b = (0, 0, ~1.5)
          (different 3D location)
        - Singleton cam_d: frames 0-29.
          * First half (frames 0-14): rays pass near point_a (low residual vs group_a,
            high residual vs group_b)
          * Second half (frames 15-29): rays pass near point_b (low residual vs group_b,
            high residual vs group_a)
        - Mean over all 30 frames: both groups > threshold
        - "Before" segment (0-14): group_a passes
        - "After" segment (15-29): group_b passes
        """
        n_frames = 30
        all_frames = tuple(range(n_frames))
        half = n_frames // 2  # 15
        n_g = n_frames

        # Group A converges near (0, 0, ~0.5) — same as the default converging luts
        # Group B converges near (0, 0, ~1.5) — shifted by using different origins

        # Group A cameras (cam_a, cam_b): converge near z=0.5
        luts: dict[str, MockForwardLUT] = {
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
        }

        # Group B cameras (cam_e, cam_f): start at z=3 and converge near z=1.5
        # origin at z=3, dir pointing to (0,0,1.5), i.e. dir=(-1,0,-1.5)/norm from (1,0,3)
        luts["cam_e"] = MockForwardLUT(
            "cam_e",
            np.array([1.0, 0.0, 3.0]),
            np.array([-1.0, 0.0, -1.5]),
        )
        luts["cam_f"] = MockForwardLUT(
            "cam_f",
            np.array([-1.0, 0.0, 3.0]),
            np.array([1.0, 0.0, -1.5]),
        )

        group_a = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet(
                    "cam_a",
                    100,
                    all_frames,
                    keypoints=_make_keypoints(n_g),
                    keypoint_conf=_make_keypoint_conf(n_g),
                ),
                _make_tracklet(
                    "cam_b",
                    200,
                    all_frames,
                    keypoints=_make_keypoints(n_g),
                    keypoint_conf=_make_keypoint_conf(n_g),
                ),
            ),
            confidence=0.9,
        )

        group_b = TrackletGroup(
            fish_id=1,
            tracklets=(
                _make_tracklet(
                    "cam_e",
                    101,
                    all_frames,
                    keypoints=_make_keypoints(n_g),
                    keypoint_conf=_make_keypoint_conf(n_g),
                ),
                _make_tracklet(
                    "cam_f",
                    201,
                    all_frames,
                    keypoints=_make_keypoints(n_g),
                    keypoint_conf=_make_keypoint_conf(n_g),
                ),
            ),
            confidence=0.9,
        )

        # Singleton cam_d:
        # - frames 0-14: pixel (0,0) -> ray passing near point_a (matches group_a)
        # - frames 15-29: pixel (500,500) -> ray passing near point_b (matches group_b)
        #
        # cam_a intersects cam_b near (0, 0, 0.5)
        # cam_e intersects cam_f near (0, 0, 1.5)
        #
        # cam_d pixel_fn:
        #   px<100: origin=(0, -1, 0), dir=(0, 1, 0.5) -> passes near (0,0,0.5) = point_a
        #   px>=100: origin=(0, -1, 3.0), dir=(0, 1, -1.5) -> passes near (0,0,1.5) = point_b

        def cam_d_pixel_fn(px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Return ray through point_a for px<100, through point_b for px>=100."""
            if float(px[0]) < 100:
                # Ray passing near (0, 0, 0.5)
                return np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.5])
            else:
                # Ray passing near (0, 0, 1.5)
                return np.array([0.0, -1.0, 3.0]), np.array([0.0, 1.0, -1.5])

        luts["cam_d"] = MockForwardLUT(
            "cam_d",
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 1.0, 0.5]),
            pixel_fn=cam_d_pixel_fn,
        )

        # Keypoints: first half at (0,0) -> ray near point_a; second half at (500,500) -> ray near point_b
        kpts_singleton = np.zeros((n_frames, 6, 2), dtype=np.float32)
        kpts_singleton[half:, :, :] = 500.0

        singleton_tracklet = _make_tracklet(
            "cam_d",
            999,
            all_frames,
            keypoints=kpts_singleton,
            keypoint_conf=_make_keypoint_conf(n_frames),
        )
        singleton = TrackletGroup(
            fish_id=2, tracklets=(singleton_tracklet,), confidence=0.1
        )

        # Threshold = 0.5m:
        # - "before" segment vs group_a: rays near point_a, residual ~0 -> passes
        # - "after" segment vs group_b: rays near point_b, residual ~0 -> passes
        # - whole singleton vs group_a: mean of (near 0) + (far from point_a) > 0.5 -> fails
        # - whole singleton vs group_b: mean of (far from point_b) + (near 0) > 0.5 -> fails
        # => whole-assign fails, split-assign succeeds
        config = MockRecoveryConfig(
            recovery_residual_threshold=0.5,
            recovery_min_shared_frames=1,
            recovery_min_segment_length=3,
        )
        result = recover_singletons([group_a, group_b, singleton], luts, config)

        # At least one group should have gained a segment (4 tracklets)
        grew = any(len(g.tracklets) > 2 for g in result)
        assert grew, "Expected at least one group to gain a segment from split-assign"
        # No singletons remaining if both segments matched
        singletons_out = [g for g in result if len(g.tracklets) == 1]
        assert len(singletons_out) == 0, "Expected both segments to be absorbed"

    def test_split_assign_requires_both_segments_to_match_different_groups(
        self,
    ) -> None:
        """If only one segment can match, the singleton remains unchanged."""
        frames = tuple(range(0, 30))
        n = len(frames)
        luts = _converging_luts()
        luts["cam_d"] = _divergent_lut("cam_d")  # diverges from the single group

        group = _make_multi_group(tuple(range(0, 30)), luts, fish_id=0)

        singleton_tracklet = _make_tracklet(
            "cam_d",
            999,
            frames,
            keypoints=_make_keypoints(n),
            keypoint_conf=_make_keypoint_conf(n),
        )
        singleton = TrackletGroup(
            fish_id=1, tracklets=(singleton_tracklet,), confidence=0.1
        )
        config = MockRecoveryConfig(
            recovery_residual_threshold=0.001,  # so tight nothing matches
            recovery_min_shared_frames=1,
            recovery_min_segment_length=3,
        )
        result = recover_singletons([group, singleton], luts, config)

        # Singleton remains unchanged
        singletons_out = [g for g in result if len(g.tracklets) == 1]
        assert len(singletons_out) == 1

    def test_split_assign_skips_short_singletons(self) -> None:
        """Short singletons below 2*min_segment_length skip split sweep.

        Uses a divergent singleton so that whole-assign also fails
        (residual > threshold). The singleton has only 5 frames but
        min_segment_length=3 requires 6+ for split sweep. It should
        remain as a singleton — neither whole-assigned nor split.
        """
        frames = tuple(range(0, 5))  # 5 frames, min_segment_length=3 -> skip (5 < 6)
        n = len(frames)
        luts = _converging_luts()
        # Divergent lut: singleton rays miss the group's convergence point
        luts["cam_d"] = _divergent_lut("cam_d")

        group = _make_multi_group(tuple(range(0, 5)), luts, fish_id=0)
        singleton_tracklet = _make_tracklet(
            "cam_d",
            999,
            frames,
            keypoints=_make_keypoints(n),
            keypoint_conf=_make_keypoint_conf(n),
        )
        singleton = TrackletGroup(
            fish_id=1, tracklets=(singleton_tracklet,), confidence=0.1
        )
        # Tight threshold so whole-assign fails (divergent rays give large residual)
        # and split-assign is skipped because len(frames)=5 < 2*min_segment_length=6
        config = MockRecoveryConfig(
            recovery_residual_threshold=1.0,
            recovery_min_shared_frames=1,
            recovery_min_segment_length=3,  # 5 < 2*3=6, so skip split
        )
        result = recover_singletons([group, singleton], luts, config)

        # Singleton can't be split (too short) and diverges too much for whole-assign
        singletons_out = [g for g in result if len(g.tracklets) == 1]
        assert len(singletons_out) == 1


# ---------------------------------------------------------------------------
# Tests: keypoints=None fallback
# ---------------------------------------------------------------------------


class TestKeypointsNoneFallback:
    """Test that singletons with keypoints=None fall back gracefully."""

    def test_no_keypoints_does_not_crash(self) -> None:
        """Recovery runs without crash when singleton keypoints are None."""
        frames = tuple(range(20))
        luts = _converging_luts()
        luts["cam_d"] = MockForwardLUT(
            "cam_d",
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 1.0, 0.5]),
        )
        group = _make_multi_group(frames, luts, fish_id=0)
        # Singleton with keypoints=None
        singleton_tracklet = _make_tracklet(
            "cam_d",
            999,
            frames,
            keypoints=None,  # no keypoints
            keypoint_conf=None,
        )
        singleton = TrackletGroup(
            fish_id=1, tracklets=(singleton_tracklet,), confidence=0.1
        )
        config = MockRecoveryConfig(
            recovery_residual_threshold=5.0,
            recovery_min_shared_frames=1,
        )
        # Should not raise
        result = recover_singletons([group, singleton], luts, config)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: fish_id assignment for remaining singletons
# ---------------------------------------------------------------------------


class TestFishIdAssignment:
    """Test that remaining singletons get unique fish_ids above existing max."""

    def test_unassigned_singleton_gets_unique_fish_id(self) -> None:
        """Singletons that remain unassigned get unique fish_ids above max."""
        frames = tuple(range(0, 10))
        group_frames = tuple(range(100, 120))  # non-overlapping frames
        n = len(frames)
        luts = _converging_luts()
        luts["cam_d"] = MockForwardLUT(
            "cam_d",
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 1.0, 0.5]),
        )

        group = _make_multi_group(group_frames, luts, fish_id=5)
        singleton_tracklet = _make_tracklet(
            "cam_d",
            999,
            frames,  # no shared frames with group
            keypoints=_make_keypoints(n),
            keypoint_conf=_make_keypoint_conf(n),
        )
        singleton = TrackletGroup(
            fish_id=-1, tracklets=(singleton_tracklet,), confidence=0.1
        )
        config = MockRecoveryConfig(recovery_min_shared_frames=3)
        result = recover_singletons([group, singleton], luts, config)

        # Check all fish_ids are unique and >= 0
        all_ids = [g.fish_id for g in result]
        assert len(all_ids) == len(set(all_ids)), "Duplicate fish_ids found"
        singletons_out = [g for g in result if len(g.tracklets) == 1]
        for sg in singletons_out:
            assert sg.fish_id >= 0


# ---------------------------------------------------------------------------
# Tests: no singletons — early return
# ---------------------------------------------------------------------------


class TestNoSingletons:
    """Test early return when there are no singletons."""

    def test_no_singletons_returns_groups_unchanged(self) -> None:
        """When all groups are multi-tracklet, return unchanged."""
        frames = tuple(range(20))
        luts = _converging_luts()
        group = _make_multi_group(frames, luts, fish_id=0)
        config = MockRecoveryConfig()
        result = recover_singletons([group], luts, config)

        assert len(result) == 1
        assert result[0] is group
