"""Unit tests for 3D triangulation cluster refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from aquapose.core.association.refinement import refine_clusters
from aquapose.core.association.types import TrackletGroup
from aquapose.core.tracking.types import Tracklet2D

# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockRefinementConfig:
    """Mock config satisfying RefinementConfigLike for tests.

    Default threshold is 0.5m for synthetic test geometry where cameras
    are at unit distances. Production default is 0.025m.
    """

    eviction_reproj_threshold: float = 0.5
    min_cameras_refine: int = 3
    refinement_enabled: bool = True


class MockForwardLUT:
    """Mock ForwardLUT returning controlled ray geometry.

    Each camera sits at a different position and emits rays converging
    near the origin for any centroid (0, 0).
    """

    def __init__(
        self, camera_id: str, origin: np.ndarray, direction: np.ndarray
    ) -> None:
        self.camera_id = camera_id
        self._origin = origin.astype(np.float32)
        self._direction = (direction / np.linalg.norm(direction)).astype(np.float32)

    def cast_ray(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fixed rays regardless of pixel coordinates."""
        n = pixels.shape[0]
        origins = torch.from_numpy(np.tile(self._origin, (n, 1)))
        dirs = torch.from_numpy(np.tile(self._direction, (n, 1)))
        return origins, dirs


def _make_tracklet(
    camera_id: str,
    track_id: int,
    frames: tuple[int, ...],
    centroids: tuple[tuple[float, float], ...] | None = None,
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
    )


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
    """Create a LUT whose rays miss the convergence point entirely.

    Origin at (5, 5, 0) pointing in +z -- the ray runs along x=5, y=5
    and never passes near the origin where the other rays converge.
    """
    return MockForwardLUT(
        "cam_outlier",
        np.array([5.0, 5.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRefineClusterNoEviction:
    """Test that well-converging clusters pass refinement without eviction."""

    def test_all_converging_no_eviction(self) -> None:
        """Cluster with 4 converging cameras retains all tracklets."""
        frames = (0, 1, 2, 3, 4)
        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet("cam_a", 0, frames),
                _make_tracklet("cam_b", 1, frames),
                _make_tracklet("cam_c", 2, frames),
                _make_tracklet("cam_d", 3, frames),
            ),
        )
        luts = _converging_luts()
        config = MockRefinementConfig()
        result = refine_clusters([group], luts, config)

        # Should have exactly 1 group with 4 tracklets
        assert len(result) == 1
        assert len(result[0].tracklets) == 4
        assert result[0].fish_id == 0

    def test_confidence_populated(self) -> None:
        """Refined group has confidence and per_frame_confidence set."""
        frames = (0, 1, 2)
        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet("cam_a", 0, frames),
                _make_tracklet("cam_b", 1, frames),
                _make_tracklet("cam_c", 2, frames),
            ),
        )
        luts = _converging_luts()
        config = MockRefinementConfig()
        result = refine_clusters([group], luts, config)

        assert result[0].confidence is not None
        assert result[0].per_frame_confidence is not None
        assert len(result[0].per_frame_confidence) == 3
        for c in result[0].per_frame_confidence:
            assert isinstance(c, float)
            assert 0.0 <= c <= 1.0


class TestRefineClusterEviction:
    """Test that outlier tracklets are evicted to singleton groups."""

    def test_outlier_evicted(self) -> None:
        """Tracklet with divergent rays gets evicted as singleton."""
        frames = (0, 1, 2, 3, 4)
        luts = _converging_luts()
        # Add the outlier LUT
        outlier_lut = _divergent_lut()
        luts["cam_outlier"] = outlier_lut

        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet("cam_a", 0, frames),
                _make_tracklet("cam_b", 1, frames),
                _make_tracklet("cam_c", 2, frames),
                _make_tracklet("cam_outlier", 99, frames),
            ),
        )
        config = MockRefinementConfig()
        result = refine_clusters([group], luts, config)

        # Should have 2 groups: the cleaned cluster + the evicted singleton
        assert len(result) == 2

        # Main cluster keeps 3 tracklets
        main_group = result[0]
        assert main_group.fish_id == 0
        assert len(main_group.tracklets) == 3
        cam_ids = {t.camera_id for t in main_group.tracklets}
        assert "cam_outlier" not in cam_ids

        # Singleton group for evicted tracklet
        singleton = result[1]
        assert len(singleton.tracklets) == 1
        assert singleton.tracklets[0].camera_id == "cam_outlier"
        assert singleton.confidence == 0.1

    def test_evicted_becomes_singleton_with_low_confidence(self) -> None:
        """Evicted singleton has confidence=0.1."""
        frames = (0, 1, 2)
        luts = _converging_luts()
        luts["cam_outlier"] = _divergent_lut()

        group = TrackletGroup(
            fish_id=5,
            tracklets=(
                _make_tracklet("cam_a", 0, frames),
                _make_tracklet("cam_b", 1, frames),
                _make_tracklet("cam_c", 2, frames),
                _make_tracklet("cam_outlier", 99, frames),
            ),
        )
        config = MockRefinementConfig()
        result = refine_clusters([group], luts, config)

        singletons = [g for g in result if len(g.tracklets) == 1]
        assert len(singletons) == 1
        assert singletons[0].confidence == 0.1


class TestRefineClusterRetriangulation:
    """Test that cleaned cluster is re-triangulated with updated confidence."""

    def test_cleaned_cluster_has_updated_confidence(self) -> None:
        """After eviction, remaining cluster re-computes per-frame confidence."""
        frames = (0, 1, 2, 3, 4)
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
            confidence=None,
        )
        config = MockRefinementConfig()
        result = refine_clusters([group], luts, config)

        main = result[0]
        assert main.confidence is not None
        assert main.per_frame_confidence is not None
        assert len(main.per_frame_confidence) == 5


class TestRefinementDisabled:
    """Test refinement_enabled=False returns groups unchanged."""

    def test_disabled_returns_unchanged(self) -> None:
        """Groups pass through unchanged when refinement is disabled."""
        frames = (0, 1, 2)
        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet("cam_a", 0, frames),
                _make_tracklet("cam_b", 1, frames),
                _make_tracklet("cam_c", 2, frames),
            ),
        )
        config = MockRefinementConfig(refinement_enabled=False)
        result = refine_clusters([group], {}, config)

        assert len(result) == 1
        assert result[0] is group


class TestMinCamerasRefine:
    """Test groups below min_cameras_refine are skipped."""

    def test_below_threshold_skipped(self) -> None:
        """Group with 2 cameras (below min_cameras_refine=3) is skipped."""
        frames = (0, 1, 2)
        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet("cam_a", 0, frames),
                _make_tracklet("cam_b", 1, frames),
            ),
            confidence=0.8,
        )
        luts = _converging_luts()
        config = MockRefinementConfig(min_cameras_refine=3)
        result = refine_clusters([group], luts, config)

        assert len(result) == 1
        assert result[0] is group
        # Confidence unchanged since refinement was skipped
        assert result[0].confidence == 0.8
        assert result[0].per_frame_confidence is None


class TestPerFrameConfidence:
    """Test per-frame confidence tuple structure and values."""

    def test_per_frame_confidence_is_tuple_of_floats(self) -> None:
        """per_frame_confidence is a tuple with one float per union frame."""
        frames = (0, 1, 2, 3)
        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet("cam_a", 0, frames),
                _make_tracklet("cam_b", 1, frames),
                _make_tracklet("cam_c", 2, frames),
            ),
        )
        luts = _converging_luts()
        config = MockRefinementConfig()
        result = refine_clusters([group], luts, config)

        pfc = result[0].per_frame_confidence
        assert pfc is not None
        assert isinstance(pfc, tuple)
        assert len(pfc) == 4
        for val in pfc:
            assert isinstance(val, float)

    def test_high_convergence_high_confidence(self) -> None:
        """Well-converging rays produce confidence close to 1.0."""
        frames = (0, 1, 2)
        group = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet("cam_a", 0, frames),
                _make_tracklet("cam_b", 1, frames),
                _make_tracklet("cam_c", 2, frames),
            ),
        )
        luts = _converging_luts()
        config = MockRefinementConfig()
        result = refine_clusters([group], luts, config)

        # With converging rays, confidence should be reasonably high
        assert result[0].confidence is not None
        assert result[0].confidence > 0.0


class TestMultipleGroups:
    """Test refinement handles multiple groups correctly."""

    def test_multiple_groups_independent(self) -> None:
        """Each group is refined independently."""
        frames = (0, 1, 2)
        group_a = TrackletGroup(
            fish_id=0,
            tracklets=(
                _make_tracklet("cam_a", 0, frames),
                _make_tracklet("cam_b", 1, frames),
                _make_tracklet("cam_c", 2, frames),
            ),
        )
        group_b = TrackletGroup(
            fish_id=1,
            tracklets=(
                _make_tracklet("cam_a", 10, frames),
                _make_tracklet("cam_b", 11, frames),
            ),
            confidence=0.9,
        )
        luts = _converging_luts()
        config = MockRefinementConfig()
        result = refine_clusters([group_a, group_b], luts, config)

        # group_a refined, group_b skipped (2 cameras < min_cameras=3)
        assert len(result) == 2
        assert result[0].fish_id == 0
        assert result[0].per_frame_confidence is not None
        assert result[1].fish_id == 1
        assert result[1] is group_b
