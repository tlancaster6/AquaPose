"""Unit tests for Leiden-based tracklet clustering and fragment merging."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from aquapose.core.tracking.types import Tracklet2D

leidenalg = pytest.importorskip("leidenalg")

from aquapose.core.association.clustering import (  # noqa: E402
    build_must_not_link,
    cluster_tracklets,
    merge_fragments,
)

# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockClusteringConfig:
    """Mock config satisfying ClusteringConfigLike for tests."""

    score_min: float = 0.3
    expected_fish_count: int = 2
    leiden_resolution: float = 1.0
    max_merge_gap: int = 30


def _make_tracklet(
    camera_id: str,
    track_id: int,
    frames: tuple[int, ...],
    centroid: tuple[float, float] = (100.0, 100.0),
    status: str = "detected",
    statuses: tuple[str, ...] | None = None,
) -> Tracklet2D:
    """Build a synthetic Tracklet2D with uniform or custom statuses."""
    centroids = tuple((centroid[0], centroid[1]) for _ in frames)
    bboxes = tuple((centroid[0] - 10.0, centroid[1] - 10.0, 20.0, 20.0) for _ in frames)
    frame_status = statuses if statuses is not None else tuple(status for _ in frames)
    return Tracklet2D(
        camera_id=camera_id,
        track_id=track_id,
        frames=frames,
        centroids=centroids,
        bboxes=bboxes,
        frame_status=frame_status,
    )


TrackletKey = tuple[str, int]


# ---------------------------------------------------------------------------
# build_must_not_link tests
# ---------------------------------------------------------------------------


class TestBuildMustNotLink:
    """Tests for build_must_not_link constraint generation."""

    def test_detected_overlap_creates_constraint(self) -> None:
        """Two same-camera tracklets with detected overlap -> must-not-link."""
        ta = _make_tracklet("cam_a", 1, (0, 1, 2, 3, 4))
        tb = _make_tracklet("cam_a", 2, (3, 4, 5, 6, 7))
        tracks_2d = {"cam_a": [ta, tb]}

        constraints = build_must_not_link(tracks_2d)

        expected_pair = frozenset({("cam_a", 1), ("cam_a", 2)})
        assert expected_pair in constraints

    def test_coasted_overlap_not_constraint(self) -> None:
        """Same-camera overlap only in coasted frames -> no constraint."""
        ta = _make_tracklet(
            "cam_a",
            1,
            (0, 1, 2, 3, 4),
            statuses=("detected", "detected", "detected", "coasted", "coasted"),
        )
        tb = _make_tracklet(
            "cam_a",
            2,
            (3, 4, 5, 6, 7),
            statuses=("coasted", "coasted", "detected", "detected", "detected"),
        )
        tracks_2d = {"cam_a": [ta, tb]}

        constraints = build_must_not_link(tracks_2d)

        # Overlap at frames 3,4: ta is coasted, tb is coasted -> no detected overlap
        assert len(constraints) == 0

    def test_different_cameras_no_constraint(self) -> None:
        """Tracklets from different cameras -> never a constraint."""
        ta = _make_tracklet("cam_a", 1, (0, 1, 2, 3, 4))
        tb = _make_tracklet("cam_b", 1, (0, 1, 2, 3, 4))
        tracks_2d = {"cam_a": [ta], "cam_b": [tb]}

        constraints = build_must_not_link(tracks_2d)

        assert len(constraints) == 0


# ---------------------------------------------------------------------------
# cluster_tracklets tests
# ---------------------------------------------------------------------------


class TestClusterTracklets:
    """Tests for Leiden-based tracklet clustering."""

    def test_two_fish_clustering(self) -> None:
        """4 tracklets from 2 cameras grouped into 2 fish clusters."""
        # Fish 1: cam_a-1, cam_b-1 (high score)
        # Fish 2: cam_a-2, cam_b-2 (high score)
        # Cross-fish: zero score
        t_a1 = _make_tracklet("cam_a", 1, tuple(range(20)))
        t_a2 = _make_tracklet("cam_a", 2, tuple(range(20)))
        t_b1 = _make_tracklet("cam_b", 1, tuple(range(20)))
        t_b2 = _make_tracklet("cam_b", 2, tuple(range(20)))

        tracks_2d = {"cam_a": [t_a1, t_a2], "cam_b": [t_b1, t_b2]}

        scores: dict[tuple[TrackletKey, TrackletKey], float] = {
            (("cam_a", 1), ("cam_b", 1)): 0.9,  # Fish 1
            (("cam_a", 2), ("cam_b", 2)): 0.85,  # Fish 2
        }

        must_not_link: set[frozenset[TrackletKey]] = set()
        config = MockClusteringConfig(expected_fish_count=2)

        groups = cluster_tracklets(scores, tracks_2d, must_not_link, config)

        assert len(groups) == 2

        # Each group should have 2 tracklets (one per camera)
        for g in groups:
            if len(g.tracklets) == 2:
                cams = {t.camera_id for t in g.tracklets}
                assert cams == {"cam_a", "cam_b"}

    def test_must_not_link_splits(self) -> None:
        """Conflicting must-not-link pair causes eviction to singleton."""
        t_a1 = _make_tracklet("cam_a", 1, tuple(range(20)))
        t_a2 = _make_tracklet("cam_a", 2, tuple(range(20)))
        t_b1 = _make_tracklet("cam_b", 1, tuple(range(20)))

        tracks_2d = {"cam_a": [t_a1, t_a2], "cam_b": [t_b1]}

        # All three have high scores — Leiden would group them together
        scores: dict[tuple[TrackletKey, TrackletKey], float] = {
            (("cam_a", 1), ("cam_b", 1)): 0.9,
            (("cam_a", 2), ("cam_b", 1)): 0.8,
        }

        # But cam_a-1 and cam_a-2 are must-not-link
        must_not_link = {frozenset({("cam_a", 1), ("cam_a", 2)})}
        config = MockClusteringConfig(expected_fish_count=2)

        groups = cluster_tracklets(scores, tracks_2d, must_not_link, config)

        # Should have at least 2 groups (the conflicting one evicted)
        assert len(groups) >= 2

        # No group should contain both cam_a-1 and cam_a-2
        for g in groups:
            cam_a_ids = [t.track_id for t in g.tracklets if t.camera_id == "cam_a"]
            assert not (1 in cam_a_ids and 2 in cam_a_ids), (
                "Must-not-link pair should not be in same group"
            )

    def test_singleton_tracklet(self) -> None:
        """Tracklet with no scored edges becomes a singleton with confidence 0."""
        t_a1 = _make_tracklet("cam_a", 1, tuple(range(20)))
        t_b1 = _make_tracklet("cam_b", 1, tuple(range(20)))
        t_c1 = _make_tracklet("cam_c", 99, tuple(range(20)))  # Isolated

        tracks_2d = {"cam_a": [t_a1], "cam_b": [t_b1], "cam_c": [t_c1]}

        scores: dict[tuple[TrackletKey, TrackletKey], float] = {
            (("cam_a", 1), ("cam_b", 1)): 0.9,
        }

        must_not_link: set[frozenset[TrackletKey]] = set()
        config = MockClusteringConfig(expected_fish_count=2)

        groups = cluster_tracklets(scores, tracks_2d, must_not_link, config)

        # cam_c-99 should be a singleton
        singletons = [g for g in groups if len(g.tracklets) == 1]
        singleton_keys = [
            (g.tracklets[0].camera_id, g.tracklets[0].track_id) for g in singletons
        ]
        assert ("cam_c", 99) in singleton_keys

        # Singleton confidence should be 0.0
        for g in singletons:
            if g.tracklets[0].camera_id == "cam_c":
                assert g.confidence == 0.0

    def test_warns_on_wrong_count(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning emitted when cluster count != expected_fish_count."""
        t_a1 = _make_tracklet("cam_a", 1, tuple(range(20)))
        t_a2 = _make_tracklet("cam_a", 2, tuple(range(20)))
        t_b1 = _make_tracklet("cam_b", 1, tuple(range(20)))

        tracks_2d = {"cam_a": [t_a1, t_a2], "cam_b": [t_b1]}

        scores: dict[tuple[TrackletKey, TrackletKey], float] = {
            (("cam_a", 1), ("cam_b", 1)): 0.9,
        }

        must_not_link: set[frozenset[TrackletKey]] = set()
        config = MockClusteringConfig(expected_fish_count=5)  # Wrong count

        with caplog.at_level(logging.WARNING):
            groups = cluster_tracklets(scores, tracks_2d, must_not_link, config)

        assert any("Cluster count" in msg for msg in caplog.messages)
        assert len(groups) != 5


# ---------------------------------------------------------------------------
# merge_fragments tests
# ---------------------------------------------------------------------------


class TestMergeFragments:
    """Tests for same-camera fragment merging."""

    def test_non_overlapping_merge(self) -> None:
        """Two non-overlapping same-camera tracklets merge with interpolation."""
        from aquapose.core.association.types import TrackletGroup

        t1 = _make_tracklet("cam_a", 1, (0, 1, 2), centroid=(100.0, 100.0))
        t2 = _make_tracklet("cam_a", 2, (5, 6, 7), centroid=(200.0, 200.0))

        group = TrackletGroup(fish_id=0, tracklets=(t1, t2), confidence=0.5)
        config = MockClusteringConfig(max_merge_gap=30)

        result = merge_fragments([group], config)

        assert len(result) == 1
        merged_group = result[0]
        # Should have 1 tracklet (merged) instead of 2
        assert len(merged_group.tracklets) == 1

        merged = merged_group.tracklets[0]
        # Should contain frames 0-7 (with 3,4 interpolated)
        assert "interpolated" in merged.frame_status

    def test_coasted_overlap_merge(self) -> None:
        """Two tracklets with coasted overlap: trim and merge."""
        from aquapose.core.association.types import TrackletGroup

        t1 = _make_tracklet(
            "cam_a",
            1,
            (0, 1, 2, 3, 4),
            centroid=(100.0, 100.0),
            statuses=("detected", "detected", "detected", "coasted", "coasted"),
        )
        t2 = _make_tracklet(
            "cam_a",
            2,
            (3, 4, 5, 6, 7),
            centroid=(200.0, 200.0),
            statuses=("coasted", "coasted", "detected", "detected", "detected"),
        )

        group = TrackletGroup(fish_id=0, tracklets=(t1, t2), confidence=0.5)
        config = MockClusteringConfig(max_merge_gap=30)

        result = merge_fragments([group], config)

        assert len(result) == 1
        # Should merge since no detection-backed overlap
        merged_group = result[0]
        assert len(merged_group.tracklets) == 1

        merged = merged_group.tracklets[0]
        # Should have detected frames from both
        detected_frames = [
            f
            for f, s in zip(merged.frames, merged.frame_status, strict=False)
            if s == "detected"
        ]
        assert 0 in detected_frames
        assert 5 in detected_frames

    def test_exceeds_max_gap(self) -> None:
        """Fragments separated by more than max_merge_gap are NOT merged."""
        from aquapose.core.association.types import TrackletGroup

        t1 = _make_tracklet("cam_a", 1, (0, 1, 2), centroid=(100.0, 100.0))
        t2 = _make_tracklet("cam_a", 2, (100, 101, 102), centroid=(200.0, 200.0))

        group = TrackletGroup(fish_id=0, tracklets=(t1, t2), confidence=0.5)
        config = MockClusteringConfig(max_merge_gap=5)

        result = merge_fragments([group], config)

        assert len(result) == 1
        # Both tracklets should remain separate
        assert len(result[0].tracklets) == 2

    def test_cross_camera_not_merged(self) -> None:
        """Tracklets from different cameras are never merged."""
        from aquapose.core.association.types import TrackletGroup

        t1 = _make_tracklet("cam_a", 1, (0, 1, 2), centroid=(100.0, 100.0))
        t2 = _make_tracklet("cam_b", 1, (0, 1, 2), centroid=(200.0, 200.0))

        group = TrackletGroup(fish_id=0, tracklets=(t1, t2), confidence=0.5)
        config = MockClusteringConfig(max_merge_gap=30)

        result = merge_fragments([group], config)

        assert len(result) == 1
        # 2 tracklets remain (different cameras)
        assert len(result[0].tracklets) == 2


# ---------------------------------------------------------------------------
# AssociationStage integration test
# ---------------------------------------------------------------------------


class TestAssociationStage:
    """Integration-level test for AssociationStage."""

    def test_produces_groups_graceful_degradation(self) -> None:
        """AssociationStage produces empty groups when LUTs are not available."""
        from aquapose.core.association.stage import AssociationStage
        from aquapose.core.context import PipelineContext
        from aquapose.engine.config import AssociationConfig, LutConfig, PipelineConfig

        config = PipelineConfig(
            calibration_path="",  # Empty path -> no LUTs
            association=AssociationConfig(),
            lut=LutConfig(),
        )

        stage = AssociationStage(config)
        ctx = PipelineContext()
        ctx.tracks_2d = {"cam_a": []}
        ctx.detections = []

        result = stage.run(ctx)

        assert result.tracklet_groups is not None
        assert result.tracklet_groups == []
