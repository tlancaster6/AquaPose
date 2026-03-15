"""Unit tests for Leiden-based tracklet clustering."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from aquapose.core.tracking.types import Tracklet2D

leidenalg = pytest.importorskip("leidenalg")

from aquapose.core.association.clustering import (  # noqa: E402
    _merge_disjoint_clusters,
    build_must_not_link,
    cluster_tracklets,
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

    def test_coasted_overlap_creates_constraint(self) -> None:
        """Same-camera overlap in coasted frames -> constraint.

        The tracker had both tracks alive simultaneously, so they are
        definitively different fish regardless of frame_status.
        """
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

        assert len(constraints) == 1

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

        assert any("cluster count" in msg.lower() for msg in caplog.messages)
        assert len(groups) != 5


# ---------------------------------------------------------------------------
# Disjoint-camera merge tests
# ---------------------------------------------------------------------------


class TestDisjointCameraMerge:
    """Tests for _merge_disjoint_clusters and integration into cluster_tracklets."""

    def _make_scores(
        self,
        pairs: list[tuple[TrackletKey, TrackletKey, float]],
    ) -> dict[tuple[TrackletKey, TrackletKey], float]:
        """Build a scores dict from (key_a, key_b, score) triples."""
        return {(ka, kb): s for ka, kb, s in pairs}

    def test_disjoint_camera_merge_basic(self) -> None:
        """Two clusters with disjoint cameras and good cross-score merge into one."""
        # Cluster 0: nodes 0,1,2  (cam_a, cam_b, cam_c)
        # Cluster 1: nodes 3,4,5  (cam_d, cam_e, cam_f)
        sub_key_list: list[TrackletKey] = [
            ("cam_a", 1),
            ("cam_b", 1),
            ("cam_c", 1),
            ("cam_d", 1),
            ("cam_e", 1),
            ("cam_f", 1),
        ]
        clusters = {0: [0, 1, 2], 1: [3, 4, 5]}

        # Build cross-cluster scores >= score_min
        scores: dict[tuple[TrackletKey, TrackletKey], float] = {}
        for i in range(3):
            for j in range(3, 6):
                ka = sub_key_list[i]
                kb = sub_key_list[j]
                scores[(ka, kb)] = 0.5

        must_not_link: set[frozenset[TrackletKey]] = set()
        config = MockClusteringConfig(score_min=0.3)

        result = _merge_disjoint_clusters(
            clusters, sub_key_list, scores, must_not_link, config
        )

        assert len(result) == 1
        merged = next(iter(result.values()))
        assert set(merged) == {0, 1, 2, 3, 4, 5}

    def test_disjoint_camera_merge_blocked_by_overlap(self) -> None:
        """Two clusters sharing a camera are NOT merged."""
        # Both clusters have cam_a -> not disjoint
        sub_key_list: list[TrackletKey] = [
            ("cam_a", 1),
            ("cam_b", 1),
            ("cam_a", 2),
            ("cam_c", 1),
        ]
        clusters = {0: [0, 1], 1: [2, 3]}

        scores: dict[tuple[TrackletKey, TrackletKey], float] = {
            (("cam_a", 1), ("cam_a", 2)): 0.8,
            (("cam_a", 1), ("cam_c", 1)): 0.8,
            (("cam_b", 1), ("cam_a", 2)): 0.8,
            (("cam_b", 1), ("cam_c", 1)): 0.8,
        }

        must_not_link: set[frozenset[TrackletKey]] = set()
        config = MockClusteringConfig(score_min=0.3)

        result = _merge_disjoint_clusters(
            clusters, sub_key_list, scores, must_not_link, config
        )

        # Should remain 2 clusters — cam_a appears in both
        assert len(result) == 2

    def test_disjoint_camera_merge_blocked_by_low_score(self) -> None:
        """Disjoint cameras but mean cross-score < score_min prevents merge."""
        sub_key_list: list[TrackletKey] = [
            ("cam_a", 1),
            ("cam_b", 1),
            ("cam_c", 1),
            ("cam_d", 1),
        ]
        clusters = {0: [0, 1], 1: [2, 3]}

        # Cross-cluster scores below score_min
        scores: dict[tuple[TrackletKey, TrackletKey], float] = {
            (("cam_a", 1), ("cam_c", 1)): 0.1,
            (("cam_a", 1), ("cam_d", 1)): 0.1,
            (("cam_b", 1), ("cam_c", 1)): 0.1,
            (("cam_b", 1), ("cam_d", 1)): 0.1,
        }

        must_not_link: set[frozenset[TrackletKey]] = set()
        config = MockClusteringConfig(score_min=0.3)

        result = _merge_disjoint_clusters(
            clusters, sub_key_list, scores, must_not_link, config
        )

        assert len(result) == 2

    def test_disjoint_camera_merge_blocked_by_mnl(self) -> None:
        """Disjoint cameras and good cross-score, but MNL violation blocks merge."""
        sub_key_list: list[TrackletKey] = [
            ("cam_a", 1),
            ("cam_b", 1),
            ("cam_c", 1),
            ("cam_d", 1),
        ]
        clusters = {0: [0, 1], 1: [2, 3]}

        scores: dict[tuple[TrackletKey, TrackletKey], float] = {
            (("cam_a", 1), ("cam_c", 1)): 0.8,
            (("cam_a", 1), ("cam_d", 1)): 0.8,
            (("cam_b", 1), ("cam_c", 1)): 0.8,
            (("cam_b", 1), ("cam_d", 1)): 0.8,
        }

        # MNL constraint between cam_a-1 and cam_c-1
        must_not_link = {frozenset({("cam_a", 1), ("cam_c", 1)})}
        config = MockClusteringConfig(score_min=0.3)

        result = _merge_disjoint_clusters(
            clusters, sub_key_list, scores, must_not_link, config
        )

        # Should remain 2 clusters due to MNL
        assert len(result) == 2

    def test_disjoint_camera_merge_iterative(self) -> None:
        """Three pairwise-disjoint clusters with good cross-scores all merge to 1."""
        sub_key_list: list[TrackletKey] = [
            ("cam_a", 1),
            ("cam_b", 1),
            ("cam_c", 1),
        ]
        clusters = {0: [0], 1: [1], 2: [2]}

        scores: dict[tuple[TrackletKey, TrackletKey], float] = {
            (("cam_a", 1), ("cam_b", 1)): 0.7,
            (("cam_a", 1), ("cam_c", 1)): 0.6,
            (("cam_b", 1), ("cam_c", 1)): 0.8,
        }

        must_not_link: set[frozenset[TrackletKey]] = set()
        config = MockClusteringConfig(score_min=0.3)

        result = _merge_disjoint_clusters(
            clusters, sub_key_list, scores, must_not_link, config
        )

        assert len(result) == 1
        merged = next(iter(result.values()))
        assert set(merged) == {0, 1, 2}

    def test_disjoint_camera_merge_integration_via_cluster_tracklets(self) -> None:
        """cluster_tracklets merges Leiden-split disjoint-camera sub-clusters."""
        # 6 cameras, 1 tracklet each
        cams = ["cam_a", "cam_b", "cam_c", "cam_d", "cam_e", "cam_f"]
        tracklets = [_make_tracklet(cam, 1, tuple(range(20))) for cam in cams]
        tracks_2d = {cam: [t] for cam, t in zip(cams, tracklets, strict=True)}

        keys: list[TrackletKey] = [(cam, 1) for cam in cams]

        # Intra-subclique: first 3 cameras pairwise 0.9, last 3 cameras pairwise 0.9
        # Cross-subclique: 0.4 (above score_min=0.3)
        scores: dict[tuple[TrackletKey, TrackletKey], float] = {}
        for i in range(3):
            for j in range(i + 1, 3):
                scores[(keys[i], keys[j])] = 0.9
        for i in range(3, 6):
            for j in range(i + 1, 6):
                scores[(keys[i], keys[j])] = 0.9
        for i in range(3):
            for j in range(3, 6):
                scores[(keys[i], keys[j])] = 0.4

        must_not_link: set[frozenset[TrackletKey]] = set()
        # High resolution to encourage over-partitioning
        config = MockClusteringConfig(
            score_min=0.3, expected_fish_count=1, leiden_resolution=5.0
        )

        groups = cluster_tracklets(scores, tracks_2d, must_not_link, config)

        # All tracklets should end up in a single group
        multi_groups = [g for g in groups if len(g.tracklets) > 1]
        assert len(multi_groups) == 1
        assert len(multi_groups[0].tracklets) == 6
        result_cams = {t.camera_id for t in multi_groups[0].tracklets}
        assert result_cams == set(cams)


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
