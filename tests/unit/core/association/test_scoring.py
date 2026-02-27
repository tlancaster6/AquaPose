"""Unit tests for pairwise cross-camera tracklet affinity scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from aquapose.core.association.scoring import (
    ray_ray_closest_point,
    score_all_pairs,
    score_tracklet_pair,
)
from aquapose.core.tracking.types import Tracklet2D

# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockAssociationConfig:
    """Mock config satisfying AssociationConfigLike for tests."""

    ray_distance_threshold: float = 0.03
    score_min: float = 0.3
    t_min: int = 3
    t_saturate: int = 100
    early_k: int = 3
    ghost_pixel_threshold: float = 50.0
    min_shared_voxels: int = 1


class MockForwardLUT:
    """Mock ForwardLUT that returns controlled ray geometry.

    Camera A at (0,0,0) looking along +z.
    Camera B at (1,0,0) looking along (-1,0,1)/sqrt(2).
    Both rays converge near (0.5, 0, 0.5) for centroid (0,0).
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


class MockDivergentForwardLUT:
    """Mock ForwardLUT that returns rays that never converge."""

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id

    def cast_ray(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return parallel rays offset in Y that will never converge closely."""
        n = pixels.shape[0]
        if self.camera_id == "cam_a":
            origins = torch.zeros(n, 3)
            dirs = torch.tensor([[0.0, 0.0, 1.0]]).expand(n, 3)
        else:
            origins = torch.tensor([[0.0, 10.0, 0.0]]).expand(n, 3)
            dirs = torch.tensor([[0.0, 0.0, 1.0]]).expand(n, 3)
        return origins, dirs


class MockInverseLUT:
    """Mock InverseLUT that returns controlled visibility.

    The midpoint of converging test rays is at (0.5, 0.0, 0.5). We set up
    the grid index so that ghost_point_lookup can find this voxel and return
    visibility data for the configured cameras.
    """

    def __init__(
        self,
        camera_ids: list[str],
        visibility: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.camera_ids = camera_ids
        self._visibility = visibility or {}
        n_cameras = len(camera_ids)
        self.voxel_resolution = 0.02
        self.grid_bounds = {
            "x_min": -1.0,
            "x_max": 1.0,
            "y_min": -1.0,
            "y_max": 1.0,
            "z_min": 0.0,
            "z_max": 1.0,
        }

        # We need a voxel that ghost_point_lookup can find for the midpoint
        # (0.5, 0.0, 0.5). Grid coords: ix=round((0.5-(-1))/0.02)=75,
        # iy=round((0-(-1))/0.02)=50, iz=round((0.5-0)/0.02)=25
        # Create a single voxel at index 0 that maps to these grid coords.
        n_voxels = 1
        self.voxel_centers = np.array([[0.5, 0.0, 0.5]], dtype=np.float32)
        self.visibility_mask = np.ones((n_voxels, n_cameras), dtype=bool)
        self._grid_to_voxel_idx: dict[tuple[int, int, int], int] = {
            (75, 50, 25): 0,
        }
        # Create projected_pixels for ghost lookups
        self.projected_pixels = np.full(
            (n_voxels, n_cameras, 2), np.nan, dtype=np.float32
        )
        for cam_idx, cam_id in enumerate(camera_ids):
            if cam_id in self._visibility:
                u, v = self._visibility[cam_id]
                self.projected_pixels[0, cam_idx, :] = [u, v]


class MockInverseLUTNonAdjacent:
    """Mock InverseLUT where only cam_a-cam_b are adjacent."""

    def __init__(self) -> None:
        self.camera_ids = ["cam_a", "cam_b", "cam_c"]
        n_voxels = 200
        n_cameras = 3
        self.visibility_mask = np.zeros((n_voxels, n_cameras), dtype=bool)
        # Only cam_a and cam_b share voxels
        self.visibility_mask[:, 0] = True  # cam_a sees all
        self.visibility_mask[:, 1] = True  # cam_b sees all
        # cam_c sees nothing
        self.visibility_mask[:, 2] = False
        self.voxel_centers = np.zeros((n_voxels, 3), dtype=np.float32)
        self.voxel_resolution = 0.02
        self.grid_bounds = {
            "x_min": -1.0,
            "x_max": 1.0,
            "y_min": -1.0,
            "y_max": 1.0,
            "z_min": 0.0,
            "z_max": 1.0,
        }
        self._grid_to_voxel_idx: dict[tuple[int, int, int], int] = {}
        self.projected_pixels = np.full(
            (n_voxels, n_cameras, 2), np.nan, dtype=np.float32
        )


def _make_tracklet(
    camera_id: str,
    track_id: int,
    frames: tuple[int, ...],
    centroid: tuple[float, float] = (100.0, 100.0),
    status: str = "detected",
) -> Tracklet2D:
    """Build a synthetic Tracklet2D with uniform centroids and bboxes."""
    centroids = tuple((centroid[0], centroid[1]) for _ in frames)
    bboxes = tuple((centroid[0] - 10.0, centroid[1] - 10.0, 20.0, 20.0) for _ in frames)
    frame_status = tuple(status for _ in frames)
    return Tracklet2D(
        camera_id=camera_id,
        track_id=track_id,
        frames=frames,
        centroids=centroids,
        bboxes=bboxes,
        frame_status=frame_status,
    )


# ---------------------------------------------------------------------------
# ray_ray_closest_point tests
# ---------------------------------------------------------------------------


class TestRayRayClosestPoint:
    """Tests for ray_ray_closest_point geometry."""

    def test_intersecting_rays(self) -> None:
        """Two rays that intersect at a known point. Distance ~0."""
        # Ray A: origin (0,0,0), direction (1,0,1)/sqrt(2)
        origin_a = np.array([0.0, 0.0, 0.0])
        dir_a = np.array([1.0, 0.0, 1.0])
        dir_a = dir_a / np.linalg.norm(dir_a)

        # Ray B: origin (1,0,0), direction (-1,0,1)/sqrt(2)
        origin_b = np.array([1.0, 0.0, 0.0])
        dir_b = np.array([-1.0, 0.0, 1.0])
        dir_b = dir_b / np.linalg.norm(dir_b)

        # Rays intersect at (0.5, 0, 0.5)
        dist, midpoint = ray_ray_closest_point(origin_a, dir_a, origin_b, dir_b)

        assert dist < 1e-6, f"Expected ~0 distance, got {dist}"
        np.testing.assert_allclose(midpoint, [0.5, 0.0, 0.5], atol=1e-6)

    def test_skew_rays(self) -> None:
        """Two skew lines with known closest distance."""
        # Ray along x-axis at y=0, z=0
        origin_a = np.array([0.0, 0.0, 0.0])
        dir_a = np.array([1.0, 0.0, 0.0])

        # Ray along z-axis at x=0, y=1
        origin_b = np.array([0.0, 1.0, 0.0])
        dir_b = np.array([0.0, 0.0, 1.0])

        dist, midpoint = ray_ray_closest_point(origin_a, dir_a, origin_b, dir_b)

        assert abs(dist - 1.0) < 1e-6, f"Expected distance 1.0, got {dist}"
        np.testing.assert_allclose(midpoint, [0.0, 0.5, 0.0], atol=1e-6)

    def test_parallel_rays(self) -> None:
        """Two parallel rays. Should not crash and return finite distance."""
        origin_a = np.array([0.0, 0.0, 0.0])
        dir_a = np.array([1.0, 0.0, 0.0])

        origin_b = np.array([0.0, 1.0, 0.0])
        dir_b = np.array([1.0, 0.0, 0.0])

        dist, midpoint = ray_ray_closest_point(origin_a, dir_a, origin_b, dir_b)

        # Parallel rays offset by 1 in y
        assert dist == pytest.approx(1.0, abs=1e-6)
        assert np.isfinite(midpoint).all()


# ---------------------------------------------------------------------------
# score_tracklet_pair tests
# ---------------------------------------------------------------------------


class TestScoreTrackletPair:
    """Tests for score_tracklet_pair function."""

    def test_perfect_match(self) -> None:
        """Rays converge within threshold on every frame -> high score."""
        frames = tuple(range(20))
        ta = _make_tracklet("cam_a", 1, frames)
        tb = _make_tracklet("cam_b", 1, frames)

        # Converging rays: A at origin along +z, B at (1,0,0) along (-1,0,1)
        lut_a = MockForwardLUT(
            "cam_a", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0])
        )
        lut_b = MockForwardLUT(
            "cam_b", np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 1.0])
        )
        forward_luts = {"cam_a": lut_a, "cam_b": lut_b}

        # Mock inverse LUT — all cameras see the midpoint, with detections nearby
        inv_lut = MockInverseLUT(
            ["cam_a", "cam_b"],
            {"cam_a": (100.0, 100.0), "cam_b": (100.0, 100.0)},
        )

        detections: list[dict[str, list[tuple[float, float]]]] = [
            {"cam_a": [(100.0, 100.0)], "cam_b": [(100.0, 100.0)]} for _ in range(20)
        ]

        config = MockAssociationConfig()
        score = score_tracklet_pair(ta, tb, forward_luts, inv_lut, detections, config)

        # Rays intersect at distance 0 < threshold, all frames inlier
        # f=1.0, ghost=0 (only scoring cameras visible), w=20/100=0.2
        assert score > 0.0, f"Expected positive score, got {score}"

    def test_no_overlap(self) -> None:
        """Two tracklets with no shared frames -> score 0."""
        ta = _make_tracklet("cam_a", 1, (0, 1, 2, 3, 4))
        tb = _make_tracklet("cam_b", 1, (10, 11, 12, 13, 14))

        lut_a = MockForwardLUT(
            "cam_a", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        )
        lut_b = MockForwardLUT(
            "cam_b", np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        )
        forward_luts = {"cam_a": lut_a, "cam_b": lut_b}

        inv_lut = MockInverseLUT(["cam_a", "cam_b"])
        config = MockAssociationConfig()

        score = score_tracklet_pair(ta, tb, forward_luts, inv_lut, [], config)
        assert score == 0.0

    def test_below_t_min(self) -> None:
        """Fewer shared frames than t_min -> score 0."""
        ta = _make_tracklet("cam_a", 1, (0, 1))  # only 2 shared frames
        tb = _make_tracklet("cam_b", 1, (0, 1))

        lut_a = MockForwardLUT(
            "cam_a", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        )
        lut_b = MockForwardLUT(
            "cam_b", np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        )
        forward_luts = {"cam_a": lut_a, "cam_b": lut_b}

        inv_lut = MockInverseLUT(["cam_a", "cam_b"])
        config = MockAssociationConfig(t_min=3)

        score = score_tracklet_pair(ta, tb, forward_luts, inv_lut, [], config)
        assert score == 0.0

    def test_early_termination(self) -> None:
        """Rays never converge -> early termination returns 0 after early_k frames."""
        frames = tuple(range(100))
        ta = _make_tracklet("cam_a", 1, frames)
        tb = _make_tracklet("cam_b", 1, frames)

        # Divergent rays: parallel offset by 10 in Y
        lut_a = MockDivergentForwardLUT("cam_a")
        lut_b = MockDivergentForwardLUT("cam_b")
        forward_luts = {"cam_a": lut_a, "cam_b": lut_b}

        inv_lut = MockInverseLUT(["cam_a", "cam_b"])
        config = MockAssociationConfig(early_k=3, t_min=3)

        score = score_tracklet_pair(ta, tb, forward_luts, inv_lut, [], config)
        assert score == 0.0


# ---------------------------------------------------------------------------
# score_all_pairs tests
# ---------------------------------------------------------------------------


class TestScoreAllPairs:
    """Tests for score_all_pairs function."""

    def test_respects_camera_adjacency(self) -> None:
        """Only cam_a-cam_b pairs are scored when cam_c is non-adjacent."""
        frames = tuple(range(20))
        tracks_2d = {
            "cam_a": [_make_tracklet("cam_a", 1, frames)],
            "cam_b": [_make_tracklet("cam_b", 1, frames)],
            "cam_c": [_make_tracklet("cam_c", 1, frames)],
        }

        lut_a = MockForwardLUT(
            "cam_a", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0])
        )
        lut_b = MockForwardLUT(
            "cam_b", np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 1.0])
        )
        lut_c = MockForwardLUT(
            "cam_c", np.array([5.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        )
        forward_luts = {"cam_a": lut_a, "cam_b": lut_b, "cam_c": lut_c}

        inv_lut = MockInverseLUTNonAdjacent()

        detections: list[dict[str, list[tuple[float, float]]]] = [
            {"cam_a": [(100.0, 100.0)], "cam_b": [(100.0, 100.0)]} for _ in range(20)
        ]

        config = MockAssociationConfig(score_min=0.0)

        scored = score_all_pairs(tracks_2d, forward_luts, inv_lut, detections, config)

        # Only cam_a-cam_b should appear
        for key_a, key_b in scored:
            cam_pair = {key_a[0], key_b[0]}
            assert "cam_c" not in cam_pair, (
                f"cam_c should not appear in scored pairs: {key_a}, {key_b}"
            )

    def test_filters_by_score_min(self) -> None:
        """Pairs below score_min should not appear in the result."""
        frames = tuple(range(20))
        tracks_2d = {
            "cam_a": [_make_tracklet("cam_a", 1, frames)],
            "cam_b": [_make_tracklet("cam_b", 1, frames)],
        }

        # Divergent rays -> score will be 0
        lut_a = MockDivergentForwardLUT("cam_a")
        lut_b = MockDivergentForwardLUT("cam_b")
        forward_luts = {"cam_a": lut_a, "cam_b": lut_b}

        inv_lut = MockInverseLUT(["cam_a", "cam_b"])

        config = MockAssociationConfig(score_min=0.3, t_min=3, early_k=3)

        scored = score_all_pairs(tracks_2d, forward_luts, inv_lut, [], config)

        assert len(scored) == 0, (
            "Divergent rays should produce 0 score, filtered by score_min"
        )


class TestGhostPenalty:
    """Tests for ghost penalty suppression of scores."""

    def test_ghost_penalty_suppresses_score(self) -> None:
        """Score should be lower when ghost cameras see no supporting detections."""
        frames = tuple(range(20))
        ta = _make_tracklet("cam_a", 1, frames)
        tb = _make_tracklet("cam_b", 1, frames)

        # Converging rays
        lut_a = MockForwardLUT(
            "cam_a", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0])
        )
        lut_b = MockForwardLUT(
            "cam_b", np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 1.0])
        )
        forward_luts = {"cam_a": lut_a, "cam_b": lut_b}

        # Ghost cameras see the midpoint but no detections exist there
        inv_lut_with_ghost = MockInverseLUT(
            ["cam_a", "cam_b", "cam_c", "cam_d"],
            {
                "cam_a": (100.0, 100.0),
                "cam_b": (100.0, 100.0),
                "cam_c": (200.0, 200.0),
                "cam_d": (300.0, 300.0),
            },
        )

        config = MockAssociationConfig(t_min=3)

        # With detections supporting ghost cameras -> lower penalty
        detections_with: list[dict[str, list[tuple[float, float]]]] = [
            {
                "cam_a": [(100.0, 100.0)],
                "cam_b": [(100.0, 100.0)],
                "cam_c": [(200.0, 200.0)],
                "cam_d": [(300.0, 300.0)],
            }
            for _ in range(20)
        ]

        score_with = score_tracklet_pair(
            ta, tb, forward_luts, inv_lut_with_ghost, detections_with, config
        )

        # Without detections at ghost cameras -> higher penalty
        detections_without: list[dict[str, list[tuple[float, float]]]] = [
            {
                "cam_a": [(100.0, 100.0)],
                "cam_b": [(100.0, 100.0)],
                # No detections for cam_c, cam_d
            }
            for _ in range(20)
        ]

        score_without = score_tracklet_pair(
            ta, tb, forward_luts, inv_lut_with_ghost, detections_without, config
        )

        assert score_with > score_without, (
            f"Score with supporting detections ({score_with}) should be higher "
            f"than without ({score_without})"
        )
