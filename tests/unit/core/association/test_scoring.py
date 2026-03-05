"""Unit tests for pairwise cross-camera tracklet affinity scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from aquapose.core.association.scoring import (
    ray_ray_closest_point,
    ray_ray_closest_point_batch,
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
# ray_ray_closest_point_batch tests
# ---------------------------------------------------------------------------


class TestRayRayClosestPointBatch:
    """Tests for ray_ray_closest_point_batch vectorized geometry."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 42, 100])
    def test_batch_identical_to_scalar(self, seed: int) -> None:
        """Batch distances match scalar distances for random non-parallel rays."""
        rng = np.random.default_rng(seed)
        n = 20
        origins_a = rng.standard_normal((n, 3))
        dirs_a = rng.standard_normal((n, 3))
        dirs_a /= np.linalg.norm(dirs_a, axis=1, keepdims=True)
        origins_b = rng.standard_normal((n, 3))
        dirs_b = rng.standard_normal((n, 3))
        dirs_b /= np.linalg.norm(dirs_b, axis=1, keepdims=True)

        batch_dists = ray_ray_closest_point_batch(origins_a, dirs_a, origins_b, dirs_b)
        scalar_dists = np.array(
            [
                ray_ray_closest_point(origins_a[i], dirs_a[i], origins_b[i], dirs_b[i])[
                    0
                ]
                for i in range(n)
            ]
        )

        np.testing.assert_allclose(
            batch_dists,
            scalar_dists,
            atol=1e-6,
            err_msg=f"Batch and scalar disagree (seed={seed})",
        )

    def test_batch_parallel_rays(self) -> None:
        """Near-parallel ray pairs produce same distance as scalar fallback."""
        n = 5
        origins_a = np.zeros((n, 3))
        dirs_a = np.tile([1.0, 0.0, 0.0], (n, 1))
        origins_b = np.zeros((n, 3))
        origins_b[:, 1] = np.arange(1, n + 1, dtype=np.float64)  # offset in y
        dirs_b = np.tile([1.0, 0.0, 0.0], (n, 1))  # same direction = parallel

        batch_dists = ray_ray_closest_point_batch(origins_a, dirs_a, origins_b, dirs_b)
        scalar_dists = np.array(
            [
                ray_ray_closest_point(origins_a[i], dirs_a[i], origins_b[i], dirs_b[i])[
                    0
                ]
                for i in range(n)
            ]
        )

        np.testing.assert_allclose(batch_dists, scalar_dists, atol=1e-6)

    def test_batch_mixed_parallel_and_skew(self) -> None:
        """Mix of parallel and non-parallel rays matches scalar element-wise."""
        # Ray 0: parallel (same direction)
        # Ray 1: skew (perpendicular)
        # Ray 2: parallel (same direction, different offset)
        # Ray 3: intersecting
        origins_a = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        dirs_a = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ]
        )
        dirs_a /= np.linalg.norm(dirs_a, axis=1, keepdims=True)

        origins_b = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 3.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        dirs_b = np.array(
            [
                [1.0, 0.0, 0.0],  # parallel to ray 0
                [0.0, 0.0, 1.0],  # perpendicular to ray 1
                [0.0, 0.0, 1.0],  # parallel to ray 2
                [-1.0, 0.0, 1.0],  # converging with ray 3
            ]
        )
        dirs_b /= np.linalg.norm(dirs_b, axis=1, keepdims=True)

        batch_dists = ray_ray_closest_point_batch(origins_a, dirs_a, origins_b, dirs_b)
        scalar_dists = np.array(
            [
                ray_ray_closest_point(origins_a[i], dirs_a[i], origins_b[i], dirs_b[i])[
                    0
                ]
                for i in range(4)
            ]
        )

        np.testing.assert_allclose(batch_dists, scalar_dists, atol=1e-6)

    def test_batch_empty_input(self) -> None:
        """N=0 arrays return an empty (0,) array without error."""
        origins_a = np.empty((0, 3))
        dirs_a = np.empty((0, 3))
        origins_b = np.empty((0, 3))
        dirs_b = np.empty((0, 3))

        result = ray_ray_closest_point_batch(origins_a, dirs_a, origins_b, dirs_b)

        assert result.shape == (0,)
        assert result.dtype == np.float64

    def test_batch_single_ray(self) -> None:
        """N=1 matches scalar exactly."""
        origin_a = np.array([[0.0, 0.0, 0.0]])
        dir_a = np.array([[1.0, 0.0, 0.0]])
        origin_b = np.array([[0.0, 1.0, 0.0]])
        dir_b = np.array([[0.0, 0.0, 1.0]])

        batch_dist = ray_ray_closest_point_batch(origin_a, dir_a, origin_b, dir_b)
        scalar_dist, _ = ray_ray_closest_point(
            origin_a[0], dir_a[0], origin_b[0], dir_b[0]
        )

        assert batch_dist.shape == (1,)
        np.testing.assert_allclose(batch_dist[0], scalar_dist, atol=1e-6)

    def test_batch_intersecting_rays(self) -> None:
        """Known intersecting rays produce distance near zero."""
        # Two rays intersecting at (0.5, 0, 0.5)
        origins_a = np.array([[0.0, 0.0, 0.0]])
        dirs_a = np.array([[1.0, 0.0, 1.0]])
        dirs_a /= np.linalg.norm(dirs_a, axis=1, keepdims=True)
        origins_b = np.array([[1.0, 0.0, 0.0]])
        dirs_b = np.array([[-1.0, 0.0, 1.0]])
        dirs_b /= np.linalg.norm(dirs_b, axis=1, keepdims=True)

        result = ray_ray_closest_point_batch(origins_a, dirs_a, origins_b, dirs_b)

        assert result.shape == (1,)
        assert result[0] < 1e-6


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

        config = MockAssociationConfig()
        score = score_tracklet_pair(ta, tb, forward_luts, config)

        # Rays intersect at distance ~0, so each frame contributes 1.0 - (0/threshold) = 1.0
        # score_sum = 20.0, f = 20/20 = 1.0, w = 20/100 = 0.2, score = 1.0 * 0.2 = 0.2
        assert score == pytest.approx(0.2)

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

        config = MockAssociationConfig()

        score = score_tracklet_pair(ta, tb, forward_luts, config)
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

        config = MockAssociationConfig(t_min=3)

        score = score_tracklet_pair(ta, tb, forward_luts, config)
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

        config = MockAssociationConfig(early_k=3, t_min=3)

        score = score_tracklet_pair(ta, tb, forward_luts, config)
        assert score == 0.0

    def test_soft_scoring_distance_sensitivity(self) -> None:
        """Closer rays produce higher scores than farther rays (soft kernel validation).

        Creates two pairs of tracklets:
        - Pair A: rays intersect at distance ~0 (perfect match)
        - Pair B: rays are near-threshold apart (dist close to threshold)

        Validates that score_close > score_far, confirming the soft kernel
        differentiates distance magnitudes unlike binary inlier counting.
        """
        frames = tuple(range(10))

        # Pair A: converging rays that intersect at ~0 distance
        ta_close = _make_tracklet("cam_a", 1, frames)
        tb_close = _make_tracklet("cam_b", 1, frames)
        lut_a_close = MockForwardLUT(
            "cam_a", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0])
        )
        lut_b_close = MockForwardLUT(
            "cam_b", np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 1.0])
        )
        forward_luts_close = {"cam_a": lut_a_close, "cam_b": lut_b_close}

        # Pair B: rays that are skew with a closest distance close to but under threshold
        # Threshold is 0.03. We create rays with closest distance ~0.02 (under threshold
        # but much farther than near-zero).
        # Ray A along z-axis at (0,0,0), Ray B parallel along z-axis offset by 0.02 in x.
        # But parallel rays always diverge so we use slight convergence: give them
        # a tiny cross-component. Instead, use the skew configuration with known distance.
        # Ray A: origin (0,0,0), direction (0,0,1) (along z)
        # Ray B: origin (0.02,0,0), direction (0,0,1) (parallel, offset by 0.02 in x)
        # These are parallel and their closest distance is 0.02 (under threshold=0.03).
        ta_far = _make_tracklet("cam_a", 2, frames)
        tb_far = _make_tracklet("cam_b", 2, frames)
        lut_a_far = MockForwardLUT(
            "cam_a", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        )
        lut_b_far = MockForwardLUT(
            "cam_b", np.array([0.02, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        )
        forward_luts_far = {"cam_a": lut_a_far, "cam_b": lut_b_far}

        config = MockAssociationConfig(ray_distance_threshold=0.03, t_min=3, early_k=3)

        score_close = score_tracklet_pair(
            ta_close, tb_close, forward_luts_close, config
        )
        score_far = score_tracklet_pair(ta_far, tb_far, forward_luts_far, config)

        assert score_close > 0.0, (
            f"Close rays should produce positive score, got {score_close}"
        )
        assert score_far > 0.0, (
            f"Far (but inlier) rays should produce positive score, got {score_far}"
        )
        assert score_close > score_far, (
            f"Close rays (score={score_close:.4f}) should outscore far rays "
            f"(score={score_far:.4f}) with soft kernel"
        )


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

        config = MockAssociationConfig(score_min=0.0)

        scored = score_all_pairs(tracks_2d, forward_luts, inv_lut, config)

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

        inv_lut = MockInverseLUTNonAdjacent()

        config = MockAssociationConfig(score_min=0.3, t_min=3, early_k=3)

        scored = score_all_pairs(tracks_2d, forward_luts, inv_lut, config)

        assert len(scored) == 0, (
            "Divergent rays should produce 0 score, filtered by score_min"
        )
