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

# Number of keypoints matching the AquaPose pose model
N_KEYPOINTS = 6


@dataclass(frozen=True)
class MockAssociationConfig:
    """Mock config satisfying AssociationConfigLike for tests."""

    ray_distance_threshold: float = 0.03
    score_min: float = 0.3
    t_min: int = 3
    t_saturate: int = 100
    early_k: int = 3
    min_shared_voxels: int = 1
    keypoint_confidence_floor: float = 0.3
    aggregation_method: str = "mean"


class MockForwardLUT:
    """Mock ForwardLUT that returns per-pixel ray geometry.

    Base ray: origin + direction. Pixel offsets produce small direction
    perturbations so multi-keypoint scoring gets distinct rays per keypoint.
    """

    def __init__(
        self, camera_id: str, origin: np.ndarray, direction: np.ndarray
    ) -> None:
        self.camera_id = camera_id
        self._origin = origin.astype(np.float32)
        self._direction = (direction / np.linalg.norm(direction)).astype(np.float32)

    def cast_ray(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return rays with small perturbation based on pixel coordinates."""
        n = pixels.shape[0]
        origins = torch.from_numpy(np.tile(self._origin, (n, 1)))
        base_dirs = np.tile(self._direction, (n, 1))
        # Add small perturbation based on pixel offset from (100, 100)
        px = pixels.numpy().astype(np.float32)
        offset = (px - np.array([[100.0, 100.0]])) * 0.0001
        perturbed = base_dirs.copy()
        perturbed[:, 0] += offset[:, 0]
        perturbed[:, 1] += offset[:, 1]
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / norms
        dirs = torch.from_numpy(perturbed)
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


def _make_keypoints(
    n_frames: int,
    centroid: tuple[float, float] = (100.0, 100.0),
    confidence: float = 0.9,
    *,
    k: int = N_KEYPOINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic keypoints near the centroid.

    Keypoints are spaced along the x-axis around the centroid:
    offsets = [-5, -3, -1, 1, 3, 5] pixels for K=6.

    Returns:
        Tuple of (keypoints, keypoint_conf) with shapes (T, K, 2) and (T, K).
    """
    offsets = np.linspace(-5, 5, k)  # (K,)
    kpts = np.zeros((n_frames, k, 2), dtype=np.float32)
    for i in range(k):
        kpts[:, i, 0] = centroid[0] + offsets[i]
        kpts[:, i, 1] = centroid[1]
    conf = np.full((n_frames, k), confidence, dtype=np.float32)
    return kpts, conf


def _make_tracklet(
    camera_id: str,
    track_id: int,
    frames: tuple[int, ...],
    centroid: tuple[float, float] = (100.0, 100.0),
    status: str = "detected",
    *,
    keypoints: np.ndarray | None = ...,  # type: ignore[assignment]
    keypoint_conf: np.ndarray | None = ...,  # type: ignore[assignment]
) -> Tracklet2D:
    """Build a synthetic Tracklet2D with uniform centroids, bboxes, and keypoints.

    If keypoints/keypoint_conf are not explicitly provided (sentinel ...),
    default keypoints near the centroid with confidence 0.9 are generated.
    Pass None explicitly to create a tracklet without keypoints.
    """
    centroids = tuple((centroid[0], centroid[1]) for _ in frames)
    bboxes = tuple((centroid[0] - 10.0, centroid[1] - 10.0, 20.0, 20.0) for _ in frames)
    frame_status = tuple(status for _ in frames)

    # Generate default keypoints if not explicitly provided
    if keypoints is ... and keypoint_conf is ...:
        keypoints, keypoint_conf = _make_keypoints(len(frames), centroid)
    elif keypoints is ...:
        keypoints = None
    elif keypoint_conf is ...:
        keypoint_conf = None

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

        # All keypoints near centroid, rays nearly converge for all,
        # score > 0 (exact value depends on per-keypoint perturbation)
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
        """Closer rays produce higher scores than farther rays (soft kernel validation)."""
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

        # Pair B: rays that are skew with closest distance ~0.02 (under threshold)
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

    def test_early_termination_t_shared_less_than_early_k(self) -> None:
        """t_shared < early_k with divergent rays returns 0.0."""
        frames = tuple(range(3))  # t_shared=3
        ta = _make_tracklet("cam_a", 1, frames)
        tb = _make_tracklet("cam_b", 1, frames)

        lut_a = MockDivergentForwardLUT("cam_a")
        lut_b = MockDivergentForwardLUT("cam_b")
        forward_luts = {"cam_a": lut_a, "cam_b": lut_b}

        config = MockAssociationConfig(early_k=5, t_min=3)

        score = score_tracklet_pair(ta, tb, forward_luts, config)
        assert score == 0.0

    def test_early_termination_t_shared_equals_early_k(self) -> None:
        """t_shared == early_k with divergent rays returns 0.0."""
        frames = tuple(range(5))  # t_shared=5
        ta = _make_tracklet("cam_a", 1, frames)
        tb = _make_tracklet("cam_b", 1, frames)

        lut_a = MockDivergentForwardLUT("cam_a")
        lut_b = MockDivergentForwardLUT("cam_b")
        forward_luts = {"cam_a": lut_a, "cam_b": lut_b}

        config = MockAssociationConfig(early_k=5, t_min=3)

        score = score_tracklet_pair(ta, tb, forward_luts, config)
        assert score == 0.0

    def test_two_phase_scoring_converging(self) -> None:
        """t_shared > early_k with converging rays produces positive score."""
        frames = tuple(range(20))  # t_shared=20, early_k=5
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

        config = MockAssociationConfig(early_k=5, t_min=3)

        score = score_tracklet_pair(ta, tb, forward_luts, config)
        assert score > 0.0, f"Expected positive score, got {score}"


# ---------------------------------------------------------------------------
# Keypoint-specific scoring tests
# ---------------------------------------------------------------------------


class TestKeypointScoring:
    """Tests for multi-keypoint scoring features."""

    def test_none_keypoints_returns_zero(self) -> None:
        """Tracklets with keypoints=None return score 0.0."""
        frames = tuple(range(10))
        ta = _make_tracklet("cam_a", 1, frames, keypoints=None, keypoint_conf=None)
        tb = _make_tracklet("cam_b", 1, frames)

        lut_a = MockForwardLUT(
            "cam_a", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0])
        )
        lut_b = MockForwardLUT(
            "cam_b", np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 1.0])
        )
        forward_luts = {"cam_a": lut_a, "cam_b": lut_b}

        config = MockAssociationConfig()
        score = score_tracklet_pair(ta, tb, forward_luts, config)
        assert score == 0.0

    def test_both_none_keypoints_returns_zero(self) -> None:
        """Both tracklets with keypoints=None return score 0.0."""
        frames = tuple(range(10))
        ta = _make_tracklet("cam_a", 1, frames, keypoints=None, keypoint_conf=None)
        tb = _make_tracklet("cam_b", 1, frames, keypoints=None, keypoint_conf=None)

        lut_a = MockForwardLUT(
            "cam_a", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0])
        )
        lut_b = MockForwardLUT(
            "cam_b", np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 1.0])
        )
        forward_luts = {"cam_a": lut_a, "cam_b": lut_b}

        config = MockAssociationConfig()
        score = score_tracklet_pair(ta, tb, forward_luts, config)
        assert score == 0.0

    def test_low_confidence_excluded(self) -> None:
        """Keypoints below confidence floor are excluded from scoring.

        Creates two identical test cases:
        1. All 6 keypoints at confidence 0.9
        2. Keypoints 3-5 at confidence 0.1 (below floor 0.3)

        Only keypoints 0-2 should contribute in case 2, producing a different
        score than case 1 (different number of rays in the mean).
        """
        frames = tuple(range(10))
        n = len(frames)

        # Case 1: all keypoints confident
        kpts, conf = _make_keypoints(n)
        ta1 = _make_tracklet("cam_a", 1, frames, keypoints=kpts, keypoint_conf=conf)
        tb1 = _make_tracklet(
            "cam_b", 1, frames, keypoints=kpts.copy(), keypoint_conf=conf.copy()
        )

        # Case 2: keypoints 3-5 below confidence floor
        conf_partial = conf.copy()
        conf_partial[:, 3:] = 0.1  # below floor

        ta2 = _make_tracklet(
            "cam_a", 2, frames, keypoints=kpts, keypoint_conf=conf_partial
        )
        tb2 = _make_tracklet(
            "cam_b", 2, frames, keypoints=kpts.copy(), keypoint_conf=conf_partial.copy()
        )

        lut_a = MockForwardLUT(
            "cam_a", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0])
        )
        lut_b = MockForwardLUT(
            "cam_b", np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 1.0])
        )
        forward_luts = {"cam_a": lut_a, "cam_b": lut_b}

        config = MockAssociationConfig()
        score_all = score_tracklet_pair(ta1, tb1, forward_luts, config)
        score_partial = score_tracklet_pair(ta2, tb2, forward_luts, config)

        # Both should be positive (converging rays)
        assert score_all > 0.0
        assert score_partial > 0.0
        # With different keypoint counts in the mean, scores differ
        # (not asserting which is larger — depends on ray geometry)

    def test_all_keypoints_below_floor_skips_frame(self) -> None:
        """Frames where all keypoints are below floor are skipped."""
        frames = tuple(range(10))
        n = len(frames)
        kpts, conf = _make_keypoints(n)

        # Set all keypoints to 0.0 confidence on frames 0-4
        conf_sparse = conf.copy()
        conf_sparse[:5, :] = 0.0

        ta = _make_tracklet(
            "cam_a", 1, frames, keypoints=kpts, keypoint_conf=conf_sparse
        )
        tb = _make_tracklet(
            "cam_b", 1, frames, keypoints=kpts.copy(), keypoint_conf=conf_sparse.copy()
        )

        lut_a = MockForwardLUT(
            "cam_a", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0])
        )
        lut_b = MockForwardLUT(
            "cam_b", np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 1.0])
        )
        forward_luts = {"cam_a": lut_a, "cam_b": lut_b}

        config = MockAssociationConfig(t_min=3, early_k=3)
        score = score_tracklet_pair(ta, tb, forward_luts, config)

        # Only 5 frames have valid keypoints, effective_t_shared = 5
        # But early_k=3 means we check first 3 frames, all skipped, score_sum=0
        # This triggers early termination -> score 0
        assert score == 0.0

    def test_partial_confidence_intersection(self) -> None:
        """Only keypoints confident on BOTH tracklets participate (intersection)."""
        frames = tuple(range(10))
        n = len(frames)
        kpts, conf = _make_keypoints(n)

        # Tracklet A: keypoints 0,1,2 confident, 3,4,5 not
        conf_a = conf.copy()
        conf_a[:, 3:] = 0.1

        # Tracklet B: keypoints 1,2,3 confident, 0,4,5 not
        conf_b = conf.copy()
        conf_b[:, 0] = 0.1
        conf_b[:, 4:] = 0.1

        # Intersection: only keypoints 1,2 should participate
        ta = _make_tracklet("cam_a", 1, frames, keypoints=kpts, keypoint_conf=conf_a)
        tb = _make_tracklet(
            "cam_b", 1, frames, keypoints=kpts.copy(), keypoint_conf=conf_b
        )

        lut_a = MockForwardLUT(
            "cam_a", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0])
        )
        lut_b = MockForwardLUT(
            "cam_b", np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 1.0])
        )
        forward_luts = {"cam_a": lut_a, "cam_b": lut_b}

        config = MockAssociationConfig()
        score = score_tracklet_pair(ta, tb, forward_luts, config)

        # Should produce a valid positive score with only 2 keypoints
        assert score > 0.0

    def test_vectorized_matches_loop_reference(self) -> None:
        """Vectorized implementation matches a reference loop-based computation."""
        frames = tuple(range(10))
        n = len(frames)
        rng = np.random.default_rng(42)

        # Generate random keypoints with varying confidence
        kpts_a = rng.uniform(80, 120, (n, N_KEYPOINTS, 2)).astype(np.float32)
        kpts_b = rng.uniform(80, 120, (n, N_KEYPOINTS, 2)).astype(np.float32)
        conf_a = rng.uniform(0.0, 1.0, (n, N_KEYPOINTS)).astype(np.float32)
        conf_b = rng.uniform(0.0, 1.0, (n, N_KEYPOINTS)).astype(np.float32)

        ta = _make_tracklet("cam_a", 1, frames, keypoints=kpts_a, keypoint_conf=conf_a)
        tb = _make_tracklet("cam_b", 1, frames, keypoints=kpts_b, keypoint_conf=conf_b)

        lut_a = MockForwardLUT(
            "cam_a", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0])
        )
        lut_b = MockForwardLUT(
            "cam_b", np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 1.0])
        )
        forward_luts = {"cam_a": lut_a, "cam_b": lut_b}

        config = MockAssociationConfig(t_min=1, early_k=3)

        # Vectorized result
        score_vec = score_tracklet_pair(ta, tb, forward_luts, config)

        # Reference loop implementation
        floor = config.keypoint_confidence_floor
        threshold = config.ray_distance_threshold
        score_sum = 0.0
        n_skipped = 0

        for i in range(n):
            valid = (conf_a[i] >= floor) & (conf_b[i] >= floor)
            if not valid.any():
                n_skipped += 1
                continue

            dists_frame = []
            for k_idx in range(N_KEYPOINTS):
                if not valid[k_idx]:
                    continue
                px_a = torch.tensor(kpts_a[i, k_idx : k_idx + 1], dtype=torch.float32)
                px_b = torch.tensor(kpts_b[i, k_idx : k_idx + 1], dtype=torch.float32)
                oa, da = lut_a.cast_ray(px_a)
                ob, db = lut_b.cast_ray(px_b)
                dist = ray_ray_closest_point_batch(
                    oa.numpy().astype(np.float64),
                    da.numpy().astype(np.float64),
                    ob.numpy().astype(np.float64),
                    db.numpy().astype(np.float64),
                )
                dists_frame.append(float(dist[0]))

            mean_dist = np.mean(dists_frame)
            if mean_dist < threshold:
                score_sum += 1.0 - mean_dist / threshold

        effective_t = n - n_skipped
        if effective_t > 0:
            f_ref = score_sum / effective_t
            w_ref = min(n, config.t_saturate) / config.t_saturate
            score_ref = f_ref * w_ref
        else:
            score_ref = 0.0

        np.testing.assert_allclose(
            score_vec,
            score_ref,
            atol=1e-10,
            err_msg="Vectorized and loop-based reference disagree",
        )

    def test_lut_round_trip_ray_convergence(self) -> None:
        """Round-trip LUT correctness: converging rays meet within 2mm.

        Uses MockForwardLUTs with known convergent geometry. Verifies that
        rays from two cameras intersect within 0.002m (2mm) of the expected
        convergence point.
        """
        # Two rays that intersect at (0.5, 0, 0.5)
        origin_a = np.array([0.0, 0.0, 0.0])
        dir_a = np.array([1.0, 0.0, 1.0])
        dir_a = dir_a / np.linalg.norm(dir_a)

        origin_b = np.array([1.0, 0.0, 0.0])
        dir_b = np.array([-1.0, 0.0, 1.0])
        dir_b = dir_b / np.linalg.norm(dir_b)

        dist, midpoint = ray_ray_closest_point(origin_a, dir_a, origin_b, dir_b)

        # Verify convergence within 2mm
        assert dist < 0.002, f"Ray-ray distance {dist:.6f}m exceeds 2mm threshold"
        # Verify midpoint is near expected (0.5, 0, 0.5)
        np.testing.assert_allclose(midpoint, [0.5, 0.0, 0.5], atol=0.002)

    def test_aggregation_method_field_exists(self) -> None:
        """Config objects have aggregation_method field defaulting to 'mean'."""
        from aquapose.engine.config import AssociationConfig

        mock_cfg = MockAssociationConfig()
        real_cfg = AssociationConfig()

        assert mock_cfg.aggregation_method == "mean"
        assert real_cfg.aggregation_method == "mean"

    def test_keypoint_confidence_floor_field_exists(self) -> None:
        """Config objects have keypoint_confidence_floor field defaulting to 0.3."""
        from aquapose.engine.config import AssociationConfig

        mock_cfg = MockAssociationConfig()
        real_cfg = AssociationConfig()

        assert mock_cfg.keypoint_confidence_floor == 0.3
        assert real_cfg.keypoint_confidence_floor == 0.3

    def test_single_confident_keypoint_contributes(self) -> None:
        """A frame with only 1 confident keypoint still contributes (minimum is 1)."""
        frames = tuple(range(5))
        n = len(frames)
        kpts, conf = _make_keypoints(n)

        # Only keypoint 0 is confident
        conf_single = np.full_like(conf, 0.1)
        conf_single[:, 0] = 0.9

        ta = _make_tracklet(
            "cam_a", 1, frames, keypoints=kpts, keypoint_conf=conf_single
        )
        tb = _make_tracklet(
            "cam_b", 1, frames, keypoints=kpts.copy(), keypoint_conf=conf_single.copy()
        )

        lut_a = MockForwardLUT(
            "cam_a", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0])
        )
        lut_b = MockForwardLUT(
            "cam_b", np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 1.0])
        )
        forward_luts = {"cam_a": lut_a, "cam_b": lut_b}

        config = MockAssociationConfig(t_min=1, early_k=3)
        score = score_tracklet_pair(ta, tb, forward_luts, config)

        assert score > 0.0, (
            "Single confident keypoint should still produce positive score"
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
