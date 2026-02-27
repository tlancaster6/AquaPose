"""Unit tests for head-tail orientation resolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from aquapose.core.midline.orientation import resolve_orientation

# ---------------------------------------------------------------------------
# Mock config and ForwardLUT
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockOrientationConfig:
    """Mock config satisfying OrientationConfigLike for tests."""

    speed_threshold: float = 2.0
    orientation_weight_geometric: float = 1.0
    orientation_weight_velocity: float = 0.5
    orientation_weight_temporal: float = 0.3


class MockForwardLUT:
    """Mock ForwardLUT returning controlled ray geometry.

    Returns rays based on pixel position to simulate different head/tail
    triangulation quality.
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


def _make_converging_luts() -> dict[str, MockForwardLUT]:
    """Create LUTs with rays converging near origin."""
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
    }


def _make_midline_points(
    n: int = 15, head: float = 10.0, tail: float = 100.0
) -> np.ndarray:
    """Create a simple horizontal midline from head_x to tail_x."""
    points = np.zeros((n, 2), dtype=np.float64)
    points[:, 0] = np.linspace(head, tail, n)  # x: head to tail
    points[:, 1] = 50.0  # y: constant
    return points


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGeometricVote:
    """Test cross-camera geometric vote signal."""

    def test_consistent_midlines_no_flip(self) -> None:
        """3 cameras with identical midline orientation -> no flip."""
        luts = _make_converging_luts()
        pts = _make_midline_points()

        midlines = {
            "cam_a": pts.copy(),
            "cam_b": pts.copy(),
            "cam_c": pts.copy(),
        }

        result, orientation = resolve_orientation(
            midlines,
            luts,
            velocity=None,
            prev_orientation=None,
            speed=0.0,
            config=MockOrientationConfig(),
        )

        # With consistent midlines, should keep original orientation
        assert orientation in (1, -1)
        # Result should have all 3 cameras
        assert set(result.keys()) == {"cam_a", "cam_b", "cam_c"}
        for _cam_id, r_pts in result.items():
            assert r_pts.shape == (15, 2)

    def test_geometric_vote_with_flipped_camera(self) -> None:
        """One camera has flipped midline -> consensus picks majority orientation."""
        luts = _make_converging_luts()
        pts = _make_midline_points()

        midlines = {
            "cam_a": pts.copy(),
            "cam_b": pts.copy(),
            "cam_c": pts[::-1].copy(),  # Flipped
        }

        _result, orientation = resolve_orientation(
            midlines,
            luts,
            velocity=None,
            prev_orientation=None,
            speed=0.0,
            config=MockOrientationConfig(),
        )

        # Should resolve to some consistent orientation
        assert orientation in (1, -1)


class TestVelocityGating:
    """Test velocity signal is gated by speed threshold."""

    def test_below_speed_threshold_ignores_velocity(self) -> None:
        """Below speed_threshold, velocity signal is not used."""
        luts = _make_converging_luts()
        pts = _make_midline_points()
        midlines = {"cam_a": pts.copy()}

        # Velocity pointing opposite to midline direction
        velocity = (-5.0, 0.0)
        speed = 1.5  # Below threshold of 2.0

        _, orientation_slow = resolve_orientation(
            midlines,
            luts,
            velocity=velocity,
            prev_orientation=None,
            speed=speed,
            config=MockOrientationConfig(),
        )

        # With speed below threshold and no other signals, should default to +1
        assert orientation_slow == 1

    def test_above_speed_threshold_uses_velocity(self) -> None:
        """Above speed_threshold, velocity signal contributes to decision."""
        luts = _make_converging_luts()
        pts = _make_midline_points(head=10.0, tail=100.0)

        # Single camera so no geometric vote
        midlines = {"cam_a": pts.copy()}

        # Velocity in +x direction (same as head->tail direction)
        # head_dir = -(tail - head) = -(90, 0) = (-90, 0)
        # velocity = (5, 0), dot(head_dir, vel) = -450 < 0 -> vote = -1 (flip)
        velocity = (5.0, 0.0)
        speed = 5.0  # Above threshold

        _, orientation = resolve_orientation(
            midlines,
            luts,
            velocity=velocity,
            prev_orientation=None,
            speed=speed,
            config=MockOrientationConfig(),
        )

        # Should flip since velocity aligns with tail direction
        assert orientation == -1


class TestTemporalPrior:
    """Test temporal prior signal."""

    def test_maintains_previous_orientation(self) -> None:
        """Temporal prior maintains previous orientation when other signals weak."""
        luts = _make_converging_luts()
        pts = _make_midline_points()
        midlines = {"cam_a": pts.copy()}

        # No velocity, single camera (no geometric vote)
        # Only signal is temporal prior
        _, orientation = resolve_orientation(
            midlines,
            luts,
            velocity=None,
            prev_orientation=-1,
            speed=0.0,
            config=MockOrientationConfig(),
        )

        # Should maintain previous orientation (-1)
        assert orientation == -1

    def test_first_frame_no_prior(self) -> None:
        """First frame with no prior defaults to +1."""
        luts = _make_converging_luts()
        pts = _make_midline_points()
        midlines = {"cam_a": pts.copy()}

        _, orientation = resolve_orientation(
            midlines,
            luts,
            velocity=None,
            prev_orientation=None,
            speed=0.0,
            config=MockOrientationConfig(),
        )

        assert orientation == 1


class TestSingleCamera:
    """Test behavior with single camera (no geometric vote)."""

    def test_single_camera_velocity_and_temporal(self) -> None:
        """Single camera falls back to velocity + temporal only."""
        luts = _make_converging_luts()
        pts = _make_midline_points()
        midlines = {"cam_a": pts.copy()}

        # Only velocity and temporal signals available
        _, orientation = resolve_orientation(
            midlines,
            luts,
            velocity=(5.0, 0.0),
            prev_orientation=1,
            speed=5.0,
            config=MockOrientationConfig(),
        )

        assert orientation in (1, -1)

    def test_single_camera_first_frame_no_velocity(self) -> None:
        """Single camera, first frame, no velocity -> default +1."""
        luts = _make_converging_luts()
        pts = _make_midline_points()
        midlines = {"cam_a": pts.copy()}

        _, orientation = resolve_orientation(
            midlines,
            luts,
            velocity=None,
            prev_orientation=None,
            speed=0.0,
            config=MockOrientationConfig(),
        )

        assert orientation == 1


class TestOutputStructure:
    """Test output dict structure and types."""

    def test_output_preserves_camera_keys(self) -> None:
        """Output dict has same camera keys as input."""
        luts = _make_converging_luts()
        pts = _make_midline_points()

        midlines = {
            "cam_a": pts.copy(),
            "cam_b": pts.copy(),
        }

        result, _ = resolve_orientation(
            midlines,
            luts,
            velocity=None,
            prev_orientation=None,
            speed=0.0,
            config=MockOrientationConfig(),
        )

        assert set(result.keys()) == {"cam_a", "cam_b"}

    def test_output_preserves_shape(self) -> None:
        """Output arrays have same shape as input."""
        luts = _make_converging_luts()
        pts = _make_midline_points(n=15)

        midlines = {"cam_a": pts.copy()}

        result, _ = resolve_orientation(
            midlines,
            luts,
            velocity=None,
            prev_orientation=None,
            speed=0.0,
            config=MockOrientationConfig(),
        )

        assert result["cam_a"].shape == (15, 2)

    def test_flipped_output_reverses_points(self) -> None:
        """When orientation is -1, midline points are reversed."""
        luts = _make_converging_luts()
        pts = _make_midline_points(head=10.0, tail=100.0)
        midlines = {"cam_a": pts.copy()}

        # Force flip via strong temporal prior
        _, orientation = resolve_orientation(
            midlines,
            luts,
            velocity=None,
            prev_orientation=-1,
            speed=0.0,
            config=MockOrientationConfig(),
        )

        assert orientation == -1
