"""Unit tests for 3D fish trajectory generation."""

from __future__ import annotations

import numpy as np

from aquapose.synthetic.trajectory import (
    FISH_BODY_LENGTH,
    MotionConfig,
    SchoolingConfig,
    TankConfig,
    TrajectoryConfig,
    TrajectoryResult,
    generate_trajectories,
)


def _default_config(**overrides: object) -> TrajectoryConfig:
    """Return a fast-running TrajectoryConfig for tests."""
    kwargs: dict[str, object] = dict(
        n_fish=3,
        duration_seconds=2.0,
        fps=10.0,
        random_seed=42,
    )
    kwargs.update(overrides)
    return TrajectoryConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Shape and type tests
# ---------------------------------------------------------------------------


def test_trajectory_shape() -> None:
    """Default config produces state array with correct shape."""
    cfg = _default_config(n_fish=4, duration_seconds=3.0, fps=10.0)
    result = generate_trajectories(cfg)

    expected_frames = round(3.0 * 10.0)
    assert isinstance(result, TrajectoryResult)
    assert result.n_fish == 4
    assert result.n_frames == expected_frames
    assert result.fps == 10.0
    assert result.states.shape == (expected_frames, 4, 7)
    assert result.states.dtype == np.float32


def test_trajectory_fish_ids_correct() -> None:
    """The 7th state dimension stores integer fish IDs 0..n_fish-1."""
    cfg = _default_config(n_fish=3)
    result = generate_trajectories(cfg)

    ids = result.states[:, :, 6]  # (n_frames, n_fish)
    for fish_idx in range(result.n_fish):
        np.testing.assert_array_equal(
            ids[:, fish_idx],
            fish_idx,
            err_msg=f"Fish {fish_idx} ID column should equal {fish_idx}",
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_trajectory_deterministic() -> None:
    """Same seed produces identical output across two calls."""
    cfg = _default_config()
    r1 = generate_trajectories(cfg)
    r2 = generate_trajectories(cfg)

    np.testing.assert_array_equal(r1.states, r2.states)


def test_trajectory_different_seeds_differ() -> None:
    """Different seeds produce different trajectories."""
    cfg_a = _default_config(random_seed=0)
    cfg_b = _default_config(random_seed=99)

    r_a = generate_trajectories(cfg_a)
    r_b = generate_trajectories(cfg_b)

    assert not np.allclose(r_a.states[:, :, :3], r_b.states[:, :, :3])


# ---------------------------------------------------------------------------
# Physical plausibility
# ---------------------------------------------------------------------------


def test_trajectory_stays_in_tank() -> None:
    """All positions remain within the tank cylinder at every frame."""
    cfg = _default_config(n_fish=5, duration_seconds=5.0)
    result = generate_trajectories(cfg)

    tank = cfg.tank
    x = result.states[:, :, 0]
    y = result.states[:, :, 1]
    z = result.states[:, :, 2]

    r_xy = np.sqrt(x**2 + y**2)
    inner_r = tank.radius - tank.wall_margin

    assert np.all(r_xy <= inner_r + 1e-4), (
        f"Some positions exceeded inner radius {inner_r:.3f}m. "
        f"Max r_xy = {r_xy.max():.4f}"
    )

    z_min = tank.water_z + tank.wall_margin
    z_max = tank.water_z + tank.depth - tank.wall_margin
    assert np.all(z >= z_min - 1e-4), (
        f"Some Z positions below z_min={z_min:.3f}. Min z = {z.min():.4f}"
    )
    assert np.all(z <= z_max + 1e-4), (
        f"Some Z positions above z_max={z_max:.3f}. Max z = {z.max():.4f}"
    )


def test_trajectory_speed_bounds() -> None:
    """All recorded speeds remain within [s_min, s_max]."""
    motion = MotionConfig(s_min=0.01, s_max=0.5)
    cfg = _default_config(n_fish=5, duration_seconds=5.0, motion=motion)
    result = generate_trajectories(cfg)

    speeds = result.states[:, :, 5]
    assert np.all(speeds >= motion.s_min - 1e-6), (
        f"Speed below s_min={motion.s_min}. Min={speeds.min():.6f}"
    )
    assert np.all(speeds <= motion.s_max + 1e-6), (
        f"Speed above s_max={motion.s_max}. Max={speeds.max():.6f}"
    )


def test_trajectory_collision_avoidance() -> None:
    """With 5 fish in a small tank, no pair is closer than 0.5*FISH_BODY_LENGTH."""
    # Small tank to stress-test collision avoidance
    tank = TankConfig(radius=0.5, depth=0.5, water_z=0.75)
    cfg = _default_config(n_fish=5, duration_seconds=5.0, fps=15.0, tank=tank)
    result = generate_trajectories(cfg)

    min_allowed = 0.5 * FISH_BODY_LENGTH
    positions = result.states[:, :, :3]  # (n_frames, n_fish, 3)

    for frame_idx in range(result.n_frames):
        pos = positions[frame_idx]  # (n_fish, 3)
        for i in range(result.n_fish):
            for j in range(i + 1, result.n_fish):
                dist = float(np.linalg.norm(pos[i] - pos[j]))
                assert dist >= min_allowed - 1e-3, (
                    f"Frame {frame_idx}: fish {i} and {j} too close "
                    f"({dist:.4f}m < {min_allowed:.4f}m)"
                )


# ---------------------------------------------------------------------------
# Schooling behaviour
# ---------------------------------------------------------------------------


def test_schooling_cohesion() -> None:
    """Tight schooling produces smaller mean inter-fish distance than independent."""
    n_fish = 5

    cfg_ind = _default_config(
        n_fish=n_fish,
        duration_seconds=3.0,
        fps=15.0,
        schooling=SchoolingConfig.independent(),
        random_seed=7,
    )
    cfg_tight = _default_config(
        n_fish=n_fish,
        duration_seconds=3.0,
        fps=15.0,
        schooling=SchoolingConfig.tight_school(),
        random_seed=7,
    )

    r_ind = generate_trajectories(cfg_ind)
    r_tight = generate_trajectories(cfg_tight)

    def mean_inter_dist(states: np.ndarray) -> float:
        """Compute mean pairwise inter-fish XY distance across all frames."""
        positions = states[:, :, :2]  # XY only, (n_frames, n_fish, 2)
        dists = []
        for frame in positions:
            for i in range(n_fish):
                for j in range(i + 1, n_fish):
                    dists.append(float(np.linalg.norm(frame[i] - frame[j])))
        return float(np.mean(dists))

    dist_ind = mean_inter_dist(r_ind.states)
    dist_tight = mean_inter_dist(r_tight.states)

    # Tight schooling should have smaller mean inter-fish distance
    assert dist_tight < dist_ind, (
        f"Tight schooling mean dist {dist_tight:.4f}m >= independent {dist_ind:.4f}m"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_fish_no_crash() -> None:
    """n_fish=1 runs without errors (no pairwise force issues)."""
    cfg = _default_config(n_fish=1, duration_seconds=3.0)
    result = generate_trajectories(cfg)

    assert result.n_fish == 1
    assert result.states.shape[1] == 1
    assert not np.any(np.isnan(result.states))


def test_schooling_config_presets() -> None:
    """SchoolingConfig class method presets return valid configs."""
    presets = [
        SchoolingConfig.independent(),
        SchoolingConfig.loose_school(),
        SchoolingConfig.tight_school(),
        SchoolingConfig.milling(),
        SchoolingConfig.streaming(),
    ]
    for preset in presets:
        assert isinstance(preset, SchoolingConfig)
        assert preset.cohesion >= 0.0
        assert preset.alignment >= 0.0
        assert preset.cohesion_radius > 0.0
        assert preset.alignment_radius > 0.0
