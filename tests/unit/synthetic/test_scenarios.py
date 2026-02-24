"""Unit tests for scenario presets and the generate_scenario() dispatcher."""

from __future__ import annotations

import pytest

from aquapose.synthetic import (
    SyntheticDataset,
    build_fabricated_rig,
    crossing_paths,
    generate_scenario,
    tight_schooling,
    track_fragmentation,
)
from aquapose.synthetic.trajectory import TrajectoryConfig


def _small_rig():
    """Return a 2x2 rig for fast scenario tests."""
    return build_fabricated_rig(n_cameras_x=2, n_cameras_y=2)


# ---------------------------------------------------------------------------
# Individual scenario config tests
# ---------------------------------------------------------------------------


def test_crossing_paths_returns_valid_config() -> None:
    """crossing_paths() returns (TrajectoryConfig, NoiseConfig) with n_fish=2."""
    cfg, noise = crossing_paths(difficulty=0.5, seed=42)

    assert isinstance(cfg, TrajectoryConfig)
    assert cfg.n_fish == 2
    assert cfg.random_seed == 42
    assert noise.base_miss_rate >= 0.0


def test_crossing_paths_difficulty_parameter() -> None:
    """crossing_paths() accepts difficulty in [0, 1] without errors."""
    for diff in [0.0, 0.3, 0.7, 1.0]:
        cfg, _ = crossing_paths(difficulty=diff, seed=0)
        assert cfg.n_fish == 2


def test_track_fragmentation_miss_rate() -> None:
    """track_fragmentation() returns NoiseConfig with the specified miss_rate."""
    for rate in [0.1, 0.25, 0.5, 0.9]:
        _, noise = track_fragmentation(miss_rate=rate, seed=0)
        assert abs(noise.base_miss_rate - rate) < 1e-9, (
            f"Expected miss_rate={rate}, got {noise.base_miss_rate}"
        )


def test_track_fragmentation_no_false_positives() -> None:
    """track_fragmentation() produces no false positives by design."""
    _, noise = track_fragmentation(miss_rate=0.25)
    assert noise.base_false_positive_rate == 0.0


def test_tight_schooling_uses_schooling_preset() -> None:
    """tight_schooling() returns config with cohesion > 0.5 (tight_school preset)."""
    cfg, _ = tight_schooling(n_fish=5, seed=42)

    assert cfg.n_fish == 5
    assert cfg.schooling.cohesion > 0.5, (
        f"tight_schooling cohesion {cfg.schooling.cohesion} should be > 0.5"
    )


def test_tight_schooling_n_fish_parameter() -> None:
    """tight_schooling() accepts custom n_fish."""
    cfg, _ = tight_schooling(n_fish=8, seed=7)
    assert cfg.n_fish == 8


# ---------------------------------------------------------------------------
# generate_scenario() dispatcher tests
# ---------------------------------------------------------------------------


def test_generate_scenario_crossing_paths() -> None:
    """generate_scenario('crossing_paths', rig) returns a SyntheticDataset."""
    rig = _small_rig()
    ds = generate_scenario("crossing_paths", rig, seed=0)

    assert isinstance(ds, SyntheticDataset)
    assert len(ds.frames) > 0

    # Frame count matches expected fps * duration
    cfg, _ = crossing_paths(seed=0)
    expected_frames = round(cfg.duration_seconds * cfg.fps)
    assert len(ds.frames) == expected_frames


def test_generate_scenario_unknown_raises() -> None:
    """generate_scenario() raises ValueError for an unknown scenario name."""
    rig = _small_rig()
    with pytest.raises(ValueError, match="Unknown scenario"):
        generate_scenario("nonexistent_scenario", rig)


def test_generate_scenario_all_known_names() -> None:
    """All four scenario names dispatch without errors."""
    rig = _small_rig()
    known = [
        "crossing_paths",
        "track_fragmentation",
        "tight_schooling",
        "startle_response",
    ]
    for name in known:
        # Use short durations for speed — override via kwargs
        ds = generate_scenario(name, rig, seed=0)
        assert isinstance(ds, SyntheticDataset), f"Scenario {name!r} failed"
        assert len(ds.frames) > 0, f"Scenario {name!r} produced empty dataset"
