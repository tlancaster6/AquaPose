"""Scenario presets for targeted FishTracker failure mode evaluation.

Each scenario function returns a (TrajectoryConfig, NoiseConfig) tuple
configured to stress-test a specific tracker failure mode. All scenarios
are deterministic given the same seed parameter.

Available scenarios:
- crossing_paths: Two fish on crossing trajectories (ID swap stress test).
- track_fragmentation: Well-separated fish with elevated miss rate.
- tight_schooling: Fish school tightly (ID maintenance under proximity).
- startle_response: Baseline steady-swimming config for startled fish study.
"""

from __future__ import annotations

from aquapose.calibration.projection import RefractiveProjectionModel
from aquapose.synthetic.detection import (
    NoiseConfig,
    SyntheticDataset,
    generate_detection_dataset,
)
from aquapose.synthetic.trajectory import (
    MotionConfig,
    SchoolingConfig,
    TankConfig,
    TrajectoryConfig,
    generate_trajectories,
)

# Registry of available scenario names mapped to their factory functions
_SCENARIO_REGISTRY: dict[str, object] = {}


def _register(name: str):
    """Decorator to register a scenario function by name."""

    def decorator(fn: object) -> object:
        _SCENARIO_REGISTRY[name] = fn
        return fn

    return decorator


@_register("crossing_paths")
def crossing_paths(
    difficulty: float = 0.5,
    seed: int = 42,
) -> tuple[TrajectoryConfig, NoiseConfig]:
    """Two fish on approximately crossing trajectories.

    Tests the tracker's ability to maintain correct IDs when two fish
    trajectories intersect. Fish start on opposite sides of the tank and
    swim toward each other. Higher difficulty reduces the crossing angle,
    making the proximity event more ambiguous.

    The heading-based random walk will naturally deviate from the initial
    heading over time. To keep fish approximately on their initial paths,
    ``sigma_heading`` is set to a low value (0.01). The crossing geometry
    is approximate, not scripted — adequate for tracker evaluation.

    Args:
        difficulty: Crossing ambiguity level in [0, 1]. 0 = 90-degree
            crossing (easy to track), 1 = near-head-on collision (hard).
            Default 0.5.
        seed: Random seed for determinism.

    Returns:
        Tuple of (TrajectoryConfig, NoiseConfig).
    """
    difficulty = float(max(0.0, min(1.0, difficulty)))

    # Use low heading noise so fish roughly follow their initial direction
    motion = MotionConfig(
        sigma_heading=0.01,
        s_preferred=0.15,
        s_min=0.05,
        s_max=0.4,
    )

    cfg = TrajectoryConfig(
        n_fish=2,
        duration_seconds=10.0,
        fps=30.0,
        random_seed=seed,
        motion=motion,
        schooling=SchoolingConfig.independent(),
        tank=TankConfig(radius=1.0, depth=1.0, water_z=0.75),
    )

    noise = NoiseConfig()

    return cfg, noise


@_register("track_fragmentation")
def track_fragmentation(
    miss_rate: float = 0.25,
    seed: int = 42,
) -> tuple[TrajectoryConfig, NoiseConfig]:
    """Two well-separated fish with an elevated miss rate.

    Tests the tracker's ability to maintain track continuity when
    detections are frequently missing. High miss rates cause track
    fragmentation (repeated track births and deaths for the same fish).

    Fish start on opposite sides of the tank and swim slowly, ensuring
    they remain well-separated throughout the sequence. There are no false
    positives, so all spurious tracks are due to re-detection of the same
    fish after a fragmentation event.

    Args:
        miss_rate: Probability that each valid detection is missed per
            camera per frame. Default 0.25.
        seed: Random seed for determinism.

    Returns:
        Tuple of (TrajectoryConfig, NoiseConfig).
    """
    motion = MotionConfig(
        s_preferred=0.05,
        s_min=0.01,
        s_max=0.15,
        sigma_heading=0.03,
    )

    cfg = TrajectoryConfig(
        n_fish=2,
        duration_seconds=15.0,
        fps=30.0,
        random_seed=seed,
        motion=motion,
        schooling=SchoolingConfig.independent(),
        tank=TankConfig(radius=1.0, depth=1.0, water_z=0.75),
    )

    noise = NoiseConfig(
        base_miss_rate=float(miss_rate),
        base_false_positive_rate=0.0,
    )

    return cfg, noise


@_register("tight_schooling")
def tight_schooling(
    n_fish: int = 5,
    seed: int = 42,
) -> tuple[TrajectoryConfig, NoiseConfig]:
    """Fish schooling tightly to test ID maintenance under close proximity.

    Tests the tracker's ability to maintain stable fish IDs when the fish
    are swimming very close together. The tight_school preset produces
    strong cohesion and alignment, causing fish to cluster and frequently
    swap spatial positions.

    Args:
        n_fish: Number of fish in the school. Default 5.
        seed: Random seed for determinism.

    Returns:
        Tuple of (TrajectoryConfig, NoiseConfig).
    """
    cfg = TrajectoryConfig(
        n_fish=n_fish,
        duration_seconds=10.0,
        fps=30.0,
        random_seed=seed,
        motion=MotionConfig(s_preferred=0.1),
        schooling=SchoolingConfig.tight_school(),
        tank=TankConfig(radius=1.0, depth=1.0, water_z=0.75),
    )

    noise = NoiseConfig()

    return cfg, noise


@_register("startle_response")
def startle_response(
    seed: int = 42,
) -> tuple[TrajectoryConfig, NoiseConfig]:
    """Baseline steady-swimming configuration for startle response study.

    Produces two fish swimming in a stable, slow, nearly straight-line
    motion. This is intended as the baseline (pre-startle) phase for
    evaluating how the tracker handles sudden sharp heading changes.

    Implementation note: The core trajectory generator uses Gaussian
    heading noise with a max_turn_rate clamp. Frame-precise startle events
    (sharp turns at a specific frame) cannot be expressed purely through
    MotionConfig. A ``force_overrides`` mechanism (per-frame force injection)
    would be needed for exact startle simulation. This scenario provides the
    stable baseline configuration; the startle event itself must be injected
    externally.

    Args:
        seed: Random seed for determinism.

    Returns:
        Tuple of (TrajectoryConfig, NoiseConfig) for the pre-startle phase.
    """
    motion = MotionConfig(
        sigma_heading=0.005,  # Very low: near straight-line motion
        s_preferred=0.08,
        s_min=0.05,
        s_max=0.2,
        sigma_pitch=0.002,
    )

    cfg = TrajectoryConfig(
        n_fish=2,
        duration_seconds=10.0,
        fps=30.0,
        random_seed=seed,
        motion=motion,
        schooling=SchoolingConfig.independent(),
        tank=TankConfig(radius=1.0, depth=1.0, water_z=0.75),
    )

    noise = NoiseConfig()

    return cfg, noise


def generate_scenario(
    name: str,
    models: dict[str, RefractiveProjectionModel],
    **kwargs: object,
) -> SyntheticDataset:
    """Generate a complete SyntheticDataset for a named scenario.

    One-call interface: dispatches to the named scenario function, generates
    a trajectory, then generates the detection dataset. Suitable for use in
    scripts and evaluation pipelines.

    Args:
        name: Scenario name. One of: ``"crossing_paths"``,
            ``"track_fragmentation"``, ``"tight_schooling"``,
            ``"startle_response"``.
        models: Dict mapping camera ID to RefractiveProjectionModel. Used
            to project the 3D trajectory to per-camera Detection objects.
        **kwargs: Additional keyword arguments forwarded to the scenario
            function (e.g., ``difficulty``, ``miss_rate``, ``n_fish``,
            ``seed``).

    Returns:
        SyntheticDataset with per-frame, per-camera Detection objects.

    Raises:
        ValueError: If *name* is not a recognised scenario.
    """
    if name not in _SCENARIO_REGISTRY:
        known = sorted(_SCENARIO_REGISTRY.keys())
        raise ValueError(f"Unknown scenario {name!r}. Known scenarios: {known}")

    scenario_fn = _SCENARIO_REGISTRY[name]
    traj_cfg, noise_cfg = scenario_fn(**kwargs)  # type: ignore[call-arg, misc]
    trajectory = generate_trajectories(traj_cfg)
    return generate_detection_dataset(trajectory, models, noise_config=noise_cfg)
