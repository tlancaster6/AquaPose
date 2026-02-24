"""3D fish trajectory generation with heading-based random walk and social forces.

Generates multi-fish 3D trajectories confined to a cylindrical tank volume.
Supports independent motion, schooling, and collision avoidance. Used to
produce ground truth paths for synthetic tracker evaluation datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Fish body length constant (matches FishConfig.scale default of 0.085m)
FISH_BODY_LENGTH: float = 0.085

# Collision avoidance trigger distance (2 * body lengths)
_COLLISION_RADIUS: float = 2.0 * FISH_BODY_LENGTH

# Schooling force scale factor — keeps forces from overwhelming heading noise
_SCHOOLING_FORCE_SCALE: float = 0.1


@dataclass
class MotionConfig:
    """Per-fish motion parameters for the heading-based random walk.

    Attributes:
        s_min: Minimum speed in m/s.
        s_max: Maximum speed in m/s.
        s_preferred: Speed AR(1) mean-reversion target in m/s.
        sigma_speed: Speed noise standard deviation (m/s per timestep).
        speed_persistence: Speed AR(1) autocorrelation (0=pure noise, 1=fixed speed).
        sigma_heading: Heading noise standard deviation in radians per timestep.
        max_turn_rate: Maximum heading change per timestep in radians.
        sigma_pitch: Pitch (vertical heading) noise standard deviation per timestep.
        max_pitch: Maximum absolute pitch angle in radians.
        pitch_reversion: Pitch mean-reversion rate toward zero per timestep.
    """

    s_min: float = 0.01
    s_max: float = 0.5
    s_preferred: float = 0.1
    sigma_speed: float = 0.02
    speed_persistence: float = 0.95
    sigma_heading: float = 0.05
    max_turn_rate: float = 0.3
    sigma_pitch: float = 0.01
    max_pitch: float = 0.25
    pitch_reversion: float = 0.8


@dataclass
class SchoolingConfig:
    """Inter-fish social force parameters for schooling behaviour.

    Attributes:
        cohesion: Weight for cohesion force (steer toward neighbour centroid).
        alignment: Weight for alignment force (match neighbour heading).
        cohesion_radius: Radius in metres within which cohesion force acts.
        alignment_radius: Radius in metres within which alignment force acts.
    """

    cohesion: float = 0.0
    alignment: float = 0.0
    cohesion_radius: float = 0.5
    alignment_radius: float = 0.3

    @classmethod
    def independent(cls) -> SchoolingConfig:
        """No schooling — fish move independently."""
        return cls(cohesion=0.0, alignment=0.0)

    @classmethod
    def loose_school(cls) -> SchoolingConfig:
        """Loose schooling with weak cohesion and moderate alignment."""
        return cls(
            cohesion=0.3, alignment=0.4, cohesion_radius=0.8, alignment_radius=0.5
        )

    @classmethod
    def tight_school(cls) -> SchoolingConfig:
        """Tight schooling — strong cohesion and alignment at close range."""
        return cls(
            cohesion=0.8, alignment=0.7, cohesion_radius=0.4, alignment_radius=0.3
        )

    @classmethod
    def milling(cls) -> SchoolingConfig:
        """Milling — strong cohesion, no alignment (fish circle each other)."""
        return cls(
            cohesion=0.9, alignment=0.0, cohesion_radius=0.5, alignment_radius=0.3
        )

    @classmethod
    def streaming(cls) -> SchoolingConfig:
        """Streaming — moderate cohesion, strong alignment (polarised group)."""
        return cls(
            cohesion=0.4, alignment=0.9, cohesion_radius=0.6, alignment_radius=0.5
        )


@dataclass
class TankConfig:
    """Cylindrical tank geometry parameters.

    Cameras are placed at Z=0 looking downward. The water surface is at
    ``water_z``. Fish are initialised in the depth range
    [water_z, water_z + depth].

    Attributes:
        radius: Tank radius in metres.
        depth: Water depth in metres.
        wall_margin: Soft-repulsion zone width near cylindrical wall in metres.
        water_z: Z coordinate of the water surface in world frame. Matches
            build_fabricated_rig convention: cameras at Z=0, water at water_z.
        center_x: X offset of the tank centre in world frame (default 0.0).
            When using a real camera rig the coverage zone centroid may not
            coincide with the world origin; set this to shift fish spawning
            into the well-covered region.
        center_y: Y offset of the tank centre in world frame (default 0.0).
    """

    radius: float = 1.0
    depth: float = 1.0
    wall_margin: float = 0.05
    water_z: float = 0.75
    center_x: float = 0.0
    center_y: float = 0.0


@dataclass
class TrajectoryConfig:
    """Full configuration for a multi-fish trajectory simulation.

    Attributes:
        n_fish: Number of fish to simulate.
        duration_seconds: Simulation duration in seconds.
        fps: Frames per second.
        random_seed: Seed for the NumPy RNG. Use None for non-deterministic.
        motion: Motion model parameters.
        schooling: Social force parameters.
        tank: Tank geometry parameters.
    """

    n_fish: int = 3
    duration_seconds: float = 10.0
    fps: float = 30.0
    random_seed: int | None = 42
    motion: MotionConfig = field(default_factory=MotionConfig)
    schooling: SchoolingConfig = field(default_factory=SchoolingConfig.independent)
    tank: TankConfig = field(default_factory=TankConfig)


@dataclass
class FishTrajectoryState:
    """Per-fish per-frame state snapshot.

    Attributes:
        position: 3D world position (x, y, z) in metres.
        heading_xy: Horizontal heading angle in radians (rotation around Z).
        heading_z: Vertical heading (pitch) angle in radians.
        speed: Current swimming speed in m/s.
    """

    position: np.ndarray  # shape (3,)
    heading_xy: float
    heading_z: float
    speed: float


@dataclass
class TrajectoryResult:
    """Output of a multi-fish trajectory simulation.

    Attributes:
        n_fish: Number of simulated fish.
        n_frames: Number of simulated frames.
        fps: Frames per second used in the simulation.
        states: Full state array, shape (n_frames, n_fish, 7). The 7 dimensions
            are [x, y, z, heading_xy, heading_z, speed, fish_id].
        config: The TrajectoryConfig used to generate this result (for
            reproducibility).
    """

    n_fish: int
    n_frames: int
    fps: float
    states: np.ndarray  # (n_frames, n_fish, 7)
    config: TrajectoryConfig


def _init_positions(
    n_fish: int,
    tank: TankConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialise fish positions and headings uniformly inside the tank cylinder.

    Args:
        n_fish: Number of fish to initialise.
        tank: Tank geometry parameters.
        rng: Seeded RNG instance.

    Returns:
        Tuple of (positions, heading_xy, heading_z, speeds):
        - positions: shape (n_fish, 3)
        - heading_xy: shape (n_fish,) radians
        - heading_z: shape (n_fish,) radians
        - speeds: shape (n_fish,)
    """
    inner_r = tank.radius - tank.wall_margin
    z_min = tank.water_z + tank.wall_margin
    z_max = tank.water_z + tank.depth - tank.wall_margin

    # Uniform sampling in cylinder: rejection sample XY within inner radius.
    # Apply tank center offset so fish spawn in the world-frame coverage zone.
    positions = np.zeros((n_fish, 3), dtype=np.float64)
    for i in range(n_fish):
        while True:
            x = rng.uniform(-inner_r, inner_r)
            y = rng.uniform(-inner_r, inner_r)
            if x * x + y * y <= inner_r * inner_r:
                positions[i, 0] = x + tank.center_x
                positions[i, 1] = y + tank.center_y
                break
        positions[i, 2] = rng.uniform(z_min, z_max)

    heading_xy = rng.uniform(-np.pi, np.pi, size=n_fish)
    heading_z = rng.normal(0.0, 0.01, size=n_fish)
    speeds = np.full(n_fish, 0.1)  # s_preferred default
    return positions, heading_xy, heading_z, speeds


def _boundary_force(
    positions: np.ndarray,
    tank: TankConfig,
    heading_xy: np.ndarray,
) -> np.ndarray:
    """Compute soft boundary repulsion toward tank centre.

    Activates within 10 cm of cylindrical wall; linearly ramps from zero at
    the outer boundary of the repulsion zone to max_turn_rate at the wall.

    Args:
        positions: shape (n_fish, 3)
        tank: Tank geometry parameters.
        heading_xy: Current horizontal headings, shape (n_fish,).

    Returns:
        Delta heading_xy corrections, shape (n_fish,).
    """
    n_fish = positions.shape[0]
    repulsion_zone = 0.10  # metres
    corrections = np.zeros(n_fish, dtype=np.float64)

    # Compute radial distance from tank centre (accounting for offset)
    x = positions[:, 0] - tank.center_x
    y = positions[:, 1] - tank.center_y
    r = np.sqrt(x * x + y * y)
    dist_to_wall = tank.radius - r

    near_wall = dist_to_wall < repulsion_zone
    if not np.any(near_wall):
        return corrections

    # Angle toward tank centre (opposite of radial outward direction)
    angle_to_centre = np.arctan2(-y, -x)

    for i in np.where(near_wall)[0]:
        # Strength ramps linearly from 0 (at repulsion_zone) to 1 (at wall)
        strength = 1.0 - dist_to_wall[i] / repulsion_zone
        strength = float(np.clip(strength, 0.0, 1.0))

        # Angular difference between current heading and direction to centre
        diff = angle_to_centre[i] - heading_xy[i]
        # Wrap to [-pi, pi]
        diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
        corrections[i] = strength * 0.3 * diff  # scale by max_turn_rate

    # Top/bottom boundary: handled via pitch reversion in main loop
    return corrections


def _collision_force(
    positions: np.ndarray,
    heading_xy: np.ndarray,
) -> np.ndarray:
    """Compute pairwise collision avoidance correction for horizontal heading.

    When two fish are within ``_COLLISION_RADIUS``, steer each fish away
    from the other. Force magnitude is inversely proportional to distance.

    Args:
        positions: shape (n_fish, 3)
        heading_xy: Current horizontal headings, shape (n_fish,).

    Returns:
        Delta heading_xy corrections, shape (n_fish,).
    """
    n_fish = positions.shape[0]
    corrections = np.zeros(n_fish, dtype=np.float64)

    if n_fish < 2:
        return corrections

    for i in range(n_fish):
        for j in range(i + 1, n_fish):
            diff = positions[i, :2] - positions[j, :2]
            dist = float(np.linalg.norm(diff))
            if dist < _COLLISION_RADIUS and dist > 1e-6:
                # Angle from j to i (fish i should steer away from j)
                away_i = np.arctan2(diff[1], diff[0])
                away_j = np.arctan2(-diff[1], -diff[0])

                strength = 1.0 - dist / _COLLISION_RADIUS
                strength = float(np.clip(strength, 0.0, 1.0))

                d_i = away_i - heading_xy[i]
                d_i = (d_i + np.pi) % (2.0 * np.pi) - np.pi
                corrections[i] += strength * 0.4 * d_i

                d_j = away_j - heading_xy[j]
                d_j = (d_j + np.pi) % (2.0 * np.pi) - np.pi
                corrections[j] += strength * 0.4 * d_j

    return corrections


def _schooling_forces(
    positions: np.ndarray,
    heading_xy: np.ndarray,
    config: SchoolingConfig,
) -> np.ndarray:
    """Compute cohesion + alignment social forces.

    Cohesion steers fish toward the centroid of visible neighbours.
    Alignment steers fish to match the mean heading of neighbours.

    Args:
        positions: shape (n_fish, 3)
        heading_xy: Current horizontal headings, shape (n_fish,).
        config: Schooling parameters.

    Returns:
        Delta heading_xy corrections from schooling, shape (n_fish,).
    """
    n_fish = positions.shape[0]
    corrections = np.zeros(n_fish, dtype=np.float64)

    if n_fish < 2 or (config.cohesion == 0.0 and config.alignment == 0.0):
        return corrections

    for i in range(n_fish):
        # Find neighbours within respective radii
        cohesion_neighbours = []
        alignment_neighbours = []

        for j in range(n_fish):
            if i == j:
                continue
            dist = float(np.linalg.norm(positions[i, :2] - positions[j, :2]))
            if dist < config.cohesion_radius:
                cohesion_neighbours.append(j)
            if dist < config.alignment_radius:
                alignment_neighbours.append(j)

        # Cohesion: steer toward neighbour centroid
        if cohesion_neighbours and config.cohesion > 0.0:
            centroid = np.mean(positions[cohesion_neighbours, :2], axis=0)
            diff = centroid - positions[i, :2]
            target_angle = np.arctan2(float(diff[1]), float(diff[0]))
            d = target_angle - heading_xy[i]
            d = (d + np.pi) % (2.0 * np.pi) - np.pi
            corrections[i] += config.cohesion * _SCHOOLING_FORCE_SCALE * d

        # Alignment: steer toward mean neighbour heading
        if alignment_neighbours and config.alignment > 0.0:
            mean_heading = np.mean(heading_xy[alignment_neighbours])
            d = mean_heading - heading_xy[i]
            d = (d + np.pi) % (2.0 * np.pi) - np.pi
            corrections[i] += config.alignment * _SCHOOLING_FORCE_SCALE * d

    return corrections


def generate_trajectories(config: TrajectoryConfig) -> TrajectoryResult:
    """Generate multi-fish 3D trajectories using a heading-based random walk.

    Simulates ``config.n_fish`` fish inside a cylindrical tank over
    ``config.duration_seconds`` seconds at ``config.fps`` frames per second.
    Each fish follows an AR(1) speed model and a heading model driven by
    Gaussian noise plus boundary repulsion, collision avoidance, and optional
    schooling forces.

    The result is deterministic given the same ``config.random_seed``.

    Args:
        config: Full simulation configuration.

    Returns:
        TrajectoryResult with state array of shape (n_frames, n_fish, 7),
        where the 7 state dimensions are
        [x, y, z, heading_xy, heading_z, speed, fish_id].
    """
    rng = np.random.default_rng(config.random_seed)
    motion = config.motion
    schooling = config.schooling
    tank = config.tank

    n_frames = round(config.duration_seconds * config.fps)
    dt = 1.0 / config.fps

    positions, heading_xy, heading_z, speeds = _init_positions(config.n_fish, tank, rng)

    # Update speed means to s_preferred
    speeds[:] = motion.s_preferred

    states = np.zeros((n_frames, config.n_fish, 7), dtype=np.float32)

    for frame_idx in range(n_frames):
        # --- Record current state ---
        states[frame_idx, :, 0] = positions[:, 0]
        states[frame_idx, :, 1] = positions[:, 1]
        states[frame_idx, :, 2] = positions[:, 2]
        states[frame_idx, :, 3] = heading_xy
        states[frame_idx, :, 4] = heading_z
        states[frame_idx, :, 5] = speeds
        states[frame_idx, :, 6] = np.arange(config.n_fish, dtype=np.float32)

        # --- Compute forces ---
        boundary = _boundary_force(positions, tank, heading_xy)
        collision = _collision_force(positions, heading_xy)
        social = _schooling_forces(positions, heading_xy, schooling)

        # --- Update headings ---
        noise_xy = rng.normal(0.0, motion.sigma_heading, size=config.n_fish)
        delta_xy = noise_xy + boundary + collision + social
        delta_xy = np.clip(delta_xy, -motion.max_turn_rate, motion.max_turn_rate)
        heading_xy = heading_xy + delta_xy
        # No wrapping needed at small sigma — but normalise to (-pi, pi)
        heading_xy = (heading_xy + np.pi) % (2.0 * np.pi) - np.pi

        noise_z = rng.normal(0.0, motion.sigma_pitch, size=config.n_fish)
        # Reversion: pitch reverts toward zero
        delta_z = noise_z - motion.pitch_reversion * heading_z
        heading_z = heading_z + delta_z
        heading_z = np.clip(heading_z, -motion.max_pitch, motion.max_pitch)

        # --- Update speed (AR(1) with mean reversion) ---
        noise_s = rng.normal(0.0, motion.sigma_speed, size=config.n_fish)
        speeds = (
            motion.speed_persistence * speeds
            + (1.0 - motion.speed_persistence) * motion.s_preferred
            + noise_s
        )
        speeds = np.clip(speeds, motion.s_min, motion.s_max)

        # --- Compute direction vector ---
        cos_z = np.cos(heading_z)
        dx = cos_z * np.cos(heading_xy)
        dy = cos_z * np.sin(heading_xy)
        dz = np.sin(heading_z)
        direction = np.stack([dx, dy, dz], axis=1)  # (n_fish, 3)

        # --- Update positions ---
        positions = positions + speeds[:, np.newaxis] * direction * dt

        # --- Hard clamp to tank volume (safety net) ---
        # Compute radial distance from tank centre (accounting for center offset)
        inner_r = tank.radius - tank.wall_margin
        cx_off = positions[:, 0] - tank.center_x
        cy_off = positions[:, 1] - tank.center_y
        r_xy = np.sqrt(cx_off**2 + cy_off**2)
        over_wall = r_xy > inner_r
        if np.any(over_wall):
            scale = inner_r / (r_xy[over_wall] + 1e-12)
            # Scale the offset from centre, then re-add centre
            positions[over_wall, 0] = cx_off[over_wall] * scale + tank.center_x
            positions[over_wall, 1] = cy_off[over_wall] * scale + tank.center_y

        z_min = tank.water_z + tank.wall_margin
        z_max = tank.water_z + tank.depth - tank.wall_margin
        positions[:, 2] = np.clip(positions[:, 2], z_min, z_max)

    return TrajectoryResult(
        n_fish=config.n_fish,
        n_frames=n_frames,
        fps=config.fps,
        states=states,
        config=config,
    )
