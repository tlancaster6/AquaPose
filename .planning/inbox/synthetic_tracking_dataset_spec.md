# Synthetic Dataset Specification: Multi-Camera Multi-Animal Tracking Evaluation

## 1. System Overview

### Purpose
This system generates synthetic datasets for evaluating a multi-camera, multi-animal (fish) 3D tracking pipeline. The tracking pipeline uses 2D object detection in each camera view followed by track-driven claiming, where existing 3D tracks are reprojected into each view and claim new detections based on reprojection proximity.

### Design Philosophy
The synthetic data generator operates in two modes:

1. **General scenario mode**: Configurable simulations with adjustable animal count, schooling behavior, and realistic noise injection. Used for broad performance characterization across operating conditions.
2. **Targeted scenario mode**: Short, 2-animal scenarios that isolate specific known failure modes. Used for diagnosing and benchmarking tracker behavior on hard cases.

Both modes share the same underlying pipeline: generate 3D trajectories → project to 2D camera coordinates → inject detection noise → output ground-truth-labeled synthetic detections.

### Integration Context
- The tracker under test has its own camera model API that handles refractive geometry through the water surface. The coding agent is familiar with this API.
- The synthetic data generator should produce output in a format consumable by the tracker's evaluation harness.
- Ground truth is comprehensive, enabling computation of standard MOT metrics (MOTA, MOTP, IDF1, ID switches, track fragmentation, etc.).

---

## 2. Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    SCENARIO CONFIGURATION                    │
│  (animal count, schooling params, duration, failure mode)    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   3D TRAJECTORY GENERATION                   │
│  Fish motion model: heading-based correlated random walk     │
│  with schooling forces, collision avoidance, and             │
│  physically-constrained depth changes                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   2D PROJECTION & GEOMETRY                   │
│  Project 3D centroids + fish body model into each camera     │
│  Compute: bounding boxes, occlusion maps, pairwise           │
│  distances in each view                                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   DETECTION NOISE MODEL                      │
│  Apply: missed detections, false positives, centroid         │
│  jitter, occlusion-dependent miss rates, velocity-           │
│  dependent degradation, detection coalescence                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                        OUTPUT                                │
│  Per-frame, per-camera: noisy detections + bounding boxes    │
│  Ground truth: 3D positions, true IDs, detection-to-GT      │
│  mapping, occlusion flags, noise annotations                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Core Components

### 3.1 Physical Environment

**Tank geometry:**
- Cylindrical volume, approximately 1m radius, 1m depth
- The tank is positioned approximately 1m below the camera array
- The coordinate system origin and axis conventions should match the existing tracker's coordinate system
- Fish are confined to the interior of the cylinder with a small margin (e.g., 5cm from walls) to avoid edge artifacts

**Fish body model:**
- Physical length: ~8cm, standard elongated fish body geometry
- For 2D bounding box computation: model the fish as a 3D oriented ellipsoid (semi-axes approximately 4cm × 1.5cm × 1cm for length × width × height)
- The ellipsoid's long axis is aligned with the fish's heading direction
- When projected into each camera view, the ellipsoid produces a 2D bounding ellipse; the non-oriented bounding box is the axis-aligned rectangle enclosing this ellipse
- This means bounding box aspect ratio varies naturally with the fish's orientation relative to each camera

**Temporal parameters:**
- Framerate: 30 fps, fully synchronized across all cameras
- All cameras observe every fish at every timestep (though detections may be dropped by the noise model)

### 3.2 Fish Motion Model

The motion model generates physically plausible 3D fish trajectories. It uses a heading-based correlated random walk with the following state per fish:

**State vector:**
```
position: (x, y, z)        — 3D position in tank coordinates
heading: (θ_xy, θ_z)       — horizontal heading angle, vertical pitch angle
speed: s                    — scalar speed along heading direction
```

**Dynamics (per timestep, dt = 1/30 s):**

```
# Heading update (autoregressive with noise)
θ_xy(t+1) = θ_xy(t) + Δθ_xy(t)
θ_z(t+1)  = θ_z(t)  + Δθ_z(t)

# Heading change is drawn from a wrapped normal distribution
Δθ_xy ~ WrappedNormal(μ_heading, σ_heading)
Δθ_z  ~ WrappedNormal(μ_pitch, σ_pitch)

# Speed update (autoregressive, clamped)
s(t+1) = clamp(s(t) + Δs(t), s_min, s_max)
Δs ~ Normal(0, σ_speed)

# Position update
velocity = s * [cos(θ_xy)cos(θ_z), sin(θ_xy)cos(θ_z), sin(θ_z)]
position(t+1) = position(t) + velocity * dt
```

**Key constraints and parameters:**

| Parameter | Description | Suggested Default | Notes |
|-----------|-------------|-------------------|-------|
| `s_min` | Minimum speed | 0.01 m/s | Fish are rarely fully stationary |
| `s_max` | Maximum speed | 0.5 m/s | ~6 body lengths/sec, reasonable burst |
| `s_preferred` | Preferred cruising speed | 0.1 m/s | Mean-reversion target |
| `σ_speed` | Speed change noise | 0.02 m/s per step | |
| `speed_persistence` | AR(1) coefficient for speed | 0.95 | High persistence = smooth speed changes |
| `σ_heading` | Heading change noise | 0.05 rad/step | ~3°/step max typical turn |
| `max_turn_rate` | Hard limit on heading change | 0.3 rad/step | ~17°/step, allows sharp turns |
| `σ_pitch` | Pitch change noise | 0.01 rad/step | Much smaller than heading noise |
| `max_pitch` | Maximum pitch angle | 0.25 rad | ~15° from horizontal |
| `pitch_reversion` | Tendency to return to level | 0.8 | Strong mean-reversion to θ_z = 0 |

**Depth-change constraint:** Fish change depth by tilting and swimming forward. This is enforced by:
1. The pitch angle `θ_z` is bounded by `max_pitch` (fish can only tilt so far)
2. The pitch noise `σ_pitch` is much smaller than `σ_heading`
3. Vertical velocity is always `s * sin(θ_z)` — a fish cannot move vertically without forward speed
4. Strong mean-reversion on pitch: `μ_pitch = -pitch_reversion * θ_z(t)` pulls the fish back to level swimming

**Boundary handling:** When a fish approaches the tank boundary (cylinder walls, top, bottom), apply a soft repulsive force to the heading that steers it away. This should be a smooth function that activates within a margin of the boundary (e.g., 10cm) and grows stronger as the fish gets closer. Do not use hard reflection/teleportation.

**Collision avoidance:** Fish cannot co-occupy the same 3D location. Apply a short-range repulsive force between all fish pairs when their centroids are closer than a minimum distance (e.g., 2 body lengths = 16cm). This force modifies the heading, not the speed, pushing fish to steer around each other.

**Speed mean-reversion:** Add a drift term to speed: `μ_speed = -speed_reversion_rate * (s(t) - s_preferred)` so fish tend to cruise at a preferred speed but can accelerate/decelerate.

### 3.3 Schooling / Flocking Model

Schooling behavior is controlled by two parameters:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `cohesion` | [0, 1] | Tendency to move toward the centroid of nearby fish |
| `alignment` | [0, 1] | Tendency to match heading/velocity of nearby fish |

These are implemented as additional forces on the heading and speed updates, following a Boids-like model:

**Cohesion force:** Steers a fish's heading toward the centroid of all fish within a `cohesion_radius` (e.g., 30cm). The force magnitude scales with `cohesion` parameter.

**Alignment force:** Steers a fish's heading toward the average heading of all fish within an `alignment_radius` (e.g., 30cm). The force magnitude scales with `alignment` parameter. Also applies a speed-matching force toward the average speed of neighbors.

**Interaction radius parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cohesion_radius` | 0.3 m | Range for cohesion force |
| `alignment_radius` | 0.3 m | Range for alignment force |

**Combined heading update with all forces:**
```
Δθ_xy = noise_term + boundary_force + collision_force + cohesion_force + alignment_force
```

Each force contributes additively to the heading change, with appropriate scaling so that forces don't overwhelm the noise term at moderate parameter values. At `cohesion=0, alignment=0`, fish move independently. At `cohesion=1, alignment=1`, fish school tightly with coordinated headings.

**Presets for convenience:**
- `independent`: cohesion=0, alignment=0
- `loose_school`: cohesion=0.3, alignment=0.3
- `tight_school`: cohesion=0.7, alignment=0.7
- `milling`: cohesion=0.7, alignment=0.1 (cluster together but don't align headings)
- `streaming`: cohesion=0.2, alignment=0.8 (parallel movement without tight clustering)

### 3.4 Camera Projection & Occlusion

**Projection:** Use the existing camera model API to project 3D fish positions (centroids) and body ellipsoids into each camera view. The API handles the refractive geometry.

**Bounding box computation:** For each fish in each camera:
1. Project the 3D oriented ellipsoid (aligned with fish heading) into the camera view
2. Compute the axis-aligned bounding box enclosing the projected ellipsoid
3. Store the bounding box center (centroid) and dimensions (width, height)

**Occlusion computation:** For each camera view, for each pair of fish:
1. Compute the 2D distance between bounding box centers
2. Compute the intersection-over-union (IoU) of the two bounding boxes
3. Compute the intersection-over-smaller-box ratio (IoS) — this better captures when a small/distant fish is hidden behind a larger/closer one
4. A fish is considered **partially occluded** when IoU > 0 (bounding boxes overlap)
5. A fish is considered **heavily occluded** when IoS > 0.5 (more than half of the smaller box is covered)
6. Use depth ordering (distance from camera) to determine which fish is the occluder vs. the occluded

Store per-frame, per-camera, per-fish:
- `occlusion_level`: float in [0, 1] representing the fraction of the fish's bounding box occluded by other fish
- `nearest_neighbor_2d_distance`: distance to the nearest other fish's centroid in this view (in pixels)
- `nearest_neighbor_3d_distance`: distance to nearest other fish in 3D (in meters)

### 3.5 Detection Noise Model

The noise model transforms perfect 2D projections into realistic noisy detections. It operates independently per camera, per frame.

**Baseline detection parameters (derived from real model performance, F1 ≈ 0.94):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_miss_rate` | 0.06 | Probability of missing a detection (no occlusion, low velocity) |
| `base_false_positive_rate` | 0.06 | Expected false positives per frame per camera, as a fraction of true fish count |
| `centroid_noise_std` | 3.0 px | Gaussian noise on detected centroid position |
| `bbox_noise_std` | 2.0 px | Gaussian noise on bounding box width/height |

**Conditional modifiers:**

1. **Occlusion-dependent miss rate:**
   ```
   miss_rate = base_miss_rate + occlusion_miss_bonus * occlusion_level
   ```
   Where `occlusion_miss_bonus` ≈ 0.5. A fully occluded fish has ~56% miss rate. A partially occluded fish has elevated miss rate proportional to occlusion.

2. **Occlusion-dependent centroid shift:**
   When a fish is partially occluded (`occlusion_level > 0`), its detected centroid is biased toward the occluding fish's centroid:
   ```
   shifted_centroid = true_centroid + centroid_shift_strength * occlusion_level * (occluder_centroid - true_centroid)
   ```
   Where `centroid_shift_strength` ≈ 0.3. This simulates the detector centering on the combined blob rather than the individual fish.

3. **Velocity-dependent miss rate:**
   ```
   velocity_miss_bonus = velocity_miss_scale * clamp(speed / speed_threshold - 1, 0, 1)
   ```
   Where `velocity_miss_scale` ≈ 0.15 and `speed_threshold` ≈ 0.3 m/s. Fast-moving fish have elevated miss rates due to motion blur.

4. **Velocity-dependent centroid noise:**
   ```
   effective_noise_std = centroid_noise_std * (1 + velocity_noise_scale * speed / s_max)
   ```
   Where `velocity_noise_scale` ≈ 0.5. Fast-moving fish have noisier position estimates.

5. **Detection coalescence (merged detections):**
   When two fish bounding boxes have IoU > `coalescence_iou_threshold` (≈ 0.3) in a camera view:
   ```
   coalescence_probability = coalescence_base_rate * (IoU / coalescence_iou_threshold)
   ```
   Where `coalescence_base_rate` ≈ 0.3. When coalescence occurs:
   - Both individual detections are removed
   - A single merged detection is placed at the IoU-weighted average of the two centroids
   - The merged bounding box is the union of the two original boxes
   - Ground truth records this as a coalescence event, linking the merged detection to both true fish

6. **False positives:**
   - Count drawn from `Poisson(base_false_positive_rate * n_fish)` per camera per frame
   - Position: uniformly sampled within the camera's field of view of the tank volume
   - Bounding box size: drawn from the empirical distribution of true detection sizes (so FPs look like real fish)
   - False positives are spatially independent (no persistent FP locations, as stated)

**Combined miss probability:**
```
final_miss_prob = 1 - (1 - base_miss_rate) * (1 - occlusion_miss_component) * (1 - velocity_miss_component)
```
This ensures miss probabilities combine naturally without exceeding 1.

**All noise parameters should be adjustable** to allow testing at different difficulty levels.

### 3.6 Output Format

The generator produces a single dataset object (or serialized file) containing:

**Per-frame, per-camera detection list:**
```python
@dataclass
class SyntheticDetection:
    frame: int
    camera_id: int
    bbox_center: Tuple[float, float]     # (u, v) in pixels, with noise applied
    bbox_size: Tuple[float, float]       # (width, height) in pixels, with noise applied
    # Ground truth linkage
    true_fish_id: Optional[int]          # None for false positives
    is_false_positive: bool
    is_coalesced: bool                   # True if this is a merged detection
    coalesced_fish_ids: Optional[List[int]]  # If coalesced, which fish were merged
    # Noise annotations
    centroid_noise_applied: Tuple[float, float]  # The actual noise vector added
    centroid_shift_applied: Tuple[float, float]  # Occlusion-induced shift vector
```

**Per-frame, per-camera miss list:**
```python
@dataclass
class MissedDetection:
    frame: int
    camera_id: int
    true_fish_id: int
    true_bbox_center: Tuple[float, float]   # Where the detection should have been
    true_bbox_size: Tuple[float, float]
    miss_reason: str                         # 'baseline', 'occlusion', 'velocity', 'coalescence'
    occlusion_level: float
    speed: float
```

**Per-frame ground truth:**
```python
@dataclass
class GroundTruthFrame:
    frame: int
    fish_states: List[FishState]

@dataclass
class FishState:
    fish_id: int
    position_3d: Tuple[float, float, float]   # True 3D position
    velocity_3d: Tuple[float, float, float]   # True 3D velocity
    heading: Tuple[float, float]               # (θ_xy, θ_z)
    speed: float
    # Per-camera visibility info
    per_camera: Dict[int, CameraVisibility]

@dataclass
class CameraVisibility:
    camera_id: int
    true_bbox_center: Tuple[float, float]     # Perfect projection, no noise
    true_bbox_size: Tuple[float, float]
    occlusion_level: float
    nearest_neighbor_2d_dist: float
    was_detected: bool
    detection_index: Optional[int]             # Index into detection list, if detected
```

**Scenario metadata:**
```python
@dataclass
class ScenarioMetadata:
    scenario_type: str                    # 'general' or name of targeted scenario
    n_fish: int
    n_frames: int
    n_cameras: int
    framerate: float
    # Motion model parameters
    motion_params: Dict[str, float]
    # Schooling parameters
    cohesion: float
    alignment: float
    # Noise model parameters
    noise_params: Dict[str, float]
    # For targeted scenarios
    failure_mode: Optional[str]
    difficulty: Optional[float]
    random_seed: int
```

**Implementation note:** Use PyTorch tensors for batch computation where beneficial (projection, noise sampling), but the output data structures can use standard Python dataclasses or dicts. The output should be serializable (e.g., to a dict of tensors or a pickle file).

---

## 4. General Scenario Configuration

The general scenario generator accepts a configuration object and produces a complete synthetic dataset.

**Configuration parameters:**

```python
@dataclass
class GeneralScenarioConfig:
    # Core
    n_fish: int = 5
    duration_seconds: float = 30.0
    random_seed: int = 42

    # Motion model
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

    # Schooling
    cohesion: float = 0.0
    alignment: float = 0.0
    cohesion_radius: float = 0.3
    alignment_radius: float = 0.3

    # Noise model
    base_miss_rate: float = 0.06
    base_false_positive_rate: float = 0.06
    centroid_noise_std: float = 3.0
    bbox_noise_std: float = 2.0
    occlusion_miss_bonus: float = 0.5
    centroid_shift_strength: float = 0.3
    velocity_miss_scale: float = 0.15
    speed_threshold: float = 0.3
    velocity_noise_scale: float = 0.5
    coalescence_iou_threshold: float = 0.3
    coalescence_base_rate: float = 0.3

    # Tank geometry
    tank_radius: float = 1.0
    tank_depth: float = 1.0
    wall_margin: float = 0.05
```

**Usage pattern:**
```python
config = GeneralScenarioConfig(n_fish=10, cohesion=0.5, alignment=0.5, duration_seconds=60.0)
dataset = generate_general_scenario(config, camera_model=my_camera_model)
```

**Recommended evaluation matrix** (run as a batch):
- Fish counts: [2, 5, 10, 20]
- Schooling presets: [independent, loose_school, tight_school, milling, streaming]
- Noise levels: [low (halved miss/FP rates), nominal, high (doubled miss/FP rates)]
- Duration: 30s per run (900 frames)
- 3 random seeds per configuration

---

## 5. Targeted Scenario Definitions

All targeted scenarios are 2-fish, short-duration simulations designed to isolate specific failure modes. Each produces the same output format as general scenarios but with additional metadata about the failure mode being tested.

### 5.1 Schooling Without Occlusion

**What it tests:** Track identity maintenance when two fish swim in close proximity with similar velocity vectors, but without significant occlusion events. Tests whether the tracker can maintain distinct IDs based on small spatial differences when velocity cues are similar.

**Setup:**
- 2 fish swim with high alignment and moderate cohesion
- **Z-depth constraint:** Both fish are initialized at the same z-depth and the pitch angle is locked to 0 (or near-0), forcing them to remain at the same depth throughout. This prevents one fish from passing in front of the other in any camera view.
- Inter-fish distance is maintained in the range [10cm, 25cm] (close but not overlapping bounding boxes in most views)
- Duration: 10 seconds (300 frames)

**Parameters:**
- `alignment`: 0.9
- `cohesion`: 0.5
- `sigma_pitch`: 0 (locked to level)
- Noise model: nominal

**Difficulty scaling:**
- Easy: inter-fish distance 20-25cm, moderate speed
- Medium: inter-fish distance 10-15cm, moderate speed
- Hard: inter-fish distance 10-15cm, high speed (more centroid noise)

### 5.2 Schooling With Occlusion

**What it tests:** Same as above, but now fish can move to different depths, producing partial and full occlusion events in some camera views. Tests ID maintenance through occlusion during coordinated movement.

**Setup:**
- Same as 5.1 but with pitch unlocked — fish can change depth normally
- Because they're schooling at close range, one will frequently pass in front of the other from various camera angles
- Duration: 15 seconds (450 frames)

**Parameters:**
- `alignment`: 0.9
- `cohesion`: 0.6
- `sigma_pitch`: normal (0.01)
- Noise model: nominal

**Difficulty scaling:**
- Easy: larger inter-fish distance (15-25cm), lower occlusion probability
- Hard: smaller inter-fish distance (8-15cm), more frequent/prolonged occlusion

### 5.3 Non-Schooling Occlusion (Crossing Paths)

**What it tests:** Identity maintenance when two independently-moving fish cross paths, creating a brief occlusion event. This is the classic "identity switch" scenario.

**Setup:**
- 2 fish start separated (>40cm apart)
- Fish trajectories are designed so they converge, pass through close proximity / occlusion, then diverge
- Duration: 10 seconds (300 frames), with the crossing event occurring around the midpoint

**Difficulty scaling** (continuous parameter `difficulty` in [0, 1]):

| Difficulty | Approach angle | Velocity profile | Special conditions |
|-----------|---------------|------------------|-------------------|
| 0.0 (easy) | Orthogonal (90°) | Constant speed, straight-line | Maximally different velocity vectors at crossing |
| 0.3 | ~60° approach | Constant speed, straight-line | |
| 0.5 | ~45° approach | Mild acceleration | |
| 0.7 | ~30° approach | Acceleration + angular velocity | Velocity vectors more similar at crossing |
| 0.9 | Near-parallel (<20°) | Acceleration + angular velocity | Very similar velocity vectors at crossing |
| 1.0 (hard) | Near-parallel, one fish stationary | Moving fish passes close to stationary fish | Stationary fish has no velocity prior to disambiguate |

**Implementation approach:** Rather than scripting exact trajectories, use waypoint-based motion:
- Define entry points, a crossing zone, and exit points
- The motion model steers fish toward waypoints while maintaining smooth, physically-realistic trajectories
- The crossing geometry (approach angle, velocities at crossing) is derived from waypoint placement
- This preserves realistic motion characteristics while controlling the scenario geometry

**Variants to generate:**
- At each difficulty level, generate multiple random seeds
- Include variants where fish cross at different depths (some views show occlusion, others don't)

### 5.4 Detection Coalescence / Track Merge

**What it tests:** Tracker behavior when two fish in close proximity produce a single merged detection rather than two individual detections. The tracker sees N-1 detections with one at an incorrect position.

**Setup:**
- 2 fish approach each other and enter close proximity (bounding box IoU > 0.3)
- During the close-proximity phase, the coalescence rate is artificially elevated to guarantee at least several coalescence events
- After the close-proximity phase, fish separate
- Duration: 10 seconds (300 frames)

**Parameters:**
- Override `coalescence_base_rate` to 0.8 during the close-proximity phase (guarantee coalescence events)
- Normal noise parameters otherwise

**Difficulty scaling:**
- Easy: short coalescence duration (5-10 frames), fish exit with different velocities
- Hard: long coalescence duration (30-60 frames), fish exit with similar velocities

### 5.5 Track Fragmentation

**What it tests:** Whether the tracker correctly bridges gaps in detections for a single animal. No occlusion — just elevated baseline miss rate causing repeated brief detection losses.

**Setup:**
- 2 fish moving independently, well-separated (no occlusion)
- The `base_miss_rate` is elevated to create frequent detection gaps
- Duration: 15 seconds (450 frames)

**Parameters:**
- `base_miss_rate`: 0.20 (elevated from nominal 0.06)
- No occlusion-related noise (fish stay far apart)
- Normal centroid noise

**Difficulty scaling:**
- Easy: miss rate 0.15, short gaps (1-2 frames)
- Medium: miss rate 0.25, gaps up to 5 frames
- Hard: miss rate 0.35, gaps up to 10 frames

**Note:** The miss rate controls gap frequency. To control gap *duration*, inject correlated (bursty) misses: when a detection is missed, the miss probability for the next frame is temporarily elevated (e.g., `burst_miss_continuation` ≈ 0.5). This creates realistic multi-frame gaps rather than independent per-frame misses.

### 5.6 Re-Identification After Long Occlusion

**What it tests:** Whether the tracker can correctly re-associate tracks after an extended full occlusion event where both fish have moved significantly.

**Setup:**
- 2 fish start separated, converge into full occlusion that lasts for an extended period, then separate
- During occlusion, the occluded fish continues moving (changing position significantly)
- Duration: 15 seconds (450 frames)

**Two sub-scenarios:**

**5.6a — Divergent exit:** Fish exit the occlusion moving in clearly different directions. Easier because velocity vectors at re-emergence are discriminative.
- Post-occlusion heading difference: > 60°

**5.6b — Parallel exit:** Fish exit the occlusion moving in similar directions. Harder because velocity cues don't help disambiguate.
- Post-occlusion heading difference: < 20°

**Difficulty scaling:**
- Easy: 15-frame occlusion (~0.5s), divergent exit, moderate displacement
- Medium: 30-frame occlusion (~1s), mixed exit angles
- Hard: 60+ frame occlusion (~2s), parallel exit, large displacement

### 5.7 Velocity Estimation Corruption

**What it tests:** Tracker recovery after a brief mis-association corrupts the velocity estimate for a track. Tests whether the tracker can recover from corrupted predictions without cascading errors.

**Setup:**
- 2 fish moving on distinct but nearby parallel paths
- For 1-3 frames, force a detection swap: fish A's detection is positioned closer to track B's predicted position, and vice versa. This is achieved by applying a large directional centroid shift toward the other fish for those frames.
- After the swap frames, detections return to normal
- Duration: 10 seconds (300 frames), with the forced swap occurring around frame 100

**Parameters:**
- `forced_swap_frames`: [100, 101] (or [100, 101, 102] for harder version)
- During swap frames: override centroid positions to be swapped between the two fish
- Normal noise otherwise

**Evaluation focus:** Track the velocity estimate error over time. A well-behaved tracker should show a spike in velocity error at the swap, followed by recovery within a few frames. A fragile tracker will show cascading ID switches.

### 5.8 Synchronized Direction Change (Startle Response)

**What it tests:** Tracker behavior when both fish abruptly change heading simultaneously, breaking the assumption that short-term velocity prediction is reliable.

**Setup:**
- 2 fish swimming with moderate separation and steady velocity
- At a specific frame, both fish execute a sharp heading change (e.g., 90-180° turn)
- Post-turn, fish continue on new headings
- Duration: 10 seconds (300 frames), startle at frame 150

**Parameters:**
- Pre-startle: steady swimming, low heading noise
- Startle: instantaneous heading change of 90-180° (sampled per fish)
- Post-startle: normal motion model resumes

**Difficulty scaling:**
- Easy: fish are well-separated (>30cm), turn in different directions
- Medium: fish are moderately close (~15-20cm), turn in similar directions
- Hard: fish are close (<15cm), turn in same direction by similar amounts (hard to distinguish)

### 5.9 Depth Ambiguity in Cross-View Claiming

**What it tests:** Whether the tracker correctly handles two fish at different depths but similar XY positions. These fish overlap heavily in some camera views but are separated in others.

**Setup:**
- 2 fish at different z-depths (e.g., one at z=-0.3m, one at z=-0.7m) but similar XY positions (within ~5cm in XY)
- Fish maintain this depth-separated, XY-proximate arrangement for a sustained period
- Some camera views show severe overlap; others (especially oblique angles) show separation
- Duration: 10 seconds (300 frames)

**Parameters:**
- Depth separation: configurable (0.2m - 0.6m)
- XY proximity: configurable (0 - 10cm)
- Fish move slowly in XY to maintain proximity
- Drop some cameras' detections via elevated miss rate to test tracker robustness with incomplete multi-view evidence

**Difficulty scaling:**
- Easy: large depth separation (0.5m), moderate XY offset (8cm), all cameras detect
- Medium: moderate depth separation (0.3m), small XY offset (3cm), 1-2 cameras miss
- Hard: small depth separation (0.2m), minimal XY offset (<2cm), 2-3 cameras miss

### 5.10 Partial Occlusion Centroid Shift

**What it tests:** Tracker behavior when detections exist but are spatially biased due to partial occlusion. The detector fires, but the centroid is shifted toward the occluder, corrupting position estimates without dropping the detection entirely.

**Setup:**
- 2 fish swim on slowly converging paths, producing sustained partial occlusion (IoU between 0.1 and 0.4) in some camera views
- The centroid shift model produces detections that are consistently biased toward the other fish
- No missed detections — the challenge is purely from biased positions
- Duration: 10 seconds (300 frames)

**Parameters:**
- Override `centroid_shift_strength` to 0.5 (elevated from 0.3)
- Override `occlusion_miss_bonus` to 0.0 (no extra misses, only centroid shift)
- Fish maintain partial occlusion for 100+ frames

**Evaluation focus:** Measure the tracker's 3D position error during the partial occlusion phase. A robust tracker should recognize and downweight biased detections from partially-occluded views.

---

## 6. Constraints & Assumptions

1. **Camera model:** The existing camera API handles all projection, including refractive geometry. The synthetic data generator calls this API and does not re-implement projection.
2. **Coordinate system:** Matches the existing tracker's conventions (the coding agent knows these).
3. **Determinism:** All scenarios are reproducible given the same random seed.
4. **Performance:** The generator does not need to run in real-time, but should be efficient enough to generate a 30-second, 10-fish scenario in under a minute on a standard workstation.
5. **Extensibility:** The architecture should make it straightforward to add new targeted scenarios or new noise model components.
6. **Backend:** Python with PyTorch for batch computation (projection, noise sampling, force computation). Standard Python for data structures and I/O.

---

## 7. Implementation Notes

- All scenarios must be fully reproducible given the same random seed.
- Targeted scenarios should reuse the core motion model, not reimplement it. The difference is in initialization, constraints, and any waypoint/force overrides specific to the scenario.
- The coding agent should determine module structure, code organization, and integration approach based on the existing codebase.
