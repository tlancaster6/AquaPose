---
phase: quick-6
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/synthetic/trajectory.py
  - src/aquapose/synthetic/detection.py
  - src/aquapose/synthetic/scenarios.py
  - src/aquapose/synthetic/stubs.py
  - src/aquapose/synthetic/__init__.py
  - tests/unit/synthetic/test_trajectory.py
  - tests/unit/synthetic/test_detection_gen.py
  - tests/unit/synthetic/test_scenarios.py
autonomous: true
requirements: []

must_haves:
  truths:
    - "Trajectory generator produces physically plausible multi-fish 3D paths confined to tank volume"
    - "Detection generator projects 3D trajectories to per-camera Detection objects with configurable noise"
    - "Scenario presets produce ready-to-use datasets for tracker evaluation (crossing, schooling, fragmentation)"
    - "Output format matches FishTracker.update() input: dict[str, list[Detection]] per frame"
    - "All scenarios are deterministic given same random seed"
  artifacts:
    - path: "src/aquapose/synthetic/trajectory.py"
      provides: "3D fish trajectory generation with motion model, boundary handling, collision avoidance, optional schooling"
      min_lines: 150
    - path: "src/aquapose/synthetic/detection.py"
      provides: "Project trajectories to per-camera Detection objects with noise model (miss, jitter, false positives)"
      min_lines: 100
    - path: "src/aquapose/synthetic/scenarios.py"
      provides: "Scenario configs and presets for targeted tracker evaluation"
      min_lines: 80
    - path: "tests/unit/synthetic/test_trajectory.py"
      provides: "Unit tests for trajectory generation"
      min_lines: 60
    - path: "tests/unit/synthetic/test_detection_gen.py"
      provides: "Unit tests for detection generation"
      min_lines: 40
    - path: "tests/unit/synthetic/test_scenarios.py"
      provides: "Unit tests for scenario presets"
      min_lines: 30
  key_links:
    - from: "src/aquapose/synthetic/trajectory.py"
      to: "src/aquapose/synthetic/detection.py"
      via: "TrajectoryResult consumed by generate_synthetic_detections"
      pattern: "TrajectoryResult"
    - from: "src/aquapose/synthetic/detection.py"
      to: "src/aquapose/segmentation/detector.py"
      via: "Produces Detection dataclass instances"
      pattern: "Detection\\(bbox="
    - from: "src/aquapose/synthetic/detection.py"
      to: "src/aquapose/calibration/projection.py"
      via: "Uses RefractiveProjectionModel.project for 3D-to-2D"
      pattern: "model\\.project"
    - from: "src/aquapose/synthetic/scenarios.py"
      to: "src/aquapose/synthetic/trajectory.py"
      via: "Scenario presets configure TrajectoryConfig"
      pattern: "TrajectoryConfig"
---

<objective>
Create a synthetic data generation system for evaluating the Cross-View Identity and 3D Tracking pipeline. The system generates multi-fish 3D trajectories with physically plausible motion, projects them through refractive cameras to produce Detection objects with configurable noise, and provides scenario presets targeting known tracker failure modes.

Purpose: Enable controlled, reproducible testing of FishTracker without real video data. Currently the synthetic module only generates static/kinematic fish shapes for midline testing (quick-2/4). This extends it to produce multi-frame tracking-compatible detection streams.

Output: Three new modules in src/aquapose/synthetic/ (trajectory.py, detection.py, scenarios.py), updated __init__.py, replaced stubs.py, and comprehensive unit tests.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@src/aquapose/synthetic/__init__.py
@src/aquapose/synthetic/fish.py
@src/aquapose/synthetic/rig.py
@src/aquapose/synthetic/stubs.py
@src/aquapose/segmentation/detector.py (Detection dataclass — the target output format)
@src/aquapose/tracking/tracker.py (FishTracker, FishTrack — consumers of synthetic detections)
@src/aquapose/tracking/associate.py (claim_detections_for_tracks, discover_births — input format)
@src/aquapose/calibration/projection.py (RefractiveProjectionModel.project — used for 3D-to-2D)
@tests/unit/synthetic/test_synthetic.py (existing test patterns)
@.planning/inbox/synthetic_tracking_dataset_spec.md (detailed reference spec — adapt for ROI)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Trajectory generator — heading-based random walk with forces</name>
  <files>
    src/aquapose/synthetic/trajectory.py
    tests/unit/synthetic/test_trajectory.py
  </files>
  <action>
Create `src/aquapose/synthetic/trajectory.py` implementing the 3D fish trajectory generator.

**Core dataclasses:**

- `MotionConfig` — all motion parameters with sensible defaults from the spec (s_min=0.01, s_max=0.5, s_preferred=0.1, sigma_speed=0.02, speed_persistence=0.95, sigma_heading=0.05, max_turn_rate=0.3, sigma_pitch=0.01, max_pitch=0.25, pitch_reversion=0.8).
- `SchoolingConfig` — cohesion, alignment, cohesion_radius, alignment_radius. Include preset classmethod helpers: `independent()`, `loose_school()`, `tight_school()`, `milling()`, `streaming()` with values from spec section 3.3.
- `TankConfig` — radius=1.0, depth=1.0, wall_margin=0.05, water_z=0.75 (matching build_fabricated_rig convention: cameras at Z=0, water at Z=water_z, fish at Z > water_z).
- `TrajectoryConfig` — n_fish, duration_seconds=10.0, fps=30.0, random_seed=42, motion: MotionConfig, schooling: SchoolingConfig, tank: TankConfig. Use field(default_factory=...) for mutable defaults.
- `FishTrajectoryState` — per-fish per-frame state: position (3,), heading_xy (float), heading_z (float), speed (float).
- `TrajectoryResult` — n_fish, n_frames, fps, states: np.ndarray of shape (n_frames, n_fish, 7) where dim 7 = [x, y, z, heading_xy, heading_z, speed, fish_id]. Also store config for reproducibility.

**Core function `generate_trajectories(config: TrajectoryConfig) -> TrajectoryResult`:**

1. Seed RNG with `np.random.default_rng(config.random_seed)`.
2. Initialize n_fish positions uniformly inside cylinder (x^2+y^2 < (radius-margin)^2, z between water_z and water_z+depth-margin). Random headings, speed=s_preferred.
3. Per timestep (dt = 1/fps):
   a. Compute forces on heading: boundary_force (soft repulsion within 10cm of walls, linear ramp), collision_force (repulse when within 2*body_length=16cm, modify heading only), cohesion_force (steer toward neighbor centroid within radius), alignment_force (match neighbor heading within radius).
   b. Update heading: theta_xy += noise + boundary + collision + cohesion + alignment, clamped by max_turn_rate. theta_z += noise + pitch_reversion_term, clamped by max_pitch.
   c. Update speed: AR(1) with mean-reversion to s_preferred, clamped [s_min, s_max].
   d. Update position: pos += speed * direction_vector * dt.
   e. Hard clamp position inside tank volume as final safety net.
4. Store all states in the result array.

**Key implementation notes:**
- Use vectorized numpy operations over all fish where possible (forces are pairwise but n_fish is small, <20).
- Heading noise is Gaussian, not wrapped normal — at the small sigma values used, wrapping is negligible.
- Body length constant: FISH_BODY_LENGTH = 0.085 (matching FishConfig.scale default).
- Schooling force scaling: multiply cohesion/alignment by a scale factor (0.1) so forces don't overwhelm noise at moderate parameter values.

**Tests in `tests/unit/synthetic/test_trajectory.py`:**

- `test_trajectory_shape`: Default config produces correct shape (n_frames, n_fish, 7).
- `test_trajectory_deterministic`: Same seed = identical output.
- `test_trajectory_stays_in_tank`: All positions within tank cylinder bounds.
- `test_trajectory_speed_bounds`: All speeds within [s_min, s_max].
- `test_trajectory_collision_avoidance`: With 5 fish in small tank, no pair closer than 0.5*FISH_BODY_LENGTH at any frame.
- `test_schooling_cohesion`: With tight_school preset and 5 fish, mean inter-fish distance is smaller than with independent preset (run both, compare).
- `test_single_fish_no_crash`: n_fish=1 runs without errors (no pairwise force issues).
  </action>
  <verify>
`hatch run pytest tests/unit/synthetic/test_trajectory.py -v` — all tests pass.
`hatch run check` — no lint or type errors in trajectory.py.
  </verify>
  <done>
TrajectoryConfig/TrajectoryResult dataclasses exist. generate_trajectories() produces multi-fish 3D paths that stay in tank, respect speed bounds, avoid collisions, and respond to schooling parameters. 7 unit tests pass.
  </done>
</task>

<task type="auto">
  <name>Task 2: Detection generator — project trajectories to noisy Detection objects</name>
  <files>
    src/aquapose/synthetic/detection.py
    src/aquapose/synthetic/stubs.py
    src/aquapose/synthetic/__init__.py
    tests/unit/synthetic/test_detection_gen.py
  </files>
  <action>
Create `src/aquapose/synthetic/detection.py` implementing the 2D detection generator.

**Core dataclasses:**

- `NoiseConfig` — all noise parameters with spec defaults: base_miss_rate=0.06, base_false_positive_rate=0.06, centroid_noise_std=3.0, bbox_noise_std=2.0, velocity_miss_scale=0.15, speed_threshold=0.3, velocity_noise_scale=0.5. OMIT occlusion-dependent and coalescence noise for now (noted as future extension in docstring) — these require bounding box overlap computation which adds significant complexity for marginal ROI at this stage.
- `DetectionGenConfig` — noise: NoiseConfig, fish_bbox_size: tuple[float, float] = (40, 25) pixels (approximate bbox size for an 8cm fish at typical depth). This is a simplification vs the spec's ellipsoid projection — adequate for tracker testing where exact bbox shape is not the variable under test.
- `SyntheticFrame` — frame_index: int, detections_per_camera: dict[str, list[Detection]], ground_truth: dict mapping camera_id to list of (fish_id, true_centroid_px, was_detected).
- `SyntheticDataset` — frames: list[SyntheticFrame], metadata: dict (config, n_fish, n_frames, seed).

**Core function `generate_detection_dataset(trajectory: TrajectoryResult, models: dict[str, RefractiveProjectionModel], noise_config: NoiseConfig | None = None, det_config: DetectionGenConfig | None = None, random_seed: int | None = None) -> SyntheticDataset`:**

1. Seed RNG.
2. For each frame, for each camera:
   a. Project each fish's 3D position through the RefractiveProjectionModel. Use `model.project(pts_torch)` where pts_torch is shape (n_fish, 3). Get pixel positions and validity flags.
   b. For each valid fish projection:
      - Compute miss probability: base_miss_rate + velocity_miss_component (from fish speed).
      - Draw miss/detect from Bernoulli(1 - miss_prob).
      - If detected: apply Gaussian centroid jitter (centroid_noise_std scaled by velocity factor), bbox size jitter (bbox_noise_std).
      - Create `Detection(bbox=(x-w//2, y-h//2, w, h), mask=None, area=w*h, confidence=1.0)`.
   c. Generate false positives: Poisson(base_fp_rate * n_fish) count, random positions within image bounds (use K matrix to determine image size: W=round(2*cx), H=round(2*cy) per Phase 04 convention).
   d. Record ground truth linkage.
3. Return SyntheticDataset.

**Replace stubs.py:** Update the stub functions in stubs.py to call the new implementation. `generate_synthetic_detections` should delegate to `generate_detection_dataset`. `generate_synthetic_tracks` remains a stub (track generation is a tracker output, not an input). Update docstrings to reflect the change.

**Update __init__.py:** Add new public exports: TrajectoryConfig, TrajectoryResult, generate_trajectories (from trajectory), NoiseConfig, DetectionGenConfig, SyntheticDataset, SyntheticFrame, generate_detection_dataset (from detection), and the scenario functions from Task 3. Keep all existing exports (FishConfig, generate_fish_3d, etc.).

**Tests in `tests/unit/synthetic/test_detection_gen.py`:**

- `test_detection_dataset_structure`: Produces correct number of frames, each with detections_per_camera dict keyed by camera IDs.
- `test_detections_are_detection_type`: Each detection is an instance of `Detection` with valid bbox tuple.
- `test_no_noise_all_detected`: With base_miss_rate=0 and base_fp_rate=0, every valid projection produces exactly one detection per camera per fish.
- `test_miss_rate_reduces_detections`: With base_miss_rate=0.5, total detections across many frames is significantly less than the no-noise case.
- `test_false_positives_added`: With base_fp_rate=0.5, total detections exceed number of true fish projections.
- `test_deterministic_with_seed`: Same seed = identical dataset.
  </action>
  <verify>
`hatch run pytest tests/unit/synthetic/test_detection_gen.py -v` — all tests pass.
`hatch run pytest tests/unit/synthetic/ -v` — all synthetic tests pass (including existing ones).
`hatch run check` — no lint or type errors.
  </verify>
  <done>
generate_detection_dataset() takes a TrajectoryResult + camera models and produces per-frame dict[str, list[Detection]] with configurable noise. Stubs replaced with real implementations. __init__.py updated. 6 unit tests pass.
  </done>
</task>

<task type="auto">
  <name>Task 3: Scenario presets — targeted failure mode configurations</name>
  <files>
    src/aquapose/synthetic/scenarios.py
    tests/unit/synthetic/test_scenarios.py
  </files>
  <action>
Create `src/aquapose/synthetic/scenarios.py` providing ready-to-use scenario configurations.

**Design:** Each scenario is a function returning a `TrajectoryConfig` (and optionally a `NoiseConfig` override). Scenarios reuse the core trajectory generator — they are just curated parameter sets, NOT reimplementations.

**Implement these 4 high-ROI scenarios (of the 10 in the spec) — the ones that test the most critical tracker failure modes:**

1. `crossing_paths(difficulty: float = 0.5, seed: int = 42) -> tuple[TrajectoryConfig, NoiseConfig]`
   - Spec section 5.3. 2 fish start separated (>40cm), converge through crossing, diverge.
   - Implementation: Use waypoint-like initialization. Fish 1 starts at (-0.3, -0.2, water_z+0.5) heading toward (+0.3, +0.2). Fish 2 starts at (+0.3, -0.2) heading toward (-0.3, +0.2). The approach angle is interpolated from 90 degrees (difficulty=0) to 10 degrees (difficulty=1). Duration 10s.
   - Set cohesion=0, alignment=0 (independent motion). Use nominal noise config.
   - NOTE: The heading-based random walk will deviate from the initial heading over time. For targeted crossing, set sigma_heading to a LOW value (0.01) so fish approximately follow their initial headings but with natural jitter. The crossing geometry is approximate, not scripted — this is acceptable for tracker evaluation.

2. `track_fragmentation(miss_rate: float = 0.25, seed: int = 42) -> tuple[TrajectoryConfig, NoiseConfig]`
   - Spec section 5.5. 2 fish well-separated, elevated miss rate.
   - Fish start at opposite sides of tank, slow speed. Duration 15s.
   - Override NoiseConfig: base_miss_rate=miss_rate. No false positives (base_fp_rate=0).

3. `tight_schooling(n_fish: int = 5, seed: int = 42) -> tuple[TrajectoryConfig, NoiseConfig]`
   - Spec sections 5.1/5.2 combined. Fish school tightly, testing ID maintenance in proximity.
   - Use SchoolingConfig.tight_school() preset. Duration 10s.
   - Nominal noise config.

4. `startle_response(seed: int = 42) -> tuple[TrajectoryConfig, NoiseConfig]`
   - Spec section 5.8. This is a config-only scenario — 2 fish swimming steadily, then sharp heading change.
   - Implementation note: The core trajectory generator uses Gaussian heading noise with max_turn_rate clamp. For startle, use a special MotionConfig with VERY low sigma_heading (0.005) for steady swimming. The actual startle event (sharp turn at a specific frame) cannot be expressed purely through MotionConfig — document this as a limitation and note that a `force_overrides` mechanism (per-frame force injection) would be needed for frame-precise startle events. For now, provide the config that produces "steady swimming near each other" as the baseline.

**Each function:**
- Returns (TrajectoryConfig, NoiseConfig) tuple.
- Has a clear docstring explaining what it tests and how difficulty affects it.
- Is deterministic via seed parameter.

**Also add a convenience function:**
`generate_scenario(name: str, models: dict[str, RefractiveProjectionModel], **kwargs) -> SyntheticDataset`
- Dispatches to the named scenario function, generates trajectory, then generates detection dataset. One-call interface for downstream scripts.

**Tests in `tests/unit/synthetic/test_scenarios.py`:**

- `test_crossing_paths_returns_valid_config`: Returns TrajectoryConfig with n_fish=2.
- `test_track_fragmentation_miss_rate`: Returned NoiseConfig has the specified miss_rate.
- `test_tight_schooling_uses_schooling_preset`: Returned config has cohesion > 0.5.
- `test_generate_scenario_dispatches`: `generate_scenario("crossing_paths", rig)` returns a SyntheticDataset with correct frame count.
- `test_generate_scenario_unknown_raises`: Unknown name raises ValueError.
  </action>
  <verify>
`hatch run pytest tests/unit/synthetic/test_scenarios.py -v` — all tests pass.
`hatch run pytest tests/unit/synthetic/ -v` — ALL synthetic tests pass (existing + new).
`hatch run check` — no lint or type errors across all new files.
  </verify>
  <done>
4 scenario presets exist (crossing_paths, track_fragmentation, tight_schooling, startle_response). generate_scenario() provides a one-call interface. 5 unit tests pass. All existing synthetic tests still pass.
  </done>
</task>

</tasks>

<verification>
1. `hatch run pytest tests/unit/synthetic/ -v` — all tests pass (existing + new).
2. `hatch run check` — no lint or type errors in any synthetic module.
3. Quick smoke test (manual or in test): `generate_scenario("crossing_paths", build_fabricated_rig())` returns a SyntheticDataset with frames containing Detection objects.
4. Verify determinism: running twice with same seed produces identical results.
</verification>

<success_criteria>
- TrajectoryResult contains multi-fish 3D paths confined to tank volume across multiple frames
- SyntheticDataset contains per-frame, per-camera Detection objects consumable by FishTracker.update()
- 4 scenario presets produce distinct trajectory/noise configurations targeting tracker failure modes
- All scenarios are deterministic given same random seed
- 18+ unit tests pass across 3 new test files
- All existing synthetic tests continue to pass
- No lint or type errors
</success_criteria>

<output>
After completion, create `.planning/quick/6-create-synthetic-data-generation-system-/6-SUMMARY.md`
</output>
