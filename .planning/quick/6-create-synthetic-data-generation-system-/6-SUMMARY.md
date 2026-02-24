---
phase: quick-6
plan: "01"
subsystem: synthetic
tags: [synthetic, tracking, trajectory, detection, scenarios]
dependency_graph:
  requires:
    - aquapose.calibration.projection (RefractiveProjectionModel.project)
    - aquapose.segmentation.detector (Detection dataclass)
    - aquapose.synthetic.rig (build_fabricated_rig)
  provides:
    - aquapose.synthetic.trajectory (TrajectoryResult, generate_trajectories)
    - aquapose.synthetic.detection (SyntheticDataset, generate_detection_dataset)
    - aquapose.synthetic.scenarios (crossing_paths, track_fragmentation, tight_schooling, startle_response, generate_scenario)
  affects:
    - aquapose.synthetic.__init__ (new public exports)
    - aquapose.synthetic.stubs (generate_synthetic_detections now delegates to real impl)
tech_stack:
  added: []
  patterns:
    - Heading-based AR(1) random walk for fish motion
    - Vectorised pairwise force computation (boundary, collision, schooling)
    - Newton-Raphson refractive projection (via RefractiveProjectionModel.project)
    - Decorator-based scenario registry (_SCENARIO_REGISTRY dict)
key_files:
  created:
    - src/aquapose/synthetic/trajectory.py
    - src/aquapose/synthetic/detection.py
    - src/aquapose/synthetic/scenarios.py
    - tests/unit/synthetic/test_trajectory.py
    - tests/unit/synthetic/test_detection_gen.py
    - tests/unit/synthetic/test_scenarios.py
  modified:
    - src/aquapose/synthetic/__init__.py (added 15 new exports)
    - src/aquapose/synthetic/stubs.py (generate_synthetic_detections now delegates)
decisions:
  - "TrajectoryResult stores (n_frames, n_fish, 7) float32 array — fish_id as dim 6 enables zero-copy frame extraction"
  - "FISH_BODY_LENGTH=0.085m used as collision radius base (2x body_length trigger) — matches FishConfig.scale default"
  - "_SCHOOLING_FORCE_SCALE=0.1 keeps social forces from overwhelming heading noise at moderate parameter values"
  - "NoiseConfig omits occlusion/coalescence noise — requires bbox overlap computation, deferred as future extension"
  - "Image size from camera K matrix: W=round(2*cx), H=round(2*cy) — matches Phase 04 convention"
  - "Scenario registry uses decorator pattern — adding new scenarios requires zero changes to generate_scenario()"
  - "startle_response provides pre-startle baseline only — frame-precise force injection deferred (noted in docstring)"
metrics:
  duration: "8 minutes"
  completed_date: "2026-02-24"
  tasks_completed: 3
  tasks_total: 3
  files_created: 6
  files_modified: 2
  tests_added: 25
  tests_total_synthetic: 42
---

# Quick Task 6: Synthetic Data Generation System Summary

Multi-fish 3D trajectory generation + refractive projection to Detection objects + scenario presets for tracker evaluation.

## What Was Built

Three new modules extending `src/aquapose/synthetic/` with multi-frame tracking-compatible data generation:

1. **trajectory.py** — Heading-based random walk with AR(1) speed model, boundary soft-repulsion, pairwise collision avoidance, and configurable schooling forces (cohesion + alignment). SchoolingConfig presets: `independent`, `loose_school`, `tight_school`, `milling`, `streaming`. Output: `TrajectoryResult` with (n_frames, n_fish, 7) float32 state array.

2. **detection.py** — Projects 3D fish positions through `RefractiveProjectionModel.project` per camera per frame. Applies Bernoulli miss model (velocity-scaled), Gaussian centroid jitter, bbox size noise, and Poisson false positives. Output: `SyntheticDataset` with `SyntheticFrame` objects containing `dict[str, list[Detection]]` — identical to `FishTracker.update()` input format.

3. **scenarios.py** — Four curated scenario configurations as `(TrajectoryConfig, NoiseConfig)` tuples targeting tracker failure modes: `crossing_paths` (ID swaps), `track_fragmentation` (elevated miss rate), `tight_schooling` (proximity maintenance), `startle_response` (pre-startle baseline). `generate_scenario()` provides one-call interface.

## Test Results

42 unit tests pass across 4 test files (25 new + 17 existing):

| File | Tests |
|------|-------|
| test_trajectory.py | 10 (new) |
| test_detection_gen.py | 6 (new) |
| test_scenarios.py | 9 (new) |
| test_synthetic.py | 17 (existing, all still pass) |

## Key Design Decisions

- TrajectoryResult stores fish_id as dimension 6 of the state array (float32 cast of int) for zero-copy frame extraction.
- FISH_BODY_LENGTH = 0.085m (matching FishConfig.scale) used as collision avoidance trigger at 2x body length.
- `_SCHOOLING_FORCE_SCALE = 0.1` prevents social forces from overwhelming heading noise — schooling shapes group but doesn't lock fish into rigid formation.
- NoiseConfig omits occlusion/coalescence noise (deferred — requires bbox overlap computation).
- Image dimensions estimated from K matrix principal point: W=round(2*cx), H=round(2*cy) — matches Phase 04 convention.
- Scenario registry uses a `@_register(name)` decorator — adding new scenarios requires no changes to `generate_scenario()`.
- `startle_response` provides only the pre-startle baseline config. Frame-precise force injection would require a `force_overrides` mechanism (noted in docstring as future extension).

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

### Files created

- [x] `src/aquapose/synthetic/trajectory.py` exists (>150 lines)
- [x] `src/aquapose/synthetic/detection.py` exists (>100 lines)
- [x] `src/aquapose/synthetic/scenarios.py` exists (>80 lines)
- [x] `tests/unit/synthetic/test_trajectory.py` exists (10 tests)
- [x] `tests/unit/synthetic/test_detection_gen.py` exists (6 tests)
- [x] `tests/unit/synthetic/test_scenarios.py` exists (9 tests)

### Commits

- 3781536: feat(quick-6): trajectory generator
- 614c6b9: feat(quick-6): detection generator
- 4c3c604: feat(quick-6): scenario presets

## Self-Check: PASSED
