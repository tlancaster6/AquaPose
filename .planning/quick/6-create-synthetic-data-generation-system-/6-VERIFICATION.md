---
phase: quick-6
verified: 2026-02-24T00:00:00Z
status: passed
score: 5/5 must-haves verified
---

# Quick Task 6: Synthetic Data Generation System Verification Report

**Task Goal:** Create a synthetic data generation system for targeted testing of Cross-View Identity and 3D Tracking implementation.
**Verified:** 2026-02-24
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Trajectory generator produces physically plausible multi-fish 3D paths confined to tank volume | VERIFIED | `generate_trajectories()` in trajectory.py (479 lines) implements boundary clamping, speed bounds, collision avoidance; `test_trajectory_stays_in_tank` and `test_trajectory_speed_bounds` pass |
| 2 | Detection generator projects 3D trajectories to per-camera Detection objects with configurable noise | VERIFIED | `generate_detection_dataset()` in detection.py (316 lines) calls `model.project()` per camera, applies Bernoulli miss model and Poisson FP; `test_no_noise_all_detected`, `test_miss_rate_reduces_detections`, `test_false_positives_added` all pass |
| 3 | Scenario presets produce ready-to-use datasets for tracker evaluation (crossing, schooling, fragmentation) | VERIFIED | scenarios.py (259 lines) implements all 4 presets via `_SCENARIO_REGISTRY` decorator; `generate_scenario()` dispatches by name; `test_generate_scenario_all_known_names` passes all 4 |
| 4 | Output format matches FishTracker.update() input: `dict[str, list[Detection]]` per frame | VERIFIED | `SyntheticFrame.detections_per_camera` is `dict[str, list[Detection]]`; `Detection` instances created with `Detection(bbox=(x,y,w,h), mask=None, area=..., confidence=...)` matching the detector.py dataclass |
| 5 | All scenarios are deterministic given same random seed | VERIFIED | `test_trajectory_deterministic` and `test_deterministic_with_seed` both pass; seed flows from `TrajectoryConfig.random_seed` through `np.random.default_rng()` |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Min Lines | Actual Lines | Status | Details |
|----------|-----------|-------------|--------|---------|
| `src/aquapose/synthetic/trajectory.py` | 150 | 479 | VERIFIED | Full motion model: AR(1) speed, boundary force, collision force, schooling forces, hard tank clamp |
| `src/aquapose/synthetic/detection.py` | 100 | 316 | VERIFIED | Full noise model: velocity-scaled miss, centroid jitter, Poisson FP, Detection creation |
| `src/aquapose/synthetic/scenarios.py` | 80 | 259 | VERIFIED | 4 scenario presets + decorator registry + `generate_scenario()` dispatcher |
| `tests/unit/synthetic/test_trajectory.py` | 60 | 232 | VERIFIED | 10 tests covering shape, determinism, physical bounds, collision avoidance, schooling |
| `tests/unit/synthetic/test_detection_gen.py` | 40 | 206 | VERIFIED | 6 tests covering structure, type, noise model variants, determinism |
| `tests/unit/synthetic/test_scenarios.py` | 30 | 115 | VERIFIED | 9 tests covering all 4 scenario configs and dispatcher |

Additionally verified:
- `src/aquapose/synthetic/__init__.py` — 15 new exports added; `__all__` updated with TrajectoryConfig, TrajectoryResult, generate_trajectories, NoiseConfig, DetectionGenConfig, SyntheticDataset, SyntheticFrame, generate_detection_dataset, all 4 scenario functions, generate_scenario
- `src/aquapose/synthetic/stubs.py` — `generate_synthetic_detections` now delegates to `generate_detection_dataset`; `generate_synthetic_tracks` raises `NotImplementedError` with informative message

### Key Link Verification

| From | To | Via | Pattern | Status | Details |
|------|----|-----|---------|--------|---------|
| `trajectory.py` | `detection.py` | TrajectoryResult consumed by generate_synthetic_detections | `TrajectoryResult` | WIRED | detection.py imports `TrajectoryResult`; `generate_detection_dataset(trajectory: TrajectoryResult, ...)` at line 139 |
| `detection.py` | `segmentation/detector.py` | Produces Detection dataclass instances | `Detection(` | WIRED | `from aquapose.segmentation.detector import Detection` at line 17; `Detection(bbox=(x, y, w, h), ...)` at lines 263 and 289 |
| `detection.py` | `calibration/projection.py` | Uses RefractiveProjectionModel.project for 3D-to-2D | `model.project` | WIRED | `pixels, valid = model.project(pts_torch)` at line 202 |
| `scenarios.py` | `trajectory.py` | Scenario presets configure TrajectoryConfig | `TrajectoryConfig` | WIRED | scenarios.py imports `TrajectoryConfig` and constructs instances in all 4 scenario functions |

### Test Results

All 42 unit tests pass (confirmed by running `hatch run pytest tests/unit/synthetic/ -v`):

| File | Tests | Result |
|------|-------|--------|
| `test_trajectory.py` | 10 | ALL PASS |
| `test_detection_gen.py` | 6 | ALL PASS |
| `test_scenarios.py` | 9 | ALL PASS |
| `test_synthetic.py` (existing) | 17 | ALL PASS |

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `stubs.py` (docstring) | "Placeholder positional/keyword arguments" | Info | Intentional: legitimate parameter names for a stub that raises NotImplementedError by design |

No blocker anti-patterns. The single note-worthy item is the word "placeholder" in a docstring within stubs.py, which is correct descriptive language for intentional stub parameters.

### Human Verification Required

None — all goal truths are programmatically verifiable via the unit tests, which pass.

Optional manual validation (for confidence only, not blocking):

1. **Smoke test with real rig** — Call `generate_scenario("crossing_paths", build_fabricated_rig())` end-to-end and confirm the returned `SyntheticDataset` frames contain `Detection` objects with reasonable pixel coordinates for a fabricated 13-camera rig. The unit tests use a 2x2 rig; the real rig has 13 cameras.

### Summary

The synthetic data generation system is fully implemented and goal is achieved. All five observable truths are verified against actual code and passing tests. The three new modules (trajectory.py, detection.py, scenarios.py) are substantive implementations — not placeholders — with line counts well above minimums. All four key links are wired with real imports and function calls. The output format (`dict[str, list[Detection]]` per frame) is compatible with `FishTracker.update()` as required.

---

_Verified: 2026-02-24_
_Verifier: Claude (gsd-verifier)_
