---
phase: quick-7
plan: "01"
subsystem: tracking
tags: [tracking, diagnostics, synthetic, metrics, visualization]
dependency_graph:
  requires:
    - aquapose.synthetic.scenarios (generate_scenario, _SCENARIO_REGISTRY)
    - aquapose.synthetic.detection (generate_detection_dataset, SyntheticDataset)
    - aquapose.synthetic.trajectory (generate_trajectories, TrajectoryResult)
    - aquapose.synthetic.rig (build_fabricated_rig)
    - aquapose.tracking.tracker (FishTracker)
    - aquapose.visualization.overlay (FISH_COLORS)
  provides:
    - scripts/diagnose_tracking.py (standalone CLI diagnostic for FishTracker evaluation)
    - tests/unit/tracking/test_diagnose_tracking.py (smoke tests for metric computation)
  affects: []
tech_stack:
  added: []
  patterns:
    - Greedy nearest-neighbour GT matching by 3D centroid proximity
    - CLEAR MOT-inspired metric computation (MOTA, ID switches, fragmentation)
    - importlib module loading for testing script-module functions
    - Visualization dispatch with per-function exception guarding
key_files:
  created:
    - scripts/diagnose_tracking.py
    - tests/unit/tracking/test_diagnose_tracking.py
  modified: []
decisions:
  - "TrackingMetrics dataclass holds all metric fields including per-fish dicts for report generation"
  - "GT matching uses greedy nearest-neighbour 3D assignment (0.15m threshold) - generous for synthetic data with noise"
  - "ID switch detection only counts actual GT identity changes (not None->GT or GT->None transitions)"
  - "vis_detection_overlay_grid uses Mapping[str, Sequence[Any]] signature for invariant dict covariance"
  - "Smoke tests use importlib.util.spec_from_file_location to load script as module without packaging"
metrics:
  duration: "11 minutes"
  completed_date: "2026-02-24"
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 0
  tests_added: 5
---

# Quick Task 7: Cross-View Identity and 3D Tracking Diagnostic Summary

Standalone FishTracker diagnostic script with CLEAR MOT-inspired metrics and 4 visualizations evaluated against synthetic ground truth from the quick-6 system.

## What Was Built

### scripts/diagnose_tracking.py (1094 lines)

A standalone CLI diagnostic that benchmarks FishTracker against synthetic ground truth. Key components:

**`TrackingMetrics` dataclass** — holds all CLEAR MOT-inspired metrics:
- MOTA, ID switches, fragmentation, TP, FN, FP
- Per-track purity, mean track purity
- Mostly tracked / mostly lost GT fish counts
- Per-fish tracked fraction dict

**`compute_tracking_metrics()`** — pure function computing metrics from:
1. Per-frame GT matching dict (track_id -> gt_fish_id)
2. Per-frame GT positions (n_gt, 3) arrays
3. Per-frame GT fish ID lists

**`_match_gt_to_tracks()`** — greedy nearest-neighbour GT assignment with 0.15m threshold.

**4 visualizations:**
1. `vis_3d_trajectories()` — 3D matplotlib plot with GT solid lines + tracker scatter markers
2. `vis_detection_overlay_grid()` — NxN camera grid with GT circles, detection squares, missed X markers
3. `vis_id_timeline()` — Horizontal timeline per GT fish, colored by assigned track_id, red lines at ID switches
4. `vis_metrics_barchart()` — Bar chart of MOTA, TP%, FN%, FP%, mean purity

**CLI interface:**
```
python scripts/diagnose_tracking.py --scenario crossing_paths --difficulty 0.7
python scripts/diagnose_tracking.py --scenario tight_schooling --n-fish 7
python scripts/diagnose_tracking.py --scenario track_fragmentation --miss-rate 0.4
python scripts/diagnose_tracking.py --scenario startle_response --n-cameras 4
```

### tests/unit/tracking/test_diagnose_tracking.py (242 lines)

5 smoke tests loading the script via importlib and testing metric computation on mock data:
1. `test_perfect_tracking_metrics` — MOTA=1.0, zero switches/FP/FN
2. `test_complete_miss_metrics` — MOTA<=0, FN=n_gt*n_frames
3. `test_id_switch_detection` — id_switches>=1 on GT reassignment
4. `test_fragmentation_detection` — fragmentation>=1 on track gap
5. `test_metrics_dataclass_fields` — all 9 required fields present

## Test Results

```
tests/unit/tracking/test_diagnose_tracking.py::test_perfect_tracking_metrics PASSED
tests/unit/tracking/test_diagnose_tracking.py::test_complete_miss_metrics PASSED
tests/unit/tracking/test_diagnose_tracking.py::test_id_switch_detection PASSED
tests/unit/tracking/test_diagnose_tracking.py::test_fragmentation_detection PASSED
tests/unit/tracking/test_diagnose_tracking.py::test_metrics_dataclass_fields PASSED
5 passed in 0.08s
```

## Script End-to-End Verification

```
python scripts/diagnose_tracking.py --scenario crossing_paths
  -> 300 frames, 2 GT fish, 16 cameras
  -> MOTA computed, 4 PNGs + tracking_report.md produced

python scripts/diagnose_tracking.py --scenario tight_schooling --n-fish 3
  -> 300 frames, 3 GT fish, 16 cameras
  -> Runs to completion without errors
```

Output files: `3d_trajectories.png`, `detection_overlay.png`, `id_timeline.png`, `metrics_barchart.png`, `tracking_report.md`

## Deviations from Plan

**Auto-fixed (Rule 1 - Type annotation):** `vis_detection_overlay_grid` initially used `dict[str, list[object]]` which is invariant — changed to `Mapping[str, Sequence[Any]]` for covariance compatibility. One additional `# type: ignore[arg-type]` applied to metadata dict int extraction.

None — plan otherwise executed exactly as written.

## Commits

- `0655121`: feat(quick-7): cross-view identity and 3D tracking diagnostic script
- `84a80ca`: test(quick-7): smoke tests for diagnose_tracking metric computation

## Self-Check

### Files created

- [x] `scripts/diagnose_tracking.py` exists (1094 lines, min 350)
- [x] `tests/unit/tracking/test_diagnose_tracking.py` exists (242 lines, min 40)

### Output artifacts

- [x] `output/tracking_diagnostic/3d_trajectories.png`
- [x] `output/tracking_diagnostic/detection_overlay.png`
- [x] `output/tracking_diagnostic/id_timeline.png`
- [x] `output/tracking_diagnostic/metrics_barchart.png`
- [x] `output/tracking_diagnostic/tracking_report.md`

### Commits

- [x] `0655121` exists in git log
- [x] `84a80ca` exists in git log

### Key links verified

- [x] `diagnose_tracking.py` imports from `aquapose.synthetic.scenarios._SCENARIO_REGISTRY`
- [x] `tracker.update()` called per frame
- [x] `SyntheticFrame.ground_truth` used for GT position extraction

## Self-Check: PASSED
