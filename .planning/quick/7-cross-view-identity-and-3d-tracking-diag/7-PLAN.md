---
phase: quick-7
plan: "01"
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/diagnose_tracking.py
  - tests/unit/tracking/test_diagnose_tracking.py
autonomous: true
requirements: []

must_haves:
  truths:
    - "Script generates a synthetic scenario, runs FishTracker on it, and produces quantitative metrics comparing tracked IDs to ground truth fish IDs"
    - "Script produces at least 4 diagnostic visualizations: 3D trajectory plot, per-camera detection overlay, ID consistency timeline, and tracking metrics summary"
    - "Script runs end-to-end with zero real data dependencies (synthetic only)"
    - "Metrics include MOTA-like accuracy, ID switch count, track fragmentation count, and false positive rate"
    - "All 4 scenario presets can be selected via CLI flag"
  artifacts:
    - path: "scripts/diagnose_tracking.py"
      provides: "Standalone diagnostic script for cross-view identity and 3D tracking evaluation on synthetic data"
      min_lines: 350
    - path: "tests/unit/tracking/test_diagnose_tracking.py"
      provides: "Smoke tests verifying the diagnostic functions produce expected output shapes"
      min_lines: 40
  key_links:
    - from: "scripts/diagnose_tracking.py"
      to: "src/aquapose/synthetic/scenarios.py"
      via: "generate_scenario() produces SyntheticDataset fed to tracker"
      pattern: "generate_scenario"
    - from: "scripts/diagnose_tracking.py"
      to: "src/aquapose/tracking/tracker.py"
      via: "FishTracker.update() processes each synthetic frame"
      pattern: "tracker\\.update"
    - from: "scripts/diagnose_tracking.py"
      to: "src/aquapose/synthetic/detection.py"
      via: "SyntheticFrame.ground_truth provides GT fish_id linkage for metric computation"
      pattern: "ground_truth"
---

<objective>
Create a diagnostic script for benchmarking the Cross-View Identity and 3D Tracking pipeline (FishTracker) using synthetic data from quick-6. The script generates a scenario, runs FishTracker frame-by-frame, computes quantitative tracking metrics (MOTA, ID switches, fragmentation, false positives) by comparing tracker output to synthetic ground truth, and produces diagnostic visualizations.

Purpose: Enable controlled, reproducible evaluation of FishTracker performance across known failure modes (crossing paths, fragmentation, schooling) without any real data dependency.

Output: `scripts/diagnose_tracking.py` with CLI interface + smoke tests.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/quick/6-create-synthetic-data-generation-system-/6-SUMMARY.md
@src/aquapose/synthetic/__init__.py
@src/aquapose/synthetic/scenarios.py
@src/aquapose/synthetic/detection.py
@src/aquapose/synthetic/trajectory.py
@src/aquapose/tracking/tracker.py
@src/aquapose/tracking/associate.py
@src/aquapose/synthetic/rig.py
@scripts/diagnose_pipeline.py (reference pattern for CLI structure, timing, visualization dispatch)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Core diagnostic script with tracking metrics and GT matching</name>
  <files>
    scripts/diagnose_tracking.py
  </files>
  <action>
Create `scripts/diagnose_tracking.py` — a standalone diagnostic script focused on benchmarking FishTracker against synthetic ground truth.

**CLI interface (follow diagnose_pipeline.py patterns):**

```
python scripts/diagnose_tracking.py
python scripts/diagnose_tracking.py --scenario crossing_paths --difficulty 0.7
python scripts/diagnose_tracking.py --scenario tight_schooling --n-fish 7
python scripts/diagnose_tracking.py --scenario track_fragmentation --miss-rate 0.4
python scripts/diagnose_tracking.py --output-dir output/tracking_diagnostic
python scripts/diagnose_tracking.py --n-cameras 4
```

Arguments:
- `--scenario`: One of crossing_paths, track_fragmentation, tight_schooling, startle_response (default: crossing_paths)
- `--difficulty`: Float 0-1 for crossing_paths (default 0.5)
- `--miss-rate`: Float for track_fragmentation (default 0.25)
- `--n-fish`: Int for tight_schooling (default 5)
- `--seed`: Random seed (default 42)
- `--n-cameras`: Cameras per axis for fabricated rig (default 4, produces NxN grid)
- `--output-dir`: Output directory (default output/tracking_diagnostic)
- `--min-hits`: FishTracker min_hits (default 3, lower than production 5 since synthetic has fewer cameras)
- `--min-cameras-birth`: FishTracker min_cameras_birth (default 2, since fabricated rig may have fewer cameras)
- `--expected-count`: FishTracker expected fish count (default: derived from scenario n_fish)

**Main flow (`main()`):**

1. Parse args, create output_dir.
2. Build fabricated rig via `build_fabricated_rig(n_cameras_x=args.n_cameras, n_cameras_y=args.n_cameras)`.
3. Dispatch to scenario function with appropriate kwargs (use a dict mapping scenario name to its valid kwargs, extracted from args). Call `generate_scenario(name, models, **kwargs)` to get `SyntheticDataset`.
4. Instantiate `FishTracker(expected_count=..., min_hits=args.min_hits, min_cameras_birth=args.min_cameras_birth)`.
5. Frame loop: for each `SyntheticFrame` in dataset.frames:
   - Call `tracker.update(frame.detections_per_camera, models, frame_index=frame.frame_index)`.
   - Record: confirmed tracks (fish_id, position), all tracks (for timeline).
   - Record timing per frame.
6. Compute tracking metrics via `compute_tracking_metrics()` (see below).
7. Generate visualizations (see below).
8. Write markdown report.

**Ground truth matching (`_match_gt_to_tracks()`):**

For each frame, match tracker output fish_ids to ground truth fish_ids. The matching logic:
- For each confirmed track, find which GT fish_id has the closest 3D centroid to the track's position. Use a greedy assignment: compute pairwise distances between track positions and GT positions (from trajectory states), assign closest pairs first.
- A match is valid if distance < 0.15m (generous threshold for synthetic data with noise).
- Build a per-frame mapping: `dict[int, int | None]` — track_fish_id -> gt_fish_id (or None if no match).
- Track ID consistency: across frames, check if a track_fish_id switches which gt_fish_id it maps to. Each such switch is an "ID swap".

**Tracking metrics (`compute_tracking_metrics()`):**

Compute CLEAR MOT-inspired metrics from the GT matching:
- `n_gt`: Total GT objects across all frames (sum of n_fish * n_frames where fish is in tank).
- `true_positives`: Frames where a GT fish has a matched confirmed track.
- `false_negatives`: Frames where a GT fish has no matched track (missed).
- `false_positives`: Confirmed tracks with no GT match.
- `id_switches`: Number of times a track changes its GT assignment.
- `fragmentation`: Number of times a GT fish's track is interrupted (goes from matched to unmatched and back).
- `MOTA`: 1 - (FN + FP + ID_switches) / n_gt (clamped to [-1, 1]).
- `track_purity`: For each track, fraction of frames where it maps to its most-frequent GT fish.
- `mostly_tracked`: GT fish tracked for >= 80% of their lifetime.
- `mostly_lost`: GT fish tracked for <= 20% of their lifetime.

Return a dataclass `TrackingMetrics` with all these fields.

**Print summary to stdout** (similar to diagnose_pipeline.py timing summary format):
```
=== Tracking Metrics Summary ===
Scenario:            crossing_paths (difficulty=0.5)
Frames:              300
GT fish:             2
Confirmed tracks:    N
MOTA:                0.XX
ID switches:         N
Fragmentations:      N
True positives:      N / M (XX.X%)
False negatives:     N (XX.X%)
False positives:     N (XX.X%)
Mostly tracked:      N / M
Mostly lost:         N / M
Mean track purity:   0.XX
```

**Visualization functions (all take output_dir Path, return None):**

1. `vis_3d_trajectories(trajectory, tracker_positions, output_path)`:
   - matplotlib 3D plot. Ground truth trajectories as solid colored lines. Tracker positions as scatter markers (x). Color by fish_id. Title: "GT vs Tracked 3D Trajectories".
   - Use `plot3d.py` style (import FISH_COLORS from overlay, convert BGR->RGB).

2. `vis_detection_overlay_grid(dataset, tracker_results, models, output_path)`:
   - For a sample frame (frame N//2), show a grid of camera views. Each camera panel is a blank image (sized from K matrix) with GT centroids as circles and detected bbox centers as squares. Color by fish_id. Mark missed detections with red X. Max 4x4 grid.

3. `vis_id_timeline(gt_matching_per_frame, n_gt_fish, output_path)`:
   - Horizontal timeline plot. X axis = frame. Y axis = GT fish ID. Color = tracker fish_id assigned to that GT fish at each frame. Gaps (unmatched frames) shown in gray. ID switches highlighted with vertical red lines.

4. `vis_metrics_barchart(metrics, output_path)`:
   - Bar chart of key metrics: TP%, FN%, FP%, MOTA, mean purity. Clean single-figure summary.

5. `write_tracking_report(output_path, metrics, args, dataset_metadata, timing)`:
   - Markdown report with: scenario config, metrics table, per-fish breakdown, image references to the 4 visualizations above, timing.

**Visualization dispatch** — follow diagnose_pipeline.py pattern:
```python
vis_funcs = [
    ("3d_trajectories.png", lambda: vis_3d_trajectories(...)),
    ("detection_overlay.png", lambda: vis_detection_overlay_grid(...)),
    ("id_timeline.png", lambda: vis_id_timeline(...)),
    ("metrics_barchart.png", lambda: vis_metrics_barchart(...)),
]
for name, func in vis_funcs:
    try:
        print(f"  Generating {name}...")
        func()
    except Exception as exc:
        print(f"  [WARN] Failed to generate {name}: {exc}")
```

**Key implementation notes:**
- Import torch lazily or only where needed — the script should start fast.
- Use `from __future__ import annotations` at top.
- For the 3D trajectory plot, extract GT positions from `trajectory.states[:, :, :3]` (TrajectoryResult stored inside SyntheticDataset is not directly accessible — you need to regenerate it OR extract GT positions from `SyntheticFrame.ground_truth`). Since SyntheticDataset does not store the TrajectoryResult, regenerate the trajectory separately: call the scenario function to get (TrajectoryConfig, NoiseConfig), then `generate_trajectories(traj_config)` to get the TrajectoryResult, then `generate_detection_dataset(trajectory, models, noise_config)` for the dataset. This avoids modifying the synthetic module. Store both trajectory and dataset.
- For fabricated rig cameras, image dimensions may be small (depends on rig K matrix). The detection overlay grid should handle this gracefully (upscale if needed).
- Use `plt.close('all')` after each visualization to prevent memory leaks.
- The script should print progress and total wall time at the end.
  </action>
  <verify>
`python scripts/diagnose_tracking.py --scenario crossing_paths --output-dir output/tracking_diagnostic` runs to completion, prints metrics summary, and produces 4 PNG files + 1 markdown report in output directory.

`python scripts/diagnose_tracking.py --scenario tight_schooling --n-fish 3` also runs to completion.

`hatch run check` — no lint or type errors in the new script.
  </verify>
  <done>
diagnose_tracking.py exists with CLI interface, runs all 4 scenarios, computes MOTA/ID-switch/fragmentation metrics from GT matching, produces 4 visualizations + markdown report. No real data dependencies.
  </done>
</task>

<task type="auto">
  <name>Task 2: Smoke tests for diagnostic functions</name>
  <files>
    tests/unit/tracking/test_diagnose_tracking.py
  </files>
  <action>
Create `tests/unit/tracking/test_diagnose_tracking.py` with lightweight smoke tests that verify the core metric computation logic without running the full script (which requires GPU for projection).

**Test approach:** Import the metric computation functions directly from the script module. Since scripts/ is not a package, use `importlib` or add the metric functions to a testable location. Preferred approach: structure `diagnose_tracking.py` so that `TrackingMetrics` dataclass and `compute_tracking_metrics()` are importable. At the top of the test file:
```python
import importlib.util
spec = importlib.util.spec_from_file_location("diagnose_tracking", "scripts/diagnose_tracking.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
```

**Tests:**

1. `test_perfect_tracking_metrics`: Create mock GT matching data where every frame has perfect 1:1 assignment. Verify MOTA=1.0, id_switches=0, fragmentation=0, FP=0, FN=0.

2. `test_complete_miss_metrics`: Create mock GT matching where no track ever matches GT. Verify MOTA <= 0, FN = n_gt.

3. `test_id_switch_detection`: Create mock GT matching where track 0 maps to GT fish 0 for frames 0-49, then switches to GT fish 1 for frames 50-99. Verify id_switches >= 1.

4. `test_fragmentation_detection`: Create mock GT matching where GT fish 0 is tracked for frames 0-20, lost for frames 21-30, tracked again for frames 31-50. Verify fragmentation >= 1.

5. `test_metrics_dataclass_fields`: Verify TrackingMetrics has all expected fields (mota, id_switches, fragmentation, true_positives, false_negatives, false_positives, mostly_tracked, mostly_lost, mean_track_purity).

**Key notes:**
- These tests do NOT require GPU or camera models — they test pure metric computation on mock data.
- Keep tests fast (no trajectory generation, no projection).
- If the import approach via importlib is too fragile, an alternative: extract `TrackingMetrics` and `compute_tracking_metrics` into a small utility in `src/aquapose/tracking/metrics.py` and import from there. But prefer the importlib approach to keep the scope minimal.
  </action>
  <verify>
`hatch run pytest tests/unit/tracking/test_diagnose_tracking.py -v` — all 5 tests pass.
`hatch run check` — no lint or type errors.
  </verify>
  <done>
5 smoke tests verify metric computation correctness: perfect tracking, total miss, ID switch detection, fragmentation detection, and dataclass completeness. All pass without GPU.
  </done>
</task>

</tasks>

<verification>
1. `python scripts/diagnose_tracking.py --scenario crossing_paths` runs end-to-end and produces output in output/tracking_diagnostic/.
2. `python scripts/diagnose_tracking.py --scenario tight_schooling --n-fish 3` also runs.
3. `hatch run pytest tests/unit/tracking/test_diagnose_tracking.py -v` — all tests pass.
4. `hatch run check` — no lint or type errors.
5. Output directory contains: 3d_trajectories.png, detection_overlay.png, id_timeline.png, metrics_barchart.png, tracking_report.md.
6. Printed metrics summary includes MOTA, ID switches, fragmentation, TP/FN/FP counts.
</verification>

<success_criteria>
- diagnose_tracking.py runs on all 4 scenario presets without errors
- Computes CLEAR MOT-inspired metrics (MOTA, ID switches, fragmentation, TP, FN, FP)
- Produces 4 diagnostic visualizations + markdown report
- Zero real data dependencies — fully synthetic
- 5 smoke tests pass for metric computation
- No lint or type errors
</success_criteria>

<output>
After completion, create `.planning/quick/7-cross-view-identity-and-3d-tracking-diag/7-SUMMARY.md`
</output>
