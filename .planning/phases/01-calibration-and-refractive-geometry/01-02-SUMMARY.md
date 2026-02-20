---
phase: 01-calibration-and-refractive-geometry
plan: 02
subsystem: calibration
tags: [pytorch, aquacal, triangulation, svd, ray-simulation, uncertainty, matplotlib]

# Dependency graph
requires:
  - "01-01: RefractiveProjectionModel.project() and cast_ray()"
  - "01-01: load_calibration_data() for real calibration loading"
  - "aquacal.datasets.synthetic.generate_camera_intrinsics()"
provides:
  - "triangulate_rays(): SVD least-squares ray intersection for N rays (moved to projection.py during post-phase cleanup)"
  - "compute_triangulation_uncertainty(): ray-simulation Z/XY error vs depth with image bounds checking"
  - "build_rig_from_calibration(): load real calibration JSON into RefractiveProjectionModel list"
  - "build_synthetic_rig(): approximate 13-camera rig (1 wide-angle center + 12 ring at ~54° roll)"
  - "generate_uncertainty_report(): matplotlib plots + markdown report"
  - "docs/reports/z_uncertainty_report.md: Z error is 7x-15x larger than XY (mean ~11x) from real calibration"
affects:
  - phase-04-analysis-by-synthesis  # informs optimizer loss weighting

# Tech tracking
tech-stack:
  added:
    - "matplotlib (Agg backend) — non-interactive plot generation for report"
  patterns:
    - "SVD least-squares ray triangulation: sum_i (I - d_i d_i^T) @ p = sum_i (I - d_i d_i^T) @ o_i"
    - "±pixel_noise perturbation in 4 directions per camera for per-depth error estimation"
    - "torch.linalg.lstsq for degenerate-safe ray intersection"

key-files:
  created:
    - "src/aquapose/calibration/uncertainty.py"
    - "tests/unit/calibration/test_uncertainty.py"
    - "docs/reports/z_uncertainty_report.md"
    - "docs/reports/z_uncertainty_xyz_error.png"
    - "docs/reports/z_uncertainty_ratio.png"
    - "docs/reports/z_uncertainty_cameras.png"
  modified:
    - "src/aquapose/calibration/__init__.py"

key-decisions:
  - "Z/XY anisotropy is ~11x mean (7x-15x range) at 0.5px noise from real calibration — confirms top-down cameras have poor Z observability; Phase 4 optimizer must weight Z differently"
  - "Report uses build_rig_from_calibration() with real calibration JSON, not synthetic rig — original synthetic rig (AquaCal generate_real_rig_array) had wrong geometry: 2 rings instead of 1, 90° roll instead of ~54°, 0.75m height instead of 1.0m, no image bounds checking, no wide-angle center camera"
  - "Real rig geometry: 1 wide-angle center camera (fx=746, ~94° FOV) + 12 ring cameras (fx~1580, ~54° FOV) at ~0.65m radius with ~54° roll from radial direction, ~1.03m above water"
  - "compute_triangulation_uncertainty defaults to test point at origin (under reference camera) with image bounds checking — not all 13 cameras see every point; visibility ramps from 4 cameras (shallow) to 6 (deep)"
  - "Report generated as a one-time artifact (not a test) — script-run approach matches plan specification"

# Metrics
duration: 5min
completed: 2026-02-19
---

# Phase 1 Plan 2: Z-Uncertainty Characterization Summary

**Ray simulation on real 13-camera top-down rig shows Z error is ~11x larger than XY (range 7x-15x), quantifying fundamental depth observability gap for Phase 4 optimizer weighting**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-02-19T22:57:36Z
- **Completed:** 2026-02-19T23:03:00Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Implemented `uncertainty.py` with four public functions: `compute_triangulation_uncertainty` (ray-simulation with image bounds checking), `build_rig_from_calibration` (loads real calibration JSON), `build_synthetic_rig` (approximate rig for tests), and `generate_uncertainty_report` (3 matplotlib plots + markdown). `triangulate_rays` was moved to `projection.py` during post-phase cleanup as a general-purpose multi-view geometry primitive.
- Generated `docs/reports/z_uncertainty_report.md` from real calibration at 20 depth samples across full tank range (1.08-2.03m), documenting Z/XY anisotropy of ~11x mean (7x-15x range), with realistic camera visibility ramp (4 cameras shallow → 6 deep)
- 16 unit tests passing: ray triangulation accuracy (known intersection geometry), parallel-ray degeneracy, rig construction (13 models, correct types, can project), output shapes, Z > XY anisotropy property, noise sensitivity

## Task Commits

1. **Task 1: Implement uncertainty computation and report generation** - `83e0b6c` (feat)
2. **Task 2: Generate the Z-uncertainty characterization report** - `4ac597c` (feat)

## Files Created/Modified

- `src/aquapose/calibration/uncertainty.py` — UncertaintyResult dataclass (with xy_position field), compute_triangulation_uncertainty (with image bounds + xy_position), build_rig_from_calibration, build_synthetic_rig, generate_uncertainty_report (with pixel_noise param)
- `src/aquapose/calibration/__init__.py` — Added 6 new exports to __all__ (including build_rig_from_calibration)
- `tests/unit/calibration/test_uncertainty.py` — 16 unit tests across 3 test classes
- `docs/reports/z_uncertainty_report.md` — Report with depth table, summary stats, interpretation paragraph
- `docs/reports/z_uncertainty_xyz_error.png` — X/Y/Z error vs depth plot
- `docs/reports/z_uncertainty_ratio.png` — Z/XY ratio vs depth plot
- `docs/reports/z_uncertainty_cameras.png` — Camera visibility vs depth plot

## Decisions Made

- **Z/XY anisotropy magnitude**: Measured at ~11x mean (7x-15x range) for 0.5px noise across 1.08-2.03m depth range from real calibration. 4-6 cameras see the reference camera position at any given depth (not all 13). Z error 0.002-0.005mm, XY error 0.0002-0.0007mm.
- **Real vs synthetic rig**: Original synthetic rig (AquaCal `generate_real_rig_array`) was incorrect: 2 rings (6+6) not 1 ring of 12, 90° roll not ~54°, no wide-angle center camera, no image bounds check, height 0.75m not 1.0m. Switched to `build_rig_from_calibration()` loading real calibration JSON. Synthetic rig kept as corrected fallback for tests.
- **Real rig geometry** (from calibration analysis): 1 wide-angle center camera (e3v8250, fx=746) + 12 ring cameras (fx~1560-1650) at ~0.65m radius with ~54° roll from radial direction, water_z=1.03m. Reference camera (e3v829d) is on the ring, not at rig center.
- **Test point at origin**: `compute_triangulation_uncertainty` defaults to (0,0,Z) — under the reference camera — matching the coordinate frame of the calibration.
- **Report generation approach**: Generated as a script-run artifact (not a recurring test) as specified in the plan. The report is committed as a static file.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ambiguous variable name `I` (ruff E741)**
- **Found during:** Task 1 (lint check before first commit)
- **Issue:** `I = torch.eye(...)` triggers ruff E741 (ambiguous name: looks like numeral 1)
- **Fix:** Renamed to `eye3` throughout `triangulate_rays`
- **Files modified:** `src/aquapose/calibration/uncertainty.py`
- **Committed in:** `83e0b6c`

**2. [Rule 1 - Bug] Removed unused `xy_mean_valid` variable (ruff F841)**
- **Found during:** Task 1 (lint check before first commit)
- **Issue:** Intermediate variable `xy_mean_valid` assigned but not used in `generate_uncertainty_report`
- **Fix:** Removed the assignment; `ratio_valid = ratio[valid]` achieves the same result directly
- **Files modified:** `src/aquapose/calibration/uncertainty.py`
- **Committed in:** `83e0b6c`

**3. [Rule 1 - Bug] Fixed non-ASCII em-dash in report template**
- **Found during:** Task 2 (report generation)
- **Issue:** Em-dash character (`—`) in the f-string report template produced a non-ASCII byte in the markdown file, causing `grep` to treat it as a binary file
- **Fix:** Replaced with a plain comma in the sentence
- **Files modified:** `src/aquapose/calibration/uncertainty.py`
- **Committed in:** `4ac597c`

---

**Total deviations:** 3 auto-fixed (2 lint issues + 1 encoding bug in report template)
**Impact on plan:** Minor code quality fixes only; implementation matches specification exactly.

## Key Finding

The Z/XY anisotropy is **~11x mean (7x-15x range)** from the real 13-camera calibration with 0.5px pixel noise. Only 4-6 cameras see a given point at any depth (not all 13), because the narrow-FOV ring cameras (~54°) with ~54° roll have limited overlap. This is a fundamental geometric property: top-down cameras have rays that are nearly parallel in Z, so small pixel perturbations have minimal effect on X/Y reconstruction but large effect on Z. Phase 4 should apply a Z-axis loss weight approximately 10x smaller than the X/Y weight.

## Next Phase Readiness

- `aquapose.calibration` module is fully complete — loader, projection, and uncertainty characterization all available
- Z/XY anisotropy (~11x) is quantified from real hardware — Phase 4 optimizer weighting is now informed
- Phase 1 blocker "Z-uncertainty budget not yet quantified" is resolved

---
*Phase: 01-calibration-and-refractive-geometry*
*Completed: 2026-02-19*
