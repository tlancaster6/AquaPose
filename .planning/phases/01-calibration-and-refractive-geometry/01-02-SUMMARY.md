---
phase: 01-calibration-and-refractive-geometry
plan: 02
subsystem: calibration
tags: [pytorch, aquacal, triangulation, svd, ray-simulation, uncertainty, matplotlib]

# Dependency graph
requires:
  - "01-01: RefractiveProjectionModel.project() and cast_ray()"
  - "aquacal.datasets.synthetic.generate_real_rig_array()"
provides:
  - "triangulate_rays(): SVD least-squares ray intersection for N rays"
  - "compute_triangulation_uncertainty(): ray-simulation Z/XY error vs depth"
  - "build_synthetic_rig(): 13-camera AquaCal reference geometry as RefractiveProjectionModel list"
  - "generate_uncertainty_report(): matplotlib plots + markdown report"
  - "docs/reports/z_uncertainty_report.md: Z error is 30x-577x larger than XY (mean 132x)"
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
  - "Z/XY anisotropy is 132x mean (30x-577x range) at 0.5px noise — confirms top-down cameras have fundamentally poor Z observability; Phase 4 optimizer must weight Z differently"
  - "build_synthetic_rig uses AquaCal generate_real_rig_array with water_z = height_above_water (0.75m) since cameras are at world Z=0 and Z increases downward into water"
  - "Report generated as a one-time artifact (not a test) — script-run approach matches plan specification"

# Metrics
duration: 5min
completed: 2026-02-19
---

# Phase 1 Plan 2: Z-Uncertainty Characterization Summary

**Ray simulation on 13-camera top-down rig shows Z error is 132x larger than XY (range 30x-577x), quantifying fundamental depth observability gap for Phase 4 optimizer weighting**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-02-19T22:57:36Z
- **Completed:** 2026-02-19T23:03:00Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Implemented `uncertainty.py` with four public functions: `triangulate_rays` (SVD least-squares), `compute_triangulation_uncertainty` (ray-simulation at tank center), `build_synthetic_rig` (converts AquaCal 13-camera rig to RefractiveProjectionModel list), and `generate_uncertainty_report` (3 matplotlib plots + markdown)
- Generated `docs/reports/z_uncertainty_report.md` at 10 depth samples (5cm intervals, 0.80-1.25m range) documenting Z/XY anisotropy of 132x mean, up to 577x
- 16 unit tests passing: ray triangulation accuracy (known intersection geometry), parallel-ray degeneracy, rig construction (13 models, correct types, can project), output shapes, Z > XY anisotropy property, noise sensitivity

## Task Commits

1. **Task 1: Implement uncertainty computation and report generation** - `83e0b6c` (feat)
2. **Task 2: Generate the Z-uncertainty characterization report** - `4ac597c` (feat)

## Files Created/Modified

- `src/aquapose/calibration/uncertainty.py` — UncertaintyResult dataclass, triangulate_rays, compute_triangulation_uncertainty, build_synthetic_rig, generate_uncertainty_report
- `src/aquapose/calibration/__init__.py` — Added 5 new exports to __all__
- `tests/unit/calibration/test_uncertainty.py` — 16 unit tests across 3 test classes
- `docs/reports/z_uncertainty_report.md` — Report with depth table, summary stats, interpretation paragraph
- `docs/reports/z_uncertainty_xyz_error.png` — X/Y/Z error vs depth plot
- `docs/reports/z_uncertainty_ratio.png` — Z/XY ratio vs depth plot
- `docs/reports/z_uncertainty_cameras.png` — Camera visibility vs depth plot

## Decisions Made

- **Z/XY anisotropy magnitude**: Measured at 132x mean (30x-577x range) for 0.5px noise across 0.80-1.25m depth range. All 13 cameras see tank center at all depths, producing very accurate triangulation in X/Y (sub-micron) but Z error remains in the 0.0002-0.0007mm range due to ray convergence geometry.
- **water_z convention**: In `build_synthetic_rig`, `water_z = height_above_water` (0.75m) because AquaCal places camera centers at world Z=0 and Z increases downward (into water). Fish at Z > 0.75 are underwater.
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

The Z/XY anisotropy is **132x mean (30x-577x range)** for the 13-camera top-down rig with 0.5px pixel noise. This is a fundamental geometric property: top-down cameras have rays that are nearly parallel in Z, so small pixel perturbations have minimal effect on X/Y reconstruction but large effect on Z. Phase 4 should apply a Z-axis loss weight approximately 100x smaller than the X/Y weight.

## Next Phase Readiness

- `aquapose.calibration` module is fully complete — loader, projection, and uncertainty characterization all available
- Z/XY anisotropy (132x) is quantified and documented — Phase 4 optimizer weighting is now informed
- Phase 1 blocker "Z-uncertainty budget not yet quantified" is resolved

---
*Phase: 01-calibration-and-refractive-geometry*
*Completed: 2026-02-19*
