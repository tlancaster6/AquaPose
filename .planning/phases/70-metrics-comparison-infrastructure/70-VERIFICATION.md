---
phase: 70
status: passed
verified: 2026-03-06
verifier: inline
score: 6/6
---

# Phase 70 Verification: Metrics & Comparison Infrastructure

## Phase Goal

All evaluation metrics extended and ready before the iteration loop starts, so every round is measured consistently.

## Requirement Cross-Reference

| Requirement | Description | Plan | Status |
|-------------|-------------|------|--------|
| EVAL-01 | Reprojection error percentiles (p50, p90, p95) | 70-01 | PASS |
| EVAL-02 | Midline confidence percentiles (p10, p50, p90) | 70-01 | PASS |
| EVAL-03 | Camera count percentiles (p50, p90) | 70-01 | PASS |
| EVAL-04 | Per-keypoint reprojection error (mean + p90 per body point) | 70-02 | PASS |
| EVAL-05 | Curvature-stratified reconstruction quality | 70-02 | PASS |
| EVAL-06 | 3D track fragmentation analysis | 70-01 | PASS |

## Must-Have Verification

### Plan 70-01 Must-Haves

| # | Truth/Artifact | Verified |
|---|----------------|----------|
| 1 | ReconstructionMetrics has p50/p90/p95_reprojection_error fields | YES - reconstruction.py lines 55-57 |
| 2 | MidlineMetrics has p10/p50/p90_confidence fields | YES - midline.py |
| 3 | AssociationMetrics has p50/p90_camera_count fields | YES - association.py |
| 4 | FragmentationMetrics dataclass exists | YES - fragmentation.py |
| 5 | evaluate_fragmentation() pure function exists | YES - fragmentation.py |
| 6 | EvalRunner wires fragmentation stage | YES - runner.py |
| 7 | Text and JSON output include all new metrics | YES - output.py |

### Plan 70-02 Must-Haves

| # | Truth/Artifact | Verified |
|---|----------------|----------|
| 1 | aquapose eval output includes per-keypoint reprojection error | YES - output.py contains Per-Keypoint table |
| 2 | aquapose eval output includes curvature-stratified quality | YES - output.py contains Curvature-Stratified table |
| 3 | Both metrics appear in text and JSON output | YES - output.py has both, to_dict() serializes both |
| 4 | reconstruction.py contains per_point_error field and compute function | YES - compute_per_point_error() present |
| 5 | runner.py loads RefractiveProjectionModel | YES - _load_projection_models() with compute_undistortion_maps |
| 6 | output.py contains per_point_error formatting | YES - Per-Keypoint Reprojection Error section |
| 7 | reconstruction.py imports compute_curvature | YES - from aquapose.training.pseudo_labels |
| 8 | reconstruction.py uses BSpline | YES - from scipy.interpolate import BSpline |

### Key Link Verification

| From | To | Via | Verified |
|------|-----|-----|----------|
| runner.py | RefractiveProjectionModel.project() | Load calibration, build models per camera | YES |
| reconstruction.py | compute_curvature() | Import from training.pseudo_labels | YES |
| reconstruction.py | BSpline | scipy.interpolate | YES |
| stages/__init__.py | compute_per_point_error, compute_curvature_stratified | Public exports | YES |

## Success Criteria (from ROADMAP.md)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | aquapose eval output includes reprojection/confidence/camera percentiles | PASS |
| 2 | aquapose eval output includes per-keypoint reprojection error breakdown | PASS |
| 3 | aquapose eval output includes curvature-stratified reconstruction quality | PASS |
| 4 | aquapose eval output includes 3D track fragmentation analysis | PASS |
| 5 | All new metrics appear in both text and JSON output formats | PASS |

## Test Coverage

- 1087 tests pass, 0 failures
- Lint: clean
- Typecheck: no errors in modified files (2 pre-existing errors in unrelated midline.py)

## Conclusion

All 6 EVAL requirements verified as implemented. Phase goal achieved: evaluation metrics are fully extended and ready for consistent measurement across iteration rounds.
