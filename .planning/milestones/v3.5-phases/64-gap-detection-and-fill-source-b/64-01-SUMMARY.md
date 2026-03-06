---
phase: 64-gap-detection-and-fill-source-b
plan: 01
subsystem: training
tags: [pseudo-labels, gap-detection, gap-classification, gap-fill, bounds-check]

requires:
  - phase: 63-pseudo-label-generation-source-a
    provides: pseudo_labels.py with reproject_spline_keypoints, compute_confidence_score, generate_fish_labels

provides:
  - training.pseudo_labels detect_gaps(), _classify_gap(), generate_gap_fish_labels(), _passes_bounds_check()
  - Unit tests for all gap detection and gap label generation behaviors

affects: [64-02, 65-dataset-assembly]

tech-stack:
  added: []
  patterns: [inverse-lut-visibility-gap-detection, 3-tier-gap-classification]

key-files:
  created: []
  modified:
    - src/aquapose/training/pseudo_labels.py
    - src/aquapose/training/__init__.py
    - tests/unit/training/test_pseudo_labels.py

key-decisions:
  - "Gap detection uses ghost_point_lookup on centroid (mean of control points) for visibility"
  - "Classification uses RefractiveProjectionModel.project() for bbox overlap, not LUT pixel coords"
  - "Bounds check requires >= 50% of visible keypoints within image bounds"
  - "Gap labels use same fish-level confidence score, no discount factor"

patterns-established:
  - "InverseLUT visibility minus contributing cameras for gap identification"
  - "Reverse pipeline stage check for gap classification"

requirements-completed: [GAP-01, GAP-02, GAP-03]

duration: 5min
completed: 2026-03-05
---

# Plan 64-01 Summary

**Added gap detection, classification, and gap-fill label generation to pseudo_labels.py with TDD coverage**

## Performance

- **Duration:** 5 min
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments
- Built detect_gaps(): identifies cameras visible via InverseLUT but absent from per_camera_residuals, respects min_cameras floor
- Built _classify_gap(): 3-tier classification (no-detection, no-tracklet, failed-midline) using projected centroid for bbox overlap
- Built generate_gap_fish_labels(): OBB/pose label generation without per-camera residual check, with bounds validation
- Built _passes_bounds_check(): filters degenerate reprojections where <50% of visible keypoints are in image bounds
- 15 new unit tests covering all specified behaviors
- Updated training/__init__.py exports

## Task Commits

1. **Task 1: Gap detection, classification, and gap-fill label generation** - `6ce03b0` (feat)

## Files Created/Modified
- `src/aquapose/training/pseudo_labels.py` - Added detect_gaps, _classify_gap, generate_gap_fish_labels, _passes_bounds_check + InverseLUT import
- `src/aquapose/training/__init__.py` - Exported detect_gaps and generate_gap_fish_labels
- `tests/unit/training/test_pseudo_labels.py` - 15 new tests for gap detection/classification/label generation/bounds check

## Decisions Made
- Used mean of control points for centroid (consistent with CONTEXT.md)
- Used .cpu().detach() for CUDA safety per CLAUDE.md
- Used try/except ValueError for tracklet frame lookup (tuple.index)

## Deviations from Plan
None - plan executed as specified.

## Issues Encountered
None.

## Next Phase Readiness
- Core functions ready for CLI wiring in Plan 64-02
- All functions are importable and tested

---
*Phase: 64-gap-detection-and-fill-source-b*
*Completed: 2026-03-05*
