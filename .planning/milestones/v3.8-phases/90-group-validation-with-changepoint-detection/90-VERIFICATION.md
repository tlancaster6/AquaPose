---
phase: 90-group-validation-with-changepoint-detection
status: passed
verified: 2026-03-11
requirement_ids: [VALID-01, VALID-02, VALID-03, VALID-04, CLEAN-02]
---

# Phase 90 Verification: Group Validation with Changepoint Detection

**Goal**: After clustering, each group is audited for temporal ID swaps and outliers; swapped tracklets are split or evicted; refinement.py is deleted

## Must-Have Verification

### Plan 90-01: validation.py with multi-keypoint residuals and changepoint detection

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Each tracklet in a group gets a per-frame residual series from multi-keypoint ray distances against the rest of the group | PASS | `_compute_tracklet_residuals()` computes per-frame mean ray-ray distance between target's confident keypoints and others; test_all_converging_no_eviction validates |
| 2 | A tracklet with a temporal swap is split at the changepoint into consistent and inconsistent segments | PASS | `_classify_tracklet()` -> `_find_changepoint_by_run()` -> `_split_tracklet_at()`; test_swap_detected_and_split validates |
| 3 | A tracklet with uniformly high residual is evicted as a singleton candidate | PASS | `_classify_tracklet()` returns "evict" when no changepoint and <50% consistent; test_outlier_evicted validates |
| 4 | A mostly-consistent tracklet (>50% frames below threshold) is kept as-is | PASS | `_classify_tracklet()` checks fraction_consistent > 0.5; test_mostly_consistent_kept validates |
| 5 | Split tracklets become new Tracklet2D instances with unique track_ids | PASS | `_split_tracklet_at()` constructs new frozen dataclasses; test_split_tracklets_have_unique_ids validates |
| 6 | Output TrackletGroups have per_frame_confidence and consensus_centroids populated | PASS | `_compute_frame_consensus()` and `_compute_per_frame_confidence()` called after membership changes; test_confidence_populated and test_consensus_centroids_populated validate |

| # | Artifact | Status | Evidence |
|---|----------|--------|----------|
| 1 | `src/aquapose/core/association/validation.py` exports ValidationConfigLike and validate_groups | PASS | `from aquapose.core.association import validate_groups, ValidationConfigLike` succeeds |
| 2 | `tests/unit/core/association/test_validation.py` with >=100 lines | PASS | 781 lines, 14 test methods |

| # | Key Link | Status | Evidence |
|---|----------|--------|----------|
| 1 | validation.py imports ray_ray_closest_point from scoring.py | PASS | Line 17: `from aquapose.core.association.scoring import ray_ray_closest_point` |
| 2 | validation.py constructs TrackletGroup instances | PASS | 4 TrackletGroup() construction calls in validate_groups() |

### Plan 90-02: Pipeline wiring and refinement deletion

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | stage.py calls validate_groups instead of refine_clusters | PASS | Lines 102-104: `from aquapose.core.association.validation import validate_groups; groups = validate_groups(...)` |
| 2 | AssociationConfig has validation_enabled, min_cameras_validate, min_segment_length fields and no longer has refinement_enabled, min_cameras_refine | PASS | `isinstance(AssociationConfig(), ValidationConfigLike)` returns True; grep for refinement fields returns no hits in src/ |
| 3 | refinement.py is deleted from the codebase | PASS | File does not exist at `src/aquapose/core/association/refinement.py` |
| 4 | association __init__.py exports ValidationConfigLike and validate_groups, not RefinementConfigLike and refine_clusters | PASS | `__all__` contains ValidationConfigLike and validate_groups; no RefinementConfigLike or refine_clusters |
| 5 | All existing tests pass (no downstream breakage) | PASS | 1168 tests pass, 3 skipped, 0 failures |

| # | Artifact | Status | Evidence |
|---|----------|--------|----------|
| 1 | config.py contains min_segment_length | PASS | `AssociationConfig().min_segment_length == 10` |
| 2 | stage.py contains validate_groups | PASS | grep confirms import and call |

| # | Key Link | Status | Evidence |
|---|----------|--------|----------|
| 1 | stage.py -> validation.py via lazy import | PASS | `from aquapose.core.association.validation import validate_groups` |
| 2 | AssociationConfig satisfies ValidationConfigLike | PASS | `isinstance(AssociationConfig(), ValidationConfigLike)` returns True |

## Requirement Traceability

| Requirement | Status | Evidence |
|-------------|--------|----------|
| VALID-01 | PASS | `_compute_tracklet_residuals()` computes per-tracklet multi-keypoint residuals against rest of group |
| VALID-02 | PASS | `_find_changepoint_by_run()` identifies temporal swap points via longest-consistent-run heuristic |
| VALID-03 | PASS | `_split_tracklet_at()` creates two new Tracklet2D instances; consistent segment stays, inconsistent becomes singleton |
| VALID-04 | PASS | `_classify_tracklet()` returns "evict" for tracklets with <50% consistent frames and no changepoint |
| CLEAN-02 | PASS | refinement.py deleted; zero references remain in src/ or tests/ |

## Test Results

- `hatch run test -x`: 1168 passed, 3 skipped, 0 failures
- `hatch run check`: All checks passed (lint + typecheck)
- `grep -r "refine_clusters\|RefinementConfigLike\|refinement_enabled\|min_cameras_refine" src/ tests/`: No matches

## Score

**5/5 requirements verified. Phase goal achieved.**
