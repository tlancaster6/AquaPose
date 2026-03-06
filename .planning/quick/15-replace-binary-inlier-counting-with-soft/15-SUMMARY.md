---
phase: quick-15
plan: 01
subsystem: association-scoring
tags: [scoring, association, soft-kernel, ghost-penalty-removal]
dependency_graph:
  requires: []
  provides: [soft-linear-scoring-kernel]
  affects: [association-stage, tune-association-script]
tech_stack:
  added: []
  patterns: [soft-linear-kernel, protocol-attribute-removal]
key_files:
  created: []
  modified:
    - src/aquapose/core/association/scoring.py
    - src/aquapose/engine/config.py
    - src/aquapose/core/association/stage.py
    - scripts/tune_association.py
    - tests/unit/core/association/test_scoring.py
decisions:
  - "Soft linear kernel 1 - (dist / threshold) replaces binary inlier counting"
  - "Ghost penalty fully removed; AssociationConfigLike no longer requires ghost_pixel_threshold"
  - "detections parameter removed from score_tracklet_pair and score_all_pairs signatures"
metrics:
  duration_minutes: 4
  tasks_completed: 2
  tasks_total: 2
  files_modified: 5
  completed_date: "2026-03-03"
---

# Quick Task 15: Replace Binary Inlier Counting with Soft Scoring Kernel

**One-liner:** Soft linear kernel `1 - (dist / threshold)` replaces binary inlier counting in association scoring; ghost penalty and `ghost_pixel_threshold` fully removed.

## What Was Done

Binary inlier counting in `score_tracklet_pair()` discarded distance magnitude — a correct pair at 0.41 cm and a wrong pair at 1.86 cm both scored ~0.98, making community detection impossible. The fix translates the 4.5x distance difference into a ~2.3x score difference via a soft kernel.

## Tasks

### Task 1: Replace Binary Inlier Counting with Soft Kernel and Remove Ghost Penalty

**Commit:** `f76e676`

Changes to `src/aquapose/core/association/scoring.py`:
- `score_tracklet_pair()`: replaced `inlier_count += 1` with `score_sum += 1.0 - (dist / config.ray_distance_threshold)`
- Removed `ghost_point_lookup` import and all ghost penalty code (`ghost_ratios`, `scoring_cameras`, `mid_tensor`, `visibility`, `other_cams`, `n_visible_other`, `n_negative`, `mean_ghost`)
- Combined score changed from `f * (1.0 - mean_ghost) * w` to `f * w`
- Removed `detections` and `inverse_lut` parameters from `score_tracklet_pair()` signature
- Removed `detections` parameter from `score_all_pairs()` signature
- Updated `AssociationConfigLike` protocol: removed `ghost_pixel_threshold` attribute
- Updated module docstring to remove ghost-point penalty references

Changes to `src/aquapose/engine/config.py`:
- Removed `ghost_pixel_threshold: float = 50.0` field from `AssociationConfig`
- Removed `ghost_pixel_threshold` from class docstring

Changes to `src/aquapose/core/association/stage.py`:
- Removed `_extract_centroids()` helper function (no longer needed)
- Updated `score_all_pairs()` call to remove `det_centroids` and `detections` arguments
- Removed "Extract detection centroids for ghost penalty" comment

Changes to `scripts/tune_association.py`:
- Removed `ghost_pixel_threshold` from `SWEEP_RANGES` dict
- Removed `ghost_pixel_threshold` from `PRIMARY_STAGES` list

### Task 2: Update Tests for Soft Scoring and Remove Ghost Penalty Tests

**Commit:** `d9e48a7`

Changes to `tests/unit/core/association/test_scoring.py`:
- `MockAssociationConfig`: removed `ghost_pixel_threshold` field
- `MockInverseLUT` class removed (no longer needed by scoring tests)
- `MockInverseLUTNonAdjacent` kept for `TestScoreAllPairs` adjacency test
- `test_perfect_match`: updated call (removed `inv_lut`, `detections` args); updated assertion to `score == pytest.approx(0.2)` (exact soft kernel result for perfect intersection at distance 0)
- `test_no_overlap`, `test_below_t_min`, `test_early_termination`: removed `inv_lut` and empty-list `detections` args
- `TestGhostPenalty` class removed entirely
- `TestScoreAllPairs`: removed `detections` arg from `score_all_pairs()` calls
- New test `test_soft_scoring_distance_sensitivity`: creates two tracklet pairs (perfect intersection vs ~0.02m separation under threshold=0.03), validates `score_close > score_far`

## Verification

Grep verification confirms ghost penalty fully removed:
```
grep -rn "ghost_pixel_threshold|ghost_ratio|ghost_point_lookup|n_visible_other|ghost_ratios" \
  src/aquapose/core/association/scoring.py src/aquapose/engine/config.py \
  tests/unit/core/association/test_scoring.py
# Returns nothing
```

Soft kernel present:
```
grep -n "1.0 - (dist" src/aquapose/core/association/scoring.py
# 191:            score_sum += 1.0 - (dist / config.ray_distance_threshold)
```

Test results: 749 passed, 0 failures, all association tests pass.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated AssociationStage.run() to match new score_all_pairs signature**
- **Found during:** Task 1
- **Issue:** `stage.py` called `score_all_pairs()` with the old signature including `det_centroids` and `detections` positional args
- **Fix:** Updated call to match new signature; removed `_extract_centroids()` helper function from `stage.py`
- **Files modified:** `src/aquapose/core/association/stage.py`
- **Commit:** `f76e676`

**2. [Rule 2 - Missing] Removed MockInverseLUT from TestScoreAllPairs.test_filters_by_score_min**
- **Found during:** Task 2
- **Issue:** The test used `MockInverseLUT` (removed) for the `score_all_pairs` call; the plan noted `inverse_lut` is still needed for overlap graph
- **Fix:** Used `MockInverseLUTNonAdjacent` instead (which has the structure `camera_overlap_graph()` expects); this is actually the correct fixture for this test since it defines cam_a-cam_b adjacency
- **Files modified:** `tests/unit/core/association/test_scoring.py`
- **Commit:** `d9e48a7`

## Self-Check: PASSED

- FOUND: src/aquapose/core/association/scoring.py
- FOUND: src/aquapose/engine/config.py
- FOUND: tests/unit/core/association/test_scoring.py
- FOUND: .planning/quick/15-replace-binary-inlier-counting-with-soft/15-SUMMARY.md
- FOUND: commit f76e676 (Task 1)
- FOUND: commit d9e48a7 (Task 2)
