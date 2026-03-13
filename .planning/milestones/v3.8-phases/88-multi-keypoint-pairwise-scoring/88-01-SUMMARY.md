# Phase 88 Plan 01 Summary: Multi-Keypoint Pairwise Scoring

**Status:** Complete
**Started:** 2026-03-11
**Completed:** 2026-03-11

## What Was Built

Replaced single-centroid ray casting in association scoring with multi-keypoint vectorized scoring. The scorer now casts rays from all 6 anatomical keypoints per detection per frame, computes matched keypoint ray-ray distances (nose-to-nose, head-to-head, etc.), aggregates per-frame via arithmetic mean, and applies the existing soft linear kernel.

## Key Changes

### Config (engine/config.py)
- Added `keypoint_confidence_floor: float = 0.3` to `AssociationConfig`
- Added `aggregation_method: str = "mean"` to `AssociationConfig`

### Protocol (core/association/scoring.py)
- Added `keypoint_confidence_floor` and `aggregation_method` to `AssociationConfigLike` protocol
- Replaced `_batch_score_frames()` with `_batch_score_frames_kpt()` returning `(score_sum, n_skipped)`
- `score_tracklet_pair()` returns 0.0 when either tracklet has `keypoints=None` (no centroid fallback)
- Frames where all keypoints fall below confidence floor are skipped (don't count toward effective_t_shared)
- Removed all centroid-based scoring code

### Tests (test_scoring.py)
- Updated `MockAssociationConfig` with new fields
- Updated `_make_tracklet` to generate default keypoints (K=6 near centroid)
- Updated `MockForwardLUT` with per-pixel ray perturbation for multi-keypoint testing
- Added `TestKeypointScoring` class with 10 new tests: None keypoints, confidence filtering, frame skipping, intersection semantics, vectorized-vs-loop equivalence, LUT round-trip, config fields, single keypoint

## Requirements Addressed

| ID | Status |
|----|--------|
| SCORE-01 | Done — rays cast from multiple keypoints per detection per frame |
| SCORE-02 | Done — keypoints below configurable threshold excluded via intersection mask |
| SCORE-03 | Done — per-keypoint distances aggregated via configurable method (mean default) |
| SCORE-04 | Done — single cast_ray call per camera per batch, no per-pair Python loop |

## Verification

- All 1173 unit tests pass (0 failures)
- Lint clean (ruff)
- Typecheck clean (basedpyright, 0 errors)
- No centroid code in scoring.py (grep verified)
- Vectorized implementation matches loop-based reference (atol=1e-10)
- LUT round-trip convergence within 2mm

## Self-Check: PASSED

## Commits
- `be7c9fa` — feat(88-01): replace centroid scoring with multi-keypoint ray casting
- `9d80b3e` — test(88-01): add multi-keypoint scoring tests and update fixtures

## Key Files

### Created
- `.planning/phases/88-multi-keypoint-pairwise-scoring/88-01-SUMMARY.md`

### Modified
- `src/aquapose/engine/config.py`
- `src/aquapose/core/association/scoring.py`
- `tests/unit/core/association/test_scoring.py`
