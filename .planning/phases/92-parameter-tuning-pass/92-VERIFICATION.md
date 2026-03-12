---
phase: 92-parameter-tuning-pass
verified: 2026-03-12T23:50:00Z
status: passed
score: 4/4 success criteria verified
gaps: []
human_verification:
  - test: "Verify overlay visualizations from run_20260312_151712 show correct fish associations"
    expected: "Fish identities are consistent across cameras and frames; no obvious ID swaps"
    why_human: "Visual correctness of association overlays cannot be verified programmatically"
---

# Phase 92: Parameter Tuning Pass Verification Report

**Phase Goal:** New config parameters are empirically calibrated on real data; the final v3.8 association configuration is documented and validated against the v3.7 baseline
**Verified:** 2026-03-12T23:50:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `aquapose tune --stage association` runs a parameter grid over new config fields on cached tracking outputs | VERIFIED | `sweep_association` in tuning.py (line 388-430) runs 27-combo 3D joint grid over ray_distance_threshold x score_min x keypoint_confidence_floor, then carry-forward for eviction_reproj_threshold, leiden_resolution, early_k. 92-RESULTS.md documents all 27 joint combos + carry-forward results. |
| 2 | Tuned parameters produce measurable reduction in singleton rate vs v3.7 baseline (target ~15%, floor: better than 27%) | VERIFIED | Singleton rate: 5.4% (v3.8) vs 27% (v3.7) = 80% reduction. Documented in 92-RESULTS.md. Exceeds target. |
| 3 | Reprojection error and grouping quality not degraded relative to v3.7 baseline | VERIFIED | Mean reproj error: 2.85px (v3.8) vs ~3.0px (v3.7) = 5% improvement. Fish yield: 102.6% (stable). |
| 4 | Tuned config defaults committed; results document records sweep ranges, selected values, and metric comparison | VERIFIED | 92-RESULTS.md documents methodology, grid ranges, all results, and conclusion. Config defaults in config.py match sweep winner for the three key parameters changed: ray_distance_threshold=0.01, score_min=0.30, keypoint_confidence_floor=0.20. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/engine/config.py` | `use_multi_keypoint_scoring` toggle, tuned defaults | VERIFIED | Line 202: `use_multi_keypoint_scoring: bool = True`. Defaults: ray_distance_threshold=0.01, score_min=0.3, keypoint_confidence_floor=0.2. |
| `src/aquapose/core/association/scoring.py` | `_score_pair_centroid_only` + toggle branch | VERIFIED | Lines 196-301: substantive centroid-only scoring (ray casting, early termination, overlap reliability). Line 333: toggle branch before keypoints-None check. |
| `src/aquapose/evaluation/stages/association.py` | `DEFAULT_GRID` with `keypoint_confidence_floor` | VERIFIED | Line 15: `keypoint_confidence_floor: [0.2, 0.3, 0.4]`. Grid has 6 keys total. |
| `src/aquapose/evaluation/tuning.py` | 3D joint grid in `sweep_association` | VERIFIED | Lines 389-393: joint_params includes ray_distance_threshold, score_min, keypoint_confidence_floor. |
| `src/aquapose/cli.py` | config.yaml fallback in tune_cmd | VERIFIED | Lines 360-367: falls back from config_exhaustive.yaml to config.yaml. |
| `.planning/phases/92-parameter-tuning-pass/92-RESULTS.md` | Tuning results document | VERIFIED | 119-line document with methodology, 27-combo grid table, carry-forward table, winner comparison, and conclusion. |
| `tests/unit/core/association/test_scoring.py` | TestCentroidOnlyScoring class | VERIFIED | Line 954: test class with tests for centroid-only scoring path (nonzero scores, value in [0,1], toggle behavior). |
| `tests/unit/engine/test_config.py` | use_multi_keypoint_scoring tests | VERIFIED | Lines 475-488: default=True and False construction tests. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| scoring.py | config.py | `use_multi_keypoint_scoring` toggle | WIRED | Line 333: `if not config.use_multi_keypoint_scoring:` branches to centroid-only path |
| tuning.py | association.py | joint_params reads from DEFAULT_GRID | WIRED | Line 394: `joint_values = [grid[p] for p in joint_params]` reads keypoint_confidence_floor from grid |
| config.py | 92-RESULTS.md | defaults match documented winner | PARTIAL | ray_distance_threshold (0.01), score_min (0.30), keypoint_confidence_floor (0.20) match. See note below. |

**Documentation-code mismatch (non-blocking):** 92-RESULTS.md "Final Configuration" section lists `early_k: 10` and `eviction_reproj_threshold: 0.02`, but actual config.py defaults are `early_k: 5` and `eviction_reproj_threshold: 0.025`. The sweep data shows these pairs are functionally equivalent (identical metrics), so the mismatch has zero impact on behavior. The code values were set in commit 15da978 during the interactive tuning session. This is a documentation accuracy issue, not a functional gap.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| EVAL-01 | 92-01, 92-02 | Parameter tuning pass on real data measuring singleton rate, reprojection error, and grouping quality vs v3.7 baseline | SATISFIED | 27-combo grid sweep + carry-forward documented in 92-RESULTS.md; singleton rate 5.4%, reproj error 2.85px, yield 102.6% |
| EVAL-02 | 92-02 | End-to-end pipeline run with tuned parameters confirms improvement over v3.7 | SATISFIED | E2E run run_20260312_151712 completed successfully per 92-RESULTS.md; singleton rate improved from 27% to 5.4% |

No orphaned requirements for Phase 92.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | - |

No TODOs, FIXMEs, placeholders, or stub implementations found in modified files.

### Human Verification Required

### 1. Visual Association Quality

**Test:** Open overlay visualizations from run_20260312_151712 and inspect fish identity assignments across cameras and frames.
**Expected:** Fish IDs are consistent across all 12 cameras; no visible ID swaps or singleton gaps in the overlays.
**Why human:** Visual correctness of multi-camera association cannot be verified programmatically from code alone.

### Gaps Summary

No blocking gaps found. All four ROADMAP success criteria are verified. Both requirements (EVAL-01, EVAL-02) are satisfied. The only noted issue is a minor documentation-code mismatch in 92-RESULTS.md for two parameters (`early_k` and `eviction_reproj_threshold`) where the documented "final configuration" values differ from the actual code defaults. The sweep data confirms these differences have zero metric impact, so this does not block goal achievement.

---

_Verified: 2026-03-12T23:50:00Z_
_Verifier: Claude (gsd-verifier)_
