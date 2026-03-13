---
phase: 89-fragment-merging-removal
plan: "01"
subsystem: association
tags: [leiden, clustering, tracklet, pipeline-cleanup]

# Dependency graph
requires:
  - phase: 88-multi-keypoint-ray-scoring
    provides: association pipeline with multi-keypoint scoring (no fragment merging needed)
provides:
  - Leiden clustering without fragment merging (merge_fragments deleted)
  - AssociationConfig without max_merge_gap field
  - ClusteringConfigLike protocol with 3 fields only (score_min, expected_fish_count, leiden_resolution)
  - Tracklet2D docstring with only "detected" and "coasted" as valid frame_status values
affects:
  - 90-group-validation
  - any consumer of AssociationConfig YAML (max_merge_gap field silently ignored if present)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ClusteringConfigLike protocol: 3 fields only — score_min, expected_fish_count, leiden_resolution"
    - "frame_status valid values: 'detected' and 'coasted' only (interpolated removed)"

key-files:
  created: []
  modified:
    - src/aquapose/core/association/clustering.py
    - src/aquapose/core/association/stage.py
    - src/aquapose/core/association/__init__.py
    - src/aquapose/engine/config.py
    - src/aquapose/core/tracking/types.py
    - tests/unit/core/association/test_clustering.py

key-decisions:
  - "Fragment merging removed permanently: works against upstream fragmentation intent (fragmented tracklets are expected and desirable for Phase 90 group validation)"
  - "max_merge_gap removed from AssociationConfig and ClusteringConfigLike protocol"
  - "No interpolated frame_status: only detected and coasted remain valid"

patterns-established:
  - "Association pipeline is now SPECSEED Steps 2-3 only (no Step 4 fragment merge)"

requirements-completed:
  - CLEAN-01

# Metrics
duration: 4min
completed: "2026-03-11"
---

# Phase 89 Plan 01: Fragment Merging Removal Summary

**Deleted `merge_fragments` and all helpers from clustering.py, removed `max_merge_gap` from config and protocol, and scrubbed all "interpolated" frame_status references from the codebase**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-11T19:18:07Z
- **Completed:** 2026-03-11T19:21:54Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Removed `merge_fragments`, `_merge_cam_fragments`, `_try_merge_pair` (~220 lines) from clustering.py
- Removed `max_merge_gap` field from `AssociationConfig` and `ClusteringConfigLike` protocol
- Removed Step 4 fragment merge call from `AssociationStage.run()`
- Deleted `TestMergeFragments` class (4 tests) from test_clustering.py
- Cleaned up `Tracklet2D` docstring to remove "interpolated" status and fragment merging references

## Task Commits

Each task was committed atomically:

1. **Task 1: Delete merge_fragments code, config field, and tests** - `9fc1c2e` (feat)
2. **Task 2: Clean up interpolated frame_status references and stale docstrings** - `fd5cb98` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `src/aquapose/core/association/clustering.py` - Removed fragment merging section (~220 lines), updated module docstring, __all__, ClusteringConfigLike protocol, and build_must_not_link docstring
- `src/aquapose/core/association/stage.py` - Removed merge_fragments import and Step 4 call; renumbered Step 5 to Step 4
- `src/aquapose/core/association/__init__.py` - Removed merge_fragments from import and __all__
- `src/aquapose/engine/config.py` - Removed max_merge_gap field and docstring entry from AssociationConfig
- `src/aquapose/core/tracking/types.py` - Updated Tracklet2D docstring: only "detected" and "coasted" valid, removed fragment merging example
- `tests/unit/core/association/test_clustering.py` - Removed merge_fragments import, max_merge_gap from MockClusteringConfig, deleted TestMergeFragments class

## Decisions Made
- Fragment merging removed permanently — it works against upstream fragmentation intent; Phase 90 group validation provides richer signal
- The `"interpolated"` frame_status value is fully retired; only `"detected"` and `"coasted"` remain valid

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- ruff-format hook reformatted clustering.py (trailing newline) on first commit attempt; re-staged and committed cleanly on second attempt.

## Next Phase Readiness
- CLEAN-01 satisfied: merge_fragments and max_merge_gap fully removed
- Phase 90 (group validation) can proceed with a cleaner association pipeline
- No pipeline breakage: all 1169 tests pass, lint and typecheck clean

---
*Phase: 89-fragment-merging-removal*
*Completed: 2026-03-11*
