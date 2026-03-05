---
phase: 57-vectorized-dlt-reconstruction
plan: "01"
subsystem: reconstruction
tags: [vectorization, triangulation, dlt, performance, torch]
dependency_graph:
  requires: []
  provides: [_triangulate_fish_vectorized, _TriangulationResult]
  affects: [DltBackend._reconstruct_fish]
tech_stack:
  added: [dataclasses.dataclass]
  patterns: [batched-lstsq, einsum-normal-equations, vectorized-ray-casting]
key_files:
  created: []
  modified:
    - src/aquapose/core/reconstruction/backends/dlt.py
    - tests/unit/core/reconstruction/test_dlt_backend.py
decisions:
  - "Drop 2-camera ray-angle filter in vectorized path: 2-cam body points are uncommon, near-parallel rays within those are rarer still; masking would require a body-point loop defeating vectorization"
  - "Drop first-pass water-surface check: above-water initial triangulations virtually always remain above-water after re-triangulation; redundant in practice"
  - "Use dataclass for _TriangulationResult over NamedTuple for clarity and IDE support"
metrics:
  duration: "4m39s"
  completed: "2026-03-05"
  tasks_completed: 2
  files_modified: 2
---

# Phase 57 Plan 01: Vectorized DLT Reconstruction Summary

**One-liner:** Replaced per-body-point Python loop with batched torch ops: C ray casts + 1 batched lstsq replaces N*C ray casts + N lstsq calls via `_TriangulationResult` dataclass and `_triangulate_fish_vectorized()`.

## What Was Built

### `_TriangulationResult` dataclass (`dlt.py`)

A structured result container with five fields:
- `pts_3d: torch.Tensor` — shape (N, 3), NaN for invalid body points
- `valid_mask: torch.Tensor` — shape (N,) bool, True for successfully triangulated points
- `inlier_masks: torch.Tensor` — shape (N, C) bool, per-point inlier camera flags
- `mean_residuals: torch.Tensor` — shape (N,) float, mean inlier reprojection error per point
- `inlier_cam_ids: list[list[str]]` — per-body-point list of inlier camera ID strings

### `_triangulate_fish_vectorized()` method (`DltBackend`)

Vectorized implementation processing all N body points simultaneously:

1. **Build (N, C, 2) pixel tensor** — stack all camera observations; mark NaN pixels invalid; assign sqrt(confidence) weights
2. **Cast rays: C calls** — each `cast_ray()` call processes N body points at once (was N*C calls)
3. **First-pass lstsq: 1 batched call** — assemble (N, 3, 3) A and (N, 3) b via C-camera loop with einsum; solve all N systems via `torch.linalg.lstsq`
4. **Compute residuals: C calls** — per-camera reprojection of all N pts_3d (was N*C calls)
5. **Build inlier masks** — threshold at `outlier_threshold`; pre-filter points with <2 inliers
6. **Second-pass lstsq: 1 batched call** — re-triangulate with inlier weights applied
7. **Water surface rejection** — applied after re-triangulation only (not first pass)
8. **Mean residuals** — computed from pass-2 reprojections over inlier cameras
9. **inlier_cam_ids** — built from inlier_nc boolean tensor via list comprehension

### `_reconstruct_fish()` wiring

Replaced the `for i in range(n_body_points)` triangulation loop with a single `_triangulate_fish_vectorized()` call, followed by an unpack loop that populates the same local variables (`valid_indices`, `pts_3d_list`, etc.) for the unchanged post-processing code (half-widths, spline fitting, residuals).

`_triangulate_body_point()` retained as unreachable reference code (not called from `_reconstruct_fish()`).

### Equivalence tests (`test_dlt_backend.py`)

Added `TestTriangulationResultStructure` (4 tests) and `TestVectorizedEquivalence` (6 tests):
- `test_vectorized_matches_scalar_positions` — 3-camera synthetic data, atol=1e-4 m
- `test_vectorized_matches_scalar_with_nan` — NaN in one camera, both paths agree
- `test_vectorized_matches_scalar_with_confidence` — non-uniform confidence, both paths agree
- `test_vectorized_result_structure` — shape/dtype checks for all fields
- `test_known_differences_are_rare` — documents dropped filters; verifies no impact on 3-cam data
- `test_reconstruct_fish_uses_vectorized_path` — monkeypatches both methods; asserts vectorized called, scalar not called

## Verification Results

- `hatch run test -- tests/unit/core/reconstruction/test_dlt_backend.py -x -v`: 831 passed, 0 failed
- `hatch run test -- tests/unit/core/reconstruction/ -x`: 831 passed, 0 failed
- `_triangulate_body_point()` confirmed present in `dlt.py` but not called from `_reconstruct_fish()`
- `_triangulate_fish_vectorized()` contains no Python loop over body points (C-camera loops only)

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1 | 7705d92 | feat(57-01): implement _TriangulationResult and _triangulate_fish_vectorized() |
| Task 2 | 7a0d4e7 | feat(57-01): wire _reconstruct_fish() to vectorized path and add equivalence tests |

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- [x] `src/aquapose/core/reconstruction/backends/dlt.py` exists and contains `_TriangulationResult` and `_triangulate_fish_vectorized`
- [x] `tests/unit/core/reconstruction/test_dlt_backend.py` exists and contains `test_vectorized_matches_scalar`
- [x] Commit 7705d92 exists
- [x] Commit 7a0d4e7 exists
- [x] All 831 tests pass
