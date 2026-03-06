---
phase: quick-17
plan: 01
subsystem: config, reconstruction, synthetic, io
tags: [config, refactor, cleanup]
dependency_graph:
  requires: []
  provides: [unified n_sample_points config with single source of truth]
  affects: [engine/config, core/reconstruction/utils, io/midline_writer, synthetic/fish, engine/pipeline]
tech_stack:
  added: []
  patterns: [single-source-of-truth config propagation]
key_files:
  created: []
  modified:
    - src/aquapose/engine/config.py
    - src/aquapose/core/reconstruction/utils.py
    - src/aquapose/io/midline_writer.py
    - src/aquapose/synthetic/fish.py
    - src/aquapose/engine/pipeline.py
    - src/aquapose/core/types/reconstruction.py
    - tests/unit/engine/test_config.py
    - tests/unit/synthetic/test_synthetic.py
    - tests/unit/io/test_midline_writer.py
decisions:
  - "PipelineConfig.n_sample_points default changed from 10 to 15 (aligning with actual N_SAMPLE_POINTS constant value)"
  - "n_sample_points propagates to reconstruction.n_sample_points (not midline.n_points)"
  - "midline.n_points YAML field now raises a did-you-mean error via _RENAME_HINTS"
  - "N_SAMPLE_POINTS constant removed entirely; consumers use hardcoded literal 15 as default"
metrics:
  duration: 15min
  completed: 2026-03-03
  tasks_completed: 2
  files_modified: 9
---

# Quick Task 17: Unify n_sample_points Config Summary

Single-source-of-truth config propagation for n_sample_points: removed N_SAMPLE_POINTS constant, removed MidlineConfig.n_points, added ReconstructionConfig.n_sample_points, and unified default to 15 throughout.

## What Was Done

### Task 1: Update config hierarchy and load_config propagation (commit 4cc3f81)

- `PipelineConfig.n_sample_points` default changed from `10` to `15`
- `MidlineConfig.n_points` field removed entirely
- `ReconstructionConfig.n_sample_points: int = 15` field added
- `load_config()` propagation block updated: now propagates `n_sample_points` to `reconstruction.n_sample_points` (was `midline.n_points`)
- `_RENAME_HINTS` updated: `"n_points": "n_sample_points (top-level)"` so YAML using `midline.n_points` gets helpful error
- `pipeline.py`: midline stage construction changed from `config.midline.n_points` to `config.n_sample_points`
- `test_config.py`: updated 3 tests (default is 15, propagation to reconstruction, midline.n_points raises error)

### Task 2: Remove N_SAMPLE_POINTS constant and update all consumers (commit 07a8314)

- `utils.py`: removed `N_SAMPLE_POINTS` constant and its `__all__` entry; kept `SPLINE_K`, `SPLINE_N_CTRL`, `SPLINE_KNOTS`, `MIN_BODY_POINTS`
- `midline_writer.py`: removed `N_SAMPLE_POINTS` import; `n_sample_points` parameter now defaults to literal `15`
- `synthetic/fish.py`: removed `N_SAMPLE_POINTS` import; `FishConfig.n_points` and `generate_fish_half_widths` default to literal `15`
- `reconstruction.py` docstring: replaced `N_SAMPLE_POINTS` references with `n_sample_points`
- `test_synthetic.py`: removed `N_SAMPLE_POINTS` import; replaced all uses with literal `15`
- `test_midline_writer.py`: removed `N_SAMPLE_POINTS` import; replaced all uses with literal `15`

## Deviations from Plan

### Minor deviation: Additional file updated

**Found during:** Task 2
**Issue:** `src/aquapose/core/types/reconstruction.py` had `N_SAMPLE_POINTS` references in docstring
**Fix:** Updated docstring to use `n_sample_points` instead
**Rule:** Rule 2 (auto-fix missing critical functionality — keeping docs accurate)

## Verification Results

All plan verifications pass:

1. `hatch run test -x` — 687 passed, 3 skipped
2. `hatch run check` — lint and typecheck pass (pre-existing typecheck errors unrelated to this change)
3. `grep -r "N_SAMPLE_POINTS" src/` — returns no hits
4. `grep -r "N_SAMPLE_POINTS" tests/` — returns no hits
5. `load_config(None, cli_overrides={"n_animals": 3})` — `c.n_sample_points == 15`, `c.reconstruction.n_sample_points == 15`

## Self-Check

- `src/aquapose/engine/config.py` — FOUND
- `src/aquapose/core/reconstruction/utils.py` — FOUND
- `src/aquapose/io/midline_writer.py` — FOUND
- `src/aquapose/synthetic/fish.py` — FOUND
- `tests/unit/engine/test_config.py` — FOUND
- `tests/unit/synthetic/test_synthetic.py` — FOUND
- `tests/unit/io/test_midline_writer.py` — FOUND
- Task 1 commit `4cc3f81` — FOUND
- Task 2 commit `07a8314` — FOUND

## Self-Check: PASSED
