---
phase: 93-config-plumbing
plan: "01"
subsystem: reconstruction
tags: [config, plumbing, n_sample_points, reconstruction]
dependency_graph:
  requires: []
  provides: [n_sample_points config wiring end-to-end]
  affects: [ReconstructionStage, pipeline.py, ReconstructionConfig, PipelineConfig]
tech_stack:
  added: []
  patterns: [TDD, config propagation, frozen dataclass defaults]
key_files:
  created: []
  modified:
    - src/aquapose/engine/config.py
    - src/aquapose/core/reconstruction/stage.py
    - src/aquapose/engine/pipeline.py
    - tests/unit/engine/test_config.py
    - tests/unit/engine/test_build_stages.py
    - tests/unit/core/reconstruction/test_reconstruction_stage.py
decisions:
  - "n_sample_points default changed from 15 to 6 to match the 6-keypoint identity mapping"
  - "Added _DEFAULT_N_SAMPLE_POINTS module constant in stage.py following existing naming pattern"
metrics:
  duration: "~8 minutes"
  completed: "2026-03-13"
  tasks_completed: 2
  files_modified: 6
---

# Phase 93 Plan 01: Wire n_sample_points Config Plumbing Summary

**One-liner:** End-to-end `n_sample_points` wiring from `ReconstructionConfig` through `pipeline.py` to `ReconstructionStage`, with default changed from 15 to 6 for the 6-keypoint identity mapping.

## What Was Built

Completed the config plumbing for `n_sample_points` across three files:

1. **`src/aquapose/engine/config.py`** — Changed `ReconstructionConfig.n_sample_points` default from 15 to 6, changed `PipelineConfig.n_sample_points` default from 15 to 6, updated propagation fallback from 15 to 6, updated docstring to say "Default 6".

2. **`src/aquapose/core/reconstruction/stage.py`** — Added `_DEFAULT_N_SAMPLE_POINTS = 6` module constant; added `n_sample_points: int = _DEFAULT_N_SAMPLE_POINTS` parameter to `__init__`; stores as `self._n_sample_points`; replaced hardcoded `n_points=15` at the `_keypoints_to_midline()` call site with `n_points=self._n_sample_points`; updated class docstring and module docstring.

3. **`src/aquapose/engine/pipeline.py`** — Added `n_sample_points=config.reconstruction.n_sample_points` to the `ReconstructionStage(...)` constructor call in `build_stages()`.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Failing tests for n_sample_points=6 wiring | 20842bd | test_config.py, test_build_stages.py, test_reconstruction_stage.py |
| 1 (GREEN) | Wire n_sample_points through stage and pipeline | c73a7b0 | config.py, stage.py, pipeline.py, test_config.py, test_reconstruction_stage.py |
| 2 | Verify no hardcoded 15; full suite passes | (via c73a7b0) | — |

## Verification

- `grep -n "n_points=15\|n_sample_points.*15" src/aquapose/core/reconstruction/stage.py` — no matches
- `ReconstructionConfig().n_sample_points == 6` — PASS
- `PipelineConfig().n_sample_points == 6` — PASS
- `hatch run test -x` — 1204 passed, 3 skipped

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- [x] `src/aquapose/engine/config.py` modified — contains `n_sample_points: int = 6`
- [x] `src/aquapose/core/reconstruction/stage.py` modified — contains `self._n_sample_points`
- [x] `src/aquapose/engine/pipeline.py` modified — contains `n_sample_points=config.reconstruction.n_sample_points`
- [x] Commits 20842bd and c73a7b0 exist
- [x] Full test suite passes (1204 passed)
