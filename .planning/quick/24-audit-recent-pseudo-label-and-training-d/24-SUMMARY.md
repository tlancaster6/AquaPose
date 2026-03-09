---
phase: 24-audit-recent-pseudo-label-and-training-d
plan: 24
subsystem: training
tags: [audit, code-quality, technical-debt]
dependency_graph:
  requires: []
  provides: [prioritized-audit-report]
  affects: [training-module]
tech_stack:
  added: []
  patterns: []
key_files:
  created:
    - .planning/quick/24-audit-recent-pseudo-label-and-training-d/24-AUDIT-REPORT.md
  modified: []
decisions:
  - No code changes made -- audit only, user decides what to pursue
metrics:
  duration: ~3min
  completed: "2026-03-09T19:40:41Z"
---

# Quick Task 24: Training Module Audit Summary

Comprehensive code quality audit of src/aquapose/training/ (21 files, ~8400 LOC) and tests/unit/training/ (15 files, ~5800 LOC) covering structural debt, interface leaks, test gaps, and data flow issues accumulated during phases 66-73.

## What Was Done

Read and analyzed all 21 source files and 15 test files. Applied the repo-scope elegance review checklist focusing on: duplicated logic, catch-all modules, missing abstractions, inconsistent patterns, dead code, leaky abstractions, test coverage gaps, and data flow issues.

## Key Findings

14 findings total, categorized and prioritized:

- **5 Structural**: Near-identical YOLO wrappers (3 files), duplicated CLI training commands (with a concrete bug: seg missing model registration), duplicated `_LutConfigFromDict`, duplicated `affine_warp_crop`/`transform_keypoints`, duplicated arc-length computation
- **2 Interface**: SampleStore internals accessed via `_connect()` in CLI code, hardcoded "pose" detection in `store.assemble()`
- **4 Test Coverage**: YOLO wrappers (23 LOC tests for 328 LOC source), elastic deform CLI (28 LOC stub), `select_diverse_subset.py` (0 tests for 330 LOC), `datasets.py`/`coco_convert.py` (no unit tests)
- **2 Data Flow**: Inline reimplementation of pose label parsing in `import_cmd`, 500-line monolithic `generate` command
- **1 Code Quality**: `import_cmd` mixes import/metadata/augmentation/reporting concerns

## Concrete Bug Found

Finding 2: The `seg` training command in `cli.py` is missing `register_trained_model` -- it was added to `obb` and `pose` but not `seg`. This means seg models are trained but never registered in the SampleStore or written to the project config.

## Recommended Priority

- **Batch 1** (Quick wins, ~1.5hr): Consolidate duplicated functions, consolidate YOLO wrappers
- **Batch 2** (Moderate, ~2hr): Fix CLI commands + seg bug, write tests for `select_diverse_subset`
- **Batch 3** (When touching, ~3hr): Add SampleStore public API, CLI integration tests
- **Batch 4** (If extending, ~4hr): Refactor pseudo-label generator, comprehensive tests

## Deviations from Plan

None -- plan executed exactly as written.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 54e166a | Audit report with 14 prioritized findings |

## Self-Check: PASSED

- [x] 24-AUDIT-REPORT.md exists (200 lines, exceeds 80-line minimum)
- [x] Report covers all 21 source files
- [x] Each finding has LOE and ROI ratings
- [x] Priority matrix and recommended execution order included
- [x] No code changes made (audit only)
