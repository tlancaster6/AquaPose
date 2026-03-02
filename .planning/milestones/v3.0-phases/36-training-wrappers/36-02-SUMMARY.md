---
phase: 36-training-wrappers
plan: 02
subsystem: training
tags: [yolo, ultralytics, ndjson, seg, pose, cli, click]

# Dependency graph
requires:
  - phase: 36-01
    provides: yolo_seg.py, yolo_pose.py, CLI seg/pose subcommands, unit tests
provides:
  - NDJSON-to-YOLO.txt conversion for seg (polygon format)
  - NDJSON-to-YOLO.txt conversion for pose (bbox + keypoints format)
  - data.yaml rewrite with absolute paths for both wrappers
  - train_yolo_seg() and train_yolo_pose() training functions
  - aquapose train seg and aquapose train pose CLI subcommands
  - Comprehensive unit tests for all conversion and rewrite logic
affects: [37-pipeline-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "NDJSON-to-YOLO.txt conversion via private _convert_*_ndjson_to_txt helpers"
    - "data.yaml rewrite pattern: absolute path: field + images/train, images/val entries"
    - "Lazy ultralytics import inside train function (same as yolo_obb.py pattern)"

key-files:
  created:
    - src/aquapose/training/yolo_seg.py
    - src/aquapose/training/yolo_pose.py
    - tests/unit/training/test_yolo_seg.py
    - tests/unit/training/test_yolo_pose.py
  modified:
    - src/aquapose/training/cli.py
    - src/aquapose/training/__init__.py
    - tests/unit/training/test_training_cli.py

key-decisions:
  - "New seg/pose CLI subcommands use --model (full model name string) instead of --model-size (suffix only) — consistent with CONTEXT.md decision"
  - "NDJSON conversion is done before calling model.train(); label files written to labels/{split}/ alongside images/{split}/"
  - "data.yaml rewritten to data_ultralytics.yaml with absolute path to prevent Ultralytics DATASETS_DIR resolution issues"
  - "Pose wrapper preserves kpt_shape, kpt_names, flip_idx from original data.yaml — nothing hardcoded"

patterns-established:
  - "Training wrapper pattern: read NDJSON refs from data.yaml, convert labels, rewrite yaml with abs path, call model.train()"
  - "Private conversion helpers: _convert_seg_ndjson_to_txt, _convert_pose_ndjson_to_txt (module-private, testable via direct import)"

requirements-completed: [TRAIN-01, TRAIN-02]

# Metrics
duration: 5min
completed: 2026-03-01
---

# Phase 36 Plan 02: Training Wrappers Summary

**YOLO-seg and YOLO-pose training wrappers with NDJSON-to-YOLO.txt conversion, data.yaml rewrite, and CLI subcommands (`aquapose train seg`, `aquapose train pose`)**

## Performance

- **Duration:** ~5 min (continuation of 36-01 work)
- **Started:** 2026-03-01T21:18:00Z
- **Completed:** 2026-03-01T21:19:00Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- `yolo_seg.py` and `yolo_pose.py` training wrappers with full NDJSON-to-YOLO.txt conversion pipelines
- `aquapose train seg` and `aquapose train pose` CLI subcommands with all required flags
- Unit tests covering NDJSON conversion, data.yaml rewrite, empty-input guards, and CLI help output
- Import boundary maintained: training/ modules do not import from aquapose.engine or aquapose.cli

## Task Commits

Each task was committed atomically:

1. **Task 1: Create yolo_seg.py and yolo_pose.py training wrappers** - `e1ecf6b` (feat)
2. **Task 2: Add seg/pose CLI subcommands, update __init__.py, add tests** - `3e802c0` (feat)

## Files Created/Modified
- `src/aquapose/training/yolo_seg.py` - YOLO-seg training wrapper: NDJSON seg conversion, data.yaml rewrite, train_yolo_seg()
- `src/aquapose/training/yolo_pose.py` - YOLO-pose training wrapper: NDJSON pose conversion, data.yaml rewrite, train_yolo_pose()
- `src/aquapose/training/cli.py` - Added `seg` and `pose` subcommands to the train group
- `src/aquapose/training/__init__.py` - Added train_yolo_seg and train_yolo_pose to exports and __all__
- `tests/unit/training/test_yolo_seg.py` - Unit tests for seg NDJSON conversion, data.yaml rewrite, empty input guard, polygon coord count
- `tests/unit/training/test_yolo_pose.py` - Unit tests for pose NDJSON conversion, data.yaml rewrite with kpt fields, empty input guard
- `tests/unit/training/test_training_cli.py` - Added seg/pose help flag tests; updated removed-commands test to check only for "unet" absence

## Decisions Made
- `--model` flag uses full model name string (e.g., `yolo26n-seg`) rather than size suffix — consistent with CONTEXT.md decision for seg/pose (OBB keeps `--model-size`)
- NDJSON conversion writes to `labels/{split}/` directory alongside `images/{split}/` — Ultralytics standard layout
- data.yaml rewritten to `data_ultralytics.yaml` with absolute `path:` to prevent DATASETS_DIR resolution bugs
- Pose wrapper reads `kpt_shape`, `kpt_names`, `flip_idx` from original data.yaml rather than hardcoding 6-keypoint structure

## Deviations from Plan

None — plan executed exactly as written. All files were already in place from the prior plan execution (36-01 committed both tasks).

## Issues Encountered
- Pre-existing test failures in `tests/unit/engine/test_pipeline.py` (config artifact writing) — unrelated to training wrappers, out-of-scope for this plan
- Pre-existing typecheck errors in engine, segmentation, visualization, reconstruction modules — none in training/

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- TRAIN-01 and TRAIN-02 satisfied: `aquapose train seg` and `aquapose train pose` launch their respective wrappers
- Phase 37 (Pipeline Integration) can now wire YOLO-seg into `segment_then_extract` backend and YOLO-pose into `direct_pose` backend
- Both wrappers follow the same interface as `train_yolo_obb` — consistent patterns throughout training/

---
*Phase: 36-training-wrappers*
*Completed: 2026-03-01*

## Self-Check: PASSED
- `src/aquapose/training/yolo_seg.py` - EXISTS
- `src/aquapose/training/yolo_pose.py` - EXISTS
- `src/aquapose/training/cli.py` - EXISTS (with seg/pose subcommands)
- `src/aquapose/training/__init__.py` - EXISTS (exports train_yolo_seg, train_yolo_pose)
- `tests/unit/training/test_yolo_seg.py` - EXISTS
- `tests/unit/training/test_yolo_pose.py` - EXISTS
- `tests/unit/training/test_training_cli.py` - EXISTS (with seg/pose help tests)
- Commit e1ecf6b - FOUND (feat(36-01): add yolo_seg/yolo_pose training wrappers and CLI commands)
- Commit 3e802c0 - FOUND (feat(36-02): add seg/pose CLI help tests and update test_training_cli.py)
- All training unit tests PASS (confirmed by test run)
- Linting: PASSED (no ruff violations)
- Typecheck: PASSED for training/ (no new errors introduced)
