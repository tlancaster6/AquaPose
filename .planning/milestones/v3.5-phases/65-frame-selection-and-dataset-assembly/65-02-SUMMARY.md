---
phase: 65-frame-selection-and-dataset-assembly
plan: 02
subsystem: training
tags: [dataset-assembly, yolo, confidence-filtering, validation-split, cli]

# Dependency graph
requires:
  - phase: 65-frame-selection-and-dataset-assembly
    provides: "Frame selection utilities (temporal_subsample, diversity_sample)"
  - phase: 63-pseudo-labeling-pipeline
    provides: "Pseudo-label output directories with confidence.json sidecars"
provides:
  - "assemble_dataset: pool manual + pseudo-labels into YOLO dataset with confidence filtering"
  - "collect_pseudo_labels: discover labels from run directories with confidence metadata"
  - "filter_by_confidence / filter_by_gap_reason: independent thresholds per source"
  - "split_manual_val: per-camera stratified validation split"
  - "CLI: aquapose pseudo-label assemble with multi-run, threshold, frame selection options"
affects: [training, yolo-training, dataset-quality]

# Tech tracking
tech-stack:
  added: []
  patterns: [multi-run collision avoidance via run_id prefix, pseudo-label val metadata sidecar]

key-files:
  created:
    - src/aquapose/training/dataset_assembly.py
    - tests/unit/training/test_dataset_assembly.py
  modified:
    - src/aquapose/training/pseudo_label_cli.py
    - src/aquapose/training/__init__.py
    - tests/unit/training/test_pseudo_label_cli.py

key-decisions:
  - "Pseudo-label val images stored in train/ with JSON sidecar (not separate val dir)"
  - "Multi-run filename collision resolved by prefixing with run_dir.name"
  - "shutil.copy2 for portability (not symlinks)"

patterns-established:
  - "Dataset assembly: collect -> filter -> split -> copy -> write dataset.yaml"
  - "Pseudo-val metadata sidecar: JSON list with source, confidence, run_id per image"

requirements-completed: [DATA-01, DATA-02, DATA-03]

# Metrics
duration: 5min
completed: 2026-03-05
---

# Phase 65 Plan 02: Dataset Assembly Summary

**YOLO dataset assembly pooling manual annotations + multi-run pseudo-labels with independent confidence thresholds, gap-reason exclusion, and per-camera stratified validation splits**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-05T20:14:53Z
- **Completed:** 2026-03-05T20:19:42Z
- **Tasks:** 2 (Task 1: TDD, Task 2: auto)
- **Files modified:** 5

## Accomplishments
- Implemented dataset assembly module with 6 public functions for collecting, filtering, splitting, and assembling YOLO datasets
- Independent confidence thresholds for consensus (Source A) and gap (Source B) labels
- Per-camera stratified manual validation split used as official val in dataset.yaml
- Multi-run collision avoidance via run_id filename prefixing
- CLI `aquapose pseudo-label assemble` with full option set including frame selection integration
- 24 tests covering all functions and CLI end-to-end

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests** - `7c93c49` (test)
2. **Task 1 (GREEN): Implementation** - `590b77c` (feat)
3. **Task 2: CLI + exports** - `a45d0fb` (feat)

## Files Created/Modified
- `src/aquapose/training/dataset_assembly.py` - Core assembly logic: collect, filter, split, assemble, copy
- `tests/unit/training/test_dataset_assembly.py` - 22 unit tests for all assembly functions
- `src/aquapose/training/pseudo_label_cli.py` - Added `assemble` CLI command with 13 options
- `src/aquapose/training/__init__.py` - Added assemble_dataset export
- `tests/unit/training/test_pseudo_label_cli.py` - Added 2 CLI smoke tests for assemble command

## Decisions Made
- Pseudo-label val images are copied into `images/train/` (still used for training) but tracked in `pseudo_val_metadata.json` sidecar for post-training analysis -- avoids polluting the official val split
- Multi-run filename collisions resolved by prefixing pseudo-label files with `run_dir.name` (e.g., `run_001_000001_cam0.jpg`)
- Used `shutil.copy2` instead of symlinks for portability across systems
- Frame selection integration in CLI is optional -- only activated when `--temporal-step > 1` or `--diversity-max-per-bin` is specified

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Dataset assembly pipeline complete and ready for use
- CLI accepts multiple run dirs, manual dir, independent thresholds
- Output is YOLO-standard: `dataset.yaml` + `images/{train,val}/` + `labels/{train,val}/`
- Phase 65 (frame selection + dataset assembly) is fully complete

---
*Phase: 65-frame-selection-and-dataset-assembly*
*Completed: 2026-03-05*
