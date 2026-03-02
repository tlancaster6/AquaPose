---
phase: 35-codebase-cleanup
verified: 2026-03-01T21:15:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 35: Codebase Cleanup Verification Report

**Phase Goal:** The codebase contains no custom U-Net, SAM2 pseudo-label, old midline backend, MOG2 detection, or legacy training CLI code — only Ultralytics-based models and the new training wrappers remain, leaving a clean foundation for v3.0 backends
**Verified:** 2026-03-01T21:15:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `segmentation/model.py`, `_UNet`, `_PoseModel`, and `BinaryMaskDataset` deleted; no import of these symbols exists anywhere | VERIFIED | Files absent from filesystem; grep returns zero hits across `src/` and `tests/` |
| 2 | SAM2 pseudo-label generation code removed; only path is COCO JSON → NDJSON | VERIFIED | `pseudo_labeler.py` deleted; `SAMPseudoLabeler`, `FrameAnnotation`, `filter_mask`, `to_coco_dataset` return zero grep hits |
| 3 | Custom model code removed from `segment_then_extract` and `direct_pose` backends; both stubbed as no-ops | VERIFIED | Both files rewritten as ~80-line no-op stubs importing only `AnnotatedDetection` and `Detection`; no model loading |
| 4 | MOG2 detection backend removed; only `yolo` and `yolo_obb` are registered backends | VERIFIED | `MOG2Detector` class absent from `detector.py`; `make_detector("mog2")` raises `ValueError`; `DetectionConfig.__post_init__` validates at construction |
| 5 | `train_unet` and `train_pose` CLI commands removed; `aquapose train --help` lists only `yolo-obb` | VERIFIED | `training/cli.py` contains only `yolo-obb` command; `test_train_help_does_not_list_removed_commands` passes |

**Score:** 5/5 truths verified

---

## Required Artifacts

### Plan 35-01 Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/segmentation/__init__.py` | Exports only crop utilities + Detection + YOLODetector + make_detector | VERIFIED | Exact exports: CropRegion, Detection, YOLODetector, compute_crop_region, extract_crop, make_detector, paste_mask |
| `src/aquapose/training/__init__.py` | Exports only common utils + CropDataset + prep_group + train_yolo_obb (no BinaryMaskDataset/train_unet/train_pose/KeypointDataset) | VERIFIED | Exact exports: CropDataset, EarlyStopping, MetricsLogger, apply_augmentation, make_loader, prep_group, save_best_and_last, stratified_split, train_yolo_obb |
| `src/aquapose/training/cli.py` | Training CLI with only yolo-obb subcommand | VERIFIED | File contains only `train_group` + `yolo_obb` command; no unet or pose functions |

### Plan 35-02 Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/core/midline/backends/segment_then_extract.py` | No-op stub returning midline=None for all detections | VERIFIED | 79-line stub; constructor logs warning; process_frame returns AnnotatedDetection(midline=None) per detection |
| `src/aquapose/core/midline/backends/direct_pose.py` | No-op stub returning midline=None for all detections | VERIFIED | 79-line stub; constructor logs warning; process_frame returns AnnotatedDetection(midline=None) per detection |
| `src/aquapose/core/midline/backends/__init__.py` | Updated registry noting stub status | VERIFIED | Module docstring and get_backend() docstring explicitly state both backends are no-op stubs pending Phase 37 |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `segmentation/__init__.py` | `segmentation/detector.py` | imports Detection, YOLODetector, make_detector (NOT MOG2Detector) | WIRED | `from .detector import Detection, YOLODetector, make_detector` — no MOG2Detector in import |
| `training/cli.py` | `training/yolo_obb.py` | `@train_group.command("yolo-obb")` only | WIRED | Single command on train_group; lazy import inside function body |
| `backends/__init__.py` | `backends/segment_then_extract.py` | lazy import in `get_backend("segment_then_extract")` | WIRED | `from aquapose.core.midline.backends.segment_then_extract import SegmentThenExtractBackend` present |
| `backends/__init__.py` | `backends/direct_pose.py` | lazy import in `get_backend("direct_pose")` | WIRED | `from aquapose.core.midline.backends.direct_pose import DirectPoseBackend` present |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CLEAN-01 | 35-01 | All custom U-Net model code removed | SATISFIED | `segmentation/model.py`, `training/unet.py`, `training/pose.py` absent; `BinaryMaskDataset` removed from `datasets.py`; zero grep hits for `UNetSegmentor`, `_UNet`, `MaskRCNNSegmentor`, `UNET_INPUT_SIZE` |
| CLEAN-02 | 35-01 | SAM2 pseudo-label pipeline removed | SATISFIED | `segmentation/pseudo_labeler.py` absent; `tests/unit/segmentation/test_pseudo_labeler.py` and `tests/integration/segmentation/test_yolo_sam_integration.py` absent; zero grep hits for SAM symbols |
| CLEAN-03 | 35-02 | Custom model code removed from midline backends; backends stubbed as no-ops | SATISFIED | Both backends rewritten as no-op stubs; 5-test suite for DirectPoseBackend passes; `MidlineConfig.__post_init__` validates backend names |
| CLEAN-04 | 35-01 | MOG2 detection backend removed | SATISFIED | `MOG2Detector` class absent from `detector.py`; `make_detector("mog2")` raises ValueError; `DetectionConfig.__post_init__` validates `detector_kind` at construction |
| CLEAN-05 | 35-01 | Old training CLI commands removed | SATISFIED | `training/cli.py` has no `unet` or `pose` functions; `test_train_help_does_not_list_removed_commands` asserts both absent; 614 tests pass |

No orphaned requirements — REQUIREMENTS.md maps CLEAN-01 through CLEAN-05 exclusively to Phase 35, and all five are addressed by plans 35-01 and 35-02.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `backends/segment_then_extract.py` | 39 | `logger.warning(...)` in constructor | Info | Intentional by design — warns operator that midline will be None; not a stub smell, it's the documented stub behavior |
| `backends/direct_pose.py` | 39 | `logger.warning(...)` in constructor | Info | Same as above — intentional |
| `engine/config.py` | 416, 442, 526, 527 | Comments mention `"mog2"` as a CLI override example | Info | Documentation artefacts only; the validation code correctly rejects mog2; no functional impact |

No blockers. No warnings. The mog2 comment references in `config.py` are in docstrings illustrating the override format — the actual validation correctly rejects mog2, and the test `test_detector_kind_mog2_raises_value_error` confirms rejection at construction time.

---

## Test Suite Verification

- **hatch run test:** 614 passed, 31 deselected, 21 warnings — zero failures
- **hatch run lint:** All checks passed

614 tests confirms a net gain of 6 tests from Phase 35 (608 after Plan 01, 614 after Plan 02 adding the `test_direct_pose_backend.py` suite with 5 tests and one new midline stage test).

---

## Human Verification Required

None. All success criteria are verifiable programmatically:
- File existence/absence is binary
- Symbol imports are grep-checkable
- Config validation is covered by existing tests
- Test suite passing is confirmed

---

## Summary

Phase 35 achieved its goal. The codebase contains no custom U-Net, SAM2, MOG2, or legacy training CLI code. All five CLEAN requirements are satisfied:

- **CLEAN-01/02:** `model.py` and `pseudo_labeler.py` deleted; all custom model symbols purged from `datasets.py`, `__init__.py` exports, and test files. Zero grep hits.
- **CLEAN-03:** Both midline backends (`segment_then_extract`, `direct_pose`) rewritten as substantive no-op stubs — they instantiate cleanly, accept `**kwargs` for API compatibility, log a warning, and return `AnnotatedDetection(midline=None)` for all detections. `MidlineConfig` validates backend names at construction.
- **CLEAN-04:** `MOG2Detector` class removed from `detector.py`; `make_detector("mog2")` raises `ValueError`; `DetectionConfig.__post_init__` enforces `{"yolo", "yolo_obb"}` at construction.
- **CLEAN-05:** `training/cli.py` reduced to `yolo-obb` only; test asserts `unet` and `pose` absent from `--help` output.

The foundation is clean and ready for Phase 36 (Training Wrappers).

---

_Verified: 2026-03-01T21:15:00Z_
_Verifier: Claude (gsd-verifier)_
