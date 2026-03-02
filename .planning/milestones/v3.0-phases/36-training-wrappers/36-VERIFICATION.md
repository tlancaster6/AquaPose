---
phase: 36-training-wrappers
verified: 2026-03-01T22:00:00Z
status: passed
score: 11/11 must-haves verified
re_verification: false
---

# Phase 36: Training Wrappers Verification Report

**Phase Goal:** A COCO-to-NDJSON segmentation data converter and training wrappers for YOLO26n-seg and YOLO26n-pose are available from the CLI, following the same pattern as the existing yolo_obb.py training wrapper
**Verified:** 2026-03-01T22:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `--mode seg` CLI flag exists in `build_yolo_training_data.py` | VERIFIED | `choices=["all", "obb", "pose", "seg"]` in `build_parser()`, line 982 |
| 2 | COCO segmentation polygons are affine-transformed into crop space and normalized to [0,1] | VERIFIED | `generate_seg_dataset()` lines 892–913: homogeneous affine multiply + `format_seg_annotation()` normalizes by dividing by crop_w/crop_h |
| 3 | Multi-ring COCO polygons keep only the largest ring | VERIFIED | `max(segmentation, key=len)` at line 883 of build_yolo_training_data.py |
| 4 | All visible fish in each crop are labeled (target + intruders) | VERIFIED | Inner loop over `for other_ann in annotations` (all image annotations) at line 876 |
| 5 | Existing `--mode all` behavior is unchanged (OBB + pose, not seg) | VERIFIED | `if mode in ("all", "obb")` and `if mode in ("all", "pose")` — seg only on `if mode == "seg"` |
| 6 | `aquapose train seg --help` shows all required flags | VERIFIED | cli.py lines 67–124: --data-dir, --output-dir, --epochs, --batch-size, --device, --val-split, --imgsz, --model, --weights all present |
| 7 | `aquapose train pose --help` shows all required flags | VERIFIED | cli.py lines 127–184: identical flag set with --model default yolo26n-pose |
| 8 | NDJSON-to-YOLO.txt seg conversion produces valid label files with normalized polygon coordinates | VERIFIED | `_convert_seg_ndjson_to_txt()` in yolo_seg.py lines 14–54 writes `class_id x1 y1 x2 y2 ...` format |
| 9 | NDJSON-to-YOLO.txt pose conversion produces valid label files with bbox + keypoints | VERIFIED | `_convert_pose_ndjson_to_txt()` in yolo_pose.py lines 14–62 writes `class_id cx cy w h x1 y1 v1...` format |
| 10 | Training wrappers rewrite data.yaml to use images/ directories with absolute path before calling model.train() | VERIFIED | Both wrappers call `_rewrite_data_yaml_seg/pose()` which sets `path: <abs>`, `train: images/train`, `val: images/val`; result passed as `data=str(rewritten_yaml)` to `model.train()` |
| 11 | yolo_seg.py and yolo_pose.py do not import from aquapose.engine or aquapose.cli | VERIFIED | grep of entire training/ directory found zero forbidden imports |

**Score:** 11/11 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/build_yolo_training_data.py` | `--mode seg` branch with `generate_seg_dataset` function | VERIFIED | Lines 791–960: full `generate_seg_dataset()` implementation; `format_seg_annotation()` at lines 509–535 |
| `tests/unit/test_build_yolo_training_data.py` | `TestSegConverter` class with polygon transform and multi-ring tests | VERIFIED | `TestFormatSegAnnotation` (4 tests, line 529) and `TestSegConverter` (5 integration tests, line 564) both present |
| `src/aquapose/training/yolo_seg.py` | YOLO26 seg training wrapper with NDJSON conversion | VERIFIED | `train_yolo_seg()` at line 103; `_convert_seg_ndjson_to_txt()` at line 14; `_rewrite_data_yaml_seg()` at line 57 |
| `src/aquapose/training/yolo_pose.py` | YOLO26 pose training wrapper with NDJSON conversion | VERIFIED | `train_yolo_pose()` at line 124; `_convert_pose_ndjson_to_txt()` at line 14; `_rewrite_data_yaml_pose()` at line 65 |
| `src/aquapose/training/cli.py` | seg and pose CLI subcommands | VERIFIED | `@train_group.command("seg")` at line 67; `@train_group.command("pose")` at line 127 |
| `tests/unit/training/test_yolo_seg.py` | Unit tests for NDJSON-to-YOLO.txt seg conversion | VERIFIED | `test_convert_seg_ndjson_to_txt`, `test_rewrite_data_yaml_seg`, `test_empty_ndjson_produces_empty_labels`, `test_convert_seg_polygon_coord_count` |
| `tests/unit/training/test_yolo_pose.py` | Unit tests for NDJSON-to-YOLO.txt pose conversion | VERIFIED | `test_convert_pose_ndjson_to_txt`, `test_rewrite_data_yaml_pose_preserves_kpt_shape`, `test_rewrite_data_yaml_pose_without_kpt_fields`, `test_empty_ndjson_produces_empty_labels_pose` |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `cli.py::seg` | `yolo_seg.py::train_yolo_seg` | lazy import in CLI handler | WIRED | `from .yolo_seg import train_yolo_seg` at cli.py line 111; called at line 113 |
| `cli.py::pose` | `yolo_pose.py::train_yolo_pose` | lazy import in CLI handler | WIRED | `from .yolo_pose import train_yolo_pose` at cli.py line 171; called at line 173 |
| `yolo_seg.py` | `ultralytics.YOLO` | `model.train(data=rewritten_yaml)` | WIRED | `from ultralytics import YOLO` lazy import at line 151; `yolo_model.train(...)` at line 189 |
| `generate_seg_dataset` | `affine_warp_crop` | reuses existing affine crop extraction | WIRED | `affine_warp_crop(img, corners, crop_w, crop_h)` called at line 872 |
| `generate_seg_dataset` | polygon affine transform | same affine_matrix math | WIRED | `affine_mat @ poly_h.T` at line 896 (homogeneous multiply, same pattern as `transform_keypoints`) |
| `training/__init__.py` | `train_yolo_seg`, `train_yolo_pose` | explicit imports + `__all__` | WIRED | Both imported at lines 9–10 and listed in `__all__` at lines 22–23 |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DATA-01 | 36-01-PLAN.md | Segmentation training data converter: COCO segmentation JSON → NDJSON-format YOLO-seg dataset matching existing OBB/pose NDJSON pattern | SATISFIED | `generate_seg_dataset()` in build_yolo_training_data.py produces `seg/images/{train,val}/`, `seg/{train,val}.ndjson`, `seg/data.yaml`; format matches pose NDJSON schema |
| TRAIN-01 | 36-02-PLAN.md | YOLO-seg training wrapper callable from CLI, following existing yolo_obb.py pattern | SATISFIED | `aquapose train seg` subcommand in cli.py calls `train_yolo_seg()`; wrapper follows same NDJSON-convert → rewrite-yaml → model.train() → copy-weights pattern as yolo_obb.py |
| TRAIN-02 | 36-02-PLAN.md | YOLO-pose training wrapper callable from CLI, following existing yolo_obb.py pattern | SATISFIED | `aquapose train pose` subcommand in cli.py calls `train_yolo_pose()`; wrapper follows same pattern |

No orphaned requirements found — all three requirement IDs declared in plan frontmatter map to verified implementations.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | No anti-patterns found |

No TODOs, FIXMEs, placeholder returns, or stub implementations found in any phase 36 modified files.

**One design note (not a blocker):** `--mode all` does not include seg (only OBB + pose). This is intentional per the plan — seg uses a different COCO annotation schema (segmentation polygons vs keypoints) and is explicitly excluded from the default mode. The CLI help text documents this correctly.

---

### Test Results

All phase 36 tests pass:

- `tests/unit/test_build_yolo_training_data.py` — all tests including `TestFormatSegAnnotation` (4 tests) and `TestSegConverter` (5 integration tests): PASS
- `tests/unit/training/test_yolo_seg.py` — 4 tests: PASS
- `tests/unit/training/test_yolo_pose.py` — 4 tests: PASS
- `tests/unit/training/test_training_cli.py` — all tests including `test_train_help_lists_seg_and_pose`, `test_train_seg_help_shows_expected_flags`, `test_train_pose_help_shows_expected_flags`: PASS
- Pre-existing failures: `tests/unit/engine/test_pipeline.py::test_pipeline_writes_config_artifact` and `test_config_artifact_written_before_stages` — confirmed pre-existing, unrelated to phase 36 (noted in both SUMMARYs as out of scope)

Full test run: **633 passed, 2 failed (pre-existing), 31 deselected**

### Human Verification Required

None. All observable behaviors for this phase are verifiable programmatically through code inspection and unit tests. The training wrappers call `model.train()` lazily and do not require a GPU run to verify correctness of NDJSON conversion, data.yaml rewrite, or CLI wiring.

---

## Gaps Summary

No gaps. All 11 must-have truths verified, all 7 artifacts confirmed substantive and wired, all 3 requirement IDs satisfied. Phase goal achieved.

---

_Verified: 2026-03-01T22:00:00Z_
_Verifier: Claude (gsd-verifier)_
