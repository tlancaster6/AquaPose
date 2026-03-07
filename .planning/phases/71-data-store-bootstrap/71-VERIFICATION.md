---
phase: 71-data-store-bootstrap
verified: 2026-03-07T18:00:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 71: Data Store Bootstrap Verification Report

**Phase Goal:** All existing manual annotations imported into the data store with baseline OBB and pose models trained, registered, and sanity-checked
**Verified:** 2026-03-07T18:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `aquapose data convert` converts COCO-JSON annotations to both YOLO-OBB and YOLO-pose formats | VERIFIED | `generate_obb_dataset` and `generate_pose_dataset` in coco_convert.py with `split_mode` parameter; convert_cmd in data_cli.py passes `--split-mode` through; OBB dataset has 40 train + 9 val, Pose dataset has 1290 train + 64 val |
| 2 | Manual annotations are in the data store as `source=manual` with correct provenance, and `data status` shows the expected sample counts | VERIFIED | OBB store: 49 manual samples; Pose store: 1354 manual samples (322 originals + 1032 augmented); `status_cmd` in data_cli.py queries both stores and displays counts |
| 3 | Baseline OBB and pose models are trained from store-assembled datasets and registered with model lineage | VERIFIED | OBB model `run_20260307_094353` tagged "baseline" (mAP50=0.931); Pose model `run_20260307_113057` tagged "baseline" (mAP50=0.991); both `best_model.pt` files exist; config.yaml updated |
| 4 | Train/val split respects temporal holdout convention (no near-duplicate leakage between splits) | VERIFIED | `parse_frame_index` + `temporal_split` in coco_convert.py; last 2 of 10 frame indices (684000, 693000) held out as val; val tagging on import preserves split through store; `test_temporal_split` and `test_assemble_tagged_split` pass |
| 5 | `aquapose data exclude --reason TAG` applies reason-tagged exclusions and `data status` shows breakdown by reason | VERIFIED | `exclude()` in store.py accepts `reason` param, adds both "excluded" and reason tags; `exclude_cmd` in data_cli.py passes `--reason`; `status_cmd` computes reason breakdown from `json_each(tags)` with reserved-tag filtering; tests `test_exclude_with_reason`, `test_exclude_without_reason_backward_compat`, `test_include_keeps_reason_tags` all pass |

**Score:** 5/5 truths verified

### Required Artifacts (Plan 01 -- Code Changes)

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/training/coco_convert.py` | parse_frame_index, temporal_split, split_mode param | VERIFIED | Functions exist at lines 238, 267; split_mode param on generate_obb_dataset (line 323) and generate_pose_dataset (line 437) |
| `src/aquapose/training/store.py` | exclude() with reason, assemble() with split_mode and val_candidates_tag | VERIFIED | exclude() has reason param (line 398); assemble() has split_mode and val_candidates_tag params (line 627) |
| `src/aquapose/training/data_cli.py` | --split-mode, --reason, val tagging, reason breakdown | VERIFIED | --split-mode on convert_cmd (line 363) and assemble_cmd (line 503); --reason on exclude_cmd (line 680); val tagging on import (line 208); reason breakdown in status_cmd (lines 594-613) |
| `src/aquapose/training/yolo_pose.py` | rect parameter | VERIFIED | rect param at line 24, passed to model.train() at line 89 |
| `src/aquapose/training/cli.py` | Updated defaults, rect flag | VERIFIED | OBB mosaic=0.3 (line 53); pose mosaic=0.1 (line 358), imgsz=128 (line 339), rect flag (line 362); rect passed to train_yolo_pose (line 432) |

### Required Artifacts (Plan 02 -- Workflow Outputs)

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `~/aquapose/projects/YH/training_data/obb/store.db` | OBB store with manual annotations | VERIFIED | 65KB, 49 manual samples, 9 val-tagged, 1 model registered |
| `~/aquapose/projects/YH/training_data/pose/store.db` | Pose store with manual + augmented | VERIFIED | 1MB, 1354 manual samples (322 originals + 1032 augmented), 64 val-tagged, 1 model registered |
| `~/aquapose/.../obb/datasets/baseline/` | Assembled OBB dataset | VERIFIED | 40 train / 9 val images+labels, dataset.yaml present |
| `~/aquapose/.../pose/datasets/baseline/` | Assembled pose dataset | VERIFIED | 1290 train / 64 val images+labels, dataset.yaml present |
| `~/aquapose/.../obb/run_20260307_094353/best_model.pt` | Trained OBB baseline model | VERIFIED | File exists |
| `~/aquapose/.../pose/run_20260307_113057/best_model.pt` | Trained pose baseline model | VERIFIED | File exists |
| `~/aquapose/projects/YH/config.yaml` | Updated model paths | VERIFIED | detection.weights_path and midline.weights_path point to baseline model files |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| data_cli.py convert_cmd | coco_convert.py generators | split_mode parameter passthrough | VERIFIED | split_mode passed at lines 419, 438 |
| data_cli.py assemble_cmd | store.py assemble() | split_mode and val_candidates_tag passthrough | VERIFIED | Both params passed at lines 551-555 |
| data_cli.py exclude_cmd | store.py exclude() | reason parameter passthrough | VERIFIED | reason passed at line 709 |
| cli.py pose command | yolo_pose.py train_yolo_pose() | rect parameter passthrough | VERIFIED | rect in cli_args (line 414), passed to train_yolo_pose (line 432) |
| data convert --split-mode temporal | YOLO-format output dirs | coco_convert.py generators | VERIFIED | OBB: 40 train, 9 val; temporal split confirmed by val frame indices 684000, 693000 |
| data import | store.db | SampleStore.import_sample() | VERIFIED | source=manual for all 49 OBB and 1354 pose samples |
| data assemble --split-mode tagged | dataset dir with symlinks | SampleStore.assemble() | VERIFIED | Baseline datasets assembled with correct train/val matching val tags |
| train obb/pose --tag baseline | store model registry | run_manager.register_trained_model() | VERIFIED | Both models registered with tag=baseline |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| BOOT-01 | 71-01 | `aquapose data convert` converts COCO-JSON to YOLO-OBB and YOLO-pose formats | SATISFIED | convert_cmd with --type both, split_mode param; tested and executed |
| BOOT-02 | 71-02 | Manual annotations imported into data store as source=manual with correct provenance | SATISFIED | OBB: 49 manual, Pose: 1354 manual; val-tagged correctly; stores populated |
| BOOT-03 | 71-02 | Baseline OBB and pose models trained and registered with model lineage | SATISFIED | Both models trained, registered with tag=baseline, config.yaml updated |
| BOOT-04 | 71-01 | Temporal split convention -- no near-duplicate leakage | SATISFIED | parse_frame_index + temporal_split; last 2/10 frame indices in val; tests pass |
| BOOT-05 | 71-01 | `data exclude --reason TAG` with reason breakdown in status | SATISFIED | exclude() with reason, status_cmd reason breakdown; 3 tests covering reason lifecycle |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No anti-patterns detected in any modified files |

### Human Verification Required

None required -- all truths verified programmatically. Training metrics (OBB mAP50=0.931, Pose mAP50=0.991) were already human-verified during Plan 02 execution (user approved results per SUMMARY self-check).

### Deviations Noted

1. **Pose imgsz 128 -> 320**: Plan 01 set CLI default to imgsz=128, but Plan 02 execution used imgsz=320 because 128px gave poor mAP50-95 (0.539). The CLI default remains 128 but the actual trained baseline used 320. This is acceptable -- CLI defaults are starting points, not constraints.
2. **Two bug fixes during execution**: Val augmentation leakage (7e9ce53) and missing kpt_shape in pose dataset.yaml (0d8545e) were found and fixed during Plan 02 workflow execution.

---

_Verified: 2026-03-07T18:00:00Z_
_Verifier: Claude (gsd-verifier)_
