---
phase: 38-stabilization-and-tech-debt-cleanup
verified: 2026-03-02T17:30:00Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 38: Stabilization and Tech-Debt Cleanup — Verification Report

**Phase Goal:** Training data and config infrastructure uses standard YOLO txt+yaml format (not NDJSON), config fields are consolidated and init-config generates correct defaults, and dead legacy code is analyzed.

**Verified:** 2026-03-02
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `DetectionConfig` has `weights_path` field and no `model_path` field | VERIFIED | `config.py` L50: `weights_path: str \| None = None`; `_RENAME_HINTS` maps old `model_path` to hint |
| 2 | `MidlineConfig` has a single `weights_path` field and no `keypoint_weights_path` field | VERIFIED | `config.py` L79-80: `weights_path` documented; grep for `keypoint_weights_path` in `src/` returns zero hits |
| 3 | `aquapose init-config` generates YAML with `detector_kind: yolo_obb`, `backend: pose_estimation`, and `weights_path` in both sections | VERIFIED | `cli.py` L174-182: explicit dict with `detector_kind: yolo_obb`, `weights_path: models/yolo_obb.pt`, `backend: pose_estimation`, `weights_path: models/yolo_pose.pt` |
| 4 | No NDJSON generation or consumption code remains anywhere in the codebase | VERIFIED | `grep -rni 'ndjson' src/` and `grep -rni 'ndjson' scripts/` both return zero hits |
| 5 | `build_yolo_training_data.py` produces `labels/` directories with per-image `.txt` files and `dataset.yaml` for all three modes (obb, seg, pose) | VERIFIED | Script L655-665, L805-816, L988-998: all three generate functions write `dataset.yaml` and `labels/{split}/` dirs |
| 6 | Training wrappers pass `dataset.yaml` to `model.train(data=...)` | VERIFIED | `yolo_obb.py` L72, `yolo_pose.py` L71, `yolo_seg.py` L71: all call `model.train(data=str(yaml_path), ...)` |
| 7 | An import analysis report exists classifying all legacy files; dead code is deleted with user approval | VERIFIED | `38-DEAD-CODE-REPORT.md` exists with AST-based analysis of 16 files across 4 directories; `visualization/diagnostics.py` deleted per user approval |

**Score:** 7/7 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/engine/config.py` | `DetectionConfig` with `weights_path`, `MidlineConfig` with `weights_path` only | VERIFIED | `weights_path` present in both dataclasses; `_RENAME_HINTS` maps old `model_path` |
| `src/aquapose/cli.py` | init-config generates `yolo_obb` defaults with `weights_path` | VERIFIED | Lines 174-182 generate correct defaults |
| `scripts/build_yolo_training_data.py` | Produces `labels/` dirs and `dataset.yaml` for obb, seg, pose modes | VERIFIED | All three generate functions write txt+yaml; no NDJSON code |
| `src/aquapose/training/yolo_obb.py` | Consumes `dataset.yaml` | VERIFIED | L63-65, 72: looks for and passes `dataset.yaml` |
| `src/aquapose/training/yolo_pose.py` | Consumes `dataset.yaml` | VERIFIED | L63-65, 71: looks for and passes `dataset.yaml` |
| `src/aquapose/training/yolo_seg.py` | Consumes `dataset.yaml` | VERIFIED | L63-65, 71: looks for and passes `dataset.yaml` |
| `.planning/phases/38-stabilization-and-tech-debt-cleanup/38-DEAD-CODE-REPORT.md` | Import analysis of legacy directories | VERIFIED | 274-line report classifying 16 files with AST evidence and recommendations |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `config.py` (DetectionConfig) | `pipeline.py` | `config.detection.weights_path` access | WIRED | `pipeline.py` L321: `weights_path=config.detection.weights_path` |
| `config.py` (MidlineConfig) | `core/midline/stage.py` | `mc.weights_path` | WIRED | `stage.py` L160: `weights_path=mc.weights_path if mc is not None else None` |
| `build_yolo_training_data.py` | `training/yolo_obb.py` | `dataset.yaml` file contract | WIRED | Build script writes `dataset.yaml`; wrapper reads `data_dir / "dataset.yaml"` |
| `build_yolo_training_data.py` | `training/yolo_pose.py` | `dataset.yaml` file contract | WIRED | Build script writes `dataset.yaml` with `kpt_shape`; wrapper reads `dataset.yaml` |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| STAB-01 | Plan 38-02 | Training data script produces standard YOLO txt labels + dataset.yaml; training wrappers consume txt+yaml | SATISFIED | Build script produces txt+yaml; all three wrappers use `dataset.yaml`; no NDJSON code remains |
| STAB-02 | Plan 38-01 | `weights_path` and `keypoint_weights_path` consolidated into single `weights_path` field | SATISFIED | `DetectionConfig.weights_path` and `MidlineConfig.weights_path` are the only path fields; grep for `keypoint_weights_path` in `src/` returns zero hits |
| STAB-03 | Plan 38-01 | `init-config` generates correct defaults (YOLO-OBB detection, explicit backend selection, valid weights path) | SATISFIED | `cli.py` init_config generates `detector_kind: yolo_obb`, `backend: pose_estimation`, `weights_path` in both sections |
| STAB-04 | Plans 38-03, 38-04 | All stale docstrings referencing U-Net, no-op stubs, or Phase 37 pending status are updated; dead code analyzed | DEFERRED (user decision) | Deferred to Phase 39 by explicit user instruction. Dead code analysis (Plan 38-04 component) WAS completed. Stale U-Net docstrings remain in `src/aquapose/reconstruction/midline.py` (lines 303, 312, 314) and `src/aquapose/core/midline/stage.py` (line 4) — deferred. |

---

## Test Suite Status

All 656 unit tests pass (3 skipped, 31 deselected as `@slow`/`@e2e`). No test failures from any phase 38 changes. Pre-existing failures noted in plan 38-02 summary (`test_pipeline_writes_config_artifact`, `test_config_artifact_written_before_stages`) were resolved by the time of verification — `tests/unit/engine/test_pipeline.py` now passes all 14 tests.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/aquapose/core/midline/stage.py` | 4 | Stale docstring: "via U-Net" | Info | Misleading but STAB-04 deferred to Phase 39 |
| `src/aquapose/reconstruction/midline.py` | 303, 312, 314 | Stale docstring: "U-Net resize" | Info | Misleading but STAB-04 deferred to Phase 39 |

These are the only anti-patterns found. Both are in scope for STAB-04 which is explicitly deferred to Phase 39 and do NOT block STAB-01/02/03 goal achievement.

Note: `src/aquapose/core/detection/backends/yolo.py` L58 contains `model_path=weights_path` — this is an internal call to the lower-level `YOLODetector` class (in `segmentation/detector.py`) which still uses `model_path` as its parameter name. This is intentional and documented in the 38-01 summary: "YOLODetector (in segmentation/detector.py) keeps its own model_path parameter since it's a lower-level class outside the scope of this plan." This is NOT a stale reference — it is a deliberate architectural boundary.

---

## Human Verification Required

None. All automated checks pass for STAB-01, STAB-02, and STAB-03.

---

## Summary

All three in-scope requirements (STAB-01, STAB-02, STAB-03) and the phase goal's "dead legacy code is analyzed" component are fully achieved:

1. **STAB-01 (NDJSON to txt+yaml):** The build script now generates standard Ultralytics YOLO directory structure with `labels/` txt files and `dataset.yaml` for all three modes. All three training wrappers consume `dataset.yaml`. Zero NDJSON references remain in any production code.

2. **STAB-02 (weights_path consolidation):** `DetectionConfig` and `MidlineConfig` both use a single `weights_path` field. `model_path` and `keypoint_weights_path` are completely eliminated from the config dataclasses and all consumers. `_RENAME_HINTS` provides migration guidance for users with old configs.

3. **STAB-03 (init-config defaults):** `aquapose init-config` generates a correct starter config with `detector_kind: yolo_obb`, `backend: pose_estimation`, and `weights_path` in both sections. No hardcoded keypoint count.

4. **Dead code analysis:** `38-DEAD-CODE-REPORT.md` provides a complete AST-based import graph of all 16 files in the legacy directories. `visualization/diagnostics.py` was deleted per user approval. All other legacy modules confirmed as canonical, load-bearing implementations.

STAB-04 (docstring/guidebook accuracy) is deferred to Phase 39 per explicit user decision. The stale U-Net references in two docstrings do not affect runtime behavior and are tracked for cleanup in the next phase.

---

_Verified: 2026-03-02T17:30:00Z_
_Verifier: Claude (gsd-verifier)_
