---
gsd_state_version: 1.0
milestone: v3.0
milestone_name: Ultralytics Unification
status: unknown
last_updated: "2026-03-02T16:48:00.000Z"
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 10
  completed_plans: 8
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-01)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 38 — Stabilization and Tech Debt Cleanup (Plan 02 complete)

## Current Position

Phase: 38 of 38 (Stabilization and Tech Debt Cleanup) — IN PROGRESS
Plan: 2/4 complete in current phase
Status: Plan 02 complete (NDJSON->txt+yaml format migration); Plan 03 next
Last activity: 2026-03-02 — Completed 38-02-PLAN.md (build script and training wrappers migrated from NDJSON to standard YOLO txt+yaml format)

Progress: [█████████░] ~92%

## Accumulated Context

### Decisions

Key decisions from Phase 36 Plan 02 (2026-03-01):

- New seg/pose CLI subcommands use `--model` (full model name string) not `--model-size` — consistent with CONTEXT.md decision; OBB keeps `--model-size` unchanged
- Training wrappers pass `dataset.ndjson` directly to `model.train(data=...)` — Ultralytics-native NDJSON format (quick-14 eliminated txt conversion)
- Pose wrapper reads `kpt_shape`, `kpt_names`, `flip_idx` from original data.yaml — nothing hardcoded about 6-keypoint structure

Key decisions from Phase 36 Plan 01 (2026-03-01):

- YOLO26n-seg and YOLO26n-pose training wrappers follow exact same pattern as existing yolo_obb.py
- Both wrappers forbidden from importing aquapose.engine or aquapose.cli (import boundary enforced by test)

Key decisions from Phase 37 Plan 01 (2026-03-01):

- Backend names changed from `segment_then_extract`/`direct_pose` to `segmentation`/`pose_estimation`
- Old names now raise ValueError at MidlineConfig construction time
- `MidlineConfig.backend` default changed to `"segmentation"`; `keypoint_confidence_floor` default raised to 0.3
- New backend stubs accept explicit kwargs (not `**kwargs`) and store as instance attributes for Plan 02 wiring

Key decisions from Phase 35 Plan 02 (2026-03-01):

- Both midline backends were no-op stubs accepting `**kwargs` — no model loading, all midlines are `None`
- Phase 37 Plan 01 renamed these backends and Phase 37 Plan 02 will wire YOLO-seg/pose into them

Key decisions from Phase 35 Plan 01 (2026-03-01):

- `make_detector()` now only accepts 'yolo' and 'yolo_obb' — raises ValueError for unknown kinds
- `DetectionConfig.__post_init__` validates `detector_kind` at construction time

Key decisions entering v3.0:

- v2.2 terminated early: custom U-Net segmentation (IoU 0.623) and custom keypoint regression both performed poorly — insufficient for production use
- Ultralytics unification chosen: YOLO11n-seg replaces U-Net segmentation, YOLO11n-pose replaces keypoint regression
- Backwards compatibility explicitly not a concern — strip old code, start fresh with Ultralytics only
- YOLO-OBB detection backend (Phase 32) is kept — already uses Ultralytics and works well
- Existing Stage Protocol architecture retained — new models plug in as backends within the 5-stage pipeline
- Training uses NDJSON format; existing `scripts/build_yolo_training_data.py` covers OBB/pose; only seg conversion is new (DATA-01)
- 6 anatomical keypoints: nose, head, spine1, spine2, spine3, tail
- [Phase 36-training-wrappers]: format_seg_annotation normalizes polygon vertices to [0,1]; generate_seg_dataset uses same affine crop pipeline as pose, OBB from keypoints + polygons from segmentation; annotations missing segmentation silently skipped
- [Phase 37-02]: SegmentationBackend uses reconstruction/midline.py helpers directly (imported, not duplicated); half_widths in PoseEstimationBackend are zeros; _keypoints_to_midline exposed as module-level function for testability
- [Phase 38-01]: DetectionConfig.model_path renamed to weights_path; _RENAME_HINTS provides user-facing hints for old YAML configs
- [Phase 38-01]: MidlineConfig.keypoint_weights_path removed; all backends now use single weights_path field
- [Phase 38-01]: init-config defaults updated to yolo_obb + pose_estimation with weights_path, matching current production architecture
- [Phase 38-02]: Training data format migrated from NDJSON to standard YOLO txt+yaml (labels/ dir + dataset.yaml); training wrappers pass dataset.yaml to model.train()
- [Phase 38-02]: Identity flip_idx used for pose dataset.yaml; kpt_shape derived from annotation data with N_KEYPOINTS fallback

### Pending Todos

17 pending todos from v2.2 — see .planning/todos/pending/ (review for relevance to v3.0)

### Roadmap Evolution

- Phase 38 added: Stabilization and Tech Debt Cleanup (NDJSON→txt labels, init-config defaults, weights_path consolidation, stale docstrings)

### Blockers/Concerns

**CRITICAL — Coordinate space conversions (cross-cutting, all phases):**
Full-image ↔ crop-space conversions are a pervasive source of error, especially with OBB affine warps. Mismatches between how crops are prepared for training vs inference cause silent accuracy failures. Existing machinery (`extract_affine_crop`, `invert_affine_point`, `transform_keypoints`) should be reused but with extreme attention to coordinate normalization at every boundary. v2.2 development surfaced and fixed many such bugs, but confidence in correctness remains low. Every phase must explicitly verify coordinate round-trips.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 12 | Add tool script for YOLO-OBB and YOLO-pose training set generation from multi-animal pose annotations | 2026-03-01 | 304bc09 | [12-add-tool-script-for-yolo-obb-and-yolo-po](./quick/12-add-tool-script-for-yolo-obb-and-yolo-po/) |
| 13 | Convert yolo_obb.py to use NDJSON format via train_yolo_ndjson; align OBB NDJSON key and CLI flags with seg/pose | 2026-03-01 | 7e98ac3 | [13-convert-yolo-obb-py-to-use-ndjson-format](./quick/13-convert-yolo-obb-py-to-use-ndjson-format/) |
| 14 | Adopt Ultralytics-native NDJSON format: build script emits dataset header + flat annotation arrays; training wrappers pass .ndjson directly to model.train() | 2026-03-01 | c8cd9eb | [14-adopt-ultralytics-native-ndjson-format-u](./quick/14-adopt-ultralytics-native-ndjson-format-u/) |

## Session Continuity

Last session: 2026-03-02
Stopped at: Completed 38-02-PLAN.md (NDJSON to txt+yaml format migration)
Resume file: None
