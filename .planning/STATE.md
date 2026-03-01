---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: Ultralytics Unification
status: unknown
last_updated: "2026-03-01T22:00:44.030Z"
progress:
  total_phases: 2
  completed_phases: 2
  total_plans: 4
  completed_plans: 4
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-01)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** Phase 36 — Training Wrappers (Plan 02 complete — phase DONE)

## Current Position

Phase: 36 of 37 (Training Wrappers) — COMPLETE
Plan: 2/2 complete in current phase
Status: Phase 36 complete; Phase 37 (Pipeline Integration) is next
Last activity: 2026-03-01 — Phase 36 Plan 02 complete (YOLO-seg/pose training wrappers, NDJSON conversion, CLI subcommands)

Progress: [████░░░░░░] ~66%

## Accumulated Context

### Decisions

Key decisions from Phase 36 Plan 02 (2026-03-01):

- New seg/pose CLI subcommands use `--model` (full model name string) not `--model-size` — consistent with CONTEXT.md decision; OBB keeps `--model-size` unchanged
- NDJSON-to-YOLO.txt conversion done before model.train(); label files written to `labels/{split}/` alongside `images/{split}/`
- data.yaml rewritten to `data_ultralytics.yaml` with absolute `path:` to prevent Ultralytics DATASETS_DIR resolution issues
- Pose wrapper reads `kpt_shape`, `kpt_names`, `flip_idx` from original data.yaml — nothing hardcoded about 6-keypoint structure

Key decisions from Phase 36 Plan 01 (2026-03-01):

- YOLO26n-seg and YOLO26n-pose training wrappers follow exact same pattern as existing yolo_obb.py
- Both wrappers forbidden from importing aquapose.engine or aquapose.cli (import boundary enforced by test)

Key decisions from Phase 35 Plan 02 (2026-03-01):

- Both midline backends (`segment_then_extract`, `direct_pose`) are now no-op stubs accepting `**kwargs` — no model loading, all midlines are `None`
- `MidlineConfig.__post_init__` validates `backend` against `{"segment_then_extract", "direct_pose"}` — rejects typos at construction time
- Stubs accept all previous kwargs silently for API compatibility with `get_backend()` kwarg forwarding
- No `sys.modules` injection needed in tests — stubs have zero model imports
- Phase 37 will wire YOLO-seg into `segment_then_extract` and YOLO-pose into `direct_pose` (backends are insertion points, not removed)

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

### Pending Todos

17 pending todos from v2.2 — see .planning/todos/pending/ (review for relevance to v3.0)

### Blockers/Concerns

**CRITICAL — Coordinate space conversions (cross-cutting, all phases):**
Full-image ↔ crop-space conversions are a pervasive source of error, especially with OBB affine warps. Mismatches between how crops are prepared for training vs inference cause silent accuracy failures. Existing machinery (`extract_affine_crop`, `invert_affine_point`, `transform_keypoints`) should be reused but with extreme attention to coordinate normalization at every boundary. v2.2 development surfaced and fixed many such bugs, but confidence in correctness remains low. Every phase must explicitly verify coordinate round-trips.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 12 | Add tool script for YOLO-OBB and YOLO-pose training set generation from multi-animal pose annotations | 2026-03-01 | 304bc09 | [12-add-tool-script-for-yolo-obb-and-yolo-po](./quick/12-add-tool-script-for-yolo-obb-and-yolo-po/) |

## Session Continuity

Last session: 2026-03-01
Stopped at: Completed 36-02-PLAN.md (YOLO-seg/pose training wrappers, NDJSON conversion helpers, CLI seg/pose subcommands — Phase 36 complete)
Resume file: None
