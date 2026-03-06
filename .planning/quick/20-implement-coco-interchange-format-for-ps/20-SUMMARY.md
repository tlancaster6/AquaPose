---
phase: 20-implement-coco-interchange-format-for-ps
plan: 01
subsystem: training
tags: [coco, interchange, pseudo-labels, export, import]
dependency_graph:
  requires: []
  provides: [coco-interchange, yolo-coco-roundtrip]
  affects: [pseudo-label-generate, pseudo-label-assemble]
tech_stack:
  added: []
  patterns: [bidirectional-format-conversion, lazy-import-fallback]
key_files:
  created:
    - src/aquapose/training/coco_interchange.py
    - tests/unit/training/test_coco_interchange.py
  modified:
    - src/aquapose/training/__init__.py
    - src/aquapose/training/pseudo_label_cli.py
    - src/aquapose/training/dataset_assembly.py
decisions:
  - PIL Image.open for dimension reads (fast, no decode, already available)
  - No pycocotools dependency (COCO format is just JSON schema)
  - Lazy import of coco_to_yolo_pose in dataset_assembly.py (only loads on COCO fallback path)
metrics:
  duration: ~5 min
  completed: 2026-03-06
---

# Quick Task 20: COCO Interchange Format for Pseudo-Labels Summary

Bidirectional YOLO-Pose to COCO Keypoints JSON conversion enabling external annotation tool editing of pseudo-labels via round-trip export/import.

## What Was Built

### coco_interchange.py (new module)

Three public functions:

- `yolo_pose_to_coco(pose_dir, n_keypoints)` -- Reads YOLO-Pose labels/images, converts normalized bbox/keypoints to absolute-pixel COCO format with proper visibility mapping.
- `write_coco_keypoints(pose_dir, n_keypoints)` -- Convenience wrapper that writes `coco_keypoints.json` next to `dataset.yaml`.
- `coco_to_yolo_pose(coco_path, output_labels_dir)` -- Converts COCO JSON back to YOLO-Pose `.txt` label files with normalized coordinates.

### Pipeline Integration

- **generate()**: Automatically writes `coco_keypoints.json` alongside each pose dataset (consensus and gap).
- **collect_pseudo_labels()**: COCO fallback -- when `coco_keypoints.json` exists but YOLO labels are absent/empty, converts on-the-fly before assembly proceeds.

## Test Coverage

5 unit tests covering:
- Basic YOLO-to-COCO conversion (structure, absolute pixel coords, sequential IDs)
- Invisible keypoint mapping (YOLO `0 0 0` <-> COCO `0 0 0`)
- Basic COCO-to-YOLO conversion (normalized coords, multi-image, multi-annotation)
- COCO invisible keypoint mapping back to YOLO
- Full round-trip identity (YOLO -> COCO -> YOLO within `atol=1e-5`)

## Task Commits

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 (RED) | Failing tests | 58d69c1 | tests/unit/training/test_coco_interchange.py |
| 1 (GREEN) | Implement converters | ba341a9 | src/aquapose/training/coco_interchange.py |
| 2 | Wire into CLI pipelines | d2e1195 | pseudo_label_cli.py, dataset_assembly.py, __init__.py |

## Deviations from Plan

None -- plan executed exactly as written.

## Verification

- All 1032 tests pass (including 5 new)
- Lint clean on all modified files
- Typecheck: no new errors (42 pre-existing, none in modified files)

## Self-Check: PASSED
