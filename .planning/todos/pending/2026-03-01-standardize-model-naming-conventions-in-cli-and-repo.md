---
created: 2026-03-01T16:32:54.545Z
title: Standardize model naming conventions in CLI and repo
area: engine
files:
  - src/aquapose/training/
  - src/aquapose/cli/
  - src/aquapose/engine/
---

## Problem

The `aquapose train` CLI uses inconsistent naming for its training modes: `unet` and `yolo-obb` are named after the model architecture, while `pose` is named after the task. This creates confusion about what each mode does and makes the naming scheme hard to extend. The inconsistency propagates through config keys, backend names, and documentation throughout the repo.

For example, a user might expect `pose` to be a model name (like "PoseNet") rather than a task, or wonder why the segmentation U-Net isn't called `segmentation` to match the task-based convention of `pose`.

## Solution

Pick one convention and propagate it consistently:
- **Option A (task-based)**: `segmentation`, `detection-obb`, `pose` — names describe what the model does
- **Option B (model-based)**: `unet`, `yolo-obb`, `pose-resnet` (or similar) — names describe the architecture
- **Option C (hybrid with clear pattern)**: `unet-seg`, `yolo-obb`, `yolo-detect`, `pose-kpt` — model+task suffix

Whichever is chosen, update:
- CLI subcommand/mode names in training entry points
- Config schema keys (e.g., `training.mode`)
- Backend registry names
- Documentation and help strings
- Any references in scripts or test fixtures

Consider backward compatibility — deprecation aliases for old names if configs exist in the wild.
