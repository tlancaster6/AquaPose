---
created: 2026-03-01T16:30:44.871Z
title: Add plain YOLO training path to CLI
area: tooling
files:
  - src/aquapose/training/
  - src/aquapose/cli/
---

## Problem

The `aquapose train` CLI currently supports three training modes: `unet`, `pose`, and `yolo-obb`. There is no path for training a plain YOLO (axis-aligned bounding box) detection model. Users who want to train or retrain a standard YOLO detector for fish detection must do so outside the CLI, losing the benefits of standardized config, logging, and reproducibility that the training CLI provides.

## Solution

- Add a `yolo` (or `yolo-aabb`) training subcommand/mode alongside the existing `unet`, `pose`, and `yolo-obb` modes
- Reuse shared training infrastructure (config loading, logging, checkpoint management) from the existing training paths
- Support standard YOLO detection dataset format (images + YOLO-format `.txt` labels with class + xywh)
- Wire up Ultralytics YOLO training API similar to the existing `yolo-obb` path but with standard `detect` task instead of `obb` task
