---
created: 2026-03-02T01:12:39.595Z
title: Consolidate weights_path and keypoint_weights_path config fields
area: engine
files:
  - src/aquapose/engine/config.py
---

## Problem

The pipeline config has two separate fields for model weights: `weights_path` (used by the segmentation backend) and `keypoint_weights_path` (used by the pose estimation backend). Since these two backends are mutually exclusive — only one runs per pipeline invocation — having separate fields is redundant and confusing. It also led to a bug where `keypoint_weights_path` wasn't being resolved relative to `project_dir` (fixed in this session).

## Solution

1. Remove `keypoint_weights_path` from the config schema.
2. Use `weights_path` for whichever backend is active (seg or pose).
3. Update `SegmentationBackend` and `PoseEstimationBackend` to both read from `weights_path`.
4. Update `init-config` CLI template and any example configs.
5. Add a deprecation/migration note or alias for existing configs that still use `keypoint_weights_path`.
