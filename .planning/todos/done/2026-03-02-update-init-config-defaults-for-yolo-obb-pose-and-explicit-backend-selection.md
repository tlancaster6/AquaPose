---
created: 2026-03-02T15:42:08.812Z
title: Update init-config defaults for yolo-obb pose and explicit backend selection
area: engine
files:
  - src/aquapose/engine/config.py
  - src/aquapose/engine/cli.py
---

## Problem

The `init-config` CLI command generates a config with outdated defaults and omits configuration for steps that have swappable backends or multiple model options. Users should get a config that reflects the current recommended pipeline (YOLO-OBB detection + keypoint pose estimation) and makes every backend/model choice explicit rather than hidden behind code-level defaults.

## Solution

1. Update default detection backend to `yolo-obb` (instead of whatever the current default is).
2. Update default pose/midline initialization to use the keypoint pose model.
3. Audit every pipeline stage for swappable backends or model options — any stage with more than one possible backend/model should appear in the generated config with the recommended option explicitly selected.
4. This complements the config cleanup todo (reorganize schema) but focuses specifically on default values and backend visibility.
