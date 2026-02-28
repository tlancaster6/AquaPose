---
created: 2026-02-28T00:17:00.000Z
title: Clean up and reorganize pipeline config schema
area: engine
files:
  - src/aquapose/engine/config.py
  - src/aquapose/engine/cli.py
---

## Problem

The config.yaml produced by the `init-config` CLI command contains unused parameters (leftover from earlier pipeline iterations) and has no logical ordering. File paths are scattered throughout, and stage-specific settings don't follow pipeline execution order. This makes configs confusing to edit and hard to reason about.

## Solution

1. Audit config schema against actual pipeline usage — remove any parameters that no stage, observer, or CLI path reads.
2. Reorganize the YAML template produced by `init-config`:
   - **File paths grouped at top** (video_dir, calibration_path, output_dir, weights paths, etc.)
   - **Stage settings in pipeline execution order** (detection → tracking → association → midline → reconstruction)
   - **Observer/diagnostic settings at bottom**
3. Update config dataclass field ordering to match.
4. Ensure existing configs still parse correctly (backward compat for field ordering is free in dataclass/dict loading).
