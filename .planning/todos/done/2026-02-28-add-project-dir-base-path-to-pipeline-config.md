---
created: 2026-02-28T00:16:45.846Z
title: Add project_dir base path to pipeline config
area: engine
files:
  - src/aquapose/engine/config.py
---

## Problem

Pipeline config (config.yaml) currently requires all paths (video_dir, calibration_path, output_dir, weights paths, etc.) to be absolute. This is fragile — moving the project directory or running on a different machine breaks every path.

A `project_dir` top-level parameter would allow all other paths in config.yaml to be specified relative to it, making configs portable across machines and directory layouts.

## Solution

Add `project_dir` as an optional first-class parameter in the pipeline config schema. When present, resolve all other path fields relative to `project_dir` during config loading. When absent, behavior is unchanged (paths interpreted as-is). Resolution should happen early in config parsing so downstream code always sees absolute paths.
