---
created: 2026-03-12T14:02:55.846Z
title: Iterate only active frames in reconstruction per-fish loop
area: reconstruction
files:
  - src/aquapose/core/reconstruction/stage.py:266
---

## Problem

In `_run_with_tracklet_groups`, the per-fish frame loop iterates over all frames in the chunk (`for frame_idx in range(frame_count)`), even though each fish is only present in a small subset of frames. With 300 frames per chunk and multiple fish, most iterations immediately hit the `frame_cameras.get(frame_idx, [])` empty-list path and skip.

## Solution

Replace `for frame_idx in range(frame_count)` with iteration over `sorted(frame_cameras.keys())` — only frames where the fish actually has detected cameras. The gap interpolation step already operates on the valid-frame set and doesn't require iterating all frames.
