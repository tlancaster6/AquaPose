---
created: 2026-03-06T22:18:10.312Z
title: Fix core import boundary violation in frame_source
area: core
files:
  - src/aquapose/core/types/frame_source.py:25
---

## Problem

`core/types/frame_source.py` has a module-level import `from aquapose.io.discovery import discover_camera_videos` (line 25). This violates the architectural rule that `core/` only imports stdlib, third-party, and core internals. The `io` package is outside core/, so this creates a hard runtime dependency from Layer 1 (core computation) to the I/O layer.

This is the only import boundary violation in the codebase — all other core/ files correctly avoid importing from engine/, evaluation/, training/, and io/.

## Solution

Options (pick one):
1. **Move discovery logic out of core**: Have the caller (engine/CLI) discover video paths and pass them as constructor arguments to `VideoFrameSource`
2. **Lazy import**: Move the import inside the method that uses it (reduces coupling but doesn't eliminate it)
3. **Move `discover_camera_videos` into core/**: If the function is pure logic with no I/O-layer dependencies itself, it may belong in core/

Option 1 is cleanest — frame source construction is already done in engine/orchestrator, so passing resolved paths there is natural.
