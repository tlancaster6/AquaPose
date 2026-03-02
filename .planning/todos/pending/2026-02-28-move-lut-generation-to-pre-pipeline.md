---
created: 2026-02-28T22:00:00.000Z
title: Move LUT generation to pre-pipeline setup
area: calibration
files:
  - src/aquapose/core/association/stage.py
  - src/aquapose/calibration/luts.py
  - src/aquapose/engine/pipeline.py
  - src/aquapose/cli.py
---

## Problem

LUT generation currently lives inside `AssociationStage.run()` as lazy initialization. Per GUIDEBOOK Section 5, LUTs are pre-pipeline input materialization — they should be resolved before the pipeline loop starts, alongside frame loading and calibration. Having them inside a stage violates the principle that stages are pure computation with no side effects (the current code uses `print()` for progress because the observer system can't reach it).

## Solution

Move LUT loading/generation to the CLI or pipeline setup layer (before the batch loop). LUTs should be loaded from cache or generated once, then passed into `PipelineContext` alongside calibration data. If LUTs are not present and cannot be generated (e.g. missing calibration), the pipeline should fail early with a clear error message rather than discovering the problem mid-run inside the association stage.

The association stage should receive LUTs as a required input, not generate them internally.
