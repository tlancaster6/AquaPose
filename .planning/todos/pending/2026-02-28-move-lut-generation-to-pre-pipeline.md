---
created: 2026-02-28T22:00:00.000Z
title: Pass pre-generated LUTs via PipelineContext instead of loading from disk in AssociationStage
area: calibration
files:
  - src/aquapose/core/association/stage.py
  - src/aquapose/engine/pipeline.py
---

## Background

LUT *generation* was moved out of the association stage into its own CLI command (`aquapose prep generate-luts`), so the performance concern is resolved. However, `AssociationStage.run()` still discovers and loads LUT files from disk itself rather than receiving them as input.

## Remaining work

Load LUTs in the pipeline setup layer (alongside calibration) and pass them into `PipelineContext`. The association stage should receive LUTs as a required context field, not reach out to disk. This gives early failure if LUTs are missing and keeps stages as pure computation.
