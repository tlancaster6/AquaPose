# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-19)

**Core value:** Accurate single-fish 3D reconstruction from multi-view silhouettes via differentiable refractive rendering
**Current focus:** Phase 1 — Calibration and Refractive Geometry

## Current Position

Phase: 1 of 6 (Calibration and Refractive Geometry)
Plan: 1 of TBD in current phase
Status: Executing
Last activity: 2026-02-19 — Completed 01-01: calibration loader and refractive projection model

Progress: [█░░░░░░░░░] 10%

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: 45 min
- Total execution time: 0.75 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-calibration-and-refractive-geometry | 1 | 45 min | 45 min |

**Recent Trend:**
- Last 5 plans: 45 min
- Trend: —

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: AquaCal is numpy-based — refractive projection must be reimplemented in PyTorch (AquaCal used only for loading calibration JSON, not for forward projection)
- [Init]: Phase 3 (Fish Mesh) depends only on Phase 1, not Phase 2 — can develop in parallel with segmentation if calendar time matters
- [Init]: Temporal smoothness loss (RECON-02) built in Phase 4 but only activates in Phase 5 when tracking provides associations — build the hook, wire it in Phase 5
- [Init]: All APIs are batch-first (list of fish states) from day one — even Phase 4 single-fish code uses single-element lists
- [01-01]: Cross-validation compares against AquaCal NumPy (not AquaMVS PyTorch) — AquaMVS not importable in hatch env due to missing open3d/lightglue; AquaCal NumPy is the actual ground truth
- [01-01]: K_inv float32 inversion tolerance set to atol=1e-4 — float32 with fx=1400 produces ~6e-5 error, expected float32 precision not a bug

### Pending Todos

None.

### Blockers/Concerns

- [Phase 1]: Z-uncertainty budget not yet quantified — must compute analytically before optimization code is written
- [Phase 2]: MOG2 female recall under worst-case conditions (stationary, low contrast) not yet measured — most likely operational failure mode
- [Phase 4]: PyTorch3D sigma/gamma hyperparameters for this rig's fish pixel sizes unknown — empirical sweep needed during Phase 4 development

## Session Continuity

Last session: 2026-02-19
Stopped at: Completed 01-01-PLAN.md (calibration loader and refractive projection model)
Resume file: .planning/phases/01-calibration-and-refractive-geometry/01-02-PLAN.md
