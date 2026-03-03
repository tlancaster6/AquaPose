---
gsd_state_version: 1.0
milestone: v3.3
milestone_name: Chunk Mode
status: roadmap_complete
last_updated: "2026-03-03T22:45:00.000Z"
progress:
  total_phases: 3
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-03)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** v3.3 Chunk Mode — Phase 51: Frame Source Refactor

## Current Position

Phase: 51 of 53 (Frame Source Refactor)
Plan: 1 of 2 complete
Status: In progress
Last activity: 2026-03-03 — Completed 51-01 FrameSource protocol + stage migration

Progress: [░░░░░░░░░░] 10% (0/3 phases, 1 plan complete)

## Accumulated Context

### Decisions

See PROJECT.md Key Decisions table for full history.

Key decisions for v3.3:
- Per-chunk association (not global) — bounds O(T²) complexity, isolates failures
- Identity stitching via track ID continuity — lightweight, leverages OC-SORT carry-forward
- Orchestrator owns HDF5 output — per-chunk observer would fire incorrectly
- Diagnostic mode and chunk mode are mutually exclusive — different purposes, bounded scope
- No FishState3D in handoff — 3D re-ID premature; add if re-ID failures observed in practice

Phase 51 Plan 01 decisions:
- VideoSet retained in io/video.py — observers still import it; Plan 02 handles removal
- MidlineStage keeps calibration_path param for ForwardLUT loading in orientation resolution
- VideoFrameSource shared by DetectionStage and MidlineStage (single construction in build_stages)
- FrameSource is runtime_checkable Protocol — enables isinstance checks without inheritance

### Pending Todos

17 pending todos from v2.2 — see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-03-03
Stopped at: Completed 51-01-PLAN.md — FrameSource protocol, VideoFrameSource, stage migration
Resume file: None
