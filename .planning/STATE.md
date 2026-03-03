---
gsd_state_version: 1.0
milestone: v3.3
milestone_name: Chunk Mode
status: unknown
last_updated: "2026-03-03T23:25:48Z"
progress:
  total_phases: 2
  completed_phases: 1
  total_plans: 5
  completed_plans: 3
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-03)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** v3.3 Chunk Mode — Phase 52: Chunk Orchestrator and Handoff (planned, awaiting Phase 51 completion)

## Current Position

Phase: 52 of 53 (Chunk Orchestrator and Handoff)
Plan: 1 of 3 complete
Status: In Progress
Last activity: 2026-03-03 — Completed 52-01: ChunkHandoff, ChunkFrameSource, write_handoff, chunk_size foundation

Progress: [████░░░░░░] 40% (1/3 phases, 3 plans complete)

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

Phase 51 Plan 02 decisions:
- VideoFrameSource created once in cli.py run command and passed to both build_stages and build_observers — avoids double construction
- VideoFrameSource imported at module level in cli.py so tests can patch aquapose.cli.VideoFrameSource
- Observers accept frame_source=None and fall back to synthetic black frames — preserves synthetic mode compatibility
- stop_frame in YAML now raises ValueError with _RENAME_HINTS pointing to max_frames on frame source

### Pending Todos

17 pending todos from v2.2 — see .planning/todos/pending/ (review for relevance)

### Blockers/Concerns

None.

### Phase 52 Planning Decisions

- ChunkHandoff placed in core/context.py (not engine/orchestrator.py) to avoid circular import: core/tracking/stage.py needs it, and core must not import from engine
- ChunkFrameSource context manager is a no-op (no open/close): orchestrator opens VideoFrameSource once for the whole run and creates ChunkFrameSource views
- track_id_to_global (dict[(camera_id, track_id) -> global_fish_id]) added to ChunkHandoff to enable OC-SORT track ID continuity stitching
- "Reclaim" (old global ID recovery after fish disappears for full chunk) deferred in favor of simpler fresh ID assignment — can revisit if needed
- next_global_id tracked separately from prev_handoff so it survives failed chunk resets
- _build_stages_for_chunk() helper in orchestrator avoids modifying build_stages() to accept an injection parameter

Phase 52 Plan 01 decisions:
- ChunkHandoff placed in engine/orchestrator.py per plan 52-01 instructions (plan note: watch for circular import if core/tracking/stage.py needs it)
- chunk_size=0 treated as None by callers via `config.chunk_size or None` convention, not coerced in load_config()
- ChunkFrameSource no-op context manager confirmed: orchestrator owns VideoFrameSource lifecycle

## Session Continuity

Last session: 2026-03-03
Stopped at: Completed 52-01 — ChunkHandoff, ChunkFrameSource, write_handoff, PipelineConfig.chunk_size. Next: Phase 52 Plan 02 (ChunkOrchestrator).
Resume file: None
