---
gsd_state_version: 1.0
milestone: v3.4
milestone_name: Performance Optimization
status: unknown
last_updated: "2026-03-05T03:08:50.123Z"
progress:
  total_phases: 5
  completed_phases: 4
  total_plans: 8
  completed_plans: 7
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-05)

**Core value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation
**Current focus:** v3.4 Performance Optimization — Phase 59: Batched YOLO Inference

## Current Position

Phase: 59 of 59 (Batched YOLO Inference)
Plan: 03 of 03 complete
Status: Complete
Last activity: 2026-03-05 — Completed 59-03 (Batched midline inference)

Progress: [##########] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 4 (v3.4)
- Average duration: 4min
- Total execution time: 18min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 58-frame-i-o-optimization | 1 | 6min | 6min |
| 59-batched-yolo-inference | 3 | 12min | 4min |

## Accumulated Context
| Phase 57 P01 | 279 | 2 tasks | 2 files |
| Phase 58 P01 | 6min | 2 tasks | 4 files |
| Phase 59 P01 | 2min | 2 tasks | 4 files |
| Phase 59 P02 | 5min | 2 tasks | 5 files |
| Phase 59 P03 | 5min | 2 tasks | 5 files |

### Decisions

See PROJECT.md Key Decisions table for full history.
- [Phase 57]: Drop 2-camera ray-angle filter in vectorized path: masking per-point would require a body-point loop, defeating vectorization for negligible yield impact
- [Phase 57]: Drop first-pass water-surface check in vectorized path: above-water initial triangulations virtually always remain above-water after re-triangulation
- [Phase 58]: Queue maxsize=2 balances memory (2 frames x 12 cameras) vs prefetch benefit
- [Phase 58]: Undistortion runs in background thread so main thread receives ready-to-use frames
- [Phase 58]: Decode failure skips camera (warning) rather than killing the frame or raising
- [Phase 59]: Mutable BatchState (not frozen) to persist batch size reductions across calls
- [Phase 59]: batch_size=0 means no limit (send all inputs in one call) for config fields
- [Phase 59]: Extract parsing into shared helpers (_parse_results, _parse_box_results) to avoid duplication between detect() and detect_batch()
- [Phase 59]: Use r.orig_shape for frame dimensions in shared batch parser
- [Phase 59]: Crop extraction (CPU) separated from batch predict (GPU) in MidlineStage for clean OOM retry boundary
- [Phase 59]: Failed crop extractions produce null AnnotatedDetection at correct position, maintaining positional correspondence

### Profiling Data (v3.4 baseline)

- Single chunk (200 frames × 12 cameras): ~916s wall time
- GPU utilization: active 51% of time, avg 30% SM when active
- Bottleneck breakdown: detection ~13%, midline ~13%, frame I/O ~12%, reconstruction ~9%, association ~5%

### Research Flags (resolve before coding)

- Phase 58: Confirm `ChunkFrameSource.read_frame(global_idx)` seek is the sole I/O path during a chunk run; verify `VideoFrameSource` has no internal position-tracking state that conflicts with sequential background-thread reading
- Phase 59: Confirm Ultralytics result ordering guarantee on the pinned version in pyproject.toml; verify batch predict behavior when all input images are the same resolution
- Phase 57: Audit TF32 state used when outlier_threshold=10.0 was calibrated; check YH run metadata before coding begins

### Pending Todos

17 pending todos from v2.2 — see .planning/todos/pending/ (review for relevance)

### Roadmap Evolution

- Phase 60 added: End-to-End Performance Validation — consolidates deferred real-data checks (SC-2 etc.) into a single milestone gate

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-03-05
Stopped at: Re-executed 59-02-PLAN.md (Batched detection backends + stage refactor)
Resume file: None
