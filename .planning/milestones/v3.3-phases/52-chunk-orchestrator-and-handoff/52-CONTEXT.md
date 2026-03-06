# Phase 52: Chunk Orchestrator and Handoff - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement ChunkOrchestrator, ChunkHandoff, identity stitching, and per-chunk HDF5 flush. Videos can be processed in fixed-size temporal chunks with state carried across boundaries. PosePipeline and stages remain unaware of chunking — the orchestrator presents each chunk as a complete short video.

Depends on Phase 51 (frame source refactor) being complete. Requirements: CHUNK-01 through CHUNK-05, IDENT-01, IDENT-02, OUT-01.

</domain>

<decisions>
## Implementation Decisions

### Identity Conflict Resolution
- When a tracklet group has tracklets matching multiple known global fish IDs (rare — indicates association error at chunk boundary), resolve via majority vote among matching tracklets
- Log a warning with the conflicting IDs; processing continues normally
- When a fish disappears for an entire chunk then reappears later: reclaim the old global ID if there's exactly one "missing" global ID that could match; otherwise assign fresh ID
- Identity stitching events (conflicts, new fish, reclaims) logged via standard Python `logging` at WARNING level — no new logging infrastructure needed

### Chunk Failure Handling
- If a chunk's pipeline run fails, skip the chunk and continue to the next
- After a failed chunk: next chunk starts with no carry-forward (fresh trackers) and fresh global ID range (starting from max_previous_id + 1) — the break in continuity is explicit in both frame data and ID numbering
- Skipped chunk frame ranges recorded in HDF5 root metadata so downstream consumers know about gaps
- No output written for failed chunks

### Progress Reporting
- Chunk-level progress only (one line per chunk): chunk number, frame range, fish count (continued/new/reclaimed), elapsed time, ETA
- ETA computed from running average of chunk processing times after first chunk completes
- Identity stitching summary is part of the chunk progress line (e.g., "Chunk 3/50 (2000-2999) — 9 fish, 0 new, 45s, ~35m left")
- New `-v` CLI flag on `aquapose run` (CLI only, not config): when set, attaches ConsoleObserver for per-stage output within each chunk; when absent, ConsoleObserver is not attached in chunk mode
- Per-stage output (ConsoleObserver) is suppressed by default in chunk mode to keep output clean for long runs

### Code Organization
- ChunkOrchestrator lives in `engine/orchestrator.py` alongside pipeline.py
- Midline3DWriter instantiated directly inside ChunkOrchestrator (not injected)
- `-v` verbosity flag is CLI-only, not a config field (runtime preference, not reproducibility concern)
- CarryForward class must be deleted and all references replaced with ChunkHandoff

### Claude's Discretion
- ChunkHandoff dataclass placement (core/context.py alongside old CarryForward, or engine/orchestrator.py)
- Identity stitching function organization (inline in orchestrator or separate helper)
- Chunk timeout for Leiden non-convergence (whether to impose, and threshold)
- Exact chunk progress line format

</decisions>

<specifics>
## Specific Ideas

- After a failed chunk, the ID gap and frame gap should make breakage obvious in the data — user explicitly wants failed chunks to be visible, not papered over
- Chunk progress line should include identity summary inline so you can watch fish counts stabilize across chunks
- The `-v` flag concept: chunk mode defaults to quiet, verbose opt-in via CLI flag

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Midline3DWriter` (io/midline_writer.py): Already supports chunked writes with 1000-frame buffer, `write_frame(frame_index, midlines)` — perfect for per-chunk HDF5 flush with global frame offset
- `OcSortTracker.get_state()/from_state()` (core/tracking/ocsort_wrapper.py): Full tracker state serialization including boxmot internals — enables exact continuation across chunks
- `HandoffState` (core/association/types.py): Pre-defined frozen dataclass stub with placeholder fields — designed for this phase
- `CarryForward` (core/context.py): Current tracker state carrier, to be absorbed into ChunkHandoff
- Standard Python `logging` used throughout codebase — no new logging setup needed

### Established Patterns
- `PosePipeline.run()` accepts optional `initial_context` with `carry_forward` — orchestrator can inject carry state per chunk
- `TrackingStage.run(context, carry)` returns `(context, new_carry)` tuple — pipeline already handles this special dispatch
- `build_stages(config)` factory constructs all 5 stages from config — orchestrator reuses this
- `build_observers(config, mode)` factory for observer attachment — orchestrator can conditionally skip ConsoleObserver
- Frozen dataclasses for cross-boundary state (CarryForward pattern)
- Atomic file writes not yet patterned but trivial (temp file + os.rename)

### Integration Points
- `PosePipeline` — orchestrator calls `run()` per chunk, no pipeline changes needed
- `PipelineConfig` — add `chunk_size` field
- `cli.py` `run` command — add `-v` flag, instantiate orchestrator instead of calling pipeline directly
- `HDF5ExportObserver` — must be disabled when chunk mode active (orchestrator owns HDF5)
- `PipelineContext.carry_forward` — will carry ChunkHandoff instead of CarryForward
- `TrackletGroup.tracklets` — each tracklet has `(camera_id, track_id)` needed for identity map lookups

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 52-chunk-orchestrator-and-handoff*
*Context gathered: 2026-03-03*
