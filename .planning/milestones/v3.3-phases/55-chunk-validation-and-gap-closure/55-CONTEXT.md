# Phase 55: Chunk Validation and Gap Closure - Context

**Gathered:** 2026-03-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Close all gaps from v3.3 milestone audit: validate chunk orchestrator correctness with stage-level mock tests (INTEG-03), fix manifest.json start_frame, and formally verify Phase 53 requirements (OUT-02, INTEG-01, INTEG-02). The tracklet builder carry-over bug has already been fixed (builders reset in `from_state()`) — this phase adds tests and closes remaining gaps.

</domain>

<decisions>
## Implementation Decisions

### INTEG-03 Validation Tests
- Stage-level mocks only — no real-data e2e tests (too slow, not meaningfully different from unit tests)
- **Degenerate case test:** `chunk_size=null` vs `chunk_size >= total_frames` — same computation window, output must be identical. Tests full orchestrator path (ChunkFrameSource wrapping, identity stitching with no prev handoff, global frame offset at 0, Midline3DWriter flush)
- **Multi-chunk mechanical correctness:** 2 chunks with mocked/synthetic pipeline contexts. Verify frame offsets correct (chunk 2 offset by chunk_size), identity map applied to fish IDs, HDF5 writes in global frame order, handoff carries state
- **Reword INTEG-03 requirement** from "numerically equivalent" to "degenerate single-chunk run produces identical output; multi-chunk runs produce structurally correct output with correct frame offsets and identity continuity"
- Mock at stage level (`PosePipeline.run()` returns canned `PipelineContext`), similar to existing `test_chunk_orchestrator.py`

### Manifest start_frame Fix
- Pass `chunk_start` through `build_observers()` to `DiagnosticObserver`
- DiagnosticObserver writes actual `start_frame` value in manifest.json chunk entries (currently always `None`)
- Orchestrator already knows `chunk_start` at line ~249 — wire it through the same path as `chunk_idx`

### Builder Bug Fix (already committed)
- `ocsort_wrapper.py:from_state()` now resets builders while preserving tracker state and ID mappings
- Empty builders are re-created for existing track IDs so `update()` doesn't KeyError
- Test updated: `test_state_roundtrip_resets_builders` verifies only batch2 frames in output
- The bug caused 98 global IDs for 9 fish over 200 frames — massive identity fragmentation at chunk boundaries

### Phase 53 Formal Verification
- Integration checker already confirmed wiring for OUT-02, INTEG-01, INTEG-02
- Formal verification = run gsd-verifier on Phase 53 to produce 53-VERIFICATION.md
- No new code needed — just documentation closure

### Claude's Discretion
- Exact mock data structures for test scenarios
- Whether to add builder-reset-specific regression test beyond the existing `test_state_roundtrip_resets_builders`
- Test file organization (extend existing `test_chunk_orchestrator.py` vs new file)

</decisions>

<specifics>
## Specific Ideas

- The builder carry-over bug was discovered from a real run (`~/aquapose/projects/YH/runs/run_20260304_172806/`) with chunk_size=100 where chunk 1 tracklets had 200 data points in a 100-frame range
- Chunk boundary produced instant drop from 9 fish to 3, with identity_map having 118 entries for 9 real fish
- The degenerate case test (chunk_size >= total) is the strongest validation — same computation path, must be byte-identical

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/unit/engine/test_chunk_orchestrator.py`: 6 existing tests for `_stitch_identities` and `_chunk_boundaries` — extend this
- `tests/unit/engine/test_chunk_handoff.py`: 8 tests for handoff, ChunkFrameSource, config — patterns to follow
- `_stitch_identities()` in `orchestrator.py:24-84` — the function under test
- `build_observers()` in `observer_factory.py` already accepts `chunk_idx: int = 0`

### Established Patterns
- Orchestrator tests mock `PosePipeline` and `build_stages` to return canned contexts
- DiagnosticObserver tests use `tmp_path` fixtures for output directory
- `TrackletGroup` and `Tracklet2D` frozen dataclasses used for synthetic test data

### Integration Points
- `orchestrator.py:265-274`: where `build_observers(chunk_idx=chunk_idx)` is called — add `chunk_start` param here
- `observer_factory.py:82-86`: where `DiagnosticObserver(chunk_idx=chunk_idx)` is constructed — add `chunk_start`
- `diagnostic_observer.py:226`: where `"start_frame": None` is written — replace with actual value

</code_context>

<deferred>
## Deferred Ideas

- Viz frame offset resilience for skipped/failed chunks (the loader could use manifest start_frame once populated, but graceful handling of missing chunks is a separate concern)
- `generate_all()` orphaned from viz CLI — CLI reimplements iteration inline; could be unified but cosmetic

</deferred>

---

*Phase: 55-chunk-validation-and-gap-closure*
*Context gathered: 2026-03-04*
