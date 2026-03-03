# Requirements: AquaPose v3.3 Chunk Mode

**Defined:** 2026-03-03
**Core Value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation

## v3.3 Requirements

Requirements for chunk processing milestone. Each maps to roadmap phases.

### Frame Source

- [x] **FRAME-01**: DetectionStage and MidlineStage receive frames from an injectable frame source instead of opening VideoSet directly
- [x] **FRAME-02**: Frame source yields `(frame_idx, dict[str, ndarray])` — local frame index plus per-camera undistorted frames
- [x] **FRAME-03**: `stop_frame` removed from pipeline config — frame windowing is a frame-source concern managed by the orchestrator

### Chunk Orchestration

- [x] **CHUNK-01**: ChunkOrchestrator processes video in fixed-size temporal chunks, each running full 5-stage pipeline independently
- [x] **CHUNK-02**: `chunk_size` config field in pipeline config with null/0 fallback to full-video mode (no behavioral change)
- [x] **CHUNK-03**: Warning emitted when `chunk_size < 100` frames (insufficient temporal evidence for reliable association scoring)
- [x] **CHUNK-04**: ChunkHandoff frozen dataclass replaces CarryForward — carries tracker state + identity map across chunk boundaries
- [x] **CHUNK-05**: Atomic handoff serialization (temp file + rename) to `handoff.pkl` after each chunk completes

### Identity

- [x] **IDENT-01**: Post-chunk identity stitching maps chunk-local fish IDs to globally consistent IDs via track ID continuity from OC-SORT carry-forward
- [x] **IDENT-02**: Unmatched tracklet groups (no known track IDs) receive fresh global IDs; matched groups inherit existing global fish IDs

### Output

- [x] **OUT-01**: Per-chunk HDF5 flush via existing Midline3DWriter with global frame offset applied
- [ ] **OUT-02**: HDF5Observer disabled when chunk mode is active — orchestrator owns HDF5 output as a run-level concern

### Integration

- [ ] **INTEG-01**: `aquapose run` uses ChunkOrchestrator; with chunk_size=null/0, single-chunk degenerate case matches current behavior exactly
- [ ] **INTEG-02**: Chunk mode and diagnostic mode are mutually exclusive — validation error if both active
- [ ] **INTEG-03**: Chunked output matches non-chunked output for the same video (validation test)

## Deferred Requirements

### Resumption

- **RESUME-01**: Restart from last completed chunk using serialized handoff
- **RESUME-02**: Detect and skip already-processed chunks on restart

### Advanced Chunking

- **ADV-01**: Adaptive chunk sizing based on tracklet density
- **ADV-02**: Parallel chunk processing across GPU workers
- **ADV-03**: Temporal overlap at chunk boundaries for smoother transitions

### Association Extensions

- **ASSOC-01**: Seed + resolve association — pre-assign known identities before Leiden clustering
- **ASSOC-02**: FishState3D in handoff for 3D re-identification of lost tracks

## Out of Scope

| Feature | Reason |
|---------|--------|
| Crash resumption | Handoff serialized to disk, but restart logic deferred — adds complexity for rare case |
| Adaptive chunk sizing | Fixed 1000-frame chunks are predictable and sufficient; optimize later if needed |
| Parallel chunk processing | Sequential processing is simpler; no GPU contention |
| Diagnostic mode + chunk mode | Different purposes — diagnostic for short-clip tuning, chunk for long-video production |
| Temporal overlap at boundaries | Carry-forward provides continuity without overlap; adds duplicate detection complexity |
| Seed + resolve association | Per-chunk association + post-hoc stitching is simpler and likely sufficient at 1000-frame chunks |
| FishState3D in handoff | 3D re-identification is premature; add only if re-ID failures observed in practice |
| Intermediate stage output persistence | Only final 3D midlines flushed per chunk; intermediate data released after each chunk |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FRAME-01 | Phase 51 | Complete (51-01) |
| FRAME-02 | Phase 51 | Complete (51-01) |
| FRAME-03 | Phase 51 | Complete |
| CHUNK-01 | Phase 52 | Complete |
| CHUNK-02 | Phase 52 | Complete |
| CHUNK-03 | Phase 52 | Complete |
| CHUNK-04 | Phase 52 | Complete |
| CHUNK-05 | Phase 52 | Complete |
| IDENT-01 | Phase 52 | Complete |
| IDENT-02 | Phase 52 | Complete |
| OUT-01 | Phase 52 | Complete |
| OUT-02 | Phase 53 | Pending |
| INTEG-01 | Phase 53 | Pending |
| INTEG-02 | Phase 53 | Pending |
| INTEG-03 | Phase 53 | Pending |

**Coverage:**
- v3.3 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-03-03*
*Last updated: 2026-03-03 after initial definition*
