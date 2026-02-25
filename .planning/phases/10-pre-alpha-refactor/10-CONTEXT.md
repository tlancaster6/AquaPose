# Phase 10: Pre-Alpha Refactor - Context

**Gathered:** 2026-02-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Transform AquaPose from a script-driven scientific pipeline into an event-driven scientific computation engine. Implement a 3-layer architecture (Core Computation → PosePipeline → Observers) as defined in the Refactor Doctrine (.planning/inbox/refactor_doctrine.md). Clean-room rebuild — not an incremental migration.

The goal is to formalize complexity, not reduce it. Research velocity must increase after this refactor.

</domain>

<decisions>
## Implementation Decisions

### Observer Protocol
- Observers subscribe to specific event types (not receive-all, not method-per-event)
- Events are typed dataclasses with a 3-tier taxonomy:
  - Pipeline lifecycle: PipelineStart, PipelineComplete, PipelineFailed
  - Stage lifecycle: StageStart, StageComplete (with summaries)
  - Frame-level: FrameProcessed (for frame-based stages)
  - Selective domain events for stable scientific transitions (e.g., OptimizerIterationComplete, TracksUpdated) — emit meaning, not mechanics
- Observers are synchronous by default — pipeline blocks on each observer call. Determinism is mandatory.
- If an observer needs non-blocking behavior (heavy viz, disk I/O), it manages its own internal queue/worker thread
- Event naming convention: present-tense (PipelineStart, StageComplete, FrameProcessed)

### Stage Interface Design
- Stages defined via `typing.Protocol` (structural typing, no inheritance required)
- Data flows through a strongly typed `PipelineContext` dataclass that accumulates results
  - Each stage accepts PipelineContext, reads what it needs, appends typed results, returns updated context
  - Not a dict (too loose), not direct stage-to-stage IO (too rigid)
- Stages are logically stateless but may cache expensive initialization (loaded models, calibration matrices)
  - `run()` is a pure function of inputs + config. No hidden mutation between runs.
  - Same config → identical behavior. That's the test.
- Stage ordering via explicit ordered list (not dependency DAG)

### Configuration System
- Frozen dataclasses, hierarchical by stage
- Loading precedence: defaults → YAML → CLI overrides → freeze
- Each stage receives only its own config subtree at construction
- Execution modes (production, diagnostic, synthetic, benchmark) are named config presets merged before freezing
- Full serialized config logged as first artifact of every run — the reproducibility contract
- Run identity: timestamp-based (run_20260225_143022)
- Artifact path: `~/aquapose/runs/{run_id}/` as default, overridable via config

### CLI Design
- Single command + flags: `aquapose run --mode diagnostic --config path.yaml`
- CLI is a thin wrapper over PosePipeline
- No subcommands per mode

### Source Layout
- New `src/aquapose/engine/` package for pipeline infrastructure:
  - stages.py (Stage Protocol, PipelineContext)
  - events.py (event dataclasses)
  - observers.py (observer base, attachment)
  - config.py (frozen config hierarchy)
  - pipeline.py (PosePipeline orchestrator)
- Strict one-way import boundary: engine/ imports computation modules, NEVER the reverse
  - No TYPE_CHECKING exceptions — strict one-way at all levels

### Migration Strategy
- Clean-room rebuild (not incremental)
- Build order:
  1. Stage interface definition (Protocol, PipelineContext, import boundary lint rule)
  2. Config dataclasses (one commit per layer if complex)
  3. Event system (types + emitter)
  4. Observer base + attachment protocol
  5. Pipeline skeleton (wires stages, emits events, no real computation)
  6. Each stage migration (one commit per stage, full cleanup)
  7. Each observer extraction (one commit per observer)
  8. CLI entrypoint
- Remove old code as you go — atomic commits enable rollback
- Existing scripts archived to scripts/legacy/ then deleted after alpha is stable
- Existing tests rewritten alongside new interfaces (not adapted)

### Refactor Boundaries
- Full cleanup during port — agents should refactor internals to match new conventions
- Nothing is sacred — everything can be refactored as long as behavior is preserved
- Minimal new dependencies allowed (but NOT pydantic — frozen dataclasses already decided)

### Commit Discipline
- Conventional commits: `refactor(engine): port detection stage`
- One commit = one independently valid state of the system
- Each commit must pass lint + typecheck
- Import boundary enforcement added phased (custom rules added when there's code to enforce against)

### Verification Criteria
- Structural checks: import boundaries + lint/typecheck + phased custom rules (no file I/O in stage run(), etc.)
- Golden data generated as standalone preparatory commit BEFORE any stage migration begins
- Each ported stage verified with:
  - Interface tests (stage.run(context) produces correct output)
  - Numerical regression tests (against golden data)
- After equivalence established: decide per-stage whether to keep regression test (pytest.mark.regression) or delete
  - Detection: quality results, keep regression
  - Segmentation: middling results, evaluate case-by-case
  - Triangulation/optimization: still under development, may not keep

### Definition of Done
- Full alpha: CLI entrypoint, all observers, all modes (production/diagnostic/synthetic), no scripts needed
- `aquapose run` produces 3D midlines from video input through the new pipeline
- Timing, diagnostic, visualization, HDF5 export all implemented as observers
- No script calls stage functions directly

### Claude's Discretion
- Observer attachment mechanism (register at construction vs add/remove dynamically)
- Exact PipelineContext field structure and typing
- Compression/serialization details for config logging
- Loading skeleton for YAML config parsing
- Specific custom lint rules and when to introduce them
- Per-stage decision on retaining regression tests

</decisions>

<specifics>
## Specific Ideas

- Doctrine document (.planning/inbox/refactor_doctrine.md) is the governing architectural reference — all decisions trace back to it
- "Complexity is allowed. Entanglement is not."
- "Build the system so that new developers extend it by attaching observers — not by writing new scripts."
- Synthetic execution is a stage adapter, not a pipeline bypass
- Modes alter behavior via configuration and observer selection, never via branching inside stage logic

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 10-pre-alpha-refactor*
*Context gathered: 2026-02-25*
