# Phase 13: Engine Core - Context

**Gathered:** 2026-02-25
**Status:** Ready for planning
**Source:** `.planning/inbox/alpha_refactor_decisions.md` and `.planning/inbox/refactor_doctrine.md`

<domain>
## Phase Boundary

The architectural skeleton exists — Stage Protocol, PipelineContext, event system, observer base, config hierarchy, pipeline orchestrator, and enforced import boundary — ready for stages to be plugged in. No stage migrations, no observers, no CLI in this phase.

</domain>

<decisions>
## Implementation Decisions

### Stage Protocol
- Defined via `typing.Protocol` (structural typing, no inheritance required)
- Single `run()` method that is a pure function of inputs + config
- Stages are logically stateless but may cache expensive initialization (loaded models, calibration matrices)
- Same config → identical behavior — that's the test
- Stage ordering via explicit ordered list (not dependency DAG)
- Each stage receives only its own config subtree at construction

### PipelineContext
- Strongly typed dataclass that accumulates results stage-by-stage
- Each stage accepts PipelineContext, reads what it needs, appends typed results, returns updated context
- Not a dict (too loose), not direct stage-to-stage IO (too rigid)
- No implicit shared state — all data flows through context fields

### Event System
- Events are typed dataclasses with a 3-tier taxonomy:
  - Pipeline lifecycle: PipelineStart, PipelineComplete, PipelineFailed
  - Stage lifecycle: StageStart, StageComplete (with summaries)
  - Frame-level: FrameProcessed (for frame-based stages)
  - Selective domain events for stable scientific transitions (e.g., OptimizerIterationComplete, TracksUpdated) — emit meaning, not mechanics
- Event naming convention: present-tense (PipelineStart, StageComplete, FrameProcessed)
- Synchronous delivery — pipeline blocks on each observer call. Determinism is mandatory.

### Observer Base
- Observers subscribe to specific event types (not receive-all, not method-per-event)
- Synchronous by default — if an observer needs non-blocking behavior, it manages its own internal queue/worker thread
- Observers are passive — may not mutate pipeline state, change stage logic, or control execution flow

### Configuration System
- Frozen dataclasses, hierarchical by stage
- Loading precedence: defaults → YAML → CLI overrides → freeze
- Raises on post-freeze mutation
- Execution modes (production, diagnostic, synthetic, benchmark) are named config presets merged before freezing
- Full serialized config logged as first artifact of every run — the reproducibility contract
- Run identity: timestamp-based (run_20260225_143022)
- Artifact path: `~/aquapose/runs/{run_id}/` as default, overridable via config

### Pipeline Orchestrator
- Single canonical entrypoint: `PosePipeline.run()`
- Defines stage order, manages execution state, emits lifecycle events, coordinates observers
- Owns artifact management
- Full serialized run config written as first artifact when `PosePipeline.run()` is called

### Source Layout and Import Boundary
- New `src/aquapose/engine/` package:
  - `stages.py` — Stage Protocol, PipelineContext
  - `events.py` — event dataclasses
  - `observers.py` — observer base, attachment
  - `config.py` — frozen config hierarchy
  - `pipeline.py` — PosePipeline orchestrator
- Strict one-way import boundary: engine/ imports computation modules, NEVER the reverse
- No TYPE_CHECKING exceptions — strict one-way at all levels

### Claude's Discretion
- Observer attachment mechanism (register at construction vs add/remove dynamically)
- Exact PipelineContext field structure and typing
- Compression/serialization details for config logging
- Loading skeleton for YAML config parsing
- Specific custom lint rules and when to introduce them

</decisions>

<specifics>
## Specific Ideas

- Doctrine document (`.planning/inbox/refactor_doctrine.md`) is the governing architectural reference — all decisions trace back to it
- "Complexity is allowed. Entanglement is not."
- "Build the system so that new developers extend it by attaching observers — not by writing new scripts."
- Not pydantic — frozen dataclasses already decided
- Clean-room rebuild, not incremental migration

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 13-engine-core*
*Context gathered: 2026-02-25*
