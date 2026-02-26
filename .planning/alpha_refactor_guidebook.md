# AquaPose Alpha Development Guide

## 1. Purpose

AquaPose is evolving from a script-driven scientific pipeline into an event-driven scientific computation engine for multi-view 3D pose inference.

---

## 2. Core Identity

AquaPose must be: deterministic, replayable, fully introspectable, stage-extensible, observer-driven, CLI-executable, and script-agnostic.

If a change violates any of these, it is architecturally suspect.

---

## 3. Architectural Layers

The system has strict one-way dependencies between three layers.

**Layer 1 — Core Computation** (`src/aquapose/core/`)
Pure computation modules: calibration, detection, segmentation, tracking, midline extraction, triangulation, optimization. They accept structured inputs, return structured outputs, have no side effects, do not write files, do not generate visualizations, and do not import dev tooling. They are blind to diagnostics.

**Layer 2 — Execution Engine** (`src/aquapose/engine/`)
The pipeline orchestrator. Defines stage order, manages execution state, emits structured lifecycle events, handles execution modes, coordinates observers, and owns artifact management. All execution flows through this layer. There is exactly one canonical entrypoint.

**Layer 3 — Observability** (observers attached to engine)
External to computation. Observers may record intermediate data, generate diagnostics, write reports, save artifacts, collect timing, and capture optimizer snapshots. Observers may not mutate pipeline state, change stage logic, or control execution flow. Observers are passive.

**Import discipline:**
```
core/   → nothing (only stdlib, third-party, and core internals)
engine/ → core/
cli/    → engine/
```
No exceptions. No `TYPE_CHECKING` backdoors. Enforced by lint rule from the first commit that touches `core/` or `engine/`.

---

## 4. Source Layout

```
src/aquapose/
  core/               # Layer 1: pure computation modules
    calibration/
    detection/
    segmentation/
    tracking/
    association/
    reconstruction/
  engine/             # Layer 2 + 3: orchestration and observability
    pipeline.py       # PosePipeline orchestrator
    stages.py         # Stage Protocol, PipelineContext
    events.py         # Event dataclasses
    observers.py      # Observer base, attachment protocol
    config.py         # Frozen config hierarchy
    artifacts.py      # Artifact path management
  cli/                # Thin wrapper
```

---

## 5. Pre-Pipeline: Input Materialization

Resolved before the pipeline loop starts. Not stages — these are input preparation.

**Frame Source**
- Loads and decodes video frames lazily via `FrameSource.batches()`
- Applies undistortion using calibration intrinsics by default
- Yields `FrameBatch` — all stages receive clean, undistorted frames
- Batch size is configurable; pipeline is stateless across batches

**Calibration**
- Loads intrinsics and extrinsics from AquaCal calibration file
- Produces structured `CalibrationData` (projection models, camera parameters)
- Available to all stages via `PipelineContext`
- Intrinsics are consumed by the frame source for undistortion; extrinsics and projection models are passed through to stages that need them

**Outer loop:**
```python
for batch in frame_source.batches(size=config.batch_size):
    context = PipelineContext(frames=batch, calibration=calibration, carry=carry, ...)
    pipeline.run(context)
    carry = context.carry
```

The pipeline sees each batch as a complete unit of work. Stages never know batching exists. Cross-batch state (tracking continuity) is handed forward explicitly via a typed `CarryForward` object, not accumulated internally. Stitching and aggregation across batches lives outside the pipeline loop.

---

## 6. Pipeline Stages

### Stage 1 — Detection
*Swappable backend: model-based*

- **In:** frames, detection config
- **Out:** `detections_per_frame` — `list[FrameDetections]` per camera per frame. Empty lists for cameras with no fish.
- Cameras are processed independently — no cross-view logic here
- Confidence threshold and NMS parameters are backend config
- A 13-camera rig with partial overlap means most cameras will have zero or few detections per frame; this is normal, not an error
- "model-based" is the current backend; no specific alternative backends are planned

### Stage 2 — Midline
*Swappable backend: segment-then-extract / direct pose estimation*

- **In:** detections, frames
- **Out:** `annotated_detections` — each detection enriched with its 15-point 2D midline + half-widths. Midlines travel with their parent detection from this point forward. Segment-then-extract runs segmentation model (U-Net, SAM, etc.) → skeletonize → BFS internally. Direct keypoint produces midlines in a single model call.
- All backends must produce the same output structure: 15 arc-length-sampled points with half-widths
- Detections that fail midline extraction (e.g. degenerate masks) produce a flagged empty midline, not an exception
- Midline orientation (head vs tail) is not resolved here — that's a reconstruction concern
- The direct pose estimation backend is planned but not currently implemented

### Stage 3 — Cross-View Association

- **In:** annotated detections, projection models
- **Out:** `associated_bundles` — groups of annotated detections matched across cameras for each physical fish per frame, with camera assignments. Currently matches on bbox centroids; midline data is carried through and available for future association methods.
- A fish visible in N cameras produces one bundle with N entries; a fish visible in only one camera still produces a bundle (single-view)
- Unassociated detections (false positives, reflections) are discarded here
- This stage was previously internal to the tracker — extracting it is a key structural change in this refactor

### Stage 4 — Tracking
*Swappable backend: Hungarian matching*

- **In:** associated bundles, CarryForward (previous batch's track state)
- **Out:** `tracks_per_frame` — associated bundles promoted to persistent fish IDs via temporal matching with population constraints. Lifecycle states (probationary → confirmed → coasting → dead). Updates CarryForward for next batch.
- Tracking is purely temporal — all cross-view correspondence is resolved by stage 3
- CarryForward is typed narrowly to what tracking needs (previous frame's track positions, IDs, lifecycle states), not a generic container
- Population constraint enforces known fish count when available
- First batch of a run has empty CarryForward; tracks bootstrap from association output
- "Hungarian" matching is the current backend; no additional backends are currently planned

### Stage 5 — Reconstruction
*Swappable backend: triangulation / curve optimizer*

- **In:** tracked bundles (annotated detections with persistent IDs), projection models
- **Out:** `midlines_3d` — per-frame `dict[fish_id, Spline3D]` (7-control-point 3D B-splines) + `dropped: dict[fish_id, DropReason]` for fish that failed reconstruction.
- Both backends must resolve head-tail orientation before or during reconstruction
- Single-view fish cannot be reconstructed; they appear in `dropped` with an appropriate reason
- Triangulation backend handles midline point correspondence across views; curve optimizer avoids this by fitting a parametric curve directly
- Output spline control point count (7) is config, not hardcoded

---

## 7. Stage Interface Design

Stages are defined via `typing.Protocol` (structural typing, no inheritance required).

Data flows through a strongly typed `PipelineContext` dataclass that accumulates results. Each stage accepts `PipelineContext`, reads what it needs, appends typed results, and returns the updated context.

Stages are logically stateless but may cache expensive initialization (loaded models, calibration matrices). `run()` is a pure function of inputs and config — no hidden mutation between runs. Same config and inputs → identical behavior. If you tear down a stage and rebuild it with the same config, you get identical behavior. That's the test.

Stages never emit events, write files, or access the observer system. The pipeline wraps each stage call and emits lifecycle events by inspecting stage outputs. Stages return data; the pipeline interprets it.

Stages validate preconditions cheaply at the top of `run()` — assert that required context fields are present. Don't validate data deeply, just confirm the contract.

Stage ordering is an explicit ordered list, not a dependency DAG.

---

## 8. Swappable Backends vs Configurable Models

A **backend** represents a fundamentally different approach to a stage's task — different algorithm, different execution pattern, or different intermediate representations. Backends are registered in a stage-level registry and resolved from config at pipeline construction time. Examples: segment-then-extract vs direct pose estimation, triangulation vs curve optimization.

A **configurable model** is a choice within a backend — a different trained model, architecture, or parameter set that doesn't change the backend's structure or data flow. Model selection is handled via backend config, not by registering a new backend. Examples: YOLO vs DETR within the model-based detection backend, U-Net vs SAM within the segment-then-extract midline backend.

The test: if swapping it in changes the internal data flow or intermediate representations, it's a new backend. If it just changes what model gets loaded, it's config.

---

## 9. Event System

Events are typed dataclasses with a 3-tier taxonomy:

- **Pipeline lifecycle:** PipelineStart, PipelineComplete, PipelineFailed
- **Stage lifecycle:** StageStart, StageComplete (with summaries), StageFailed
- **Frame-level:** FrameProcessed, FrameSkipped (for tolerant mode)
- **Selective domain events** for stable scientific transitions (e.g. OptimizerIterationComplete, TracksUpdated) — emit meaning, not mechanics

Event naming convention: present-tense (PipelineStart, StageComplete, FrameProcessed).

Stages never emit events. The pipeline emits all events by inspecting stage outputs. If frame-level granularity is needed, the stage returns per-frame structured output and the pipeline emits FrameProcessed events by iterating that output.

---

## 10. Observer Protocol

Observers subscribe to specific event types (not receive-all, not method-per-event).

Observers are synchronous by default — the pipeline blocks on each observer call. Determinism is mandatory. If an observer needs non-blocking behavior (heavy visualization, disk I/O), it manages its own internal queue or worker thread.

Observers may not mutate pipeline state, change stage logic, or control execution flow. They are passive consumers of events.

Alpha observers: timing, raw export (HDF5), visualization, diagnostics.

---

## 11. Configuration System

Frozen dataclasses, hierarchical by stage. Loading precedence: defaults → YAML → CLI overrides → freeze.

Each stage receives only its own config subtree at construction. Stages never read YAML files directly — config parsing is the CLI/entrypoint's job.

Execution modes (production, diagnostic, synthetic, benchmark) are named config presets merged before freezing. The mode selects a preset, the preset merges into config, and from that point forward the pipeline doesn't know or care what "mode" it's in. No branching inside stage logic.

The full serialized config is logged as the first artifact of every run — the reproducibility contract. Given identical inputs, identical configuration, and identical random seeds, the pipeline must produce identical outputs.

Run identity: timestamp-based (`run_20260225_143022`).

---

## 12. Error Handling

Configurable, default to fail-fast.

- **`strict` (default):** Exception propagates, run fails, partial artifacts preserved for debugging. Pipeline emits StageFailed event before dying so observers can capture diagnostics.
- **`tolerant`:** Pipeline logs the failure, emits FrameSkipped event with the exception, inserts an empty result for that frame, and continues. A run-level report tallies all skipped frames so nothing is silently lost.

Strict for production and benchmarking. Tolerant for exploratory runs on messy data. Never tolerant by default.

---

## 13. Artifacts

Artifacts are first-class citizens — structured, named consistently, associated with a run ID, reproducible from inputs.

File writing is not allowed inside stage functions. All artifacts are managed centrally by the pipeline. Stages and observers that need to persist artifacts request a write path from the pipeline.

Default output location: `~/aquapose/runs/{run_id}/`, overridable via `AQUAPOSE_OUTPUT_DIR` environment variable or `config.output_dir`.

```
~/aquapose/runs/{run_id}/
  config.yaml
  stages/
    detection/
    midline/
    association/
    tracking/
    reconstruction/
  observers/
    timing/
    diagnostics/
    visualization/
  logs/
```

---

## 14. CLI

Single command + flags: `aquapose run --mode diagnostic --config path.yaml`

The CLI is a thin wrapper over PosePipeline. No subcommands per mode. No script may call stage functions directly, reimplement orchestration logic, or bypass stage sequencing.

---

## 15. Migration Strategy

Clean-room rebuild — not an incremental migration.

### Build Order

1. Stage interface definition (Protocol, PipelineContext) + import boundary lint rule
2. Config dataclasses
3. Event system (types + emitter)
4. Observer base + attachment protocol
5. Pipeline skeleton (wires stages, emits events, no real computation)
6. Each stage migration (one commit per stage)
7. Each observer extraction (one commit per observer)
8. CLI entrypoint

### Priority

Stage interfaces first — everything else is shaped by them. Start with two or three real stages (detection, midline, reconstruction), define the interface they share, and resist the urge to abstract until you see the actual commonality.

### Commit Discipline

Conventional commits: `refactor(engine): port detection stage`

One commit = one independently valid state of the system. Every commit leaves the project importable and tests passing. Never commit a half-migrated stage. If a stage migration is too big for one commit, split it into "define new stage wrapping old logic" and "remove old callsite" as two commits.

### Structural Checks Per Commit

**From commit 1:** Import boundary enforcement + full lint/typecheck (`ruff` + `basedpyright`).

**Phased custom rules** (added in the same commit as the code they govern):
- No file I/O in stage `run()` → add when first stage is migrated
- No mutable class attrs on stages → add with stage interface commit
- Observers don't import from `core/` internals → add with observer base class

### Verification

Golden data generated as a standalone preparatory commit before any stage migration begins.

Each ported stage verified with:
- **Interface tests:** stage.run(context) produces correct output types, honors the contract
- **Numerical regression tests:** against golden data, asserting numerical equivalence within tolerance

Regression tests kept and marked `@pytest.mark.regression` — run outside the fast test loop but kept as a safety net for subtle numerical drift.

**Golden data tiers:**
- Trusted stages (detection, segmentation): snapshot outputs, assert tight tolerances
- Suspect stages (known accuracy issues): snapshot inputs only, write interface tests during migration, snapshot outputs after accuracy is fixed

### Code Disposition

- Full cleanup during port — refactor internals to match new conventions
- Nothing is sacred — everything can be refactored as long as behavior is preserved
- Existing scripts archived to `scripts/legacy/` then deleted after alpha is stable
- Existing tests rewritten alongside new interfaces
- Minimal new dependencies allowed (not pydantic — frozen dataclasses already decided)

---

## 16. Definition of Done

Alpha stabilization is complete when:

- There is exactly one canonical pipeline entrypoint
- All scripts invoke PosePipeline
- `aquapose run` produces 3D midlines from video input through the new pipeline
- Diagnostic functionality is implemented via observers
- Synthetic mode runs through the pipeline (as a stage adapter, not a pipeline bypass)
- Timing, raw export (HDF5), visualization, and diagnostics are all implemented as observers
- No stage imports dev tooling; dev tooling depends on core, never vice versa
- The CLI is a thin wrapper over PosePipeline
- No script calls stage functions directly

---

## 17. Governing Principles

Complexity is allowed. Entanglement is not.

Observability is encouraged. Pipeline duplication is forbidden.

The pipeline is sacred. Everything else is modular.

New developers extend AquaPose by attaching observers — not by writing new scripts.

---

## 18. Discretionary Items

Left to implementation judgment:

- Observer attachment mechanism (register at construction vs add/remove dynamically)
- Exact PipelineContext field structure and typing
- Compression/serialization details for config logging
- Loading skeleton for YAML config parsing
- Specific custom lint rules and when to introduce them
- Per-stage decision on retaining regression tests beyond the guidelines above
