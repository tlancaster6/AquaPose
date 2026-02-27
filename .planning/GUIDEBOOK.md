# AquaPose Development Guide

## 1. Purpose

AquaPose is an event-driven scientific computation engine for multi-view 3D fish pose inference. It reconstructs 3D fish midlines from multi-view silhouettes using a 13-camera aquarium rig with refractive calibration.

---

## 2. Core Identity

AquaPose must be: deterministic, replayable, fully introspectable, stage-extensible, observer-driven, CLI-executable, and script-agnostic.

If a change violates any of these, it is architecturally suspect.

---

## 3. Architectural Layers

The system has strict one-way dependencies between three layers.

**Layer 1 — Core Computation** (`src/aquapose/core/`)
Pure computation modules: calibration, detection, segmentation, 2D tracking, cross-camera association, midline extraction, triangulation, optimization. They accept structured inputs, return structured outputs, have no side effects, do not write files, do not generate visualizations, and do not import dev tooling. They are blind to diagnostics.

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
    calibration/      # AquaCal loading, refractive projection, ray casting, LUTs
    detection/        # YOLO detection, detector factory
    segmentation/     # U-Net inference, SAM pseudo-labels, crop utilities
    tracking/         # Per-camera 2D tracking (OC-SORT)
    association/      # Cross-camera tracklet association (scoring, clustering, refinement)
    midline/          # Midline extraction (segment-then-extract, direct pose)
    reconstruction/   # Triangulation, B-spline fitting
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

**Refractive Lookup Tables**
- **Forward LUT (pixel → ray):** For each camera, a precomputed grid mapping 2D pixel coordinates to 3D ray (origin, direction) in water-space. Bilinear interpolation at query time. Eliminates per-frame refraction math during association.
- **Inverse LUT (voxel → pixel):** The tank volume (2 m diameter × 1 m tall cylinder) discretized into a regular voxel grid (default 2 cm). Each voxel records which cameras can see it and the projected pixel coordinates in each. Provides: camera overlap graph, ghost-point lookups, fast reprojection.
- Both LUTs are computed once per camera setup, serialized to disk, and reloaded on subsequent runs. They are available to stages via `PipelineContext` alongside calibration data.

**Outer loop:**
```python
for batch in frame_source.batches(size=config.batch_size):
    context = PipelineContext(frames=batch, calibration=calibration, luts=luts, carry=carry, ...)
    pipeline.run(context)
    carry = context.carry
```

The pipeline sees each batch as a complete unit of work. Stages never know batching exists. Cross-batch state (2D tracking continuity) is handed forward explicitly via a typed `CarryForward` object, not accumulated internally. Stitching and aggregation across batches lives outside the pipeline loop.

---

## 6. Pipeline Stages

```
Detection → 2D Tracking → Cross-Camera Association → Midline → Reconstruction
```

### Stage 1 — Detection
*Swappable backend: model-based*

- **In:** frames, detection config
- **Out:** `detections` — per-camera per-frame bounding boxes with confidence scores. Empty lists for cameras with no fish.
- Cameras are processed independently — no cross-view logic here
- Confidence threshold and NMS parameters are backend config
- A 13-camera rig with partial overlap means most cameras will have zero or few detections per frame; this is normal, not an error

### Stage 2 — 2D Tracking
*Swappable backend: OC-SORT*

- **In:** `detections`, CarryForward (previous batch's per-camera track state)
- **Out:** `tracks_2d` — per-camera list of 2D tracklets, each a time-series of detections with a local track ID. Updates CarryForward for next batch.
- Each camera is tracked independently — no cross-camera awareness at this stage
- Tracklets carry per-frame status tags: `"detected"` (matched to a real detection) or `"coasted"` (Kalman prediction only). This distinction is consumed by association for must-not-link constraints and fragment merging.
- Local track IDs are scoped to a single camera — they have no cross-camera meaning
- OC-SORT extends standard SORT with observation-centric re-update, momentum, and lost-track recovery. No appearance model — within a single top-down camera view, fish are distinguishable by position and velocity alone.
- Fallback: plain SORT if OC-SORT proves problematic

### Stage 3 — Cross-Camera Association

- **In:** `tracks_2d`, calibration, LUTs
- **Out:** `tracklet_groups` — groups of tracklets matched across cameras, each representing one physical fish with a global ID. Also emits `handoff_state` for future chunk-aware operation.
- Assigns **global fish IDs** that are authoritative for all downstream stages. Local per-camera IDs are internal bookkeeping from this point forward.
- Algorithm overview (see `MS3-SPECSEED.md` for full design):
  1. Camera overlap graph from inverse LUT
  2. Pairwise tracklet scoring via ray-ray distance + ghost-point penalty
  3. Affinity graph construction + Leiden clustering with must-not-link constraints
  4. Same-camera fragment merging within clusters
  5. 3D consistency refinement (triangulate per-cluster, evict outlier tracklets)
- Per-frame confidence estimates (reprojection residuals, camera count, close-encounter flags) are attached to each group
- Handles partial observability (fish visible in 4–5 of 12 cameras) and tracklet fragmentation (multiple short tracklets per fish per camera)
- The pipeline accepts an optional `prior_context` for chunk-aware seeding and always emits `handoff_state`. Chunk orchestration is deferred; the stage operates as a single-chunk full-batch processor.

### Stage 4 — Midline
*Swappable backend: segment-then-extract / direct pose estimation*

- **In:** `tracklet_groups`, `detections`, frames
- **Out:** `annotated_detections` — midlines for detections belonging to confirmed tracklet-groups only. Ungrouped detections are skipped entirely.
- Segment-then-extract: crop → segmentation model (U-Net, SAM, etc.) → skeletonize → BFS → arc-length resample to N points + half-widths
- Direct pose estimation: crop → encoder + keypoint head → N keypoint coordinates (planned, not yet implemented)
- All backends must produce the same output structure: N arc-length-sampled points with optional half-widths
- Detections that fail midline extraction (e.g. degenerate masks) produce a flagged empty midline, not an exception
- Cross-camera group membership (from Stage 3) provides a head-tail consistency signal — if most cameras agree on head direction, flip outliers

### Stage 5 — Reconstruction
*Swappable backend: triangulation / curve optimizer*

- **In:** `tracklet_groups`, `annotated_detections`, calibration
- **Out:** `midlines_3d` — per-frame `dict[fish_id, Spline3D]` (B-spline 3D midlines) + `dropped: dict[fish_id, DropReason]` for fish that failed reconstruction.
- Triangulates using only the cameras known to observe each fish (from tracklet association). Per-fish, per-frame with known correspondence — no RANSAC needed for cross-view matching.
- Both backends must resolve head-tail orientation before or during reconstruction
- Single-view fish cannot be reconstructed; they appear in `dropped` with an appropriate reason
- Output spline control point count is config, not hardcoded

### PipelineContext Data Flow

| Stage | Reads | Writes |
|-------|-------|--------|
| 1. Detection | frames, calibration | `detections` |
| 2. 2D Tracking | `detections`, CarryForward | `tracks_2d`, CarryForward |
| 3. Association | `tracks_2d`, calibration, LUTs | `tracklet_groups`, `handoff_state` |
| 4. Midline | `tracklet_groups`, `detections`, frames | `annotated_detections` |
| 5. Reconstruction | `tracklet_groups`, `annotated_detections`, calibration | `midlines_3d` |

### Identity Model

- **Stage 2** assigns **local per-camera IDs** — meaningful only within a single camera's timeline
- **Stage 3** assigns **global fish IDs** — authoritative from this point forward
- Downstream stages reference global fish IDs only

### CarryForward

Per-camera 2D track state (positions, velocities, lifecycle per local tracklet). Each camera's tracker is independent — no cross-camera state in carry. Cross-camera association operates on complete tracklets within the batch, not incrementally.

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
    tracking/
    association/
    midline/
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

## 15. Milestone History

### v1.0 MVP (shipped 2026-02-25)
Clean-room rebuild from scripts to event-driven pipeline. 12 phases, 28 plans. Established the Stage protocol, PipelineContext accumulator, observer system, and CLI entrypoint.

### v2.0 Alpha (shipped 2026-02-27)
Stabilized architecture, implemented all 5 stages and 5 observers, synthetic mode, full diagnostic pipeline. 10 phases, 34 plans. Revealed that 3D reconstruction is broken due to structural cross-view correspondence failure.

### v2.1 Identity (in progress)
Pipeline reorder: Detection → 2D Tracking → Cross-Camera Association → Midline → Reconstruction. Replaces RANSAC centroid association and 3D bundle-claiming tracker with OC-SORT 2D tracking and trajectory-based tracklet association. Adds refractive LUTs for efficient geometric queries. Defers OBB detection and keypoint pose estimation to a future milestone.

---

## 16. Definition of Done

v2.1 Identity is complete when:

- Pipeline runs in the new stage order (Detection → 2D Tracking → Association → Midline → Reconstruction)
- Per-camera 2D tracking produces tracklets with local IDs and detected/coasted frame tags
- Cross-camera association produces globally consistent fish IDs from tracklet-level evidence
- Midline extraction operates only on associated tracklet-group detections
- Reconstruction uses pre-established correspondence (no RANSAC for cross-view matching)
- Refractive LUTs (forward and inverse) are precomputed, serialized, and reloaded
- Diagnostic visualization shows 2D tracklet trails and association group coloring
- Old tracking/association code (FishTracker, ransac_centroid_cluster, v2.0 AssociationStage/TrackingStage) is deleted
- `aquapose run` produces 3D midlines from video input through the reordered pipeline

---

## 17. Governing Principles

Complexity is allowed. Entanglement is not.

Observability is encouraged. Pipeline duplication is forbidden.

The pipeline is sacred. Everything else is modular.

New developers extend AquaPose by attaching observers — not by writing new scripts.

---

## 18. Discretionary Items

Left to implementation judgment:

- OC-SORT implementation choice (third-party package vs vendored)
- Leiden resolution parameter tuning
- Voxel grid resolution (default 2 cm, configurable)
- LUT serialization format (numpy .npz, HDF5, etc.)
- Forward LUT grid density per camera
- Tracklet association score thresholds (s_min, τ, T_min, T_saturate)
- Eviction threshold for 3D consistency refinement
- Exact PipelineContext field structure and typing
- Observer attachment mechanism (register at construction vs add/remove dynamically)
- Compression/serialization details for config logging
- Loading skeleton for YAML config parsing
- Specific custom lint rules and when to introduce them
