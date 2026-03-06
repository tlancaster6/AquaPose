# Phase 15 Bug Ledger — v1.0 Quirks Preserved During Stage Migrations

This ledger documents all known v1.0 behavioral quirks, design debts, and
intentional deviations from "clean" design that were preserved during the
Phase 15 stage migration to ensure numerical equivalence with the v1.0 pipeline.

---

## Stage 3 (AssociationStage) — Output Not Consumed by Stage 4

**Quirk:** Stage 3 (`AssociationStage`) produces `context.associated_bundles`,
but Stage 4 (`TrackingStage`) does NOT consume them for its primary tracking
logic. Instead, `TrackingStage` (via `HungarianBackend`) reads
`context.detections` (Stage 1 raw output) and re-derives cross-camera
association internally through `FishTracker.update()`.

**Why:** `FishTracker` in v1.0 performed both association and tracking in a
single monolithic call. Separating these into Stages 3 and 4 would require
restructuring `FishTracker` internals, which risks breaking numerical
equivalence. The Stage 3 output (`associated_bundles`) is a data product for
future backends and observers only.

**Impact:** Stage 3 is effectively a pass-through for the default pipeline. Its
output is computed but not used by Stage 4's primary path. This means Stage 3
adds computational overhead without affecting Stage 4 output in the default
configuration.

**Resolution path:** A future "clean" tracking backend would consume
`associated_bundles` directly, eliminating the redundant association in Stage 4.
Until then, this is an accepted design debt.

---

## Stage 4 (TrackingStage) — Hungarian Backend Reads Stage 1, Not Stage 3

**Quirk:** `HungarianBackend.track_frame()` receives both `bundles` (Stage 3
output) and `detections_per_camera` (Stage 1 output) but uses only
`detections_per_camera` for the actual tracking via `FishTracker.update()`.

**Why:** Preserving exact v1.0 numerical equivalence. The `bundles` parameter
is passed for future extensibility.

**Status:** Documented in `core/tracking/stage.py` module docstring and the
`TrackingStage` class docstring.

---

## Stage 5 (ReconstructionStage) — MidlineSet Assembly from Decoupled Stages

**Quirk:** In v1.0, `MidlineExtractor` had direct access to both tracks and
raw masks in a single monolithic call. In the new 5-stage model,
`ReconstructionStage` must assemble `MidlineSet` from two separate upstream
outputs: `context.tracks` (Stage 4) and `context.annotated_detections` (Stage 2).

**Assembly logic:**
1. For each `FishTrack` in `context.tracks[frame_idx]`, read `track.camera_detections`
   (cam_id → det_idx mapping).
2. For each (cam_id, det_idx), look up `context.annotated_detections[frame_idx][cam_id][det_idx]`.
3. Extract `annotated.midline` (a `Midline2D`) if non-None.
4. Build `dict[fish_id, dict[cam_id, Midline2D]]`.

**Why:** The Stage 2/4 decoupling means midline data and track identities are
in separate context fields. This assembly step reconstructs the MidlineSet that
v1.0's orchestrator assembled implicitly.

**Note:** Only fish with at least one camera midline are included in the
MidlineSet. Fish that are coasting (no active detections) will have empty
`camera_detections` and are skipped, matching v1.0 behavior where coasting
fish received no reconstruction update.

---

## Hardcoded Thresholds Extracted to Config

The following thresholds were hardcoded in v1.0 and have been extracted to
`ReconstructionConfig`:

| Threshold | v1.0 Value | Config Field |
|-----------|-----------|--------------|
| RANSAC inlier threshold | 50.0 px | `reconstruction.inlier_threshold` |
| Epipolar snap threshold | 20.0 px | `reconstruction.snap_threshold` |
| Max depth bound | None (disabled) | `reconstruction.max_depth` |

---

## Camera Skip Hardcoded to "e3v8250"

**Quirk:** All 5 stages default to `skip_camera_id="e3v8250"` (the centre
top-down wide-angle camera). This is not configurable via `PipelineConfig`
at the top level — each stage has its own default.

**Why:** The v1.0 orchestrator hardcoded this skip at multiple points. The
5-stage model preserves this by embedding the default in each stage constructor.

**Resolution path:** A future `skip_camera_id` top-level config field would
allow centralised override.

---

## CurveOptimizer Statefulness Preserved

**Quirk:** `CurveOptimizerBackend` maintains a single `CurveOptimizer`
instance across all frames, preserving warm-start state. This is stateful
behavior — the Stage 5 output for frame N depends on the optimizer's internal
state from frames 0..N-1.

**Why:** v1.0 instantiated `CurveOptimizer` once per video run and called
`optimize_midlines()` repeatedly. Warm-starting dramatically improves
convergence speed and reconstruction quality.

**Impact:** `ReconstructionStage` with the curve_optimizer backend is NOT
idempotent across pipeline runs on the same stage instance. A fresh pipeline
run should construct a new `ReconstructionStage`.

---

## Association Config Was Empty Placeholder in Phase 14.1

**Quirk:** `AssociationConfig` was initially defined as an empty frozen
dataclass placeholder (Phase 14.1). In Phase 15-03, it was extended with
`expected_count`, `min_cameras`, and `reprojection_threshold`. The Stage 3
`RansacCentroidBackend` delegates to the existing `discover_births()` function
rather than reimplementing the centroid clustering logic.

---

*Last updated: Phase 15-05 (2026-02-26)*
