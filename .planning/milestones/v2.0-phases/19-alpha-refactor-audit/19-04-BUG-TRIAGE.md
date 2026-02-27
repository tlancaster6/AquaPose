# Phase 15 Bug Ledger — Triage

**Triaged:** 2026-02-26
**Input:** .planning/phases/15-stage-migrations/15-BUG-LEDGER.md
**Purpose:** Feed into 19-AUDIT.md (produced by Plan 19-03)

---

## Summary

| # | Item | Status | Severity |
|---|------|--------|----------|
| 1 | Stage 3 output not consumed by Stage 4 | Open | Warning |
| 2 | Hungarian Backend reads Stage 1, not Stage 3 | Open | Warning |
| 3 | MidlineSet assembly from decoupled stages | Accepted | — |
| 4 | Hardcoded thresholds extracted to config | Resolved | — |
| 5 | Camera skip hardcoded to "e3v8250" | Open | Warning |
| 6 | CurveOptimizer statefulness preserved | Accepted | — |
| 7 | AssociationConfig was empty placeholder | Resolved | — |

---

## Detailed Triage

### 1. Stage 3 Output Not Consumed by Stage 4

**Original:** Stage 3 (`AssociationStage`) produces `context.associated_bundles`, but Stage 4 (`TrackingStage`) does NOT consume them for its primary tracking logic. The Stage 3 output is a data product for future backends and observers only, adding computational overhead without affecting Stage 4 output in the default configuration.

**Status:** Open

**Evidence:**
- `src/aquapose/core/tracking/stage.py:86-138` — `TrackingStage.run()` reads `context.detections` (Stage 1 output) and passes `associated_bundles` only as a positional argument to `track_frame()`, which the HungarianBackend ignores.
- `src/aquapose/core/tracking/stage.py:117-119` — `bundles_per_frame` is constructed from `context.associated_bundles` only to satisfy `zip()` and pass to `track_frame()`; it does not influence tracking output.
- Module docstring (`stage.py:10-16`) correctly documents this as "documented design debt."

**Notes:** No change since Phase 15. The debt is well-documented and non-breaking. A future "clean" tracking backend consuming `associated_bundles` directly would eliminate the redundant association pass in `FishTracker.update()`.

---

### 2. Hungarian Backend Reads Stage 1, Not Stage 3

**Original:** `HungarianBackend.track_frame()` receives both `bundles` (Stage 3 output) and `detections_per_camera` (Stage 1 output) but uses only `detections_per_camera` for the actual tracking via `FishTracker.update()`.

**Status:** Open

**Evidence:**
- `src/aquapose/core/tracking/stage.py:124-129` — `self._backend.track_frame()` is called with both `bundles=frame_bundles` and `detections_per_camera=frame_dets`, but the HungarianBackend (wrapping `FishTracker`) uses only `detections_per_camera`.
- `src/aquapose/core/tracking/stage.py:50-54` — `TrackingStage` class docstring: "run() reads context.detections (Stage 1 raw output), not context.associated_bundles (Stage 3 output). The Hungarian backend re-derives cross-camera association internally to preserve v1.0 equivalence."
- This is the same root cause as Item 1 — both items stem from `FishTracker` performing association and tracking in a single monolithic call.

**Notes:** Both Items 1 and 2 are manifestations of the same design debt and should be remediated together. The documentation is accurate. No regression risk — the current behavior is numerically equivalent to v1.0.

---

### 3. MidlineSet Assembly from Decoupled Stages

**Original:** In v1.0, `MidlineExtractor` had direct access to both tracks and raw masks. In the 5-stage model, `ReconstructionStage` must assemble a `MidlineSet` from two separate upstream outputs: `context.tracks` (Stage 4) and `context.annotated_detections` (Stage 2). The assembly logic bridges the Stage 2/4 decoupling.

**Status:** Accepted

**Evidence:**
- `src/aquapose/core/reconstruction/stage.py:166-225` — `_assemble_midline_set()` is a well-defined private method with a clear docstring, bounds checking (`det_idx >= len(cam_annots)`), and debug logging.
- `src/aquapose/core/reconstruction/stage.py:44-54` — `ReconstructionStage` class docstring explicitly describes the assembly logic as the intended bridge pattern between Stage 2 and Stage 4.
- The assembly handles edge cases: coasting fish (empty `camera_detections`) are skipped, out-of-range `det_idx` are logged and skipped, cameras with no annotated detections are skipped.

**Notes:** This is now the intended architecture, not a quirk. The decoupled model is cleaner than v1.0's monolithic `MidlineExtractor`. The bridge pattern is well-documented and correct. No remediation needed.

---

### 4. Hardcoded Thresholds Extracted to Config

**Original:** RANSAC inlier threshold, epipolar snap threshold, and max depth bound were hardcoded in v1.0. The ledger recorded their extraction to `ReconstructionConfig` as a positive change.

**Status:** Resolved

**Evidence:**
- `src/aquapose/engine/config.py:142-163` — `ReconstructionConfig` has all three fields:
  - `inlier_threshold: float = 50.0` (matches v1.0 hardcoded value)
  - `snap_threshold: float = 20.0` (matches v1.0 hardcoded value)
  - `max_depth: float | None = None` (disabled by default, matching v1.0 behavior)
- `src/aquapose/engine/pipeline.py:310-316` — `build_stages()` passes all three fields from config to `ReconstructionStage()`.
- `src/aquapose/core/reconstruction/stage.py:79-81` — `ReconstructionStage.__init__()` accepts all three as constructor arguments.

**Notes:** Fully resolved. Thresholds are configurable via YAML and CLI overrides. No action required.

---

### 5. Camera Skip Hardcoded to "e3v8250"

**Original:** All 5 stages default to `skip_camera_id="e3v8250"`. This is not configurable via `PipelineConfig` at the top level — each stage has its own default. A future `skip_camera_id` top-level config field would allow centralized override.

**Status:** Open

**Evidence:**
- Hardcoded `_DEFAULT_SKIP_CAMERA_ID = "e3v8250"` found in 10 locations across `src/`:
  - `src/aquapose/core/detection/stage.py:29`
  - `src/aquapose/core/midline/stage.py:28`
  - `src/aquapose/core/association/stage.py:28`
  - `src/aquapose/core/association/backends/ransac_centroid.py:20`
  - `src/aquapose/core/tracking/stage.py:36`
  - `src/aquapose/core/tracking/backends/hungarian.py:82`
  - `src/aquapose/core/reconstruction/stage.py:31`
  - `src/aquapose/core/reconstruction/backends/curve_optimizer.py:22`
  - `src/aquapose/core/reconstruction/backends/triangulation.py:26`
  - `src/aquapose/pipeline/orchestrator.py:22` (legacy orchestrator)
- `src/aquapose/engine/config.py:170-203` — `PipelineConfig` has no `skip_camera_id` field.
- `src/aquapose/engine/pipeline.py:270-324` — `build_stages()` never passes `skip_camera_id` to any stage constructor, relying entirely on each stage's `_DEFAULT_SKIP_CAMERA_ID` constant.

**Notes:** Cannot override the skip camera via YAML or CLI without code changes. Adding `skip_camera_id: str = "e3v8250"` to `PipelineConfig` and threading it through `build_stages()` would centralize the override. This is a configuration completeness gap — not a correctness bug, since the default is appropriate for the hardware. Recommended for Phase 20 as a low-effort improvement.

---

### 6. CurveOptimizer Statefulness Preserved

**Original:** `CurveOptimizerBackend` maintains a single `CurveOptimizer` instance across all frames for warm-starting. This means `ReconstructionStage` with the `curve_optimizer` backend is NOT idempotent across pipeline runs on the same stage instance.

**Status:** Accepted

**Evidence:**
- `src/aquapose/core/reconstruction/backends/curve_optimizer.py:71-72` — `self._optimizer = CurveOptimizer(config=optimizer_config)` is constructed once in `__init__()` and persists.
- `src/aquapose/core/reconstruction/backends/curve_optimizer.py:25-34` — Class docstring explicitly states: "Maintains a stateful `CurveOptimizer` for warm-starting across frames (persists across `reconstruct_frame()` calls)."
- `src/aquapose/core/reconstruction/backends/curve_optimizer.py:77-99` — `reconstruct_frame()` delegates to `self._optimizer.optimize_midlines()`, which warm-starts from the previous frame's solution automatically.

**Notes:** Statefulness is the correct design for performance — warm-starting dramatically improves convergence speed and reconstruction quality. The documentation is accurate. Users constructing `PosePipeline` for multiple sequential runs on the same data should construct a new `ReconstructionStage` (and therefore new backend) per run. This is noted in the Phase 15-05 decisions in STATE.md. No remediation needed.

---

### 7. AssociationConfig Was Empty Placeholder in Phase 14.1

**Original:** `AssociationConfig` was initially defined as an empty frozen dataclass placeholder in Phase 14.1. In Phase 15-03, it was extended with `expected_count`, `min_cameras`, and `reprojection_threshold`.

**Status:** Resolved

**Evidence:**
- `src/aquapose/engine/config.py:78-94` — `AssociationConfig` has three real fields:
  - `expected_count: int = 9`
  - `min_cameras: int = 3`
  - `reprojection_threshold: float = 15.0`
- `src/aquapose/engine/pipeline.py:291-295` — `build_stages()` passes all three fields from config to `AssociationStage()`.
- `src/aquapose/core/association/stage.py:58-75` — `AssociationStage.__init__()` accepts all three as constructor parameters and forwards them to the backend.

**Notes:** Fully resolved. `AssociationConfig` is no longer a placeholder. No action required.

---

## Open Items for 19-AUDIT.md

The following items remain open and should be incorporated into the Phase 19 Audit Report under the appropriate findings section.

---

### OPEN-1: Stage 3 Output Not Consumed by Stage 4

**Severity:** Warning

**Summary:** `AssociationStage` (Stage 3) computes `associated_bundles` every frame, but `TrackingStage` (Stage 4) ignores them and re-derives cross-camera association internally via `FishTracker.update()`. Stage 3 adds computational overhead (a full RANSAC centroid pass per frame) without affecting Stage 4 output in the default configuration.

**Files:** `src/aquapose/core/tracking/stage.py:86-138`

**Remediation:** Implement a "clean" tracking backend (e.g., `AssociationAwareBackend`) that consumes `context.associated_bundles` directly, bypassing the redundant internal association in `FishTracker`. This would require restructuring `FishTracker` internals and is a larger undertaking. Until then, the computational waste is bounded and the documentation is accurate.

---

### OPEN-2: Hungarian Backend Ignores Stage 3 Bundles

**Severity:** Warning

**Summary:** `HungarianBackend.track_frame()` receives both `bundles` (Stage 3) and `detections_per_camera` (Stage 1) but only uses `detections_per_camera`. The `bundles` parameter is a no-op placeholder for future extensibility.

**Files:** `src/aquapose/core/tracking/stage.py:124-129`, `src/aquapose/core/tracking/backends/hungarian.py`

**Remediation:** Same as OPEN-1 — these are two aspects of the same design debt. Resolve both together when implementing a bundles-aware tracking backend.

---

### OPEN-3: Camera Skip ID Not Configurable via PipelineConfig

**Severity:** Warning

**Summary:** `skip_camera_id="e3v8250"` is hardcoded as a default in 10 files across the `src/` tree. `PipelineConfig` has no `skip_camera_id` field, so users cannot override the camera skip via YAML config or CLI without modifying source code.

**Files:**
- `src/aquapose/engine/config.py` (missing field)
- `src/aquapose/engine/pipeline.py` (build_stages does not pass skip_camera_id)
- `src/aquapose/core/detection/stage.py:29`, `src/aquapose/core/midline/stage.py:28`, `src/aquapose/core/association/stage.py:28`, `src/aquapose/core/association/backends/ransac_centroid.py:20`, `src/aquapose/core/tracking/stage.py:36`, `src/aquapose/core/tracking/backends/hungarian.py:82`, `src/aquapose/core/reconstruction/stage.py:31`, `src/aquapose/core/reconstruction/backends/curve_optimizer.py:22`, `src/aquapose/core/reconstruction/backends/triangulation.py:26`

**Remediation:** Add `skip_camera_id: str = "e3v8250"` to `PipelineConfig` (and to the template YAML). Update `build_stages()` to pass `config.skip_camera_id` to each stage constructor. This is a small, low-risk change that completes the configuration story for multi-tank deployments where the skip camera may differ.

---

*End of triage. 3 Open items (all Warning severity) ready for incorporation into 19-AUDIT.md.*
*4 items resolved or accepted — no action required.*
