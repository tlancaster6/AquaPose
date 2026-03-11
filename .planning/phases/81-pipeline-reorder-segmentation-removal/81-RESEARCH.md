# Phase 81: Pipeline Reorder & Segmentation Removal - Research

**Researched:** 2026-03-10
**Domain:** Internal pipeline refactor — Python module surgery, no external libraries
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Pose-on-all-detections**
- Pose runs on ALL detections that pass the OBB confidence threshold — no secondary confidence gate
- False positives are rare enough with the production OBB model that filtering isn't worth the complexity
- Tracking and association downstream can ignore low-quality pose results if needed

**Data flow change**
- Pose stage enriches Detection objects in-place within `context.detections` — no new PipelineContext field
- `annotated_detections` field is removed from PipelineContext entirely (clean break, no dead fields)
- All code referencing `annotated_detections` must be updated to read from `detections` directly

**Tracking passthrough**
- Tracking (OC-SORT) remains OBB-only — pose data is present on detections but tracking ignores it
- Wiring keypoints into tracking cost is Phase 83's job
- No speculative heading computation on Detection — leave that for Phase 83

**Stage rename: MidlineStage → PoseStage**
- Class renamed from `MidlineStage` to `PoseStage`
- Directory renamed from `core/midline/` to `core/pose/`
- Config keys renamed from `config.midline.*` to `config.pose.*`
- Types renamed: `Midline2D` → `PoseResult2D`, `AnnotatedDetection` reviewed for removal or rename
- Full rename across imports, tests, observers, config, and docstrings

**Orientation resolution: removed entirely**
- `orientation.py` deleted in full — cross-camera vote, velocity prior, temporal prior, all of it
- Pose keypoints have fixed anatomical meaning (nose=0, tail=5) — no head-tail ambiguity to resolve
- No heading direction added to Detection in this phase (Phase 83 computes it when needed)

**Segmentation removal: full cleanup**
- Delete `backends/segmentation.py`, `orientation.py`
- Audit `midline.py` — keep utilities the pose backend actually uses (spline fitting, arc-length sampling), delete segmentation-only code (skeletonization, mask-to-midline)
- Move retained utilities into `core/pose/` directory
- Remove all references in config, backend registry, imports, and docstrings across 14+ files
- Training code for YOLO-seg stays (general training wrapper, not segmentation-backend-specific)
- Backend selector kept in config with one option (`pose_estimation`) — maintains swappable architecture pattern

**Synthetic mode: deferred**
- Synthetic mode pipeline update is deferred — mark as broken/TODO, do not update stage ordering for synthetic mode in this phase
- Reduces blast radius; synthetic mode can be fixed in a later cleanup phase

### Claude's Discretion

None specified — all decisions were locked during discussion.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PIPE-01 | Pose estimation runs immediately after detection (Stage 2), before tracking | New pipeline order: Detection → PoseStage → Tracking → Association → Reconstruction. `build_stages()` and `_STAGE_OUTPUT_FIELDS` must be updated. PoseStage runs before TrackingStage and enriches `context.detections` in place. |
| PIPE-02 | Segmentation midline backend removed (backends/segmentation.py, skeletonization code, orientation resolution) | Delete `core/midline/backends/segmentation.py`, `core/midline/orientation.py`. Audit and strip `core/midline/midline.py` of segmentation-only functions. Remove `SegmentationBackend` from registry and `__init__.py`. |
| PIPE-03 | PipelineContext and stage interfaces updated for new stage ordering | Remove `annotated_detections` field from `PipelineContext`. `PoseStage.run()` enriches `context.detections` in place and returns context. `ReconstructionStage._run_with_tracklet_groups()` reads `.midline` from `context.detections[frame][cam]` entries. |
</phase_requirements>

## Summary

Phase 81 is pure structural surgery: reorder the pipeline so pose estimation runs before tracking, and remove the segmentation backend entirely. No new capabilities are added. The work falls into five parallel concerns: (1) rename the `midline` module to `pose`, (2) move `PoseStage` to position 2 in `build_stages()`, (3) strip `annotated_detections` from `PipelineContext` and update all readers, (4) delete `segmentation.py`, `orientation.py`, and the skeletonization code from `midline.py`, and (5) update tests that hardcode old stage ordering or import deleted modules.

The codebase is well-structured for this surgery. The `PipelineContext` is a plain dataclass — adding and removing fields is straightforward. The `build_stages()` factory is the single canonical wiring point for stage order. The `_STAGE_OUTPUT_FIELDS` dict controls skip-detection for warm-start runs and must be updated. `ReconstructionStage._run_with_tracklet_groups()` currently reads from `context.annotated_detections`; it must instead read the `.midline` attribute directly from `context.detections` entries after PoseStage has enriched them.

The key data-flow impact: PoseStage enriches Detection objects in-place (`.midline` attribute), so `context.detections` remains the single truth for all downstream stages. The `annotated_detections` field and `AnnotatedDetection` wrapper type are eliminated. The `Midline2D` type itself stays (used by reconstruction and I/O); only the wrapping layer is removed.

**Primary recommendation:** Execute as three sequential waves — (Wave 1) rename module + PoseStage interface; (Wave 2) pipeline reorder + PipelineContext cleanup; (Wave 3) segmentation/orientation deletion + test updates.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib | 3.11+ | `importlib`, `dataclasses`, `pathlib` | Already in use; no new deps |
| Ruff | (project-pinned) | Lint/format after refactor | Project formatter |
| basedpyright | (project-pinned) | Typecheck after field removals | Project typechecker |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | (project-pinned) | `hatch run test` for validation | After each wave |
| unittest.mock | stdlib | Mock stage constructors in build_stages tests | Already used in test_build_stages.py |

### Alternatives Considered
None — this is an internal refactor with no external library choices.

**Installation:** No new packages needed.

## Architecture Patterns

### Recommended Project Structure (after rename)

```
src/aquapose/core/pose/          ← renamed from core/midline/
├── __init__.py                  ← exports PoseStage, PoseResult2D
├── backends/
│   ├── __init__.py              ← get_backend() with only "pose_estimation"
│   └── pose_estimation.py      ← unchanged, moved here
├── crop.py                     ← unchanged, moved here
├── pose.py                     ← renamed from midline.py; seg functions deleted
├── stage.py                    ← renamed MidlineStage → PoseStage
└── types.py                    ← AnnotatedDetection removed OR renamed
```

Files deleted:
- `core/midline/backends/segmentation.py`
- `core/midline/orientation.py`
- `core/midline/` directory (replaced by `core/pose/`)

### Pattern 1: In-Place Detection Enrichment

**What:** PoseStage writes `.midline` directly onto Detection objects stored in `context.detections`, rather than producing a separate `annotated_detections` list.

**When to use:** This eliminates the parallel data structure problem. Downstream stages (Reconstruction) find midline data on the same Detection they already have.

**Implication for ReconstructionStage:** `_run_with_tracklet_groups()` currently calls `_find_matching_annotated(frame_ann[cam_id], centroid)` to find an `AnnotatedDetection` and then reads `.midline`. After Phase 81, it reads from `context.detections[frame_idx][cam_id]` directly, using the same centroid-matching logic against Detection objects.

### Pattern 2: `_STAGE_OUTPUT_FIELDS` Update

The pipeline skip-detection mechanism in `pipeline.py` uses `_STAGE_OUTPUT_FIELDS` to detect already-populated stages. After Phase 81:

- Remove `"MidlineStage": ("annotated_detections",)` entry
- Add `"PoseStage": ()` — PoseStage enriches in-place, no dedicated output field
  - **Critical:** An empty tuple means `already_populated = False` always (the `bool(output_fields)` check), so PoseStage will never be auto-skipped. This is intentional if `annotated_detections` is removed from context. Consider adding a sentinel field or accepting that PoseStage always re-runs in warm-start mode.
- Update `"SyntheticDataStage"` entry to remove `"annotated_detections"` from its output fields tuple (synthetic mode is deferred but the context field is gone)

### Pattern 3: `build_stages()` Reorder

Current (positions 0-4): Detection → Tracking → Association → Midline → Reconstruction

New (positions 0-4): Detection → PoseStage → Tracking → Association → Reconstruction

```python
stages = _truncate([
    detection_stage,
    pose_stage,       # NEW position 1 (was position 3)
    tracking_stage,   # NEW position 2 (was position 1)
    AssociationStage(config),  # NEW position 3 (was position 2)
    reconstruction_stage,
])
```

**Impact on `stop_after`:** The `STAGE_NAMES` dict must add `"pose"` and remove `"midline"`. The CLI `stop_after` option choices list (`["detection", "tracking", "association", "midline"]`) must be updated to `["detection", "pose", "tracking", "association"]`.

### Pattern 4: Config Rename

`MidlineConfig` → `PoseConfig`. The `PipelineConfig.midline` field → `PipelineConfig.pose`.

Fields to remove from `PoseConfig` (segmentation-only):
- `min_area` (segmentation mask area check)
- `speed_threshold`, `orientation_weight_geometric`, `orientation_weight_velocity`, `orientation_weight_temporal` (orientation resolution)
- `midline_batch_crops` → rename to `pose_batch_crops`

Fields to keep (pose_estimation backend uses them):
- `confidence_threshold` / `conf_threshold`
- `weights_path`
- `backend` (keep selector, single option `pose_estimation`)
- `n_keypoints`
- `keypoint_t_values`
- `keypoint_confidence_floor`
- `min_observed_keypoints`
- `detection_tolerance` (used by stage for tracklet→detection matching)

YAML backward compat: The existing `load_config()` already accepts `"midline"` as a key. After rename, it should accept `"pose"` and optionally retain `"midline"` as deprecated alias.

### Anti-Patterns to Avoid

- **Leaving `annotated_detections` as None-but-present:** The decision is a clean break — remove the field entirely, not just stop populating it. Leaving it as a dead field violates the phase goal and confuses future readers.
- **Skipping `_find_matching_annotated` helper in ReconstructionStage:** The centroid-matching logic is needed to find the right Detection in `context.detections[frame][cam]`, since multiple fish may be in the same camera frame. The helper just needs to be adapted to work with `Detection` objects that have a `.midline` attribute instead of `AnnotatedDetection` wrappers.
- **Forgetting `SyntheticDataStage`:** Synthetic mode is deferred, but `SyntheticDataStage` still populates `annotated_detections` in the current code. After removing the field from `PipelineContext`, `SyntheticDataStage.run()` will break at `context.annotated_detections = ...`. It must be marked with a `TODO(Phase 81 synthetic): broken` comment or guarded.
- **Renaming `Midline2D` to `PoseResult2D`:** The CONTEXT.md notes this as something to "review." `Midline2D` is used pervasively in reconstruction, I/O, and evaluation code. Renaming it would massively expand the blast radius. HIGH confidence recommendation: leave `Midline2D` named as-is (it's in `core/types/midline.py`, not `core/midline/`). Only rename the stage-level wrapper types.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Finding the matching Detection for a tracklet centroid | Custom distance search | Adapt existing `_find_matching_annotated()` / `_find_closest_detection_idx()` helpers | Already in `midline/stage.py` and `reconstruction/stage.py`; just update signature |
| Batch inference orchestration | New batching code | Reuse existing `predict_with_oom_retry()` and `BatchState` from `core/inference.py` | Already handles OOM retry and adaptive batch sizing |

**Key insight:** This phase is module renaming + deletion + wiring change. The actual inference logic (`pose_estimation.py`, `crop.py`) is untouched and simply moves directories.

## Common Pitfalls

### Pitfall 1: PoseStage Running Before Frame Source Is Available

**What goes wrong:** PoseStage (now position 1) needs a `frame_source` to read video frames, just like the old MidlineStage. But the frame source was created after detection stage in `build_stages()`. No change needed — the frame source is created once in `build_stages()` and passed to both `DetectionStage` and `PoseStage`.

**How to avoid:** Confirm `frame_source` is created before both stage constructors are called. The current `build_stages()` already does this correctly for `MidlineStage`; the wiring just moves up.

### Pitfall 2: ReconstructionStage Now Reads `context.detections` Instead of `context.annotated_detections`

**What goes wrong:** `_run_with_tracklet_groups()` and `_run_legacy()` both check `context.annotated_detections is None` and raise `ValueError`. After removing that field, `context.detections` becomes the source, and the guard condition must change.

**How to avoid:** In `_run_with_tracklet_groups()`, change `annotated = context.annotated_detections` to `annotated = context.detections`. The centroid-matching logic that finds `AnnotatedDetection` objects by centroid must now find `Detection` objects and read their `.midline` attribute. The `_find_matching_annotated()` helper in `reconstruction/stage.py` must be updated.

### Pitfall 3: `_STAGE_OUTPUT_FIELDS` Skip-Detection Breaks for PoseStage

**What goes wrong:** With `annotated_detections` removed, there is no context field that PoseStage "owns." The pipeline skip-detection code checks `all(getattr(context, f, None) is not None for f in output_fields)` — an empty tuple always returns True for `all()`, but `bool(())` is `False`, so `already_populated` would be `False` (never skipped). This is safe but means warm-start won't skip PoseStage.

**How to avoid:** Either accept this (PoseStage is fast compared to Detection), or add a boolean sentinel field like `pose_complete: bool = False` to PipelineContext. Decision should be documented — the CONTEXT.md decision to "enrich in-place" implies accepting that PoseStage can't be auto-skipped.

### Pitfall 4: DiagnosticObserver References `annotated_detections`

**What goes wrong:** `diagnostic_observer.py` builds a `StageSnapshot` dataclass with an `annotated_detections` field, assigned from `getattr(context, "annotated_detections", None)`. Removing the field from `PipelineContext` makes this silently return `None` forever (not a crash, but dead code).

**How to avoid:** Remove `annotated_detections` from `StageSnapshot` in `diagnostic_observer.py`. Check `_SNAPSHOT_FIELDS` list and `StageSnapshot` dataclass for references.

### Pitfall 5: `test_build_stages.py` Hardcodes Old Stage Ordering

**What goes wrong:** 11+ test functions in `test_build_stages.py` patch `"aquapose.core.MidlineStage"` and assert it occupies `stages[3]`. After the rename and reorder, `PoseStage` is at `stages[1]` and the patch target changes to `"aquapose.core.PoseStage"`.

**How to avoid:** Update all `test_build_stages.py` tests to use new stage class name and position. Similarly update `test_stages.py`, `test_diagnostic_observer.py`, `test_timing.py` (hardcodes `"MidlineStage"` as stage name string).

### Pitfall 6: `midline.py` Segmentation-Only Functions Used by Tests

**What goes wrong:** `tests/unit/test_midline.py` likely tests functions from `midline.py` including segmentation-specific ones (`_adaptive_smooth`, `_skeleton_and_widths`, `_longest_path_bfs`). Deleting those functions breaks the tests.

**How to avoid:** Before deleting, check which `midline.py` functions are used by `pose_estimation.py` vs only by `segmentation.py`. The pose backend imports: `extract_affine_crop`, `invert_affine_points` (from `crop.py`), `interp1d` (scipy). It does NOT use any function from `midline.py`. Only the segmentation backend imports from `midline.py`. This means all of `midline.py`'s exported functions can be deleted; check `tests/unit/test_midline.py` and `tests/unit/core/midline/test_direct_pose_backend.py` for any direct imports to update.

### Pitfall 7: `evaluation/tuning.py` Reads `annotated_detections` Extensively

**What goes wrong:** `tuning.py` has 15+ references to `ctx.annotated_detections` as the midline cache. After removing the field, this code breaks entirely.

**How to avoid:** After Phase 81, the "midline cache" becomes a snapshot of `context.detections` (after PoseStage has enriched it). Update `tuning.py` to read from `context.detections` instead. The `_build_midline_sets()` function that iterates `annotated_detections[frame_idx][cam_id]` and reads `.midline` will work identically with `detections[frame_idx][cam_id]` after the rename, since Detection objects now carry `.midline` directly.

### Pitfall 8: `SyntheticDataStage` Writes `annotated_detections`

**What goes wrong:** `SyntheticDataStage.run()` sets `context.annotated_detections = ...`. After removing that field, this assignment raises `AttributeError` (frozen context) or silently does nothing (regular dataclass). Synthetic mode will be broken.

**How to avoid:** Add a `TODO(Phase 81 synthetic deferred)` comment and either guard the assignment or convert it to a no-op. The phase decision is to mark synthetic mode as broken rather than fix it.

## Code Examples

### In-Place Detection Enrichment Pattern

The PoseStage will enrich Detection objects in-place. The `Detection` datatype (in `core/types/detection.py`) must have a `.midline` field added:

```python
# core/types/detection.py (add field)
@dataclass
class Detection:
    ...
    midline: "Midline2D | None" = None  # populated by PoseStage
```

PoseStage.run() writes to each Detection's `.midline`:

```python
# core/pose/stage.py (PoseStage.run sketch)
def run(self, context: PipelineContext) -> PipelineContext:
    detections = context.get("detections")
    camera_ids = context.get("camera_ids")
    # ... batch inference over all detections ...
    # For each detection, set det.midline = midline_or_None
    return context  # detections modified in-place
```

### Updated ReconstructionStage Midline Lookup

```python
# core/reconstruction/stage.py (after Phase 81)
# Was: annotated = context.annotated_detections
annotated = context.detections  # detections now carry .midline

# Was: ad = _find_matching_annotated(frame_ann[cam_id], centroid)
#       if ad is not None: midline = ad.midline
# Now: (same centroid-match helper, but against Detection objects)
det = _find_matching_detection(annotated[frame_idx][cam_id], centroid)
if det is not None and det.midline is not None:
    cam_midlines[cam_id] = det.midline
```

### Updated `_STAGE_OUTPUT_FIELDS`

```python
_STAGE_OUTPUT_FIELDS: dict[str, tuple[str, ...]] = {
    "DetectionStage": ("frame_count", "camera_ids", "detections"),
    "SyntheticDataStage": ("frame_count", "camera_ids", "detections"),  # annotated_detections removed
    "PoseStage": (),            # enriches in-place, no dedicated field
    "TrackingStage": ("tracks_2d",),
    "AssociationStage": ("tracklet_groups",),
    "ReconstructionStage": ("midlines_3d",),
}
```

### `cli.py` stop_after Update

```python
# cli.py
"stop_after",
type=click.Choice(
    ["detection", "pose", "tracking", "association"],  # "midline" removed, "pose" added
    case_sensitive=False
),
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Segmentation backend (YOLO-seg mask → skeleton → midline) | Pose estimation backend (YOLO-pose keypoints → spline → midline) | Phase 73+ | Segmentation path now dead code; this phase removes it |
| MidlineStage runs after Association (Stage 4) | PoseStage runs after Detection (Stage 2) | Phase 81 | Tracking can now use keypoints in future phases |
| `annotated_detections` separate list | `.midline` on Detection objects | Phase 81 | Simpler data model, no parallel structure |

**Deprecated/outdated:**
- `SegmentationBackend`: Replaced by `PoseEstimationBackend` in Phase 73+; now deleted
- `orientation.py`: Pose keypoints have fixed anatomical ordering; orientation resolution was only needed for symmetric midlines from skeletonization
- `MidlineExtractor` class in `midline.py`: Legacy stateful extractor from pre-stage architecture; used by `MidlineStage` only via `_skeleton_and_widths` etc. — all segmentation-specific; delete with the module

## Open Questions

1. **`Midline2D` rename to `PoseResult2D`**
   - What we know: CONTEXT.md says "reviewed for removal or rename" — ambiguous
   - What's unclear: Renaming `Midline2D` would touch reconstruction, I/O, evaluation — very wide blast radius
   - Recommendation: **Do not rename `Midline2D` in Phase 81.** It lives in `core/types/midline.py` (not `core/midline/`), is used by reconstruction and HDF5 I/O, and renaming it is Phase 85 territory (INTEG-03 code quality audit). Scope this out.

2. **`AnnotatedDetection` type: remove or keep as alias**
   - What we know: If `.midline` moves onto `Detection`, `AnnotatedDetection` becomes vestigial
   - What's unclear: Any external code (scripts, notebooks) may import it
   - Recommendation: Delete `AnnotatedDetection` from `core/pose/types.py` (the new location). If tests import it, update them to import `Detection` instead.

3. **PoseStage skip-detection in warm-start**
   - What we know: Empty `_STAGE_OUTPUT_FIELDS` for PoseStage means it never auto-skips
   - What's unclear: Is this acceptable for Phase 81 or does warm-start need to work?
   - Recommendation: Accept for Phase 81. The phase decision specifies no new PipelineContext fields. Document with a TODO for Phase 85 if needed.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| Quick run command | `hatch run test` |
| Full suite command | `hatch run test-all` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PIPE-01 | PoseStage is Stage 1 (after Detection) in `build_stages()` production mode | unit | `hatch run test tests/unit/engine/test_build_stages.py` | ✅ (needs update) |
| PIPE-01 | Pipeline executes Detection → Pose → Tracking → Association → Reconstruction without error | unit (smoke) | `hatch run test tests/unit/engine/test_pipeline.py` | ✅ (needs update) |
| PIPE-02 | `backends/segmentation.py` does not exist in codebase | unit (import check) | `hatch run test tests/unit/core/midline/` → delete + update | ✅ (delete test) |
| PIPE-02 | `orientation.py` does not exist in codebase | unit (import check) | `hatch run test tests/unit/core/midline/test_orientation.py` → delete | ✅ (delete test) |
| PIPE-02 | `get_backend()` raises on `"segmentation"` | unit | `hatch run test tests/unit/core/pose/test_pose_stage.py` | ❌ Wave 0 |
| PIPE-03 | `PipelineContext` has no `annotated_detections` field | unit | `hatch run test tests/unit/engine/test_stages.py` | ✅ (needs update) |
| PIPE-03 | `ReconstructionStage` reads `.midline` from `context.detections` entries | unit | `hatch run test tests/unit/core/reconstruction/test_reconstruction_stage.py` | ✅ (needs update) |
| PIPE-03 | `PoseStage` satisfies Stage Protocol | unit | `hatch run test tests/unit/core/pose/test_pose_stage.py` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `hatch run test` (excludes @slow)
- **Per wave merge:** `hatch run test` — full non-slow suite
- **Phase gate:** `hatch run check` (lint + typecheck) + `hatch run test` green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/unit/core/pose/__init__.py` — new test package for renamed module
- [ ] `tests/unit/core/pose/test_pose_stage.py` — covers PIPE-03 (PoseStage protocol + context enrichment), PIPE-02 (backend registry)
- [ ] Delete `tests/unit/core/midline/test_segmentation_backend.py` — covers deleted code
- [ ] Delete `tests/unit/core/midline/test_orientation.py` — covers deleted code
- [ ] `tests/unit/core/midline/test_midline_stage.py` → move to `tests/unit/core/pose/test_pose_stage.py` with updated assertions

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection — `src/aquapose/` (all files read above)
- `engine/pipeline.py` — `build_stages()`, `_STAGE_OUTPUT_FIELDS`, `PosePipeline.run()`
- `core/context.py` — `PipelineContext` dataclass, `_STAGE_OUTPUT_FIELDS` skip logic
- `core/midline/stage.py` — `MidlineStage` full implementation
- `core/midline/backends/segmentation.py` — all functions to delete
- `core/midline/orientation.py` — all functions to delete
- `core/midline/midline.py` — segmentation-only functions (delete), arc-length/spline helpers (keep)
- `core/reconstruction/stage.py` — reads `annotated_detections`; must be updated
- `engine/diagnostic_observer.py` — `StageSnapshot` with `annotated_detections` field
- `evaluation/tuning.py` — 15+ reads of `ctx.annotated_detections`
- `tests/unit/engine/test_build_stages.py` — 11+ tests hardcoding `MidlineStage` and stage positions

### Secondary (MEDIUM confidence)
- None needed — all findings from direct code inspection

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — internal refactor, no external libraries
- Architecture: HIGH — all patterns derived from direct code inspection
- Pitfalls: HIGH — each pitfall verified against actual code locations

**Research date:** 2026-03-10
**Valid until:** Until codebase changes (stable internal domain — no external API concerns)

---

## Complete Blast Radius Inventory

Files to **delete**:
- `src/aquapose/core/midline/backends/segmentation.py`
- `src/aquapose/core/midline/orientation.py`
- `src/aquapose/core/midline/midline.py` (or strip to just arc-length/spline utils if any survive)
- `tests/unit/core/midline/test_segmentation_backend.py`
- `tests/unit/core/midline/test_orientation.py`
- `tests/unit/test_midline.py` (if it tests only segmentation functions)

Files to **rename** (move + update imports inside):
- `src/aquapose/core/midline/` → `src/aquapose/core/pose/`
- `src/aquapose/core/midline/stage.py` → `src/aquapose/core/pose/stage.py` (class: `MidlineStage` → `PoseStage`)
- `src/aquapose/core/midline/backends/pose_estimation.py` → `src/aquapose/core/pose/backends/pose_estimation.py`
- `src/aquapose/core/midline/crop.py` → `src/aquapose/core/pose/crop.py`
- `src/aquapose/core/midline/types.py` → `src/aquapose/core/pose/types.py` (delete `AnnotatedDetection`)
- `tests/unit/core/midline/` → `tests/unit/core/pose/`

Files to **update**:
| File | Change |
|------|--------|
| `src/aquapose/core/__init__.py` | `MidlineStage` → `PoseStage`, update import path |
| `src/aquapose/core/context.py` | Remove `annotated_detections` field from `PipelineContext` |
| `src/aquapose/core/types/detection.py` | Add `midline: Midline2D | None = None` field |
| `src/aquapose/engine/pipeline.py` | `build_stages()` reorder; `_STAGE_OUTPUT_FIELDS` update; `_check_model_weights()` use `config.pose.*`; `STAGE_NAMES` dict update |
| `src/aquapose/engine/config.py` | `MidlineConfig` → `PoseConfig`; remove seg-only fields; `PipelineConfig.midline` → `PipelineConfig.pose`; YAML loading key `"midline"` → `"pose"` |
| `src/aquapose/engine/diagnostic_observer.py` | Remove `annotated_detections` from `StageSnapshot` and `_SNAPSHOT_FIELDS` |
| `src/aquapose/core/reconstruction/stage.py` | Read `.midline` from `context.detections` instead of `context.annotated_detections`; update guard conditions |
| `src/aquapose/cli.py` | `stop_after` choices: `"midline"` → `"pose"`; config init template: `"midline"` → `"pose"` |
| `src/aquapose/evaluation/tuning.py` | Replace all `ctx.annotated_detections` with `ctx.detections` |
| `src/aquapose/core/synthetic/stage.py` | Mark `context.annotated_detections = ...` as broken TODO |
| `tests/unit/engine/test_build_stages.py` | Update all `MidlineStage` patches → `PoseStage`; fix stage position assertions |
| `tests/unit/engine/test_stages.py` | Remove `annotated_detections` assertion |
| `tests/unit/engine/test_diagnostic_observer.py` | Update `MidlineStage` stage name, remove `annotated_detections` snapshot field |
| `tests/unit/engine/test_timing.py` | Update `"MidlineStage"` string → `"PoseStage"` |
| `tests/unit/engine/test_config.py` | Update `config.midline.*` → `config.pose.*` assertions |
| `tests/unit/core/midline/test_midline_stage.py` | Move to `test_pose_stage.py`, update class/field names |
| `tests/unit/core/midline/test_pose_estimation_backend.py` | Move, update import paths |
| `tests/unit/core/test_synthetic.py` | Remove `annotated_detections` assertions or update to use `detections` |
| `tests/unit/core/reconstruction/test_reconstruction_stage.py` | Update to use `context.detections` with `.midline` |
| `tests/unit/test_calibrate_keypoints.py` | Update `"midline"` config key → `"pose"` |
| `src/aquapose/evaluation/stages/midline.py` | No rename needed (pure function, no stage class reference) |
