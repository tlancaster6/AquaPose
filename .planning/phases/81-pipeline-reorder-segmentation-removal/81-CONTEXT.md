# Phase 81: Pipeline Reorder & Segmentation Removal - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Reorder the pipeline so pose estimation runs immediately after detection (before tracking), and fully remove the segmentation midline backend from the codebase. The pipeline changes from Detection → Tracking → Association → Midline → Reconstruction to Detection → Pose → Tracking → Association → Reconstruction. No new capabilities — this is structural surgery.

</domain>

<decisions>
## Implementation Decisions

### Pose-on-all-detections
- Pose runs on ALL detections that pass the OBB confidence threshold — no secondary confidence gate
- False positives are rare enough with the production OBB model that filtering isn't worth the complexity
- Tracking and association downstream can ignore low-quality pose results if needed

### Data flow change
- Pose stage enriches Detection objects in-place within `context.detections` — no new PipelineContext field
- `annotated_detections` field is removed from PipelineContext entirely (clean break, no dead fields)
- All code referencing `annotated_detections` must be updated to read from `detections` directly

### Tracking passthrough
- Tracking (OC-SORT) remains OBB-only — pose data is present on detections but tracking ignores it
- Wiring keypoints into tracking cost is Phase 83's job
- No speculative heading computation on Detection — leave that for Phase 83

### Stage rename: MidlineStage → PoseStage
- Class renamed from `MidlineStage` to `PoseStage`
- Directory renamed from `core/midline/` to `core/pose/`
- Config keys renamed from `config.midline.*` to `config.pose.*`
- Types renamed: `Midline2D` → `PoseResult2D`, `AnnotatedDetection` reviewed for removal or rename
- Full rename across imports, tests, observers, config, and docstrings

### Orientation resolution: removed entirely
- `orientation.py` deleted in full — cross-camera vote, velocity prior, temporal prior, all of it
- Pose keypoints have fixed anatomical meaning (nose=0, tail=5) — no head-tail ambiguity to resolve
- No heading direction added to Detection in this phase (Phase 83 computes it when needed)

### Segmentation removal: full cleanup
- Delete `backends/segmentation.py`, `orientation.py`
- Audit `midline.py` — keep utilities the pose backend actually uses (spline fitting, arc-length sampling), delete segmentation-only code (skeletonization, mask-to-midline)
- Move retained utilities into `core/pose/` directory
- Remove all references in config, backend registry, imports, and docstrings across 14+ files
- Training code for YOLO-seg stays (general training wrapper, not segmentation-backend-specific)
- Backend selector kept in config with one option (`pose_estimation`) — maintains swappable architecture pattern

### Synthetic mode: deferred
- Synthetic mode pipeline update is deferred — mark as broken/TODO, do not update stage ordering for synthetic mode in this phase
- Reduces blast radius; synthetic mode can be fixed in a later cleanup phase

</decisions>

<specifics>
## Specific Ideas

- The rename from "midline" to "pose" throughout the codebase is a clean break from the segmentation era — the system now does keypoint pose estimation, not midline extraction from masks
- Existing project config files (config.yaml) will break due to `midline` → `pose` key rename — acceptable since those are per-run artifacts

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `core/midline/backends/pose_estimation.py`: Stays as-is, moves to `core/pose/backends/`
- `core/midline/crop.py`: OBB-aligned crop utilities — stays, moves to `core/pose/`
- `core/midline/midline.py`: Audit for pose-used utilities (spline fitting, arc-length sampling) — keep those, delete rest
- `core/midline/types.py`: `AnnotatedDetection` and `Midline2D` — rename or refactor

### Established Patterns
- `PipelineContext` is a mutable dataclass accumulator — stages write fields, downstream stages read them
- `_STAGE_OUTPUT_FIELDS` in `pipeline.py` maps stage class names to output fields — must be updated for PoseStage
- `build_stages()` in `pipeline.py` constructs the stage list — reorder here
- `TrackingStage` has special `(context, carry)` dispatch via `isinstance` check in `pipeline.py`
- Backend selector pattern: config key → `importlib.import_module()` in factory

### Integration Points
- `pipeline.py:build_stages()` — reorder stage construction and rename midline_stage → pose_stage
- `pipeline.py:_STAGE_OUTPUT_FIELDS` — update for PoseStage (enriches detections, no separate output field)
- `engine/config.py` — rename MidlineConfig section to PoseConfig
- `core/__init__.py` — update public exports (MidlineStage → PoseStage)
- `engine/observers/` — diagnostic observer references to midline/annotated_detections
- `evaluation/` — any evaluation code reading annotated_detections
- `cli.py` — config key references

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 81-pipeline-reorder-segmentation-removal*
*Context gathered: 2026-03-10*
