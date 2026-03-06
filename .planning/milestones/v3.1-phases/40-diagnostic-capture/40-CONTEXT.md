# Phase 40: Diagnostic Capture - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Expand the DiagnosticObserver to capture and serialize MidlineSet data from pipeline runs as loadable fixtures. The fixture file contains the assembled per-frame MidlineSets that the reconstruction backend receives. Loading a fixture produces structured data that the evaluation harness (Phase 41) can feed directly to a reconstruction backend without running the pipeline.

</domain>

<decisions>
## Implementation Decisions

### Serialization Format
- NPZ via `np.savez_compressed` — consistent with existing `centroid_correspondences.npz` export pattern
- Single file per run (`midline_fixtures.npz`), not per-frame files
- Include basic metadata: frame_count, camera_ids, timestamp
- Claude designs the NPZ key naming convention to encode the fish_id/camera_id/frame_index hierarchy into flat string keys, parseable back into structured data at load time

### What Data to Capture
- Assembled MidlineSets — the per-frame `dict[fish_id, dict[camera_id, Midline2D]]` that `reconstruct_frame()` receives
- Decoupled from association logic — do not serialize tracklet_groups or annotated_detections
- All frames with midline data included (no min_cameras filtering); the eval harness does its own frame selection

### Capture Trigger
- Auto-export whenever DiagnosticObserver has `output_dir` configured — same trigger as centroid correspondences export
- No new config flags needed
- Output lands in the run output directory alongside other diagnostic artifacts: `{output_dir}/midline_fixtures.npz`

### Loader API
- Returns a typed dataclass (e.g. `MidlineFixture`) with fields for frames (list of MidlineSet), frame_count, camera_ids, and metadata
- Loader function lives in `aquapose.io` — importable without engine/pipeline dependencies
- Basic validation on load: check metadata presence, array shapes, version compatibility; raise clear errors on mismatch

### Claude's Discretion
- NPZ key naming convention design
- MidlineFixture dataclass field names and exact structure
- MidlineSet assembly logic within DiagnosticObserver (reassembly from snapshots vs. other approach)
- Validation specifics (what counts as shape mismatch, version compat rules)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `DiagnosticObserver` (`engine/diagnostic_observer.py`): Already captures StageSnapshot references and auto-exports centroid_correspondences.npz on PipelineComplete — extend with midline fixture export
- `StageSnapshot`: Holds references to `annotated_detections` and `tracklet_groups` — the raw ingredients for MidlineSet assembly
- `centroid_correspondences.npz` export: Proven pattern for NPZ serialization in the observer
- `MidlineSet` type alias (`core/types/reconstruction.py`): `dict[int, dict[str, Midline2D]]`
- `Midline2D` (`core/types/midline.py`): Has numpy arrays (points, half_widths, point_confidence) plus scalar fields

### Established Patterns
- Observer auto-export on PipelineComplete with output_dir guard
- NPZ with `np.savez_compressed` for array-heavy data
- Artifact output to run directory (`{output_dir}/`)
- Layer discipline: core/ has no side effects, io/ handles persistence, engine/ orchestrates

### Integration Points
- `DiagnosticObserver._on_pipeline_complete()` — add midline fixture export alongside centroid export
- `ReconstructionStage._run_with_tracklet_groups()` — reference for MidlineSet assembly logic (observer must replicate this from snapshot data)
- `aquapose.io` package — new fixture loading module
- `aquapose.core.types` — possible new MidlineFixture dataclass (or in io/)

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. The key constraint is consistency with the existing centroid export pattern in DiagnosticObserver.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 40-diagnostic-capture*
*Context gathered: 2026-03-02*
