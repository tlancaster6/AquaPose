# Phase 41: Evaluation Harness - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Build an offline evaluation framework that loads serialized MidlineSet fixtures + calibration data and computes Tier 1 (reprojection error) and Tier 2 (leave-one-out camera stability) metrics without running the full pipeline. Requirements: EVAL-01 through EVAL-05.

First task extends the Phase 40 fixture format to bundle calibration parameters, since the harness needs self-contained fixtures.

</domain>

<decisions>
## Implementation Decisions

### Frame Selection
- Uniform temporal sampling via np.linspace (deterministic, no randomness)
- Same fixture always evaluates the same frames — enables regression comparison
- Default 15 frames, configurable via parameter
- If fixture has fewer frames than requested, evaluate all available with a warning (no error)
- No quality filtering — partial camera coverage per fish is expected and informative

### Metric Computation
- Tier 1 reprojection error: call existing reconstruction backend's reconstruct_frame() directly, then reproject via RefractiveProjectionModel. Measures the actual backend being evaluated.
- Tier 2 leave-one-out: drop each observing camera in turn (not all 12), re-triangulate, measure max control-point displacement in world metres (Euclidean 3D distance)
- Leave-one-out runs that fail reconstruction (too few cameras after dropout) reported as N/A — useful signal about camera redundancy

### Output Format
- Human-readable summary: per-camera breakdown as primary axis, per-fish within each camera. Overall aggregates at bottom.
- Machine-diffable regression data: JSON format, aggregated metrics only (no per-frame detail). Per-fish and per-camera aggregates (mean, max).
- Results saved next to fixture file (e.g., fixture_dir/eval_results.json)

### Harness Architecture
- New `src/aquapose/evaluation/` package — dedicated module for harness, metrics, frame selection
- Python API only (no CLI command this phase)
- Calibration parameters bundled in fixture (intrinsics + extrinsics per camera) — reconstruction backends use RefractiveProjectionModel on-the-fly, no LUTs needed
- Phase 41's first task: extend Phase 40 fixture format to include calibration data

### Claude's Discretion
- Internal module structure within evaluation/ package
- Exact summary table formatting
- JSON schema details for regression data
- Test structure for the harness itself

</decisions>

<specifics>
## Specific Ideas

- Reconstruction backends already use RefractiveProjectionModel directly (confirmed via code inspection) — no LUT dependency, so bundling calibration params is sufficient for fully self-contained fixtures
- Fish tend to group together in the tank, so many timepoints have only a subset of cameras observing each fish — this is expected behavior, not an error condition
- The 13-camera rig provides sufficient redundancy that leave-one-out dropout should succeed for most fish/frame combinations

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `MidlineSet = dict[int, dict[str, Midline2D]]` — already defined in `core/types/reconstruction.py`
- `Midline3D` dataclass — has `mean_residual`, `max_residual`, `per_camera_residuals`, `is_low_confidence` fields
- `RefractiveProjectionModel` in `calibration/projection.py` — handles pixel↔ray conversion and 3D→2D reprojection
- Reconstruction backends in `core/reconstruction/backends/` — `reconstruct_frame()` API takes `frame_idx` + `midline_set`

### Established Patterns
- Backend instantiation: loads calibration JSON → builds `dict[str, RefractiveProjectionModel]` per camera
- B-spline fitting: 7 control points, cubic, clamped knot vector via `make_lsq_spline`
- Triangulation: confidence-weighted DLT with outlier rejection via `_weighted_triangulate_rays()`

### Integration Points
- Phase 40 fixture serialization — Phase 41 extends format to include calibration params
- Reconstruction backend's `reconstruct_frame()` — called directly by the harness
- `DiagnosticObserver` captures `StageSnapshot` data — fixtures originate from diagnostic mode runs

</code_context>

<deferred>
## Deferred Ideas

- CLI command (`aquapose evaluate`) — add when Python API is stable
- Tier 3 synthetic ground-truth evaluation — tracked as EVAL-T3-01/T3-02 in requirements
- Per-frame detailed output — may add later if aggregated metrics prove insufficient for debugging

</deferred>

---

*Phase: 41-evaluation-harness*
*Context gathered: 2026-03-02*
