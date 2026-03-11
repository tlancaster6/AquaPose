# Phase 84: Integration & Evaluation - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the Phase 83 custom keypoint tracker into the pipeline as a selectable backend and evaluate it against the Phase 80 OC-SORT baselines (27 tracks, 93.1% coverage, 0 gaps on e3v83eb frames 3300-4500). Conditionally implement BYTE-style secondary pass if coverage drops. Produce a comparison document with metrics and interpretation.

</domain>

<decisions>
## Implementation Decisions

### Evaluation methodology
- Same script, same clip as Phase 80 baseline (camera e3v83eb, frames 3300-4500, 1200 frames)
- Same metrics: track count, duration distribution, coast frequency, detection coverage, fragmentation (gaps, births, deaths)
- No additional metrics beyond Phase 80 set
- Side-by-side markdown comparison table with brief interpretation paragraph
- Standalone script in `scripts/` (not integrated into eval CLI)

### Pass/fail criteria
- Pass threshold: any improvement in track count over the 27-track OC-SORT baseline
- Detection coverage is a soft target — minor regression below 93.1% is acceptable if track count improves
- 1-2 parameter tuning rounds if initial results don't show improvement (adjust max_age, OKS lambda, n_init)
- If tracker fails to improve after tuning: document findings and ship anyway — keep custom tracker as a configurable backend alongside OC-SORT

### BYTE-style secondary pass (TRACK-10)
- Trigger: coverage drops below ~90% with the custom tracker, suggesting valid low-confidence detections are being missed
- Implementation scope: minimal — a second OKS matching pass that tries unmatched low-confidence detections against unmatched tracks, using same cost function with lower confidence threshold
- Confidence threshold split: Claude's discretion based on Phase 78 confidence analysis
- If implemented: re-run evaluation and add a third column to the comparison table (OC-SORT vs custom vs custom+BYTE)

### Pipeline wiring
- Register custom tracker as `tracker_kind: keypoint_bidi` alongside existing `ocsort`
- OC-SORT remains available as fallback — config selects which tracker runs
- Default `tracker_kind` switches to `keypoint_bidi` in this phase
- Custom tracker parameters added flat in TrackingConfig with sensible defaults (consistent with existing config pattern)
- Evaluation scope: tracking metrics only (Stage 2 output). Full E2E pipeline validation is Phase 85's job.

### Claude's Discretion
- BYTE-style confidence threshold values
- Exact parameter adjustment strategy during tuning rounds
- Comparison document structure and interpretation depth
- How to handle the evaluation script's integration with existing `evaluate_tracking()` / `evaluate_fragmentation()` utilities

</decisions>

<specifics>
## Specific Ideas

- Phase 80 baseline is the single source of truth for comparison: 27 tracks, 93.1% detection coverage, 0 gaps, 1.000 continuity ratio, 18 births / 17 deaths
- Phase 83 deferred TRACK-10 explicitly to this phase — the coverage-based trigger is the decision mechanism
- The tracker should ship regardless of performance — the investigation has value even if results are mixed

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `evaluate_tracking()` and `evaluate_fragmentation()` in `evaluation/stages/`: Ready-made metrics matching Phase 80 format
- Phase 80 baseline script pattern: Standalone script that runs OcSortTracker on a clip and prints metrics
- `OcSortTracker` wrapper interface (`update`, `get_tracklets`, `get_state`, `from_state`): Custom tracker must match this interface
- `TrackingStage` (`core/tracking/stage.py`): Per-camera dispatch, backend selection via `tracker_kind` — extend for `keypoint_bidi`
- `_STAGE_OUTPUT_FIELDS` in `engine/pipeline.py`: No changes needed — TrackingStage output fields unchanged

### Established Patterns
- Backend selection: `tracker_kind` config field selects implementation (same pattern as `detector_kind`, `midline_backend`)
- Config: Flat fields in frozen dataclass with YAML defaults
- Stage protocol: `run(context, carry) -> (context, carry)` for tracking (special-cased in pipeline.py)

### Integration Points
- `TrackingConfig` in `engine/config.py`: Add `tracker_kind` field (default `keypoint_bidi`), OKS/KF/bidi params with defaults
- `TrackingStage.__init__`: Dispatch to OcSortTracker or KeypointBidiTracker based on config
- `Detection.keypoints` and `Detection.keypoint_conf`: Available from PoseStage (Phase 81 reorder)
- `Tracklet2D.centroids`: Must be populated identically by custom tracker (keypoint-derived per Phase 82)

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 84-integration-evaluation*
*Context gathered: 2026-03-10*
