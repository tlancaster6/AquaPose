# Phase 80: Baseline Metrics - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish quantitative OC-SORT 2D tracking baselines on a single camera (e3v83eb, frames 3300-4500) so the custom tracker built in Phases 83-84 has numbers to beat. This is a single-camera 2D tracking evaluation — no multi-view association or 3D reconstruction involved.

</domain>

<decisions>
## Implementation Decisions

### Metric scope
- Use existing `TrackingMetrics` (track count, length stats, coast frequency) for 2D tracking
- Adapt `FragmentationMetrics` gap/continuity analysis for 2D tracklets (gaps, births, deaths, continuity ratio within the single camera)
- Global aggregates only — no per-camera breakdown (there's only one camera)
- No new metrics (no ID switches, MOTA, etc.) — existing evaluators cover the success criteria

### Execution method
- Standalone script in `scripts/` (similar pattern to `investigate_occlusion.py`)
- Use Phase 78.1 production models (OBB + pose) — matches what the rest of the milestone will use
- Single run, no variance analysis
- Script outputs both metrics (to console/file) and an annotated video with track IDs overlaid on each detection

### Document format
- Baseline metrics document at `.planning/phases/80-baseline-metrics/80-BASELINE.md`
- Structured metrics table followed by gap analysis section stating delta to 9-track zero-fragmentation target
- Text and numbers only — no embedded screenshots (annotated video exists separately)

### Target definition
- 9 fish are visible in e3v83eb during frames 3300-4500
- Qualitative target: "9 tracks with zero fragmentation" — no numeric thresholds defined
- No minimum improvement thresholds for Phase 84; Phase 84 compares freely against the baseline numbers
- Phase 84 success criteria already require "measurable improvement on at least one primary metric"

### Claude's Discretion
- Exact script structure and argument handling
- How to adapt FragmentationMetrics gap analysis for 2D tracklets (may reuse logic or write new)
- Video annotation style (colors, font, overlay positioning)

</decisions>

<specifics>
## Specific Ideas

- Script pattern should be similar to `scripts/investigate_occlusion.py` — standalone, runs detection + tracking on a configurable clip
- The annotated video should show track IDs so fragmentation/ID switches are visually obvious

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `evaluation/stages/tracking.py`: `TrackingMetrics` and `evaluate_tracking()` — computes track count, length stats, coast frequency from `Tracklet2D` list
- `evaluation/stages/fragmentation.py`: `FragmentationMetrics` and `evaluate_fragmentation()` — gap analysis, continuity ratios, births/deaths (currently operates on 3D `Midline3D`, needs adaptation for 2D tracklets)
- `scripts/investigate_occlusion.py`: Existing standalone script that runs detection + pose on a single camera — can serve as structural template
- `core/tracking/stage.py`: OC-SORT tracking stage with `run()` method

### Established Patterns
- Evaluation stages are pure functions: accept typed data, return frozen dataclass metrics
- Scripts in `scripts/` are standalone, not imported by the main package
- Video annotation patterns exist in `investigate_occlusion.py`

### Integration Points
- Detection stage → OC-SORT tracking stage → metric computation
- Config system provides model paths (Phase 78.1 production models already configured)
- `Tracklet2D` is the data structure bridging tracking output to metric evaluation

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 80-baseline-metrics*
*Context gathered: 2026-03-10*
