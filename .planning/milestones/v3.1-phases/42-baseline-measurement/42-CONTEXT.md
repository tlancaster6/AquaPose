# Phase 42: Baseline Measurement - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Run the evaluation harness (Phase 41) against the current (pre-rebuild) reconstruction backend to establish Tier 1 and Tier 2 reference metrics. Save results as the regression baseline for Phase 44 comparison. Requirements: EVAL-06.

</domain>

<decisions>
## Implementation Decisions

### Fixture Source
- Single fixture file, path passed as argument to the baseline script
- Fixture will exist by execution time (from a diagnostic pipeline run via Phase 40)
- No fixture validation in the baseline script — trust Phase 40 serialization and Phase 41 harness validation

### Outlier Flagging
- Statistical threshold: flag entries exceeding 2 standard deviations from mean (fixed, not configurable)
- Inline markers in the summary table (e.g. asterisk next to flagged values) — no separate outlier section
- Outlier detection applied to both Tier 1 (reprojection error) and Tier 2 (leave-one-out stability)

### Baseline Persistence
- Regression reference JSON saved next to the fixture file (consistent with Phase 41's eval_results.json pattern)
- Baseline JSON committed to git — stable reference that survives across branches and machines
- Includes metadata: timestamp, fixture path, backend identifier
- Human-readable report saved as .txt next to fixture AND printed to console

### Invocation Method
- Standalone Python script: `scripts/measure_baseline.py`
- Uses argparse with fixture path as required argument
- Output location is deterministic (next to fixture) — no output dir argument
- No smoke test — manual execution only, this is a one-off measurement

### Claude's Discretion
- Exact argparse flag names and help text
- Report table formatting details
- Metadata schema in the baseline JSON
- How to identify the "backend identifier" for metadata (e.g. module path, config hash)

</decisions>

<specifics>
## Specific Ideas

- The script is essentially a thin wrapper: load fixture → call Phase 41 harness → add outlier flags → save results + report
- Phase 41 harness already computes Tier 1 and Tier 2 metrics — this phase adds outlier flagging, persistence, and the script entrypoint
- Baseline JSON must be machine-diffable for Phase 44 regression comparison

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 41 evaluation harness: loads fixture + calibration, computes Tier 1 and Tier 2 metrics
- `load_midline_fixture()` in `io/midline_fixture.py`: NPZ deserialization from Phase 40
- Phase 41 outputs `eval_results.json` next to fixture — baseline follows same convention

### Established Patterns
- Reconstruction backends in `core/reconstruction/backends/` with `reconstruct_frame()` API
- `RefractiveProjectionModel` for 3D→2D reprojection (used by Tier 1 metric)
- Phase 41 context specifies: JSON format, per-fish and per-camera aggregates (mean, max)

### Integration Points
- Phase 41 harness API: takes fixture path, returns metric results
- Phase 44 reads the baseline JSON to compare against new backend results
- Fixture files from Phase 40 diagnostic capture

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 42-baseline-measurement*
*Context gathered: 2026-03-02*
