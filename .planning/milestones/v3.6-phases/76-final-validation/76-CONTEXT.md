# Phase 76: Final Validation - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Confirm best iteration models at full 5-minute scale with showcase outputs produced. Generate overlay videos from existing run caches and produce a comprehensive summary document covering the v3.5-v3.6 model iteration journey. No new pipeline execution required.

</domain>

<decisions>
## Implementation Decisions

### Pipeline run scope
- Reuse existing run `run_20260309_175421` (Phase 74 round 1 run) — no new pipeline execution
- Use existing eval_results.json from Phase 74 — no re-run of `aquapose eval`
- Run already covers 9000 frames (30 chunks × 300), diagnostic mode, with round 1 winner models

### Overlay video content
- Generate all three viz outputs: overlay mosaic, trail videos, detection overlay PNGs
- Overlay: mosaic only (per-camera overlay videos deferred — not currently implemented)
- Trails: use existing fast-mode fade trails (solid points, no per-point alpha blending)
- Run `aquapose viz --overlay --trails --fade-trails --detections` on the existing run

### Summary document
- Full methodology report covering v3.5 (pseudo-labeling infrastructure) + v3.6 (iteration loop)
- File: `.planning/phases/76-final-validation/76-REPORT.md` (not SUMMARY.md — reserved for GSD)
- Audience: personal reference (assumes familiarity with AquaPose internals)
- Content: methodology, per-phase outcomes, model provenance chain, metrics tables (round 0 vs round 1), known limitations
- Consolidate Phase 74 DECISION.md comparison tables into the report

### Validation criteria
- Sanity check only — Phase 74 already accepted the models with documented rationale
- Trust tooling: if `aquapose viz` completes without error, outputs are valid
- Include a "known limitations" section documenting remaining weaknesses (singleton rate ~27%, high-curvature tail error, any visual artifacts)

### Claude's Discretion
- Report structure and section ordering
- Level of detail in methodology narrative
- Which specific metrics to highlight vs include in appendix tables
- How to present the model provenance chain

</decisions>

<specifics>
## Specific Ideas

- SUMMARY.md is reserved for GSD plan summaries — use 76-REPORT.md instead
- The Phase 74 DECISION.md already has comprehensive comparison tables that should be incorporated rather than regenerated
- Trail videos should use the existing fast-mode implementation (solid points) rather than alpha-blended fading

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `aquapose viz --overlay`: generates 2D reprojection overlay mosaic video from run caches
- `aquapose viz --trails --fade-trails`: generates per-camera trail videos with fast-mode fading
- `aquapose viz --detections`: generates detection overlay mosaic PNGs
- `aquapose eval-compare`: cross-run comparison with delta computation and table formatting
- `evaluation/compare.py`: `load_eval_results()`, `compute_deltas()`, `format_comparison_table()`
- `evaluation/viz/overlay.py`: overlay generation with deterministic color palette per fish ID

### Established Patterns
- Viz outputs go to `{run_dir}/viz/` by default
- Eval results stored as `eval_results.json` in run directory
- Run caches contain all diagnostic data needed for post-hoc viz generation

### Integration Points
- Run directory: `~/aquapose/projects/YH/runs/run_20260309_175421/`
- Baseline run for comparison: `~/aquapose/projects/YH/runs/run_20260307_140127/`
- Phase 74 DECISION.md: `.planning/phases/74-round-1-evaluation-decision/74-DECISION.md`
- Phase 73 RESULTS.md: `.planning/phases/73-round-1-pseudo-labels-retraining/73-RESULTS.md`

</code_context>

<deferred>
## Deferred Ideas

- Per-camera overlay videos (individual camera overlays in addition to mosaic) — future enhancement to viz tooling

</deferred>

---

*Phase: 76-final-validation*
*Context gathered: 2026-03-09*
