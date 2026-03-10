# Phase 76: Final Validation - Research

**Researched:** 2026-03-09
**Domain:** Pipeline visualization, report writing, CLI tooling
**Confidence:** HIGH

## Summary

Phase 76 is an operational phase with no new code to write. The work consists of three tasks: (1) generating visualization outputs from an existing pipeline run using `aquapose viz`, (2) producing a comprehensive summary report document, and (3) confirming the existing eval results satisfy final validation requirements.

The existing run `run_20260309_175421` already has `eval_results.json` and `eval_comparison.json`. No new pipeline execution or eval run is needed. The viz outputs (overlay mosaic video, trail videos, detection PNGs) do not yet exist and must be generated. The report must consolidate metrics from Phase 72 (baseline), Phase 73 (training results), and Phase 74 (pipeline evaluation decision) into a single coherent document.

**Primary recommendation:** Execute viz generation first (long-running), then write the report while referencing existing Phase 74 DECISION.md tables.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Reuse existing run `run_20260309_175421` (Phase 74 round 1 run) -- no new pipeline execution
- Use existing eval_results.json from Phase 74 -- no re-run of `aquapose eval`
- Generate all three viz outputs: overlay mosaic, trail videos, detection overlay PNGs
- Overlay: mosaic only (per-camera overlay videos deferred)
- Trails: use existing fast-mode fade trails (solid points, no per-point alpha blending)
- Run `aquapose viz --overlay --trails --fade-trails --detections` on the existing run
- Full methodology report covering v3.5 (pseudo-labeling infrastructure) + v3.6 (iteration loop)
- File: `.planning/phases/76-final-validation/76-REPORT.md` (not SUMMARY.md -- reserved for GSD)
- Audience: personal reference (assumes familiarity with AquaPose internals)
- Content: methodology, per-phase outcomes, model provenance chain, metrics tables (round 0 vs round 1), known limitations
- Consolidate Phase 74 DECISION.md comparison tables into the report
- Sanity check only -- Phase 74 already accepted the models with documented rationale
- Trust tooling: if `aquapose viz` completes without error, outputs are valid
- Include a "known limitations" section documenting remaining weaknesses (singleton rate ~27%, high-curvature tail error, any visual artifacts)

### Claude's Discretion
- Report structure and section ordering
- Level of detail in methodology narrative
- Which specific metrics to highlight vs include in appendix tables
- How to present the model provenance chain

### Deferred Ideas (OUT OF SCOPE)
- Per-camera overlay videos (individual camera overlays in addition to mosaic) -- future enhancement to viz tooling

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FINAL-01 | Full 5-minute pipeline run with best iteration models and full `aquapose eval` report | Already satisfied: existing run `run_20260309_175421` covers 9000 frames (5 min at 30fps) with eval_results.json. Context decision: reuse existing run, no re-execution. |
| FINAL-02 | Overlay videos generated for all 12 cameras from final run | `aquapose viz --overlay --trails --fade-trails --detections` generates mosaic overlay video (all 12 cameras in grid), per-camera trail videos (12 separate), and detection PNG. Note: CONTEXT.md clarifies "overlay for all 12 cameras" means mosaic format, not per-camera individual overlays. |
| FINAL-03 | Summary document with metrics table (round 0 vs round 1 vs round 2 vs final), key observations, and known limitations | Write 76-REPORT.md consolidating Phase 74 DECISION.md tables. Round 2 column is N/A (skipped per Phase 74 decision). Round 0 = baseline, Round 1 = final. |

</phase_requirements>

## Standard Stack

No new libraries or dependencies needed. This phase uses only existing CLI commands.

### Core Tools
| Tool | Purpose | Why Standard |
|------|---------|--------------|
| `aquapose viz` | Generate overlay mosaic, trail videos, detection PNGs | Existing CLI command, outputs to `{run_dir}/viz/` |
| `aquapose eval-compare` | Cross-run metric comparison | Already used in Phase 74, produces formatted tables |

### Key CLI Commands

```bash
# Generate all viz outputs from existing run
hatch run aquapose viz --overlay --trails --fade-trails --detections \
  ~/aquapose/projects/YH/runs/run_20260309_175421

# Compare runs (already done in Phase 74, data available)
hatch run aquapose eval-compare \
  ~/aquapose/projects/YH/runs/run_20260307_140127 \
  ~/aquapose/projects/YH/runs/run_20260309_175421
```

## Architecture Patterns

### Viz Output Structure
```
~/aquapose/projects/YH/runs/run_20260309_175421/
├── eval_results.json          # Already exists
├── eval_comparison.json       # Already exists
├── viz/                       # Will be created by `aquapose viz`
│   ├── overlay_mosaic.mp4     # 12-camera grid with reprojected midlines
│   ├── trails/                # Per-camera trail videos
│   │   ├── {cam_id}_trails.mp4  # One per camera (12 total)
│   │   └── ...
│   └── detections/            # Detection overlay PNGs
│       ├── {cam_id}_detections.png
│       └── ...
```

### Report Structure (76-REPORT.md)
```
.planning/phases/76-final-validation/
├── 76-REPORT.md               # Final methodology + metrics report
```

### Data Sources for Report
| Source | What to Extract |
|--------|----------------|
| Phase 74 `74-DECISION.md` | Full metric comparison table (round 0 vs round 1), per-keypoint table, curvature-stratified table, go/no-go rationale |
| Phase 73 `73-RESULTS.md` | Training metrics (mAP tables), A/B comparison results, data composition |
| Phase 72 baseline run | Baseline metric snapshot (already in 74-DECISION.md) |
| `eval_results.json` (round 1 run) | Raw eval metrics for sanity check |

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Metric comparison tables | Manual diff of JSON files | Phase 74 DECISION.md already has formatted tables | Avoids recomputation and potential transcription errors |
| Overlay generation | Custom frame-by-frame rendering | `aquapose viz --overlay` | Handles refractive projection, color palette, mosaic layout |
| Trail videos | Custom tracking visualization | `aquapose viz --trails --fade-trails` | Handles chunk boundaries, camera layout, fade effects |

## Common Pitfalls

### Pitfall 1: Viz Command Timeouts
**What goes wrong:** The overlay mosaic processes 9000 frames x 12 cameras. This is a long-running operation that could take 10-30+ minutes.
**How to avoid:** Run as a background task or use TaskCreate subagent. Do not run in a synchronous bash call with a 2-minute timeout.

### Pitfall 2: SUMMARY.md Name Collision
**What goes wrong:** GSD reserves `*-SUMMARY.md` for plan summaries. Writing the report to SUMMARY.md would conflict.
**How to avoid:** Use `76-REPORT.md` as specified in CONTEXT.md.

### Pitfall 3: Regenerating Already-Available Data
**What goes wrong:** Re-running `aquapose eval` or `aquapose run` when results already exist, wasting GPU time.
**How to avoid:** CONTEXT.md explicitly says reuse existing run and eval_results.json. Only `aquapose viz` needs to run.

### Pitfall 4: Report Metrics Mismatch
**What goes wrong:** Manually transcribing metrics from eval_results.json and getting different numbers than Phase 74 DECISION.md.
**How to avoid:** Copy tables directly from 74-DECISION.md rather than re-extracting from JSON.

## Code Examples

### Running Viz Generation
```bash
# All three viz types in one command
hatch run aquapose viz --overlay --trails --fade-trails --detections \
  ~/aquapose/projects/YH/runs/run_20260309_175421
```

### Verifying Viz Outputs
```bash
# Check that outputs were created
ls ~/aquapose/projects/YH/runs/run_20260309_175421/viz/
# Should contain: overlay_mosaic.mp4, trails/ directory, detections/ directory
```

## Report Content Outline

Recommended structure for 76-REPORT.md (Claude's discretion on exact ordering):

1. **Overview** -- v3.5-v3.6 scope, what was accomplished
2. **Model Provenance Chain** -- Baseline models -> pseudo-labels -> curation -> round 1 models
3. **Training Results** -- From Phase 73: OBB and pose A/B comparison tables
4. **Pipeline Evaluation** -- From Phase 74: full metric comparison table, per-keypoint, curvature-stratified
5. **Key Outcomes** -- Headline improvements: singleton -12.5%, p50 reproj -28.4%, p90 reproj -19.8%
6. **Known Limitations** -- Singleton rate still ~27%, high-curvature tail residual error, algae domain shift
7. **Visualization Outputs** -- List of generated files with descriptions

### Known Limitations to Document
From existing project context:
- Singleton rate ~27% (down from 31.3% but still notable)
- High-curvature tail keypoint error: reduced 38% but still highest per-keypoint error (4.51 px mean, 8.12 px P90)
- Algae domain shift between manual annotations (clean tank) and current conditions
- Inlier ratio slightly decreased (87.1% -> 84.4%) with round 1 models
- Only 1 iteration round performed (round 2 skipped)
- Q4 (curved) fish still have higher reprojection error than straight fish (2.94 vs 2.68 px mean)

## Open Questions

1. **Viz runtime estimate**
   - What we know: 9000 frames x 12 cameras, overlay requires refractive projection per frame
   - What's unclear: Exact wall-clock time for viz generation
   - Recommendation: Run as background/subagent task, expect 10-30 minutes

2. **Trail video file naming**
   - What we know: Trails are per-camera
   - What's unclear: Exact output filenames (may vary based on camera IDs)
   - Recommendation: Verify after generation by listing viz/ directory

## Sources

### Primary (HIGH confidence)
- `src/aquapose/cli.py` -- CLI command signatures for `viz` and `eval-compare`
- `src/aquapose/evaluation/viz/overlay.py` -- Overlay implementation
- `src/aquapose/evaluation/viz/trails.py` -- Trails implementation
- `.planning/phases/74-round-1-evaluation-decision/74-DECISION.md` -- All metric comparison tables
- `.planning/phases/73-round-1-pseudo-labels-retraining/73-RESULTS.md` -- Training results
- `~/aquapose/projects/YH/runs/run_20260309_175421/eval_results.json` -- Existing eval data

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all tools are existing CLI commands already exercised in prior phases
- Architecture: HIGH -- output paths and data sources fully known
- Pitfalls: HIGH -- based on direct project experience and CONTEXT.md constraints

**Research date:** 2026-03-09
**Valid until:** 2026-04-09 (stable -- no external dependencies)
