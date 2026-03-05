# Phase 60: End-to-End Performance Validation - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Validate that the v3.4 Performance Optimization milestone delivered measurable speedups across the four optimized bottlenecks (batched YOLO inference, frame I/O, vectorized DLT, vectorized association scoring) while preserving correctness. Produce a markdown summary report with before/after timing and eval pass/fail.

</domain>

<decisions>
## Implementation Decisions

### Measurement approach
- Use the existing TimingObserver for per-stage wall-clock timing
- Baseline is saved from pre-optimization run `run_20260304_180748` (single chunk, 200 frames, 12 cameras, total 914.2s) — already copied to `baseline-timing.txt` in the phase directory
- Post-optimization run uses identical YH project config, single chunk, same 200-frame window
- Correctness validated by running `aquapose eval` on the post-optimization diagnostic run and comparing to baseline eval metrics
- Run mode: `--mode diagnostic` to generate caches for eval

### Report format
- Markdown summary document saved in `.planning/phases/60-end-to-end-performance-validation/`
- Per-stage speedup ratios: before time, after time, speedup factor (e.g. "Association: 35.9s -> Xs, Y.Zx faster")
- Focus on the four optimized stages: detection, association, midline, reconstruction
- Total wall-clock speedup ratio
- Eval correctness: pass/fail summary only (not full metrics side-by-side)

### Pass/fail criteria
- No hard minimum speedup target — document whatever improvement was achieved
- Correctness: threshold tolerance for eval metric differences (small floating-point variance from GPU non-determinism is acceptable)
- If a correctness regression is detected beyond tolerance, the milestone is blocked — cannot be marked complete until resolved
- Phase creates an issue/task for any regression found

### Scope of runs
- Single chunk, same YH config as baseline
- No GPU warm-up run — baseline didn't have one either, apples-to-apples
- Single iteration, no averaging — run-to-run variance is minimal for GPU-bound workloads of this duration

### Claude's Discretion
- Exact floating-point tolerance thresholds for eval metric comparison
- Report formatting details and section organization
- How to present the eval baseline (capture from existing run or reference prior eval output)

</decisions>

<specifics>
## Specific Ideas

- Baseline timing already saved: `.planning/phases/60-end-to-end-performance-validation/baseline-timing.txt`
- Baseline source: `~/aquapose/projects/YH/runs/run_20260304_180748/` (pre-optimization, single chunk)
- All runs in `~/aquapose/projects/YH/runs/` are pre-optimization and can be used for eval baseline comparison

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TimingObserver` at `engine/timing.py` — already records per-stage wall-clock time and generates formatted reports
- `EvalRunner` at `evaluation/runner.py` — orchestrates per-stage evaluation from diagnostic run directories
- `aquapose eval` CLI command at `cli.py:204` — evaluates a diagnostic run directory and prints quality report
- `aquapose run` CLI command — runs pipeline with `--mode diagnostic --max-chunks 1` for single-chunk runs

### Established Patterns
- TimingObserver writes `timing.txt` to run output directory automatically
- EvalRunner loads chunk caches from `diagnostics/chunk_NNN/cache.pkl` and computes per-stage metrics
- Diagnostic mode generates all caches needed for evaluation

### Integration Points
- Post-optimization run: `aquapose run --config ~/aquapose/projects/YH/config.yaml --mode diagnostic --max-chunks 1`
- Eval comparison: `aquapose eval <run_dir>` on both baseline and post-optimization runs
- Report output: `.planning/phases/60-end-to-end-performance-validation/60-REPORT.md`

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 60-end-to-end-performance-validation*
*Context gathered: 2026-03-05*
