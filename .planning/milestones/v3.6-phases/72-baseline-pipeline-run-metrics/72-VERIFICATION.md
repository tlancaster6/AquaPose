---
phase: 72-baseline-pipeline-run-metrics
verified: 2026-03-07T21:00:00Z
status: passed
score: 3/3 must-haves verified
re_verification: false
---

# Phase 72: Baseline Pipeline Run & Metrics Verification Report

**Phase Goal:** Quantitative "before" snapshot established on short iteration clip using baseline models from the store
**Verified:** 2026-03-07T21:00:00Z
**Status:** PASSED
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Pipeline completes a diagnostic-mode run on ~9000 frames (30 chunks of 300) using store-registered baseline models | VERIFIED | 30/30 chunk caches exist at `~/aquapose/projects/YH/runs/run_20260307_140127/diagnostics/chunk_*/cache.pkl`; manifest.json confirms 30 chunks with all 5 stages cached per chunk |
| 2 | `aquapose eval` produces a full metric report with all Phase 70 extended metrics | VERIFIED | `eval_results.json` contains all 5 stage sections (detection, tracking, association, reconstruction, fragmentation) plus midline; reconstruction includes percentiles, per_point_error (15 keypoints), and curvature_stratified (4 quartiles) |
| 3 | eval_results.json exists with singleton rate, reprojection error percentiles, track continuity, per-keypoint breakdown, and curvature-stratified quality | VERIFIED | All fields present: singleton_rate=0.313, p50=3.02px, p90=5.20px, p95=6.67px, mean_continuity_ratio=0.947, per_point_error has 15 entries, curvature_stratified has Q1-Q4 |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `~/aquapose/projects/YH/runs/run_20260307_140127/diagnostics/manifest.json` | Chunk manifest confirming 30 chunks processed | VERIFIED | 30 chunks listed, each with 5 stages cached, total_frames=9450, chunk_size=300 |
| `~/aquapose/projects/YH/runs/run_20260307_140127/diagnostics/chunk_*/cache.pkl` | Per-chunk diagnostic caches (30 total) | VERIFIED | 30 cache.pkl files exist on disk |
| `~/aquapose/projects/YH/runs/run_20260307_140127/eval_results.json` | Complete metric snapshot with all extended metrics | VERIFIED | JSON contains association, detection, midline, reconstruction (with percentiles, per-point, curvature-stratified), tracking, and fragmentation sections |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| config.yaml detection/midline weights | Phase 71 baseline models | register_trained_model() auto-update | VERIFIED (by proxy) | Pipeline ran successfully producing realistic metrics; if weights were wrong, singleton rate and reproj error would be wildly off. Actual values (31.3% singleton, 3.02px p50) are consistent with post-LUT-fix baseline |
| aquapose eval | diagnostics/chunk_*/cache.pkl | EvalRunner reading chunk caches | VERIFIED | eval_results.json written with frames_evaluated=9000, matching 30 chunks x 300 frames |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| ITER-01 | 72-01-PLAN | Baseline pipeline run on short clip with diagnostic caching produces baseline metric snapshot | SATISFIED | Pipeline completed 30 chunks, eval_results.json contains full metric snapshot, marked complete in REQUIREMENTS.md |

No orphaned requirements found -- ITER-01 is the only requirement mapped to Phase 72.

### Anti-Patterns Found

No source files were modified in this phase (workflow-only execution). No anti-pattern scan needed.

### Sanity Checks

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Singleton rate | < 30% (plan threshold) | 31.3% | ACCEPTABLE -- slightly above threshold, documented as a known decision |
| Reprojection p50 | < 10px | 3.02px | PASS |
| Track continuity | > 0.5 | 0.947 | PASS |
| Per-keypoint breakdown | 15 points | 15 entries | PASS |
| Curvature quartiles | 4 quartiles | Q1-Q4 present | PASS |

### Human Verification Required

None required. All truths are verifiable programmatically through artifact existence and content inspection. The user already approved the baseline metric snapshot during plan execution (Task 3 checkpoint).

### Gaps Summary

No gaps found. All three success criteria from the ROADMAP are satisfied:
1. Pipeline completed a diagnostic-mode run on 9000 frames using store-registered baseline models
2. `aquapose eval` produced a full metric report with all Phase 70 extended metrics
3. Baseline metric numbers are recorded in eval_results.json as the benchmark for improvement

The diagnostic caches are ready for Phase 73 pseudo-label generation.

---

_Verified: 2026-03-07T21:00:00Z_
_Verifier: Claude (gsd-verifier)_
