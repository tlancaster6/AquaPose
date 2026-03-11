# Requirements: AquaPose

**Defined:** 2026-03-11
**Core Value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation

## v3.8 Requirements

Requirements for the Improved Association milestone. Each maps to roadmap phases.

### Data Contract

- [ ] **DATA-01**: Tracklet2D carries per-frame keypoints array (T, K, 2) from tracker to association stage
- [ ] **DATA-02**: Tracklet2D carries per-frame keypoint confidence array (T, K) with 0.0 for coasted frames

### Scoring

- [ ] **SCORE-01**: Association scorer casts rays from multiple keypoints per detection per frame, not just one centroid
- [ ] **SCORE-02**: Low-confidence keypoints (below configurable threshold) are excluded from scoring per frame
- [ ] **SCORE-03**: Per-keypoint ray-ray distances are aggregated into a single pairwise score via configurable method (mean, trimmed mean)
- [ ] **SCORE-04**: Multi-keypoint scoring is vectorized (NumPy broadcasting, no per-pair Python loop)

### Group Validation

- [ ] **VALID-01**: After clustering, each tracklet's multi-keypoint residuals are computed against its group
- [ ] **VALID-02**: Changepoint detection identifies temporal ID swap points in per-tracklet residual series
- [ ] **VALID-03**: Swapped tracklets are split at the changepoint — consistent segment stays, inconsistent segment becomes singleton candidate
- [ ] **VALID-04**: Tracklets with high overall residual (no changepoint) are evicted to singleton status

### Singleton Recovery

- [ ] **RECOV-01**: Each singleton is scored against all existing groups using multi-keypoint residuals
- [ ] **RECOV-02**: Singletons with strong overall match to one group are assigned to that group
- [ ] **RECOV-03**: Singletons with no overall match but a temporal split matching two different groups are split and assigned (swap-aware recovery)
- [ ] **RECOV-04**: Same-camera overlap constraint is enforced during singleton assignment (no two tracklets from same camera with overlapping frames in one group)

### Cleanup

- [ ] **CLEAN-01**: Fragment merging code and max_merge_gap config field removed
- [ ] **CLEAN-02**: Refinement module replaced by group validation (refinement.py deleted after consumer audit)

### Evaluation

- [ ] **EVAL-01**: Parameter tuning pass on real data measuring singleton rate, reprojection error, and grouping quality vs v3.7 baseline
- [ ] **EVAL-02**: End-to-end pipeline run with tuned parameters confirms improvement over v3.7

## Future Requirements

### Association v2+

- **ASSOC-V2-01**: Appearance-based scoring for fish in close parallel (geometric ambiguity)
- **ASSOC-V2-02**: Adaptive keypoint subset selection (learn which keypoints carry most signal per camera pair)

## Out of Scope

| Feature | Reason |
|---------|--------|
| 3D volumetric voting / per-frame 3D association | Re-introduces fragility that caused v1.0→v2.0 restructuring |
| Appearance features for re-identification | Requires training an appearance model; deferred to future milestone |
| ruptures library for changepoint detection | Simple max-split sweep is sufficient for single-swap case; avoids new dependency |
| Multi-swap detection within single tracklet | Rare edge case; recursive binary split handles it if needed |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | — | Pending |
| DATA-02 | — | Pending |
| SCORE-01 | — | Pending |
| SCORE-02 | — | Pending |
| SCORE-03 | — | Pending |
| SCORE-04 | — | Pending |
| VALID-01 | — | Pending |
| VALID-02 | — | Pending |
| VALID-03 | — | Pending |
| VALID-04 | — | Pending |
| RECOV-01 | — | Pending |
| RECOV-02 | — | Pending |
| RECOV-03 | — | Pending |
| RECOV-04 | — | Pending |
| CLEAN-01 | — | Pending |
| CLEAN-02 | — | Pending |
| EVAL-01 | — | Pending |
| EVAL-02 | — | Pending |

**Coverage:**
- v3.8 requirements: 18 total
- Mapped to phases: 0
- Unmapped: 18 ⚠️

---
*Requirements defined: 2026-03-11*
*Last updated: 2026-03-11 after initial definition*
