# Phase 33: Keypoint Midline Backend - Context

**Gathered:** 2026-02-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Pipeline supports a `direct_pose` midline backend that produces N ordered midline points with per-point confidence via keypoint regression (U-Net encoder + regression head → 6 keypoints → spline → resample). Both reconstruction backends (triangulation, curve optimizer) weight observations by that confidence. Selecting `midline_backend: segment_then_extract` restores original behavior. Training data generation and model training are out of scope (manual annotations exist, training infra is Phase 31).

</domain>

<decisions>
## Implementation Decisions

### Anatomical keypoints
- The 6 keypoints are: **nose, head, spine1, spine2, spine3, tail** (NOT the hallucinated anatomical names previously in the guidebook — already corrected)
- Each keypoint has a fixed `t` value in [0, 1] representing its arc-length fraction along the body
- t-values are a **configurable field** in the keypoint backend config, not hardcoded
- A new CLI command `aquapose prep calibrate-keypoints` auto-calibrates t-values from training data
- The `prep` CLI group is new — will eventually also house manual LUT generation (deferred)
- Manual keypoint annotations already exist for training data

### Partial visibility handling
- Two-tier approach: hard confidence floor (configurable, default 0.1) removes truly terrible inferences → NaN+conf=0
- Remaining points pass through with their actual confidence values for soft weighting in reconstruction
- Minimum observed keypoint count: **3 of 6** (configurable via `min_observed_keypoints`)
- Below the minimum → empty midline (same as degenerate mask in segment-then-extract)
- Output is always exactly `n_sample_points` — NaN+conf=0 outside `[t_min_observed, t_max_observed]`, never a shorter array
- The n_points correspondence contract from PITFALLS.md is respected: point index i always means the same anatomical position across cameras

### Backend failure mode
- Degenerate keypoint output → flagged empty midline, consistent with segment-then-extract
- No cross-backend fallback (no falling back to segment-then-extract on failure)
- Trust the model's confidence — no separate sanity checks on keypoint spatial distribution
- Empty midlines produced silently (no warning logs); observers/diagnostics track via events

### Confidence weighting in reconstruction
- Both triangulation and curve optimizer use **sqrt(confidence)** to scale observation weights
- Triangulation: scale each view's A-matrix rows by sqrt(conf) before SVD (standard weighted least squares)
- Curve optimizer: same sqrt weighting on fitting objective terms
- Points with NaN coordinates are **excluded entirely** from the DLT system (not passed as zero-weight rows)
- When confidence is None (segment-then-extract), uniform weights apply — no config override needed
- Identical output to previous version when confidence is None (backward compatibility)

### Claude's Discretion
- Regression head architecture (layers, activation, output format)
- Spline degree and fitting method for keypoints → N-point midline
- Internal structure of `aquapose prep calibrate-keypoints` implementation
- Config field names and default values (except those specified above)

</decisions>

<specifics>
## Specific Ideas

- The guidebook's example keypoint names (snout, pectoral fin, mid-body, caudal peduncle, tail tip) were hallucinated — the real ones are nose, head, spine1, spine2, spine3, tail. Guidebook already corrected.
- Research PITFALLS.md documents the skeleton-squashing risk: if keypoints are omitted instead of NaN-padded, triangulation misaligns points across views. The min-3-observed + fixed-N-output contract prevents this.

</specifics>

<deferred>
## Deferred Ideas

- Manual LUT generation under `aquapose prep` CLI group — future phase
- Body-model extrapolation for partial midlines (QUAL-01) — future requirement
- Confidence calibration via temperature scaling (QUAL-02) — future requirement

</deferred>

---

*Phase: 33-keypoint-midline-backend*
*Context gathered: 2026-02-28*
