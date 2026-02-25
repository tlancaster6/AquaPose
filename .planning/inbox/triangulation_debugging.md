# Triangulation Debugging Findings

## Executive Summary

The previously reported "143px mean residual / 95% low-confidence rate" was **an artifact of two compounding bugs**, not a fundamental triangulation quality problem. The actual pipeline residual (with epipolar refinement active and NaN-filtered) is **13.3px**, and the true bottleneck is a quality-vs-coverage tradeoff caused by coupled thresholds.

## Context

- Diagnostic script: `scripts/diagnose_triangulation.py` (created for this investigation)
- Source of truth: `src/aquapose/reconstruction/triangulation.py`
- Data: 30 frames, 233 fish-frame instances, 12 cameras (6 active), 3-4 cameras per fish
- Step 0 already applied: `scripts/diagnose_pipeline.py` was de-inlined to call `triangulate_midlines()` directly (previously it inlined the loop and **skipped `_refine_correspondences_epipolar` entirely**)

---

## Bug 1: NaN Contamination in Residual Computation (CRITICAL)

### Location
`src/aquapose/reconstruction/triangulation.py`, lines 765-767 inside `triangulate_midlines()`:

```python
# Current code (buggy):
for j in range(N_SAMPLE_POINTS):
    if valid_np[j] and not np.any(np.isnan(proj_np[j])):
        err = float(np.linalg.norm(proj_np[j] - obs_pts[j]))
```

### Problem
`obs_pts` comes from `cam_midlines[cid].points`, which contains NaN values injected by `_refine_correspondences_epipolar()` (line 573: `new_points = np.full_like(tgt_midline.points, np.nan)`). The condition checks for NaN in `proj_np[j]` (the reprojected spline point) but NOT in `obs_pts[j]` (the observed 2D point). When `obs_pts[j]` is NaN, `np.linalg.norm(proj_np[j] - obs_pts[j])` returns NaN, which poisons `all_residuals`, `mean_residual`, `max_residual`, and `arc_length` (via the NaN propagating through `np.mean`/`np.max`).

### Impact
- **125 of 139 successful fish** have NaN residuals
- All downstream metrics are wrong: `mean_residual`, `max_residual`, `is_low_confidence`, `per_camera_residuals`
- The `arc_length` field is also NaN (separate propagation path — needs verification)
- Any code consuming `Midline3D.mean_residual` gets NaN

### Fix
Add NaN check for `obs_pts[j]`:

```python
for j in range(N_SAMPLE_POINTS):
    if valid_np[j] and not np.any(np.isnan(proj_np[j])) and not np.any(np.isnan(obs_pts[j])):
        err = float(np.linalg.norm(proj_np[j] - obs_pts[j]))
```

### Verification
After fix, re-run:
```bash
python scripts/diagnose_triangulation.py --stop-frame 30
```
- Baseline mean_residual should be ~13px (matching the NaN-filtered diagnostic value)
- NaN residual count should drop to 0
- H3 sweep should show actual values instead of NaN

---

## Bug 2: Coupled snap_threshold and inlier_threshold (DESIGN FLAW)

### Location
`src/aquapose/reconstruction/triangulation.py`, line 645-646 in `triangulate_midlines()`:

```python
cam_midlines = _refine_correspondences_epipolar(
    cam_midlines, models, snap_threshold=inlier_threshold  # <-- coupled
)
```

### Problem
`snap_threshold` controls how far a target body point can be from the epipolar curve before being rejected (set to NaN). `inlier_threshold` controls the max reprojection error for RANSAC inlier classification. These serve different purposes but are set to the same value.

- Tight snap (good correspondences) + tight inlier (few cameras survive) = low residuals but many fish fail
- Loose snap (poor correspondences) + loose inlier (many cameras survive) = high residuals but most fish succeed

### Evidence (H3 threshold sweep)

| Threshold | Mean Residual | Success Rate | Median Cameras |
|-----------|--------------|-------------|---------------|
| 10px      | 8.7px        | 37.8%       | 2             |
| 20px      | 8.8px        | 48.1%       | 2             |
| 50px (default) | 13.9px  | 59.7%       | 2             |
| 100px     | 32.4px       | 72.5%       | 2             |
| 200px     | 68.3px       | 93.6%       | 3             |
| 300px     | 101.6px      | 98.3%       | 4             |
| 500px     | 117.8px      | 99.6%       | 4             |

At the default threshold (50px), **40% of fish fail** spline fitting because too many body points are killed by NaN (epipolar snap rejects them).

### Fix
Decouple the two thresholds. Add a separate `snap_threshold` parameter to `triangulate_midlines()`:

```python
def triangulate_midlines(
    midline_set: MidlineSet,
    models: dict[str, RefractiveProjectionModel],
    frame_index: int = 0,
    inlier_threshold: float = DEFAULT_INLIER_THRESHOLD,
    snap_threshold: float = 20.0,  # NEW — independent of inlier_threshold
) -> dict[int, Midline3D]:
```

Then on line 645-646:
```python
cam_midlines = _refine_correspondences_epipolar(
    cam_midlines, models, snap_threshold=snap_threshold
)
```

Suggested values: `snap_threshold=20px` (tight epipolar matching), `inlier_threshold=150-200px` (permissive inlier gating). This should give ~9px residuals with >90% success rate.

---

## Finding 3: Orientation Alignment Cannot Be Improved Pre-Epipolar (H1 — DEPRIORITIZED)

### Location
`src/aquapose/reconstruction/triangulation.py`, `_align_midline_orientations()` (lines 396-455)

### Problem
The greedy algorithm is optimal only **36.5% of the time** (H1 diagnostic). However, both brute-force alternatives (chord-length sum and RANSAC residual scoring) **regressed to ~143px** — the same as no alignment at all.

### Root cause
Orientation alignment runs **before** epipolar refinement, so it operates on raw correspondences with ~148px median epipolar noise (H5). At this noise level, no global scoring function can reliably distinguish correct from incorrect flip orientations. The greedy pairwise approach is paradoxically more robust because it only compares each camera against a single reference — the noise partially cancels in pairwise comparison but compounds in global aggregation.

### Why this doesn't matter much
H2 showed that epipolar refinement **does not need orientation alignment**:
- Variant A (alignment only, no epipolar): **143.7px**
- Variant C (epipolar only, no alignment): **8.4px**
- Variant B (alignment + epipolar, current pipeline): **13.3px** (NaN-filtered)

Epipolar refinement snaps each body point independently to the nearest skeleton point along the epipolar curve. It doesn't care about head-to-tail ordering. The H2 data suggests alignment may even slightly hurt (13.3px vs 8.4px) — possibly because a bad flip causes epipolar snapping to find a worse local match.

### Recommended fix
**Option A (safe)**: Keep greedy alignment as-is. It's not great but it's the least-bad pre-epipolar strategy, and epipolar refinement compensates. Focus effort on Bugs 1-2 instead.

**Option B (aggressive)**: Remove `_align_midline_orientations` from the pipeline entirely and rely solely on epipolar refinement. This would need validation that the 8.4px variant-C result holds with decoupled thresholds (Bug 2 fix). Risk: variant C only achieved 98/233 success (42%) — lower than the full pipeline's 139/233 (60%) — so alignment may help spline fitting even if it doesn't help residuals.

**Option C (deferred)**: Move alignment to run AFTER epipolar refinement, on the cleaned correspondences. At that point the data is clean enough (~8px noise) for any scoring function to work. This would only matter for the few body points where epipolar snapping preserved the wrong ordering.

---

## Bug 3: Epipolar Snap Collisions — 17% of Observations Are Duplicates

### Location
`src/aquapose/reconstruction/triangulation.py`, `_refine_correspondences_epipolar()` lines 586-593

### Problem
Each body point on the reference camera is snapped independently to the nearest target skeleton point along its epipolar curve:

```python
best_idx = int(min_dist_to_curve.argmin().item())  # line 588
best_dist = float(min_dist_to_curve[best_idx].item())  # line 589

if best_dist <= snap_threshold:
    new_points[i] = tgt_midline.points[best_idx]  # line 592
    new_hw[i] = tgt_midline.half_widths[best_idx]  # line 593
```

Nothing prevents two different reference body points (e.g., `i=3` and `i=5`) from snapping to the same `best_idx` on the target. When this happens, the target camera provides the **same 2D observation** for two different 3D body points.

### Evidence
Measured on 10 frames:
- Total valid snaps: 628
- Collisions (multiple ref points → same target point): **108 (17.2%)**
- NaN count is identical with/without orientation alignment (1757 vs 1757) — alignment has zero effect on snap behavior

### Impact
17% of camera observations are duplicated. When body point 3 and body point 5 both map to target skeleton point 4:
- Both 3D points receive the same 2D observation from that camera
- The triangulation is biased toward that shared observation
- The spline fit receives correlated errors at those two parameter positions
- This inflates residuals and distorts the reconstructed midline shape

### Fix
Replace independent greedy snapping with a **bijective assignment** that enforces one-to-one mapping between reference body points and target skeleton points.

**Option A — Sequential exclusion (simple)**: Process reference points in order, mark target points as claimed:

```python
claimed = set()
for i in range(n_pts):
    ref_px = torch.from_numpy(ref_midline.points[i]).float()
    epi_curve = _trace_epipolar_curve(
        ref_px, models[ref_id], models[tgt_id], depth_samples
    )
    if epi_curve is None:
        continue

    dists = torch.cdist(tgt_pts_torch.unsqueeze(0), epi_curve.unsqueeze(0))[0]
    min_dist_to_curve, _ = dists.min(dim=1)

    # Mask already-claimed target points
    for idx in claimed:
        min_dist_to_curve[idx] = float("inf")

    best_idx = int(min_dist_to_curve.argmin().item())
    best_dist = float(min_dist_to_curve[best_idx].item())

    if best_dist <= snap_threshold:
        new_points[i] = tgt_midline.points[best_idx]
        new_hw[i] = tgt_midline.half_widths[best_idx]
        claimed.add(best_idx)
```

**Option B — Hungarian assignment (optimal)**: Build a cost matrix `C[i, j]` = distance from reference point `i`'s epipolar curve to target skeleton point `j`, then solve with `scipy.optimize.linear_sum_assignment`. More expensive but globally optimal. With N=15 points, the N³ cost is negligible.

**Recommendation**: Start with Option A (sequential exclusion). It preserves the existing code structure, is easy to verify, and eliminates the collision problem. If residuals don't improve enough, upgrade to Option B.

### Interaction with Bug 2
The collision rate will change when snap_threshold is decoupled from inlier_threshold. A tighter snap_threshold reduces collisions (fewer valid snaps overall) but also increases NaN. Fix Bug 2 first, then measure collision rate at the new threshold to decide whether Bug 3 still matters.

---

## Finding 4: Only 6 of 12 Cameras Produce Midlines (UPSTREAM)

### Evidence
Camera coverage in midline_sets (30 frames):

| Camera  | Midlines | Camera  | Midlines |
|---------|----------|---------|----------|
| e3v831e | 233      | e3v829d | 0        |
| e3v83f0 | 233      | e3v82f9 | 0        |
| e3v8334 | 207      | e3v832e | 0        |
| e3v83eb | 155      | e3v83e9 | 0        |
| e3v82e0 | 78       | e3v83ee | 0        |
| e3v83f1 | 12       | e3v83ef | 0        |

### Impact
Every fish is seen by only 3-4 cameras instead of potentially 6-8. More cameras would improve RANSAC robustness and spline fit quality.

### Investigation needed
The dropout could occur at detection (YOLO misses), segmentation (U-Net fails), tracking (cross-view identity doesn't assign), or midline extraction (skeleton too short). This requires separate investigation — not a triangulation bug. Check detection counts per camera first.

---

## Finding 5: Raw Correspondence Quality (H5 FAIL — Expected)

### Evidence
- Median epipolar distance for raw index-based correspondences: **147.8px**
- P90: 376.6px
- This is expected: BFS skeleton traversal produces arbitrary head-tail ordering

### Implication
This confirms that epipolar refinement is **necessary** and is doing its job well (reduces to 8.4px). No fix needed here — just confirms the pipeline design is correct.

---

## Finding 6: Camera-Pair Calibration Quality (H4 PASS)

### Evidence
Pairwise residual matrix (direct 2-camera triangulation, no RANSAC):

| Pair | Residual |
|------|----------|
| e3v83eb — e3v83f1 | 30px |
| e3v831e — e3v8334 | 38px |
| e3v831e — e3v83f0 | 42px |
| e3v83eb — e3v83f0 | 43px |
| e3v82e0 — e3v8334 | 46px |
| e3v8334 — e3v83eb | 57px |
| e3v83f0 — e3v83f1 | 63px |
| e3v831e — e3v83eb | 66px |
| e3v82e0 — e3v83f0 | 67px |
| e3v82e0 — e3v831e | 112px |
| e3v831e — e3v83f1 | 118px |

Median marginal: 59px. No camera exceeds 2x the median. The calibration quality is acceptable — the 30-60px range is consistent with the refractive model's expected accuracy.

The e3v82e0—e3v831e (112px) and e3v831e—e3v83f1 (118px) pairs are notably worse but not outliers by the 2x-median criterion. These may have poor baseline geometry (nearly collinear camera positions).

---

## Recommended Fix Order

1. **Fix NaN bug** (Bug 1) — trivial one-line fix, unblocks all residual-based metrics
2. **Decouple snap/inlier thresholds** (Bug 2) — ~5 lines changed in `triangulate_midlines()` signature + call site
3. **Fix snap collisions** (Bug 3) — sequential exclusion in `_refine_correspondences_epipolar` (~5 lines changed). Measure collision rate after Bug 2 to confirm it's still needed.
4. **Leave orientation alignment as-is** (Finding 3) — brute-force approaches were tried and regressed; epipolar refinement compensates adequately
5. **Investigate dark cameras** (Finding 4) — separate upstream investigation

Fixes 1-3 are in `src/aquapose/reconstruction/triangulation.py`. Expected outcome: ~9px mean residual, >90% success rate, 0 NaN residuals, <2% collision rate. Finding 3 can be revisited later (Option C: post-epipolar alignment) once Fixes 1-3 are validated.

---

## Files Referenced

- `src/aquapose/reconstruction/triangulation.py` — all bugs and fixes
- `scripts/diagnose_triangulation.py` — diagnostic script (created for this investigation)
- `scripts/diagnose_pipeline.py` — de-inlined in Step 0 (already modified)
- `output/triangulation_diagnostic/report.md` — full diagnostic output
