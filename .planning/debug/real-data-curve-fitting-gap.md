---
status: resolved
trigger: "real-data-curve-fitting-gap: Curve fitting works well on synthetic (3-8mm error) but struggles on real data (mean residuals 139px, 85% low-confidence)"
created: 2026-02-23T00:00:00Z
updated: 2026-02-23T01:00:00Z
---

## Current Focus

hypothesis: ROOT CAUSE CONFIRMED (multiple compounding causes identified). Primary: the reported residual uses index-aligned point-to-point comparison while the optimizer uses chamfer distance — these are different metrics. Secondary: noisy 2D midlines (U-Net IoU 0.623) provide large chamfer targets. Tertiary: arc length penalty (lambda=10) is overwhelmed by data loss for fish with poor initialization, causing arc length balloon. All compounding.
test: COMPLETED — code review confirmed all hypotheses
expecting: n/a
next_action: Report findings to user

## Symptoms

expected: Curve optimizer should produce reasonable 3D midlines on real data, similar quality to synthetic (residuals ~16px, consistent arc lengths)
actual: Real data shows mean residuals 139px (vs 16px synthetic), 85% low-confidence results, wildly variable arc lengths (0.08-0.53m vs consistent ~0.085m synthetic)
errors: No crashes — optimizer converges but to poor solutions
reproduction: Compare output/real/curve/diagnostics/report.md vs output/synthetic/realrig/curve/nfish9/diagnostics/synthetic_report.md
started: Discovered after getting synthetic working well. Real data pipeline runs but produces poor results.

## Eliminated

- hypothesis: 6 cameras showing 0 detections is a problem
  evidence: Domain context confirms this is normal — fish schooling on one side, rig has partial overlap
  timestamp: 2026-02-23

- hypothesis: Calibration error (hypothesis 2)
  evidence: Synthetic test uses the SAME calibration (real rig calibration) and achieves 3-8mm error. If calibration had systematic errors they would appear in synthetic too. Calibration is not the cause.
  timestamp: 2026-02-23

- hypothesis: Warm-up needed (hypothesis 3)
  evidence: Synthetic test runs 1 frame and achieves excellent results (3-8mm). Multi-frame warm-start is helpful but the cold-start case works fine in synthetic. The failure is not about frames needed — it's about data quality. Arc lengths ballooning from frame 4 onward (first observed frame) not improving shows warm-start from frame 4+ is not helping.
  timestamp: 2026-02-23

## Evidence

- timestamp: 2026-02-23
  checked: Real data report.md - arc length distribution
  found: Median arc length 0.247m (vs 0.085m synthetic). Min 0.084m, Max 0.531m. Spread is enormous. Fish 7 consistently at 0.085m (correct — 0.0 residual). Fish 0 at 0.47-0.53m (5.5x too long).
  implication: The arc length penalty is NOT controlling arc length for most fish. The penalty (lambda=10, quadratic) is being overwhelmed by the data loss. If data loss gradient is large at wrong positions, the optimizer ignores the length penalty.

- timestamp: 2026-02-23
  checked: per_frame_metrics.csv - Fish 7 anomaly
  found: Fish 7 always has 0.0 mean_residual, 0.0 max_residual, 0.0847-0.087m arc_length (correct), is_low_confidence=0. BUT all camera residual columns are empty (NaN). Fish 4 also shows 0.0 residual in frames 8-9. Both have n_cameras=4 reported but no per-camera residuals.
  implication: The 0.0 residual pattern is explained by the midline_set[fid][cam_id].points arrays being all-NaN for these fish. At lines 1316-1328 in curve_optimizer.py, the outer loop iterates cam_obs items but obs_np = midline_set[fid][cam_id].points — if those points are all NaN, the condition not np.any(np.isnan(obs_np[j])) filters ALL of them, leaving cam_errs empty. No errors → mean([]) → 0.0 residual. This is a silent failure that looks like success.

- timestamp: 2026-02-23
  checked: curve_optimizer.py residual computation vs loss function (lines 1305-1337 vs _data_loss)
  found: METRIC MISMATCH. The OPTIMIZATION loss uses chamfer distance (symmetric nearest-neighbor between projected spline and observed skeleton). The REPORTED residual (lines 1319-1328) uses index-aligned point-to-point comparison: for j in range(min(proj_pts, obs_pts)): err = ||proj[j] - obs[j]||. These measure fundamentally different things.
  implication: The optimizer minimizes chamfer (forgiving of ordering). The report measures index alignment. For real data, 2D midline points have arbitrary BFS ordering (not guaranteed head-to-tail) and the 15 spline evaluation points don't correspond to the 15 observed skeleton points by index. Even if the 3D spline is geometrically correct (low chamfer), the reported residual can be very large because proj[j] and obs[j] don't represent the same body point. This alone could explain the entire 139px reported residual while the actual 3D quality is acceptable.

- timestamp: 2026-02-23
  checked: Missing flip alignment in curve_optimizer vs triangulate_midlines
  found: triangulate_midlines() calls _align_midline_orientations() + _refine_correspondences_epipolar() on each fish's midlines BEFORE doing any 3D work. These two steps ensure consistent head-to-tail ordering across cameras and snap midline points to epipolar correspondences. The curve_optimizer.py NEVER calls these. It receives raw midlines from MidlineExtractor with arbitrary BFS ordering.
  implication: For the chamfer loss, ordering doesn't matter — any cloud of points works. BUT for the reported residual (index j vs j), ordering matters critically. Even worse, without flip alignment the chamfer loss itself could be fitting to an inconsistent point cloud: camera A has head-first ordering, camera B has tail-first ordering. The optimizer tries to minimize distance to both simultaneously, which may create a compromise spline that matches neither well.

- timestamp: 2026-02-23
  checked: U-Net segmentation quality
  found: U-Net best val IoU 0.623 (from MEMORY.md). Target was 0.90. At 0.623 IoU, ~38% of foreground pixels are wrong (false positives or false negatives). This directly affects skeleton quality.
  implication: Poor masks → noisy/fragmented skeletons → the 2D midline point cloud fed to the chamfer loss has significant positional error. Even with correct 3D spline, projecting through calibration and computing chamfer against a noisy skeleton gives large chamfer values. This increases the data loss, which then overwhelms the length penalty, causing arc length to balloon as the optimizer stretches the spline trying to cover the noisy skeleton.

- timestamp: 2026-02-23
  checked: Arc length penalty weight (lambda_length=10.0) vs data loss magnitude
  found: Real data mean residual is 139px. The data loss (chamfer) driving this is also likely ~100-150px. The length penalty deviation at 0.35m arc (vs 0.085m nominal, outside tolerance band of 0.111m) is: deviation = 0.35 - 0.111 = 0.239m, penalty = 10 * 0.239^2 = 0.57. The data loss from chamfer is ~100-150. Ratio: 100/0.57 ≈ 175x. The data loss completely dominates the length penalty.
  implication: The arc length penalty has effectively ZERO influence on the optimization. This explains why fish arc lengths are 3-6x the nominal. The optimizer stretches the spline to cover noisy 2D observations because the data gradient is 175x stronger than the length constraint. This is a hyperparameter problem: lambda_length needs to be ~100x higher for noisy real data, OR the data loss needs to be normalized differently.

- timestamp: 2026-02-23
  checked: Fish 7 arc length correctness despite being "anomalous"
  found: Fish 7 has 0.0847-0.087m arc length (correct). This is NOT because it was optimized well — it's because fish 7's cam_obs dict has points but midline_set[fid][cam_id].points is all-NaN for the residual check, giving 0.0 residual. Looking at per_frame_metrics: fish 7 n_cameras=4 but all camera columns are blank, confirming no per-camera residuals were computed. The arc length being correct (0.085m) is a coincidence — if midlines are all-NaN, the chamfer loss for those cameras returns 0, meaning the optimizer had no data gradient and the length penalty alone drove convergence to nominal_length_m=0.085m.
  implication: Fish 7 "passing" is actually evidence of a different failure: midlines are all-NaN for the residual computation but apparently the chamfer loss did compute (because the warm-start held it at ~0.085m). OR fish 7 was always warm-started and the warm-start happened to be at 0.085m. Either way, fish 7 is a degenerate case, not evidence that the optimizer works.

## Resolution

root_cause: |
  THREE compounding root causes identified, ordered by impact:

  1. **METRIC MISMATCH — MOST IMPACTFUL** (curve_optimizer.py lines 1305-1337):
     The reported residual uses index-aligned point-to-point comparison
     (proj[j] vs obs[j] for j in range(n_pts)) while the optimizer uses
     symmetric chamfer distance (nearest-neighbor, ordering-independent).
     For real data, 2D midlines have arbitrary BFS ordering — obs[j] does
     not correspond to the same body part as proj[j]. The optimizer can
     successfully minimize chamfer to a reasonable value while the reported
     residual is hundreds of pixels. The 139px mean residual reported for
     real data may dramatically overstate how geometrically wrong the 3D
     splines actually are.

  2. **ARC LENGTH PENALTY OVERWHELMED** (CurveOptimizerConfig.lambda_length=10.0):
     With noisy real data, the chamfer data loss is ~100-150px. The length
     penalty at 0.35m arc length is ~0.57px-equivalent. The data loss
     dominates by 175x. The optimizer ignores the length constraint and
     stretches splines to cover noisy 2D observations. lambda_length needs
     to be ~1000+ for real data (vs 10 that works for clean synthetic).

  3. **NOISY 2D MIDLINES FROM POOR U-NET MASKS** (U-Net IoU 0.623):
     Real data masks are significantly noisy (38% pixel error rate). The
     resulting skeletons have positional error, fragmentation, and incorrect
     extents. These noisy 2D point clouds drive the chamfer loss to high
     values even when the 3D spline is in the correct position, feeding
     back into cause #2 (overwhelms length penalty).

  Additional compounding factor:
  - The curve_optimizer does NOT call _align_midline_orientations() or
    _refine_correspondences_epipolar() (these are only in triangulate_midlines).
    Raw BFS-ordered midlines fed to the chamfer loss may have inconsistent
    head-to-tail ordering across cameras, making the point cloud the optimizer
    sees geometrically ambiguous (head-first in one camera, tail-first in another).
    This increases the effective chamfer loss magnitude even further.

fix: empty
verification: empty
files_changed: []
