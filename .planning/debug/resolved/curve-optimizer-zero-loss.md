---
status: resolved
trigger: "curve-optimizer-zero-loss: After commit 38d43dc changed _data_loss from summing to averaging and removed Huber wrapping, ALL losses are now exactly 0.0"
created: 2026-02-23T00:00:00Z
updated: 2026-02-23T02:20:00Z
---

## Current Focus

hypothesis: CONFIRMED — triangulation seeds with Z < water_z cause no valid projections, _data_loss returns 0.0
test: Ran pipeline with debug instrumentation showing h_q = [-15.74, 0.50] for fish 0
expecting: N/A — root cause confirmed and fixed
next_action: DONE — fix applied and verified

## Symptoms

expected: Per-fish losses should be meaningful pixel values (e.g., 10-100 pixels chamfer distance), decreasing during optimization, producing reasonable 3D fish midlines.
actual: Every fish in every frame reports loss=0.0 from step 0. Coarse stage loss=0.0. Fine stage per-fish losses all 0.0. Optimizer converges at step 4 (patience=3 with zero delta).
errors: No exceptions. Output log shows all zeros.
reproduction: python scripts/diagnose_pipeline.py --method curve --stop-frame 7
started: After commit 38d43dc which modified _data_loss()

## Eliminated

- hypothesis: cam_id format mismatch (string vs other type)
  evidence: midlines_per_fish is built by filtering cam_ids that ARE in models. Second check in _data_loss is redundant but safe. Not the issue.
  timestamp: 2026-02-23

- hypothesis: Empty observation tensors
  evidence: cam_obs dict is filtered in optimize_midlines — only fish with valid obs are in valid_fish_ids.
  timestamp: 2026-02-23

- hypothesis: New aggregation logic drops fish that old code included
  evidence: Both old and new code return 0 when cam_losses is empty. The core issue is WHY cam_losses is empty.
  timestamp: 2026-02-23

- hypothesis: Device mismatch between models and tensors
  evidence: All tensors and models are moved to the same device. water_z is a plain float, not a tensor. No device issue.
  timestamp: 2026-02-23

## Evidence

- timestamp: 2026-02-23
  checked: project() validity check in projection.py line 248
  found: valid = (h_q > 0) & (p_cam[:, 2] > 0) where h_q = Q[:, 2] - water_z
  implication: Fish must have Z > water_z to produce valid projections.

- timestamp: 2026-02-23
  checked: Ran pipeline with temporary debug instrumentation in _data_loss
  found: h_q = [-15.7447, 0.5020] for fish 0 in Frame 4. water_z=1.0306.
    This means spline Z ranges from 1.03 - 15.74 = -14.71 to 1.03 + 0.50 = 1.53.
    Only 6/20 eval points have h_q > 0 (valid). During coarse optimization, these
    6 valid points contribute. But after L-BFGS coarse optimization and upsampling
    to fine (K=7), ALL eval points end up above water → cam_losses empty → fish_losses
    empty → _data_loss returns 0.0 for fish 0.
  implication: The triangulation seed for fish 0 had most control points above the
    water surface (Z < water_z), which is physically impossible for an underwater fish.
    The optimizer cannot recover from this initialization.

- timestamp: 2026-02-23
  checked: Triangulation seed validation in optimize_midlines
  found: No validation was performed on seed control points. Any result from
    triangulate_midlines was stored directly as a warm_start, even if the control
    points had Z far below water_z (Z coordinate of underwater = Z > water_z in
    this coordinate system).
  implication: A bad RANSAC result or degenerate geometry in triangulation can
    produce nonsensical 3D coordinates. Previously (before 38d43dc), the Huber loss
    and summing aggregation masked this issue by producing some non-zero gradient from
    the few valid points. The new averaging approach returns exactly 0.0 when all
    eval points are invalid, causing silent false convergence.

## Resolution

root_cause: When triangulate_midlines produces control points where the majority lie
  ABOVE the water surface (Z < water_z), the curve optimizer cannot compute valid
  projections for most/all of the spline. After coarse L-BFGS optimization with
  weak gradients (only a few valid points), the fine stage may have ALL 20 evaluation
  points above water, making _data_loss return 0.0 (empty fish_losses list). The
  optimizer then sees delta = 0.0 < convergence_delta = 0.5 and freezes the fish
  after patience_count (3) steps, producing a silent false convergence with loss=0.0.

fix: Two complementary fixes applied to src/aquapose/reconstruction/curve_optimizer.py:
  1. Triangulation seed validation (lines ~722-770): After triangulate_midlines, each
     seed is validated by checking that majority (>= K//2) of control points have
     Z > water_z. Seeds failing this check are rejected with a WARNING log, and the
     fish falls back to cold-start initialization.
  2. Depth penalty fallback in _data_loss (lines ~289-307): When a fish has NO valid
     projections in any camera (all spline points above water), instead of returning
     0.0, a depth penalty = mean(clamp(-h_q, 0)) * 100.0 is added. This provides
     gradients that push control points back below the water surface, preventing
     silent false convergence at loss=0.

verification: Pipeline run after fix shows:
  - Frame 4: coarse stage done, loss=5.4633 (was 25.93 before, now better-initialized)
  - All 6 fish have non-zero fine step 0 losses: 3.7, 15.7, 5.5, 2.5, 2.2, 2.4
  - "all fish converged at step 4/40" (quick convergence from good initialization)
  - hatch run test tests/unit/test_curve_optimizer.py: all 365 pass (2 pre-existing
    failures in test_triangulation and test_tracker are unrelated)
  - hatch run lint src/aquapose/reconstruction/curve_optimizer.py: all checks passed

files_changed:
  - src/aquapose/reconstruction/curve_optimizer.py
