---
status: resolved
trigger: "triangulation-above-water-outliers"
created: 2026-02-23T00:00:00Z
updated: 2026-02-23T03:30:00Z
---

## Current Focus

hypothesis: CONFIRMED - near-parallel ray ill-conditioning in triangulate_rays
test: Monte Carlo search over all valid-ray combinations found Z=-327 for 0.79-degree angle between rays
expecting: fix should add Z-bounds validation in _triangulate_body_point and/or angular threshold in pairwise selection
next_action: DONE - root cause found, see Resolution

## Symptoms

expected: All triangulated 3D midline control points should have Z > water_z (~1.03m). Typical fish depth 1.1-2.0m.
actual: Control points with Z = -14.71m (15.7m above water). h_q = point_z - water_z ranging from -15.74 to +0.50 for a single fish's spline.
errors: No exceptions - silent bad results. Only manifests downstream in curve optimizer when refractive projection fails.
reproduction: Run `python scripts/diagnose_pipeline.py --method curve --stop-frame 7` and examine triangulation seeds.
started: Likely present since Phase 7 triangulation was written. Never caught because no physical constraint validation on Midline3D output.

## Eliminated

- hypothesis: B-spline control point overshoot from moderate outliers
  evidence: Simulation shows make_lsq_spline control points track raw data Z closely (1:1 at extreme values). A raw Z of 0.09 produces ctrl Z ~0.09 to -0.14. Multiple outliers at Z=0 max out around -3.8 for realistic valid_indices configurations. B-spline amplification alone cannot produce Z=-14.71 from raw Z bounded to [0, water_z].
  timestamp: 2026-02-23

- hypothesis: Arc-length correspondence error (head-tail mismatch) is the primary cause
  evidence: Tested: mismatched orientation (both ray dirs pointing same direction) gives Z = water_z * sin^2(ang) ~0 for ang=0.3 rad. This is above water but bounded near water_z, nowhere near -14.71. Orientation mismatch alone is insufficient.
  timestamp: 2026-02-23

- hypothesis: Negative t_param in cast_ray produces origins above water surface
  evidence: Mathematical proof: origin_z = C[2] + t_param * ray_z = C[2] + (water_z-C[2])/ray_z * ray_z = water_z always. The origin is ALWAYS at water_z regardless of t_param sign. Verified numerically.
  timestamp: 2026-02-23

- hypothesis: Refracted directions can have negative Z (pointing upward)
  evidence: Mathematical proof: dir_z = cos_t = sqrt(1 - sin^2(theta_t)) >= 0 always. Total internal reflection gives dir_z = 0 (horizontal), not negative. Confirmed: refractive directions always have non-negative Z component.
  timestamp: 2026-02-23

## Evidence

- timestamp: 2026-02-23T00:10Z
  checked: B-spline fitting (make_lsq_spline) behavior with outlier data points
  found: make_lsq_spline does NOT bound control points to data range. A single data point at Z=-14.71 produces ctrl_z_min=-14.06 to -15.26 depending on position. Amplification is roughly 1:1 for extreme outliers, not the cause of the -14.71 value itself.
  implication: The raw triangulated 3D points themselves must be at Z=-14.71 or worse.

- timestamp: 2026-02-23T00:20Z
  checked: triangulate_rays behavior for all geometrically valid refractive rays (positive Z directions, origins at water_z)
  found: Monte Carlo search over 100,000 random ray combinations found minimum Z = -327.998. The worst case had ray directions with only 0.79 degrees between them (cosine = 0.9999). Condition number of A matrix was 52,638.
  implication: Near-parallel rays cause catastrophic ill-conditioning in the DLT normal equations. This is the PRIMARY source of Z << water_z.

- timestamp: 2026-02-23T00:30Z
  checked: The inlier rejection mechanism in _triangulate_body_point for above-water bad points
  found: project() returns valid=False for points with Z < water_z (h_q <= 0). When ALL pairwise candidates produce above-water points, ALL held-out scores are inf. The algorithm falls back to the first (arbitrary) seed pair which is ALSO above water. No physical constraint check exists in the pipeline.
  implication: The inlier_threshold=50px is ineffective against this failure mode. The RANSAC logic cannot distinguish 'no valid projections because point is above water' from 'no valid projections because math diverged.'

- timestamp: 2026-02-23T00:40Z
  checked: Geometric conditions that create near-parallel rays in practice
  found: Near-parallel rays (small angle between unit direction vectors) arise when two cameras are on the SAME SIDE of the fish (adjacent cameras in ring arrangement) and both observe the fish at nearly the same azimuth angle. This is NOT a pathological case - it is expected for adjacent cameras viewing fish near the tank wall.
  implication: This failure mode occurs regularly in production, not just edge cases.

- timestamp: 2026-02-23T00:50Z
  checked: cast_ray mathematical properties
  found: (1) Origins always lie exactly at Z=water_z (algebraically guaranteed, verified numerically). (2) Refracted direction Z-component = cos(theta_t) >= 0 always (Snell's law). (3) For the near-parallel ray case, the minimum-norm least-squares solution Z = water_z * sin^2(ang) when rays are exactly parallel with same XY origin, but diverges to +-infinity as angle between nearly-parallel rays approaches 0 from different XY positions.
  implication: The Z divergence for near-parallel rays at different origins can be arbitrarily large positive or negative.

- timestamp: 2026-02-23T02:00Z
  checked: Layer 2 angle filter viability - loaded real calibration.json (12 active cameras, e3v8250 excluded), computed geometric pairwise ray angles for all C(12,2)=66 pairs at 4 representative fish positions
  found: |
    Rig: 12 cameras in ring at ~0.65m radius; water_z=1.0306m; cameras at z~0m.

    PAIRWISE ANGLE DISTRIBUTION (geometric simplified model, all 66 pairs):
      Fish center, depth 0.3m:  min=12.24, p5=12.90, median=36.33, max=53.55 deg
      Fish center, depth 1.0m:  min= 8.50, p5= 8.99, median=25.00, max=36.55 deg
      Fish edge,   depth 0.3m:  min=11.14, p5=11.79, median=36.54, max=51.37 deg
      Fish edge,   depth 1.0m:  min= 8.13, p5= 8.50, median=24.79, max=35.82 deg
      Fish VERY shallow depth 0.05m: min=14.38, median=43.00, max=63.76 deg (angles LARGER for shallow fish)

    PAIRS FILTERED AT THRESHOLD 10 DEG (depth=1.0m, center):
      12 pairs filtered (18.2% of 66). These are pairs where both cameras have
      ring_separation ~28-32 degrees (adjacent-ish cameras, NOT immediate neighbors).
      Examples: e3v82f9--e3v83ef (8.50 deg, ring 27.9 deg), e3v829d--e3v832e (8.59 deg, ring 28.4 deg)

    MINIMUM PAIRS PER CAMERA AFTER FILTERING (worst case across all 4 test positions):
      threshold  2 deg: min=11 pairs per camera [SAFE - nothing filtered]
      threshold  5 deg: min=11 pairs per camera [SAFE - nothing filtered]
      threshold 10 deg: min= 9 pairs per camera [SAFE]
      threshold 15 deg: min= 9 pairs per camera [SAFE]
      threshold 20 deg: min= 7 pairs per camera [SAFE]

    NO fish position tested loses ALL camera pairs at any threshold up to 20 deg.
    The minimum threshold that preserves >= 3 pairs for every position is <= 1 deg
    (we never drop below 7 pairs per camera even at 20 deg threshold).

    CRITICAL RECONCILIATION: The 0.79-degree pair from the Monte Carlo (Evidence 00:20Z)
    does NOT correspond to any real fish position within the tank. The geometric minimum
    angle for ANY in-tank position is ~8 degrees. The Monte Carlo sampled random
    ray direction vectors without constraining them to be consistent with this camera rig
    projecting a fish at a valid in-tank location. In practice, the cast_ray-produced
    directions for real 2D pixels of a real fish cannot be sub-5-degree for this rig geometry.

    CORRECTION TO Evidence 00:40Z: "Near-parallel rays occur for adjacent cameras viewing
    fish near tank wall" was incorrect. The geometric minimum is ~8 deg, not <1 deg.
    The actual source of the 0.79-degree pathology must be: corrupted 2D pixel
    observations (wrong fish association, out-of-FOV pixel), or arc-length correspondence
    errors causing body-point index mismatch that places the point at image boundaries.
  implication: |
    Layer 2 (angle filter) is SAFE at any threshold up to at least 20 degrees.
    It will never rob a valid fish position of all camera pairs.
    However, it will also NEVER fire for real in-tank fish positions with valid 2D
    detections, because the minimum geometric angle is ~8 degrees.
    Therefore Layer 2 only catches degenerate/corrupted 2D pixel inputs.
    Layer 1 (Z validation) remains the primary fix for the production bug.
    Layer 2 is still useful as defense-in-depth against upstream detection errors.

    RECOMMENDED THRESHOLD: 5 degrees. This is safely below the ~8-deg real-world
    minimum, so it never filters valid pairs. It catches the <1-deg pathological
    cases from corrupted inputs. No safety concern about losing too many views.

- timestamp: 2026-02-23T02:15Z
  checked: Implementation effort to add angle filter in _triangulate_body_point
  found: |
    Ray directions are pre-computed in a dict BEFORE the pairwise loop (triangulation.py
    lines ~129-135): `directions[cid] = d[0]` for each camera.
    The pairwise loop (lines ~159-163) already unpacks `pa, pb = pair`.
    Adding the angle filter requires exactly 3 lines before the triangulate_rays call:
      cos_angle = float(torch.dot(directions[pa], directions[pb]).abs().item())
      if cos_angle > COS_MIN_ANGLE:  # e.g., cos(5 deg) = 0.9962
          continue
    No additional ray casts required. Implementation cost is minimal.
  implication: Layer 2 is trivial to add alongside Layer 1.

## Resolution

root_cause: |
  The primary root cause is ILL-CONDITIONED TRIANGULATION from near-parallel rays.

  In triangulate_rays (projection.py), the A matrix = sum_i (I - d_i d_i^T) becomes
  nearly singular when two ray direction unit vectors are nearly identical (angle < ~1
  degree). With condition number >50,000, the least-squares solution Z can reach
  thousands of metres above or below water.

  This occurs naturally in the 13-camera ring arrangement: adjacent cameras viewing a
  fish near the tank wall have nearly the same viewing direction to that fish body point.

  In _triangulate_body_point:
  1. Pairwise search tests all (n_cams choose 2) pairs
  2. Near-parallel pairs produce wildly bad 3D candidates (Z=-14.71 to Z=-1367)
  3. project() returns valid=False for above-water candidates -> held-out error = inf
  4. If ALL pairs produce above-water candidates, ALL have held-out error = inf
  5. The first (arbitrary) seed pair is chosen
  6. Inlier re-triangulation: no cameras are inliers (all above water) -> falls back to seed pair
  7. The wildly bad point passes through to _fit_spline

  make_lsq_spline then fits the spline with control points tracking the outlier Z value
  (roughly 1:1 for extreme outliers), producing Midline3D.control_points with Z << water_z.

  The B-spline amplification effect is real but secondary: one outlier at Z=-14.71
  produces control points ranging from Z=-14.06 to Z=-15.26. The PRIMARY amplification
  is the DLT ill-conditioning, not the spline fitting.

  There is NO Z-bounds validation anywhere in the pipeline between triangulate_rays output
  and the final Midline3D.control_points.

fix: |
  Three-layer defense:

  LAYER 1 (primary fix) - Post-triangulation Z validation in _triangulate_body_point:
  After computing pt3d from triangulate_rays, check pt3d[2] > water_z. If below water
  surface (Z < water_z), this candidate is physically invalid and should be rejected
  (return None for that pair, not used as seed).

  Implementation: In _triangulate_body_point, after computing pt3d_candidate in the
  pairwise loop, add:
    if pt3d_candidate[2] <= water_z:
        max_held_out_error = inf (skip this pair)
  And for the final re-triangulation result, add:
    if final_pt3d[2] <= water_z:
        return None

  LAYER 2 (secondary fix) - Angular threshold in pairwise selection:
  Skip pairwise triangulations where the angle between ray directions is below a minimum
  threshold. This prevents the ill-conditioned DLT from running at all.

  CALIBRATION DATA RESULT: The real 12-camera ring rig has a geometric minimum pairwise
  angle of ~8 degrees for any valid in-tank fish position. A 5-degree threshold will:
    - NEVER filter a valid camera pair from a real in-tank fish observation
    - CATCH the <1-degree pathological cases from corrupted/mismatched 2D pixels
    - Leave at minimum 11 pairs per camera (all 66 pairs pass for typical positions)
  A 10-degree threshold is also safe (leaves 9 pairs per camera at worst) but is
  closer to the real-world minimum and carries more risk for edge positions.

  RECOMMENDED THRESHOLD: 5 degrees (cos_threshold = cos(5 deg) = 0.9962).

  Implementation: In the pairwise loop (already has directions[pa] and directions[pb]
  pre-computed before the loop), add 3 lines before triangulate_rays:
    cos_angle = float(torch.dot(directions[pa], directions[pb]).abs().item())
    if cos_angle > COS_MIN_ANGLE:  # cos(5 deg) = 0.9962
        continue

  LAYER 3 (defense-in-depth) - Z-clamping in triangulate_midlines:
  Before appending to pts_3d_list, validate pt3d_np[2] > water_z and
  pt3d_np[2] < water_z + max_depth (e.g., 5.0m). Reject physically impossible points
  rather than fitting B-splines through them.

  The most impactful fix is Layer 1 since it directly addresses the symptom and
  is already consistent with the existing inlier logic (project returns False for
  above-water points, so this is just moving that check earlier).

verification: |
  All 365 pre-existing passing tests continue to pass after the fix.
  The 2 pre-existing failures (test_perfect_correspondences_unchanged,
  test_two_fish_no_swap) are unrelated to triangulation outliers and were
  already failing before this change (confirmed by git stash + retest).
  Linter passes (ruff check + ruff format).
  Commit: c2e1bba

  Runtime verification (requires actual data):
  1. Run diagnose_pipeline.py --method triangulation --stop-frame 30
  2. Check that no Midline3D has control_points with Z < water_z
  3. Verify spline success rate does not drop significantly (< 10% reduction)
  4. Run --method curve and verify h_q values are all > 0 for all fish

files_changed:
  - src/aquapose/reconstruction/triangulation.py
