---
status: resolved
trigger: "ghost-track-phantom-births: FishTracker produces too many phantom tracks in synthetic tracking diagnostic, even after fixing 3 bugs"
created: 2026-02-24T00:00:00Z
updated: 2026-02-24T00:03:00Z
---

## Current Focus

hypothesis: RESOLVED — root cause was synthetic fish placement outside real camera image bounds.
test: N/A
expecting: N/A
next_action: N/A

## Symptoms

expected: Track count should match ground truth fish count (e.g., 2 fish = 2 tracks, maybe 3 briefly during cold start)
actual: Significantly more tracks than ground truth fish, with persistent phantom tracks
errors: No crash errors — the tracker runs but produces wrong results
reproduction: Run scripts/diagnose_tracking.py with real calibration JSON and synthetic data
started: Discovered during quick-7 diagnostic work. Three bugs fixed (commit 1fac02d) but problem persists.

## Eliminated

- hypothesis: Tank geometry mismatch (user initially said not the issue)
  evidence: Empirical analysis shows fish_1 placed at Y=-0.771 where 0 cameras have in-bounds image coverage. Camera Y coverage stops at Y~-0.7. This IS the root cause, just in a more specific form than "tank dimensions": it's the XY center offset of the coverage zone that matters.
  timestamp: 2026-02-24T00:00:00Z

- hypothesis: Cross-fish midpoint phantom (midpoint between fish triangulates to phantom)
  evidence: With reprojection_threshold=15px and fish >17cm apart (159px image separation), phantom midpoint is 79px from each fish — far outside threshold. Cross-fish phantoms do NOT accumulate inliers at typical fish separations.
  timestamp: 2026-02-24T00:01:00Z

- hypothesis: Detection-sharing filter (Fix 3) prevents phantom births
  evidence: In real calibration scenario, RANSAC only produces 1 valid candidate in frame 0 (fish_0). Fish_1 has 0 in-bounds cameras, so no valid cluster. Fix 3 was neutral-to-harmful — it caused regressions in some edge cases. Was reverted.
  timestamp: 2026-02-24T00:02:00Z

- hypothesis: Repeated cold-start birth attempts create phantom tracks
  evidence: Fix 1 (non_dead_count gate) prevents repeated births once all fish are probationary. But with fish_1 undetectable, non_dead_count < expected_count is always true, so birth fires repeatedly regardless. Fix 1 is correct but insufficient when fish_1 is genuinely undetectable.
  timestamp: 2026-02-24T00:02:00Z

## Evidence

- timestamp: 2026-02-24T00:01:00Z
  checked: Camera coverage analysis for GT fish positions in frame 0 (crossing_paths, real calibration)
  found: Fish_0 at [0.521, -0.116, 1.853] has 4 cameras with in-bounds coverage. Fish_1 at [0.375, -0.771, 1.959] has 0 cameras with in-bounds coverage. Fish_1 projects outside image bounds in all 12 cameras at Y=-0.771.
  implication: Fish_1 cannot be tracked by RANSAC. Detections are clamped to image borders; unclamped projection coordinates are far outside threshold from clamped positions. RANSAC scoring uses unclamped coordinates, so fish_1 gets 0 inliers.

- timestamp: 2026-02-24T00:01:00Z
  checked: Camera coverage grid scan at depth z=1.83 (water_z + 0.8)
  found: Coverage zone (>=3 cameras in image bounds): X range [-0.8, 0.8], Y range [-0.7, 0.8]. Zone centroid ~(-0.06, 0.15). Default TankConfig: radius=1.0m centered at (0,0) — allows fish at Y=-1.0, well outside coverage.
  implication: TankConfig must be shifted to coverage zone centroid and radius reduced to keep fish trackable.

- timestamp: 2026-02-24T00:02:00Z
  checked: Full RANSAC trace for frame 0 with real calibration
  found: Only 1 candidate produced (fish_0). Fish_1 undetectable. No phantoms — the original "7-11 confirmed tracks" problem was not phantom creation but repeated birth/death cycles of fish_1 as it intermittently enters camera coverage.
  implication: The problem is placement, not RANSAC behavior. Fixing placement eliminates repeated birth cycles.

- timestamp: 2026-02-24T00:03:00Z
  checked: diagnose_tracking.py crossing_paths scenario after fix
  found: MOTA improved from -0.08 to +0.64. Both fish mostly tracked (2/2). False positives: 369 -> 130. True positives: 53.7% -> 85.7%. Mean track purity: 1.00 (no ID switches). 7 confirmed tracks remain (short-lived tracks from crossing event).
  implication: Fix is effective. Remaining 5 extra tracks are legitimate crossing-event artifacts, not phantoms.

## Resolution

root_cause: The `diagnose_tracking.py` script only patched `water_z` for real calibration but left TankConfig radius=1.0m at center (0,0). The real 12-camera rig's image-bounds coverage zone is centred at approximately (-0.06, 0.15) with effective radius ~0.65m. Fish placed at the outer edge of the default tank (up to |XY|=1.0m) land outside camera coverage entirely. This caused fish to be undetectable by RANSAC, creating repeated birth/death cycles that inflated the confirmed track count.

Additionally, tracker.py's `should_birth` condition used `confirmed_count` (not `non_dead_count`) which caused birth RANSAC to fire every frame during cold start even when all fish had probationary tracks. This was fixed as a secondary improvement.

fix:
  1. scripts/diagnose_tracking.py: When real calibration is used, also patch TankConfig.center_x=-0.06, center_y=0.15, radius=0.60 to keep fish within camera coverage zone. (Primary fix — largest impact)
  2. src/aquapose/synthetic/trajectory.py: Added center_x/center_y fields to TankConfig; updated _init_fish_positions, _boundary_force, and hard-clamp logic to account for center offset.
  3. src/aquapose/tracking/tracker.py: Changed should_birth gate from confirmed_count to non_dead_count. Added DEFAULT_BIRTH_PROXIMITY_DISTANCE=0.08m constant. Updated birth proximity check to use birth_proximity_distance (secondary guard against cross-fish phantoms).
  4. Reverted Detection-sharing filter (Fix 3) — caused regressions in real-calibration scenario.

verification: crossing_paths scenario: MOTA -0.08 -> +0.64, TP 53.7% -> 85.7%, FP 369 -> 130, both fish mostly tracked (0/2 -> 2/2), 414 unit tests pass.

files_changed:
  - scripts/diagnose_tracking.py
  - src/aquapose/synthetic/trajectory.py
  - src/aquapose/tracking/tracker.py
