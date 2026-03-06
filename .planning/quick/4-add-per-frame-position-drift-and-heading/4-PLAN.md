---
phase: quick-4
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/synthetic/fish.py
  - scripts/diagnose_pipeline.py
  - tests/unit/synthetic/test_synthetic.py
autonomous: true
requirements: [QUICK-4]
must_haves:
  truths:
    - "Multi-frame synthetic fish positions change linearly per frame according to velocity"
    - "Multi-frame synthetic fish headings change linearly per frame according to angular_velocity"
    - "Single-frame and zero-drift configs produce identical results to current behavior"
  artifacts:
    - path: "src/aquapose/synthetic/fish.py"
      provides: "FishConfig with velocity and angular_velocity fields, drift logic in generate_synthetic_midline_sets"
      contains: "velocity"
    - path: "tests/unit/synthetic/test_synthetic.py"
      provides: "Tests verifying drift behavior"
      contains: "drift"
  key_links:
    - from: "src/aquapose/synthetic/fish.py"
      to: "generate_fish_3d"
      via: "per-frame drifted FishConfig copy passed to generate_fish_3d"
      pattern: "position.*frame_offset.*velocity"
---

<objective>
Add per-frame position drift (velocity) and heading drift (angular_velocity) to FishConfig so synthetic fish evolve over time instead of remaining static across frames.

Purpose: Enable realistic multi-frame synthetic testing where fish swim through the scene, exercising the tracking and reconstruction pipeline under motion.
Output: Updated FishConfig dataclass, drift logic in generate_synthetic_midline_sets, updated diagnose_pipeline defaults, and drift unit tests.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/synthetic/fish.py
@scripts/diagnose_pipeline.py
@tests/unit/synthetic/test_synthetic.py
@src/aquapose/synthetic/__init__.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add drift fields to FishConfig and apply per-frame drift in generate_synthetic_midline_sets</name>
  <files>src/aquapose/synthetic/fish.py</files>
  <action>
1. Add two new fields to FishConfig dataclass:
   - `velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)` with docstring "Per-frame position displacement in metres (dx, dy, dz)."
   - `angular_velocity: float = 0.0` with docstring "Per-frame heading change in radians."

2. In `generate_synthetic_midline_sets()`, inside the frame loop (line ~307), replace the direct use of `cfg` with a per-frame drifted copy. Before `pts_3d = generate_fish_3d(cfg)`, compute:
   ```python
   # Apply per-frame drift
   bx, by, bz = cfg.position
   drifted_cfg = FishConfig(
       position=(
           bx + frame_offset * cfg.velocity[0],
           by + frame_offset * cfg.velocity[1],
           bz + frame_offset * cfg.velocity[2],
       ),
       heading_rad=cfg.heading_rad + frame_offset * cfg.angular_velocity,
       curvature=cfg.curvature,
       scale=cfg.scale,
       n_points=cfg.n_points,
       velocity=cfg.velocity,
       angular_velocity=cfg.angular_velocity,
   )
   ```
   Then use `drifted_cfg` instead of `cfg` in the `generate_fish_3d(drifted_cfg)` call and `generate_fish_half_widths(n_points=drifted_cfg.n_points, scale=drifted_cfg.scale)` call.

3. Update the FishConfig docstring to mention drift fields.

No changes to __init__.py needed since FishConfig is already exported.
  </action>
  <verify>hatch run test tests/unit/synthetic/test_synthetic.py — all existing tests still pass (drift defaults are zero so behavior is unchanged)</verify>
  <done>FishConfig has velocity and angular_velocity fields; generate_synthetic_midline_sets applies linear drift per frame; all existing tests pass unchanged.</done>
</task>

<task type="auto">
  <name>Task 2: Update diagnose_pipeline.py defaults and add drift unit tests</name>
  <files>scripts/diagnose_pipeline.py, tests/unit/synthetic/test_synthetic.py</files>
  <action>
1. In `scripts/diagnose_pipeline.py` `_run_synthetic()`, update the fish config loop (lines ~328-338) to add drift values. For alternating fish (odd index), add motion; even fish stay still:
   ```python
   for i in range(args.n_fish):
       x_pos = i * 0.1 - (args.n_fish - 1) * 0.05
       curvature = 0.0 if i % 2 == 0 else 15.0
       # Alternate between stationary and drifting fish
       if i % 2 == 0:
           velocity = (0.0, 0.0, 0.0)
           angular_vel = 0.0
       else:
           velocity = (0.002, 0.001, 0.0)
           angular_vel = 0.05
       fish_configs.append(
           FishConfig(
               position=(x_pos, 0.0, 1.25),
               heading_rad=0.0,
               curvature=curvature,
               scale=0.085,
               velocity=velocity,
               angular_velocity=angular_vel,
           )
       )
   ```

2. In `tests/unit/synthetic/test_synthetic.py`, add these tests after the existing MidlineSet tests:

   **test_drift_position_changes_across_frames**: Create a FishConfig with velocity=(0.01, 0.0, 0.0) and angular_velocity=0.0. Generate 5 frames via generate_synthetic_midline_sets with build_fabricated_rig(). For each frame, get the ground truth Midline3D control points centroid (mean of control_points). Assert that the X centroid increases monotonically across frames. Assert frame 0 centroid X is close to initial position X, and frame 4 centroid X is approximately initial_x + 4 * 0.01 (within 1mm tolerance).

   **test_drift_heading_changes_across_frames**: Create two FishConfigs: one with angular_velocity=0.0 (stationary heading) and one with angular_velocity=pi/10. Generate 3 frames. For the rotating fish, assert that the GT control points in frame 0 vs frame 2 have different spatial orientation (e.g., the direction vector from first to last control point rotates). For the stationary fish, assert control points are identical across frames.

   **test_zero_drift_matches_static**: Create a FishConfig with default velocity=(0,0,0) and angular_velocity=0.0. Generate 3 frames. Assert that ground truth Midline3D control_points are identical (np.allclose) across all 3 frames — confirming backward compatibility.
  </action>
  <verify>hatch run test tests/unit/synthetic/test_synthetic.py — all tests pass including new drift tests</verify>
  <done>diagnose_pipeline.py uses drift velocities for odd-indexed synthetic fish; 3 new unit tests verify position drift, heading drift, and zero-drift backward compatibility.</done>
</task>

</tasks>

<verification>
hatch run test tests/unit/synthetic/test_synthetic.py
hatch run check
</verification>

<success_criteria>
- FishConfig has velocity and angular_velocity fields with zero defaults
- Multi-frame synthetic generation produces drifting fish when velocity/angular_velocity are non-zero
- Zero-drift configs produce identical results to previous behavior (backward compatible)
- All existing and new unit tests pass
- diagnose_pipeline.py uses non-zero drift for alternating synthetic fish
</success_criteria>

<output>
After completion, create `.planning/quick/4-add-per-frame-position-drift-and-heading/4-SUMMARY.md`
</output>
