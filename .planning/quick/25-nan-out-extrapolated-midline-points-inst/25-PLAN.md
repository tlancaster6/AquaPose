---
phase: 25-nan-out-extrapolated-midline-points
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/core/midline/backends/pose_estimation.py
  - tests/unit/core/midline/test_pose_estimation_backend.py
autonomous: true
requirements: [QUICK-25]
must_haves:
  truths:
    - "Midline points outside the visible keypoint t-range are NaN with confidence 0"
    - "Midline points within the visible keypoint t-range are unchanged (interpolated normally)"
    - "Downstream DLT triangulation treats NaN points as invalid (weight=0)"
    - "All existing tests still pass"
  artifacts:
    - path: "src/aquapose/core/midline/backends/pose_estimation.py"
      provides: "_keypoints_to_midline with NaN extrapolation masking"
      contains: "np.nan"
    - path: "tests/unit/core/midline/test_pose_estimation_backend.py"
      provides: "Tests for NaN-out behavior on extrapolated points"
      contains: "test_keypoints_to_midline_nan"
  key_links:
    - from: "_keypoints_to_midline"
      to: "DLT triangulation"
      via: "Midline2D.point_confidence = 0 for NaN points"
      pattern: "valid_nc.*False"
---

<objective>
Modify `_keypoints_to_midline` to NaN-out extrapolated midline points instead of clamping confidence, so that cameras with dropped endpoints abstain from triangulation rather than voting for a straight-line extrapolation.

Purpose: Prevent rigid-looking 3D fish tails/noses caused by extrapolated geometry receiving ~0.77 DLT weight from clamped confidence values.
Output: Updated `_keypoints_to_midline` function and new unit tests.
</objective>

<execution_context>
@/home/tlancaster6/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlancaster6/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/core/midline/backends/pose_estimation.py
@tests/unit/core/midline/test_pose_estimation_backend.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add tests for NaN-out extrapolation behavior</name>
  <files>tests/unit/core/midline/test_pose_estimation_backend.py</files>
  <behavior>
    - test_keypoints_to_midline_nan_outside_visible_range: When t_values span [0.2, 0.8] (nose and tail dropped), output points at t < 0.2 and t > 0.8 should be NaN with confidence 0.0, while points in [0.2, 0.8] should be finite with positive confidence.
    - test_keypoints_to_midline_full_range_no_nan: When t_values span [0.0, 1.0] (all keypoints visible), no NaN in output -- all points finite with positive confidence.
    - test_keypoints_to_midline_tail_only_dropped: When t_values span [0.0, 0.6] (tail dropped at t=0.8, 1.0), only points beyond t=0.6 are NaN with confidence 0, points at t <= 0.6 are normal.
    - test_keypoints_to_midline_shape_preserved: Output shape is still (n_points, 2) and (n_points,) regardless of NaN presence.
  </behavior>
  <action>
Add four new test functions to the `_keypoints_to_midline unit tests` section of the test file. Use the existing helper patterns. For the partial-range tests, construct keypoints with t_values that don't start at 0.0 or don't end at 1.0 (simulating dropped nose/tail). Assert that output xy values outside the visible t-range are np.nan and confidence is 0.0, while interior points remain finite.

Key test setup for "nose and tail dropped":
- 4 keypoints at t_values = [0.2, 0.4, 0.6, 0.8]
- n_points = 15 (t_eval = linspace(0, 1, 15))
- Points at t < 0.2 (indices 0, 1) and t > 0.8 (indices 13, 14) should be NaN/conf=0
- Points at 0.2 <= t <= 0.8 should be finite with positive confidence
  </action>
  <verify>
    <automated>cd /home/tlancaster6/Projects/AquaPose && hatch run test tests/unit/core/midline/test_pose_estimation_backend.py -x</automated>
  </verify>
  <done>Four new tests exist and initially fail (RED), then pass after Task 2 (GREEN).</done>
</task>

<task type="auto">
  <name>Task 2: Modify _keypoints_to_midline to NaN-out extrapolated points</name>
  <files>src/aquapose/core/midline/backends/pose_estimation.py</files>
  <action>
Modify `_keypoints_to_midline` to NaN-out points outside the visible keypoint range:

1. Keep the existing `interp_x`, `interp_y`, `interp_c` interpolation (which still extrapolates for the x/y computation).

2. After computing `x_out`, `y_out`, `conf_out`, add a masking step:
   ```python
   # NaN-out extrapolated points beyond the visible keypoint range.
   # Cameras that didn't see the endpoint abstain from triangulation
   # rather than voting for a straight-line extrapolation.
   t_first = t_values[0]
   t_last = t_values[-1]
   extrapolated = (t_eval < t_first) | (t_eval > t_last)
   x_out[extrapolated] = np.nan
   y_out[extrapolated] = np.nan
   conf_out[extrapolated] = 0.0
   ```

3. Place this block between the `conf_out = ...` line (75) and the `xy = np.stack(...)` line (77).

4. No changes needed to the interp1d calls themselves -- let them extrapolate, then mask. This is simpler than changing fill_value and avoids scipy behavior differences.

5. Update the docstring to document that points outside the visible keypoint range are set to NaN with confidence 0.
  </action>
  <verify>
    <automated>cd /home/tlancaster6/Projects/AquaPose && hatch run test tests/unit/core/midline/test_pose_estimation_backend.py -x</automated>
  </verify>
  <done>All tests pass including the four new NaN-out tests. Points outside `[t_first_visible, t_last_visible]` are NaN with confidence 0. Points inside the range are unchanged. Full test suite passes with `hatch run test`.</done>
</task>

</tasks>

<verification>
- `hatch run test tests/unit/core/midline/test_pose_estimation_backend.py` -- all tests pass
- `hatch run test` -- full test suite passes (no regressions)
- `hatch run check` -- lint + typecheck passes
</verification>

<success_criteria>
- `_keypoints_to_midline` returns NaN xy and 0.0 confidence for points outside the visible keypoint t-range
- When all keypoints are visible (t spans [0, 1]), output is identical to previous behavior (no NaN)
- 15-point array shape is preserved regardless of NaN presence
- All existing and new tests pass
</success_criteria>

<output>
After completion, create `.planning/quick/25-nan-out-extrapolated-midline-points-inst/25-SUMMARY.md`
</output>
