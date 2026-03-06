---
phase: quick-8
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/tracking/tracker.py
  - tests/unit/tracking/test_tracker.py
autonomous: true
requirements: [QUICK-8]

must_haves:
  truths:
    - "FishTrack.velocity reflects a smoothed average of recent frame-to-frame deltas, not just the last single-frame delta"
    - "Velocity prediction accuracy improves for tracks with noisy per-frame positions"
    - "Coasting prediction still works correctly with smoothed velocity"
    - "Single-view update_position_only still freezes velocity (no smoothing applied)"
    - "Existing tracker behavior is unchanged when velocity_window=1 (backward compatible)"
  artifacts:
    - path: "src/aquapose/tracking/tracker.py"
      provides: "Windowed velocity smoothing in FishTrack"
      contains: "velocity_history"
    - path: "tests/unit/tracking/test_tracker.py"
      provides: "Tests for windowed velocity smoothing"
      contains: "test_windowed_velocity"
  key_links:
    - from: "src/aquapose/tracking/tracker.py"
      to: "FishTrack.update_from_claim"
      via: "velocity_history ring buffer averaging"
      pattern: "velocity_history"
    - from: "src/aquapose/tracking/tracker.py"
      to: "FishTrack.predict"
      via: "self.velocity (now smoothed)"
      pattern: "self\\.velocity"
---

<objective>
Add windowed velocity smoothing to FishTrack so that the velocity used for prediction
is the mean of recent frame-to-frame position deltas rather than a single noisy delta.

Purpose: Single-frame velocity (centroid_3d - prev) is noisy because 3D triangulated
positions have measurement error (especially in Z, which is 132x noisier than XY).
Smoothing over a short window (e.g., 5 frames) produces more stable predictions,
improving track claiming accuracy and reducing ID swaps during close encounters.

Output: Modified tracker.py with velocity_history ring buffer, updated tests.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/tracking/tracker.py
@tests/unit/tracking/test_tracker.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add velocity_history ring buffer and windowed smoothing to FishTrack</name>
  <files>src/aquapose/tracking/tracker.py</files>
  <action>
1. Add a module-level default constant:
   ```python
   DEFAULT_VELOCITY_WINDOW: int = 5
   """Number of recent frame-to-frame deltas to average for velocity smoothing."""
   ```

2. Add `velocity_window` parameter to `FishTracker.__init__` (default `DEFAULT_VELOCITY_WINDOW`),
   store as `self.velocity_window`, and pass it through to `_create_track`.

3. Add two new fields to the `FishTrack` dataclass:
   - `velocity_history: deque[np.ndarray]` with `maxlen=velocity_window` (default 5).
     Each entry is a frame-to-frame delta vector, shape (3,).
   - `velocity_window: int = DEFAULT_VELOCITY_WINDOW`

4. In `FishTrack.update_from_claim`, change the velocity computation block (lines 203-208):
   - Still compute `delta = centroid_3d - prev` as the raw frame delta.
   - Append `delta` to `self.velocity_history`.
   - Set `self.velocity = np.mean(list(self.velocity_history), axis=0).astype(np.float32)`
     (mean of all deltas in the ring buffer).
   - When `len(self.positions) == 0`, set velocity to zeros and clear velocity_history.

5. In `FishTrack.update_position_only`, do NOT append to velocity_history and do NOT
   update self.velocity (velocity is frozen for single-view updates -- this is existing behavior).

6. In `FishTracker._create_track`, pass `velocity_window=self.velocity_window` to the
   `FishTrack` constructor.

7. Update the `FishTrack` docstring to document `velocity_history` and `velocity_window`.

8. Update the `FishTracker` docstring to document the `velocity_window` parameter.

Key constraint: `self.velocity` remains the authoritative velocity used by `predict()`
and `mark_missed()`. The smoothing only changes HOW velocity is computed in
`update_from_claim`, not how it is consumed. This means coasting, prediction, and
damping all work exactly as before with no changes needed.
  </action>
  <verify>
Run `hatch run test tests/unit/tracking/test_tracker.py` -- all 17 existing tests must pass.
Run `hatch run typecheck` on tracker.py -- no new type errors.
  </verify>
  <done>
FishTrack.velocity is computed as the mean of the last N frame-to-frame deltas (default N=5).
All existing tests pass unchanged (backward compatible). velocity_window=1 reproduces old behavior exactly.
  </done>
</task>

<task type="auto">
  <name>Task 2: Add unit tests for windowed velocity smoothing</name>
  <files>tests/unit/tracking/test_tracker.py</files>
  <action>
Add the following tests after the existing test_coasting_velocity_damping test:

1. `test_windowed_velocity_smoothing_averages_deltas`:
   - Create a FishTrack with velocity_window=3.
   - Call update_from_claim 4 times with positions [0,0,0], [1,0,0], [2,0,0], [5,0,0].
   - Deltas are: [1,0,0], [1,0,0], [3,0,0]. Window of 3 holds all three.
   - Assert velocity == mean([1,1,3], axis=0) = [5/3, 0, 0] (atol=1e-5).

2. `test_windowed_velocity_window_1_matches_raw_delta`:
   - Create FishTrack with velocity_window=1.
   - Update with positions [0,0,0], [1,0,0], [2,0,0], [5,0,0].
   - Assert velocity == [3, 0, 0] (last delta only, same as old behavior).

3. `test_windowed_velocity_single_view_does_not_update_history`:
   - Create FishTrack with velocity_window=3.
   - update_from_claim with [0,0,0] then [1,0,0] (velocity=[1,0,0]).
   - Call update_position_only with [2,0,0].
   - Assert velocity is still [1,0,0] (frozen).
   - Assert len(velocity_history) == 1 (not 2).

4. `test_windowed_velocity_prediction_uses_smoothed`:
   - Create FishTrack with velocity_window=3.
   - update_from_claim with [0,0,1], [0.1,0,1], [0.2,0,1], [0.5,0,1].
   - Deltas: [0.1,0,0], [0.1,0,0], [0.3,0,0]. Mean = [1/6, 0, 0].
   - predict() should return last_pos + velocity = [0.5 + 1/6, 0, 1].
   - Assert with atol=1e-5.

5. `test_windowed_velocity_coasting_uses_smoothed`:
   - Create FishTrack with velocity_window=3, velocity_damping=1.0 (no damping for easy math).
   - Feed 4 positions to get smoothed velocity.
   - mark_missed() once.
   - Assert predict() == last_pos + smoothed_velocity (since damping=1.0).
  </action>
  <verify>
Run `hatch run test tests/unit/tracking/test_tracker.py` -- all tests pass including the 5 new ones.
Run `hatch run test tests/unit/tracking/test_associate.py` -- all association tests still pass.
  </verify>
  <done>
5 new tests validate: smoothed averaging, window=1 backward compat, single-view freeze,
prediction integration, and coasting integration. All pass alongside existing 17 tests.
  </done>
</task>

</tasks>

<verification>
1. `hatch run test tests/unit/tracking/` -- all tracker and associate tests pass
2. `hatch run check` -- lint + typecheck clean
3. Manual review: FishTrack with velocity_window=1 produces identical behavior to pre-change code
</verification>

<success_criteria>
- FishTrack.velocity uses windowed mean of recent deltas (default window=5)
- All 17 existing tracker tests pass unchanged
- 5 new tests cover smoothing, backward compat, freeze, prediction, coasting
- No changes to associate.py (velocity is consumed, not produced, there)
- typecheck and lint pass
</success_criteria>

<output>
After completion, create `.planning/quick/8-windowed-velocity-smoothing-for-tracking/8-SUMMARY.md`
</output>
