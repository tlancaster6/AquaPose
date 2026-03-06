---
phase: quick-15
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/core/association/scoring.py
  - src/aquapose/engine/config.py
  - tests/unit/core/association/test_scoring.py
  - scripts/tune_association.py
autonomous: true
requirements: [SOFT-SCORE, REMOVE-GHOST]

must_haves:
  truths:
    - "score_tracklet_pair uses linear soft kernel (1 - dist/threshold) instead of binary inlier counting"
    - "Ghost penalty computation is completely removed from scoring"
    - "ghost_pixel_threshold removed from AssociationConfig and AssociationConfigLike"
    - "All existing tests pass with updated logic"
  artifacts:
    - path: "src/aquapose/core/association/scoring.py"
      provides: "Soft scoring kernel, no ghost penalty"
      contains: "1.0 - (dist / config.ray_distance_threshold)"
    - path: "src/aquapose/engine/config.py"
      provides: "AssociationConfig without ghost_pixel_threshold"
    - path: "tests/unit/core/association/test_scoring.py"
      provides: "Updated tests for soft scoring behavior"
  key_links:
    - from: "src/aquapose/core/association/scoring.py"
      to: "src/aquapose/engine/config.py"
      via: "AssociationConfigLike protocol"
      pattern: "ghost_pixel_threshold should NOT appear"
---

<objective>
Replace binary inlier counting with soft linear scoring kernel in association scoring, and remove the ghost penalty entirely.

Purpose: Binary inlier counting discards distance magnitude -- a correct pair at 0.41cm and a wrong pair at 1.86cm both score ~0.98, making community detection impossible. Soft scoring translates the 4.5x distance difference into a ~2.3x score difference. Ghost penalty is ineffective with densely packed fish.
Output: Updated scoring.py with soft kernel, cleaned config, updated tests.
</objective>

<execution_context>
@/home/tlancaster6/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlancaster6/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/core/association/scoring.py
@src/aquapose/engine/config.py
@tests/unit/core/association/test_scoring.py
@scripts/tune_association.py
@.planning/inbox/association_scoring_diagnosis.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Replace binary inlier counting with soft kernel and remove ghost penalty</name>
  <files>src/aquapose/core/association/scoring.py, src/aquapose/engine/config.py</files>
  <action>
In `scoring.py`, modify `score_tracklet_pair()`:

1. Replace binary inlier counting with soft linear kernel. Change the loop body from:
   ```python
   if dist < config.ray_distance_threshold:
       inlier_count += 1
       # ... ghost penalty code ...
   ```
   To:
   ```python
   if dist < config.ray_distance_threshold:
       score_sum += 1.0 - (dist / config.ray_distance_threshold)
   ```
   Initialize `score_sum = 0.0` before the loop (replacing `inlier_count = 0`).

2. Remove ALL ghost penalty code:
   - Remove `from aquapose.calibration.luts import ghost_point_lookup` import
   - Remove `ghost_ratios: list[float] = []` and `scoring_cameras` set
   - Remove the entire ghost penalty block inside the inlier branch (the `mid_tensor`, `visibility`, `other_cams`, `n_visible_other`, `n_negative` logic)
   - Remove `mean_ghost = ...` line after the loop

3. Update early termination to track whether ANY contribution was made. Change from checking `inlier_count == 0` to checking `score_sum == 0.0` (or equivalently, keep a simple counter for early termination only).

4. Compute final score as `f = score_sum / t_shared` instead of `f = inlier_count / t_shared`. The combined score becomes `score = f * w` (no ghost term).

5. Remove `detections` and `inverse_lut` parameters from `score_tracklet_pair()` signature since ghost penalty no longer needs them. Update the docstring accordingly.

6. Remove `detections` and `inverse_lut` parameters from `score_all_pairs()` signature. Update its call to `score_tracklet_pair()` and remove the `ghost_point_lookup` import. Update docstring.

7. Update `AssociationConfigLike` protocol: remove `ghost_pixel_threshold` attribute.

8. Update module docstring to remove references to "ghost-point penalties".

9. Update `__all__` if needed (it should remain the same since no public names change).

In `config.py`, modify `AssociationConfig`:
- Remove `ghost_pixel_threshold: float = 50.0` field
- Remove the ghost_pixel_threshold line from the class docstring

In `scripts/tune_association.py`:
- Remove `ghost_pixel_threshold` from `PARAM_GRID` dict
- Remove `ghost_pixel_threshold` from the ordered keys list
- If `detections` or `inverse_lut` are passed to scoring functions, update those calls

In `src/aquapose/core/association/__init__.py`: no changes needed (no ghost symbols exported).
  </action>
  <verify>
    <automated>cd /home/tlancaster6/Projects/AquaPose && hatch run check</automated>
  </verify>
  <done>
  - score_tracklet_pair uses `1.0 - (dist / threshold)` accumulation
  - No ghost_point_lookup import, no ghost_ratios, no ghost penalty computation
  - ghost_pixel_threshold removed from config and protocol
  - Lint and typecheck pass
  </done>
</task>

<task type="auto">
  <name>Task 2: Update tests for soft scoring and remove ghost penalty tests</name>
  <files>tests/unit/core/association/test_scoring.py</files>
  <action>
Update `tests/unit/core/association/test_scoring.py`:

1. Update `MockAssociationConfig`: remove `ghost_pixel_threshold` field.

2. Update `TestScoreTrackletPair` tests to match new signature (no `inverse_lut`, no `detections` params):
   - `test_perfect_match`: Remove `inv_lut`, `detections` args from `score_tracklet_pair()` call. The score should now be `f * w` where f=1.0 (all frames contribute 1.0 since dist=0) and w=20/100=0.2, so score=0.2. Assert `score == pytest.approx(0.2)`.
   - `test_no_overlap`: Remove `inv_lut` arg. Keep assertion score==0.0.
   - `test_below_t_min`: Remove `inv_lut` arg. Keep assertion score==0.0.
   - `test_early_termination`: Remove `inv_lut` arg. Keep assertion score==0.0.

3. Remove the entire `TestGhostPenalty` class -- ghost penalty no longer exists.

4. Remove `MockInverseLUT` class (no longer needed by scoring tests). Keep `MockInverseLUTNonAdjacent` since it is used by `TestScoreAllPairs`.

5. Update `TestScoreAllPairs` tests:
   - `test_respects_camera_adjacency`: Remove `detections` arg from `score_all_pairs()` call. Keep adjacency assertions.
   - `test_filters_by_score_min`: Remove `inv_lut` arg from `score_all_pairs()` call (but still needs `MockInverseLUTNonAdjacent` or equivalent for overlap graph -- check if `score_all_pairs` still needs `inverse_lut` for `camera_overlap_graph`).

   NOTE: `score_all_pairs` still needs `inverse_lut` for `camera_overlap_graph()` -- only `detections` is removed from its signature, and `inverse_lut` is NOT passed through to `score_tracklet_pair` anymore but is still used for overlap graph. So keep `inverse_lut` in `score_all_pairs` signature, just remove `detections`.

6. Add a new test `test_soft_scoring_distance_sensitivity` to verify that closer rays produce higher scores than farther rays (the core improvement):
   - Create two MockForwardLUT pairs: one producing rays that intersect at dist=0 (perfect), another producing rays with dist close to but under threshold (e.g., dist ~0.02 with threshold 0.03).
   - Assert score_close > score_far, validating the soft kernel differentiates distances.

7. Remove unused imports (`MockInverseLUT` if removed).
  </action>
  <verify>
    <automated>cd /home/tlancaster6/Projects/AquaPose && hatch run test -- tests/unit/core/association/test_scoring.py -v</automated>
  </verify>
  <done>
  - All scoring tests pass with updated signatures
  - TestGhostPenalty class removed
  - New test validates soft kernel distance sensitivity
  - No references to ghost_pixel_threshold in test file
  </done>
</task>

</tasks>

<verification>
```bash
cd /home/tlancaster6/Projects/AquaPose && hatch run test -- tests/unit/core/association/ -v
cd /home/tlancaster6/Projects/AquaPose && hatch run check
```

Grep verification:
```bash
# Ghost penalty fully removed
grep -rn "ghost_pixel_threshold\|ghost_ratio\|ghost_point_lookup\|n_visible_other\|ghost_ratios" src/aquapose/core/association/scoring.py src/aquapose/engine/config.py tests/unit/core/association/test_scoring.py
# Should return nothing

# Soft kernel present
grep -n "1.0 - (dist" src/aquapose/core/association/scoring.py
# Should show the soft kernel line
```
</verification>

<success_criteria>
- Soft linear kernel replaces binary inlier counting in score_tracklet_pair
- Ghost penalty fully removed (no ghost_point_lookup, no ghost_ratios, no ghost_pixel_threshold in config)
- All association tests pass
- Lint + typecheck pass
- New test confirms distance sensitivity of soft kernel
</success_criteria>

<output>
After completion, create `.planning/quick/15-replace-binary-inlier-counting-with-soft/15-SUMMARY.md`
</output>
