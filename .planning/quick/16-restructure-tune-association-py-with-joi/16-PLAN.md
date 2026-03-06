---
phase: quick-16
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/tune_association.py
autonomous: true
requirements: ["QUICK-16"]
must_haves:
  truths:
    - "ray_distance_threshold and score_min are swept jointly as a 2D grid (7x8 = 56 combos)"
    - "Best pair from joint grid is carried forward into sequential eviction_reproj_threshold sweep"
    - "Secondary stages (leiden_resolution, early_k) sweep sequentially after primary stages"
    - "A 2D results matrix is printed showing yield at each (ray_distance_threshold, score_min) cell"
    - "All parameter ranges match the widened specification"
  artifacts:
    - path: "scripts/tune_association.py"
      provides: "Restructured tuning script with joint grid sweep"
      contains: "_run_joint_grid_sweep"
  key_links:
    - from: "_run_joint_grid_sweep"
      to: "carry_forward"
      via: "returns best (ray_distance_threshold, score_min) pair locked into carry_forward"
      pattern: "carry_forward.*ray_distance_threshold.*score_min"
---

<objective>
Restructure tune_association.py to use a joint 2D grid sweep for ray_distance_threshold x score_min,
then carry the winning pair into sequential sweeps for remaining params. Widen all parameter ranges
for an overnight run.

Purpose: Soft scoring couples ray_distance_threshold and score_min, so sweeping them independently
with carry-forward misses interaction effects. A joint grid captures the true optimum.

Output: Updated scripts/tune_association.py
</objective>

<execution_context>
@/home/tlancaster6/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlancaster6/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@scripts/tune_association.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Restructure tune_association.py with joint grid sweep and widened ranges</name>
  <files>scripts/tune_association.py</files>
  <action>
Modify scripts/tune_association.py with these changes:

1. **Update SWEEP_RANGES** to widened values:
   - ray_distance_threshold: [0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15]
   - score_min: [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
   - eviction_reproj_threshold: [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10]

2. **Update SECONDARY_RANGES** to widened values:
   - leiden_resolution: [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
   - early_k: [5, 10, 15, 20, 25, 30]

3. **Replace PRIMARY_STAGES** list with two constants:
   - JOINT_PARAMS: tuple[str, str] = ("ray_distance_threshold", "score_min")
   - SEQUENTIAL_PRIMARY: list[str] = ["eviction_reproj_threshold"]

4. **Add `_run_joint_grid_sweep` function** with signature:
   ```python
   def _run_joint_grid_sweep(
       param_a: str,
       values_a: list[float],
       param_b: str,
       values_b: list[float],
       yaml_config: Path,
       n_frames: int,
       backend: str,
       outlier_threshold: float,
   ) -> tuple[dict[str, float], list[tuple[dict[str, float], EvalResults, tuple[float, float], dict[str, object]]]]:
   ```
   - Iterates all combinations of values_a x values_b (nested loops, a outer, b inner).
   - For each combo, creates overrides = {param_a: val_a, param_b: val_b}, generates fixture,
     runs evaluation (skip_tier2=True), computes association metrics and score.
   - Prints progress: `[{i}/{total}] {param_a}={val_a}, {param_b}={val_b} ... yield=X/Y (Z%), mean=M px, singleton=S%`
   - Stores results in a list of (overrides_dict, results, score, assoc_metrics).
   - After all combos, sorts by score tuple.
   - Returns (best_overrides_dict, all_results_list).

5. **Add `_print_joint_grid_matrix` function** with signature:
   ```python
   def _print_joint_grid_matrix(
       param_a: str,
       values_a: list[float],
       param_b: str,
       values_b: list[float],
       results: list[tuple[dict[str, float], EvalResults, tuple[float, float], dict[str, object]]],
   ) -> None:
   ```
   - Builds a 2D lookup from the results list keyed by (val_a, val_b).
   - Prints a matrix with param_a values as rows, param_b values as columns.
   - Each cell shows yield percentage (e.g., "67%"). Use right-aligned fixed-width columns.
   - Header row: param_b values. Left column: param_a values.
   - Mark the best cell with an asterisk (e.g., "67%*").
   - Print the matrix header: "2D Grid: {param_a} (rows) x {param_b} (cols) — Yield %"

6. **Update `main()` sweep section** to replace the PRIMARY_STAGES loop:
   - Stage 1: Call `_run_joint_grid_sweep` for ray_distance_threshold x score_min.
     Print stage header "Stage 1: Joint grid sweep ray_distance_threshold x score_min (NxM = T combos)".
     Call `_print_joint_grid_matrix` after the sweep.
     Lock both params into carry_forward from the best pair.
   - Stage 2+: Loop over SEQUENTIAL_PRIMARY (just eviction_reproj_threshold) using existing
     `_run_sweep_stage` with carry_forward.
   - Secondary stages: Keep existing logic using SECONDARY_STAGES and SECONDARY_RANGES (unchanged).

7. **Fix stage numbering**: Joint grid is stage 1, eviction_reproj_threshold is stage 2,
   secondary stages start at 3.

8. **Update top-N collection logic** to include joint grid results:
   - The joint grid returns all combos with scores. Include the top combos (not just the winner)
     as candidates for the top-N full evaluation section.

Keep all existing helper functions (_compute_score, _compute_association_metrics, _print_stage_header,
_print_stage_table, _print_final_report, _print_camera_distribution) unchanged.
Keep all existing CLI arguments unchanged.
Keep the existing baseline evaluation, top-N full evaluation, and final report sections unchanged
(they work on carry_forward dict which is the same shape).
  </action>
  <verify>
    <automated>cd /home/tlancaster6/Projects/AquaPose && python -c "import ast; ast.parse(open('scripts/tune_association.py').read()); print('Syntax OK')" && hatch run check</automated>
  </verify>
  <done>
    - tune_association.py parses without syntax errors
    - Joint grid sweep function exists and iterates all combinations
    - 2D matrix printer shows yield at each (ray_distance_threshold, score_min) cell
    - Widened ranges match specification exactly
    - Flow: joint grid -> lock best pair -> sequential eviction_reproj_threshold -> secondary stages
    - Lint and typecheck pass
  </done>
</task>

</tasks>

<verification>
- `python -c "import ast; ast.parse(open('scripts/tune_association.py').read())"` passes
- `hatch run check` passes (lint + typecheck)
- Script contains `_run_joint_grid_sweep` and `_print_joint_grid_matrix` functions
- SWEEP_RANGES and SECONDARY_RANGES match the widened specification
- PRIMARY_STAGES replaced with JOINT_PARAMS + SEQUENTIAL_PRIMARY
</verification>

<success_criteria>
tune_association.py restructured with joint 2D grid for ray_distance_threshold x score_min,
widened parameter ranges, 2D results matrix output, and sequential carry-forward for remaining params.
Script passes syntax check and hatch run check.
</success_criteria>

<output>
After completion, create `.planning/quick/16-restructure-tune-association-py-with-joi/16-SUMMARY.md`
</output>
