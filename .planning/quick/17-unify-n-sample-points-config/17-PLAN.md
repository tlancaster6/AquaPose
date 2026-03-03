---
phase: quick-17
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/engine/config.py
  - src/aquapose/core/reconstruction/utils.py
  - src/aquapose/core/reconstruction/stage.py
  - src/aquapose/core/reconstruction/backends/dlt.py
  - src/aquapose/io/midline_writer.py
  - src/aquapose/synthetic/fish.py
  - src/aquapose/engine/pipeline.py
  - tests/unit/synthetic/test_synthetic.py
  - tests/unit/io/test_midline_writer.py
autonomous: true
requirements: [QUICK-17]
must_haves:
  truths:
    - "PipelineConfig.n_sample_points defaults to 15 (not 10)"
    - "MidlineConfig no longer has n_points field; n_sample_points propagates from top-level only"
    - "N_SAMPLE_POINTS constant removed from utils.py"
    - "ReconstructionConfig has n_sample_points field, propagated from top-level"
    - "DLT backend, synthetic fish, HDF5 writer, and midline stage receive n_sample_points from config, not hardcoded constant"
    - "SPLINE_K, SPLINE_N_CTRL, SPLINE_KNOTS remain as constants in utils.py"
  artifacts:
    - path: "src/aquapose/engine/config.py"
      provides: "PipelineConfig.n_sample_points=15, ReconstructionConfig.n_sample_points=15, no MidlineConfig.n_points"
    - path: "src/aquapose/core/reconstruction/utils.py"
      provides: "SPLINE_K, SPLINE_N_CTRL, SPLINE_KNOTS, MIN_BODY_POINTS (no N_SAMPLE_POINTS)"
  key_links:
    - from: "src/aquapose/engine/config.py"
      to: "load_config propagation"
      via: "propagate n_sample_points to reconstruction.n_sample_points"
      pattern: "rec_kwargs.*n_sample_points"
    - from: "src/aquapose/engine/pipeline.py"
      to: "ReconstructionStage, MidlineStage, Midline3DWriter"
      via: "config.n_sample_points passed through build_stages"
      pattern: "n_sample_points"
---

<objective>
Unify n_sample_points configuration so it flows from a single top-level PipelineConfig.n_sample_points (default 15) to all consumers: midline stage, DLT backend, synthetic fish, and HDF5 writer. Remove the scattered N_SAMPLE_POINTS constant and MidlineConfig.n_points field.

Purpose: Eliminate config duplication and inconsistency (top-level default was 10, MidlineConfig.n_points default was 15, N_SAMPLE_POINTS constant was 15).
Output: Clean single-source-of-truth config propagation for n_sample_points.
</objective>

<execution_context>
@/home/tlancaster6/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlancaster6/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/engine/config.py
@src/aquapose/core/reconstruction/utils.py
@src/aquapose/core/reconstruction/backends/dlt.py
@src/aquapose/core/reconstruction/stage.py
@src/aquapose/io/midline_writer.py
@src/aquapose/synthetic/fish.py
@src/aquapose/engine/pipeline.py
@tests/unit/synthetic/test_synthetic.py
@tests/unit/io/test_midline_writer.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Update config hierarchy and load_config propagation</name>
  <files>src/aquapose/engine/config.py</files>
  <action>
  1. Change `PipelineConfig.n_sample_points` default from `10` to `15`. Update its docstring to say "Default is 15" and remove mention of propagating to midline.n_points.

  2. Remove `n_points` field entirely from `MidlineConfig`. Remove it from the docstring. This field is being replaced by top-level n_sample_points.

  3. Add `n_sample_points: int = 15` field to `ReconstructionConfig`. Docstring: "Number of sample points along each midline for triangulation output. Default 15. Propagated from top-level n_sample_points when not explicitly overridden."

  4. In `load_config()`:
     - Remove the existing `# --- propagate n_sample_points to midline.n_points ---` block (lines 640-642 that set `mid_kwargs["n_points"]`).
     - Add a new propagation block: if `"n_sample_points"` is not already in `rec_kwargs`, set `rec_kwargs["n_sample_points"] = top_kwargs.get("n_sample_points", 15)`.

  5. In the `_RENAME_HINTS` dict, add `"n_points": "n_sample_points (top-level)"` so that any YAML using `midline.n_points` gets a helpful error.
  </action>
  <verify>
    <automated>hatch run python -c "from aquapose.engine.config import PipelineConfig, ReconstructionConfig, MidlineConfig; import dataclasses; fields = {f.name for f in dataclasses.fields(MidlineConfig)}; assert 'n_points' not in fields, 'n_points still in MidlineConfig'; assert PipelineConfig().n_sample_points == 15; assert ReconstructionConfig().n_sample_points == 15; print('OK')"</automated>
  </verify>
  <done>PipelineConfig.n_sample_points defaults to 15, MidlineConfig.n_points removed, ReconstructionConfig.n_sample_points added with propagation from top-level.</done>
</task>

<task type="auto">
  <name>Task 2: Remove N_SAMPLE_POINTS constant and update all consumers</name>
  <files>
    src/aquapose/core/reconstruction/utils.py,
    src/aquapose/core/reconstruction/backends/dlt.py,
    src/aquapose/core/reconstruction/stage.py,
    src/aquapose/io/midline_writer.py,
    src/aquapose/synthetic/fish.py,
    src/aquapose/engine/pipeline.py,
    tests/unit/synthetic/test_synthetic.py,
    tests/unit/io/test_midline_writer.py
  </files>
  <action>
  **utils.py:**
  - Remove `N_SAMPLE_POINTS` constant and its entry from `__all__`.
  - Keep SPLINE_K, SPLINE_N_CTRL, SPLINE_KNOTS, MIN_BODY_POINTS unchanged.

  **dlt.py:**
  - Add `n_sample_points: int = 15` parameter to `DltBackend.__init__()` and `DltBackend.from_models()`.
  - Store as `self._n_sample_points`.
  - In `_reconstruct_fish`, the current code derives `n_body_points = len(first_midline.points)` from input midlines -- this is correct and does NOT need changing (it reads from actual data). The `n_sample_points` parameter is NOT used by DltBackend internally (it produces output sized by the input midline point count). So actually, DLT backend does not need an n_sample_points parameter -- the output size matches the input midline size, which is already controlled upstream. **Skip adding n_sample_points to DltBackend** -- it is unnecessary.

  **stage.py (ReconstructionStage):**
  - Add `n_sample_points: int = 15` parameter to `__init__`. Store as `self._n_sample_points`.
  - Pass `n_sample_points` is NOT needed for the backend (see above). But store it for potential future use and to match ReconstructionConfig. Actually, since the backend doesn't use it and the stage doesn't use it, skip adding it to stage.py as well.

  **Revised approach for dlt.py and stage.py:** After analysis, DLT backend derives body point count from input midlines (`len(first_midline.points)`), and the stage just delegates to the backend. Neither needs an explicit n_sample_points parameter -- the value flows through the midline data itself. No changes needed to dlt.py or stage.py.

  **midline_writer.py:**
  - Remove the import of `N_SAMPLE_POINTS` from `aquapose.core.reconstruction.utils`.
  - Change `Midline3DWriter.__init__` default: `n_sample_points: int = 15` (hardcoded default instead of constant reference). The actual value will be passed from config at construction time.
  - Update the module docstring's shape comments from `(N, max_fish, 15)` to `(N, max_fish, n_sample_points)`.

  **synthetic/fish.py:**
  - Remove `from aquapose.core.reconstruction.utils import N_SAMPLE_POINTS` (keep SPLINE_K and SPLINE_KNOTS imports).
  - Change `FishConfig.n_points: int = N_SAMPLE_POINTS` to `FishConfig.n_points: int = 15`.
  - Change `generate_fish_half_widths(n_points: int = N_SAMPLE_POINTS, ...)` to `generate_fish_half_widths(n_points: int = 15, ...)`.
  - Update docstrings to say "Default 15" instead of "Default N_SAMPLE_POINTS".

  **pipeline.py (build_stages):**
  - The midline stage is currently constructed with `n_points=config.midline.n_points`. Since we removed `MidlineConfig.n_points`, change to `n_points=config.n_sample_points`.
  - The synthetic stage is already using `n_points=config.n_sample_points` -- no change needed.
  - The reconstruction stage construction does not pass n_sample_points -- no change needed (backend reads from input data).
  - Check if the HDF5 writer / diagnostic observer constructs Midline3DWriter and pass n_sample_points from config. Search for `Midline3DWriter(` usages and update if needed.

  **tests/unit/synthetic/test_synthetic.py:**
  - Remove `N_SAMPLE_POINTS` from the import line (currently imports from `aquapose.synthetic.fish`).
  - Replace all uses of `N_SAMPLE_POINTS` in assertions with the literal `15`.

  **tests/unit/io/test_midline_writer.py:**
  - Remove `N_SAMPLE_POINTS` from the import line (currently imports from `aquapose.core.reconstruction.utils`).
  - Replace all uses of `N_SAMPLE_POINTS` with the literal `15`.
  - Keep SPLINE_K and SPLINE_KNOTS imports.
  </action>
  <verify>
    <automated>hatch run test -x</automated>
  </verify>
  <done>N_SAMPLE_POINTS constant removed from utils.py. All consumers updated to use either config-provided value or hardcoded default of 15. All tests pass.</done>
</task>

</tasks>

<verification>
1. `hatch run test -x` -- all unit tests pass
2. `hatch run check` -- lint and typecheck pass
3. `grep -r "N_SAMPLE_POINTS" src/` returns no hits
4. `grep -r "N_SAMPLE_POINTS" tests/` returns no hits
5. `hatch run python -c "from aquapose.engine.config import load_config; c = load_config(None, cli_overrides={'n_animals': 3}); assert c.n_sample_points == 15; assert c.reconstruction.n_sample_points == 15; print('propagation OK')"` passes
</verification>

<success_criteria>
- PipelineConfig.n_sample_points defaults to 15
- MidlineConfig has no n_points field
- ReconstructionConfig has n_sample_points field, propagated from top-level in load_config
- N_SAMPLE_POINTS constant is gone from utils.py and all imports
- SPLINE_K, SPLINE_N_CTRL, SPLINE_KNOTS remain as constants
- All existing tests pass
- Lint and typecheck pass
</success_criteria>

<output>
After completion, create `.planning/quick/17-unify-n-sample-points-config/17-SUMMARY.md`
</output>
