---
phase: 19-wire-frame-selection
plan: 19
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/training/dataset_assembly.py
  - src/aquapose/training/pseudo_label_cli.py
  - tests/unit/training/test_dataset_assembly.py
autonomous: true
requirements: [TODO-19]

must_haves:
  truths:
    - "assemble_dataset accepts selected_frames dict and filters pseudo-labels by frame index"
    - "CLI assemble command exposes --temporal-step, --diversity-bins, --diversity-max-per-bin flags"
    - "When frame selection flags are active, diagnostic caches are loaded and frame_selection functions are called"
    - "When no frame selection flags are set, behavior is unchanged (no diagnostic cache loading)"
  artifacts:
    - path: "src/aquapose/training/dataset_assembly.py"
      provides: "selected_frames parameter on assemble_dataset, _filter_by_frames helper"
      contains: "selected_frames"
    - path: "src/aquapose/training/pseudo_label_cli.py"
      provides: "CLI flags and wiring for frame selection into assemble command"
      contains: "temporal-step"
    - path: "tests/unit/training/test_dataset_assembly.py"
      provides: "Tests for frame filtering in assemble_dataset"
      contains: "_filter_by_frames"
  key_links:
    - from: "src/aquapose/training/pseudo_label_cli.py"
      to: "src/aquapose/training/frame_selection.py"
      via: "import temporal_subsample, diversity_sample"
      pattern: "from aquapose.training.frame_selection import"
    - from: "src/aquapose/training/pseudo_label_cli.py"
      to: "src/aquapose/training/dataset_assembly.py"
      via: "passes selected_frames to assemble_dataset"
      pattern: "selected_frames="
---

<objective>
Wire the existing but uncalled frame selection functions (temporal_subsample, diversity_sample) into the pseudo-label assembly CLI and dataset_assembly module.

Purpose: Frame selection logic was implemented in frame_selection.py but never connected. The assemble command has no frame selection flags, assemble_dataset has no selected_frames parameter, and curvature-aware diversity sampling never runs.

Output: Working CLI flags on `aquapose pseudo-label assemble` that load diagnostic caches, run frame selection, and filter pseudo-labels by selected frame indices.
</objective>

<execution_context>
@/home/tlancaster6/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlancaster6/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/training/dataset_assembly.py
@src/aquapose/training/pseudo_label_cli.py
@src/aquapose/training/frame_selection.py
@tests/unit/training/test_dataset_assembly.py
@.planning/todos/pending/2026-03-05-wire-frame-selection-into-pseudo-label-assembly-cli.md
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add selected_frames parameter to assemble_dataset with filtering</name>
  <files>src/aquapose/training/dataset_assembly.py, tests/unit/training/test_dataset_assembly.py</files>
  <behavior>
    - Test: assemble_dataset with selected_frames={"run_001": {1, 2}} only includes pseudo-labels whose stem starts with a frame index in the set (frame index parsed as int(stem[:6]))
    - Test: assemble_dataset with selected_frames=None includes all pseudo-labels (no change from current behavior)
    - Test: assemble_dataset with selected_frames for one run_id but not another filters only the specified run
    - Test: _filter_by_frames helper correctly parses frame index from first 6 chars of stem and filters
  </behavior>
  <action>
1. Add a `_filter_by_frames` helper function in `dataset_assembly.py`:
   ```python
   def _filter_by_frames(
       labels: list[dict],
       selected_frames: dict[str, set[int]],
   ) -> list[dict]:
   ```
   For each label, parse frame index as `int(label["stem"][:6])`. Keep the label if its `run_id` is not in `selected_frames` (runs not in the dict are unfiltered, per Phase 65 decision) OR if the frame index is in `selected_frames[run_id]`.

2. Add `selected_frames: dict[str, set[int]] | None = None` parameter to `assemble_dataset()` after `max_frames`.

3. Apply `_filter_by_frames` to `all_pseudo` after confidence/gap-reason filtering but BEFORE `max_frames` cap. Only apply if `selected_frames is not None`.

4. Write tests in `TestAssembleDataset` class and a new `TestFilterByFrames` class in `test_dataset_assembly.py`.
  </action>
  <verify>
    <automated>hatch run test -- tests/unit/training/test_dataset_assembly.py -x</automated>
  </verify>
  <done>assemble_dataset accepts selected_frames dict, filters pseudo-labels by frame index parsed from first 6 chars of stem, and all new and existing tests pass.</done>
</task>

<task type="auto">
  <name>Task 2: Add CLI flags and wire frame selection into assemble command</name>
  <files>src/aquapose/training/pseudo_label_cli.py</files>
  <action>
1. Add three CLI options to the `assemble` command (before `--seed`):
   - `--temporal-step` (int, default=1, help="Take every Nth frame. 1 = no subsampling.")
   - `--diversity-bins` (int, default=5, help="Number of curvature bins for diversity sampling.")
   - `--diversity-max-per-bin` (int, default=None, help="Max frames per curvature bin. None = no diversity cap.")

2. Add these parameters to the `assemble()` function signature.

3. Inside `assemble()`, AFTER creating `run_dir_paths` but BEFORE calling `assemble_dataset`:
   - Check if frame selection is active: `temporal_step > 1 or diversity_max_per_bin is not None`
   - If active, build `selected_frames: dict[str, set[int]]` by iterating over each run_dir:
     a. Dynamically import `load_run_context` from `aquapose.evaluation.runner` (same pattern as `generate` command, line 148)
     b. Load diagnostic cache: `context, _ = load_run_context(run_dir_path)`
     c. If context is None or context.midlines_3d is None, log warning and skip (no filtering for that run)
     d. Build frame_indices as `list(range(len(context.midlines_3d)))`
     e. Import `temporal_subsample`, `diversity_sample` from `aquapose.training.frame_selection`
     f. Apply `temporal_subsample(frame_indices, temporal_step)`
     g. If `diversity_max_per_bin is not None`, apply `diversity_sample(context.midlines_3d, selected_indices, diversity_bins, diversity_max_per_bin, seed)`
     h. Store result: `selected_frames[run_dir_path.name] = set(selected_indices)`
     i. Log the counts: `logger.info("Run %s: %d/%d frames selected", run_dir_path.name, len(selected_indices), len(frame_indices))`
   - If not active, set `selected_frames = None`

4. Pass `selected_frames=selected_frames` to the `assemble_dataset()` call.

5. In the summary output, if frame selection was active, add a line: `click.echo(f"  Frame selection: {sum(len(v) for v in selected_frames.values())} frames across {len(selected_frames)} runs")`
  </action>
  <verify>
    <automated>hatch run test -- tests/unit/training/test_dataset_assembly.py -x && hatch run check</automated>
  </verify>
  <done>CLI `aquapose pseudo-label assemble` accepts --temporal-step, --diversity-bins, --diversity-max-per-bin flags. When active, diagnostic caches are loaded and frame selection filters pseudo-labels. When inactive (defaults), behavior is unchanged. Lint and typecheck pass.</done>
</task>

</tasks>

<verification>
- `hatch run test -- tests/unit/training/test_dataset_assembly.py -x` passes (all existing + new tests)
- `hatch run check` passes (lint + typecheck)
- `aquapose pseudo-label assemble --help` shows --temporal-step, --diversity-bins, --diversity-max-per-bin options
</verification>

<success_criteria>
- frame_selection.py functions (temporal_subsample, diversity_sample) have callers in pseudo_label_cli.py
- assemble_dataset accepts and applies selected_frames filtering
- CLI flags exposed and wired end-to-end
- All tests pass, lint + typecheck clean
</success_criteria>

<output>
After completion, create `.planning/quick/19-wire-frame-selection-into-pseudo-label-a/19-SUMMARY.md`
</output>
