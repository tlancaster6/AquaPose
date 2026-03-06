---
created: 2026-03-05T21:41:51.660Z
title: Wire frame selection into pseudo-label assembly CLI
area: training
files:
  - src/aquapose/training/pseudo_label_cli.py
  - src/aquapose/training/dataset_assembly.py
  - src/aquapose/training/frame_selection.py
---

## Problem

The frame selection functions (`temporal_subsample`, `diversity_sample`, `compute_curvature`) in `frame_selection.py` exist but have zero callers. Plans 65-02 and 65-03 specified:

1. **CLI flags** on `aquapose pseudo-label assemble`: `--temporal-step`, `--diversity-bins`, `--diversity-max-per-bin` (65-02, Task 2, lines 179-182)
2. **`selected_frames` parameter** on `assemble_dataset()` mapping `run_id -> set[int]` of allowed frame indices (65-03, Task 1)
3. **CLI wiring** to load diagnostic caches, run `temporal_subsample`/`diversity_sample`, build the `selected_frames` dict, and pass it to `assemble_dataset` (65-03, Task 1)

None of this was implemented. The `assemble` command has no frame selection flags, `assemble_dataset` has no `selected_frames` parameter, and curvature-aware pose diversity sampling never happens.

## Solution

Follow plans 65-02 Task 2 and 65-03 Task 1:

1. Add `selected_frames: dict[str, set[int]] | None = None` parameter to `assemble_dataset()` with `_filter_by_frames` helper that filters pseudo-labels by `int(stem[:6])` frame index
2. Add `--temporal-step`, `--diversity-bins`, `--diversity-max-per-bin` CLI flags to the `assemble` command
3. When frame selection flags are active, load diagnostic caches via `load_run_context` (dynamic import pattern already used in `generate`), run `temporal_subsample` then `diversity_sample`, build `selected_frames` dict keyed by `run_dir.name`
4. Pass `selected_frames` (or None if no frame selection) to `assemble_dataset()`
5. Add tests for frame filtering in `test_dataset_assembly.py`
