# Phase 77: Training Module Code Quality - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Consolidate duplicated code in `src/aquapose/training/`, fix the `seg` CLI missing model registration bug, and add missing tests. Implements audit findings 1-5, 8, 10, 12 from the training module audit (`.planning/quick/24-audit-recent-pseudo-label-and-training-d/24-AUDIT-REPORT.md`). No new features, no behavioral changes — pure refactoring and test coverage.

</domain>

<decisions>
## Implementation Decisions

### YOLO training wrapper consolidation (Findings 1, 2)
- Create a single `train_yolo()` function in a new `yolo_training.py` file
- `train_yolo()` takes a `model_type` parameter ("obb", "seg", "pose") that controls defaults
- Pose-specific parameters (e.g. `rect`) are explicit optional kwargs on `train_yolo()`, not a defaults dict
- Delete `yolo_obb.py`, `yolo_seg.py`, `yolo_pose.py` entirely — no thin wrapper files
- Provide `train_yolo_obb`, `train_yolo_seg`, `train_yolo_pose` as convenience aliases in the same file
- Update `__init__.py` to export from `yolo_training.py`

### CLI command consolidation (Finding 2)
- Extract a shared `_run_training()` orchestrator function that handles: run dir creation, logging setup, config snapshot, training call, summary writing, model registration, next steps printing
- Keep 3 separate Click commands (`train obb`, `train seg`, `train pose`) — no CLI interface change
- Each command becomes ~10 lines of option parsing + one `_run_training()` call
- This automatically fixes the `seg` command missing `register_trained_model` (concrete bug)

### Shared function deduplication (Findings 3, 4, 5, 12)
- `affine_warp_crop` and `transform_keypoints`: keep canonical versions in `geometry.py`, delete duplicates from `coco_convert.py`, import from `geometry.py`
- `compute_arc_length`: consolidate into `geometry.py`, import from both `coco_convert.py` and `pseudo_labels.py`
- `_LutConfigFromDict`: move to `common.py`, import from both `pseudo_label_cli.py` and `prep.py`
- Inline pose label parsing in `data_cli.py`: replace with call to `parse_pose_label()` from `elastic_deform.py`, catch `ValueError` for multi-line labels. `parse_pose_label` stays in `elastic_deform.py`

### Arc-length return convention (Finding 5)
- Canonical `compute_arc_length()` in `geometry.py` returns `0.0` (not `None`) when fewer than `min_visible` keypoints are visible
- Return type is `float`, not `float | None`
- Callers that previously checked `is None` should check `== 0.0` instead

### Claude's Discretion
- Internal organization of `yolo_training.py` (helper placement, constant naming)
- Exact parameter forwarding structure in `_run_training()`
- Test fixture design details

</decisions>

<specifics>
## Specific Ideas

- The `common.py` file already exists with `EarlyStopping`, `MetricsLogger`, `save_best_and_last`, `make_loader` — these may be legacy (pre-Ultralytics). Adding `_LutConfigFromDict` there is fine; don't refactor the existing contents.
- The `_run_training()` orchestrator should use a try/except around `register_trained_model` (matching the current obb/pose pattern) so registration failures are warnings, not crashes.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `geometry.py`: Already exports `pca_obb`, `format_obb_annotation`, `format_pose_annotation`, `extrapolate_edge_keypoints` — natural home for consolidated functions
- `common.py`: Shared training utilities (EarlyStopping, MetricsLogger) — natural home for `_LutConfigFromDict`
- `run_manager.py`: Has `create_run_dir`, `snapshot_config`, `write_summary`, `print_next_steps`, `register_trained_model` — all used by the CLI orchestrator
- `elastic_deform.py`: Has `parse_pose_label()` — reuse for data_cli.py curvature computation

### Established Patterns
- Training functions accept `Path` for data_dir/output_dir, return `Path` to best weights
- CLI commands use `@click.pass_context` and resolve project dir via `get_project_dir(ctx)`
- Lazy imports (`from ultralytics import YOLO`) inside function bodies to avoid import-time GPU init
- The training subsystem must not import from `engine/` (enforced by pre-commit AST boundary)

### Integration Points
- `__init__.py` must be updated when renaming/removing public symbols
- `cli.py` imports training functions via relative imports (`from .yolo_obb import train_yolo_obb`)
- Tests in `tests/unit/training/` mirror source structure

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 77-training-module-code-quality*
*Context gathered: 2026-03-09*
