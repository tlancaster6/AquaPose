# Training Module Audit Report

## Summary

Audited 21 source files (~8,400 LOC) in `src/aquapose/training/` and 15 test files (~5,800 LOC) in `tests/unit/training/`. The module is in reasonable shape for research code that grew rapidly across phases 66-73. The core abstractions (SampleStore, geometry, pseudo-labels) are solid and well-tested. The main debt is in three areas: (1) near-identical YOLO training wrappers that should be consolidated, (2) duplicated functions across modules that diverged subtly, and (3) significant test coverage gaps in the CLI and thin-wrapper modules. 14 findings total: 5 structural, 2 interface, 4 test coverage, 2 data flow, 1 code quality.

## Findings

### Category: Structural

#### Finding 1: Three nearly identical YOLO training wrappers

- **Files:** `yolo_obb.py` (108 LOC), `yolo_pose.py` (112 LOC), `yolo_seg.py` (108 LOC)
- **Issue:** These three files are 90%+ identical. The only differences are: (a) `yolo_pose.py` has an extra `rect` parameter, and (b) the model name defaults differ. The entire body -- device detection, yaml validation, YOLO init, `model.train()` call, weight copying -- is copy-pasted.
- **Impact:** Any bug fix or improvement (e.g., adding a new Ultralytics parameter) must be applied in three places. Divergence is guaranteed over time.
- **LOE:** Small (<30min) -- extract a `_train_yolo_common()` function, then each wrapper becomes a 5-line function calling it with model-specific defaults.
- **ROI:** High -- eliminates 200+ LOC of duplication, prevents future bugs.
- **Recommendation:** Create a shared `_train_yolo()` in a common location (e.g., `yolo_common.py` or inline in one of the files). Each public function becomes a thin wrapper passing model-specific defaults.

#### Finding 2: Three nearly identical CLI training commands

- **Files:** `cli.py` lines 17-154 (`yolo_obb`), 201-317 (`seg`), 320-464 (`pose`)
- **Issue:** The three `@train_group.command` handlers follow an identical pattern: resolve project dir, create run dir, setup logging, build cli_args dict, snapshot config, call training function, write summary, register model, print next steps. The only differences are the model_type string, the training function called, and pose having an extra `--rect` flag. This is ~400 LOC of boilerplate.
- **Impact:** Adding a new training option (e.g., `--lr`) requires editing three places. The `seg` command is missing model registration (it was added to `obb` and `pose` but not `seg`), which is a latent bug.
- **LOE:** Medium (30-90min) -- extract a shared `_run_training()` helper that takes model_type and the training callable as parameters.
- **ROI:** High -- reduces ~400 LOC to ~150 LOC, and the missing `seg` registration is a concrete bug.
- **Recommendation:** Extract a shared training orchestration function. Also fix the `seg` command to include `register_trained_model` (currently missing).

#### Finding 3: Duplicated `_LutConfigFromDict` class

- **Files:** `pseudo_label_cli.py` lines 39-53, `prep.py` lines 24-37
- **Issue:** The exact same class (same name, same fields, same docstring pattern) is defined in two files. Both exist to satisfy the `LutConfigLike` protocol without importing from `aquapose.engine.config`.
- **Impact:** Low -- the class is trivial and unlikely to diverge. But it's a clear DRY violation that confuses readers.
- **LOE:** Small (<30min) -- move to a shared location (e.g., `common.py` or a new `_compat.py`).
- **ROI:** Medium -- small effort, small payoff, but easy and clean.
- **Recommendation:** Move to `common.py` as a shared utility class.

#### Finding 4: Duplicated `affine_warp_crop` and `transform_keypoints` functions

- **Files:** `geometry.py` lines 93-172, `coco_convert.py` lines 165-230
- **Issue:** `coco_convert.py` has its own copies of `affine_warp_crop` and `transform_keypoints` that are identical to the versions in `geometry.py`. The `coco_convert.py` file imports `pca_obb`, `format_obb_annotation`, `format_pose_annotation`, and `extrapolate_edge_keypoints` from `geometry.py` but re-implements these two functions locally.
- **Impact:** Any fix to these functions must be applied in both places. The `coco_convert.py` versions are used by the COCO-to-YOLO conversion pipeline; the `geometry.py` versions are used by pseudo-label generation. A divergence could cause subtle differences between manual annotation conversion and pseudo-label generation.
- **LOE:** Small (<30min) -- delete the duplicates from `coco_convert.py` and import from `geometry.py`.
- **ROI:** High -- eliminates a real risk of divergence in a correctness-critical path.
- **Recommendation:** Remove the duplicate functions from `coco_convert.py` and add imports from `geometry.py`. This is a pure cleanup with no behavior change.

#### Finding 5: Duplicated arc-length computation

- **Files:** `coco_convert.py` `compute_arc_length()` (lines 104-128), `pseudo_labels.py` `_compute_arc_length()` (lines 145-167)
- **Issue:** Two implementations of the same algorithm. The `coco_convert.py` version returns `None` for fewer than 2 visible keypoints; the `pseudo_labels.py` version returns `0.0`. The `pseudo_labels.py` version explicitly notes it "matches `coco_convert.compute_arc_length` for consistency" -- but the return types actually differ (`float | None` vs `float`).
- **Impact:** Subtle inconsistency in return type. The `pseudo_labels.py` version is only used internally to compute `lateral_pad`, where `0.0` triggers the floor at 5px, so the divergence is benign in practice. But it's a maintenance hazard.
- **LOE:** Small (<30min) -- consolidate into `geometry.py` with one canonical signature.
- **ROI:** Medium -- low risk today, prevents future confusion.
- **Recommendation:** Move arc-length computation to `geometry.py` and import from both modules. Use a single return convention.

### Category: Interface

#### Finding 6: `SampleStore` internals accessed by CLI code

- **Files:** `data_cli.py` lines 128-133, 147-149, 213-225; `data_cli.py` `_resolve_ids_by_filename` lines 483-495
- **Issue:** CLI code directly calls `sample_store._connect()` to execute raw SQL queries. The `_connect()` method is private (underscore prefix) but is used 4 times in `data_cli.py`. Examples: counting children before upserts (lines 128-133), looking up samples by content hash (lines 147-149), tagging val samples (lines 213-225), and resolving filenames to IDs (lines 487-495).
- **Impact:** Couples the CLI to the database schema. If the schema or connection management changes, the CLI breaks. The `_resolve_ids_by_filename` function iterates ALL parent samples into memory for each filename lookup, which is O(N*M) instead of O(M) with proper SQL.
- **LOE:** Medium (30-90min) -- add proper public methods to SampleStore: `tag()`, `query_by_stem()`, `count_children()`.
- **ROI:** Medium -- the current code works, but it's fragile and the filename resolution is inefficient.
- **Recommendation:** Add `SampleStore.add_tag(sample_id, tag)`, `SampleStore.query_by_stem(stems)`, and `SampleStore.count_children(sample_id)`. The CLI should not execute raw SQL.

#### Finding 7: `store.assemble()` hardcodes "pose" store detection

- **Files:** `store.py` lines 743-751
- **Issue:** `if self.root.name == "pose":` is used to detect whether to add `kpt_shape`/`flip_idx` to `dataset.yaml`. This is fragile -- it depends on the directory naming convention and breaks if the store is at a non-standard path.
- **Impact:** Low in practice (the convention is stable), but it's a code smell. The method mixes dataset assembly logic with format detection.
- **LOE:** Small (<30min) -- pass `model_type` as a parameter to `assemble()`, or detect from label file content.
- **ROI:** Low -- not likely to cause real issues, but easy to fix.
- **Recommendation:** Add an optional `model_type` parameter to `assemble()` to make format detection explicit.

### Category: Test Coverage

#### Finding 8: YOLO training wrappers have trivial tests only

- **Files:** `test_yolo_pose.py` (23 LOC), `test_yolo_seg.py` (23 LOC), no `test_yolo_obb.py`
- **Source files:** `yolo_pose.py` (112 LOC), `yolo_seg.py` (108 LOC), `yolo_obb.py` (108 LOC)
- **Issue:** Tests only verify importability and the `FileNotFoundError` for missing `dataset.yaml`. No test for the weight-copying logic (lines 92-108 in each file), which has edge cases: what if `best.pt` doesn't exist but `last.pt` does? What if neither exists? There is no `test_yolo_obb.py` at all.
- **Impact:** The weight-copying fallback logic is untested. A regression there would silently leave training runs without usable weights.
- **LOE:** Small (<30min) -- add parametric tests for the weight-copying logic. Can mock the YOLO training call.
- **ROI:** Medium -- weight copying is a critical post-training step.
- **Recommendation:** Add tests for weight-copying scenarios: both exist, only best exists, only last exists, neither exists. After consolidating the wrappers (Finding 1), only one test suite is needed.

#### Finding 9: Elastic deformation CLI has a 28-line stub test

- **Files:** `test_elastic_deform_cli.py` (28 LOC)
- **Source files:** `data_cli.py` augmentation path (lines 113-285), `elastic_deform.py` (378 LOC)
- **Issue:** The elastic deform CLI test only covers `parse_pose_label`. The import-with-augmentation codepath in `data_cli.py` (lines 113-285) -- which reads images, calls `generate_variants`, writes temp files, and calls `add_augmented` -- has no integration test. The `generate_variants` function itself is tested in `test_elastic_deform.py` (269 LOC), which is reasonable, but the CLI integration is untested.
- **Impact:** The CLI augmentation path has had multiple bugs (TPS singular matrix, S-curve shape, lateral blur) that were caught manually. An integration test would catch regressions.
- **LOE:** Medium (30-90min) -- needs a fixture with a fake store, sample image, and pose label.
- **ROI:** Medium -- the augmentation path is actively maintained and has had bugs.
- **Recommendation:** Add a CLI integration test that runs `import_cmd` with `--augment` on a synthetic pose dataset.

#### Finding 10: `select_diverse_subset.py` has no dedicated tests

- **Files:** No test file exists for `select_diverse_subset.py` (330 LOC)
- **Issue:** The diversity-maximizing subset selection (both OBB and pose) is completely untested. It has complex logic: temporal binning, curvature quartile stratification, proportional allocation, flex picks, train/val splitting, file copying. This was used in phase 72-73 and is part of the pseudo-label workflow.
- **Impact:** High -- any regression in selection logic would silently produce biased training data.
- **LOE:** Medium (30-90min) -- needs synthetic confidence.json and image/label directories.
- **ROI:** High -- complex logic, high impact, actively used.
- **Recommendation:** Create `test_select_diverse_subset.py` covering: (a) OBB: camera balance, temporal spread, flex allocation; (b) Pose: curvature stratification, val fraction; (c) edge cases: fewer entries than target, single camera.

#### Finding 11: `datasets.py` and `coco_convert.py` have no dedicated tests

- **Files:** No test file for `datasets.py` (270 LOC). `coco_convert.py` (596 LOC) is partially tested via `test_data_cli.py` (the `convert_cmd` test) but has no unit tests for individual functions.
- **Issue:** `datasets.py` contains `CropDataset`, `stratified_split`, and `apply_augmentation` -- a PyTorch Dataset implementation with RLE mask decoding and augmentation. These are untested except indirectly. `coco_convert.py` has `parse_keypoints`, `compute_arc_length`, `temporal_split`, `generate_obb_dataset`, `generate_pose_dataset` -- none have unit tests (only tested indirectly through CLI integration tests).
- **Impact:** `datasets.py` is legacy (Mask R-CNN training) and may not be actively used. `coco_convert.py` is actively used but tested indirectly. Risk is moderate.
- **LOE:** Large (2-4hr) for full coverage of both. Small (<30min) for high-value `coco_convert.py` unit tests only.
- **ROI:** Medium for `coco_convert.py` unit tests, Low for `datasets.py` (likely legacy).
- **Recommendation:** Add unit tests for `coco_convert.py` core functions (`parse_keypoints`, `compute_arc_length`, `temporal_split`). Evaluate whether `datasets.py` is still used; if not, consider deprecation.

### Category: Data Flow

#### Finding 12: `import_cmd` curvature computation is a reimplementation

- **Files:** `data_cli.py` lines 174-196
- **Issue:** The `import_cmd` function in `data_cli.py` manually parses YOLO pose labels to extract keypoints for curvature computation. This is a reimplementation of `elastic_deform.parse_pose_label()` -- but with different assumptions (it doesn't call `parse_pose_label`, parses keypoints inline, uses a different visibility threshold). The code extracts keypoints into `pts` list by manually parsing `kp_tokens` in groups of 3.
- **Impact:** Two codepaths for the same task (parsing YOLO pose keypoints) with subtly different behavior. If `parse_pose_label` is updated (e.g., to handle a new format), the inline version won't benefit.
- **LOE:** Small (<30min) -- replace the inline parsing with a call to `parse_pose_label()`, catching `ValueError` for multi-line labels.
- **ROI:** Medium -- reduces code and eliminates a subtle divergence risk.
- **Recommendation:** Refactor to use `parse_pose_label()` for consistency.

#### Finding 13: `pseudo_label_cli.py` generate command is a monolith

- **Files:** `pseudo_label_cli.py` `generate()` function, lines 124-629
- **Issue:** The `generate` command is a single 500-line function that handles: config loading, calibration loading, projection model construction, LUT loading, tracklet indexing, directory creation, frame iteration, consensus label generation, gap label generation, frame reading, OBB completeness filtering, pose crop writing, confidence sidecar writing, dataset YAML writing, visualization, and summary printing. It has deeply nested loops (3+ levels) and manages 15+ local variables.
- **Impact:** Difficult to test, modify, or debug. Changes to one part (e.g., gap detection) risk breaking unrelated parts. The function cannot be reused outside the CLI context.
- **LOE:** Large (2-4hr) -- extract into a `PseudoLabelGenerator` class or a set of composable functions. The CLI handler becomes a thin driver.
- **ROI:** Medium -- the function works and has been stable since phase 72, but any future changes (e.g., adding a new label source) will be painful.
- **Recommendation:** Extract a `PseudoLabelGenerator` class with methods for each phase: `load_context()`, `generate_consensus_labels()`, `generate_gap_labels()`, `write_outputs()`. The CLI handler orchestrates these calls. This also makes the pipeline testable without mocking Click.

### Category: Code Quality

#### Finding 14: `data_cli.py` `import_cmd` mixes concerns

- **Files:** `data_cli.py` lines 70-299
- **Issue:** The `import_cmd` function is 230 lines and handles: image scanning, sidecar metadata loading, per-sample metadata assembly (from sidecar, curvature, and static JSON), import with source priority, val tagging via raw SQL, cascade deletion tracking, augmentation with temp file lifecycle, and summary reporting. This is the longest Click command handler and mixes data import, metadata enrichment, augmentation, and reporting in a single function.
- **Impact:** Hard to test individual concerns. The val-tagging logic (lines 212-225) directly accesses `_connect()` (see Finding 6). The augmentation path is tightly coupled to the import loop.
- **LOE:** Medium (30-90min) -- extract metadata enrichment and augmentation into separate functions.
- **ROI:** Medium -- improves testability and readability, but not blocking.
- **Recommendation:** Extract `_enrich_metadata()` (sidecar + curvature + static overlay) and `_augment_sample()` (temp file lifecycle + variant generation) as standalone functions. The import loop becomes a clean orchestration of scan, enrich, import, tag, augment.

## Priority Matrix

| # | Finding | LOE | ROI | Category |
|---|---------|-----|-----|----------|
| 1 | Three identical YOLO training wrappers | Small | High | Structural |
| 2 | Three identical CLI training commands (+ seg registration bug) | Medium | High | Structural |
| 4 | Duplicated `affine_warp_crop`/`transform_keypoints` in `coco_convert.py` | Small | High | Structural |
| 10 | `select_diverse_subset.py` has no tests | Medium | High | Test Coverage |
| 5 | Duplicated arc-length computation | Small | Medium | Structural |
| 3 | Duplicated `_LutConfigFromDict` | Small | Medium | Structural |
| 6 | SampleStore internals accessed by CLI | Medium | Medium | Interface |
| 8 | YOLO wrappers have trivial tests only | Small | Medium | Test Coverage |
| 9 | Elastic deform CLI has stub test | Medium | Medium | Test Coverage |
| 12 | `import_cmd` reimplements pose label parsing | Small | Medium | Data Flow |
| 14 | `import_cmd` mixes concerns | Medium | Medium | Code Quality |
| 11 | `datasets.py`/`coco_convert.py` missing unit tests | Large | Medium | Test Coverage |
| 13 | `pseudo_label_cli.py` generate is a monolith | Large | Medium | Data Flow |
| 7 | `store.assemble()` hardcodes pose detection | Small | Low | Interface |

## Recommended Execution Order

### Batch 1: Quick wins, high ROI (estimated ~1.5 hours)

These can be done in a single session and yield the most improvement per minute:

1. **Finding 4** -- Delete duplicate `affine_warp_crop`/`transform_keypoints` from `coco_convert.py`, import from `geometry.py`. Pure cleanup, no behavior change. (<15min)
2. **Finding 5** -- Move arc-length computation to `geometry.py`, import from both consumers. (<15min)
3. **Finding 3** -- Move `_LutConfigFromDict` to `common.py`. (<10min)
4. **Finding 1** -- Consolidate YOLO training wrappers into shared function. (<30min)
5. **Finding 12** -- Replace inline pose label parsing in `import_cmd` with `parse_pose_label()`. (<15min)

### Batch 2: Moderate effort, high value (estimated ~2 hours)

These require more thought but fix real issues:

6. **Finding 2** -- Consolidate CLI training commands + fix missing `seg` registration. (~45min)
7. **Finding 10** -- Write tests for `select_diverse_subset.py`. (~45min)
8. **Finding 8** -- Write weight-copying tests for YOLO wrappers (after consolidation). (~20min)

### Batch 3: Nice to have, do when touching these files (estimated ~3 hours)

Lower urgency, but worth doing when the relevant code is being modified for other reasons:

9. **Finding 6** -- Add public SampleStore methods, stop using `_connect()` in CLI. (~60min)
10. **Finding 9** -- Write elastic deform CLI integration test. (~45min)
11. **Finding 14** -- Extract `_enrich_metadata()` and `_augment_sample()` from `import_cmd`. (~30min)

### Batch 4: Large refactors, do only if needed (estimated ~4 hours)

Only worth pursuing if the relevant code is being extended significantly:

12. **Finding 13** -- Extract `PseudoLabelGenerator` class from the generate command. (~3hr)
13. **Finding 11** -- Write comprehensive tests for `datasets.py` and `coco_convert.py`. (~2hr, but `datasets.py` may be legacy)
14. **Finding 7** -- Add `model_type` parameter to `store.assemble()`. (~15min, trivial but low value)
