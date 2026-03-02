# Phase 35: Codebase Cleanup - Research

**Researched:** 2026-03-01
**Domain:** Python codebase refactoring — removing custom model implementations, maintaining pipeline runability
**Confidence:** HIGH (all findings from direct codebase inspection)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Phase Boundary**
Remove custom model implementations (U-Net, custom _PoseModel, SAM2 pseudo-labeler, MOG2 detector) and their associated training commands. Leave the backend orchestration layer intact — `segment_then_extract` and `direct_pose` survive as approaches, they just lose their custom model code. Stub in no-op YOLO replacements so the pipeline stays runnable. Consolidate surviving shared utilities.

**Critical clarification:** CLEAN-03 does NOT mean "delete the midline backends." It means "remove the custom model implementations from within the backends." The segment_then_extract approach (segment → skeletonize → arc-length sample) and the direct_pose approach (keypoints → spline) both survive. Phase 37 wires in YOLO-seg and YOLO-pose as their new model providers. ROADMAP.md success criteria must be updated to reflect this.

**Shared code triage**
- Keep utilities if any non-removed code imports them; delete only truly orphaned code
- Extract reusable pieces from `segmentation/` package (e.g., crop helpers) to a shared location before deleting the rest
- Keep shared training infrastructure (base datasets, utilities) in `training/`; only remove `train_unet.py` and `train_pose.py`
- Opportunistic cleanup: also remove clearly dead adjacent code discovered during removal
- Verify whether top-level `reconstruction/`, `tracking/`, `visualization/` directories are legacy duplicates of their `core/` counterparts — remove if truly redundant and doable without major churn

**Two-pass structure**
- **Pass 1 — Separate and delete**: Extract potentially reusable utilities, then delete removed modules
- **Pass 2 — Audit survivors**: For everything preserved: (1) judge keep-as-shared vs fold-into-callers, (2) delete if actually unneeded, (3) verify correctness. Consolidate survivors into a clear shared location with clean import paths.

**Test disposition**
- Tests for removed code: delete with the code
- Integration/E2E tests referencing removed models: update to use remaining backends (not delete)
- Note: segment_then_extract and direct_pose are still valid backends — tests may just need model references updated

**Migration messaging**
- Generic "unknown" errors, not migration hints (e.g., "Unknown detector_kind: 'mog2'. Available: yolo, yolo_obb")
- Strict config validation — unknown fields or invalid backend names raise errors at config load time, no silent ignoring
- Update `init-config` templates and example YAML files to remove references to deleted models

**Removal verification**
- Tests must pass AND manual `aquapose run` smoke test with valid config
- Stub both `yolo_seg` and `yolo_pose` as no-op backends so the pipeline executes end-to-end without errors (empty/null midline results are acceptable)
- Atomic commits: one commit per CLEAN requirement for easy bisection

**ROADMAP.md corrections**
- Update CLEAN-03 success criteria: "custom model code removed from midline backends" instead of "backends removed"
- Update any Phase 37 references that imply segment_then_extract/direct_pose don't exist

### Claude's Discretion
- No-op stub output format (empty Midline2D vs skip entirely) — pick based on what downstream stages handle gracefully
- Exact shared utility relocation target (e.g., `core/utils/` or similar)
- Order of removal within the atomic commit structure
- Whether top-level `reconstruction/`, `tracking/`, `visualization/` are actually legacy duplicates worth removing

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CLEAN-01 | All custom U-Net model code removed (`segmentation/model.py`, `training/_UNet`, `_PoseModel`, `BinaryMaskDataset`) | Files fully mapped; cascade imports identified in training/unet.py, training/pose.py, training/viz.py, core/midline/backends/segment_then_extract.py, core/midline/backends/direct_pose.py |
| CLEAN-02 | SAM2 pseudo-label pipeline removed — only path from raw video to training data is COCO JSON → NDJSON | `segmentation/pseudo_labeler.py` fully mapped; cascade imports identified in segmentation/__init__.py, tests/unit/segmentation/test_pseudo_labeler.py, tests/integration/segmentation/test_yolo_sam_integration.py |
| CLEAN-03 | Custom model code removed from midline backends (NOT the backends themselves); backends stubbed with no-op YOLO placeholders | `core/midline/backends/segment_then_extract.py` and `direct_pose.py` imports of UNetSegmentor/_PoseModel mapped; backend registry and orchestration layer survives |
| CLEAN-04 | MOG2 detection backend removed; only `yolo` and `yolo_obb` remain | `MOG2Detector` class identified in `segmentation/detector.py`; `make_detector` already in `segmentation/detector.py` (NOT `core/detection/backends/`); detection backend registry in `core/detection/backends/__init__.py` already only has yolo/yolo_obb |
| CLEAN-05 | `train_unet` and `train_pose` CLI commands removed from `aquapose train` group | `training/cli.py` has `unet` and `pose` subcommands; `training/unet.py` and `training/pose.py` to be deleted; `training/__init__.py` exports need pruning |
</phase_requirements>

## Summary

Phase 35 is a surgical deletion phase. The codebase has been fully mapped and every symbol-to-file relationship is known. The work falls into five independent atomic removal units (one per CLEAN requirement), each touching a well-scoped set of files with no hidden cross-dependencies.

The key architectural insight is that the detection and midline stages have a two-tier structure: (1) a `core/` pipeline layer with a proper backend registry and Stage Protocol, and (2) a legacy `segmentation/detector.py` file that contains both the surviving `YOLODetector` and the to-be-deleted `MOG2Detector`. The `core/detection/backends/__init__.py` registry already only registers `yolo` and `yolo_obb` — MOG2 was never wired into the core pipeline registry, only into the old `make_detector()` factory. CLEAN-04 is therefore narrowly scoped to removing the `MOG2Detector` class from `segmentation/detector.py` and updating `make_detector()` to reject `"mog2"`.

The top-level `reconstruction/`, `tracking/`, and `visualization/` directories are confirmed NOT legacy duplicates. They are actively imported by `core/`, `engine/`, `io/`, and tests. They must be preserved.

The `SegmentationResult` dataclass in `segmentation/model.py` is only imported by `segmentation/__init__.py` — no external non-test code uses it. It can be deleted with `model.py`. The `Detection` dataclass in `segmentation/detector.py` is used pervasively and must survive.

**Primary recommendation:** Proceed with five atomic commits in CLEAN-01 → CLEAN-02 → CLEAN-04 → CLEAN-05 → CLEAN-03 order. CLEAN-03 last because it requires the no-op stubs that need the most thought about output format.

## Standard Stack

### Core (already in use — no new dependencies)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| click | existing | CLI definition — `train_group`, `@click.group`, `@click.command` | Already used for all CLI commands |
| pytest | existing | Test framework for verifying removals | Already used project-wide |
| hatch | existing | Test runner (`hatch run test`) | Project standard per CLAUDE.md |

### No New Dependencies
This phase introduces no new libraries. It is purely a deletion + stub phase.

## Architecture Patterns

### Current Structure (what exists)

```
src/aquapose/
├── segmentation/
│   ├── crop.py              # SURVIVE — used by core/, reconstruction/, tests
│   ├── detector.py          # PARTIAL — keep YOLODetector + Detection; delete MOG2Detector
│   ├── model.py             # DELETE (entire file: _UNet, UNetSegmentor, MaskRCNNSegmentor, SegmentationResult)
│   ├── pseudo_labeler.py    # DELETE (entire file: SAMPseudoLabeler, FrameAnnotation, AnnotatedFrame, etc.)
│   └── __init__.py          # UPDATE — remove model.py and pseudo_labeler.py exports
├── training/
│   ├── cli.py               # UPDATE — remove 'unet' and 'pose' subcommands
│   ├── common.py            # SURVIVE — EarlyStopping, MetricsLogger, make_loader, save_best_and_last
│   ├── datasets.py          # PARTIAL — delete BinaryMaskDataset; keep _CocoDataset, CropDataset, stratified_split, apply_augmentation
│   ├── pose.py              # DELETE (entire file: _PoseModel, KeypointDataset, train_pose, etc.)
│   ├── prep.py              # SURVIVE — prep_group CLI (calibrate-keypoints command)
│   ├── unet.py              # DELETE (entire file: train_unet and helpers)
│   ├── viz.py               # PARTIAL — delete save_unet_augmented_grid, save_unet_val_grid; keep save_pose_augmented_grid, save_pose_val_grid only if pose.py callers are gone
│   ├── yolo_obb.py          # SURVIVE — train_yolo_obb
│   └── __init__.py          # UPDATE — remove deleted exports
├── core/
│   └── midline/
│       └── backends/
│           ├── __init__.py               # UPDATE — update get_backend() docstring/registry to reflect stubs
│           ├── segment_then_extract.py   # UPDATE — remove UNetSegmentor import; replace segmentation block with no-op stub
│           └── direct_pose.py            # UPDATE — remove _PoseModel import; replace model block with no-op stub
```

### Confirmed Survivors (do not touch)
```
src/aquapose/
├── reconstruction/          # SURVIVE — NOT a legacy duplicate; actively imported by core/, engine/, io/, tests
├── tracking/                # SURVIVE — NOT a legacy duplicate; ocsort_wrapper.py imported by core/tracking/stage.py
├── visualization/           # SURVIVE — NOT a legacy duplicate; imported by engine/observers, tests
├── segmentation/crop.py     # SURVIVE — used everywhere (extract_affine_crop, invert_affine_points, CropRegion)
├── training/common.py       # SURVIVE — EarlyStopping, MetricsLogger used by remaining training code
├── training/datasets.py     # PARTIAL SURVIVE — _CocoDataset base class, stratified_split survive for Phase 36
├── training/yolo_obb.py     # SURVIVE — existing working YOLO-OBB training wrapper
└── training/prep.py         # SURVIVE — calibrate-keypoints CLI command
```

### Pattern 1: No-Op Backend Stub

The `SegmentThenExtractBackend` and `DirectPoseBackend` constructors currently load models eagerly. After removing the model code, they become stubs that return `AnnotatedDetection` with `midline=None` for every detection.

**What downstream handles:** The association stage, tracking stage, and reconstruction stage all accept `midline=None` (already handled in production when extraction fails). A no-op stub is functionally equivalent to a backend where all midline extractions fail — valid, non-crashing behavior.

**Stub pattern (no-op for segment_then_extract):**
```python
class SegmentThenExtractBackend:
    """Segment-then-extract midline backend (stub — awaiting YOLO-seg model in Phase 37)."""

    def __init__(self, **kwargs: Any) -> None:
        logger.warning(
            "SegmentThenExtractBackend: no segmentation model loaded — "
            "all midlines will be None until Phase 37 wires in YOLO-seg."
        )

    def process_frame(self, frame_idx, frame_dets, frames, camera_ids):
        annotated = {}
        for cam_id in camera_ids:
            annotated[cam_id] = [
                AnnotatedDetection(
                    detection=det,
                    mask=None,
                    crop_region=None,
                    midline=None,
                    camera_id=cam_id,
                    frame_index=frame_idx,
                )
                for det in frame_dets.get(cam_id, [])
            ]
        return annotated
```

### Pattern 2: Factory with Clear Unknown-Kind Error

`make_detector()` in `segmentation/detector.py` currently accepts `"mog2"`. After CLEAN-04, it should reject it:

```python
def make_detector(kind: str, **kwargs: Any) -> YOLODetector:
    if kind == "yolo":
        return YOLODetector(**kwargs)
    raise ValueError(
        f"Unknown detector_kind: {kind!r}. Available: 'yolo', 'yolo_obb'"
    )
```

Note: `"yolo_obb"` is handled by `core/detection/backends/__init__.py`, not by `make_detector()`. The `make_detector` function should document this split.

### Pattern 3: Atomic Commit Structure

Per user decision: one commit per CLEAN requirement. Recommended order:

1. **CLEAN-01** — Delete `segmentation/model.py` + cascade: `training/unet.py`, fix `training/pose.py` (remove _UNet import, delete _PoseModel), fix `training/viz.py`, fix `core/midline/backends/segment_then_extract.py`, fix `segmentation/__init__.py`, fix `training/__init__.py`, fix `training/datasets.py` (delete BinaryMaskDataset), delete `tests/unit/segmentation/test_model.py`, delete `tests/unit/training/test_pose.py`
2. **CLEAN-02** — Delete `segmentation/pseudo_labeler.py` + cascade: fix `segmentation/__init__.py`, delete `tests/unit/segmentation/test_pseudo_labeler.py`, delete `tests/integration/segmentation/test_yolo_sam_integration.py`
3. **CLEAN-04** — Remove `MOG2Detector` from `segmentation/detector.py`, update `make_detector()`, update `segmentation/__init__.py`, update `tests/unit/segmentation/test_detector.py`, update `tests/unit/engine/test_config.py`, update `tests/unit/engine/test_cli.py`
4. **CLEAN-05** — Remove `unet` and `pose` subcommands from `training/cli.py`, update `training/__init__.py`, update `tests/unit/training/test_training_cli.py`
5. **CLEAN-03** — Replace model internals in `core/midline/backends/segment_then_extract.py` and `direct_pose.py` with no-op stubs; update `core/midline/backends/__init__.py` docstring; update `engine/config.py` docstring; update `tests/unit/core/midline/test_midline_stage.py` and `tests/unit/core/midline/test_direct_pose_backend.py`

Then: ROADMAP.md + REQUIREMENTS.md corrections, `engine/config.py` field cleanup.

### Anti-Patterns to Avoid

- **Deleting `segmentation/crop.py`**: `CropRegion`, `extract_affine_crop`, `invert_affine_points`, `compute_crop_region`, `extract_crop`, `paste_mask` are used throughout `core/`, `reconstruction/`, engine observers, and tests. Do not touch this file.
- **Touching `reconstruction/`, `tracking/`, `visualization/`**: These are NOT legacy duplicates. `core/tracking/stage.py` imports from `aquapose.tracking.ocsort_wrapper`; `core/reconstruction/` imports from `aquapose.reconstruction.triangulation`; `visualization/` is imported by engine observers and tests. Removing them would break the pipeline.
- **Removing `training/datasets.py` entirely**: `_CocoDataset`, `CropDataset`, `stratified_split`, and `apply_augmentation` are general utilities needed by Phase 36 training wrappers. Only `BinaryMaskDataset` goes away.
- **Removing `training/viz.py` entirely**: Check whether `save_pose_augmented_grid` and `save_pose_val_grid` have any callers after `pose.py` is deleted. If not, delete `viz.py` too. But do not assume — verify first.
- **Silent mog2 config acceptance**: The config dataclass currently has `detector_kind: str = "yolo"` with no validation. The engine must raise at stage construction time (or config load time) if `"mog2"` is specified.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CLI command removal | New CLI framework | Just delete `@click.command` decorated functions and their imports | Click groups are declarative; removing a subcommand is just deletion |
| Backend stub | Complex mock infrastructure | Minimal class with `process_frame` returning `AnnotatedDetection(midline=None)` | Downstream already handles `midline=None` — no infrastructure needed |
| Import graph analysis | Custom tooling | Direct grep + Read inspection (already done) | The codebase is small enough to trace manually |

## Common Pitfalls

### Pitfall 1: The Detection Dataclass Lives in `segmentation/detector.py`

**What goes wrong:** Deleting `segmentation/detector.py` entirely in pursuit of removing MOG2, destroying `Detection`, `YOLODetector`, and `make_detector()` which are used by dozens of files.

**Why it happens:** `MOG2Detector` is in the same file as the surviving `Detection` dataclass and `YOLODetector`.

**How to avoid:** CLEAN-04 is a surgical edit to `segmentation/detector.py`, not a file deletion. Only `MOG2Detector` and its watershed helper methods go away. The `Detection` dataclass, `YOLODetector`, and `make_detector()` (now without `mog2` branch) survive.

**Warning signs:** Any diff that removes `Detection` or `YOLODetector` is wrong.

### Pitfall 2: `training/viz.py` May Become Orphaned

**What goes wrong:** After deleting `training/unet.py` and `training/pose.py`, `training/viz.py` may have no callers, but it's not obvious at a glance.

**Why it happens:** `viz.py` exports `save_unet_augmented_grid`, `save_unet_val_grid` (called from `unet.py`) and `save_pose_augmented_grid`, `save_pose_val_grid` (called from `pose.py`).

**How to avoid:** After deleting both `unet.py` and `pose.py`, grep for remaining callers of `viz.py` functions. If none exist, delete `viz.py` too and remove it from `__init__.py`. If some survive, keep only those functions.

**Warning signs:** Imports of `from .viz import` left in `__init__.py` after training module cleanup.

### Pitfall 3: `engine/config.py` Has Stale Documentation and Defaults

**What goes wrong:** After removing the model backends, `MidlineConfig` still has fields like `weights_path`, `backend: str = "segment_then_extract"`, `keypoint_weights_path`, etc., and `DetectionConfig` still mentions `"mog2"` in its docstring.

**Why it happens:** Config field removal wasn't explicitly called out in CLEAN requirements but is implicit — a config that accepts deleted backend names is confusing.

**How to avoid:** After completing CLEAN-03 and CLEAN-04, audit `engine/config.py` `MidlineConfig` and `DetectionConfig` docstrings. Remove `"mog2"` mentions from `DetectionConfig`. Consider whether backend-specific fields (`weights_path`, `keypoint_weights_path`) should be removed or left for Phase 37 to repurpose.

**Decision needed:** The user said "strict config validation — unknown fields or invalid backend names raise errors at config load time." The current `MidlineConfig.backend` has no validation. This needs a `__post_init__` validator or equivalent.

**Warning signs:** `aquapose run --set detection.detector_kind=mog2` silently accepts `mog2` even after CLEAN-04.

### Pitfall 4: Tests Asserting Existence of Removed Commands

**What goes wrong:** `tests/unit/training/test_training_cli.py` has `test_train_help_lists_unet()`, `test_train_help_lists_pose()`, `test_train_unet_help_shows_expected_flags()`, `test_train_pose_help_shows_expected_flags()`. These fail after CLEAN-05 unless deleted.

**How to avoid:** Delete tests for removed commands; update `test_train_help_lists_yolo_obb()` and the overall `--help` assertion tests to not assert `unet` or `pose` appear.

### Pitfall 5: `test_config.py` Uses `"mog2"` as a Test Value

**What goes wrong:** `tests/unit/engine/test_config.py` uses `detector_kind: "mog2"` in YAML override tests and CLI override tests. After CLEAN-04 introduces validation, these tests will fail with a `ValueError` instead of passing.

**How to avoid:** After CLEAN-04, update `test_config.py` to:
- If testing that unknown detector kinds raise an error: change these to `pytest.raises(ValueError)` assertions
- If testing config parsing (not the detector itself): use a valid kind like `"yolo"` instead

### Pitfall 6: `ROADMAP.md` and `REQUIREMENTS.md` Stale Success Criteria

**What goes wrong:** ROADMAP.md Phase 35 success criterion #3 says "segment_then_extract and direct_pose midline backends are removed" — but per CONTEXT.md, only the custom model code inside them is removed. This contradicts REQUIREMENTS.md CLEAN-03 and will confuse future phases.

**How to avoid:** Update ROADMAP.md and REQUIREMENTS.md as the first or last commit of this phase. Specifically:
- ROADMAP.md Phase 35 criterion #3: change to "custom model code (UNetSegmentor, _PoseModel) removed from segment_then_extract and direct_pose backends; both backends stubbed as no-ops pending Phase 37 model wiring"
- REQUIREMENTS.md CLEAN-03: update description to match

## Code Examples

### Verified current file locations (from direct inspection)

**Files to delete entirely:**
- `src/aquapose/segmentation/model.py` — `_UNet`, `_DecoderBlock`, `UNetSegmentor`, `MaskRCNNSegmentor`, `SegmentationResult`, `UNET_INPUT_SIZE`
- `src/aquapose/segmentation/pseudo_labeler.py` — `SAMPseudoLabeler`, `FrameAnnotation`, `AnnotatedFrame`, `filter_mask`, `to_coco_dataset`
- `src/aquapose/training/unet.py` — `train_unet` and helpers
- `src/aquapose/training/pose.py` — `_PoseModel`, `KeypointDataset`, `train_pose`, and helpers
- `tests/unit/segmentation/test_model.py`
- `tests/unit/segmentation/test_pseudo_labeler.py`
- `tests/unit/training/test_pose.py`
- `tests/integration/segmentation/test_yolo_sam_integration.py`

**Files to surgically edit:**

`src/aquapose/segmentation/detector.py` — delete `MOG2Detector` class and `warm_up`/watershed helpers; update `make_detector()`:
```python
def make_detector(kind: str, **kwargs: Any) -> YOLODetector:
    if kind == "yolo":
        return YOLODetector(**kwargs)
    raise ValueError(
        f"Unknown detector_kind: {kind!r}. Available: 'yolo', 'yolo_obb'"
    )
```

`src/aquapose/segmentation/__init__.py` — remove exports for `MOG2Detector`, `SAMPseudoLabeler`, `FrameAnnotation`, `AnnotatedFrame`, `UNetSegmentor`, `MaskRCNNSegmentor`, `SegmentationResult`, `filter_mask`, `to_coco_dataset`

`src/aquapose/training/__init__.py` — remove `BinaryMaskDataset`, `train_pose`, `train_unet`, `KeypointDataset` from imports and `__all__`

`src/aquapose/training/cli.py` — delete `unet()` and `pose()` subcommand functions and their `@train_group.command` decorators; keep `yolo_obb()` and `_parse_input_size()`

`src/aquapose/training/datasets.py` — delete `BinaryMaskDataset` class; keep everything else

`src/aquapose/core/midline/backends/segment_then_extract.py` — replace `UNetSegmentor` import and all model usage with no-op stub; keep `AnnotatedDetection` imports and `process_frame` signature

`src/aquapose/core/midline/backends/direct_pose.py` — replace `_PoseModel` import (from `aquapose.training.pose`) and all model usage with no-op stub; keep `AnnotatedDetection` imports and `process_frame` signature

`src/aquapose/core/midline/backends/__init__.py` — update docstring to remove "Both fully implemented" and note stubs; keep `get_backend()` factory with both kinds still registered (they just return stubs now)

**Tests to update (not delete):**
- `tests/unit/core/midline/test_midline_stage.py` — remove test for fail-fast on missing U-Net weights; update any assertions about UNetSegmentor being loaded
- `tests/unit/core/midline/test_direct_pose_backend.py` — update to test no-op stub behavior; remove tests for _PoseModel loading
- `tests/unit/engine/test_config.py` — change `"mog2"` test values to `"yolo"` or add `pytest.raises(ValueError)` where validation is tested
- `tests/unit/engine/test_cli.py` — update mog2 override test if it expects specific behavior
- `tests/unit/training/test_training_cli.py` — delete tests for `unet` and `pose` subcommands; update `--help` assertions

### Config Validation Approach

The `engine/config.py` `DetectionConfig` and `MidlineConfig` should validate backend names in `__post_init__`. Example for `DetectionConfig`:

```python
def __post_init__(self) -> None:
    if not isinstance(self.extra, dict):
        object.__setattr__(self, "extra", dict(self.extra))
    valid_kinds = {"yolo", "yolo_obb"}
    if self.detector_kind not in valid_kinds:
        raise ValueError(
            f"Unknown detector_kind: {self.detector_kind!r}. "
            f"Available: {sorted(valid_kinds)}"
        )
```

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Custom _UNet (MobileNetV3 encoder) for segmentation | Will be replaced by YOLO11n-seg in Phase 37 | Clean slate — no backward compat required |
| SAM2 pseudo-labels → Label Studio annotation flow | COCO JSON direct annotation → NDJSON conversion (Phase 36) | Simpler pipeline, no SAM2 dependency |
| Custom _PoseModel (encoder + regression head) for keypoints | Will be replaced by YOLO11n-pose in Phase 37 | Existing test data from Phase 33 may still be used |
| MOG2 background subtraction detector | YOLO-OBB detection (already operational) | YOLO-OBB already works well |

**Confirmed NOT legacy duplicates (keep as-is):**
- `src/aquapose/reconstruction/` — canonical location for `Midline2D`, triangulation, curve optimizer. `core/reconstruction/` imports FROM here.
- `src/aquapose/tracking/` — canonical location for `OcSortTracker` wrapper. `core/tracking/stage.py` imports FROM here.
- `src/aquapose/visualization/` — canonical location for overlay, midline_viz, triangulation_viz, diagnostics. Engine observers import FROM here.

## Open Questions

1. **`training/viz.py` fate**
   - What we know: `viz.py` has `save_unet_augmented_grid`, `save_unet_val_grid` (from `unet.py`) and `save_pose_augmented_grid`, `save_pose_val_grid` (from `pose.py`)
   - What's unclear: Are the pose viz functions called from anywhere else after `pose.py` is deleted?
   - Recommendation: After deleting `unet.py` and `pose.py`, grep for callers of `viz.py` functions. If none remain, delete `viz.py` and remove from `__init__.py`. Otherwise keep only live functions.

2. **`engine/config.py` field cleanup scope**
   - What we know: `MidlineConfig` has `backend: str = "segment_then_extract"` as default, plus `weights_path`, `keypoint_weights_path`, `n_keypoints`, etc.
   - What's unclear: Should these fields be removed now or left for Phase 37 to repurpose? The default `"segment_then_extract"` will still be a valid backend name (it's just a stub).
   - Recommendation: Keep fields but change the default `backend` to remain `"segment_then_extract"` (valid, just returns no-op). Add validation to reject unknown backend names. Leave U-Net-specific fields in place for now — Phase 37 may rename/reuse them.

3. **`MidlineConfig` field `weights_path` semantics after CLEAN-03**
   - What we know: Currently points to U-Net weights. After stubbing, it is unused.
   - Recommendation: Leave the field but update its docstring to say "deprecated — will be repurposed in Phase 37 for YOLO-seg model path."

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (via hatch) |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| Quick run command | `hatch run test` |
| Full suite command | `hatch run test-all` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CLEAN-01 | `_UNet`, `UNetSegmentor`, `BinaryMaskDataset` not importable | unit | `hatch run test tests/unit/segmentation/ tests/unit/training/ -x` | Existing tests to be DELETED |
| CLEAN-01 | No import of removed symbols in any surviving file | static (grep) | `grep -r "UNetSegmentor\|BinaryMaskDataset\|_UNet" src/ tests/` | Manual verification |
| CLEAN-02 | `SAMPseudoLabeler`, `FrameAnnotation` not importable | unit | `hatch run test tests/unit/segmentation/ -x` | Existing tests to be DELETED |
| CLEAN-03 | `segment_then_extract` backend runs without errors, returns `midline=None` | unit | `hatch run test tests/unit/core/midline/ -x` | ✅ (to be updated) |
| CLEAN-03 | `direct_pose` backend runs without errors, returns `midline=None` | unit | `hatch run test tests/unit/core/midline/ -x` | ✅ (to be updated) |
| CLEAN-04 | `make_detector("mog2")` raises `ValueError` | unit | `hatch run test tests/unit/segmentation/test_detector.py -x` | ✅ (to be updated) |
| CLEAN-04 | Pipeline config with `detector_kind: "mog2"` raises at load time | unit | `hatch run test tests/unit/engine/test_config.py -x` | ✅ (to be updated) |
| CLEAN-05 | `aquapose train --help` does not list `unet` or `pose` | unit | `hatch run test tests/unit/training/test_training_cli.py -x` | ✅ (to be updated) |
| CLEAN-05 | `aquapose train yolo-obb --help` still works | unit | `hatch run test tests/unit/training/test_training_cli.py -x` | ✅ (to be updated) |

### Sampling Rate
- **Per task commit:** `hatch run test tests/unit/ -x`
- **Per wave merge:** `hatch run test` (full unit suite, excludes @slow)
- **Phase gate:** `hatch run test` green before `/gsd:verify-work`

### Wave 0 Gaps
None — existing test infrastructure covers all phase requirements. The work is updating existing tests (some deletions, some rewrites), not creating new test files.

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection via Read tool — all file paths, class names, import chains verified against actual source
- `src/aquapose/segmentation/model.py` — full content read; symbols: `_UNet`, `_DecoderBlock`, `UNetSegmentor`, `MaskRCNNSegmentor`, `SegmentationResult`, `UNET_INPUT_SIZE`
- `src/aquapose/segmentation/pseudo_labeler.py` — full content read; symbols: `SAMPseudoLabeler`, `FrameAnnotation`, `AnnotatedFrame`, `filter_mask`, `to_coco_dataset`
- `src/aquapose/segmentation/detector.py` — full content read; `MOG2Detector` with watershed logic, `YOLODetector`, `make_detector()`
- `src/aquapose/training/cli.py` — full content read; `unet`, `pose`, `yolo_obb` subcommands confirmed
- `src/aquapose/training/datasets.py` — full content read; `BinaryMaskDataset` isolated from general `_CocoDataset`/`CropDataset`
- `src/aquapose/core/midline/backends/` — both backends fully read; UNetSegmentor and _PoseModel import locations confirmed
- `src/aquapose/core/detection/backends/__init__.py` — confirmed already only registers `yolo` and `yolo_obb`
- Grep results for `aquapose.reconstruction`, `aquapose.tracking`, `aquapose.visualization` imports — confirmed these are NOT legacy duplicates

### Secondary (MEDIUM confidence)
- REQUIREMENTS.md text vs CONTEXT.md interpretation of CLEAN-03: the REQUIREMENTS.md text says "old midline backends removed" but CONTEXT.md (user clarification) says only custom model code inside them is removed. CONTEXT.md takes precedence as the authoritative locked decision.

## Metadata

**Confidence breakdown:**
- File locations and symbol names: HIGH — all verified by direct file reads
- Import cascade (what imports what): HIGH — all verified by grep + read
- No-op stub approach viability: HIGH — downstream `midline=None` handling verified by reading association/reconstruction stage code
- Top-level reconstruction/tracking/visualization are NOT legacy duplicates: HIGH — confirmed by grep of import patterns
- `training/viz.py` fate: MEDIUM — need to re-verify after deletion of callers (one open question remains)
- Config validation approach: MEDIUM — pattern follows existing `__post_init__` in `DetectionConfig`; no external reference needed

**Research date:** 2026-03-01
**Valid until:** 2026-04-01 (codebase changes during Phase 35 itself will supersede)
