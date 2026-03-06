# Phase 38: Stabilization and Tech Debt Cleanup - Research

**Researched:** 2026-03-02
**Domain:** Python codebase cleanup — label format migration, config field consolidation, docstring hygiene, dead code analysis
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### NDJSON → txt+yaml label format
- Remove all NDJSON generation and consumption code entirely — no fallback, no dual format
- All 3 build_yolo_training_data modes (obb, seg, pose) output standard YOLO txt labels + dataset.yaml
- flip_idx in dataset.yaml for pose mode only (identity mapping `[0, 1, 2, 3, 4, 5]` so Ultralytics enables fliplr augmentation)
- OBB and seg modes don't need flip_idx — Ultralytics handles their augmentation natively
- User will delete and regenerate existing NDJSON datasets externally — those files are outside codebase scope
- Files to modify: `scripts/build_yolo_training_data.py`, `tmp/convert_all_annotations.py`, `src/aquapose/training/yolo_obb.py`, `yolo_pose.py`, `yolo_seg.py`, `src/aquapose/training/cli.py`

#### Config field consolidation
- Remove `keypoint_weights_path` from MidlineConfig — just delete the field, no deprecation logic (pre-release code)
- Both segmentation and pose_estimation backends read from the single `weights_path` field
- Rename `model_path` → `weights_path` in DetectionConfig for consistency across all stages
- Generic docstring on unified `weights_path`: "Path to model weights for the active midline backend (segmentation or pose estimation)"

#### init-config defaults
- Default detection backend: `yolo_obb`
- Default midline backend: `pose_estimation`
- Do not hardcode keypoint count (6) in generated config — let it come from config or model metadata
- Show essential fields + all backend selections with brief comments; hide internal tuning params
- Sensible project-relative default paths:
  - `video_dir`: `"videos"`
  - `geometry_path`: `"geometry/calibration.json"`
  - `output_dir`: `"runs"`
  - Detection `weights_path`: `"models/yolo_obb.pt"`
  - Midline `weights_path`: `"models/yolo_pose.pt"`

#### Stale docstring cleanup
- Targeted grep for known stale terms: U-Net, UNet, no-op stub, "Phase 37 pending", legacy backend names
- Fix all matches in src/ — update to reflect current Ultralytics-based implementation
- Known locations: `core/midline/stage.py`, `reconstruction/midline.py`, `core/reconstruction/stage.py`

#### GUIDEBOOK.md audit
- Full accuracy pass against current codebase — every section checked
- Regenerate source layout tree from actual filesystem
- Allow restructuring if sections are unclear or poorly organized
- Preserve the document's role as the authoritative architecture reference
- Update milestone history to include v3.0 completion
- Fix any references to removed code (U-Net, MOG2, SAM2, custom models)

#### Dead code cleanup
- Audit legacy top-level directories: `reconstruction/`, `segmentation/`, `tracking/`, `visualization/` (and any others found)
- Analysis-first approach: produce an import analysis report showing which files are unused, which are thin wrappers of core/ code, and which contain unique logic
- Report must present evidence before any deletion
- Files with unique logic not duplicated elsewhere: migrate into `core/`, `engine/`, or `io/` as appropriate, then delete the legacy file
- Purely dead or wrapper-only files: delete after report confirms no external imports

### Claude's Discretion
- Exact grep patterns for stale docstring scan
- Order of operations across the 6 work areas
- Whether to consolidate small changes into shared plans or keep them separate
- GUIDEBOOK.md restructuring decisions (section ordering, heading levels, content grouping)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| STAB-01 | Training data script produces standard YOLO txt labels + dataset.yaml (not NDJSON); training wrappers consume txt+yaml | NDJSON output format found in all three generate_* functions; wrappers look for `dataset.ndjson`; format swap is well-defined |
| STAB-02 | `weights_path` and `keypoint_weights_path` consolidated into single `weights_path` field | Both fields located in `MidlineConfig` (lines 110, 120); `keypoint_weights_path` read in `pipeline.py` (line 161) and `config.py` (line 621); `DetectionConfig.model_path` → rename to `weights_path` |
| STAB-03 | `init-config` generates correct defaults (YOLO-OBB detection, explicit backend selection, valid weights path) | `init_config` command in `cli.py` currently outputs wrong defaults (`detector_kind: yolo`, `weights_path: models/unet_best.pth`) |
| STAB-04 | All stale docstrings referencing U-Net, no-op stubs, or Phase 37 pending status are updated | Confirmed stale references in `core/midline/stage.py` (module docstring), `reconstruction/midline.py` (two function docstrings) |
</phase_requirements>

---

## Summary

Phase 38 is a pure cleanup phase with six work areas: label format migration (NDJSON → txt+yaml), config field consolidation, init-config default correction, stale docstring updates, GUIDEBOOK.md accuracy pass, and dead code analysis/removal. There are no new algorithms or integrations — every change is mechanical code modification or deletion.

The codebase audit reveals that all six work areas have well-understood scope. The NDJSON format is implemented in three `generate_*` functions in `scripts/build_yolo_training_data.py` and three training wrappers in `src/aquapose/training/`; switching to txt+yaml requires replacing file-output logic and updating wrapper entry points. Config consolidation touches `engine/config.py`, `engine/pipeline.py`, and `cli.py` in a single coherent sweep. Stale docstrings exist in exactly three files confirmed by grep. The legacy directories (`reconstruction/`, `segmentation/`, `tracking/`, `visualization/`) are heavily used by active `core/` code as import sources, meaning they are NOT dead — they are the canonical implementations that `core/` wraps; this is the key finding that shapes the dead code plan.

The dead code work requires an import analysis pass before any deletion. Based on current imports, `segmentation/detector.py`, `segmentation/crop.py`, `reconstruction/midline.py`, `reconstruction/triangulation.py`, and `visualization/overlay.py` are all actively imported from `core/` and cannot be deleted without a migration. The analysis plan must produce evidence before any action.

**Primary recommendation:** Execute in dependency order — config consolidation first (STAB-02), then init-config fix (STAB-03), then NDJSON→txt migration (STAB-01), then docstring pass (STAB-04), then GUIDEBOOK audit, then dead code analysis.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python dataclasses | stdlib | Config hierarchy (frozen dataclasses) | Already used throughout; `frozen=True` enforces immutability |
| PyYAML | installed | YAML serialization/deserialization | Used by `engine/config.py` and `cli.py` already |
| Ultralytics | installed | YOLO training API; txt+yaml dataset format | Standard format is `model.train(data="dataset.yaml")` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pathlib.Path | stdlib | File path manipulation | All file output in the build script |
| click | installed | CLI generation | All CLI command modifications |
| ast | stdlib | Import analysis for dead code audit | Static analysis pass in Wave 0 of dead code plan |

---

## Architecture Patterns

### Pattern 1: Standard YOLO txt+yaml Dataset Format

**What:** Each split gets a `labels/train/` and `labels/val/` directory. Each image has a corresponding `.txt` file with one annotation per line. A `dataset.yaml` declares the split paths, class names, and optional metadata like `kpt_shape` and `flip_idx`.

**When to use:** All three modes (obb, pose, seg) must output this format.

**YOLO OBB txt format** (one line per fish):
```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```
Coordinates normalized to [0, 1].

**YOLO Pose txt format** (one line per fish):
```
<class_id> <cx> <cy> <w> <h> <x1> <y1> <v1> <x2> <y2> <v2> ...
```
Coordinates normalized to [0, 1].

**YOLO Seg txt format** (one line per fish):
```
<class_id> <x1> <y1> <x2> <y2> ...
```
Polygon vertices normalized to [0, 1].

**dataset.yaml for OBB:**
```yaml
path: /abs/path/to/obb
train: images/train
val: images/val
nc: 1
names:
  0: fish
```

**dataset.yaml for Pose** (includes flip_idx for fliplr augmentation):
```yaml
path: /abs/path/to/pose
train: images/train
val: images/val
nc: 1
names:
  0: fish
kpt_shape: [6, 3]
flip_idx: [0, 1, 2, 3, 4, 5]
```

**dataset.yaml for Seg:**
```yaml
path: /abs/path/to/seg
train: images/train
val: images/val
nc: 1
names:
  0: fish
```

**Training wrapper invocation (txt+yaml):**
```python
results = yolo_model.train(data=str(yaml_path), ...)
```

### Pattern 2: Config Field Consolidation

**What:** Remove `keypoint_weights_path` from `MidlineConfig`; update all consumers to use `weights_path` instead. Rename `DetectionConfig.model_path` → `weights_path`.

**Why:** Pre-release code — no backwards compat needed. One field name across all stage configs is cleaner.

**Current state of `MidlineStage.__init__` (line 161):**
```python
weights_path=mc.keypoint_weights_path if mc is not None else None,
```
Must become:
```python
weights_path=mc.weights_path if mc is not None else None,
```

**Current state of `config.py` (line 620-621) path resolution:**
```python
(det_kwargs, ["model_path"]),
(mid_kwargs, ["weights_path", "keypoint_weights_path"]),
```
Must become:
```python
(det_kwargs, ["weights_path"]),
(mid_kwargs, ["weights_path"]),
```

**Current state of `pipeline.py` (line 321):**
```python
model_path=config.detection.model_path,
```
Must become:
```python
weights_path=config.detection.weights_path,
```

**The detection backends** (`core/detection/backends/yolo.py` and `yolo_obb.py`) accept `model_path` as a constructor parameter — these must also be updated to accept `weights_path`.

### Pattern 3: YOLO txt Label Output Structure

**What:** For each dataset split, create `labels/{split}/` alongside `images/{split}/`. For each image, write a `.txt` file with the same stem as the image.

**Example directory layout for pose:**
```
pose/
  images/
    train/
      frame001_000.jpg
    val/
      frame002_000.jpg
  labels/
    train/
      frame001_000.txt
    val/
      frame002_000.txt
  dataset.yaml
```

**The key difference from NDJSON:** Instead of one big `.ndjson` file bundling all image metadata, each image gets its own `.txt` label file, and the dataset root `dataset.yaml` points to the split directories. Ultralytics auto-discovers labels from the mirrored `labels/` directory.

### Anti-Patterns to Avoid

- **Dual-format output:** Don't emit both `.ndjson` and `.txt` as a transition measure. Decision is hard cut to txt+yaml.
- **Hardcoding keypoint count:** Don't put `n_keypoints: 6` in the generated `config.yaml`. The count comes from the model or dataset.yaml.
- **Deprecation shims:** Don't add `@deprecated` wrappers for removed config fields. This is pre-release code — remove cleanly.
- **Deleting legacy modules before import audit:** Legacy `reconstruction/`, `segmentation/` etc. directories are actively imported by `core/` code. Do NOT delete them without first confirming all consumers.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Ultralytics txt label discovery | Custom path resolver | Ultralytics auto-discovery: `labels/` mirrors `images/` | Already how Ultralytics works internally |
| Import graph analysis | Manual file-by-file grep | `ast.parse()` + `ast.walk()` on all `.py` files | Reliable, exhaustive, no false negatives |

---

## Common Pitfalls

### Pitfall 1: `model_path` Rename Scope Creep

**What goes wrong:** `DetectionConfig.model_path` → `weights_path` rename touches the dataclass, the path-resolution loop in `load_config`, the `pipeline.py` call to `DetectionStage`, and both detection backends (`yolo.py`, `yolo_obb.py`) which also accept `model_path` as constructor params. Missing any of these will cause a runtime AttributeError or type error at pipeline build time.

**Why it happens:** The field is used in four separate locations, each of which must be updated atomically.

**How to avoid:** Grep for `model_path` across the entire codebase before declaring the rename done. Verify: `engine/config.py` (field definition + path resolution), `engine/pipeline.py` (config access), `core/detection/backends/yolo.py` (constructor param), `core/detection/backends/yolo_obb.py` (constructor param), tests, and existing test YAML fixtures.

**Warning signs:** `AttributeError: 'DetectionConfig' object has no attribute 'model_path'` at test time.

### Pitfall 2: `keypoint_weights_path` Still Read After Removal

**What goes wrong:** After removing `keypoint_weights_path` from `MidlineConfig`, the `MidlineStage.__init__` in `core/midline/stage.py` still reads `mc.keypoint_weights_path` at line 161. Also, `config.py` line 621 still lists `"keypoint_weights_path"` in the path-resolution loop.

**Why it happens:** Two separate consumers both read the field.

**How to avoid:** After removing the field from `MidlineConfig`, search for `keypoint_weights_path` throughout the entire codebase and eliminate every reference.

### Pitfall 3: NDJSON Tests Break After Format Migration

**What goes wrong:** Existing tests in `tests/unit/training/test_yolo_pose.py` and `test_yolo_seg.py` check for `FileNotFoundError` with message `dataset.ndjson`. After migration to txt+yaml, the error message and lookup path will change to `dataset.yaml`.

**Why it happens:** Tests were written for the NDJSON format.

**How to avoid:** Update test assertions from `dataset\.ndjson` to `dataset\.yaml` when updating the training wrappers. Similarly, `test_build_yolo_training_data.py` may reference NDJSON output paths — audit and update.

### Pitfall 4: Dead Code — Legacy Directories Are Active Import Sources

**What goes wrong:** Assuming `reconstruction/`, `segmentation/`, `tracking/`, and `visualization/` are dead just because they're "legacy". In reality, `core/` imports heavily from them:
- `core/midline/backends/segmentation.py` imports from `reconstruction/midline.py` and `segmentation/crop.py`
- `core/detection/backends/yolo.py` imports from `segmentation/detector.py`
- `core/reconstruction/stage.py` imports from `reconstruction/midline.py` and `reconstruction/triangulation.py`
- `engine/` imports from `visualization/`

**Why it happens:** The "legacy" label in the GUIDEBOOK describes their original role, not their current import status.

**How to avoid:** Produce the full import graph BEFORE any deletion decisions. The analysis-first rule from CONTEXT.md is essential here.

### Pitfall 5: GUIDEBOOK Source Layout Tree Out of Date

**What goes wrong:** The current GUIDEBOOK (line 49) lists `core/midline/backends/` as containing `(segment_then_extract, direct_pose)` — those backend names were replaced in Phase 37 Plan 01 with `segmentation` and `pose_estimation`.

**Why it happens:** GUIDEBOOK was not updated when Phase 37 renamed the backends.

**How to avoid:** Regenerate the source tree from `find src/aquapose -name "*.py" | sort` during the GUIDEBOOK audit. Validate every file reference against the filesystem.

### Pitfall 6: init-config Generates Stale Field Names

**What goes wrong:** After renaming `DetectionConfig.model_path` → `weights_path`, the `init-config` command currently writes `{"detector_kind": "yolo", "model_path": "models/best.pt"}` (line 174 of `cli.py`). This will produce a config YAML that fails validation if a user tries to use it.

**Why it happens:** `init_config` hardcodes the old field names.

**How to avoid:** Update `init_config` as part of the config consolidation work (STAB-02 and STAB-03 are tightly coupled).

---

## Code Examples

### Standard YOLO txt Label Writing (Pose Mode)

```python
# Source: derived from Ultralytics dataset format documentation
# For each crop:
label_dir = pose_root / "labels" / split
label_dir.mkdir(parents=True, exist_ok=True)
label_path = label_dir / f"{crop_stem}.txt"
# Each line: class cx cy w h x1 y1 v1 x2 y2 v2 ...
line = " ".join(str(v) for v in pose_row)
label_path.write_text(line + "\n")
```

### dataset.yaml Writing (Pose Mode with flip_idx)

```python
import yaml
from pathlib import Path

def write_pose_yaml(output_dir: Path, n_train: int, n_val: int) -> Path:
    yaml_data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "fish"},
        "kpt_shape": [6, 3],
        "flip_idx": [0, 1, 2, 3, 4, 5],  # identity — enables fliplr augmentation
    }
    yaml_path = output_dir / "dataset.yaml"
    with yaml_path.open("w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    return yaml_path
```

### Updated init-config Defaults

```python
# In cli.py init_config():
data["detection"] = {
    "detector_kind": "yolo_obb",
    "weights_path": "models/yolo_obb.pt",  # renamed from model_path
}
data["midline"] = {
    "backend": "pose_estimation",
    "weights_path": "models/yolo_pose.pt",  # single unified field
}
```

### MidlineStage pose_estimation branch (after consolidation)

```python
# core/midline/stage.py — after keypoint_weights_path removal
self._backend = get_backend(
    "pose_estimation",
    weights_path=mc.weights_path if mc is not None else None,  # was mc.keypoint_weights_path
    device=device,
    ...
)
```

### Stale Docstring Fixes

**`core/midline/stage.py` module docstring (lines 1-10) — current:**
```
MidlineStage — Stage 4 of the v2.1 5-stage AquaPose pipeline.

Reads detection bounding boxes from Stage 1, crops and segments each detection
via U-Net, then extracts 15-point 2D midlines...
```
**Should become:**
```
MidlineStage — Stage 4 of the 5-stage AquaPose pipeline.

Reads detection bounding boxes from Stage 1, crops each detection,
and runs the configured YOLO backend (YOLO-seg or YOLO-pose) to produce
2D midlines with half-widths...
```

**`core/midline/stage.py` class docstring (lines 58, 80) references `keypoint_weights_path`:**
Must be updated to reference the unified `weights_path` field.

**`reconstruction/midline.py` `_crop_to_frame` docstring (lines 302-314):**
Two references to "U-Net resize" — should read "YOLO-seg inference" or just "the mask resolution".

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| NDJSON dataset format | Standard YOLO txt+yaml | Phase 38 (this phase) | Training wrappers pass `dataset.yaml` to `model.train()` |
| `keypoint_weights_path` + `weights_path` dual fields | Single `weights_path` | Phase 38 (this phase) | Cleaner config; both backends read the same field |
| `model_path` in DetectionConfig | `weights_path` | Phase 38 (this phase) | Field name consistency across all stage configs |
| U-Net segmentation backend | YOLO-seg backend | Phase 35/37 | All U-Net docstring references are now stale |
| `segment_then_extract` / `direct_pose` backend names | `segmentation` / `pose_estimation` | Phase 37 Plan 01 | GUIDEBOOK still lists old names |

---

## Detailed Findings by Work Area

### Work Area 1: NDJSON → txt+yaml (STAB-01)

**Files to modify:**
- `scripts/build_yolo_training_data.py`: All three `generate_*` functions currently write to `dataset.ndjson`. Replace with:
  - Creating `labels/{split}/` directories alongside `images/{split}/`
  - Writing per-image `.txt` files under `labels/{split}/{stem}.txt`
  - Writing a `dataset.yaml` at the dataset root
- `src/aquapose/training/yolo_obb.py`: Change `ndjson_path = data_dir / "dataset.ndjson"` → `yaml_path = data_dir / "dataset.yaml"`; pass yaml path to `model.train(data=...)`
- `src/aquapose/training/yolo_pose.py`: Same change
- `src/aquapose/training/yolo_seg.py`: Same change
- `src/aquapose/training/cli.py`: Help text for `--data-dir` in all three subcommands currently says "dataset.ndjson" — update to "dataset.yaml"
- `tmp/convert_all_annotations.py`: Context mentions this file but it is NOT in the repo (`tmp/` is absent). Likely only in the user's local working directory. SKIP in-repo changes; note this in the plan.

**`format_obb_annotation`, `format_pose_annotation`, `format_seg_annotation`:** These functions return flat `list[float]` — the format of individual annotation rows is already correct for txt output (space-separated normalized values). Only the file-writing infrastructure changes.

**Test impact:**
- `tests/unit/training/test_yolo_pose.py`: `pytest.raises(FileNotFoundError, match=r"dataset\.ndjson")` → update to `dataset\.yaml`
- `tests/unit/training/test_yolo_seg.py`: Same
- `tests/unit/test_build_yolo_training_data.py`: Audit for NDJSON output path assertions

### Work Area 2: Config Field Consolidation (STAB-02)

**Files to modify:**

1. `src/aquapose/engine/config.py`:
   - `DetectionConfig`: Remove `model_path: str | None = None`; add `weights_path: str | None = None`; update docstring
   - `MidlineConfig`: Remove `keypoint_weights_path: str | None = None` (line 120); update docstring (lines 98-99)
   - `_RENAME_HINTS`: Add entry `"model_path": "weights_path"` (helps users with old YAML configs)
   - `load_config` path resolution loop (lines 619-626): Change `["model_path"]` → `["weights_path"]`; remove `"keypoint_weights_path"` from mid list

2. `src/aquapose/engine/pipeline.py` (line 321):
   - `model_path=config.detection.model_path` → `weights_path=config.detection.weights_path`

3. `src/aquapose/core/detection/backends/yolo.py`:
   - Constructor: `model_path: str | Path` → `weights_path: str | Path`; update all internal references

4. `src/aquapose/core/detection/backends/yolo_obb.py`:
   - Same rename as `yolo.py`

5. `src/aquapose/core/detection/stage.py`:
   - Check if it passes `model_path` to backend constructor (line 45 docstring references it) — update accordingly

6. `src/aquapose/core/midline/stage.py` (line 161):
   - `mc.keypoint_weights_path` → `mc.weights_path`

7. `src/aquapose/cli.py` (line 174):
   - `"model_path": "models/best.pt"` → `"weights_path": "models/yolo_obb.pt"` (combined with STAB-03)

**Test impact:**
- `tests/unit/engine/test_config.py`: Any test that constructs `DetectionConfig(model_path=...)` must be updated
- Any YAML fixture files in tests that contain `model_path:` under `detection:` must be updated

### Work Area 3: init-config Defaults (STAB-03)

**File to modify:** `src/aquapose/cli.py` — the `init_config` function (lines 141-186).

**Current issues:**
1. `detector_kind: "yolo"` should be `"yolo_obb"`
2. `model_path: "models/best.pt"` should be `weights_path: "models/yolo_obb.pt"` (after STAB-02)
3. `weights_path: "models/unet_best.pth"` should be `"models/yolo_pose.pt"`
4. No `backend: pose_estimation` in the midline section
5. Path names differ from decisions: `calibration_path` key in data should be `geometry/calibration.json` (currently is)

**New init-config output structure:**
```yaml
project_dir: /home/user/aquapose/projects/myproject
video_dir: videos
calibration_path: geometry/calibration.json
output_dir: runs
n_animals: SET_ME  # required integer
detection:
  detector_kind: yolo_obb  # oriented bounding box detection
  weights_path: models/yolo_obb.pt
midline:
  backend: pose_estimation  # segmentation or pose_estimation
  weights_path: models/yolo_pose.pt
```

### Work Area 4: Stale Docstrings (STAB-04)

**Confirmed stale locations (from grep):**

1. `src/aquapose/core/midline/stage.py`:
   - Module docstring (line 4): "via U-Net" — should describe YOLO-seg or YOLO-pose backends
   - Class docstring (lines 56-81): References `keypoint_weights_path` by name in the Args section — must be updated after STAB-02

2. `src/aquapose/reconstruction/midline.py`:
   - `_crop_to_frame` docstring (lines 302-314): Two references to "U-Net resize"

**Grep patterns to sweep after fixing known issues:**
- `U-Net` (exact case)
- `UNet`
- `no-op stub`
- `Phase 37 pending`
- `segment_then_extract`
- `direct_pose`
- `unet_best`
- `keypoint_weights_path` (in any docstring or comment — not just code)

### Work Area 5: GUIDEBOOK.md Audit

**Current inaccuracies found:**

1. **Source layout tree (section 4, line 49):** `core/midline/backends/` listed as `(segment_then_extract, direct_pose)` — actual files are `segmentation.py` and `pose_estimation.py`
2. **Source layout note (line 68-73):** `segmentation/` annotated as "U-Net model, training, crop, dataset, detector, pseudo-labeler" — U-Net and pseudo-labeler are removed; the surviving files are `crop.py` and `detector.py`
3. **Milestone history:** v3.0 not yet mentioned — should be added

**Suggested GUIDEBOOK accuracy audit sequence:**
1. Regenerate the source layout tree from `find src/aquapose -name "*.py" | sort`
2. Walk each section of the GUIDEBOOK, cross-referencing against current code
3. Specifically check: backend names, config field names, removed modules, stage descriptions

### Work Area 6: Dead Code Analysis

**Import graph findings (from codebase grep):**

The legacy top-level directories are NOT dead — they are the **canonical implementation locations** for code that `core/` wraps or directly imports:

| Legacy Module | Actively Imported By |
|---------------|---------------------|
| `reconstruction/midline.py` | `core/midline/backends/segmentation.py`, `core/midline/backends/pose_estimation.py`, `core/reconstruction/stage.py`, `core/midline/__init__.py`, `core/midline/types.py`, `core/reconstruction/types.py` |
| `reconstruction/triangulation.py` | `core/reconstruction/backends/triangulation.py`, `core/reconstruction/stage.py`, `core/reconstruction/types.py`, `core/reconstruction/__init__.py`, `io/midline_writer.py` |
| `reconstruction/curve_optimizer.py` | `core/reconstruction/backends/curve_optimizer.py` |
| `segmentation/detector.py` | `core/detection/backends/yolo.py`, `core/detection/backends/yolo_obb.py`, `core/detection/types.py`, `core/midline/backends/segmentation.py`, `core/midline/types.py` |
| `segmentation/crop.py` | `reconstruction/midline.py`, `core/midline/backends/segmentation.py`, `core/midline/backends/pose_estimation.py`, `core/midline/types.py` |
| `tracking/ocsort_wrapper.py` | `core/tracking/stage.py`, `tracking/__init__.py` |
| `visualization/overlay.py` | `visualization/midline_viz.py`, `visualization/triangulation_viz.py` |
| `visualization/frames.py` | `engine/overlay_observer.py`, `engine/tracklet_trail_observer.py` |

**Conclusion:** No wholesale directory deletion is possible. The dead code plan must be an import analysis report that identifies:
1. Files that are thin-wrapper re-exports (candidates for inlining or migration)
2. Files that are canonical implementations imported by `core/` (must be kept or migrated)
3. Files that are genuinely unreachable (if any)

The analysis report should be produced in Wave 0 of the dead code plan, with migration/deletion tasks defined by the findings.

---

## Open Questions

1. **`tmp/convert_all_annotations.py`**
   - What we know: CONTEXT.md lists it as a file to modify, but `tmp/` directory does not exist in the repo
   - What's unclear: Whether this file is user-local only or was accidentally omitted from git tracking
   - Recommendation: Note in plan that it's absent from the tracked codebase; skip in-repo changes; add a comment in the plan for the user to apply the same txt+yaml migration if they have this file locally

2. **Detection backend constructor signatures**
   - What we know: `core/detection/backends/yolo.py` and `yolo_obb.py` accept `model_path` as a constructor param, which is passed from `engine/pipeline.py`
   - What's unclear: Whether `core/detection/stage.py` also passes `model_path` by name to the backend constructor (line 45 in the docstring references it but the actual `make_detector()` call may use `**kwargs`)
   - Recommendation: Read `core/detection/stage.py` fully before implementing the rename; confirm the call site

3. **GUIDEBOOK restructuring scope**
   - What we know: Some sections are outdated (source layout, backend names, module descriptions)
   - What's unclear: Whether the GUIDEBOOK's section ordering needs significant restructuring or just spot fixes
   - Recommendation: Do a full read of all 391 lines before making structural decisions; prefer minimal restructuring to preserve continuity

---

## Validation Architecture

> `workflow.nyquist_validation` not present in `.planning/config.json`. Skipping Validation Architecture section.

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection — all file paths and line numbers verified against actual source
- `src/aquapose/engine/config.py` — DetectionConfig, MidlineConfig field inventory
- `src/aquapose/core/midline/stage.py` — stale docstring locations confirmed
- `src/aquapose/reconstruction/midline.py` — U-Net docstring references confirmed
- `scripts/build_yolo_training_data.py` — NDJSON output format confirmed
- `src/aquapose/cli.py` — init-config stale defaults confirmed
- `tests/unit/training/` — test impact confirmed

### Secondary (MEDIUM confidence)
- Ultralytics YOLO txt+yaml format — standard format well-documented; `flip_idx` behavior for pose models derived from standard Ultralytics pose training documentation

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new libraries; all changes use existing tooling
- Architecture: HIGH — all impacted files located and verified; import graph traced
- Pitfalls: HIGH — all pitfalls derived from direct code inspection, not speculation

**Research date:** 2026-03-02
**Valid until:** 2026-04-01 (stable codebase, no fast-moving external dependencies)
