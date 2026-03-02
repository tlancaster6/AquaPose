# Phase 36: Training Wrappers - Research

**Researched:** 2026-03-01
**Domain:** Ultralytics YOLO26 training API, COCO segmentation format, project NDJSON format conversion
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Seg data conversion:**
- Add `--mode seg` to existing `scripts/build_yolo_training_data.py` (not a new CLI subcommand)
- Source format: standard COCO polygon JSON (`segmentation` field with polygon arrays)
- Training on **per-detection crops** (not full frames) — crops extracted around each OBB detection
- **All visible fish** in each crop get labeled (target + intruders)
- Multi-ring COCO polygons: **keep largest ring only**, drop small fragments
- Data converter produces NDJSON files + dataset YAML; training wrapper reads the YAML — **fully decoupled**, YAML is the contract

**Training wrappers:**
- **YOLO26 models** (not YOLO11) — `yolo26n-seg` and `yolo26n-pose` as defaults
- `--model` flag for variant selection (e.g., `yolo26s-seg`, `yolo26m-pose`)
- `--weights` flag for pretrained weight loading (transfer learning); no `--resume` support
- Ultralytics default augmentation — no custom augmentation flags exposed
- Keypoint definition (names, count, skeleton) read from dataset YAML, **not hardcoded** in wrapper
- Random 80/20 train/val split via `--val-split` flag (same as existing yolo_obb pattern)

**File structure:**
- Separate files: `yolo_seg.py` and `yolo_pose.py` alongside existing `yolo_obb.py`
- Existing `pose.py` (old custom trainer) removed in Phase 35; new file is `yolo_pose.py`
- Fully decoupled from data converter — no shared utility code

**Output and validation:**
- Default training output directory: `~/aquapose/yolo/` (overridable via `--output-dir`)
- Quality thresholds for "good enough": seg mAP > 0.5, pose mAP > 0.4
- Phase 36 scope is **smoke test only** (1-2 epochs to verify wrappers work)

**Existing pose data:**
- `--mode pose` already produces NDJSON from COCO keypoint JSON — verify compatibility with YOLO26-pose format and use as-is

### Claude's Discretion
- Exact Ultralytics `.train()` parameter mapping
- Dataset YAML structure details (follows Ultralytics conventions)
- Error handling for malformed COCO annotations
- Test structure for smoke tests

### Deferred Ideas (OUT OF SCOPE)
- CLI formalization of training data prep (`aquapose data convert-seg`)
- Full training runs with quality tuning
- Resume from checkpoint (`--resume`)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Segmentation training data converter takes COCO segmentation JSON (polygon annotations) and produces NDJSON-format YOLO-seg dataset (matching existing OBB/pose NDJSON pattern) | COCO `segmentation` polygon format documented; project NDJSON schema reverse-engineered from existing OBB/pose code; per-crop architecture matches existing `generate_pose_dataset` pattern |
| TRAIN-01 | YOLO-seg training wrapper callable from CLI, following existing yolo_obb.py pattern | YOLO26n-seg.pt confirmed available; Ultralytics `model.train()` API confirmed; NDJSON→YOLO.txt conversion requirement identified and documented |
| TRAIN-02 | YOLO-pose training wrapper callable from CLI, following existing yolo_obb.py pattern | YOLO26n-pose.pt confirmed available; kpt_shape and kpt_names YAML schema confirmed from coco-pose.yaml; existing pose NDJSON format analyzed |
</phase_requirements>

## Summary

Phase 36 adds three things: a `--mode seg` branch to `scripts/build_yolo_training_data.py`, and two new training wrapper modules `yolo_seg.py` and `yolo_pose.py` mirroring the existing `yolo_obb.py` pattern. The primary technical challenge is a format conversion step that training wrappers must perform: the project's custom NDJSON label format (used as an intermediate storage format) is not directly consumed by Ultralytics `model.train()`. The wrappers must convert project NDJSON records to standard YOLO `.txt` label files before invoking training.

YOLO26 models are confirmed available in Ultralytics 8.4.17 (currently installed). The `model.train()` API is identical for seg and pose tasks, with `task` determined implicitly by the model variant. The only seg/pose-specific differences are in the dataset YAML structure: pose requires `kpt_shape` and optionally `kpt_names`/`flip_idx`; seg requires no additional fields beyond standard detection YAML.

The seg data converter follows the same per-detection-crop architecture as `generate_pose_dataset`: for each OBB detection, warp the crop, then record all visible fish polygon masks in that crop. The critical implementation subtlety is that COCO segmentation polygons are in full-image coordinates and must be transformed via the same affine warp used for the crop.

**Primary recommendation:** Implement `--mode seg` in `build_yolo_training_data.py` first (pure Python, testable without GPU), then implement the two training wrappers (`yolo_seg.py`, `yolo_pose.py`) with inline NDJSON→YOLO.txt conversion before calling `model.train(data=data_yaml)`.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ultralytics | 8.4.17 (installed) | YOLO26 training API | Already in pyproject.toml; confirmed working |
| click | >=8.1 (installed) | CLI group/commands | Already used in `training/cli.py` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| cv2 | >=4.8 (installed) | Affine warp for crop extraction | Seg crops follow same warp as pose crops |
| numpy | >=1.24 (installed) | Polygon coordinate transforms | Same as existing build script |
| json | stdlib | NDJSON parsing | Same as existing build script |
| shutil | stdlib | Weight file copying, tmp cleanup | Same as existing yolo_obb.py |
| torch | >=2.0 (installed) | Device detection | Same as existing yolo_obb.py |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| YOLO26n-seg | YOLO11n-seg | Context.md explicitly says YOLO26 — not a choice |
| Project NDJSON (intermediate) + conversion | Direct YOLO.txt output from converter | Conversion in wrapper adds complexity; keeping NDJSON as intermediate maintains the decoupling contract from CONTEXT.md |
| `--model` flag (full name) | `--model-size` flag (size suffix) | CONTEXT.md says `--model` with full name; more flexible than yolo_obb pattern |

**Installation:** All dependencies already in `pyproject.toml`. No new packages needed.

## Architecture Patterns

### Recommended Project Structure
```
src/aquapose/training/
├── yolo_obb.py        # Existing OBB wrapper (reference pattern)
├── yolo_seg.py        # New: YOLO26 seg wrapper (Wave 1)
├── yolo_pose.py       # New: YOLO26 pose wrapper (Wave 1)
├── cli.py             # Add seg/pose subcommands (Wave 1)
├── common.py          # Existing shared utilities (MetricsLogger)
└── __init__.py        # Add train_yolo_seg, train_yolo_pose to __all__

scripts/
└── build_yolo_training_data.py  # Add --mode seg branch (Wave 1)
```

### Pattern 1: Training Wrapper with NDJSON→YOLO.txt Conversion

**What:** Training wrappers read the data.yaml (which references `.ndjson` files), convert
those `.ndjson` files to standard YOLO `.txt` labels in-place in a `labels/` subdirectory,
rewrite the YAML to point to `images/` dirs, then call `model.train(data=rewritten_yaml)`.

**When to use:** Whenever the dataset was produced by `build_yolo_training_data.py`.

**Example (seg wrapper core logic):**
```python
# Source: reverse-engineered from Ultralytics BaseTrainer.get_dataset() and project NDJSON schema
from ultralytics import YOLO
import json, shutil
from pathlib import Path

def _convert_ndjson_to_yolo_txt(
    ndjson_path: Path,
    images_dir: Path,
    labels_dir: Path,
    mode: str,  # "seg" or "pose" or "obb"
) -> None:
    """Convert project NDJSON to standard YOLO .txt label files."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    for line in ndjson_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        img_name = Path(record["image"]).name
        stem = Path(img_name).stem
        label_path = labels_dir / f"{stem}.txt"
        lines = []
        if mode == "seg":
            for ann in record.get("annotations", []):
                cls = ann["class_id"]
                pts = " ".join(f"{x:.6f} {y:.6f}" for x, y in ann["polygon"])
                lines.append(f"{cls} {pts}")
        elif mode == "pose":
            for ann in record.get("annotations", []):
                cls = ann["class_id"]
                cx, cy, w, h = ann["bbox"]
                kps = " ".join(
                    f"{k[0]:.6f} {k[1]:.6f} {k[2]}" for k in ann["keypoints"]
                )
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kps}")
        label_path.write_text("\n".join(lines) + "\n")
```

### Pattern 2: Dataset YAML for Seg/Pose (after conversion)

**Seg data.yaml** (standard, no extra keys needed):
```yaml
path: /abs/path/to/seg/dataset
train: images/train
val: images/val
nc: 1
names:
  0: fish
```

**Pose data.yaml** (from existing `generate_pose_dataset`, read verbatim — REWRITE for images/ path):
```yaml
path: /abs/path/to/pose/dataset
train: images/train
val: images/val
nc: 1
names:
  0: fish
kpt_shape: [6, 3]
kpt_names:
  0:
    - nose
    - head
    - spine1
    - spine2
    - spine3
    - tail
flip_idx: []
```

The wrapper reads `kpt_shape` and `kpt_names` from the existing YAML — does not hardcode them.

### Pattern 3: CLI Subcommand (matches existing yolo-obb)

**What:** `train seg` and `train pose` subcommands in `training/cli.py`, registered in `train_group`.

**New flags vs existing yolo-obb:**
- Replace `--model-size` (n/s/m/l/x suffix) with `--model` (full name string, e.g., `yolo26n-seg`)
- Add `--weights` (optional path to pretrained .pt for transfer learning)
- Keep `--data-dir`, `--output-dir`, `--epochs`, `--batch-size` (note: `--batch` not `--batch-size` for parity — see pitfall below), `--device`, `--imgsz` identical

**Verified CLI flag semantics (Ultralytics `model.train()` parameter names):**
```python
model.train(
    data=str(data_yaml),      # rewritten to images/ dirs
    epochs=epochs,
    batch=batch_size,         # Ultralytics uses 'batch', not 'batch_size'
    device=device,
    project=str(output_dir / "_ultralytics"),
    name="train",
    imgsz=imgsz,
)
```

### Anti-Patterns to Avoid
- **Passing project NDJSON path directly to model.train():** Ultralytics only handles its OWN NDJSON format (requires `aiohttp`, async download). Project NDJSON is a different schema. Always convert first.
- **Hardcoding kpt_shape in wrapper:** CONTEXT.md says keypoint definition must come from data.yaml. Read it; don't replicate it.
- **Re-generating OBB crops for seg data:** The seg converter needs full-frame images and masks, then crops them (same OBB warp as pose). It does NOT re-run OBB inference at conversion time — it takes COCO polygon annotations as input.
- **Using `val_split` in model.train():** Ultralytics `model.train()` has no `val_split` parameter. The split is determined by the `val:` path in data.yaml. The `val_split` arg is used only during dataset generation (in `build_yolo_training_data.py`) and recorded in MetricsLogger for bookkeeping.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Training loop | Custom epoch loop | `model.train()` (Ultralytics) | Handles mixed precision, grad clipping, LR scheduling, augmentation internally |
| Seg/pose loss functions | Custom loss | Ultralytics seg/pose losses | Task-specific loss already wired to model variant |
| Polygon affine transform | Custom transform | Reuse `affine_warp_crop` / `transform_keypoints` from `build_yolo_training_data.py` | Already implemented and tested; coord-space bugs already fixed |
| YOLO.txt format writer | Custom format | Simple f-string join (no library needed) | Format is trivial: `class_id x1 y1 ...` normalized coords |
| COCO polygon parser | Custom parser | Standard COCO `segmentation` field access | It's just `ann["segmentation"]` — a list of flat coordinate lists |

**Key insight:** The only non-trivial new code in this phase is (1) the affine-warping of polygon masks into crop space for `--mode seg`, and (2) the NDJSON→YOLO.txt conversion step in the training wrappers. Everything else is mechanical pattern-matching against existing code.

## Common Pitfalls

### Pitfall 1: NDJSON Format Confusion
**What goes wrong:** Developer passes `train.ndjson` path to `model.train(data=...)`, expecting Ultralytics to handle it. Ultralytics only handles its own NDJSON platform format (async download with `aiohttp`). Project NDJSON uses a different schema and different field names.
**Why it happens:** Ultralytics does support `.ndjson` files — but only its own format. The name collision is misleading.
**How to avoid:** Training wrappers ALWAYS convert project NDJSON → YOLO `.txt` labels + standard YAML before calling `model.train()`.
**Warning signs:** `aiohttp` import errors, `KeyError` on NDJSON parsing, or "images not found" errors pointing to `.ndjson` paths.

### Pitfall 2: data.yaml path must be absolute for Ultralytics
**What goes wrong:** Ultralytics resolves `path:` relative to `DATASETS_DIR` if it's not absolute and doesn't exist. This causes "images not found" errors when training from arbitrary directories.
**Why it happens:** `check_det_dataset` resolves relative paths relative to `DATASETS_DIR` (typically `~/ultralytics/datasets`), not the current working directory.
**How to avoid:** Always write `path: /absolute/path/to/dataset` in the rewritten data.yaml. Use `Path(data_dir).resolve()`.
**Warning signs:** "Dataset images not found" error pointing to `~/ultralytics/datasets/...`.

### Pitfall 3: Seg polygon coordinates must be affine-transformed into crop space
**What goes wrong:** COCO polygon coordinates are in full-frame pixel space. If copied directly into a crop-space label, the polygon is wildly wrong.
**Why it happens:** The seg converter extracts crops via affine warp (same OBB warp as pose). All annotations must be transformed by the same affine matrix.
**How to avoid:** Reuse or replicate `transform_keypoints` logic from `build_yolo_training_data.py` — apply affine matrix to each polygon vertex pair.
**Warning signs:** Training loss spikes from garbage masks; visual inspection shows misaligned polygon overlays on crops.

### Pitfall 4: Multi-ring COCO polygons
**What goes wrong:** Some COCO annotations have multiple polygon rings (e.g., fish partially occluded, creating disjoint visible regions). Taking the first ring or averaging gives a bad mask.
**Why it happens:** COCO `segmentation` is a list of polygons, where each polygon is a flat `[x1,y1,x2,y2,...]` list. A single annotation can have multiple rings.
**How to avoid:** Select the largest ring by vertex count (i.e., `max(ann["segmentation"], key=len)`). This is a locked decision from CONTEXT.md.
**Warning signs:** Very small or fragmented mask labels appearing in converted dataset.

### Pitfall 5: `--batch` vs `--batch-size` naming
**What goes wrong:** The existing `yolo-obb` CLI uses `--batch-size` (Click parameter) which maps to `batch_size` (Python arg) which maps to `batch=` in `model.train()`. New wrappers must maintain CLI parity.
**Why it happens:** Click uses `--batch-size` (hyphenated), Python uses `batch_size` (underscored), Ultralytics uses `batch` (no suffix).
**How to avoid:** Keep `--batch-size` as the CLI flag name for all wrappers (matches existing yolo-obb). Pass as `batch=batch_size` to `model.train()`.
**Warning signs:** CLI inconsistency noticed during smoke test.

### Pitfall 6: Polygon normalization
**What goes wrong:** COCO polygons are in pixel space; YOLO seg labels require normalized `[0,1]` coordinates relative to crop dimensions.
**Why it happens:** Different coordinate systems. COCO is absolute pixels; YOLO is normalized.
**How to avoid:** After affine-transforming polygon vertices into crop space, divide by `(crop_w, crop_h)`. Clip to `[0, 1]`.
**Warning signs:** Segmentation masks appear off-screen or clipped to image edges during training visualization.

## Code Examples

Verified patterns from official sources and codebase:

### YOLO26 Seg Training (Ultralytics API)
```python
# Source: Ultralytics docs/YOLO26 docs confirmed; BaseTrainer.get_dataset() source verified
from ultralytics import YOLO
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo26n-seg.pt")       # downloads if not cached
results = model.train(
    data="/abs/path/to/seg/data.yaml",   # must be absolute, must use images/ dirs
    epochs=2,
    batch=16,
    device=device,
    project="/abs/path/to/output/_ultralytics",
    name="train",
    imgsz=640,
)
save_dir = results.save_dir if results is not None else Path(project) / name
best_weights = Path(str(save_dir)) / "weights" / "best.pt"
```

### YOLO26 Pose Training (Ultralytics API)
```python
# Source: identical to seg except model name; data.yaml must include kpt_shape
model = YOLO("yolo26n-pose.pt")
results = model.train(
    data="/abs/path/to/pose/data.yaml",  # must include kpt_shape: [6, 3]
    epochs=2,
    batch=16,
    device=device,
    project="/abs/path/to/output/_ultralytics",
    name="train",
    imgsz=640,
)
```

### Project NDJSON Schema (reverse-engineered from existing code)

**OBB record** (`train.ndjson` lines in `output/obb/`):
```json
{"image":"images/train/foo.jpg","width":1920,"height":1080,"obbs":[{"class_id":0,"corners":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}]}
```

**Pose record** (`train.ndjson` lines in `output/pose/`):
```json
{"image":"images/train/foo_001.jpg","width":128,"height":64,"annotations":[{"class_id":0,"bbox":[0.5,0.5,1.0,1.0],"keypoints":[[x1,y1,v1],[x2,y2,v2],...]}]}
```

**Seg record** (new, to be defined for `output/seg/`):
```json
{"image":"images/train/foo_001.jpg","width":128,"height":64,"annotations":[{"class_id":0,"polygon":[[x1,y1],[x2,y2],...]}]}
```

All coordinates normalized to `[0,1]` relative to image dimensions.

### Rewritten data.yaml for Ultralytics (seg example)
```yaml
# Written by training wrapper (overrides original which uses .ndjson paths)
path: /absolute/path/to/output/seg
train: images/train
val: images/val
nc: 1
names:
  0: fish
```

### Rewritten data.yaml for Ultralytics (pose example)
```yaml
# Written by training wrapper; kpt_shape/kpt_names read from original, preserved here
path: /absolute/path/to/output/pose
train: images/train
val: images/val
nc: 1
names:
  0: fish
kpt_shape: [6, 3]
kpt_names:
  0:
    - nose
    - head
    - spine1
    - spine2
    - spine3
    - tail
flip_idx: []
```

### NDJSON → YOLO.txt Conversion (seg)
```python
# Converts project NDJSON seg records to standard YOLO seg .txt label files
import json
from pathlib import Path

def convert_seg_ndjson_to_txt(ndjson_path: Path, labels_dir: Path) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for line in ndjson_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        stem = Path(record["image"]).stem
        label_path = labels_dir / f"{stem}.txt"
        rows = []
        for ann in record.get("annotations", []):
            cls = ann["class_id"]
            pts = " ".join(f"{x:.6f} {y:.6f}" for x, y in ann["polygon"])
            rows.append(f"{cls} {pts}")
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
```

### NDJSON → YOLO.txt Conversion (pose)
```python
def convert_pose_ndjson_to_txt(ndjson_path: Path, labels_dir: Path) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for line in ndjson_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        stem = Path(record["image"]).stem
        label_path = labels_dir / f"{stem}.txt"
        rows = []
        for ann in record.get("annotations", []):
            cls = ann["class_id"]
            cx, cy, w, h = ann["bbox"]
            kps = " ".join(
                f"{k[0]:.6f} {k[1]:.6f} {int(k[2])}" for k in ann["keypoints"]
            )
            rows.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kps}")
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| YOLO11n-seg / YOLO11n-pose (roadmap plan) | YOLO26n-seg / YOLO26n-pose | User decision, 2026-03-01 | Better CPU inference (43% faster), improved seg proto module, RLE keypoint localization |
| Custom U-Net segmentation | YOLO26n-seg | v3.0 decision | Ultralytics handles training loop, augmentation, loss internally |
| Custom keypoint regression | YOLO26n-pose | v3.0 decision | Direct regression via YOLO-pose superior to heatmap approach |

**Deprecated/outdated:**
- `yolov8s-obb.pt` in existing `yolo_obb.py`: Still functional but model naming convention changed. YOLO26 uses `yolo26n-*.pt` naming.
- `--model-size` pattern in `yolo_obb.py`: New wrappers use `--model` (full name string) per CONTEXT.md.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `hatch run test` |
| Full suite command | `hatch run test-all` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | `--mode seg` added to `build_yolo_training_data.py` | unit | `hatch run test tests/unit/test_build_yolo_training_data.py -k seg` | Partial (file exists, no seg tests yet) |
| DATA-01 | Seg NDJSON records contain correct polygon fields | unit | `hatch run test tests/unit/test_build_yolo_training_data.py` | Partial |
| DATA-01 | Multi-ring polygon: largest ring kept, others dropped | unit | `hatch run test tests/unit/test_build_yolo_training_data.py::TestSegConverter` | ❌ Wave 0 |
| DATA-01 | Polygon affine-transformed to crop space correctly | unit | `hatch run test tests/unit/test_build_yolo_training_data.py::TestSegConverter` | ❌ Wave 0 |
| TRAIN-01 | `aquapose train seg --help` lists expected flags | unit | `hatch run test tests/unit/training/test_training_cli.py -k seg` | ❌ Wave 0 |
| TRAIN-01 | `yolo_seg.py` does not import from `aquapose.engine` | unit | `hatch run test tests/unit/training/test_training_cli.py::test_training_modules_do_not_import_engine` | Partial (test exists; file doesn't) |
| TRAIN-01 | NDJSON→YOLO.txt conversion produces valid labels | unit | `hatch run test tests/unit/training/test_yolo_seg.py` | ❌ Wave 0 |
| TRAIN-02 | `aquapose train pose --help` lists expected flags | unit | `hatch run test tests/unit/training/test_training_cli.py -k pose` | ❌ Wave 0 |
| TRAIN-02 | `yolo_pose.py` does not import from `aquapose.engine` | unit | `hatch run test tests/unit/training/test_training_cli.py::test_training_modules_do_not_import_engine` | Partial |
| TRAIN-02 | NDJSON→YOLO.txt conversion preserves kpt_shape from YAML | unit | `hatch run test tests/unit/training/test_yolo_pose.py` | ❌ Wave 0 |
| TRAIN-01/02 | Smoke: 1 epoch seg/pose training completes without error | smoke (marked `@slow`) | `hatch run test-all tests/integration/test_training_smoke.py` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `hatch run test tests/unit/training/` (fast, no GPU)
- **Per wave merge:** `hatch run test` (full unit suite)
- **Phase gate:** Full suite green before `/gsd:verify-work`; smoke tests run manually

### Wave 0 Gaps
- [ ] `tests/unit/training/test_yolo_seg.py` — NDJSON→YOLO.txt conversion + data.yaml rewrite for seg (REQ DATA-01, TRAIN-01)
- [ ] `tests/unit/training/test_yolo_pose.py` — NDJSON→YOLO.txt conversion + data.yaml rewrite for pose (REQ TRAIN-02)
- [ ] `tests/unit/test_build_yolo_training_data.py` — Add `TestSegConverter` class with polygon transform and multi-ring tests (REQ DATA-01)
- [ ] `tests/unit/training/test_training_cli.py` — Add tests for `seg` and `pose` CLI subcommands (REQ TRAIN-01, TRAIN-02)
- [ ] Smoke test file (optional, marked `@slow`): `tests/integration/test_training_smoke.py`

## Open Questions

1. **Does the existing pose NDJSON format need any changes for YOLO26-pose compatibility?**
   - What we know: The existing `generate_pose_dataset` produces NDJSON with `kpt_shape: [6, 3]` in data.yaml. Ultralytics pose training requires `kpt_shape` in data.yaml. The format appears compatible.
   - What's unclear: Whether YOLO26-pose has any additional validation on keypoint format beyond what YOLO11-pose required. May surface only during the smoke test.
   - Recommendation: Proceed assuming compatibility; the smoke test will catch any format mismatch.

2. **How should the seg converter handle annotations where the polygon doesn't intersect the OBB crop?**
   - What we know: After affine-warping polygon vertices into crop space, some points may be out-of-bounds. The pose wrapper clips OOB keypoints and marks them invisible.
   - What's unclear: YOLO seg doesn't have per-point visibility. An entirely OOB polygon should produce an empty label (no annotation for that fish in this crop). A partially OOB polygon should be clipped to crop bounds.
   - Recommendation: Skip annotations where fewer than 3 polygon vertices fall within the crop bounds (minimum needed for a valid polygon). Clip partial polygons.

3. **Should `yolo_seg.py` and `yolo_pose.py` inline the NDJSON conversion or factor it into `common.py`?**
   - What we know: CONTEXT.md says "fully decoupled from data converter — no shared utility code."
   - What's unclear: Whether "no shared utility code" means no shared code between seg and pose wrappers, or only between wrappers and the data converter.
   - Recommendation: Factor NDJSON→YOLO.txt conversion into private helper functions within each wrapper module. If the two implementations are nearly identical, consider a `_convert_ndjson_labels` function in `common.py` with a `mode` parameter — this is within the wrapper layer, not shared with the converter.

## Sources

### Primary (HIGH confidence)
- Installed ultralytics 8.4.17 (`inspect.getsource()`) — `BaseTrainer.get_dataset()`, `convert_ndjson_to_yolo()`, `YOLODataset.__init__()`, `BaseDataset.get_img_files()`, `check_det_dataset()`
- `site-packages/ultralytics/cfg/datasets/coco-pose.yaml` — canonical pose data.yaml schema including `kpt_shape`, `kpt_names`, `flip_idx`
- `src/aquapose/training/yolo_obb.py` — reference implementation for wrapper pattern
- `src/aquapose/training/cli.py` — reference implementation for CLI pattern
- `scripts/build_yolo_training_data.py` — project NDJSON schema (OBB and pose formats)

### Secondary (MEDIUM confidence)
- [Ultralytics YOLO26 Docs](https://docs.ultralytics.com/models/yolo26/) — model variants `yolo26n-seg.pt`, `yolo26n-pose.pt` confirmed
- [Ultralytics Segment Task Docs](https://docs.ultralytics.com/tasks/segment/) — `model.train()` API parameters
- [Ultralytics Pose Task Docs](https://docs.ultralytics.com/tasks/pose/) — `model.train()` API parameters, `kpt_shape` requirement

### Tertiary (LOW confidence)
- WebSearch finding: "Ultralytics v8.4.17 NDJSON conversion" — active development confirmed but project NDJSON is a distinct custom format

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified from installed packages
- Architecture patterns: HIGH — reverse-engineered from installed ultralytics source; existing codebase patterns read directly
- Pitfalls: HIGH — critical NDJSON format mismatch verified by reading `BaseTrainer.get_dataset()` source and project NDJSON schemas
- Data format: HIGH — verified from `coco-pose.yaml` and ultralytics source
- YOLO26 availability: MEDIUM — confirmed from docs and web search, not from local download/test

**Research date:** 2026-03-01
**Valid until:** 2026-03-31 (Ultralytics releases frequently; YOLO26 API may change)
