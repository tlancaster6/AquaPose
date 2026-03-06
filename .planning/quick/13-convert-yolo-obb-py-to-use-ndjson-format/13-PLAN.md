---
phase: quick-13
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/training/yolo_obb.py
  - src/aquapose/training/cli.py
  - scripts/build_yolo_training_data.py
  - tests/unit/training/test_training_cli.py
autonomous: true
requirements: []

must_haves:
  truths:
    - "yolo_obb.py uses train_yolo_ndjson from common.py like seg and pose do"
    - "OBB CLI uses --model flag instead of --model-size"
    - "build_yolo_training_data.py OBB output uses 'annotations' key so convert_ndjson_to_txt works"
    - "All existing tests pass after changes"
  artifacts:
    - path: "src/aquapose/training/yolo_obb.py"
      provides: "OBB training wrapper using NDJSON pipeline"
      contains: "train_yolo_ndjson"
    - path: "src/aquapose/training/cli.py"
      provides: "Updated OBB CLI with --model flag"
      contains: "--model"
    - path: "scripts/build_yolo_training_data.py"
      provides: "OBB NDJSON output with annotations key"
      contains: "annotations"
  key_links:
    - from: "src/aquapose/training/yolo_obb.py"
      to: "src/aquapose/training/common.py"
      via: "train_yolo_ndjson import"
      pattern: "from .common import.*train_yolo_ndjson"
---

<objective>
Convert yolo_obb.py to use NDJSON format and follow the same pattern as yolo_pose.py and yolo_seg.py via train_yolo_ndjson from common.py.

Purpose: Unify all three YOLO training wrappers to use the same NDJSON-based pipeline, eliminating the inconsistency where OBB was the only wrapper that expected pre-converted YOLO .txt labels.
Output: Consistent yolo_obb.py, updated CLI, and fixed build script NDJSON key.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/training/yolo_obb.py
@src/aquapose/training/yolo_pose.py
@src/aquapose/training/yolo_seg.py
@src/aquapose/training/common.py
@src/aquapose/training/cli.py
@src/aquapose/training/__init__.py
@scripts/build_yolo_training_data.py
@tests/unit/training/test_training_cli.py

<interfaces>
<!-- Key functions from common.py that yolo_obb.py must use -->
From src/aquapose/training/common.py:
```python
def convert_ndjson_to_txt(
    ndjson_path: Path,
    labels_dir: Path,
    format_line: Callable[[dict], str],
) -> None: ...

def rewrite_data_yaml(
    data_dir: Path,
    original_yaml_path: Path,
    preserve_keys: Sequence[str] = (),
) -> Path: ...

def train_yolo_ndjson(
    data_dir: Path,
    output_dir: Path,
    *,
    format_label_line: Callable[[dict], str],
    yaml_preserve_keys: Sequence[str] = (),
    epochs: int = 100,
    batch_size: int = 16,
    device: str | None = None,
    val_split: float = 0.2,
    imgsz: int = 640,
    model: str,
    weights: Path | None = None,
) -> Path: ...
```

<!-- NDJSON record format from common.py convert_ndjson_to_txt -->
Each NDJSON line: {"image": "...", "annotations": [...], ...}
convert_ndjson_to_txt reads record["annotations"] and calls format_line(ann) for each.

<!-- Current OBB NDJSON from build_yolo_training_data.py uses "obbs" key -->
Current: {"image": "...", "obbs": [...], "width": N, "height": N}
Must change to: {"image": "...", "annotations": [...], "width": N, "height": N}
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Fix build_yolo_training_data.py OBB NDJSON to use "annotations" key</name>
  <files>scripts/build_yolo_training_data.py</files>
  <action>
In `generate_obb_dataset()`, change the NDJSON record dict key from `"obbs"` to `"annotations"` (line ~632). This aligns OBB NDJSON output with the pose and seg formats, which both use `"annotations"` as the key. The `convert_ndjson_to_txt` function in common.py reads `record["annotations"]`, so this key must match.

Change:
```python
record = {
    "image": f"images/{split}/{Path(file_name).name}",
    "width": img_w,
    "height": img_h,
    "obbs": obb_annots,
}
```
To:
```python
record = {
    "image": f"images/{split}/{Path(file_name).name}",
    "width": img_w,
    "height": img_h,
    "annotations": obb_annots,
}
```
  </action>
  <verify>hatch run test tests/unit/training/ -x</verify>
  <done>"obbs" key replaced with "annotations" in generate_obb_dataset NDJSON output.</done>
</task>

<task type="auto">
  <name>Task 2: Rewrite yolo_obb.py to use train_yolo_ndjson and update CLI</name>
  <files>
    src/aquapose/training/yolo_obb.py
    src/aquapose/training/cli.py
    tests/unit/training/test_training_cli.py
  </files>
  <action>
**yolo_obb.py:** Rewrite to follow the exact same pattern as yolo_seg.py:

1. Replace all imports — remove `shutil`, `torch`, `MetricsLogger`. Import only `Path` and `convert_ndjson_to_txt, rewrite_data_yaml, train_yolo_ndjson` from `.common`.

2. Add `_format_obb_label_line(ann: dict) -> str` that formats one OBB annotation as a YOLO-OBB label line. OBB NDJSON annotation format is `{"class_id": 0, "corners": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}` (normalized). Output format: `class_id x1 y1 x2 y2 x3 y3 x4 y4` (8 corner values, already normalized from build script).
   ```python
   def _format_obb_label_line(ann: dict) -> str:
       class_id = int(ann["class_id"])
       parts: list[str] = []
       for corner in ann["corners"]:
           parts.append(f"{float(corner[0]):.6f}")
           parts.append(f"{float(corner[1]):.6f}")
       return f"{class_id} " + " ".join(parts)
   ```

3. Add `_convert_obb_ndjson_to_txt` and `_rewrite_data_yaml_obb` thin wrappers (matching seg/pose pattern).

4. Replace the `train_yolo_obb` function signature: change `model_size: str = "s"` to `model: str = "yolo11s-obb"` and add `weights: Path | None = None`. Keep the same default OBB behavior (yolo11s-obb, not yolo26n since OBB uses v11 not v8). The body becomes a single call to `train_yolo_ndjson(...)` exactly like `train_yolo_seg`.

5. Remove the `model_size` parameter entirely. The default model string should be `"yolo11s-obb"` to match the existing `yolov8{model_size}-obb.pt` default of size "s" but using the current Ultralytics naming convention. (Note: check what the existing default model resolves to — `yolov8s-obb.pt` — so use `"yolov8s-obb"` as the default to maintain backwards compatibility with the same architecture. Ultralytics resolves `YOLO("yolov8s-obb.pt")` from either the full name or short name.)

Default model string: `"yolov8s-obb"` (preserves existing behavior; train_yolo_ndjson will do `YOLO(f"{model}.pt")`).

**cli.py:** Update the `yolo-obb` CLI command:
- Replace `--model-size` with `--model` (type=str, default="yolov8s-obb", help text describing full model variant name)
- Add `--weights` option (same as seg/pose commands)
- Update the function signature and the call to `train_yolo_obb` to pass `model=` and `weights=` instead of `model_size=`

**tests/unit/training/test_training_cli.py:**
- In `test_train_yolo_obb_help_shows_expected_flags`, replace `"--model-size"` with `"--model"` in the expected_flags list, and add `"--weights"`.
  </action>
  <verify>hatch run test tests/unit/training/test_training_cli.py -x</verify>
  <done>yolo_obb.py uses train_yolo_ndjson like seg/pose. CLI uses --model instead of --model-size. All CLI tests pass.</done>
</task>

</tasks>

<verification>
```bash
hatch run test tests/unit/training/ -x
hatch run check
```
</verification>

<success_criteria>
- yolo_obb.py follows identical pattern to yolo_seg.py (imports from common, delegates to train_yolo_ndjson)
- CLI yolo-obb command uses --model and --weights flags (consistent with seg/pose)
- build_yolo_training_data.py OBB NDJSON uses "annotations" key (compatible with convert_ndjson_to_txt)
- All tests pass, lint/typecheck clean
</success_criteria>

<output>
After completion, create `.planning/quick/13-convert-yolo-obb-py-to-use-ndjson-format/13-SUMMARY.md`
</output>
