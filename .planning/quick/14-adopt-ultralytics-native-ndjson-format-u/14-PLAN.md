---
phase: quick-14
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/build_yolo_training_data.py
  - src/aquapose/training/common.py
  - src/aquapose/training/yolo_obb.py
  - src/aquapose/training/yolo_pose.py
  - src/aquapose/training/yolo_seg.py
  - src/aquapose/training/cli.py
  - src/aquapose/training/__init__.py
  - tests/unit/training/test_yolo_pose.py
  - tests/unit/training/test_yolo_seg.py
  - tests/unit/training/test_common.py
  - tests/unit/training/test_training_cli.py
autonomous: true
requirements: [QUICK-14]
must_haves:
  truths:
    - "build_yolo_training_data.py emits Ultralytics-native NDJSON with dataset header line and flat annotation arrays"
    - "train_yolo_obb/seg/pose pass the .ndjson path directly to model.train(data=...) with no txt conversion"
    - "convert_ndjson_to_txt, rewrite_data_yaml, format_label_line functions are removed from codebase"
    - "All unit tests pass with hatch run test"
  artifacts:
    - path: "scripts/build_yolo_training_data.py"
      provides: "Ultralytics-native NDJSON output"
      contains: '"type": "dataset"'
    - path: "src/aquapose/training/common.py"
      provides: "Simplified common utilities (no NDJSON-to-txt plumbing)"
    - path: "src/aquapose/training/yolo_obb.py"
      provides: "Direct NDJSON training without format conversion"
    - path: "src/aquapose/training/yolo_pose.py"
      provides: "Direct NDJSON training without format conversion"
    - path: "src/aquapose/training/yolo_seg.py"
      provides: "Direct NDJSON training without format conversion"
  key_links:
    - from: "scripts/build_yolo_training_data.py"
      to: "model.train(data=...)"
      via: "Ultralytics-native NDJSON format"
      pattern: '"type": "dataset".*"type": "image"'
    - from: "src/aquapose/training/yolo_obb.py"
      to: "ultralytics.YOLO"
      via: "model.train(data=ndjson_path)"
      pattern: 'model\.train\(.*data='
---

<objective>
Adopt Ultralytics-native NDJSON format across the training pipeline. The build script
emits the native format directly (dataset header + flat annotation arrays), and the
training wrappers pass the .ndjson file path straight to model.train(data=...) without
any intermediate txt conversion or data.yaml rewriting.

Purpose: Eliminate ~150 lines of conversion plumbing (convert_ndjson_to_txt,
rewrite_data_yaml, format_label_line functions, train_yolo_ndjson orchestrator)
that exist only because our custom NDJSON format differs from what Ultralytics expects.

Output: Simplified training pipeline where data flows directly from build script to
Ultralytics without format translation.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@scripts/build_yolo_training_data.py
@src/aquapose/training/common.py
@src/aquapose/training/yolo_obb.py
@src/aquapose/training/yolo_pose.py
@src/aquapose/training/yolo_seg.py
@src/aquapose/training/cli.py
@src/aquapose/training/__init__.py
@tests/unit/training/test_yolo_pose.py
@tests/unit/training/test_yolo_seg.py
@tests/unit/training/test_common.py
@tests/unit/training/test_training_cli.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Convert build_yolo_training_data.py to emit Ultralytics-native NDJSON</name>
  <files>scripts/build_yolo_training_data.py</files>
  <action>
Update the three generate_*_dataset functions to emit Ultralytics-native NDJSON format.

**Dataset header line (first line of each .ndjson file):**
Each .ndjson file must start with a dataset metadata line:
```json
{"type": "dataset", "task": "obb|segment|pose", "class_names": {"0": "fish"}}
```
- OBB: `"task": "obb"`
- Seg: `"task": "segment"`
- Pose: `"task": "pose"`, also include `"kpt_shape": [6, 3]`

**Image lines (one per image/crop):**
Replace the current `{"image": ..., "annotations": [...]}` format with:
```json
{"type": "image", "file": "images/train/img.jpg", "width": W, "height": H, "split": "train", "annotations": {...}}
```

Note: `"annotations"` is a DICT (not a list) containing the task-specific key with flat arrays.

**OBB annotations format:**
Current: `{"class_id": 0, "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}`
New: `{"obb": [[cls, x1, y1, x2, y2, x3, y3, x4, y4], ...]}`
All coords remain normalized 0-1.

**Seg annotations format:**
Current: `{"class_id": 0, "polygon": [[x1,y1], [x2,y2], ...]}`
New: `{"segments": [[cls, x1, y1, x2, y2, ...], ...]}`
All coords remain normalized 0-1.

**Pose annotations format:**
Current: `{"class_id": 0, "bbox": [cx,cy,w,h], "keypoints": [[x,y,v], ...]}`
New: `{"pose": [[cls, cx, cy, w, h, x1, y1, v1, x2, y2, v2, ...], ...]}`
All coords remain normalized 0-1.

**Remove data.yaml generation** from each generate_*_dataset function — the NDJSON
header line replaces it entirely. The functions should still write to
`output_dir/{obb,pose,seg}/train.ndjson` and `output_dir/{obb,pose,seg}/val.ndjson`.

**Write a single combined .ndjson** per task type (not per split). The dataset header
goes first, then all image lines with their `"split"` field indicating train/val.
Output path: `output_dir/{obb,pose,seg}/dataset.ndjson` (single file).

Update the format_obb_annotation, format_pose_annotation, format_seg_annotation
functions to return the new flat-array format, OR inline the formatting since these
functions become simpler.

Keep the CLI, argument parsing, geometry utilities (pca_obb, affine_warp_crop, etc.)
unchanged.
  </action>
  <verify>
    python scripts/build_yolo_training_data.py --help runs without error.
    Verify format_obb_annotation / format_pose_annotation / format_seg_annotation
    produce flat arrays by inspecting the code (no runtime test needed since these
    require COCO data).
  </verify>
  <done>
    build_yolo_training_data.py generates Ultralytics-native NDJSON with:
    (1) dataset header line with type/task/class_names,
    (2) image lines with flat annotation arrays under obb/segments/pose keys,
    (3) no data.yaml files generated.
  </done>
</task>

<task type="auto">
  <name>Task 2: Simplify training wrappers to pass NDJSON directly to model.train()</name>
  <files>
    src/aquapose/training/common.py
    src/aquapose/training/yolo_obb.py
    src/aquapose/training/yolo_pose.py
    src/aquapose/training/yolo_seg.py
    src/aquapose/training/cli.py
    src/aquapose/training/__init__.py
    tests/unit/training/test_yolo_pose.py
    tests/unit/training/test_yolo_seg.py
    tests/unit/training/test_common.py
    tests/unit/training/test_training_cli.py
  </files>
  <action>
**common.py — Remove dead plumbing:**
- Delete `convert_ndjson_to_txt` function entirely
- Delete `rewrite_data_yaml` function entirely
- Delete `train_yolo_ndjson` function entirely
- Keep: EarlyStopping, MetricsLogger, save_best_and_last, make_loader (still used)

**yolo_obb.py — Simplify:**
- Delete `_format_obb_label_line`, `_convert_obb_ndjson_to_txt`, `_rewrite_data_yaml_obb`
- Remove import of `convert_ndjson_to_txt, rewrite_data_yaml, train_yolo_ndjson`
- Rewrite `train_yolo_obb` to:
  1. Find the `.ndjson` file in data_dir (e.g. `data_dir / "dataset.ndjson"`)
  2. Initialize YOLO model (same logic: `YOLO(str(weights))` if weights else `YOLO(f"{model}.pt")`)
  3. Call `model.train(data=str(ndjson_path), epochs=..., batch=..., device=..., project=..., name="train", imgsz=...)`
  4. Copy best/last weights to output_dir (same logic as current)
  5. Return best_model.pt path
- The `data_dir` parameter becomes the directory containing `dataset.ndjson`
- Keep same function signature (data_dir, output_dir, epochs, batch_size, device, val_split, imgsz, model, weights)

**yolo_pose.py — Same simplification as yolo_obb.py:**
- Delete `_format_pose_label_line`, `_convert_pose_ndjson_to_txt`, `_rewrite_data_yaml_pose`, `_POSE_YAML_KEYS`
- Rewrite `train_yolo_pose` same pattern: find ndjson, init model, train(data=ndjson), copy weights

**yolo_seg.py — Same simplification as yolo_seg.py:**
- Delete `_format_seg_label_line`, `_convert_seg_ndjson_to_txt`, `_rewrite_data_yaml_seg`
- Rewrite `train_yolo_seg` same pattern

**cli.py — Update help text only:**
- Change `--data-dir` help text from "Directory with data.yaml and NDJSON..." to "Directory containing dataset.ndjson file"
- No structural changes needed — CLI delegates to the simplified functions

**__init__.py — Remove dead exports:**
- Remove `train_yolo_ndjson` from imports and `__all__`
- Keep all other exports

**Tests — Update to match new code:**

test_common.py:
- Remove any tests that reference convert_ndjson_to_txt or rewrite_data_yaml (none currently exist, but verify imports)
- No changes needed if tests only cover EarlyStopping, MetricsLogger, save_best_and_last

test_yolo_pose.py:
- Remove tests for `_convert_pose_ndjson_to_txt` and `_rewrite_data_yaml_pose` (these functions no longer exist)
- The entire test file may reduce to just verifying `train_yolo_pose` is importable, or can be deleted if all tests tested removed functions
- If keeping the file, update imports to only reference `train_yolo_pose`

test_yolo_seg.py:
- Same treatment: remove tests for `_convert_seg_ndjson_to_txt` and `_rewrite_data_yaml_seg`
- Update or delete file accordingly

test_training_cli.py:
- Keep as-is — CLI help text tests still valid (seg, pose, yolo-obb subcommands still exist)
- Verify the import boundary test still passes
  </action>
  <verify>hatch run test -- tests/unit/training/ -x</verify>
  <done>
    (1) convert_ndjson_to_txt, rewrite_data_yaml, train_yolo_ndjson deleted from common.py,
    (2) _format_*_label_line / _convert_*_ndjson_to_txt / _rewrite_data_yaml_* deleted from all three wrappers,
    (3) train_yolo_{obb,pose,seg} pass ndjson directly to model.train(data=...),
    (4) hatch run test passes for training tests.
  </done>
</task>

</tasks>

<verification>
hatch run test -- tests/unit/training/ -x
hatch run check (lint + typecheck passes)
Grep codebase for convert_ndjson_to_txt, rewrite_data_yaml, format_label_line — zero hits outside git history
</verification>

<success_criteria>
- Zero references to convert_ndjson_to_txt, rewrite_data_yaml, or format_*_label_line in src/
- build_yolo_training_data.py output includes dataset header line with "type": "dataset"
- Training wrappers call model.train(data=str(ndjson_path)) directly
- All training tests pass
- Lint and typecheck pass
</success_criteria>

<output>
After completion, create `.planning/quick/14-adopt-ultralytics-native-ndjson-format-u/14-SUMMARY.md`
</output>
