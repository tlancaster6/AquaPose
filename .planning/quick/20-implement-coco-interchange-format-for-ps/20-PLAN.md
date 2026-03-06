---
phase: 20-implement-coco-interchange-format-for-ps
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/training/coco_interchange.py
  - src/aquapose/training/__init__.py
  - src/aquapose/training/pseudo_label_cli.py
  - src/aquapose/training/dataset_assembly.py
  - tests/unit/training/test_coco_interchange.py
autonomous: true
requirements: [COCO-EXPORT, COCO-IMPORT]

must_haves:
  truths:
    - "pseudo-label generate writes coco_keypoints.json next to each pose dataset.yaml"
    - "pseudo-label assemble converts COCO JSON back to YOLO labels when YOLO labels are absent"
    - "YOLO -> COCO -> YOLO round-trip preserves annotation data within float tolerance"
  artifacts:
    - path: "src/aquapose/training/coco_interchange.py"
      provides: "yolo_pose_to_coco, write_coco_keypoints, coco_to_yolo_pose"
      exports: ["yolo_pose_to_coco", "write_coco_keypoints", "coco_to_yolo_pose"]
    - path: "tests/unit/training/test_coco_interchange.py"
      provides: "Unit tests for COCO interchange"
      min_lines: 80
  key_links:
    - from: "src/aquapose/training/pseudo_label_cli.py"
      to: "src/aquapose/training/coco_interchange.py"
      via: "write_coco_keypoints call in generate()"
      pattern: "write_coco_keypoints"
    - from: "src/aquapose/training/dataset_assembly.py"
      to: "src/aquapose/training/coco_interchange.py"
      via: "coco_to_yolo_pose fallback in collect_pseudo_labels()"
      pattern: "coco_to_yolo_pose"
---

<objective>
Implement COCO Keypoints interchange format for pseudo-label round-trip editing.

Purpose: Enable editing pseudo-labels in external annotation tools (CVAT, Label Studio) by exporting YOLO-Pose to COCO Keypoints JSON during `generate` and importing COCO JSON back during `assemble`.

Output: New `coco_interchange.py` module with bidirectional conversion, wired into the existing CLI pipeline, with full test coverage.
</objective>

<execution_context>
@/home/tlancaster6/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlancaster6/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/coco-interchange-plan.md
@src/aquapose/training/__init__.py
@src/aquapose/training/pseudo_label_cli.py
@src/aquapose/training/dataset_assembly.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Create coco_interchange.py with bidirectional converters</name>
  <files>src/aquapose/training/coco_interchange.py, tests/unit/training/test_coco_interchange.py</files>
  <behavior>
    - test_yolo_pose_to_coco_basic: Create minimal YOLO-Pose dir (1 image 128x64, 1 label with 2 fish each having 13 keypoints), convert, verify: 1 image entry with correct dims, 2 annotations with absolute-pixel bbox [x_min, y_min, w, h] and absolute-pixel keypoints, visibility preserved, category has 13 keypoint names
    - test_yolo_pose_to_coco_invisible_keypoints: YOLO "0 0 0" entries map to COCO "0 0 0" (not-labeled)
    - test_coco_to_yolo_pose_basic: Create COCO JSON with 2 images and 3 annotations, convert, verify 2 .txt files with correct normalized bbox and keypoints
    - test_coco_to_yolo_pose_invisible: COCO visibility=0 maps to YOLO "0 0 0"
    - test_round_trip: YOLO -> COCO -> YOLO produces identical label content within float tolerance (atol=1e-5)
  </behavior>
  <action>
    Create `src/aquapose/training/coco_interchange.py` with three public functions per the detailed plan at `.planning/coco-interchange-plan.md`:

    1. `yolo_pose_to_coco(pose_dir: Path, n_keypoints: int) -> dict` — Reads labels/train/*.txt and images/train/*.jpg. For each label file: find matching image, read dimensions with PIL Image.open().size (fast, no decode). Parse each YOLO line `[cls, cx, cy, w, h, x1, y1, v1, ...]`. Convert normalized bbox to absolute `[x_min, y_min, w, h]`. Convert normalized keypoints to absolute pixels. YOLO visibility 2 -> COCO 2 (visible), YOLO visibility 0 -> COCO 0 (not labeled). Build images/annotations/categories arrays with sequential IDs. Category keypoints: `["kp_0", "kp_1", ..., "kp_{n-1}"]`, skeleton: [].

    2. `write_coco_keypoints(pose_dir: Path, n_keypoints: int) -> Path` — Calls yolo_pose_to_coco, writes JSON with indent=2 to pose_dir/coco_keypoints.json.

    3. `coco_to_yolo_pose(coco_path: Path, output_labels_dir: Path, img_dir: Path | None = None) -> int` — Load COCO JSON. Build image_id -> image_info lookup. Group annotations by image_id. For each image: get width/height from COCO entry, convert absolute bbox to normalized [cx, cy, w, h], convert absolute keypoints to normalized, write .txt file named by image file_name stem. Return count of files written.

    Use PIL.Image for reading dimensions (already available via ultralytics dependency). No pycocotools needed.

    Write tests FIRST following TDD red-green-refactor cycle. Use tmp_path fixture to create synthetic YOLO directories with small dummy JPEG images (create with PIL).
  </action>
  <verify>
    <automated>hatch run test -- tests/unit/training/test_coco_interchange.py -x</automated>
  </verify>
  <done>All 5+ tests pass. yolo_pose_to_coco produces valid COCO structure, coco_to_yolo_pose produces valid YOLO labels, round-trip is identity within float tolerance.</done>
</task>

<task type="auto">
  <name>Task 2: Wire converters into generate and assemble CLI pipelines</name>
  <files>src/aquapose/training/pseudo_label_cli.py, src/aquapose/training/dataset_assembly.py, src/aquapose/training/__init__.py</files>
  <action>
    1. **pseudo_label_cli.py** — In `generate()`, after the existing block at ~lines 577-583 that writes dataset.yaml and confidence.json for pose consensus and gap, add:

    ```python
    from aquapose.training.coco_interchange import write_coco_keypoints

    # After line 578 (_write_confidence_json for consensus):
    write_coco_keypoints(pseudo_dir / "pose" / "consensus", n_keypoints)

    # After line 583 (_write_confidence_json for gap), inside the `if gaps:` block:
    write_coco_keypoints(pseudo_dir / "pose" / "gap", n_keypoints)
    ```

    Place the import at the top of the file with other local imports.

    2. **dataset_assembly.py** — In `collect_pseudo_labels()`, add COCO fallback at the top of the `for run_dir in run_dirs:` loop body (after computing `run_id`, before checking `conf_path`). For pose model types (base_path starts with "pose/"):

    ```python
    coco_path = run_dir / "pseudo_labels" / base_path / "coco_keypoints.json"
    labels_dir = run_dir / "pseudo_labels" / base_path / "labels" / "train"
    if coco_path.exists() and (not labels_dir.exists() or not any(labels_dir.iterdir())):
        labels_dir.mkdir(parents=True, exist_ok=True)
        from aquapose.training.coco_interchange import coco_to_yolo_pose
        n_converted = coco_to_yolo_pose(coco_path, labels_dir)
        logger.info("Converted %d COCO annotations to YOLO-Pose in %s", n_converted, labels_dir)
    ```

    This is a lazy import to avoid circular imports and because it only runs in the COCO fallback path.

    3. **__init__.py** — Add imports and __all__ entries for `yolo_pose_to_coco`, `write_coco_keypoints`, and `coco_to_yolo_pose` from `.coco_interchange`.
  </action>
  <verify>
    <automated>hatch run test -- tests/unit/training/ -x</automated>
  </verify>
  <done>All training tests pass including the new coco_interchange tests. `hatch run check` passes (lint + typecheck). write_coco_keypoints is called in generate() for both consensus and gap pose dirs. collect_pseudo_labels() auto-converts COCO sources when YOLO labels are absent.</done>
</task>

</tasks>

<verification>
1. `hatch run test -- tests/unit/training/test_coco_interchange.py -x` — all converter tests pass
2. `hatch run test -- tests/unit/training/ -x` — no regressions in training tests
3. `hatch run check` — lint and typecheck pass
</verification>

<success_criteria>
- coco_interchange.py exists with yolo_pose_to_coco, write_coco_keypoints, coco_to_yolo_pose
- Round-trip YOLO -> COCO -> YOLO preserves annotations within float tolerance
- generate() writes coco_keypoints.json for consensus and gap pose dirs
- collect_pseudo_labels() falls back to COCO conversion when YOLO labels missing
- All tests pass, lint clean, types check
</success_criteria>

<output>
After completion, create `.planning/quick/20-implement-coco-interchange-format-for-ps/20-SUMMARY.md`
</output>
