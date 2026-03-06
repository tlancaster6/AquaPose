# COCO Interchange Format — Implementation Plan

## Goal

Enable round-trip editing of pseudo-labels in external annotation tools (CVAT,
Label Studio) by:

1. **Exporting** YOLO-Pose pseudo-labels to COCO Keypoints JSON during
   `pseudo-label generate`
2. **Importing** COCO Keypoints JSON (potentially hand-corrected) during
   `pseudo-label assemble`

---

## Part 1: YOLO-Pose → COCO Keypoints Export

### What

After generating pose pseudo-labels, write a `coco_keypoints.json` file next to
each `dataset.yaml` (i.e., one per pose source directory). This runs
unconditionally — no flag needed.

### Output locations

```
run_dir/pseudo_labels/pose/consensus/coco_keypoints.json
run_dir/pseudo_labels/pose/gap/coco_keypoints.json
```

### COCO Keypoints JSON structure

```json
{
  "images": [
    {"id": 1, "file_name": "images/train/000001_cam0_000.jpg", "width": 128, "height": 64}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, w, h],
      "area": float,
      "keypoints": [x1, y1, v1, x2, y2, v2, ...],
      "num_keypoints": int,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "fish",
      "supercategory": "animal",
      "keypoints": ["kp_0", "kp_1", ...],
      "skeleton": []
    }
  ]
}
```

Key differences from YOLO-Pose format:
- `bbox` is absolute pixels `[x_min, y_min, width, height]` (not normalized center)
- `keypoints` are absolute pixels (not normalized), visibility uses COCO convention
  (0=not labeled, 1=labeled-not-visible, 2=labeled-visible)
- Multiple annotations per image (multi-fish crops) each get their own entry
- Image dimensions are explicit in the `images` array

### Implementation

**New file**: `src/aquapose/training/coco_interchange.py`

Two public functions:

```python
def yolo_pose_to_coco(
    pose_dir: Path,
    n_keypoints: int,
) -> dict:
    """Convert a YOLO-Pose dataset directory to COCO Keypoints dict.

    Reads all label files from pose_dir/labels/train/*.txt and
    image dimensions from pose_dir/images/train/*.jpg.

    Args:
        pose_dir: Directory containing images/train/ and labels/train/.
        n_keypoints: Number of keypoints per annotation.

    Returns:
        COCO-format dict ready for json.dump().
    """
```

Logic:
1. List all `.txt` files in `labels/train/`
2. For each label file, find the matching image and read its dimensions
   with `cv2.imread` (just shape, or use PIL `Image.open().size` for speed)
3. Parse each line: `[cls, cx, cy, w, h, x1, y1, v1, ...]`
4. Convert normalized bbox to absolute `[x_min, y_min, w, h]`
5. Convert normalized keypoints to absolute pixels
6. Assign sequential image IDs and annotation IDs
7. Return the assembled COCO dict

```python
def write_coco_keypoints(pose_dir: Path, n_keypoints: int) -> Path:
    """Write coco_keypoints.json next to dataset.yaml in pose_dir."""
    coco = yolo_pose_to_coco(pose_dir, n_keypoints)
    out = pose_dir / "coco_keypoints.json"
    out.write_text(json.dumps(coco, indent=2))
    return out
```

**Modification**: `src/aquapose/training/pseudo_label_cli.py` — `generate()`

After the existing `_write_pose_dataset_yaml` / `_write_confidence_json` calls
(around line 568-574), add:

```python
from aquapose.training.coco_interchange import write_coco_keypoints

# Write COCO Keypoints JSON for external annotation tools
write_coco_keypoints(pseudo_dir / "pose" / "consensus", n_keypoints)
if gaps:
    write_coco_keypoints(pseudo_dir / "pose" / "gap", n_keypoints)
```

**Update `__init__.py`**: Add `write_coco_keypoints` and `yolo_pose_to_coco` to
`src/aquapose/training/__init__.py` and `__all__`.

### Tests

**New file**: `tests/unit/training/test_coco_interchange.py`

- `test_yolo_pose_to_coco_basic`: Create a minimal YOLO-Pose directory (1 image,
  1 label with 2 fish), convert, verify COCO structure: correct image count,
  annotation count, bbox in absolute pixels, keypoints in absolute pixels,
  visibility values preserved.
- `test_yolo_pose_to_coco_invisible_keypoints`: Verify `0 0 0` YOLO entries map
  to `0 0 0` in COCO (not-labeled).
- `test_round_trip`: YOLO → COCO → YOLO produces identical label files
  (within float rounding).

---

## Part 2: COCO Keypoints → YOLO-Pose Import in `assemble`

### What

When `pseudo-label assemble --model-type pose` encounters a
`coco_keypoints.json` instead of (or alongside) YOLO label files, convert the
COCO annotations back to YOLO-Pose format on-the-fly before assembly.

### Detection strategy

In `dataset_assembly.py`'s `collect_pseudo_labels()`, after resolving the
label/image directories, check for `coco_keypoints.json` in the source dir.
If found AND the labels/train/ directory is empty or missing, treat it as a
COCO-format source and convert.

More specifically, add a new function:

```python
def _maybe_convert_coco_source(source_dir: Path, n_keypoints: int) -> None:
    """If source_dir has coco_keypoints.json but no YOLO labels, convert in-place."""
```

This should be called at the start of `collect_pseudo_labels()` for pose model
type. It writes YOLO label files into `labels/train/` so the rest of the
assembly pipeline works unchanged.

### Implementation

**Add to `coco_interchange.py`**:

```python
def coco_to_yolo_pose(
    coco_path: Path,
    output_labels_dir: Path,
    img_dir: Path | None = None,
) -> int:
    """Convert COCO Keypoints JSON to YOLO-Pose label files.

    Args:
        coco_path: Path to coco_keypoints.json.
        output_labels_dir: Directory to write .txt label files.
        img_dir: Optional image directory to read dimensions from.
            If None, uses dimensions from COCO images entries.

    Returns:
        Number of label files written.
    """
```

Logic:
1. Load COCO JSON
2. Build `image_id → image_info` lookup
3. Group annotations by `image_id`
4. For each image:
   - Get `width`, `height` from COCO image entry
   - For each annotation: convert absolute bbox → normalized `[cx, cy, w, h]`,
     convert absolute keypoints → normalized, map visibility
   - Write one `.txt` file per image (filename = image `file_name` stem)
5. Return count of files written

**Modification**: `src/aquapose/training/dataset_assembly.py`

In `collect_pseudo_labels()`, before scanning for YOLO label files, add:

```python
from aquapose.training.coco_interchange import coco_to_yolo_pose

coco_path = base_dir / "coco_keypoints.json"
labels_dir = base_dir / "labels" / "train"
if coco_path.exists() and (not labels_dir.exists() or not any(labels_dir.iterdir())):
    labels_dir.mkdir(parents=True, exist_ok=True)
    n_converted = coco_to_yolo_pose(coco_path, labels_dir)
    logger.info("Converted %d COCO annotations to YOLO-Pose in %s", n_converted, labels_dir)
```

This keeps the rest of the assembly pipeline untouched — it just sees YOLO
label files as usual.

**Update `__init__.py`**: Add `coco_to_yolo_pose`.

### Tests

Add to `tests/unit/training/test_coco_interchange.py`:

- `test_coco_to_yolo_pose_basic`: Create a COCO JSON with 2 images, 3
  annotations, convert, verify YOLO label files exist with correct content.
- `test_coco_to_yolo_pose_invisible`: Verify COCO visibility=0 maps to
  `0 0 0` in YOLO.
- `test_round_trip` (shared with Part 1): YOLO → COCO → YOLO identity.
- `test_collect_pseudo_labels_coco_fallback`: Mock a pseudo-label directory
  with coco_keypoints.json but no YOLO labels, verify `collect_pseudo_labels`
  converts and loads correctly.

---

## File Change Summary

| File | Action |
|------|--------|
| `src/aquapose/training/coco_interchange.py` | **New** — `yolo_pose_to_coco`, `write_coco_keypoints`, `coco_to_yolo_pose` |
| `src/aquapose/training/__init__.py` | Add new public symbols to imports and `__all__` |
| `src/aquapose/training/pseudo_label_cli.py` | Call `write_coco_keypoints` in `generate()` (~3 lines) |
| `src/aquapose/training/dataset_assembly.py` | Add COCO fallback in `collect_pseudo_labels()` (~6 lines) |
| `tests/unit/training/test_coco_interchange.py` | **New** — ~6 test functions |

### Dependencies

No new dependencies. Uses only `json`, `pathlib`, `cv2` (or `PIL`) already in
the environment. COCO format is just a JSON schema — no `pycocotools` needed
for basic read/write.

---

## Execution Order

1. Create `coco_interchange.py` with all three functions
2. Write tests for the converter functions
3. Wire `write_coco_keypoints` into `generate()`
4. Wire COCO fallback into `collect_pseudo_labels()`
5. Run full test suite
6. Commit
