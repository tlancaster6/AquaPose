# Switch from NDJSON to YOLO txt label format throughout the library

## Context
Ultralytics accepts NDJSON as an input format but internally converts it to the classic YOLO layout (images/ + labels/ with per-image .txt files + dataset.yaml) before training. This conversion relies on the `datasets_dir` setting, which is fragile and caused a training failure when it pointed to the wrong directory.

## What to change
1. **`scripts/build_yolo_training_data.py`** — output YOLO txt labels directly instead of NDJSON. Each image gets a `.txt` file with one annotation per line. Generate `dataset.yaml` with paths/nc/names.
2. **`tmp/convert_all_annotations.py`** — same change for the multi-source converter.
3. **`src/aquapose/training/yolo_obb.py`**, **`yolo_pose.py`**, **`yolo_seg.py`** — pass `data=dataset.yaml` instead of `data=dataset.ndjson`. Remove any NDJSON-specific logic.
4. **`src/aquapose/training/cli.py`** — update `--data-dir` help text to reference txt labels.

## YOLO txt format reference
- OBB: `class_id x1 y1 x2 y2 x3 y3 x4 y4` (normalized, space-separated)
- Pose: `class_id cx cy w h x1 y1 v1 x2 y2 v2 ...` (normalized)
- Seg: `class_id x1 y1 x2 y2 ...` (polygon vertices, normalized)
- One `.txt` per image, same stem, in a parallel `labels/` directory
- `dataset.yaml` with `path`, `train`, `val`, `nc`, `names`, and `flip_idx`
- **Pose `dataset.yaml` must include `flip_idx: [0, 1, 2, 3, 4, 5]`** (identity — midline keypoints have no handedness). Without this, Ultralytics silently disables `fliplr` augmentation, causing the model to only see nose-left crops and learn a positional bias instead of visual features.

## Why
- Eliminates dependency on ultralytics NDJSON conversion (which requires `datasets_dir` to be correct)
- YOLO txt is the universal standard — works with any Ultralytics version
- Simpler debugging (can inspect individual label files)
