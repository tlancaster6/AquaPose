---
created: 2026-02-20T17:34:31.064Z
title: Consolidate scripts into CLI workflow
area: tooling
files:
  - scripts/test_mog2.py
  - scripts/diagnose_mog2.py
  - scripts/test_sam2.py
  - scripts/verify_pseudo_labels.py
  - scripts/verify_mog2_recall.py
  - scripts/sample_yolo_frames.py
  - scripts/organize_yolo_dataset.py
  - scripts/train_yolo.py
  - scripts/eval_yolo_vs_mog2.py
---

## Problem

9 ad-hoc scripts accumulated in `scripts/` during segmentation phases (02, 02.1, 02.1.1). No clear entry point, some overlap, no documented run order. A future contributor (or agent) would struggle to know which scripts are still relevant and how they relate.

**Script inventory by origin:**

| Script | Phase | Purpose | Reusable? |
|--------|-------|---------|-----------|
| `test_mog2.py` | 02.1 | Visual MOG2 detection test on 2 cameras | One-shot debug |
| `diagnose_mog2.py` | 02.1 | Detailed MOG2 diagnostics | One-shot debug |
| `verify_mog2_recall.py` | 02.1 | MOG2 recall against GT masks | One-shot eval |
| `test_sam2.py` | 02.1 | SAM2 pseudo-label evaluation | One-shot eval |
| `verify_pseudo_labels.py` | 02.1 | Pseudo-label quality check | One-shot eval |
| `sample_yolo_frames.py` | 02.1.1 | MOG2-guided frame sampling for annotation | Reusable if retraining |
| `organize_yolo_dataset.py` | 02.1.1 | Label Studio export → ultralytics split | Reusable if retraining |
| `train_yolo.py` | 02.1.1 | YOLOv8 training wrapper | Reusable |
| `eval_yolo_vs_mog2.py` | 02.1.1 | YOLO vs MOG2 recall comparison | Reusable |

## Solution

Consider:

1. **Triage**: Remove or archive one-shot debug scripts (test_mog2, diagnose_mog2, verify_mog2_recall, test_sam2, verify_pseudo_labels) — they served phase 02.1 troubleshooting and won't be needed post-v1.

2. **Document the YOLO retraining workflow** as a clear sequence:
   `sample_yolo_frames.py` → Label Studio annotation → `organize_yolo_dataset.py` → `train_yolo.py` → `eval_yolo_vs_mog2.py`

3. **Optional consolidation**: CLI entry point (`python -m aquapose.cli train-detector`, `python -m aquapose.cli eval-detector`) or Makefile/hatch script targets wrapping the reusable scripts.

4. **Add `--resume` flag** to `train_yolo.py` (missing — had to use ultralytics CLI directly to resume interrupted training).
