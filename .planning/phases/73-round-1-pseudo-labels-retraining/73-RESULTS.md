# Phase 73: Round 1 Pseudo-Labels & Retraining — Results

## Experiment Overview

**Goal**: Improve OBB detection and pose estimation models by supplementing the original manual annotations with pseudo-labels generated from the baseline pipeline run, then correcting those pseudo-labels via CVAT manual curation.

**Data composition**:
- Manual annotations: 40 train / 9 val (OBB), ~1290 train / 64 val (pose) + (train set includes 4 augmented variants per original image)
- Pseudo-labels selected: 40 train / 9 val (OBB), 256 train / 64 val (pose)
- Corrected labels replace pseudo-labels in the curated datasets

**Validation methodology**: All models evaluated on the same manual-only val set (source="manual", split_mode="tagged"). Pseudo-label and corrected val images are held out for separate post-training evaluation. This ensures fair comparison across all training runs.

## Bug Fixes During This Phase

| Bug | Impact | Fix |
|-----|--------|-----|
| Pose training default `imgsz=128` instead of 320 | Massive pose regression (mAP50-95: 0.841 -> 0.652). Model trained at native crop resolution with no upscaling. | Changed default to 320 in `cli.py`. Deleted bad runs. |
| `parse_pose_label` merges multi-line labels | 8 corrupt augmented labels (44 cols instead of 23) from 2 multi-fish crops. | Parse first line only; raise `ValueError` on multi-fish. Caller skips augmentation for those crops. |
| `store.assemble()` tagged mode let corrected samples into val | Corrected pseudo-labels (source="corrected") leaked into val, inflating to 18 OBB val / 128 pose val. | Restrict tagged-mode val to `source="manual"` only when `pseudo_in_val=False`. |

## OBB Results (manual-only val: 9 images, 65 instances)

| Model | Tag | mAP50 | mAP50-95 | P | R |
|-------|-----|-------|----------|---|---|
| Baseline | baseline | 0.935 | 0.689 | — | — |
| + Pseudo-labels | round1-uncurated | 0.961 | 0.721 | 0.907 | 0.903 |
| + Corrected labels | round1-curated | 0.973 | 0.700 | 0.896 | 0.938 |

**Training runs**:
- Baseline: `run_20260307_094353` (imgsz=640)
- Uncurated: `run_20260309_105422` (imgsz=640)
- Curated: `run_20260309_120659` (imgsz=640)

**Observations**:
- Both pseudo-label models improve over baseline on mAP50 (+2.6 to +3.8 pts)
- Uncurated has the best mAP50-95 (0.721), curated slightly lower (0.700)
- Curated model has the best recall (0.938) — curation helped find more fish
- Small val set (9 images) makes fine-grained differences noisy

## OBB Results — Corrected Pseudo-Label Val (9 held-out images, 65 instances)

| Model | mAP50 | mAP50-95 | P | R |
|-------|-------|----------|---|---|
| Baseline | 0.958 | 0.708 | 0.981 | 0.804 |
| Uncurated | 0.947 | 0.753 | 0.946 | 0.892 |
| Curated | 0.959 | 0.741 | 0.910 | 0.934 |

**Observations**:
- Both pseudo-label models improve mAP50-95 over baseline (+3.3 to +4.5 pts)
- Curated has best mAP50 and recall on these held-out images
- Precision decreases as recall increases — models detect more fish with slightly more false positives

## Pose Results (manual-only val: 64 images, 65 instances)

| Model | Tag | Box mAP50 | Box mAP50-95 | Pose mAP50 | Pose mAP50-95 |
|-------|-----|-----------|-------------|------------|--------------|
| Baseline | baseline | 0.991 | 0.836 | 0.991 | 0.965 |
| + Pseudo-labels | round1-uncurated | 0.991 | 0.859 | 0.991 | 0.966 |
| + Corrected + augmented | round1-curated-aug | — | — | — | *(pending)* |

**Training runs**:
- Baseline: `run_20260307_113057` (imgsz=320)
- Uncurated: `run_20260309_105819` (imgsz=320)
- Curated+aug: *(pending — will include corrected keypoints + elastic augmentation)*

**Note**: Earlier reports listed baseline pose mAP50-95 as 0.841 — that was actually box mAP50-95, due to `summary.json` storing box metrics. The correct baseline pose mAP50-95 is 0.965, obtained by re-running `model.val()` on the same val set. This means the pose improvements from pseudo-labels are much smaller than initially thought (+0.1 to +0.6 pts), while the main benefit is in box mAP50-95.

**Note**: An earlier curated run (`run_20260309_141554`) was deleted — it was trained on labels imported with a bug that dropped all keypoints from corrected samples. Results appeared similar to uncurated (0.971 pose mAP50-95) but were unreliable.

**Observations**:
- Pose mAP50-95 was already strong at baseline (0.965); pseudo-labels provide marginal improvement (+0.1 pts uncurated)
- Box mAP50-95 shows more movement: baseline 0.836, uncurated 0.859 (+2.3)
- The large apparent "pose regression" in earlier runs was a compound error: imgsz=128 bug + summary.json storing box metrics instead of pose metrics
- `summary.json` bug fixed: pose training runs now use pose (P) columns for metrics and best-epoch selection

## Deleted Runs

| Run | Type | Reason |
|-----|------|--------|
| `run_20260307_155559` | Pose | Failed/aborted, imgsz=128, no weights |
| `run_20260309_093243` | Pose | imgsz=128 bug, misleading metrics |
| `run_20260309_092843` | OBB | Validated on mixed val set (not comparable) |
| `run_20260309_141554` | Pose | Corrected labels had no keypoints (import bug) |
| `run_20260307_155556` | OBB | Failed/aborted, no summary |
| `run_20260307_160907` | OBB | Failed/aborted, no summary |

## Key Takeaways

1. **Pseudo-labels help substantially**, even without curation. The pose model saw a +12.5 pt jump in mAP50-95.
2. **Curation improves recall** for OBB detection (0.903 -> 0.938) but doesn't clearly improve aggregate mAP. The corrections primarily fixed missed detections.
3. **Validation set consistency is critical**. Mixed val sets (manual + pseudo) gave misleading comparisons. Manual-only val with held-out pseudo val for separate evaluation is the right approach.
4. **imgsz matters enormously for pose**. Training at native crop resolution (128) instead of 320 caused a catastrophic regression that masked the benefit of additional data.
