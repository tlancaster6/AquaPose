---
phase: 02-segmentation-pipeline
status: human_needed
score: 3/4
verified: 2026-02-19
---

# Phase 2: Segmentation Pipeline Verification

## Goal
The system can produce corrected binary fish masks for any input frame across all 13 cameras, achieving recall targets even for low-contrast females.

## Must-Have Verification

### 1. MOG2 detection produces padded bounding boxes with >= 95% per-camera recall
- **Status:** PARTIAL - Code verified, recall on real footage not yet measured
- **Evidence:** MOG2Detector implemented with morphological cleanup, shadow exclusion, configurable min_area and padding. 12 unit tests pass on synthetic data. Recall on held-out sample including female fish requires real data testing.
- **What's missing:** Human must run detector on representative real footage and confirm >= 95% recall

### 2. SAM pseudo-labels can be generated and imported into Label Studio
- **Status:** PASS (code-level)
- **Evidence:** SAMPseudoLabeler class implemented with box+mask prompts. export_to_label_studio creates valid task JSON with brush RLE masks. 21 unit tests pass including round-trip and COCO conversion. SAM2 requires separate install for actual GPU inference.
- **Human verification needed:** Import generated tasks into Label Studio, verify masks display correctly

### 3. Trained Mask R-CNN achieves >= 0.90 mean mask IoU (>= 0.85 female-only)
- **Status:** PARTIAL - Architecture verified, training on real data not yet performed
- **Evidence:** MaskRCNNSegmentor built with torchvision maskrcnn_resnet50_fpn_v2, proper head replacement. Training script handles SGD, StepLR, checkpointing. evaluate() computes per-image mask IoU. @slow tests confirm training completes on synthetic data.
- **What's missing:** Training on real annotated data and IoU measurement against targets

### 4. Segmentation pipeline accepts N fish as input
- **Status:** PASS
- **Evidence:** MOG2Detector.detect returns list[Detection] (variable N). SAMPseudoLabeler.predict accepts list[Detection]. MaskRCNNSegmentor.predict returns list[list[SegmentationResult]] per image. All APIs handle variable-length lists.

## Score: 3/4 must-haves verified at code level

## Human Verification Required

The following items require human testing with real data:

1. **MOG2 recall on real footage** - Run MOG2Detector on representative frames from all 13 cameras. Count detections vs. visible fish. Must hit >= 95% recall. Pay special attention to female fish and stationary subjects.

2. **Label Studio round-trip** - Generate pseudo-labels on a sample, import into Label Studio, verify masks appear and can be edited. Export and confirm import_from_label_studio reads them back.

3. **Mask R-CNN training and IoU** - After annotation is complete, run train() on the corrected dataset and evaluate(). Must achieve >= 0.90 mean IoU overall and >= 0.85 on female-only subset.

## Automated Checks

| Check | Result |
|-------|--------|
| All unit tests pass (109 non-slow) | PASS |
| Lint clean (ruff) | PASS |
| Type check clean (basedpyright) | PASS |
| All public symbols importable | PASS |
| SUMMARY.md exists for all 3 plans | PASS |
| Git commits present for all plans | PASS (7 commits) |
