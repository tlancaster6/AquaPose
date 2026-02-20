# Phase 2: Segmentation Pipeline - Context

**Gathered:** 2026-02-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Produce corrected binary fish masks for any input frame across all 13 cameras. Covers MOG2 detection, SAM pseudo-label generation, Label Studio annotation workflow, and Mask R-CNN training/inference. Must achieve recall targets including low-contrast females and stationary subjects. Pipeline accepts N fish as input even in single-fish v1 mode.

</domain>

<decisions>
## Implementation Decisions

### Detection strategy
- Global MOG2 parameters across all 13 cameras — no per-camera tuning
- Balanced recall/precision — avoid over-aggressive detection that pollutes pseudo-labels
- No secondary fallback for stationary fish absorbed into background — accept the gap; trained Mask R-CNN handles those cases later
- MOG2 outputs both bounding boxes AND rough foreground masks — both feed into SAM as prompts

### Annotation workflow
- All SAM pseudo-labels get human review and correction in Label Studio — no confidence-based skip
- Temporal sampling for frame selection (every Nth frame) — simple and predictable effort
- Include negative (empty/no-fish) frames in annotation set so model learns to predict "no fish"
- Random 80/20 train/validation split across all annotated frames

### Model training
- Mask R-CNN operates on fixed 256x256 crops around MOG2 detections — one fish per crop
- ImageNet-pretrained ResNet-50 backbone — standard transfer learning
- Single "fish" class (no male/female distinction) — track female IoU separately during evaluation, only intervene if target not met
- Standard augmentation: flips, rotations, brightness/contrast jitter

### Pipeline interface
- Return all detections above 0.1 confidence threshold — callers handle N-fish logic with cross-view global context
- Fish count per frame is N-max (9), but often fewer visible per camera — segmentation pipeline is per-camera, per-frame only
- RLE-encoded masks per detection, with bounding box and confidence score
- Batch frame API — accept batches of frames for GPU throughput
- Expect pre-extracted frames (numpy arrays/tensors) as input — caller handles video I/O

### Claude's Discretion
- MOG2 hyperparameters (history length, variance threshold, learning rate)
- SAM model variant and prompt engineering details
- Label Studio project configuration specifics
- Mask R-CNN training hyperparameters (learning rate schedule, epochs, etc.)
- Exact temporal sampling rate (every Nth frame — N TBD based on dataset size)
- Augmentation intensity parameters

</decisions>

<specifics>
## Specific Ideas

- Confidence threshold of 0.1 chosen to filter only the worst predictions while keeping high recall
- Female fish are the hardest case due to low contrast — the 0.85 IoU female subset target is the binding constraint
- Stationary fish will be missed by MOG2 but caught by the trained model — acceptable gap in the detection stage

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-segmentation-pipeline*
*Context gathered: 2026-02-19*
