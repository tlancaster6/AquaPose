# Phase 2: Segmentation Pipeline - Context

**Gathered:** 2026-02-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Produce binary fish masks for any input frame across all cameras. Covers YOLO/MOG2 detection (mutually exclusive), SAM2 pseudo-label generation, Mask R-CNN training/inference, and pipeline cleanup. Must achieve recall targets including low-contrast fish and stationary subjects. Pipeline accepts N fish as input even in single-fish v1 mode.

</domain>

<decisions>
## Implementation Decisions

### Detection strategy
- YOLO and MOG2 are mutually exclusive detector options — caller picks one
- MOG2 also serves a frame-selection role for YOLO training data (frames with high MOG2 activity are preferred)
- Global YOLO confidence threshold across all cameras — no per-camera tuning
- Camera exclusions handled by data organization (move bad cameras out of input folder), not code
- Crop dimensions match bbox dimensions in pixel coordinates — width and height vary independently (not fixed 256x256)

### Pseudo-label quality
- SAM2 uses crop+box-only prompt (no mask prompt) — dramatically better masks
- When SAM2 produces multiple masks, keep the largest area mask
- Quality filtering: reject masks with low YOLO detection confidence, bbox fill ratio outside [min_fill%, max_fill%], or mask area below min_area pixels
- Keep only the single largest connected region per mask
- Configurable visualization flag: `draw_pseudolabels=true/false`
- Opt-in caching: `create_cache` and `use_cache` flags on producer/consumer functions, both default to false (developer convenience, not user-facing)

### Training data pipeline
- Dataset format: COCO JSON
- Per-camera stratified 80/20 train/val split — each camera represented proportionally
- MOG2 activity-based frame sampling is for YOLO training only; Mask R-CNN uses all YOLO-detected crops
- Include negative examples (background crops with no fish) so model learns to predict "no fish"
- Single "fish" class — no sex-specific labels or evaluation subsets
- ImageNet-pretrained ResNet-50 backbone — standard transfer learning
- Standard augmentation: flips, rotations, brightness/contrast jitter

### Inference pipeline API
- Separate callable stages: detect(), crop(), segment() — not a single opaque pipeline
- Return crop-space mask + bbox/crop metadata — caller reconstructs full-frame via crop.py paste_mask()
- Default confidence threshold 0.1 — low threshold, high recall, callers filter further
- Batch frame API — accept multiple frames for GPU throughput
- Expect pre-extracted frames (numpy arrays/tensors) as input — caller handles video I/O
- Fish count per frame is N-max (9), but pipeline is per-camera, per-frame

### Cleanup
- Delete all Label Studio code completely — functions, imports, and label-studio-sdk dependency from pyproject.toml
- Delete all debug/test scripts: _debug_mask.py, _test_single.py, diagnose_mog2.py, verify_mog2_recall.py, etc.
- Refactor segmentation module into submodules: detector/, pseudo_labeler/, mask_rcnn/ — maps to separate pipeline stages

### Claude's Discretion
- MOG2 hyperparameters (history length, variance threshold, learning rate)
- SAM2 model variant selection
- Mask R-CNN training hyperparameters (learning rate schedule, epochs, etc.)
- Exact temporal sampling rate for YOLO training frames
- Augmentation intensity parameters
- Pseudo-label quality filter thresholds (min_fill, max_fill, min_area defaults)

</decisions>

<specifics>
## Specific Ideas

- Confidence threshold of 0.1 chosen to filter only the worst predictions while keeping high recall
- Crop utilities already exist in src/aquapose/segmentation/crop.py (CropRegion, compute_crop_region, extract_crop, paste_mask)
- The `scripts/` folder contains disorganized but relevant code from Phase 2 exploration — researchers and planners MUST scan scripts/*.py for reusable logic (pseudo-labeling, SAM2 invocation, dataset building, visualization) before writing new code

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-segmentation-pipeline*
*Context gathered: 2026-02-20*
