# v2.1 Prospective: AquaPose

**Status:** Draft — interactive refinement in progress

---

## Bottleneck Ordering

Two major thrusts, roughly parallel:

1. **Evaluation infrastructure** — regression suite in CI, measurement pipelines. Must come first so improvements are measurable.
2. **Pipeline reordering** — move 2D tracking before midline extraction, associate on tracklets instead of per-frame. Fixes the root cause of 3D reconstruction failure (broken cross-view correspondence). Can begin immediately.
3. **Detection quality** — OBB detector for tighter crops, reducing multi-fish contamination and mask bleed.
4. **Midline quality** — keypoint pose estimation as the primary path, segmentation preserved as a backend.

The pipeline reorder and keypoint work are the two biggest changes. The reorder is arguably more urgent — 2D midlines and tracking already look decent, but 3D reconstruction is broken despite both backends (triangulation and curve optimizer) being available. The problem is structural, not algorithmic.

---

## Pipeline Reordering

### Current pipeline
```
Detection → Midline Extraction → Association → Tracking → Reconstruction
```

### Proposed pipeline
```
Detection → 2D Tracking → Cross-Camera Association → Midline Extraction → Reconstruction
```

**Why:** The current pipeline extracts midlines before knowing which fish it's looking at, then tries to associate noisy midlines across 12 cameras per frame. This produces broken cross-view correspondence, which makes 3D reconstruction a mess regardless of backend.

**What's broken now:** 3D splines are in the right general area of the tank but jump wildly frame-to-frame. Triangulation produces multi-meter splines that appear and disappear. Curve optimizer produces reasonable-length splines but they never land on fish and jump randomly. Both symptoms point to cross-view correspondence failure.

**The fix:** Track in 2D first (easy, well-solved problem — Hungarian matching on bboxes within a single camera view). Then associate *tracklets* across cameras using reprojected trajectory consistency (project tracklet A's centroid trajectory into camera B's view, check if it matches tracklet B). This uses many frames of evidence instead of single-frame geometry, making it far more robust.

**Key benefits of reordering:**
- 2D tracking within one camera is nearly trivial — smooth motion, rare identity confusion
- Tracklet association uses trajectory shape (hundreds of frames) instead of single-frame centroids
- Midline extraction happens only for confirmed tracklet-groups, not every detection
- Cross-camera association can inform head-tail ordering consistency
- 3D reconstruction inherits temporal smoothness from 2D tracklets by construction
- One bad frame doesn't break association because the tracklet has hundreds of frames of evidence

**Architectural note:** The Stage protocol and PipelineContext accumulator pattern are unchanged. This reorders stages, not the architecture. PipelineContext fields will shift (e.g., `tracks_2d` appears earlier, `annotated_detections` moves later).

---

## Candidate Requirements

### Phase 1: Foundations

**EVAL-01: Make regression suite runnable in CI**
7 regression tests skip without production video data. Provide a synthetic fixture so numerical equivalence is verified automatically. Fix the 1 xfail (midline golden data regeneration). CI already exists (test.yml, slow-tests.yml) — the gap is that regression tests silently skip.

**FLEX-01: Flexible keypoint count through the pipeline**
Current pipeline hardcodes 15 midline points. Refactor so `n_points` flows from config through MidlineStage, Midline2D, association, tracking, and reconstruction. Required before keypoint head (which outputs 6 points) can plug in.

### Phases 2-3: Pipeline Reorder

**REORDER-01: Per-camera 2D tracking**
Implement lightweight per-camera 2D tracking using Hungarian matching on bbox IoU/centroid distance. Each camera produces independent tracklets. This is a well-solved problem — smooth motion within a single camera view with rare identity confusion.

Output: per-camera list of 2D tracklets, each a sequence of detections over time with a local track ID.

**REORDER-02: Cross-camera tracklet association**
Associate tracklets across cameras using reprojected trajectory consistency. For each pair of tracklets with temporal overlap, project one tracklet's centroid trajectory into the other camera's view via calibration, compute median distance across shared frames. Low median = same fish.

Handles tracklet fragmentation at the cross-camera level — short high-confidence 2D tracklets are associated using multi-camera information to resolve gaps.

Output: tracklet groups, where each group represents one physical fish with its associated camera tracklets.

**REORDER-03: Deferred midline extraction**
Move midline extraction after association. Extract midlines only for detections belonging to confirmed tracklet-groups. Cross-camera association can inform head-tail ordering consistency (the arbitrary BFS endpoint problem).

**REORDER-04: Reconstruct from associated tracklets**
Triangulate using only the cameras known to observe each fish (from tracklet association). Per-fish, per-frame triangulation with known correspondence. No RANSAC needed for cross-view matching — correspondence is already established.

### Phase 4: Detection Improvement

**DET-01: Oriented bounding box detector**
Replace axis-aligned YOLO bounding boxes with an oriented bounding box (OBB) or ellipse-based detector. Benefits:
- Tighter crops reduce multi-fish contamination (current major error source)
- Less background in crop reduces mask bleed (current major error source)
- Free rough heading estimate from orientation angle
- Improves crop quality for both segmentation and keypoint backends

YOLO-OBB is trainable with the existing data pipeline. Same detection→crop flow, just tighter boxes.

### Phase 5: Keypoint Pose Estimation

**KP-01: Keypoint regression head on shared encoder**
Add a lightweight keypoint head to the existing MobileNetV3-Small encoder:
- Encoder bottleneck → AdaptiveAvgPool → FC(128) → FC(12) for 6 keypoints × 2 coords
- ~75K new parameters on top of existing 2.5M encoder
- Train keypoint head with encoder frozen on manually annotated data (starting ~200-500 crops)
- Unfreeze encoder for fine-tuning if accuracy insufficient
- Output: normalized crop-relative coordinates, transformed to full-frame via existing CropRegion

Architecture:
```
                    ┌─→ Segmentation Decoder → mask (existing, preserved)
Crop → Encoder →────┤
                    └─→ Keypoint Head → 6 × (x, y) (new)
```

Segmentation head weights are preserved. At inference, backend config selects which head to use. Shared encoder forward pass means running both heads adds negligible cost.

**KP-02: Keypoint training data (manual annotation)**
Hand-label 6 midline keypoints on ~200-500 fish crops. Small but high-quality dataset. Annotate with CVAT or similar tool. These become ground truth for both training and evaluation.

**KP-03: Midline keypoint backend**
Implement `"direct_pose"` backend (stub already exists in MidlineStage) that:
- Runs encoder + keypoint head on crops
- Outputs Midline2D with 6 points in full-frame coords
- Sets half_widths to None (acceptable loss — not load-bearing for reconstruction)
- Swappable with `"segment_then_extract"` via config

### Diagnostic Tooling (built alongside reorder phases)

**DIAG-01: Tracklet and association visualization**
Visualize 2D tracklets per camera (draw centroid trails on video) and cross-camera associations (color by association group). Essential for validating the reorder work — without this you're flying blind on whether tracklet association is working. Build incrementally as each REORDER step lands, not as an afterthought.

### Deferred (post-evaluation)

These become relevant after pipeline reorder and keypoint results are assessed:

- **Segmentation improvements** (encoder capacity, augmentation, training data) — pursue only if keypoint path doesn't meet accuracy needs
- **3D reconstruction benchmark** — synthetic ground truth for Chamfer/point-to-curve error
- **Curve optimizer vs. triangulation benchmark** — depends on reconstruction benchmark existing
- **Refractive calibration validation** — if pipeline reorder fixes cross-view correspondence but 3D reconstruction is still inaccurate, calibration error is the next suspect. Validate by triangulating a known static object visible in multiple cameras.
- **HDF5 schema documentation** — low effort, standalone, do whenever convenient

---

## Key Design Decisions

**Why reorder the pipeline?**
3D reconstruction is broken despite decent 2D midlines and working 2D tracking. The symptoms (multi-meter splines, frame-to-frame jumping, neither backend producing usable output) all point to cross-view correspondence failure. The current pipeline tries to associate per-frame across 12 cameras — a hard problem. Tracking in 2D first (easy) then associating tracklets (robust, uses trajectory-level evidence) fixes the correspondence problem structurally rather than trying to tune RANSAC parameters.

**Why keypoints over better segmentation?**
Mask bleed (loose masks bleeding off the fish body) is a fundamental boundary estimation problem made worse by training on pseudo-labels that themselves have bleed. Keypoint estimation reframes the problem as "where are N points on the body" — inherently more robust to noisy boundaries. The ordered chain topology (head→tail) also allows regularization that penalizes out-of-order points, which is free prior knowledge.

**Why keep segmentation as a backend?**
Segmentation still provides masks (useful for visualization) and half-widths. The shared encoder architecture means both backends are available at negligible additional cost. If keypoints work well, segmentation becomes a secondary diagnostic output rather than the primary midline source.

**Why OBB before keypoints?**
Multi-fish crop contamination and excess background both degrade any Stage 2 approach. OBB is upstream of the segmentation-vs-keypoints decision and benefits both paths. Better crops are a prerequisite for accurate keypoint annotation too — annotators need clean single-fish crops.

---

## Out of Scope for v2.1

- New module structure — Stage protocol and PipelineContext pattern unchanged
- GUI or web interface
- Real-time processing
- Dataset collection (hardware)
- Deployment/packaging (PyPI, Docker)
- New output formats beyond HDF5
- Multi-species generalization
- MOG2 backend validation

---

*Remaining requirements to be refined interactively.*
