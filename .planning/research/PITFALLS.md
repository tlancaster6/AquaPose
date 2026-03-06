# Pitfalls Research

**Domain:** Iterative pseudo-label retraining, curvature-stratified evaluation, and mixed-source training data management for multi-view fish pose estimation
**Researched:** 2026-03-06
**Confidence:** HIGH — pitfalls derived from codebase inspection (store.py, pseudo_labels.py, elastic_deform.py, evaluation stages), project memory (association tuning, augmentation experiment), and verified semi-supervised learning literature.

> **Scope note:** This file covers pitfalls specific to v3.6 Model Iteration & QA: running the pseudo-label retraining loop end-to-end, adding curvature-stratified and per-keypoint evaluation metrics, bootstrapping the data store from manual annotations, and mixing manual + pseudo + augmented labels. Prior milestone pitfalls (v3.4) are preserved below the milestone-specific section.

---

## v3.6 Model Iteration & QA Pitfalls

The primary risk sources for this milestone are:

1. **Confirmation bias in pseudo-label retraining** — the model reinforces its own errors across rounds, especially for underrepresented body poses.
2. **Train/val contamination from shared image sources** — pseudo-labels generated from the same video frames used for validation create information leakage.
3. **Elastic augmentation double-counting with pseudo-labels** — augmented samples of pseudo-labeled data compound noise without increasing effective diversity.
4. **Curvature-stratified evaluation with insufficient bin counts** — small sample sizes per bin produce unstable metrics that look like real signals.
5. **Short-run vs full-run behavioral differences** — models and metrics tuned on 1-minute clips may not generalize to full 5-minute runs.
6. **Algae domain shift between manual annotations and current conditions** — clean-tank training labels vs. algae-laden current conditions.
7. **OBB corner order mismatch between training and pseudo-label generation** — `pca_obb` returns [TL, TR, BR, BL] but Ultralytics inference returns [RB, RT, LT, LB].
8. **Store dedup silently dropping pseudo-labels that match manual annotation images** — content-hash dedup means same-frame pseudo-labels are skipped if a manual annotation exists.

---

### Pitfall P1: Confirmation Bias Amplifies Errors Across Retraining Rounds

**What goes wrong:**
Round 0 model produces pseudo-labels that reflect its systematic biases (e.g., underestimating curvature, missing low-contrast females, hallucinating detections on algae). Round 1 model trains on these biased labels. When round 1 generates pseudo-labels, those same biases are now reinforced with higher confidence. By round 2, the model has "learned" that curved fish are rare and algae patches are fish. The confidence scores in `compute_confidence_score` (pseudo_labels.py) measure reconstruction quality, not detection correctness — a confidently-wrong detection of algae at 3-camera consensus still gets score > 0.5.

**Why it happens:**
The confidence scoring formula (50% residual + 30% camera count + 20% variance) measures geometric consistency of the 3D reconstruction. An algae patch visible from multiple cameras can produce a geometrically consistent but semantically wrong reconstruction. The confidence score cannot distinguish "well-reconstructed wrong object" from "well-reconstructed fish." Additionally, the ~22% singleton rate means reconstructions come from a biased subset of fish appearances (those visible to 3+ cameras in good conditions), underrepresenting hard cases.

**How to avoid:**
- After each pseudo-label generation round, use `pseudo-label inspect` to visually audit at least 50 labels, specifically looking for: (a) non-fish objects with high confidence, (b) curved fish with low confidence being filtered out, (c) repeated detection of the same background feature across frames.
- Track per-round statistics: if the curvature distribution of pseudo-labels shifts toward straighter fish across rounds, confirmation bias is occurring.
- Use a fixed held-out validation set of manual annotations only (`pseudo_in_val=False` in `assemble()` — this is already the default). Never let pseudo-labels contaminate val.
- Consider capping pseudo-label contribution at a fixed ratio (e.g., no more than 3:1 pseudo:manual) to prevent the training signal from being dominated by self-generated labels.
- Compare round N metrics to round 0 baseline on the exact same val set. If val metrics plateau or degrade, stop iterating.

**Warning signs:**
- Val loss or mAP improves on pseudo-label-heavy training metrics but stagnates or regresses on manual-only val.
- Curvature distribution of pseudo-labels in round N is narrower (more straight-fish biased) than round N-1.
- Algae false positive rate does not decrease across rounds despite appearing to be filtered by confidence threshold.
- Detection count per frame increases across rounds (model becomes less selective, hallucinating more detections).

**Phase to address:**
Phase 73 (Round 1 Pseudo-Label Generation). Must establish monitoring before first retraining. Phase 74 decision checkpoint should explicitly check for these signs.

---

### Pitfall P2: Train/Val Leakage Through Temporal Proximity of Pseudo-Labeled Frames

**What goes wrong:**
Pseudo-labels are generated from pipeline diagnostic caches that process continuous video (200-frame chunks at 30fps). Temporal subsampling selects frames for the training set, but adjacent frames (even 1 second apart) contain nearly identical fish positions, poses, and backgrounds. If frame 100 ends up in training and frame 103 ends up in validation (both from the same chunk), the model memorizes the near-identical scene and val metrics are inflated. The `assemble()` method in `store.py` splits by shuffled sample order, not by temporal block — it has no awareness of frame temporal proximity.

Additionally, manual annotations (~50 full-frame images) come from the same video sequences. If any manual annotation frames overlap temporally with pseudo-label frames, train-val split can place a manual frame in val and a pseudo-labeled near-duplicate in train.

**Why it happens:**
The `SampleStore.assemble()` uses a random split of val-eligible samples. The content-hash dedup prevents exact duplicates, but near-duplicates (adjacent frames) have different pixel content and different hashes. The store has no temporal metadata (frame index, timestamp) in its split logic. The `pseudo_in_val=False` flag correctly keeps pseudo-labels out of val, but if a pseudo-labeled frame is 1 second from a manual val frame, the model has effectively seen the val scene during training.

**How to avoid:**
- For the short iteration clip (~1 min), use a temporal holdout: reserve the first or last 15 seconds as val-only. No pseudo-labels from those frames enter training. This is a stronger guarantee than random splitting.
- Add `frame_index` and `camera_id` to pseudo-label metadata at import time (the metadata dict in `import_sample` supports arbitrary keys). Use these to enforce temporal separation in dataset assembly.
- For this milestone's 1-2 rounds, the practical mitigation is simple: generate pseudo-labels only from the training temporal window, not the val window. Run pipeline on the full clip but exclude val-window frames from pseudo-label export.
- At minimum, verify: no pseudo-label sample ID shares the same `(camera_id, frame_index)` as any val sample. A post-assembly sanity check script should enforce this.

**Warning signs:**
- Val mAP or OKS is suspiciously high (> 0.95 on a dataset with known hard cases like low-contrast females).
- Val metrics improve dramatically when pseudo-labels are added to training, even though pseudo-labels are not in val — this suggests the pseudo-labels contain near-duplicates of val frames.
- Model performs well on val but poorly on the full 5-minute validation run (Phase 76), indicating overfitting to the short clip's temporal neighborhood.

**Phase to address:**
Phase 71 (Data Store Bootstrap) — establish temporal split convention before any data enters the store. Phase 72 (Baseline Pipeline Run) — confirm val set is temporally disjoint from training region.

---

### Pitfall P3: Elastic Augmentation Applied to Pseudo-Labels Compounds Noise

**What goes wrong:**
The seed document (v3.6-SEED.md Phase 73, step 5) specifies "Pose: manual + elastic augmentation + pseudo (Source A with confidence filtering)." If elastic augmentation (`generate_variants` in elastic_deform.py) is applied to pseudo-labeled samples, the TPS warp deforms images whose keypoint labels already have sub-pixel errors from the reprojection chain (3D spline -> refractive projection -> pixel). The deformation assumes ground-truth keypoint positions are accurate control points for the TPS warp. Pseudo-label keypoint errors of 2-5 pixels propagate through the TPS warp, producing misaligned deformed images where the fish body is warped but the keypoints do not match the warped anatomy. This creates training samples where the model learns to associate incorrect keypoint positions with the fish appearance.

**Why it happens:**
Elastic augmentation was designed and tested on manual annotations (v3.5 experiment: OKS slope improvement from -0.71 to -0.30). Manual annotations have sub-pixel accuracy. Pseudo-labels from 3D reprojection have residual errors of ~3 pixels (the per-camera residual threshold). The `generate_variants` function takes coordinates and visibility arrays at face value; it has no notion of label confidence or uncertainty. Applying the same augmentation pipeline to both manual (high quality) and pseudo (medium quality) labels without differentiation treats all labels as equally trustworthy.

**How to avoid:**
- Apply elastic augmentation only to manual-source samples, not pseudo-labeled samples. The `parent_id` and `source` fields in the store make this query straightforward: `store.query(source="manual")` for augmentation, then `store.query()` for assembly.
- The v3.6-SEED.md already suggests evaluating "whether elastic augmentation is still needed in round 2 — pseudo-label curvature diversity sampling may make it redundant." Take this seriously: if round 1 pseudo-labels already contain diverse curvatures (check with `compute_curvature` from pseudo_labels.py), elastic augmentation adds noise without adding diversity.
- If augmenting pseudo-labels is ever desired, restrict to high-confidence samples only (confidence > 0.8) and reduce the deformation angle range to (2.0, 5.0) degrees to limit the TPS warp magnitude.

**Warning signs:**
- OKS scores on curved fish decrease after adding augmented pseudo-labels, despite improving on the original augmented-manual experiment.
- Visual inspection of augmented pseudo-label crops shows misalignment between fish body and keypoint positions (keypoints floating off the fish midline).
- Training loss for pose model does not converge as cleanly as the baseline (noisy loss curve indicates conflicting gradient signals from misaligned labels).

**Phase to address:**
Phase 73 (Round 1 dataset assembly). Decision: apply augmentation only to manual samples. Phase 75 (Round 2) — re-evaluate whether pseudo-label curvature diversity makes augmentation redundant.

---

### Pitfall P4: Curvature-Stratified Evaluation With Unstable Bin Statistics

**What goes wrong:**
Phase 70 adds curvature-stratified reconstruction quality: "compute curvature from 3D spline, bin reconstructions into curvature quantiles, report reprojection error per bin." On a 1-minute clip (1800 frames, 9 fish, ~22% singleton rate), the reconstruction yield is roughly 1800 * 9 * 0.78 = ~12,600 fish-frame reconstructions. If binned into 5 curvature quantiles, each bin has ~2,500 samples — sufficient. But if binned into 10 quantiles, the extreme curvature bins (very straight and very curved) may have only 200-400 samples due to the non-uniform curvature distribution. Reporting mean reprojection error per bin with N=200 gives a standard error of ~0.2 px (assuming 3 px std), which is comparable to the differences between bins you are trying to measure. Worse: on a short clip, a single fish that swims in circles for 5 seconds dominates the high-curvature bin, making the metric a measurement of one individual, not a population statistic.

**Why it happens:**
Curvature-stratified evaluation is conceptually appealing but the sample sizes within extreme bins are small by nature — most fish are approximately straight most of the time, so the high-curvature bins are inherently data-poor. Using quantile binning (rather than fixed-threshold binning) ensures equal counts per bin, but the extreme-curvature bin still measures a narrow curvature range that may not be representative of the curvature range that matters for evaluation.

**How to avoid:**
- Use quantile binning (not fixed thresholds) to guarantee minimum sample count per bin. Report N per bin alongside mean and p90 error.
- Use 3-5 bins maximum for the short iteration clip. Reserve finer binning (10+) for the full 5-minute validation run in Phase 76.
- Report bootstrap confidence intervals (95% CI) per bin, not just point estimates. A 95% CI that overlaps between round 0 and round 1 means the difference is not significant — do not claim improvement.
- For the short clip, use the metric directionally (does the trend improve?) rather than as an absolute measurement. The Phase 76 full-run evaluation is where fine-grained curvature analysis becomes statistically meaningful.

**Warning signs:**
- Extreme curvature bins have fewer than 100 samples — any per-bin metric is noise.
- Mean error per bin fluctuates wildly between runs on the same clip (unstable due to small N).
- One bin shows dramatic improvement but visual inspection reveals it is dominated by one fish in one temporal segment.

**Phase to address:**
Phase 70 (Metrics & Comparison Infrastructure). Choose bin count and document minimum sample size requirements before implementation.

---

### Pitfall P5: Per-Keypoint Reprojection Error Conflates Z-Uncertainty With Pose Error

**What goes wrong:**
Phase 70 adds per-keypoint reprojection breakdown (head through tail). The tail keypoint will systematically show higher reprojection error than the head keypoint, and this will be interpreted as "the model is less accurate at the tail." But the true cause is different: the tail occupies a larger z-range during swimming (tail oscillation amplitude > head oscillation amplitude), and z-reconstruction uncertainty is 132x larger than XY. The tail's higher reprojection error reflects z-reconstruction noise propagated through refractive projection, not pose model inaccuracy. The head may actually be less accurate in the model (harder to localize on a featureless fish head) but appear better due to smaller z-oscillation.

**Why it happens:**
Reprojection error conflates two sources: (a) model localization error in 2D, and (b) 3D reconstruction error propagated back to 2D. For the AquaPose rig with top-down cameras, z-error is the dominant reconstruction error source. The tail moves more in z (swimming oscillation), amplifying source (b) for tail keypoints. A naive per-keypoint breakdown attributes all reprojection error to model quality, when the dominant variance is actually geometric.

**How to avoid:**
- Report per-keypoint error alongside per-keypoint z-range (standard deviation of z-coordinate across frames for each keypoint index). If high-error keypoints also have high z-variance, note that the error is z-dominated, not model-dominated.
- Consider XY-only reprojection error as a supplementary metric: project only the XY components of 3D points to 2D (ignoring z) and compare to observed keypoints. This isolates model accuracy from z-reconstruction noise.
- When comparing round 0 vs round 1, look at the relative change per keypoint, not absolute values. If the tail error decreases proportionally to the head error, the model improved uniformly; if the tail does not improve, the bottleneck is z-uncertainty, not the model.

**Warning signs:**
- Tail keypoints consistently show 2-3x higher reprojection error than head keypoints, independent of model version — this is the z-uncertainty baseline, not a model deficiency.
- Per-keypoint improvement from retraining is concentrated in the middle keypoints (spine1-spine3) and absent at head/tail — suggests the improvement is in body pose, not localization.
- Per-keypoint error is interpreted as a reason to add more training data for tail keypoints when the real bottleneck is z-reconstruction geometry.

**Phase to address:**
Phase 70 (Metrics & Comparison Infrastructure). Document the z-uncertainty confound when implementing per-keypoint analysis.

---

### Pitfall P6: Short-Run Metrics Do Not Predict Full-Run Performance

**What goes wrong:**
The iteration loop (Phases 72-75) uses ~1 minute clips (~1800 frames, ~9 chunks). The final validation (Phase 76) uses the full 5-minute clip (~9450 frames, ~47 chunks). Several pipeline behaviors change at longer durations:

- **OC-SORT tracker state accumulation**: Track IDs are carried across chunks via ChunkHandoff. In short runs, few track births/deaths occur. In long runs, ID swaps from tracker drift accumulate, and the association stage may see more fragmented tracklets.
- **Chunk boundary artifacts**: Identity stitching at chunk boundaries is tested 8 times in a short run vs 46 times in a full run. Rare stitching bugs that occur 1-in-20 boundaries are invisible in short runs but produce 2-3 visible artifacts in the full run.
- **Temporal variation in fish behavior**: A 1-minute clip may capture fish in a limited behavioral repertoire (e.g., all swimming slowly). The full clip may contain bursts of fast swimming, territorial interactions, or occlusion events that the short clip never samples.
- **Algae visibility is time-varying**: Suspended particles settle over time; lighting conditions shift slightly. The short clip's algae false positive rate may not represent the full clip.

**Why it happens:**
The seed document explicitly chose short clips for "fast loop turnaround" (~12 min runtime vs ~12 hours). This is a reasonable engineering tradeoff, but the risk is that iteration decisions are made on unrepresentative data. Pipeline stages that have O(T) or O(T^2) failure modes (tracking drift, association graph complexity) are masked in short runs.

**How to avoid:**
- Treat short-run iteration metrics as directional (better/worse than baseline) not absolute (final quality). State this explicitly in Phase 72 output.
- In Phase 76 (Full Validation), prepare for metrics that are worse than the short-run numbers. If reprojection error on the full run is 20% higher than the short run, that is expected — do not re-iterate based on this.
- Run `aquapose eval` on at least 3 non-overlapping 1-minute segments of the full clip to check for temporal stability. If metrics vary >30% across segments, the short run was unrepresentative.
- Use the short run to validate that models are not catastrophically worse, and the full run to validate that they are production-ready.

**Warning signs:**
- Short-run singleton rate is 18% but full-run singleton rate is 28% (tracker drift creates more fragmented tracklets → more association failures).
- Short-run overlay videos look clean but full-run overlay videos show ID swaps in the second half.
- Per-round improvement measured on short clips does not transfer to the full validation run.

**Phase to address:**
Phase 72 (Baseline Pipeline Run) — document that metrics are short-run-specific. Phase 76 (Final Validation) — expect and plan for metric regression from short-run numbers.

---

### Pitfall P7: Manual Annotation Domain Shift (Clean Tank vs. Algae Conditions)

**What goes wrong:**
The ~50 manual annotation images were created "when tank was freshly cleaned" (v3.6-SEED.md). Current conditions include "algae on tank walls causing persistent false positives." The OBB detection model's training data shows a clean background; the pseudo-labels come from a pipeline running on algae-contaminated video. If the baseline model (trained on clean-tank manual annotations) produces many algae false positives, those false detections propagate through tracking, association, and potentially into pseudo-labels (if the algae patch is visible from enough cameras to form a plausible 3D reconstruction). The confidence scoring will filter some but not all: a stationary algae blob visible from 4+ cameras with consistent reprojection will score above typical confidence thresholds.

**Why it happens:**
The domain gap between training data (clean tank) and inference data (algae present) is a classic distribution shift. The model has never seen algae-on-glass textures during training, so it cannot distinguish them from low-contrast fish. The confidence scoring measures geometric reconstruction quality, which is unrelated to semantic correctness — a well-triangulated algae patch scores just like a well-triangulated fish.

**How to avoid:**
- After Phase 72 (baseline pipeline run), manually count false positives in the overlay video. If algae FPs are significant (>5% of detections), add negative examples: crop algae regions, create empty-label files, and import into the store as `source=manual` with tag `negative_example`.
- Pseudo-label filtering should include a temporal consistency check: real fish move; algae patches are stationary. If a pseudo-labeled "fish" has centroid displacement < 1 pixel across 10+ consecutive frames, flag it for exclusion.
- For OBB pseudo-labels specifically, consider excluding labels whose 3D centroid is within 2cm of the tank wall (using the known tank geometry: cylindrical, 2m diameter, 1m tall). Algae grows on walls; fish do not rest against walls for extended periods.

**Warning signs:**
- Detection count per frame is higher than expected (9 fish, but model detects 12-15 per camera consistently).
- Pseudo-label generation produces labels at fixed positions across many frames (same pixel location, frame after frame — this is background, not a fish).
- After round 1 retraining, false positive rate does not decrease because the pseudo-labels themselves contained algae false positives that trained the round 1 model to detect algae.

**Phase to address:**
Phase 72 (Baseline Pipeline Run) — quantify algae FP rate before generating pseudo-labels. Phase 73 — add temporal-consistency and wall-proximity filters before importing pseudo-labels.

---

### Pitfall P8: OBB Corner Order Mismatch Between Training and Inference

**What goes wrong:**
This is a known project pitfall (CLAUDE.md): "Ultralytics OBB `obb.xyxyxyxy` returns corners as `[right-bottom, right-top, left-top, left-bottom]` — not `[TL, TR, BR, BL]`." The pseudo-label generation code (`pca_obb` in geometry.py) computes OBB corners and formats them via `format_obb_annotation`. If the corner order from `pca_obb` does not match what Ultralytics expects during training, the model learns a different corner convention than what it predicts at inference. The YOLO OBB format is `class x1 y1 x2 y2 x3 y3 x4 y4` with normalized coordinates, and Ultralytics internally converts to xywhr. If corners are provided in a different winding order, the internal xywhr conversion produces a different rotation angle, and the model trains on incorrect box rotations.

**Why it happens:**
`pca_obb` returns [TL, TR, BR, BL] order (consistent with training data convention). `format_obb_annotation` normalizes and formats for YOLO OBB. This path has been used for manual annotation conversion (`generate_obb_dataset` in coco_convert.py) and presumably works correctly there. The risk is in the pseudo-label path: `generate_fish_labels` and `generate_gap_fish_labels` in pseudo_labels.py call the same `pca_obb` and `format_obb_annotation`, so the corner order should be consistent. But any new code that constructs OBB annotations from Ultralytics inference results (e.g., converting detections back to training labels) must reverse the Ultralytics corner order, not use it directly.

**How to avoid:**
- Verify that `format_obb_annotation` produces the same corner order for both manual COCO conversion and pseudo-label generation. A unit test that round-trips a known OBB through `pca_obb` -> `format_obb_annotation` -> Ultralytics training -> Ultralytics inference -> compare should catch mismatches.
- Never use Ultralytics `obb.xyxyxyxy` directly for training labels. Always go through `pca_obb` which produces the training convention.
- When visualizing pseudo-labels for inspection, draw the OBB corners as a numbered polygon (corner 0, 1, 2, 3) to visually confirm the winding order matches the fish orientation (corner 0 should be at the head end, consistent with how `pca_obb` orients the box along the midline principal component).

**Warning signs:**
- OBB predictions at inference are rotated 90 degrees or 180 degrees from expected orientation.
- Training mAP for OBB is unusually low despite correct labels when visualized.
- Pseudo-labeled OBB boxes look correct when drawn as rectangles (position/size right) but the rotation angle disagrees with the fish body axis.

**Phase to address:**
Phase 71 (Data Store Bootstrap) — verify corner order consistency between manual conversion and pseudo-label paths.

---

### Pitfall P9: Store Content-Hash Dedup Silently Drops Same-Frame Pseudo-Labels

**What goes wrong:**
The `SampleStore.import_sample` uses SHA-256 of the image file for dedup. OBB pseudo-labels are full-frame images (the entire camera frame). If a manual annotation exists for the same frame (from the original ~50 annotated images), the image content hash will match, and the source-priority upsert logic (`SOURCE_PRIORITY = {"pseudo": 0, "corrected": 1, "manual": 2}`) will skip the pseudo-label import (manual priority 2 > pseudo priority 0). This is the correct behavior for that specific frame — you want the manual label, not the pseudo-label.

However, for pose pseudo-labels, the situation is different. Pose labels are crop-space, and each annotation generates a unique crop. If the crop geometry (OBB corners) differs slightly between manual annotation and pseudo-label (due to different OBB computation paths), the crop images will have different hashes and both will be imported — creating a near-duplicate with conflicting labels. The same fish in the same frame would have two label files with slightly different keypoint positions.

**Why it happens:**
The store was designed for image-level dedup, which works perfectly for full-frame OBB labels. Crop-level pose labels break the assumption: same fish, same frame, slightly different crop geometry produces a different image hash. The content-hash dedup does not catch semantic duplicates (same fish, same frame, different crop).

**How to avoid:**
- For pose label import, add metadata fields `{"camera_id": ..., "frame_index": ..., "fish_id": ...}` and check for semantic duplicates before import. If a manual-source sample exists for the same (camera, frame, fish), skip the pseudo-label.
- Alternatively, filter pseudo-labels at generation time: do not generate pseudo-labels for frames that are in the manual annotation set. The COCO JSON contains the annotated frame list; exclude those frames from pseudo-label generation.
- The simpler approach for v3.6: since there are only ~50 manual annotation frames and ~1800 pseudo-label candidate frames, the overlap is tiny. Import manual annotations first, then import pseudo-labels with awareness of which frames are already covered.

**Warning signs:**
- Store `summary()` shows more samples than expected after importing pseudo-labels for a clip that includes manually annotated frames.
- Pose training shows conflicting gradients (same crop, different keypoint positions) manifesting as oscillating loss.
- `data assemble` includes both manual and pseudo crops for the same fish/frame, placing them in different splits.

**Phase to address:**
Phase 71 (Data Store Bootstrap) — import manual annotations first. Phase 73 — exclude manual annotation frames from pseudo-label generation.

---

### Pitfall P10: Track Fragmentation Metric Confuses Occlusion Gaps With Detection Failures

**What goes wrong:**
Phase 70 adds "3D track fragmentation — post-association track count, gap count/duration stats, and continuity ratio." A gap occurs when a fish has no 3D reconstruction for one or more frames. Gaps have fundamentally different causes: (a) genuine occlusion (fish behind another fish, visible in <3 cameras), (b) detection failure (model missed the fish in enough cameras), (c) association failure (fish was detected but not associated across cameras). The gap detection code in `detect_gaps` (pseudo_labels.py) already classifies gaps as `no-detection`, `no-tracklet`, or `failed-midline`, but only for cameras where the InverseLUT says the fish should be visible. The track fragmentation metric needs to aggregate these gap classifications at the fish level, not just the camera level.

If gaps are counted without classification, an "improvement" in track continuity could mean: (a) the model genuinely detects more fish (good), (b) the model hallucinates more detections including false positives (bad), or (c) the association stage is more aggressive and merges unrelated tracklets (bad). Without gap classification, track continuity is ambiguous as a quality metric.

**Why it happens:**
Track continuity is an intuitively appealing metric — higher is better. But in multi-view pose estimation, the pipeline has multiple failure modes that can masquerade as improved continuity. A model with more false positives and a looser association threshold will report higher track continuity than a more selective model, despite producing worse 3D reconstructions.

**How to avoid:**
- Report track continuity alongside singleton rate, detection count per frame, and reprojection error. If continuity improves but singleton rate or error worsens, the improvement is spurious.
- Break down gap statistics by gap reason using the existing `detect_gaps` classifications. Report separate metrics: detection-gap-rate, tracking-gap-rate, midline-gap-rate.
- Define a "clean reconstruction frame" as: fish has reconstruction AND reprojection error < threshold. Report clean-frame continuity, not just reconstruction-present continuity.

**Warning signs:**
- Track continuity improves across rounds while reprojection error increases — suggests more false associations, not better detection.
- Gap count decreases but gap reason shifts from `no-detection` to `failed-midline` — means more detections but same midline failure rate.
- Detection count per frame increases beyond 9 (there are only 9 fish) — hallucinated detections filling gaps.

**Phase to address:**
Phase 70 (Metrics & Comparison Infrastructure). Gap classification must be part of the fragmentation metric design.

---

### Pitfall P11: COCO-to-YOLO Annotation Format Conversion Loses Visibility Information

**What goes wrong:**
Phase 71 converts manual COCO-JSON annotations to YOLO-OBB and YOLO-pose formats. The COCO keypoint format uses 3 visibility levels: `v=0` (not labeled), `v=1` (labeled but occluded), `v=2` (labeled and visible). The YOLO pose format also uses 3 levels but with different semantics: `0` (not visible), `1` (occluded), `2` (visible). The `parse_keypoints` function in `coco_convert.py` collapses visibility to boolean (`v > 0`), then `format_pose_annotation` in `geometry.py` writes `2` for visible and `0` for invisible. This correctly handles the dominant case but loses the COCO `v=1` (occluded but labeled) distinction, writing it as `v=2` (visible) in YOLO format. During training, the YOLO-pose loss treats `v=1` and `v=2` differently (COCO OKS uses visibility for sigma weighting). If occluded keypoints are marked as fully visible, the model is penalized for predicting positions of occluded keypoints with the same strictness as visible keypoints.

For pseudo-labels, this is less of an issue because `generate_fish_labels` uses the refractive projection validity check (which determines geometric visibility, not occlusion). But for manual annotations with hand-labeled occluded keypoints, the distinction matters.

**Why it happens:**
The conversion was likely written for the common case (all keypoints visible in the training crops) and the occluded-keypoint edge case was not tested. The existing COCO annotations for AquaPose may have very few `v=1` keypoints if the annotator only labeled visible keypoints.

**How to avoid:**
- Check the existing COCO annotations for `v=1` occurrences. If none exist, this pitfall is academic. If they do exist, update `format_pose_annotation` to preserve the 3-level visibility.
- For pseudo-labels, the current 2-level approach (visible=2, invisible=0) is correct — 3D-projected keypoints are either geometrically visible or not; there is no "occluded but labeled" concept in reprojection.

**Warning signs:**
- OKS evaluation treats all keypoints with equal sigma weighting when some should be down-weighted (occluded).
- Manual annotations that include partially-occluded fish have unexpectedly low OKS scores because occluded keypoints are marked visible and penalized for imprecise position.

**Phase to address:**
Phase 71 (Data Store Bootstrap) — check COCO annotations for v=1 keypoints during conversion.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip algae FP audit before pseudo-label generation | Saves 30 min of manual inspection | Algae FPs propagate into training data, amplified across rounds, requiring full data store cleanup | Never |
| Apply elastic augmentation to all samples uniformly | Simpler data assembly query | Noisy pseudo-label keypoints are deformed, producing misaligned training samples that degrade model accuracy on curved fish | Never — augment manual only |
| Use random train/val split without temporal separation | Simpler implementation | Inflated val metrics from near-duplicate leakage; model appears better than it is; false confidence in iteration results | Only if pseudo-labels are excluded from val AND manual annotations are temporally separated from the iteration clip |
| Hardcode curvature bin count | Simpler metric code | Unstable metrics on short clips; false claims of improvement in extreme curvature bins | Only as a default with N-per-bin reported alongside |
| Skip round 0 baseline metrics | Jump straight to iteration | No basis for comparison; impossible to measure improvement; cannot detect regression | Never |
| Trust `aquapose train compare` without visual inspection | Saves time on overlay review | Training metrics improve but pipeline metrics degrade (model learns to match training distribution, not real fish) | Never — visual inspection is the ground truth for this domain |

---

## Integration Gotchas

Common mistakes when connecting existing subsystems for the retraining loop.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| COCO-to-store import | Import images before labels, breaking the (image, label) atomicity expected by `import_sample` | Always provide matching image+label pairs to `import_sample`; validate label line count matches annotation count |
| Pseudo-label to store | Import all frames including those in the manual annotation set | Filter out frames that overlap with manual annotations before import |
| Dataset assembly | Assemble with `pseudo_in_val=True` for "more val data" | Always `pseudo_in_val=False` — pseudo-labels in val inflate metrics and mask model deficiencies |
| Elastic augmentation + store | Call `add_augmented` for pseudo-label parents | Only augment `source=manual` parents; pseudo-label keypoints are too noisy for TPS warping |
| Model registration | Register model without linking to dataset name | Always provide `dataset_name` — provenance tracking is the core benefit of the store; without it, you cannot trace which data produced which model |
| Config auto-update after training | Trust the auto-updated config without verifying weights_path resolves | After `train compare`, verify the new model's `weights_path` exists and is loadable before running the pipeline with it |

---

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Curvature-stratified eval on short clip | Extreme bins have <100 samples; metrics are noise | Use 3-5 bins max on short clips; reserve fine-grained binning for full run | Clips shorter than 2 minutes |
| Per-keypoint analysis without z-correction | Tail error always looks worst due to z-uncertainty, not model error | Report per-keypoint z-variance alongside error | Always — this is a fundamental confound of the rig geometry |
| Full pseudo-label generation for all cameras | 12 cameras * 1800 frames * 9 fish = 194k candidate labels; SQLite import takes 20+ min | Subsample temporally (every 5th frame) or spatially (best 6 cameras) for iteration runs | Datasets > 50k samples |
| Training on full pseudo-label set without confidence filtering | Large dataset but many low-quality labels drag down model | Filter at confidence > 0.5 for OBB, > 0.6 for pose (pose is more sensitive to label noise) | When pseudo-label count exceeds 3x manual count |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Data store bootstrap:** Manual annotations imported and baseline model trained — verify the train/val split in the store matches the curvature-stratified split used in the v3.5 augmentation experiment (otherwise baseline is not comparable).
- [ ] **Pseudo-label generation:** Labels produced and imported — verify no labels exist for frames in the manual annotation set (check for semantic duplicates, not just hash duplicates).
- [ ] **Round 1 training:** Model trained and registered — verify `aquapose train compare` was run against baseline AND overlay video was inspected for algae FPs.
- [ ] **Curvature-stratified metrics:** Implemented and producing output — verify N per bin is reported and extreme bins have statistically meaningful sample sizes.
- [ ] **Per-keypoint analysis:** Reprojection error per keypoint computed — verify z-variance is reported alongside to avoid misinterpreting geometric confounds as model deficiency.
- [ ] **Track fragmentation:** Gap count and continuity ratio reported — verify gaps are classified by reason (detection/tracking/midline) and reported separately.
- [ ] **Final validation run:** Full 5-minute run completed — verify metrics are compared to the same metrics from short-run baseline (not expected to match exactly; document the gap).
- [ ] **Provenance tracking:** All models and datasets registered in store — verify `store.list_models()` shows complete lineage from manual -> baseline -> round 1 -> round 2.

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Confirmation bias detected in round 2 | MEDIUM | Discard round 2 pseudo-labels; use round 1 model as final; add manual corrections for worst failure modes |
| Train/val leakage discovered post-training | HIGH | Re-split data with temporal separation; re-train from scratch; prior metrics are invalid |
| Augmented pseudo-labels degraded model | LOW | Remove augmented pseudo-label children from store via `exclude()`; re-assemble and re-train without augmented pseudo-labels |
| Curvature metrics unstable | LOW | Reduce bin count to 3 (low/medium/high); report only directional trends, not absolute values |
| Short-run metrics do not predict full-run | LOW | Expected and documented; use full-run metrics as ground truth; do not re-iterate based on short/full gap |
| Algae FPs propagated into pseudo-labels | MEDIUM | Query store for pseudo-labels with low centroid displacement across frames; `exclude()` stationary labels; re-assemble and re-train |
| OBB corner order mismatch | HIGH | Must re-convert and re-import all affected annotations; verify with visualization before re-training |
| Store semantic duplicates causing conflicting labels | MEDIUM | Query store for samples sharing (camera_id, frame_index) metadata; `exclude()` lower-priority duplicates |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| P1: Confirmation bias | Phase 73-74 (retraining + evaluation) | Track curvature distribution of pseudo-labels per round; stop if val metrics plateau |
| P2: Train/val temporal leakage | Phase 71 (bootstrap) | Assert no pseudo-label frames within 30 frames of any val frame |
| P3: Augmentation of pseudo-labels | Phase 73 (dataset assembly) | Assembly query explicitly restricts augmentation to `source=manual` |
| P4: Curvature bin instability | Phase 70 (metrics infrastructure) | Report N per bin; CI overlap check between rounds |
| P5: Per-keypoint z-confound | Phase 70 (metrics infrastructure) | Per-keypoint z-variance column in output alongside error |
| P6: Short vs full run divergence | Phase 72 + Phase 76 | Document expected metric gap in Phase 72; verify in Phase 76 |
| P7: Algae domain shift | Phase 72 (baseline run) | Count algae FPs in overlay video; add negative examples if >5% of detections |
| P8: OBB corner order | Phase 71 (conversion) | Round-trip test: manual -> YOLO -> Ultralytics train -> predict -> compare corners |
| P9: Store semantic dedup | Phase 71 + Phase 73 | Post-import query for (camera, frame) duplicates across sources |
| P10: Track fragmentation ambiguity | Phase 70 (metrics) | Gap-reason breakdown in fragmentation output |
| P11: Visibility level collapse | Phase 71 (conversion) | Check COCO annotations for v=1 counts; preserve if present |

---

## Sources

- [Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning](https://arxiv.org/pdf/1908.02983) — foundational work on confirmation bias in pseudo-labeling
- [Confidence-Driven Pseudo-Label Optimization](https://www.emergentmind.com/topics/confidence-driven-pseudo-label-optimization) — threshold-free adaptive pseudo-label selection
- [Semi Supervised Learning: 2025 Best Practices](https://labelyourdata.com/articles/semi-supervised-learning) — current best practices overview
- [Data Leakage in Visual Datasets](https://arxiv.org/html/2508.17416v1) — near-duplicate and split-level leakage analysis
- [Class Imbalance in Object Detection: An Experimental Diagnosis](https://arxiv.org/html/2403.07113v1) — class imbalance in one-stage detectors
- [Ultralytics OBB Datasets Overview](https://docs.ultralytics.com/datasets/obb/) — OBB format specification and corner conventions
- [Ultralytics YOLO Training Docs](https://docs.ultralytics.com/modes/train/) — training configuration and data format
- AquaPose codebase direct inspection: `store.py` (dedup logic, assemble), `pseudo_labels.py` (confidence scoring, gap detection), `elastic_deform.py` (augmentation pipeline), `coco_convert.py` (format conversion), `evaluation/stages/reconstruction.py` (current metrics) — all integration points verified from source
- AquaPose project memory: association tuning results, elastic augmentation experiment outcomes, known LUT coordinate space bug — project-specific context

---

> **Previous milestone pitfalls (v3.4) are preserved below for reference.**

---

## v3.4 Performance Optimization Pitfalls (archived)

> These pitfalls covered retrofitting batching, async I/O, and vectorization into the existing synchronous pipeline. See git history for the full v3.4 pitfalls document. Key pitfalls that remain relevant:
>
> - **P2 (CUDA OOM from over-batching):** Still relevant for YOLO training with large datasets. Training batch size must be tunable.
> - **P6 (TF32 precision):** Still relevant for evaluation metric consistency across GPU generations. Document TF32 state for all training and evaluation runs.
> - **P8 (GPU tensor leak):** Still relevant when processing large numbers of pseudo-label candidate frames through the pipeline.

---
*Pitfalls research for: Iterative pseudo-label retraining, curvature-stratified evaluation, and mixed-source training data management (v3.6 Model Iteration & QA)*
*Researched: 2026-03-06*
