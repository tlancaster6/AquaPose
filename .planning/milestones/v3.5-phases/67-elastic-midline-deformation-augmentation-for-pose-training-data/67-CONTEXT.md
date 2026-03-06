# Phase 67: Elastic Midline Deformation Augmentation - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate synthetically curved variants of manually annotated pose training images and labels to counteract the straight-fish bias in the training set. Offline pre-generation of deformed image+label pairs in YOLO format, consumed by the existing dataset assembly pipeline.

</domain>

<decisions>
## Implementation Decisions

### Deformation geometry
- C-curves: uniform arc (single signed curvature parameter), symmetric bending across the whole midline
- S-curves: sinusoidal (half-period sine wave along midline), amplitude + phase offset
- Curvature magnitude: moderate range, 10-30 degrees total bend
- Fish body length in crop is ~90-150px (not 200-400px) -- calibrate warp magnitudes accordingly
- Both head and tail shift symmetrically (no anchoring)

### Image warping
- Thin-plate spline (TPS) warp using 6 keypoints as control points
- Corner anchors: pin 4 crop corners as identity control points to constrain background distortion
- Background fill: cv2 BORDER_REPLICATE for regions pulled from outside the original crop
- OBB bounding box recomputed from deformed keypoints via pca_obb (not kept from original)

### Integration
- Augment manual annotations only (not pseudo-labels, which carry the straight-fish bias)
- 4 variants per original image: 2 C-curves (opposite signs) + 2 S-curves (opposite signs)
- Output includes original (undeformed) images + labels alongside deformed variants
- Standalone CLI command: `aquapose train augment-elastic --input-dir ... --output-dir ...`
- Output is YOLO-format directory that can be passed to `assemble` as the manual dir

### Validation
- Grid preview output: CLI generates a preview grid PNG showing original + deformed variants with keypoints overlaid
- A/B training comparison via documented workflow using existing train/compare infrastructure
- Dual metrics: overall pose mAP (no regression check) + curvature-stratified reprojection error (improvement target)

### Claude's Discretion
- Exact TPS implementation choice (scipy vs OpenCV)
- Random sampling distribution for curvature magnitude within the 10-30 degree range
- Preview grid layout details (rows, columns, image size)
- Naming convention for deformed image/label files

</decisions>

<specifics>
## Specific Ideas

- The problem: model predicts straight fish consistently across views, getting low reprojection error via consistency rather than accuracy. Curved fish are rare in training data.
- 4 variants give left/right symmetry for each curve type, ensuring no directional bias
- Body length ~90-150px means even moderate angular deformations are only a few pixels of displacement at the endpoints -- warp artifacts should be minimal

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `geometry.py:pca_obb()` -- computes OBB from keypoints, reuse for deformed keypoint OBB
- `geometry.py:format_pose_annotation()` -- formats YOLO pose label lines from keypoints
- `geometry.py:format_obb_annotation()` -- formats YOLO OBB label lines
- `geometry.py:affine_warp_crop()` and `transform_keypoints()` -- reference for coordinate transform patterns
- `dataset_assembly.py` -- existing assembly pipeline that will consume the augmented output

### Established Patterns
- YOLO-format dataset structure: `images/train/`, `labels/train/`, `dataset.yaml`
- Pose label format: `[cls, cx, cy, w, h, x1, y1, v1, x2, y2, v2, ...]` normalized to [0,1]
- 6 keypoints: nose, head, spine1, spine2, spine3, tail
- Manual annotation naming: `{frame}_{cam_id}.jpg` stems
- CLI commands live in `training/cli.py` as click groups under `aquapose train`

### Integration Points
- Output directory fed to `assemble_dataset(manual_dir=...)` as the manual annotation source
- `training/cli.py` for new `augment-elastic` subcommand registration
- Existing `train compare` CLI for A/B evaluation

</code_context>

<deferred>
## Deferred Ideas

- Hard example mining using pseudo-labeling machinery to focus manual annotation effort -- separate future phase
- Applying elastic deformations to pseudo-labels (currently manual-only due to bias concerns, may revisit when pseudo-label quality improves)

</deferred>

---

*Phase: 67-elastic-midline-deformation-augmentation-for-pose-training-data*
*Context gathered: 2026-03-05*
