# Phase 37: Pipeline Integration - Context

**Gathered:** 2026-03-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire YOLO26n-seg and YOLO26n-pose Ultralytics models into the existing midline backend stubs as selectable backends within Stage 4 (Midline). Both backends operate on OBB-aligned crops from Stage 1 detections — tracking and cross-camera association are already solved before these models see any image. The pipeline produces `Midline2D` objects compatible with Stage 5 (Reconstruction).

</domain>

<decisions>
## Implementation Decisions

### Backend naming
- Rename backends from `segment_then_extract` / `direct_pose` to `segmentation` / `pose_estimation`
- Rename files: `segmentation.py` → `SegmentationBackend`, `pose_estimation.py` → `PoseEstimationBackend`
- Config key: `midline.backend: segmentation` or `midline.backend: pose_estimation`
- Default backend: `segmentation`
- Names describe the algorithmic approach, not the model — allows future non-YOLO models without config changes

### Failure handling
- Failed extractions return `AnnotatedDetection(midline=None)` — flagged empty midline, no exceptions
- Segmentation backend: skip skeletonization below a configurable minimum area threshold (existing config field likely already present)
- Pose backend: require at least 3 visible keypoints to attempt spline fitting — fewer can't define a meaningful curve
- Keypoint visibility determined by a configurable `confidence_floor` in MidlineConfig (default ~0.3) — keypoints below this threshold are treated as not visible

### Crop preparation
- Let Ultralytics handle all preprocessing (resize, pad, normalize) — pass raw OBB crop directly to model
- Use rotation-aligned (affine warp) crops from OBB detections, not axis-aligned bounding rects
- Coordinate back-projection (crop-space → full-frame) lives inside each backend, not shared — each backend is self-contained
- Critical: verify coordinate round-trips at crop→model→back-project boundary (cross-cutting v3.0 concern)

### Claude's Discretion
- Exact Ultralytics API calls for inference (model.predict vs model() etc.)
- Skeletonization algorithm choice for segmentation backend
- Spline fitting implementation details for pose backend
- Whether to batch crops or process one at a time per frame

</decisions>

<specifics>
## Specific Ideas

- Both backends are drop-in replacements for the old U-Net/modified-U-Net models — same crop-in, Midline2D-out contract
- The existing no-op stubs in `segment_then_extract.py` and `direct_pose.py` provide the exact interface to implement against
- GUIDEBOOK.md has been updated to reference YOLO26n-seg and YOLO26n-pose (replacing all U-Net/SAM references)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 37-pipeline-integration*
*Context gathered: 2026-03-01*
