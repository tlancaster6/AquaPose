# Phase 32: YOLO-OBB Detection Backend - Context

**Gathered:** 2026-02-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Add YOLO-OBB as a selectable detection model (`detector_kind: yolo_obb`) that produces rotation-aligned affine crops and OBB polygon overlays in diagnostic mode. Affine crop utilities support back-projection from crop to frame coordinates. The regular `yolo` detector remains the default and is unaffected.

</domain>

<decisions>
## Implementation Decisions

### Affine crop behavior
- Proportional padding around the OBB (percentage of OBB dimensions, not fixed pixels)
- Fixed rectangular canvas with aspect ratio preserved via letterbox padding — fish are elongated, square crops distort proportions
- Crop dimensions are configurable via config (e.g. `crop_size: [256, 128]`), not hardcoded
- Always use the affine crop utility, even for non-OBB detections (angle=0 produces identity rotation) — one unified code path

### OBB overlay styling
- OBB polygon color matches fish ID color (same palette as tracklet trails)
- OBB polygon replaces axis-aligned bounding box — no reason to draw both, the OBB IS the detection
- Labels show fish ID + confidence score (same as existing AABB overlays)
- No orientation axis line — the polygon shape itself conveys orientation

### Model loading & training
- Fine-tune a pre-trained YOLOv8-OBB model on fish data (not training from scratch)
- Training set already exists at: `C:\Users\tucke\aquapose\projects\YH\models\obb\training_set`
- Model output convention (applies to ALL models, not just OBB):
  - Training output goes to: `<project_dir>/models/<model_shorthand>/` (internal structure per model defaults)
  - Best weights copied to: `<project_dir>/models/<model_shorthand>_best.pt` (or `.pth` as appropriate)
  - For OBB: shorthand is `obb`, so `projects/YH/models/obb/` and `projects/YH/models/obb_best.pt`

### Fallback & compatibility
- Same confidence threshold filtering as regular YOLO — no separate OBB threshold
- Always use affine transform regardless of rotation angle (no threshold fallback to axis-aligned)
- Both midline backends (segment-then-extract AND direct_pose) use OBB rotation-aligned crops when OBB data is available
- Unified crop code path: non-OBB detections flow through the same affine utility with angle=0

### Claude's Discretion
- Exact proportional padding percentage (likely 10-20% range)
- Affine interpolation method (bilinear vs bicubic)
- OBB polygon line thickness and style details
- Internal YOLO-OBB inference details (batch size, NMS parameters)

</decisions>

<specifics>
## Specific Ideas

- Fish have very uneven aspect ratios — rectangular crops with letterbox padding are essential, not square
- The model weights convention (`<shorthand>_best.pt` copied up to models root) is a project-wide standard, not OBB-specific
- Training set is already prepared — no label generation pipeline needed for this phase

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 32-yolo-obb-detection-backend*
*Context gathered: 2026-02-28*
