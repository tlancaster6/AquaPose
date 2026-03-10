# Phase 78: Occlusion Investigation - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Characterize how the OBB detector and pose model behave when fish partially occlude each other. Produce a standalone investigation script, annotated video, written findings with go/no-go recommendation, and confidence threshold recommendation. No pipeline changes, no remediation — just observation and documentation.

</domain>

<decisions>
## Implementation Decisions

### Video visualization
- Crop-only view of the (263,225)-(613,525) region — no full frame, no side-by-side
- Annotations: OBB boxes (colored by track ID for tracked, gray for low-conf untracked, red for high-conf untracked), keypoints with connections (nose→head→spine1→spine2→spine3→tail), keypoint circle size encodes per-keypoint confidence (larger = higher)
- No detection confidence text overlay — keep frames clean
- Secondary (multi-instance) pose detections from within a single OBB crop are drawn in a distinct style (dashed connections, lighter shade) to distinguish from primary instance

### Go/no-go criteria
- Primary concern: keypoint identity jumps (keypoints from fish A assigned to fish B's detection during occlusion)
- No-go threshold: >20% of occlusion frames have single-frame keypoint glitches (single-frame jumps are tolerable individually since Kalman filter smooths them, but high frequency is not)
- OBB merging: acceptable if rare (<5 consecutive frames); only flag as a problem if persistent
- Confidence collapse during occlusion: document the pattern but not a no-go criterion on its own — low confidence during occlusion is expected and useful as a signal for the tracker to down-weight uncertain frames

### Confidence sweep
- Threshold range: 0.1 to 0.5 in 0.05 steps (9 levels, focused on the low end where quality vs false-positive tradeoff is most interesting)
- Metric: count-based — total detections, detections-per-frame distribution at each threshold, visual inspection of what gets filtered
- Scope: full 20-second clip (threshold recommendation must work globally, not just during occlusions)
- Results presented as a markdown table in the findings document (no plots)

### Multi-instance pose detection
- The current `PoseEstimationBackend._extract_keypoints()` takes only `kp.xy[0]`, discarding secondary instances detected by YOLO-pose within a single OBB crop
- Investigation script should extract all instances from `kp.xy` (not just `[0]`)
- Visualize secondary instances with distinct styling alongside primary
- Count frequency of multi-instance detections per frame, especially during occlusion windows
- This data informs whether leveraging secondary poses could be a remediation strategy (Phase 79) if go/no-go is borderline

### Summary format
- Location: `.planning/phases/78-occlusion-investigation/78-FINDINGS.md`
- Structure: observation-first — OBB behavior observations, keypoint behavior observations, confidence patterns, confidence sweep table, then go/no-go recommendation as conclusion
- Evidence: frame-number references in text plus a `screenshots/` directory with saved key frames from the annotated video
- If no-go: include a "Recommended Remediation" section listing specific failure modes and suggested approaches, directly feeding Phase 79

### Claude's Discretion
- Script CLI argument design (argparse flags, defaults)
- Exact color palette for track IDs (can reuse existing `_PALETTE_BGR` from evaluation viz)
- Screenshot selection criteria (which frames to save)
- How to detect/quantify keypoint identity jumps programmatically vs visual inspection
- Frame rate and codec for output video

</decisions>

<specifics>
## Specific Ideas

- The current pose backend at `src/aquapose/core/midline/backends/pose_estimation.py:541` takes `kp.xy[0]` — only first instance. YOLO-pose returns `(N_instances, K, 2)`. The investigation script should access all instances to evaluate multi-instance detection quality during occlusions.
- Target occlusion events are at the ~13-14 second mark of `e3v831e-20260218T145915-150429.mp4`
- The user's concern is specifically about whether the new OKS-based keypoint tracker (Phase 83) will be viable — keypoint identity jumps would poison tracking cost and cause ID switches

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `evaluation/viz/overlay.py`: `_PALETTE_BGR` color palette, `_fish_color()` helper, `cv2.VideoWriter` usage patterns
- `evaluation/viz/detections.py`: `_confidence_color()` red-to-green mapping, `_render_detection_mosaic()` OBB drawing logic
- `core/midline/backends/pose_estimation.py`: `PoseEstimationBackend` with `_extract_crop()` and `_extract_keypoints()` methods — crop extraction and YOLO-pose inference logic can be reused directly
- `core/midline/crop.py`: `extract_affine_crop()`, `invert_affine_points()` for coordinate transforms
- `core/detection/backends/yolo_obb.py`: OBB detection inference

### Established Patterns
- YOLO models loaded via `ultralytics.YOLO(weights_path)` with `.predict()` API
- Affine crop pipeline: OBB corners → 3-point affine → stretch-fill canvas (matching training data)
- All tensor→numpy via `.cpu().numpy()` (CUDA safety)
- Ultralytics OBB corner order: `[right-bottom, right-top, left-top, left-bottom]` — true TL is `pts[2]`

### Integration Points
- Script is standalone in `scripts/` — no pipeline integration needed
- Reads project config for model weights paths and camera setup
- Uses `FrameSource` for video frame loading with undistortion
- OC-SORT tracker at `core/tracking/ocsort_wrapper.py` for track ID assignment

</code_context>

<deferred>
## Deferred Ideas

- Leveraging multi-instance YOLO-pose detections as a remediation strategy for occlusion — evaluate feasibility in Phase 79 if no-go, or note as a future tracker enhancement if go

</deferred>

---

*Phase: 78-occlusion-investigation*
*Context gathered: 2026-03-10*
