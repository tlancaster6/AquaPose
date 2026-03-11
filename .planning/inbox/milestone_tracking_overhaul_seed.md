# Milestone Seed: Tracking Overhaul

## Goal

Replace the current OC-SORT-on-OBB-centroids tracking with a custom bidirectional keypoint-based tracker that exploits the full pose data. Reorder the pipeline so pose estimation precedes tracking, drop the legacy segmentation midline backend, and upgrade the association stage to use anatomical keypoints instead of OBB centroids.

## Context

- Current pipeline order: Detection → Tracking → Association → Midline → Reconstruction
- Tracking uses BoxMot's OC-SORT on OBB centroids — loses all pose information
- Segmentation-based midline extraction has been superseded by pose estimation for multiple versions
- Cross-view association uses OBB centroids, which shift under occlusion/clipping
- Design doc for the custom tracker: `.planning/inbox/fish_tracker_v1_design.md`

## Phases

### Phase 1: Occlusion Investigation

Informal evaluation of how the OBB detector and pose model behave when fish partially occlude each other:

- Does the OBB detector produce separate boxes for overlapping fish?
- When two fish share a crop, does pose estimation anchor to the focal fish?
- How do per-keypoint confidences behave during occlusion?
- Identify any failure modes that would undermine a keypoint-based tracker.

Approach: Standalone script in `scripts/` that takes a camera ID, frame range, and optional crop region. Runs detection on all frames, then pose estimation on every detection (not just tracked ones). Generates a video visualization where:
- Tracked detections: OBB + midline in per-track-ID color with track ID label
- Untracked detections: OBB + midline in gray/red
- Keypoint confidence encoded as point size/opacity

Test clip: `e3v831e-20260218T145915-150429.mp4`, ~13-14 second mark. A stationary fish at the upper-left of the school has two fish swim underneath it sequentially — gives two occlusion events on a predictable target. Crop region for visualization: (263,225)→(613,525).

Deliverable: written summary of findings with example frames, go/no-go for proceeding.

### Phase 2: Occlusion Remediation (conditional)

If Phase 1 reveals problems (e.g., merged boxes, keypoints jumping between fish), address them before building the tracker. Scope TBD based on findings — could range from pose model retraining to NMS tuning to detection filtering heuristics.

Skip this phase if Phase 1 findings are acceptable.

### Phase 3: Baseline Metrics

Establish quantitative tracking baselines from the current pipeline for comparison after the overhaul.

**Perfect-tracking target**: `e3v83eb-20260218T145915-150429.mp4`, 1:50-2:30 (frames 3300-4500, 1200 frames at 30fps, 40 seconds). All 9 fish remain in frame. Less white-wall exposure and fewer out-of-tank false positives than the previous e3v831e clip.

Target: 9 tracks, each ~40s long, zero fragmentation.

Metrics:
- Track count (target: exactly 9)
- Track duration distribution (target: all ~20s)
- Fragmentation count (number of ID switches)
- Total coverage (fraction of fish × frames with a track assignment)

Run the current OC-SORT tracker on this clip and record actual performance as the gap to close. The full 5-minute multi-camera diagnostic run (`~/aquapose/projects/YH/runs/run_20260307_140127/`) is available as a secondary reference if needed, but the single-camera perfect-tracking target is the primary baseline to avoid downstream pipeline stages confusing the metrics.

Deliverable: baseline metrics document with specific numbers to compare against.

### Phase 4: Pipeline Reorder & Segmentation Removal

- Move pose estimation (currently in MidlineStage, Stage 4) to run immediately after detection (Stage 2), before tracking.
- Remove the segmentation midline backend entirely (backends/segmentation.py, skeletonization code in midline.py, orientation resolution logic that only applied to segmentation).
- Ensure MidlineStage (or its replacement) no longer depends on tracklet_groups for filtering — pose runs on all detections.
- Update PipelineContext and stage interfaces accordingly.

### Phase 5: Association Upgrade (Keypoint Centroid)

Minimally-invasive upgrade to cross-view association:

- Replace the OBB centroid used for ray-based cross-view matching with a mid-body keypoint from the pose estimate.
- The mid-body keypoint (interpolated index ~7-8, raw keypoint ~3) is anatomically stable and doesn't shift under partial occlusion or frame-edge clipping like OBB centroids do.
- Determine the optimal keypoint index empirically (highest average confidence across the dataset).
- The rest of the association machinery (forward LUTs, ray-ray distance, Leiden clustering) remains unchanged.
- Populate Tracklet2D.centroids from the selected keypoint instead of OBB center.

### Phase 6: Custom Tracker Implementation

Build the bidirectional batched keypoint tracker per the V1 design doc:

- OKS-based association cost (replaces IoU on OBBs)
- OCM direction consistency via spine heading vector
- Forward + backward pass with greedy merge
- 60-dim Kalman filter (15 keypoints × position + velocity) — or consider 24-dim (6 raw keypoints only) to avoid correlated measurement noise
- Asymmetric track birth/death based on frame-edge proximity
- Chunk boundary handoff via serialized KF state
- OC-SORT mechanisms: ORU (re-update after occlusion), OCR (secondary association for lost tracks)

### Phase 7: Integration & Evaluation

- Wire the custom tracker into the reordered pipeline
- Run on both baseline scenarios (full 5-min clips and the 60-second perfect-tracking target)
- Compare against Phase 3 baselines
- Iterate on tracker parameters if needed

### Phase 8: Code Quality Audit & CLI Smoke Test

Post-integration review to ensure the overhaul hasn't introduced regressions or rough edges:

- Code quality audit: dead code from removed segmentation backend, unused imports, broken cross-references, type errors, test coverage gaps
- CLI smoke test: confirm the full pipeline runs end-to-end from the CLI with the new stage ordering, config options are documented and validated, error messages are clear

Deliverable: list of issues found (if any).

### Phase 9: Cleanup (conditional)

Address issues identified in Phase 8. Scope TBD based on findings.

Skip this phase if Phase 8 finds nothing actionable.

## Design Decisions to Resolve During Planning

- **KF state dimension**: Track all 15 interpolated keypoints (60-dim) or only the 6 raw keypoints (24-dim)? The latter avoids correlated noise but loses body-shape discrimination in OKS.
- **Which mid-body keypoint** for association centroid: determine empirically from confidence statistics.
- **BoxMot dependency**: After the custom tracker is complete, boxmot can be removed as a dependency. Decide whether to keep OC-SORT as a fallback or remove entirely.

## Out of Scope

- Appearance/ReID embeddings (V2)
- Global optimization (min-cost flow) for tracklet merge (V2)
- Chunk overlap for boundary reconciliation (V2, if needed)
- Full 15-point pose-aware cross-view association (beyond the keypoint centroid swap)
