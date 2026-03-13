---
phase: 78-occlusion-investigation
plan: 02
subsystem: investigation
tags: [occlusion, yolo-obb, yolo-pose, ocsort, findings, go-no-go]

requires:
  - phase: 78-01
    provides: investigation script
provides:
  - Written findings characterizing OBB and pose behavior under occlusion
  - GO recommendation for proceeding to tracker implementation
  - Confidence threshold recommendation (0.25, or 0.1 with polygon NMS)
  - Detection of white-wall background dropout issue
affects: [78.1-obb-pose-production-retrain, 80-baseline-metrics, 83-custom-tracker]

tech-stack:
  added: []
  patterns: []

key-files:
  created:
    - .planning/phases/78-occlusion-investigation/78-FINDINGS.md
    - .planning/phases/78-occlusion-investigation/screenshots/
    - .planning/phases/78-occlusion-investigation/frame_stats.json
    - .planning/phases/78-occlusion-investigation/confidence_sweep.md
    - .planning/phases/78-occlusion-investigation/occlusion_investigation.mp4
  modified:
    - scripts/investigate_occlusion.py

key-decisions:
  - "GO recommendation: no keypoint identity jumps, no OBB merging during occlusion"
  - "Multi-instance pose output is Gaussian NMS artifact (duplicate focal fish), not cross-fish detection"
  - "Confidence threshold can drop to 0.1 with polygon NMS applied"
  - "White-wall background detection dropout identified as a training data gap"
  - "Phase 79 (remediation) skipped per GO decision"
  - "Phase 78.1 inserted for OBB/pose production retrain to address white-wall recall"

patterns-established:
  - "Per-keypoint confidence weighting for occlusion-aware tracking"
  - "Endpoint confidence collapse as reliable occlusion detection signal"

requirements-completed: [INV-02, INV-04]

duration: 15min
completed: 2026-03-10
---

# Plan 78-02: Execute Investigation and Produce Findings

**Occlusion investigation yields GO recommendation: no identity jumps (0%), no box merging, confidence patterns useful for tracker design. White-wall dropout and Gaussian NMS artifacts documented as known limitations.**

## Performance

- **Duration:** 15 min (including user review and revision)
- **Tasks:** 2 (1 auto + 1 checkpoint)
- **Files created:** 7

## Accomplishments
- Ran investigation script on e3v831e frames 300-500, producing annotated video and per-frame statistics
- Ran confidence sweep on 600 frames (0.10-0.50 in 9 steps)
- Characterized multi-instance pose output as Gaussian NMS artifact (not cross-fish detection)
- Identified white-wall background detection dropout issue
- Produced GO recommendation with explicit criteria evaluation
- User reviewed and approved findings with corrections incorporated

## Task Commits

1. **Task 1: Execute investigation and produce findings** - `93e8cfa` (docs)
2. **Task 2: User checkpoint** - Approved with revisions (findings updated during review)

## Files Created/Modified
- `.planning/phases/78-occlusion-investigation/78-FINDINGS.md` — Investigation findings with GO recommendation
- `.planning/phases/78-occlusion-investigation/occlusion_investigation.mp4` — Annotated crop video
- `.planning/phases/78-occlusion-investigation/frame_stats.json` — Per-frame statistics
- `.planning/phases/78-occlusion-investigation/confidence_sweep.md` — Threshold sweep results
- `.planning/phases/78-occlusion-investigation/screenshots/` — 6 key frame screenshots
- `scripts/investigate_occlusion.py` — Updated: primary-only pose, polygon NMS, --nms-threshold arg

## Decisions Made
- GO recommendation based on 0% keypoint identity jumps (threshold: >20% = no-go)
- Multi-instance detections are Gaussian NMS artifacts, not useful for occlusion detection
- Script updated to match production pipeline (primary-only pose, polygon NMS)
- Confidence threshold can be 0.1 with polygon NMS
- Phase 79 skipped, Phase 78.1 inserted for production retrain

## Deviations from Plan
- Findings document revised during user review to correct multi-instance characterization
- Script updated with polygon NMS and primary-only pose extraction (improvements over original plan)
- Phase 78.1 inserted for production retrain (recommended follow-up beyond original scope)

## Issues Encountered
None during execution. User review identified the Gaussian NMS artifact mischaracterization which was corrected.

## Next Phase Readiness
- Phase 78.1 (OBB & Pose Production Retrain) is next — needs planning
- Phase 80 (Baseline Metrics) depends on Phase 78.1

---
*Phase: 78-occlusion-investigation*
*Completed: 2026-03-10*
