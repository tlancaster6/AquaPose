---
phase: 78-occlusion-investigation
plan: 01
subsystem: scripts
tags: [ultralytics, yolo-obb, yolo-pose, ocsort, opencv, investigation]

requires:
  - phase: 73-round-1-pseudo-labels-retraining
    provides: trained OBB and pose model weights
provides:
  - Standalone occlusion investigation CLI script with annotated video and confidence sweep modes
affects: [78-02-investigation-execution]

tech-stack:
  added: []
  patterns:
    - "Standalone investigation scripts in scripts/ directory"
    - "Multi-instance pose extraction from kp.xy (all instances, not just [0])"

key-files:
  created:
    - scripts/investigate_occlusion.py
  modified: []

key-decisions:
  - "Script is fully standalone — copies affine crop logic inline rather than importing from aquapose.core.midline to avoid pipeline initialization"
  - "Uses OcSortTracker with min_hits=1 for investigation (more sensitive than pipeline default of 3)"
  - "Pose model conf=0.1 for maximum keypoint visibility during investigation"
  - "Two-pass approach: detection+tracking+pose in pass 1, rendering in pass 2"

patterns-established:
  - "Investigation scripts use argparse CLI with --project-config for config loading"

requirements-completed: [INV-01]

duration: 5min
completed: 2026-03-10
---

# Plan 78-01: Occlusion Investigation Script

**Standalone CLI script for OBB detection, OC-SORT tracking, and multi-instance pose estimation with annotated crop video output and confidence sweep mode**

## Performance

- **Duration:** 5 min
- **Tasks:** 1
- **Files created:** 1

## Accomplishments
- Built 883-line standalone investigation script with two modes: single-threshold annotated video and confidence sweep
- Implemented multi-instance pose extraction (iterates all kp.xy instances, not just [0])
- Annotated video shows OBB boxes colored by track ID (palette), red for high-conf untracked, gray for low-conf untracked
- Keypoint circles encode per-keypoint confidence via radius; secondary instances drawn with dashed connections and lighter color
- Confidence sweep tests 9 thresholds (0.10-0.50) and produces a markdown table

## Task Commits

1. **Task 1: Build occlusion investigation script** - `69dd1b0` (feat)

## Files Created/Modified
- `scripts/investigate_occlusion.py` — Standalone CLI investigation script

## Decisions Made
- Script copies the 3-point affine crop logic inline (~20 lines) rather than importing from aquapose to stay standalone
- OcSortTracker min_hits=1 for investigation sensitivity
- Frame number rendered in top-left corner of crop (small, unobtrusive)

## Deviations from Plan
None - plan executed as specified.

## Issues Encountered
- Ruff caught unused loop variable and unused assignment — fixed before successful commit.

## Next Phase Readiness
- Script ready to execute on target occlusion clip in Plan 78-02
- All CLI arguments documented and tested via --help

---
*Phase: 78-occlusion-investigation*
*Completed: 2026-03-10*
