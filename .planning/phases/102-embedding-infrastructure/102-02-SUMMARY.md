---
phase: 102-embedding-infrastructure
plan: 02
subsystem: reid
tags: [embedding-runner, zero-shot-eval, affine-crop, npz, retrieval-metrics]

requires:
  - phase: 102-embedding-infrastructure
    provides: FishEmbedder class, ReidConfig
provides:
  - EmbedRunner class for batch crop extraction and embedding from completed runs
  - Zero-shot evaluation (within/between similarity, Rank-1, mAP)
  - reid/embeddings.npz output format with metadata arrays
affects: [103, 104, reid-training, identity-matching]

tech-stack:
  added: []
  patterns: [chunk-sequential processing for video I/O efficiency]

key-files:
  created:
    - src/aquapose/core/reid/runner.py
    - src/aquapose/core/reid/eval.py
  modified:
    - src/aquapose/core/reid/__init__.py

key-decisions:
  - "Accept either midlines_stitched.h5 or midlines.h5 (fallback) for flexibility"
  - "Process chunks sequentially, embed per-chunk for memory efficiency"
  - "Exclude same-frame same-camera matches in Rank-1/mAP to avoid trivial self-retrieval"

patterns-established:
  - "NPZ output with parallel arrays: embeddings, frame_index, fish_id, camera_id, detection_confidence"

requirements-completed: [EMBED-01, EMBED-03]

duration: 12min
completed: 2026-03-25
---

# Plan 102-02: EmbedRunner & Zero-shot Eval Summary

**Batch embed runner extracts OBB-aligned crops from chunk caches, embeds via MegaDescriptor-T, writes NPZ with zero-shot retrieval report**

## Performance

- **Duration:** 12 min
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- EmbedRunner loads H5 fish-frame mapping, iterates chunk caches, matches tracklets to detections
- OBB-aligned crops extracted via extract_affine_crop, embedded through FishEmbedder
- NPZ output with embeddings, frame_index, fish_id, camera_id, detection_confidence arrays
- Zero-shot eval computes within/between cosine similarity, Rank-1, mAP on 100 random frames

## Task Commits

1. **Task 1+2: EmbedRunner, eval module, package exports** - `440a9ec` (feat)

## Files Created/Modified
- `src/aquapose/core/reid/runner.py` - EmbedRunner class with full pipeline
- `src/aquapose/core/reid/eval.py` - compute_reid_metrics and print_reid_report
- `src/aquapose/core/reid/__init__.py` - Updated exports

## Decisions Made
- Accepts both midlines_stitched.h5 and midlines.h5 (fallback) since not all runs have stitched output
- Handles both single-column and multi-column fish_id layouts in H5

## Deviations from Plan

### Auto-fixed Issues

**1. H5 file flexibility**
- **Found during:** Task 1 (EmbedRunner implementation)
- **Issue:** Plan specified midlines_stitched.h5 as required, but not all runs have it
- **Fix:** Fall back to midlines.h5 if stitched version doesn't exist
- **Verification:** FileNotFoundError only if neither file exists

---

**Total deviations:** 1 auto-fixed (practical robustness)
**Impact on plan:** More robust without changing semantics.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Complete embedding infrastructure ready for Phase 103 (CLI integration) and Phase 104 (evaluation)
- Zero-shot baseline will be established when run on actual data

---
*Phase: 102-embedding-infrastructure*
*Completed: 2026-03-25*
