---
phase: 107-unfrozen-backbone-fine-tuning
plan: 02
subsystem: cli
tags: [click, reid, fine-tuning, cli, embedding]

requires:
  - phase: 107-unfrozen-backbone-fine-tuning
    provides: train_reid_end_to_end, combined checkpoint format
provides:
  - --unfreeze-blocks and --lr-backbone-factor CLI options for reid fine-tune
  - Backbone-aware re-embedding via monkey-patched EmbedRunner
affects: [swap-detection, reid-pipeline]

tech-stack:
  added: []
  patterns: [monkey-patch-embedder-for-reembed, backup-restore-npz]

key-files:
  created: []
  modified:
    - src/aquapose/core/reid/cli.py

key-decisions:
  - "Monkey-patch FishEmbedder in runner module to inject FineTunedEmbedder for re-embedding"
  - "Back up and restore embeddings.npz to preserve zero-shot embeddings during re-embed"
  - "FineTunedEmbedder applies identical normalization as FishEmbedder but forwards through backbone+head"

patterns-established:
  - "Checkpoint format detection: if backbone_state_dict present, use fine-tuned backbone path"

requirements-completed: [UNFREEZE-03]

duration: 8min
completed: 2026-03-25
---

# Plan 107-02: CLI Integration with --unfreeze-blocks Summary

**CLI `reid fine-tune --unfreeze-blocks N` runs end-to-end training and re-embeds using fine-tuned backbone+head stack**

## Performance

- **Duration:** 8 min
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Added `--unfreeze-blocks` (default 0) and `--lr-backbone-factor` (default 0.1) to `reid fine-tune`
- When `unfreeze_blocks > 0`: skips feature caching, calls `train_reid_end_to_end`, re-embeds with fine-tuned backbone+head
- When `unfreeze_blocks == 0`: existing frozen path unchanged (build_feature_cache + train_reid_head + head-only re-embed)
- Re-embedding uses monkey-patched EmbedRunner with `_FineTunedEmbedder` wrapper for crop extraction reuse
- Original `embeddings.npz` preserved via backup/restore during re-embed

## Task Commits

1. **Task 1: Wire unfrozen backbone into CLI** - `038fb44` (feat)

## Files Created/Modified
- `src/aquapose/core/reid/cli.py` - Added --unfreeze-blocks/--lr-backbone-factor options, branched training and re-embedding paths, added _reembed_finetuned helper

## Decisions Made
- Monkey-patching FishEmbedder at module level to inject fine-tuned embedder (avoids modifying EmbedRunner API)
- Backing up/restoring embeddings.npz during re-embed to preserve zero-shot embeddings
- TYPE_CHECKING imports for Path, np, ReidTrainingConfig to avoid circular imports

## Deviations from Plan
None - plan executed as specified.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Full fine-tuning pipeline available via CLI
- embeddings_finetuned.npz produced with same schema as embeddings.npz (SwapDetector compatible)

---
*Phase: 107-unfrozen-backbone-fine-tuning*
*Completed: 2026-03-25*
