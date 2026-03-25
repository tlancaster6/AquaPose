---
phase: 104-backbone-fine-tuning
plan: 01
subsystem: training
tags: [metric-learning, arcface, pytorch-metric-learning, reid, projection-head]

requires:
  - phase: 103-training-data-mining
    provides: reid_crops directory with group manifests and cropped fish images
provides:
  - ProjectionHead module (768->256->128, L2-normalized)
  - CachedFeatureDataset for metric learning with MPerClassSampler
  - build_feature_cache function for backbone feature extraction and caching
  - split_by_group temporal holdout splitting
  - compute_female_auc and compute_per_pair_auc evaluation functions
  - train_reid_head full training loop with SubCenterArcFace + BatchHardMiner
affects: [104-02, 105-swap-repair]

tech-stack:
  added: [pytorch-metric-learning>=2.9]
  patterns: [cached-feature training, dual-optimizer metric learning]

key-files:
  created:
    - src/aquapose/training/reid_training.py
    - tests/unit/training/test_reid_training.py
  modified:
    - pyproject.toml
    - src/aquapose/training/__init__.py

key-decisions:
  - "Use raw fish_id (0-8) as ArcFace labels with num_classes=9, accepting minor label noise in post-swap groups"
  - "Define _EmbedderConfig locally to satisfy ReidConfigLike protocol without importing from engine"
  - "Feature cache saved as NPZ for simplicity (adequate for ~10K crops)"

patterns-established:
  - "Dual optimizer pattern: Adam for head params, SGD for ArcFace loss params"
  - "Cached-feature training: backbone runs once, head training reads only small vectors"

requirements-completed: [TRAIN-03]

duration: 8min
completed: 2026-03-25
---

# Plan 104-01: ReID Training Module Summary

**SubCenterArcFace metric learning module with cached-feature training, temporal group split, and AUC-gated female fish evaluation**

## Performance

- **Duration:** ~8 min
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments
- ProjectionHead produces L2-normalized 128-dim embeddings from 768-dim backbone features
- Full training loop with SubCenterArcFace loss, BatchHardMiner, dual optimizers, and early stopping on holdout AUC
- Temporal group-based train/val split prevents leakage from overlapping miner windows
- Female-only AUC computation with per-pair breakdown identifies discriminability bottleneck pairs
- Feature cache built from reid_crops manifests via FishEmbedder, cached to NPZ for fast iteration
- 9 unit tests covering all core behaviors (1 additional smoke test marked @slow)

## Task Commits

1. **Task 1: Unit tests (RED)** - `12bb067` (test)
2. **Task 2: Implementation (GREEN)** - `e8a1872` (feat)

## Files Created/Modified
- `src/aquapose/training/reid_training.py` - Core module with all training functions
- `tests/unit/training/test_reid_training.py` - 10 unit tests (9 fast + 1 slow)
- `pyproject.toml` - Added pytorch-metric-learning>=2.9 dependency
- `src/aquapose/training/__init__.py` - Updated public API exports

## Decisions Made
- Used raw fish_id (0-8) as labels consistent with num_classes=9 ArcFace locked decision
- Defined local _EmbedderConfig dataclass to avoid importing from engine (IB-004 boundary)
- Val set constructed from val_ds but evaluation uses direct cache indexing for AUC (no need for DataLoader on val)

## Deviations from Plan
None - plan executed as specified.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- reid_training.py module ready for Plan 02 standalone driver script
- All public functions exported through training __init__.py

---
*Phase: 104-backbone-fine-tuning*
*Completed: 2026-03-25*
