---
phase: 102-embedding-infrastructure
plan: 01
subsystem: reid
tags: [timm, megadescriptor, swin-tiny, embeddings, protocol]

requires:
  - phase: none
    provides: n/a
provides:
  - FishEmbedder class for L2-normalized 768-dim embeddings from BGR crops
  - ReidConfig frozen dataclass in engine/config.py
  - reid package structure at src/aquapose/core/reid/
affects: [102-02, reid-runner, stitch, identity-matching]

tech-stack:
  added: [timm]
  patterns: [Protocol-based config interface for core/engine boundary]

key-files:
  created:
    - src/aquapose/core/reid/__init__.py
    - src/aquapose/core/reid/embedder.py
  modified:
    - pyproject.toml
    - src/aquapose/engine/config.py

key-decisions:
  - "Used Protocol (ReidConfigLike) instead of direct ReidConfig import to respect core/engine import boundary"
  - "MegaDescriptor normalization uses [-1,1] range (not ImageNet stats) per model card"

patterns-established:
  - "Protocol-based config: core modules define a Protocol matching the engine config dataclass shape, avoiding import boundary violations"

requirements-completed: [EMBED-02]

duration: 8min
completed: 2026-03-25
---

# Plan 102-01: FishEmbedder & ReidConfig Summary

**MegaDescriptor-T backbone wrapper producing L2-normalized 768-dim embeddings via Protocol-based config interface**

## Performance

- **Duration:** 8 min
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- FishEmbedder loads MegaDescriptor-T via timm, verifies output dim at construction
- Accepts arbitrary-size BGR uint8 crops, preprocesses to 224x224, normalizes to [-1,1]
- Returns L2-normalized float32 embeddings with sub-batch processing
- ReidConfig frozen dataclass with model_name, batch_size, crop_size, device, embedding_dim

## Task Commits

1. **Task 1+2: Add timm dep, ReidConfig, FishEmbedder, reid package** - `7a1d39b` (feat)

## Files Created/Modified
- `src/aquapose/core/reid/embedder.py` - FishEmbedder class with ReidConfigLike Protocol
- `src/aquapose/core/reid/__init__.py` - Package public API
- `src/aquapose/engine/config.py` - ReidConfig frozen dataclass
- `pyproject.toml` - Added timm>=0.9 dependency

## Decisions Made
- Used structural Protocol (ReidConfigLike) to avoid core->engine import boundary violation. The import-boundary hook forbids both runtime imports and TYPE_CHECKING backdoors from core/ to engine/. Protocol provides the same type safety without coupling.

## Deviations from Plan

### Auto-fixed Issues

**1. [Import boundary] core/engine import violation**
- **Found during:** Task 2 (FishEmbedder implementation)
- **Issue:** Plan specified `from aquapose.engine.config import ReidConfig` in embedder.py, but import-boundary hook forbids core->engine imports (both runtime and TYPE_CHECKING)
- **Fix:** Created ReidConfigLike Protocol in embedder.py matching ReidConfig attributes
- **Files modified:** src/aquapose/core/reid/embedder.py
- **Verification:** Import boundary hook passes, ReidConfig satisfies Protocol structurally
- **Committed in:** 7a1d39b

---

**Total deviations:** 1 auto-fixed (import boundary compliance)
**Impact on plan:** Necessary for project architecture rules. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- FishEmbedder ready for Plan 102-02's EmbedRunner to use
- ReidConfig ready for caller-side configuration

---
*Phase: 102-embedding-infrastructure*
*Completed: 2026-03-25*
