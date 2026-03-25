---
phase: 106-cli-integration
plan: 01
subsystem: cli
tags: [click, reid, embedding, cli, frame-stride]

# Dependency graph
requires:
  - phase: 105-swap-detection-repair
    provides: SwapDetector, SwapDetectorConfig, ReidEvent in core/reid/swap_detector.py
  - phase: 104-reid-fine-tuning
    provides: ReidTrainingConfig, build_feature_cache, train_reid_head in training/reid_training.py
  - phase: 103-reid-embed-runner
    provides: EmbedRunner, TrainingDataMiner, FishEmbedder in core/reid/
provides:
  - "reid_group Click group with embed, mine-crops, fine-tune, repair subcommands in core/reid/cli.py"
  - "frame_stride parameter on EmbedRunner for subsampled embedding"
  - "aquapose reid embed/mine-crops/fine-tune/repair CLI surface"
affects: [106-02, future-phases-using-reid-cli]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Reid CLI group follows same lazy-import pattern as train_group/data_group"
    - "core/reid/cli.py imports from cli_utils but uses SimpleNamespace instead of engine.ReidConfig (import boundary rule)"

key-files:
  created:
    - src/aquapose/core/reid/cli.py
  modified:
    - src/aquapose/core/reid/runner.py
    - src/aquapose/cli.py
    - src/aquapose/core/reid/__init__.py

key-decisions:
  - "Use SimpleNamespace for embed config to avoid forbidden engine/ import in core/ module (import boundary rule)"
  - "Remove top-level mine-reid-crops command entirely; migrated to reid mine-crops with no backward-compat shim"
  - "Delete scripts/train_reid_head.py; fully superseded by reid fine-tune command"
  - "reid_group NOT re-exported in core/reid/__init__.py to avoid heavy eager imports at CLI startup"

patterns-established:
  - "CLI groups for core modules live in core/<module>/cli.py, not in training/"
  - "Avoid engine/ imports inside core/ CLI modules; use SimpleNamespace or inline defaults"

requirements-completed: [CLI-01]

# Metrics
duration: 20min
completed: 2026-03-25
---

# Phase 106 Plan 01: Reid CLI Group Summary

**Click group `aquapose reid` with embed, mine-crops, fine-tune, repair subcommands plus EmbedRunner frame_stride for subsampled embedding**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-03-25T20:00:00Z
- **Completed:** 2026-03-25T20:18:38Z
- **Tasks:** 2
- **Files modified:** 4 (1 created, 3 modified, 1 deleted)

## Accomplishments
- Added `frame_stride: int = 1` to `EmbedRunner.__init__` with stride-based frame skipping in run loop
- Created `src/aquapose/core/reid/cli.py` with all four subcommands using lazy imports
- Registered `reid_group` in main `cli.py` via `cli.add_command()`
- Removed top-level `mine-reid-crops` command; deleted `scripts/train_reid_head.py`

## Task Commits

Each task was committed atomically:

1. **Task 1: Add frame_stride to EmbedRunner** - `81694e6` (feat)
2. **Task 2: Create reid_group Click group with all four subcommands** - `20a6ff0` (feat)

## Files Created/Modified
- `src/aquapose/core/reid/cli.py` - reid_group with embed, mine-crops, fine-tune, repair subcommands
- `src/aquapose/core/reid/runner.py` - Added frame_stride parameter to EmbedRunner
- `src/aquapose/cli.py` - Import and register reid_group; remove old mine-reid-crops command
- `src/aquapose/core/reid/__init__.py` - Updated docstring (reid_group NOT re-exported; avoid eager heavy imports)
- `scripts/train_reid_head.py` - Deleted (superseded by reid fine-tune command)

## Decisions Made
- Used `SimpleNamespace` for the embed config object instead of importing `ReidConfig` from `engine/config.py`. The import boundary checker (`IB-001`) forbids runtime imports from `engine/` inside `core/` modules. Since `EmbedRunner` accepts `config: Any`, a namespace with the same field names and defaults works correctly.
- Did not re-export `reid_group` from `core/reid/__init__.py`. If added there, importing the package would trigger eager loading of the embedder (PIL, timm) at CLI startup. Keeping it out avoids this overhead; `cli.py` imports directly from `core.reid.cli`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Forbidden engine/ import in core/ module**
- **Found during:** Task 2 (Create reid_group Click group)
- **Issue:** Pre-commit hook `import-boundary` flagged `IB-001`: `core/reid/cli.py` used `from aquapose.engine.config import ReidConfig` inside `embed_cmd`. Core modules cannot import from engine/ at runtime.
- **Fix:** Replaced `ReidConfig()` with `SimpleNamespace(model_name=..., batch_size=..., ...)` using the same default values as `ReidConfig`. `EmbedRunner.__init__` accepts `config: Any` so this works transparently.
- **Files modified:** `src/aquapose/core/reid/cli.py`
- **Verification:** Pre-commit `import-boundary` hook passes; 1273 unit tests pass.
- **Committed in:** `20a6ff0` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - import boundary enforcement)
**Impact on plan:** Fix was necessary for code correctness per project import boundary rules. No scope creep; behavior identical.

## Issues Encountered
- Ruff formatter reformatted `runner.py` stride condition into multi-line form during pre-commit; re-staged and committed successfully on second attempt.

## Next Phase Readiness
- `aquapose reid embed/mine-crops/fine-tune/repair` commands are ready to use
- Plan 02 (if any) can integrate these commands or wire them into existing pipeline phases
- No blockers

---
*Phase: 106-cli-integration*
*Completed: 2026-03-25*
