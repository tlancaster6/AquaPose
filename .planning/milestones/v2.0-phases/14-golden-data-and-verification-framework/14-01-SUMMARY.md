---
phase: 14-golden-data-and-verification-framework
plan: 01
subsystem: testing
tags: [golden-data, regression, pytorch, pipeline, fixtures]

# Dependency graph
requires:
  - phase: 13-engine-core
    provides: PosePipeline orchestrator and v1.0 stage functions in stages.py
provides:
  - scripts/generate_golden_data.py — standalone CLI to produce .pt fixtures
  - tests/golden/ directory scaffolded with .gitkeep
  - Frozen stage output .pt files committed as regression baseline (pending user run)
affects:
  - 14-02 (verification framework reads these .pt files as expected outputs)
  - 15-stage-migrations (ported stages must match golden outputs numerically)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Deterministic seed setup (random + numpy + torch + CUDA) before any pipeline imports"
    - "Stage-by-stage output serialization via torch.save (pickle-based, handles dataclasses/ndarrays)"
    - "Environment metadata snapshot in metadata.pt (GPU, CUDA, PyTorch, numpy versions + seed)"

key-files:
  created:
    - scripts/generate_golden_data.py
    - tests/golden/.gitkeep
  modified: []

key-decisions:
  - "Seeds set before pipeline module imports to guarantee determinism even if imports trigger CUDA init"
  - "Each stage output saved independently (not bundled) so partial re-runs can compare individual stages"
  - "Camera e3v8250 excluded in script to match orchestrator.py behavior exactly"
  - "metadata.pt records generation environment — required to interpret tolerance differences across GPUs"

patterns-established:
  - "Golden data generation: run with --seed 42 --stop-frame 30 for reproducible 30-frame regression fixture"
  - "Commit golden data as standalone commit with message: data(14): commit golden reference outputs from v1.0 pipeline"

requirements-completed: [VER-01]

# Metrics
duration: 8min
completed: 2026-02-25
---

# Phase 14 Plan 01: Golden Data Generation Script Summary

**Standalone CLI script (`scripts/generate_golden_data.py`) that runs the v1.0 pipeline stage-by-stage with deterministic seeding and saves 5 stage outputs + metadata as `.pt` fixture files in `tests/golden/`**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-02-25T21:51:49Z
- **Completed:** 2026-02-25T22:00:00Z
- **Tasks:** 1 of 2 committed (Task 2 is a human-verify checkpoint awaiting user execution)
- **Files modified:** 2

## Accomplishments

- Created `scripts/generate_golden_data.py` — full CLI with argparse, deterministic seed setup, stage-by-stage execution following orchestrator.py exactly
- Created `tests/golden/.gitkeep` to scaffold the golden data directory in version control
- Script validated: syntax OK, `--help` shows all 9 expected CLI arguments

## Task Commits

Each task was committed atomically:

1. **Task 1: Create golden data generation script** - `ed79f27` (feat)

**Plan metadata:** pending final commit

## Files Created/Modified

- `scripts/generate_golden_data.py` - CLI script that runs v1.0 pipeline stages, saves golden_detection.pt, golden_segmentation.pt, golden_tracking.pt, golden_midline_extraction.pt, golden_triangulation.pt, metadata.pt
- `tests/golden/.gitkeep` - Directory scaffold for golden fixture files

## Decisions Made

- Seeds set before pipeline imports (not just before running stages) — ensures any CUDA initialization triggered by imports is also seeded
- Each stage output saved as a separate `.pt` file so individual stage comparisons are possible without loading the full bundle
- Camera `e3v8250` excluded in script, matching orchestrator.py behavior exactly

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Ruff auto-fixed datetime.timezone.utc import style**
- **Found during:** Task 1 commit (pre-commit hook)
- **Issue:** Used `from datetime import datetime, timezone` with `datetime.now(timezone.utc)` — ruff F401 prefers `from datetime import UTC, datetime` with `datetime.now(UTC)`
- **Fix:** Ruff pre-commit hook auto-fixed the import; re-staged and committed cleanly
- **Files modified:** scripts/generate_golden_data.py
- **Verification:** Ruff passed on second commit attempt
- **Committed in:** ed79f27 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - style/lint)
**Impact on plan:** Cosmetic-only. No behavior change.

## Issues Encountered

None — plan executed cleanly. Task 2 is a blocking human checkpoint (user must run the script with local video data and model weights, then commit the resulting .pt files).

## User Setup Required

**Task 2 requires manual execution.** Run the golden data generation script with local data paths:

```bash
cd C:/Users/tucke/PycharmProjects/AquaPose
python scripts/generate_golden_data.py \
  --video-dir "C:/Users/tucke/Desktop/Aqua/AquaPose/raw_videos" \
  --calibration "path/to/calibration.json" \
  --stop-frame 30 \
  --detector-kind yolo \
  --yolo-weights "runs/detect/output/yolo_fish/train_v1/weights/best.pt" \
  --unet-weights "C:/Users/tucke/Desktop/Aqua/AquaPose/unet/best_model.pth" \
  --seed 42 \
  --output-dir tests/golden/
```

Then commit the output:
```bash
git add tests/golden/
git commit -m "data(14): commit golden reference outputs from v1.0 pipeline"
```

## Next Phase Readiness

- `scripts/generate_golden_data.py` is complete and ready for user execution
- Once golden .pt files are committed, Phase 14 Plan 02 (verification framework) can begin
- Phase 15-16 stage migrations are gated on committed golden data (VER-01)

---
*Phase: 14-golden-data-and-verification-framework*
*Completed: 2026-02-25*
