---
phase: 45-dead-code-cleanup
plan: 02
subsystem: tests, docs
tags: [dead-code, dlt, tests, guidebook]

requires:
  - phase: 45-dead-code-cleanup
    plan: 01
    provides: Dead modules deleted, production code updated
provides:
  - Tests updated for DLT-only reconstruction (no dead imports)
  - GUIDEBOOK.md reflects DLT-only architecture
  - pipeline.py build_stages() compatible with simplified ReconstructionConfig
---

## Summary

Completed test and documentation cleanup for DLT-only reconstruction.

### Changes

**Deleted test files:**
- `tests/unit/test_triangulation.py` — tested only dead `triangulate_midlines` and related functions
- `tests/unit/test_curve_optimizer.py` — tested only dead `CurveOptimizer`

**Updated test files:**
- `test_reconstruction_stage.py` — removed 4 dead test functions, updated import boundary module list
- `test_harness.py` — changed mock patches from `TriangulationBackend.from_models` to `DltBackend.from_models`
- `test_config.py` — changed expected backend from `"triangulation"` to `"dlt"`
- `test_midline_writer.py` — import from `utils` instead of `triangulation`
- `test_synthetic.py` — import from `utils` instead of `triangulation`
- `test_confidence_weighting.py` — kept only `TestWeightedTriangulateRays` (3 tests), removed dead curve optimizer tests
- `test_dlt_backend.py` — updated `DEFAULT_OUTLIER_THRESHOLD` assertion from 50.0 to 10.0

**Production code fix (missed by Plan 01):**
- `src/aquapose/engine/pipeline.py` — removed references to deleted `inlier_threshold`, `snap_threshold`, `max_depth` from `build_stages()`

**Constants:**
- Added `SPLINE_N_CTRL = 7` to `src/aquapose/core/reconstruction/utils.py` (needed by test_synthetic.py)

**Documentation:**
- `.planning/GUIDEBOOK.md` — updated reconstruction backends listing (dlt.py only), removed triangulation_viz from visualization listing, updated Stage 5 description to "Backend: DLT", updated backend examples to remove triangulation/curve optimizer references

### Verification

- `hatch run test` — 687 passed, 3 skipped, 30 deselected
- `hatch run check` — lint clean, typecheck has only pre-existing errors (40 errors unrelated to this phase)
- `grep` for deleted module imports — zero results in src/ and tests/
- YH config already had `backend: dlt` — no changes needed

### Commits

- `42c72e7` — feat(45-02): update tests and docs for DLT-only reconstruction
