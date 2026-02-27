# Phase 28 Bug Log

Non-blocking bugs and known limitations discovered during Phase 28 e2e testing.

## NB-001: AssociationStage requires pre-built LUTs

**Severity:** Non-blocking (degrades to empty output, no crash)
**Observed during:** Task 1 execution, synthetic pipeline run
**Symptom:** `LUTs not available -- association producing empty groups` warning logged. `tracklet_groups = []`, `midlines_3d` contains only empty dicts.
**Root cause:** AssociationStage calls `load_forward_luts` and `load_inverse_luts`, which return `None` if `.npz` files are not cached in `<calibration_dir>/luts/`. No LUT files exist on this machine for the release calibration.
**Impact:** Synthetic tests pass (since they only require tracks_2d to be non-empty). Real-data tests requiring 3D splines will fail if LUTs are not built.
**Fix required:** Run `aquapose build-luts --calibration <path>` to generate LUT cache. This is a one-time setup step. Not fixed in Phase 28 (out of scope).
**File:** `src/aquapose/core/association/stage.py` (graceful degradation already in place)

## Fixed: BUG-001: SyntheticDataStage wrong fish z-coordinate

**Severity:** Blocking (no detections generated, pipeline produces empty output)
**Observed during:** Task 1 investigation
**Symptom:** All cameras produced 0 detections, resulting in 0 tracklets and 0 groups.
**Root cause:** `_generate_fish_splines()` in `src/aquapose/core/synthetic.py` placed fish at `cz = rng.uniform(0.02, 0.12)` (z = 2-12 cm). The air-water interface in release_calibration is at `water_z = 1.03m`. Fish above the interface (z < water_z) produce `valid=False` in refractive projection.
**Fix:** Pass `water_z` from calibration to `_generate_fish_splines()`. Fish now placed at `cz = water_z + rng.uniform(0.05, 0.35)` (5-35 cm below water surface). Committed in b7adf90.
**File:** `src/aquapose/core/synthetic.py`

## NB-002: Golden regression tests reference deleted v1.0 modules

**Severity:** Non-blocking (tests skipped, no crash)
**Observed during:** Task 2 human-verify checkpoint
**Symptom:** `ModuleNotFoundError: No module named 'aquapose.tracking.tracker'` when pytest collects `tests/golden/test_stage_harness.py`.
**Root cause:** Golden `.pt` fixture files were serialized from the v1.0 pipeline via `torch.save`. They contain pickled objects from `aquapose.tracking.tracker` (FishTrack), `aquapose.segmentation` (CropRegion), and other modules deleted or restructured in v2.1. Python's unpickler fails on import.
**Impact:** All 8 golden regression tests are non-functional. Module-level `pytestmark = pytest.mark.skip(...)` applied to prevent collection errors.
**Fix required:** Regenerate golden data from the v2.1 pipeline (`scripts/generate_golden_data.py` needs rewrite for new stage boundaries). Out of scope for Phase 28.
**File:** `tests/golden/test_stage_harness.py`, `tests/golden/conftest.py`
