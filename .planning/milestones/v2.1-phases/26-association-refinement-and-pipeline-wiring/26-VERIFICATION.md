---
phase: 26-association-refinement-and-pipeline-wiring
requirements: [ASSOC-03, PIPE-02, PIPE-03]
plans_completed: [26-01, 26-02, 26-03]
verified: 2026-02-27
---

# Phase 26 Verification: Association Refinement and Pipeline Wiring

## Success Criteria Verification

### SC-1: Tracklet cluster refinement via 3D triangulation with eviction and per-frame confidence
**Status:** PASS

Evidence:
- `src/aquapose/core/association/refinement.py` implements `refine_clusters()` with robust consensus computation and tracklet eviction
- `refine_clusters()` is called from `AssociationStage.run()` in `src/aquapose/core/association/stage.py` after `merge_fragments()`
- Eviction uses median ray-to-consensus-point distance with configurable `eviction_reproj_threshold` (default 0.025m)
- Per-frame confidence computed from ray convergence quality and stored in `TrackletGroup.per_frame_confidence`
- 10 unit tests in `tests/unit/core/association/test_refinement.py` verify eviction, re-triangulation, disabled mode, and confidence
- Configuration: `AssociationConfig.eviction_reproj_threshold`, `min_cameras_refine`, `refinement_enabled`

### SC-2: Midline extraction filters to confirmed tracklet groups with head-tail orientation resolution
**Status:** PASS

Evidence:
- `src/aquapose/core/midline/stage.py` `MidlineStage.run()` builds `group_det_index` from `tracklet_groups` and filters detections via `_filter_frame_detections()`
- `_apply_orientation()` resolves head-tail using `resolve_orientation()` from `src/aquapose/core/midline/orientation.py`
- 3-signal weighted vote: geometric (cross-camera ray convergence), velocity alignment, temporal prior
- Configuration: `MidlineConfig.speed_threshold`, `orientation_weight_geometric/velocity/temporal`
- 11 unit tests in `tests/unit/core/midline/test_orientation.py`
- Fallback mode processes all detections when `tracklet_groups` is empty/None

### SC-3: Reconstruction uses known camera membership from tracklet_groups (no RANSAC)
**Status:** PASS

Evidence:
- `src/aquapose/core/reconstruction/stage.py` fully rewritten to consume `tracklet_groups`
- `_run_with_tracklet_groups()` determines per-frame camera membership from `Tracklet2D.frames` and `frame_status`
- Only `"detected"` frames count toward camera membership (coasted frames excluded)
- Frames below `min_cameras` (default 3) are dropped with reason `"insufficient_views"`
- Short gaps (<=`max_interp_gap`, default 5) linearly interpolated with `is_low_confidence=True`
- No RANSAC matching step -- camera membership comes directly from `tracklet_groups`
- Configuration: `ReconstructionConfig.min_cameras`, `max_interp_gap`, `n_control_points`
- 16 unit tests in `tests/unit/core/reconstruction/test_reconstruction_stage.py`

### SC-4: Full pipeline runs end-to-end and produces HDF5 output with fish IDs
**Status:** PASS (structural)

Evidence:
- `build_stages()` in `src/aquapose/engine/pipeline.py` returns 5 stages in correct order: Detection -> Tracking -> Association -> Midline -> Reconstruction
- All stages are wired with correct constructor args from config
- `HDF5ExportObserver` writes fish-first layout (`/fish_{id}/spline_controls[T,N,3]`, `/fish_{id}/confidence[T]`) when `tracklet_groups` present
- Root attributes include `config_hash`, `run_timestamp`, `calibration_path`, `run_id`
- Frame-major layout preserved as backward-compatible fallback
- 14 HDF5 tests in `tests/unit/engine/test_hdf5_observer.py`
- e2e real-video testing deferred to Phase 28 (pipeline must be functionally complete first)

## Requirements Completed

| Requirement | Description | Status |
|-------------|-------------|--------|
| ASSOC-03 | Cluster refinement via 3D triangulation error | COMPLETE |
| PIPE-02 | Midline filtering by tracklet group membership + orientation | COMPLETE |
| PIPE-03 | Reconstruction from tracklet_groups with known camera membership | COMPLETE |

## Test Summary

- 543 tests passing (0 failures)
- 16 new reconstruction tests
- 14 HDF5 observer tests (7 new fish-first tests)
- Lint and format clean

## Key Files

| File | Purpose |
|------|---------|
| `src/aquapose/core/association/refinement.py` | Cluster refinement + eviction |
| `src/aquapose/core/midline/orientation.py` | Head-tail orientation resolver |
| `src/aquapose/core/midline/stage.py` | Tracklet-group filtering + orientation |
| `src/aquapose/core/reconstruction/stage.py` | Tracklet-group-driven reconstruction |
| `src/aquapose/engine/hdf5_observer.py` | Fish-first HDF5 layout |
| `src/aquapose/engine/config.py` | Extended config for all 3 plans |
| `src/aquapose/engine/pipeline.py` | build_stages() with new config params |

---
*Phase: 26-association-refinement-and-pipeline-wiring*
*Verified: 2026-02-27*
