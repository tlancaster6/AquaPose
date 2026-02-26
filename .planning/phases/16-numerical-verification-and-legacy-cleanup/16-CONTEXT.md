# Phase 16: Numerical Verification and Legacy Cleanup - Context

**Gathered:** 2026-02-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Confirm the migrated pipeline is numerically equivalent to v1.0 on real data using existing golden data artifacts, then archive all legacy scripts that bypass the pipeline. The golden data files already exist per-stage in `tests/golden/`. This phase writes regression tests against them, updates the generation script to use PosePipeline, and moves legacy scripts to `scripts/legacy/`.

</domain>

<decisions>
## Implementation Decisions

### Tolerance thresholds
- Per-stage tolerances, not uniform
- Small epsilon (~1e-6) everywhere including detections — accounts for potential model inference nondeterminism
- Reconstruction (3D spline control points): ~1e-3 absolute tolerance — sub-millimeter agreement
- Tests fail hard on tolerance violations — violations are bugs until golden data is explicitly updated

### Golden data usage
- Existing golden files: `golden_detection.pt`, `golden_segmentation.pt.gz`, `golden_midline_extraction.pt`, `golden_tracking.pt`, `golden_triangulation.pt`
- Per-stage comparison AND full end-to-end pipeline comparison against final 3D output
- Update `generate_golden_data.py` to use PosePipeline instead of v1.0 scripts
- Tests marked with `@pytest.mark.regression` (new marker, separate from `@slow`)

### Divergence handling
- Intentional divergences (bug fixes): update golden data with descriptive commit message — git history is the documentation
- Improvements: flag for review (test fails), investigate, confirm improvement, then update golden data
- `pytest.mark.xfail` with reason allowed during active development — all xfails must be resolved before phase completion
- Seed all nondeterministic operations (RANSAC, model inference) for deterministic comparison — reproducibility is a guidebook requirement

### Legacy script policy
- Only archive scripts that duplicate or bypass pipeline functionality
- Keep training/dataset scripts (`train_yolo.py`, `sample_yolo_frames.py`, `organize_yolo_dataset.py`, `build_training_data.py`) — they serve a different purpose
- Archive to `scripts/legacy/`: `diagnose_pipeline.py`, `diagnose_tracking.py`, `diagnose_triangulation.py`, `per_camera_spline_overlay.py`
- `generate_golden_data.py` stays but gets updated to use PosePipeline
- Import audit after archival: verify no active code (src/, tests/) imports from archived scripts

### Claude's Discretion
- Exact per-stage tolerance values beyond the decisions above
- Test file organization within `tests/regression/` or similar
- How to structure the end-to-end golden data comparison
- Specific seeding mechanism for each nondeterministic operation

</decisions>

<specifics>
## Specific Ideas

- Guidebook Section 14 (Migration Strategy > Verification) specifies: "Golden data generated as a standalone preparatory commit before any stage migration begins" — this was already done
- Guidebook requires regression tests marked `@pytest.mark.regression` — run outside the fast test loop
- The pipeline reproducibility contract: "Given identical inputs, identical configuration, and identical random seeds, the pipeline must produce identical outputs"

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 16-numerical-verification-and-legacy-cleanup*
*Context gathered: 2026-02-25*
