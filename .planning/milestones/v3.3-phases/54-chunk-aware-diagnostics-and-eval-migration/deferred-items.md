# Deferred Items - Phase 54

## Pre-existing Test Failures (Out of Scope)

The following test failures exist before phase 54 work began and are not caused by any changes in this phase:

- `tests/unit/evaluation/test_stage_association.py::test_default_grid_ray_distance_threshold_values`
- `tests/unit/evaluation/test_stage_association.py::test_default_grid_score_min_values`
- `tests/unit/evaluation/test_stage_association.py::test_default_grid_eviction_reproj_threshold_values`
- `tests/unit/evaluation/test_stage_association.py::test_default_grid_leiden_resolution_values`
- `tests/unit/evaluation/test_stage_association.py::test_default_grid_early_k_values`

These assert specific values in `DEFAULT_GRID` that have been changed since the tests were written. They are evaluation tuning tests and need updating to match current grid values.

## Pre-existing Typecheck Errors (Out of Scope)

42 pre-existing basedpyright errors exist in `src/aquapose/` — none in files modified by this phase. Not addressed here.
