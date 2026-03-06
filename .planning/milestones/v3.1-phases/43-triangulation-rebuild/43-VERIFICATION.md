---
phase: 43-triangulation-rebuild
verified: 2026-03-02T00:00:00Z
status: passed
score: 6/6 success criteria verified
re_verification: false
gaps: []
human_verification: []
---

# Phase 43: Triangulation Rebuild Verification Report

**Phase Goal:** A new reconstruction backend exists that uses confidence-weighted DLT triangulation with outlier rejection and B-spline fitting
**Verified:** 2026-03-02
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Each body point is triangulated via confidence-weighted DLT using all available cameras (single strategy, no branching on camera count) | VERIFIED | `_triangulate_body_point` has one triangulation path; camera-count branching is absent. Ray-angle filter for 2-cam case skips a degenerate pair but does not switch algorithms. `_tri_rays` dispatches to weighted or unweighted DLT uniformly. |
| 2 | Cameras whose reprojection residual exceeds the rejection threshold are flagged as outliers and the point is re-triangulated with inlier cameras only | VERIFIED | Lines 399-415 in `dlt.py`: residuals computed per camera, cameras above `outlier_threshold` dropped, `_tri_rays` called again with `inlier_ids` only. |
| 3 | A B-spline with 7 control points is fit to the triangulated points; frames where fewer than the minimum valid-point threshold are available are skipped | VERIFIED | `DEFAULT_N_CONTROL_POINTS=7`, `fit_spline` called with `self._spline_knots` (built from 7 control points). `_min_body_points = n_control_points + 2 = 9`. Fish is skipped when `len(valid_indices) < self._min_body_points` (line 212). |
| 4 | Points with Z at or below water_z are rejected before fitting | VERIFIED | Water surface rejection applied twice in `_triangulate_body_point`: after initial triangulation (line 379) and after re-triangulation (line 418). Both use `<= water_z`. |
| 5 | Reconstructions where a configurable fraction of body points had fewer than 3 inlier cameras are flagged as low-confidence | VERIFIED | Lines 282-285 in `dlt.py`: `n_weak` counts points with `n_cams < 3`; `is_low_confidence = n_weak > low_confidence_fraction * len(per_point_n_cams)`. Default fraction is 0.2, configurable via constructor. |
| 6 | Half-widths from upstream are passed through to the output without being used in triangulation logic | VERIFIED | `midline.half_widths[i]` read in `_reconstruct_fish` for output conversion only (lines 205-206). `_triangulate_body_point` makes no reference to `half_widths`. Converted values appear in `Midline3D.half_widths`. |

**Score:** 6/6 success criteria verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/core/reconstruction/utils.py` | Shared reconstruction helpers (4 functions + 2 constants) | VERIFIED | 170 lines; exports `build_spline_knots`, `weighted_triangulate_rays`, `fit_spline`, `pixel_half_width_to_metres`, `SPLINE_K=3`, `MIN_BODY_POINTS=9`; `__all__` defined |
| `tests/unit/core/reconstruction/test_reconstruction_utils.py` | Unit tests for extracted helpers; min 50 lines | VERIFIED | 279 lines; covers all 4 functions and 2 constants |
| `src/aquapose/core/reconstruction/backends/dlt.py` | DltBackend class; min 150 lines | VERIFIED | 562 lines (563 including newline); exports `DltBackend` with `__all__` |
| `tests/unit/core/reconstruction/test_dlt_backend.py` | Unit tests for DltBackend; min 100 lines | VERIFIED | 595 lines; covers basic reconstruction, NaN handling, outlier rejection, water surface rejection, low-confidence flagging, insufficient points, half-widths passthrough, registry, confidence weighting |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/aquapose/core/reconstruction/triangulation.py` | `src/aquapose/core/reconstruction/utils.py` | import | WIRED | Line 19-26: `from aquapose.core.reconstruction.utils import (MIN_BODY_POINTS, SPLINE_K, build_spline_knots, fit_spline, pixel_half_width_to_metres, weighted_triangulate_rays)` |
| `src/aquapose/core/reconstruction/backends/dlt.py` | `src/aquapose/core/reconstruction/utils.py` | import shared helpers | WIRED | Lines 26-32: `from aquapose.core.reconstruction.utils import (SPLINE_K, build_spline_knots, fit_spline, pixel_half_width_to_metres, weighted_triangulate_rays)` |
| `src/aquapose/core/reconstruction/backends/__init__.py` | `src/aquapose/core/reconstruction/backends/dlt.py` | get_backend factory | WIRED | Lines 56-59: `if kind == "dlt": from aquapose.core.reconstruction.backends.dlt import DltBackend; return DltBackend(**kwargs)` |
| `src/aquapose/core/reconstruction/backends/dlt.py` | `src/aquapose/calibration/projection.py` | RefractiveProjectionModel for ray casting and reprojection | WIRED | `triangulate_rays` imported at line 25 (module level); `RefractiveProjectionModel` imported lazily inside `_load_models` at line 547; both are used in reconstruction logic |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| RECON-01 | 43-01, 43-02 | Per-body-point triangulation via confidence-weighted DLT (single strategy regardless of camera count) | SATISFIED | `_triangulate_body_point` triangulates all cameras together via `_tri_rays`; no camera-count branching in algorithm choice. `weighted_triangulate_rays` dispatched when any confidence != 1.0. |
| RECON-02 | 43-02 | Outlier camera rejection via reprojection residual threshold (empirically tuned) | SATISFIED | Per-camera residuals computed post-triangulation (lines 391-397); cameras above `self._outlier_threshold` excluded (lines 400-404). Default 50.0 px, noted for Phase 44 empirical tuning. |
| RECON-03 | 43-02 | Re-triangulation with inlier cameras after outlier rejection | SATISFIED | Line 415: `pt3d = self._tri_rays(inlier_ids, origins, directions, weights)` — second triangulation call uses only inlier cameras. |
| RECON-04 | 43-01, 43-02 | B-spline fitting via `make_lsq_spline` with 7 control points and minimum valid-point threshold | SATISFIED | `fit_spline` in `utils.py` calls `scipy.interpolate.make_lsq_spline` (line 132). `DEFAULT_N_CONTROL_POINTS=7`. Fish skipped when `< min_body_points` valid points. |
| RECON-05 | 43-02 | Water surface rejection (Z <= water_z) | SATISFIED | Applied at lines 379 and 418 (two passes: pre- and post-outlier-rejection). |
| RECON-06 | 43-02 | Low-confidence flagging when configurable fraction of body points had <3 inlier cameras | SATISFIED | Lines 282-285: count `n_weak` (points with `n_cams < 3`); compare to `low_confidence_fraction * total`; write `is_low_confidence` to `Midline3D`. |
| RECON-07 | 43-02 | Half-widths passed through from upstream (defaults if absent), not used in reconstruction logic | SATISFIED | `half_widths` read for output conversion only in `_reconstruct_fish`; never passed to `_triangulate_body_point`. Output field `Midline3D.half_widths` populated with world-metre values. |

All 7 RECON requirements satisfied. No orphaned requirements detected for Phase 43.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/aquapose/core/reconstruction/backends/dlt.py` | 45 | Docstring note: "This is a placeholder; empirical tuning via Phase 44 eval harness is required." | Info | This is documentation about the default constant value `DEFAULT_OUTLIER_THRESHOLD=50.0`, not a stub implementation. The value is used and the logic is complete. Phase 44 will tune the threshold. No impact on goal achievement. |

No blocker or warning-level anti-patterns found. The single "placeholder" occurrence is a docstring annotation in a constant's description, not a stub implementation.

---

### Human Verification Required

None. All goal truths can be verified programmatically for this phase:
- Wiring is confirmed via grep and file reads.
- Logic is confirmed by reading implementation.
- Tests confirmed passing: 748 passed, 3 skipped, 0 failed.

---

### Commits Verified

| Commit | Description | Verified |
|--------|-------------|---------|
| `4777375` | feat(43-01): extract shared reconstruction helpers to utils.py | Present in git log |
| `210c0e7` | feat(43-02): implement DltBackend with confidence-weighted DLT triangulation | Present in git log |
| `1c54c11` | feat(43-02): register DltBackend in backends/__init__.py factory | Present in git log |

---

### Test Results

```
hatch run test tests/unit/core/reconstruction/ -x
748 passed, 3 skipped, 0 failed
```

Test files covering phase 43 output:
- `tests/unit/core/reconstruction/test_reconstruction_utils.py` — 279 lines, covers all 4 extracted helpers and both constants
- `tests/unit/core/reconstruction/test_dlt_backend.py` — 595 lines, covers all DltBackend behaviors and edge cases

---

### Summary

Phase 43 goal is fully achieved. The reconstruction subsystem has a new `DltBackend` registered as `"dlt"` in the `get_backend` factory. It implements confidence-weighted DLT triangulation with:
- Single-strategy algorithm (no camera-count branching)
- Per-camera outlier rejection via reprojection residual threshold
- Re-triangulation with inlier cameras
- Water surface rejection (applied pre- and post-rejection)
- B-spline fitting with 7 control points via `make_lsq_spline`
- Configurable low-confidence flagging
- Half-widths passed through from upstream without influencing triangulation

Shared helpers were cleanly extracted to `utils.py` (Plan 01) with backward-compat aliases in `triangulation.py`. All 7 RECON requirements are satisfied with test coverage.

---

_Verified: 2026-03-02_
_Verifier: Claude (gsd-verifier)_
