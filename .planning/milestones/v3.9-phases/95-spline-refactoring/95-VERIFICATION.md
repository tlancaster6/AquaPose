---
phase: 95-spline-refactoring
verified: 2026-03-13T23:00:00Z
status: passed
score: 9/9 must-haves verified
re_verification: null
gaps: []
human_verification: []
---

# Phase 95: Spline Refactoring Verification Report

**Phase Goal:** B-spline fitting is no longer in the core reconstruction path — reconstruction produces raw triangulated keypoints directly, and spline fitting is available as a separate optional utility
**Verified:** 2026-03-13T23:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | DltBackend with spline_enabled=False returns Midline3D with raw triangulated keypoints in points field, control_points=None | VERIFIED | dlt.py lines 372-399: raw path sets points=pts_3d_arr_full, control_points=None; runtime check passes |
| 2  | DltBackend with spline_enabled=True returns Midline3D with spline control points (backward compat) | VERIFIED | dlt.py lines 401-483: spline path populates control_points, knots, degree, arc_length; 30 DLT tests pass |
| 3  | Midline3D supports both raw-keypoint and spline-fitted states without type errors | VERIFIED | reconstruction.py: all formerly-required fields (control_points, knots, degree, arc_length) now optional with None defaults; runtime instantiation confirmed; 0 typecheck errors |
| 4  | fit_spline() is callable as a standalone utility on raw 3D keypoints | VERIFIED | utils.py line 106: fit_spline defined and exported in __all__ (line 30); runtime callable check passes |
| 5  | ReconstructionConfig has spline_enabled toggle defaulting to False | VERIFIED | config.py line 358: spline_enabled: bool = False; runtime assertion passes |
| 6  | Midline3DWriter correctly writes raw-keypoint Midline3D to HDF5 points dataset | VERIFIED | midline_writer.py line 184: branch on midline.points is not None writes raw keypoints; HDF5 round-trip confirmed: points shape (1, 2, 6, 3), control_points all NaN |
| 7  | HDF5 output contains a points dataset for raw keypoints | VERIFIED | midline_writer.py line 105: _make("points"...) always creates dataset; reader returns points key; backward-compat: None for legacy files without dataset |
| 8  | Evaluation metrics work with raw-keypoint Midline3D | VERIFIED | evaluation/stages/reconstruction.py line 384-402: branches on control_points is not None for spline mode, midline.points for raw mode |
| 9  | spline_enabled flows from config through pipeline.py to ReconstructionStage to DltBackend | VERIFIED | pipeline.py line 346: spline_enabled=config.reconstruction.spline_enabled; stage.py line 171: forwarded in combined_kwargs to get_backend(); dlt.py __init__ receives and stores it |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/core/types/reconstruction.py` | Midline3D with optional control_points and new points field | VERIFIED | Both fields present; field order correct (required first, optional with defaults after); class docstring describes both modes |
| `src/aquapose/engine/config.py` | spline_enabled toggle in ReconstructionConfig | VERIFIED | spline_enabled: bool = False at line 358 |
| `src/aquapose/core/reconstruction/backends/dlt.py` | Conditional spline fitting in _reconstruct_fish | VERIFIED | if not self._spline_enabled branch at line 372; spline-enabled backward-compat path at line 401 |
| `src/aquapose/core/reconstruction/stage.py` | spline_enabled parameter accepted and forwarded | VERIFIED | __init__ parameter at line 153; stored and forwarded to backend via combined_kwargs |
| `src/aquapose/engine/pipeline.py` | spline_enabled wired from config to ReconstructionStage | VERIFIED | Line 346: spline_enabled=config.reconstruction.spline_enabled |
| `src/aquapose/io/midline_writer.py` | HDF5 writer handling both raw-keypoint and spline-fitted Midline3D | VERIFIED | Dual-dataset pattern: both points and control_points datasets always created; unused one NaN-filled |
| `src/aquapose/evaluation/stages/reconstruction.py` | Evaluation handles raw-keypoint Midline3D | VERIFIED | Branch at lines 384-402 handles both spline and raw-keypoint modes |
| `tests/unit/core/reconstruction/test_dlt_backend.py` | Tests for both spline-enabled and spline-disabled paths | VERIFIED | TestSplineDisabled (3 tests) and TestSplineEnabled (2 tests) added; dlt_backend fixture updated to spline_enabled=True; dlt_backend_raw fixture added |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/aquapose/engine/config.py` | `src/aquapose/core/reconstruction/backends/dlt.py` | spline_enabled passed through ReconstructionStage to DltBackend | WIRED | pipeline.py line 346 -> stage.py line 171 -> dlt.py __init__ line 155 |
| `src/aquapose/core/reconstruction/backends/dlt.py` | `src/aquapose/core/types/reconstruction.py` | DltBackend populates Midline3D.points for raw-keypoint mode | WIRED | dlt.py line 389: points=pts_3d_arr_full in raw path |
| `src/aquapose/engine/pipeline.py` | `src/aquapose/core/reconstruction/stage.py` | spline_enabled kwarg in ReconstructionStage constructor | WIRED | pipeline.py line 346; stage.py accepts at line 153 |
| `src/aquapose/io/midline_writer.py` | `src/aquapose/core/types/reconstruction.py` | write_frame reads midline.points or midline.control_points | WIRED | midline_writer.py lines 183-190: if midline.points is not None; line 193: if midline.control_points is not None |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| SPL-01 | 95-01, 95-02 | B-spline fitting moved out of core reconstruction path into optional post-processing utility | SATISFIED | fit_spline() remains importable from aquapose.core.reconstruction.utils; DltBackend with default spline_enabled=False skips it entirely |
| SPL-02 | 95-01, 95-02 | Reconstruction produces raw triangulated keypoints as primary Midline3D output when spline is disabled | SATISFIED | DltBackend._reconstruct_fish raw path returns Midline3D with points=pts_3d_arr_full; pipeline default (spline_enabled=False) is raw mode |
| SPL-03 | 95-01 | Midline3D type updated to support both spline-based and raw-keypoint representations | SATISFIED | Midline3D has both points and control_points optional fields; docstring describes both modes; all formerly-required spline fields are now None-defaulted |

No orphaned requirements. All SPL-01, SPL-02, SPL-03 explicitly claimed in plan frontmatter and satisfied by implementation.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/aquapose/core/reconstruction/stage.py` | 5, 112 | Docstring says "produces 3D B-spline midlines" — stale since spline is now disabled by default | Warning | Documentation-only; wiring is correct; confusing for readers but does not affect behavior |

### Human Verification Required

None. All key behaviors are verifiable programmatically. The full test suite (1203 pass, 0 failures) and typecheck (0 errors) confirm correctness. The HDF5 round-trip was verified via runtime script.

### Gaps Summary

No gaps. All 9 observable truths verified, all 8 artifacts pass all three levels (exists, substantive, wired), all 4 key links confirmed wired, all 3 requirement IDs satisfied.

The only notable finding is a stale docstring in `stage.py` (lines 5 and 112) that still says "B-spline midlines" — this is a warning-level documentation inaccuracy. It does not block goal achievement. The DOC-01 requirement (stage.py module docstring updated) is mapped to Phase 96, so this is expected to be fixed in the next phase.

---

_Verified: 2026-03-13T23:00:00Z_
_Verifier: Claude (gsd-verifier)_
