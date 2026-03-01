---
phase: 33-keypoint-midline-backend
verified: 2026-03-01T02:15:00Z
status: passed
score: 13/13 must-haves verified
re_verification: false
---

# Phase 33: Keypoint Midline Backend Verification Report

**Phase Goal:** Pipeline supports a keypoint regression backend that produces N ordered midline points with per-point confidence, and both reconstruction backends weight observations by that confidence
**Verified:** 2026-03-01T02:15:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

#### Plan 01 Truths (MID-01 through MID-04)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | DirectPoseBackend.__init__ no longer raises NotImplementedError | VERIFIED | `src/aquapose/core/midline/backends/direct_pose.py` lines 65-118: full `__init__` loading `_PoseModel`, validating weights path, storing config — no `NotImplementedError` anywhere in file. `test_midline_stage.py` line 6 confirms "no longer raises NotImplementedError". |
| 2 | DirectPoseBackend.process_frame() returns dict[str, list[AnnotatedDetection]] with Midline2D containing exactly n_sample_points and non-None point_confidence | VERIFIED | `process_frame` (line 120) returns `dict[str, list[AnnotatedDetection]]`. `_process_single_detection` builds `Midline2D` with `points=midline_pts` of shape `(self._n_points, 2)` and `point_confidence=midline_conf` of shape `(self._n_points,)` — never None on the success path. 9 tests in `test_direct_pose_backend.py` cover this. |
| 3 | Partial visibility: keypoints below confidence_floor replaced by NaN+conf=0; fewer than min_observed_keypoints yields empty midline | VERIFIED | Lines 269-283: `visible_mask = conf >= self._conf_floor`; if `n_visible < self._min_observed` returns `None`. Lines 352-365: `midline_pts[outside_span] = np.nan`; `midline_conf[outside_span] = 0.0`. Tests `test_partial_visibility_nan_padding` and `test_below_min_observed_returns_none_midline` cover both branches. |
| 4 | Output midline always has exactly n_sample_points points; NaN+conf=0 outside [t_min_observed, t_max_observed] | VERIFIED | Line 343: `t_eval = np.linspace(0.0, 1.0, self._n_points)` — always exactly `n_points` elements. Lines 352-365: `outside_span = (t_eval < t_min_obs) \| (t_eval > t_max_obs)` marks NaN+0 outside span. Test `test_output_always_n_sample_points` verifies shape. |
| 5 | MidlineConfig has keypoint_weights_path, keypoint_t_values, keypoint_confidence_floor, min_observed_keypoints fields | VERIFIED | `src/aquapose/engine/config.py` lines 112-115: all four fields present with correct types and defaults (`None`, `None`, `0.1`, `3`). Docstring at lines 91-99 documents each field. |
| 6 | build_stages() passes direct_pose-specific fields to DirectPoseBackend when backend='direct_pose' | VERIFIED | `src/aquapose/engine/pipeline.py` line 335 passes `midline_config=config.midline`. `src/aquapose/core/midline/stage.py` lines 150-184: branches on `backend == "direct_pose"` and extracts `keypoint_weights_path`, `keypoint_t_values`, `keypoint_confidence_floor`, `min_observed_keypoints` from `midline_config`, passing them to `get_backend("direct_pose", ...)`. |
| 7 | aquapose prep calibrate-keypoints --help shows --annotations and --output flags | VERIFIED | `src/aquapose/training/prep.py` lines 27-44: `--annotations` (required, `click.Path(exists=True)`) and `--output` (required, `click.Path()`) options defined. `src/aquapose/cli.py` lines 19, 190: `prep_group` imported and registered via `cli.add_command(prep_group)`. `src/aquapose/training/__init__.py` lines 13, 17-25: `prep_group` in imports and `__all__`. |
| 8 | Both backends produce Midline2D with identical shape and field structure | VERIFIED | `DirectPoseBackend` produces `Midline2D(points=(n_points,2), half_widths=(n_points,), fish_id, camera_id, frame_index, is_head_to_tail=True, point_confidence=(n_points,))`. This matches the `Midline2D` dataclass contract used by `SegmentThenExtractBackend`. Test `test_both_backends_same_shape` in `test_direct_pose_backend.py` (line 403) verifies shape equivalence. |

#### Plan 02 Truths (RECON-01 and RECON-02)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 9 | triangulate_midlines() with point_confidence produces different (better-weighted) 3D points than without | VERIFIED | `triangulation.py` lines 912-916: extracts `sqrt(point_confidence[i])` per body point, builds `pt_weights` dict, passes to `_triangulate_body_point(..., weights=pt_weights)`. `_tri_rays()` (line 241) dispatches to `_weighted_triangulate_rays` when `_use_weights=True`. Test `test_confidence_vs_no_confidence_differ_when_noisy` covers this. |
| 10 | triangulate_midlines() with point_confidence=None produces identical output to the previous version (backward compatibility) | VERIFIED | Lines 915-916: `else: pt_weights[cam_id] = 1.0` — all-uniform weights. `_triangulate_body_point` line 237-239: `_use_weights = weights is not None and any(v != 1.0 ...)` — all-1.0 weights → `_use_weights=False` → calls `triangulate_rays(origs, dirs)` (unchanged code path). Tests `test_uniform_weights_match_unweighted` and `test_none_confidence_backward_compat` cover this. |
| 11 | CurveOptimizer with point_confidence applies sqrt weighting to chamfer obs->proj direction | VERIFIED | `curve_optimizer.py` line 300: `_weighted_chamfer_distance_2d` defined. Lines 987-989: `conf_clean = midline2d.point_confidence[valid_mask]`; line 990: `cam_conf[cam_id] = torch.from_numpy(np.sqrt(conf_clean)).float()`. Lines 420-427: `_weighted_chamfer_distance_2d(proj_valid, obs_pts, conf_weights)` called when weights available. |
| 12 | CurveOptimizer with point_confidence=None produces identical output to the previous version (backward compatibility) | VERIFIED | `_data_loss` signature (line 350): `confidence_per_fish: ... | None = None`. Lines 422-429: only uses weighted chamfer when `confidence_per_fish is not None` and `conf_weights is not None`; otherwise falls back to `_chamfer_distance_2d`. Tests `test_none_confidence_backward_compat` and `test_all_none_weights_uses_unweighted_chamfer` cover this. |
| 13 | NaN-coordinate body points are excluded entirely from the DLT system; _weighted_triangulate_rays() is a local helper in triangulation.py — triangulate_rays() in calibration/projection.py is NOT modified | VERIFIED | `triangulation.py` line 908: `if np.any(np.isnan(pt)): continue` — NaN points skipped entirely before `pixels` dict is populated. `_weighted_triangulate_rays` defined at line 130 in `triangulation.py`. `calibration/projection.py` grep for "weight" or "confidence" returns nothing — unmodified. Test `test_nan_points_excluded_from_dlt` covers NaN exclusion. |

**Score:** 13/13 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/core/midline/backends/direct_pose.py` | Full DirectPoseBackend implementation with inference, spline fitting, NaN-padding | VERIFIED | 384 lines. Full implementation: `__init__` loads `_PoseModel`, `process_frame` dispatches to `_process_single_detection`, which runs inference, CubicSpline/linear fallback, NaN-padding, confidence interpolation. No stubs. |
| `src/aquapose/training/prep.py` | prep_group CLI with calibrate-keypoints subcommand | VERIFIED | 151 lines. `prep_group` click group + `calibrate-keypoints` subcommand with `--annotations`, `--output`, `--n-keypoints`. Full COCO parsing, arc-length fraction computation, YAML output. |
| `tests/unit/core/midline/test_direct_pose_backend.py` | Unit tests for DirectPoseBackend inference, partial visibility, output shape | VERIFIED | 472 lines. 9 test functions covering constructor validation, output shape, partial visibility NaN-padding, below-min-observed None return, angle=None handling, confidence heuristic, both-backends-same-shape. |
| `src/aquapose/reconstruction/triangulation.py` | Confidence-weighted DLT triangulation via _weighted_triangulate_rays | VERIFIED | `_weighted_triangulate_rays` at line 130 (normal equations with per-camera weights). `_triangulate_body_point` accepts `weights: dict[str, float] | None = None`. `triangulate_midlines` extracts `sqrt(confidence)` at lines 912-916. |
| `src/aquapose/reconstruction/curve_optimizer.py` | Confidence-weighted chamfer in _data_loss via _weighted_chamfer_distance_2d | VERIFIED | `_weighted_chamfer_distance_2d` at line 300. `_data_loss` accepts `confidence_per_fish` kwarg. `optimize_midlines` builds `confidence_per_fish` at lines 968-998 and propagates to all 9 `_data_loss` call sites. |
| `tests/unit/core/reconstruction/test_confidence_weighting.py` | Unit tests for confidence weighting in both reconstruction backends | VERIFIED | 665 lines. 12 test cases across 4 test classes: `TestWeightedTriangulateRays`, `TestTriangulateMidlinesConfidence`, `TestWeightedChamfer`, `TestDataLossConfidence`. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/aquapose/core/midline/backends/direct_pose.py` | `src/aquapose/training/pose.py` | imports `_PoseModel` for inference | WIRED | Line 23 (module-level): `from aquapose.segmentation.crop import ...`. Line 79 (lazy inside `__init__`): `from aquapose.training.pose import _PoseModel`. Both imports confirmed present. |
| `src/aquapose/core/midline/backends/direct_pose.py` | `src/aquapose/segmentation/crop.py` | uses `extract_affine_crop` and `invert_affine_points` | WIRED | Line 23: `from aquapose.segmentation.crop import extract_affine_crop, invert_affine_points`. Both used in `_process_single_detection` (lines 230, 292). |
| `src/aquapose/engine/pipeline.py` | `src/aquapose/engine/config.py` | reads `keypoint_weights_path` from MidlineConfig for direct_pose backend | WIRED | `pipeline.py` line 335 passes `midline_config=config.midline` to `MidlineStage`. `stage.py` lines 153-176 reads `keypoint_weights_path`, `keypoint_t_values`, `keypoint_confidence_floor`, `min_observed_keypoints` from `midline_config`. All four fields confirmed in `MidlineConfig` (config.py lines 112-115). |
| `src/aquapose/reconstruction/triangulation.py` | `src/aquapose/reconstruction/midline.py` | reads `Midline2D.point_confidence` for per-view weights | WIRED | Pattern `midline.point_confidence` found at line 913. Also accesses `midline.points[i]` (line 907), `midline.half_widths[i]` (line 911). Full wiring confirmed. |
| `src/aquapose/reconstruction/curve_optimizer.py` | `src/aquapose/reconstruction/midline.py` | reads `Midline2D.point_confidence` for weighted chamfer | WIRED | Pattern `midline2d.point_confidence` found at line 988. Full wiring: reads, takes `np.sqrt`, converts to tensor, builds `confidence_per_fish` structure passed to all `_data_loss` calls. |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| MID-01 | 33-01 | Pipeline supports a keypoint regression backend as a swappable alternative to segment-then-extract, selectable via config | SATISFIED | `get_backend("direct_pose", ...)` in `backends/__init__.py`. `MidlineConfig.backend` field. `MidlineStage` branches on `backend == "direct_pose"` at stage.py line 150. Config YAML can set `midline.backend: direct_pose`. |
| MID-02 | 33-01 | Keypoint backend produces N ordered midline points with per-point confidence from a learned regression model | SATISFIED | `_PoseModel` loaded in `DirectPoseBackend.__init__`. `process_frame` runs inference and produces `Midline2D` with `point_confidence` array of shape `(n_points,)` — non-None on success path. |
| MID-03 | 33-01 | Keypoint backend handles partial visibility by marking unobserved regions with NaN coordinates and zero confidence, always outputting exactly `n_sample_points` | SATISFIED | `t_eval = np.linspace(0.0, 1.0, self._n_points)` always produces exactly `n_points`. `outside_span` mask sets NaN and zero confidence outside `[t_min_obs, t_max_obs]`. |
| MID-04 | 33-01 | Both midline backends produce the same output structure (N-point Midline2D) so reconstruction is backend-agnostic | SATISFIED | Both backends return `Midline2D` with identical fields. `DirectPoseBackend` produces `half_widths=np.zeros(n_points)` to maintain structural compatibility. `get_backend` returns same type for both. |
| RECON-01 | 33-02 | Triangulation backend weights per-point observations by confidence when available, falling back to uniform weights when confidence is None | SATISFIED | `triangulate_midlines` builds `pt_weights` with `sqrt(confidence)` or `1.0` fallback. `_triangulate_body_point` dispatches to `_weighted_triangulate_rays` or `triangulate_rays` based on `_use_weights` flag. |
| RECON-02 | 33-02 | Curve optimizer backend weights observations by confidence when available, falling back to uniform weights when confidence is None | SATISFIED | `optimize_midlines` builds `confidence_per_fish` parallel structure. `_data_loss` dispatches to `_weighted_chamfer_distance_2d` or `_chamfer_distance_2d` based on `conf_weights` availability. |

**All 6 requirement IDs satisfied. No orphaned requirements found.**

REQUIREMENTS.md status column confirms all 6 marked `[x] Complete` for Phase 33.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/aquapose/reconstruction/triangulation.py` | 1074 | `STUB: Returns midline_3d unchanged` in `refine_midline_lm` | Info | Pre-existing stub, not phase 33 work. Unrelated to phase goal. No impact on RECON-01 or RECON-02. |

No blocker anti-patterns found in phase 33 artifacts.

---

### Human Verification Required

None. All phase goal truths are verifiable programmatically from the codebase. The 4 commits documented in the summaries are confirmed in git (`4926f4d`, `cf1b42d`, `8de6a04`, `ec31191`).

---

### Gaps Summary

No gaps. All 13 must-have truths verified, all 6 artifacts present and substantive, all 5 key links wired, all 6 requirement IDs satisfied.

---

## Commit Evidence

| Commit | Message | Phase work |
|--------|---------|------------|
| `4926f4d` | feat(33-01): implement DirectPoseBackend with keypoint inference and NaN-padding | Task 1 of Plan 01 |
| `cf1b42d` | feat(33-01): extend MidlineConfig, wire build_stages, add prep CLI | Task 2 of Plan 01 |
| `8de6a04` | feat(33-02): add confidence-weighted DLT triangulation to triangulation.py | Task 1 of Plan 02 |
| `ec31191` | feat(33-02): add confidence-weighted chamfer to curve_optimizer.py | Task 2 of Plan 02 |

---

_Verified: 2026-03-01T02:15:00Z_
_Verifier: Claude (gsd-verifier)_
