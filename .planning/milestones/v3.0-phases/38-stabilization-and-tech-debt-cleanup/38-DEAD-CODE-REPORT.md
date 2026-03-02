# Dead Code Analysis Report — Phase 38

## Methodology

- `ast.parse()` import graph analysis of all `src/aquapose/**/*.py` files (86 files total)
- Both direct (`from aquapose.X.Y import ...`) and relative import forms resolved
- Cross-checked against `tests/` directory for test-only consumers
- Date: 2026-03-02

## Summary

- Total legacy files analyzed: 16 (across 4 directories)
- Canonical implementations (active, cannot delete without migration): 8
- Thin wrappers / re-exports (no unique logic): 5
- Dead code (not imported by any file): 1
- Package `__init__.py` files (re-export aggregators): 4 (see notes below)

---

## Detailed Findings

### reconstruction/

#### reconstruction/midline.py — CANONICAL

**Imported by (non-legacy consumers):**
- `core/midline/__init__.py` L19: `from aquapose.reconstruction.midline import Midline2D`
- `core/midline/backends/pose_estimation.py` L21: `from aquapose.reconstruction.midline import Midline2D`
- `core/midline/backends/segmentation.py` L20: `from aquapose.reconstruction.midline import (Midline2D, _adaptive_smooth, _longest_path_bfs, _resample_arc_length, _skeleton_and_widths)`
- `core/midline/types.py` L15: `from aquapose.reconstruction.midline import Midline2D`
- `core/reconstruction/stage.py` L25: `from aquapose.reconstruction.midline import Midline2D`
- `core/reconstruction/types.py` L9: `from aquapose.reconstruction.midline import Midline2D`
- `core/synthetic.py` L20: `from aquapose.reconstruction.midline import Midline2D`
- `synthetic/fish.py` L18: `from aquapose.reconstruction.midline import Midline2D`
- (+ visualization/midline_viz.py, visualization/triangulation_viz.py)

**Contains:** `Midline2D` dataclass (the canonical 2D midline type), `MidlineExtractor` class (skeleton extraction logic), and private helpers `_adaptive_smooth`, `_skeleton_and_widths`, `_longest_path_bfs`, `_resample_arc_length` — these private helpers are directly imported by `core/midline/backends/segmentation.py` (not just `Midline2D`).

**Recommendation:** Keep in place. This is the canonical home for `Midline2D` and 2D skeleton extraction logic. All of `core/` imports directly from it. Migration path would require updating 8+ non-legacy consumers and should be a dedicated future plan.

---

#### reconstruction/triangulation.py — CANONICAL

**Imported by (non-legacy consumers):**
- `core/reconstruction/__init__.py` L9: `from aquapose.reconstruction.triangulation import Midline3D`
- `core/reconstruction/backends/curve_optimizer.py` L15: `from aquapose.reconstruction.triangulation import Midline3D, MidlineSet`
- `core/reconstruction/backends/triangulation.py` L14: `from aquapose.reconstruction.triangulation import (DEFAULT_INLIER_THRESHOLD, SPLINE_N_CTRL, Midline3D, MidlineSet, triangulate_midlines)`
- `core/reconstruction/stage.py` L26: `from aquapose.reconstruction.triangulation import DEFAULT_INLIER_THRESHOLD, Midline3D`
- `core/reconstruction/types.py` L10: `from aquapose.reconstruction.triangulation import Midline3D, MidlineSet`
- `io/midline_writer.py` L15,22: `from aquapose.reconstruction.triangulation import ...`
- `synthetic/fish.py` L19: `from aquapose.reconstruction.triangulation import ...`
- (+ visualization/overlay.py, visualization/plot3d.py, visualization/triangulation_viz.py)

**Contains:** `Midline3D` and `MidlineSet` canonical types, `triangulate_midlines()` function (RANSAC + B-spline), `refine_midline_lm()` (Levenberg-Marquardt refinement), plus constants `SPLINE_K`, `SPLINE_N_CTRL`, `DEFAULT_INLIER_THRESHOLD`.

**Recommendation:** Keep in place. This is the canonical home for `Midline3D`, `MidlineSet`, and all triangulation logic. The `core/reconstruction/backends/` modules are thin wrappers that call into this file. Migration would require updating 7+ non-legacy consumers.

---

#### reconstruction/curve_optimizer.py — CANONICAL

**Imported by (non-legacy consumers):**
- `core/reconstruction/backends/curve_optimizer.py` L14: `from aquapose.reconstruction.curve_optimizer import CurveOptimizer, CurveOptimizerConfig`
- (+ reconstruction/__init__.py, visualization/triangulation_viz.py for OptimizerSnapshot)

**Contains:** `CurveOptimizer` class (correspondence-free L-BFGS curve fitting), `CurveOptimizerConfig` dataclass, `OptimizerSnapshot` dataclass, `optimize_midlines()` function, and extensive helper loss functions.

**Recommendation:** Keep in place. The `core/reconstruction/backends/curve_optimizer.py` is a thin stage-protocol wrapper that imports directly from this file. Only 1 non-legacy consumer but it is a load-bearing import.

---

#### reconstruction/__init__.py — THIN WRAPPER (re-export aggregator)

**Imported by (non-legacy consumers):** None that use it as a package-level import. All non-legacy code imports directly from the submodules (`reconstruction.midline`, `reconstruction.triangulation`, `reconstruction.curve_optimizer`).

**Contains:** Re-exports from the three submodules via `__all__`.

**Recommendation:** Keep for now as it provides a clean package API. No consumers are coupled to it from non-legacy code. It is not dead — it can be used by external users — but deleting it would not break any current imports.

---

### segmentation/

#### segmentation/detector.py — CANONICAL

**Imported by (non-legacy consumers):**
- `core/detection/backends/yolo.py` L15: `from aquapose.segmentation.detector import Detection, YOLODetector`
- `core/detection/backends/yolo_obb.py` L20: `from aquapose.segmentation.detector import Detection`
- `core/detection/types.py` L9: `from aquapose.segmentation.detector import Detection`
- `core/midline/backends/pose_estimation.py` L27: `from aquapose.segmentation.detector import Detection`
- `core/midline/backends/segmentation.py` L32: `from aquapose.segmentation.detector import Detection`
- `core/midline/types.py` L17: `from aquapose.segmentation.detector import Detection`
- `core/synthetic.py` L22: `from aquapose.segmentation.detector import Detection`
- `synthetic/detection.py` L17: `from aquapose.segmentation.detector import Detection`

**Contains:** `Detection` dataclass (the canonical detection type used throughout the pipeline), `YOLODetector` class (YOLO inference wrapper for v2.x), `make_detector()` factory function.

**Recommendation:** Keep in place. `Detection` is the canonical detection type imported by 8 non-legacy consumers. The `YOLODetector` class may be v2.x legacy (Stage 1 now uses backends), but `Detection` itself is essential.

---

#### segmentation/crop.py — CANONICAL

**Imported by (non-legacy consumers):**
- `core/midline/backends/pose_estimation.py` L22: `from aquapose.segmentation.crop import (AffineCrop, extract_affine_crop, invert_affine_points)`
- `core/midline/backends/segmentation.py` L27: `from aquapose.segmentation.crop import (AffineCrop, extract_affine_crop, invert_affine_points)`
- `core/midline/types.py` L16: `from aquapose.segmentation.crop import CropRegion`
- `core/synthetic.py` L21: `from aquapose.segmentation.crop import CropRegion`

**Contains:** `CropRegion` dataclass, `AffineCrop` class, `extract_affine_crop()`, `invert_affine_point()`, `invert_affine_points()`, `compute_crop_region()`, `extract_crop()`, `paste_mask()`.

**Recommendation:** Keep in place. Multiple `core/midline/backends/` files import the affine crop machinery directly from this module. Migration would require updating 4 non-legacy consumers.

---

#### segmentation/__init__.py — THIN WRAPPER (re-export aggregator)

**Imported by (non-legacy consumers):** None. All non-legacy code imports directly from `segmentation.crop` or `segmentation.detector`.

**Contains:** Re-exports from `crop.py` and `detector.py` via `__all__`.

**Recommendation:** Keep for now. Not dead — provides a package API — but deleting it would not break any current non-legacy imports.

---

### tracking/

#### tracking/ocsort_wrapper.py — CANONICAL

**Imported by (non-legacy consumers):**
- `core/tracking/stage.py` L76: `from aquapose.tracking.ocsort_wrapper import OcSortTracker`

**Contains:** `OcSortTracker` class (the boxmot OC-SORT wrapper) and `_TrackletBuilder` helper. This is the single file that isolates boxmot internals — by design.

**Recommendation:** Keep in place. `TrackingStage` in `core/` delegates entirely to this wrapper. The isolation boundary is intentional per the tracking package docstring.

---

#### tracking/__init__.py — THIN WRAPPER (re-export aggregator)

**Imported by (non-legacy consumers):** None. `core/tracking/stage.py` imports directly from `tracking.ocsort_wrapper`, not from `tracking.__init__`.

**Contains:** Re-exports `OcSortTracker` from `ocsort_wrapper.py` via `__all__`.

**Recommendation:** Keep for now. Provides a clean package API. Could be deleted without breaking current imports.

---

### visualization/

#### visualization/frames.py — CANONICAL

**Imported by (non-legacy consumers):**
- `engine/overlay_observer.py` L155: `from aquapose.visualization.frames import synthetic_frame_iter`
- `engine/tracklet_trail_observer.py` L399,561: `from aquapose.visualization.frames import synthetic_frame_iter` (imported twice — at function body level with a conditional guard)

**Contains:** `synthetic_frame_iter()` function — the only consumer of visualization from `engine/`.

**Recommendation:** Keep in place. Used by two engine observer files.

---

#### visualization/overlay.py — CANONICAL

**Imported by (non-legacy consumers):**
- `tests/unit/visualization/test_overlay.py`: `from aquapose.visualization.overlay import FISH_COLORS, draw_midline_overlay`
- (+ visualization/midline_viz.py, visualization/plot3d.py, visualization/triangulation_viz.py)

**Contains:** `FISH_COLORS` constant, `draw_midline_overlay()`, `render_overlay_video()`.

**Recommendation:** Keep in place. Has a unit test and is used throughout the visualization subpackage.

---

#### visualization/midline_viz.py — CANONICAL

**Imported by (non-legacy consumers):** None directly from non-legacy code. All access goes through `visualization/__init__.py` (re-exported) or `visualization/diagnostics.py` (also a re-export).

**Contains:** `TrackSnapshot` class, `vis_detection_grid()`, `vis_confidence_histogram()`, `vis_claiming_overlay()`, `vis_midline_extraction_montage()`, `vis_skip_reason_pie()` — diagnostic visualization functions that implement unique logic.

**Recommendation:** Keep in place. Contains unique diagnostic visualization logic. The fact that no external code imports it directly is fine — it is surfaced via `visualization/__init__.py`.

---

#### visualization/triangulation_viz.py — CANONICAL

**Imported by (non-legacy consumers):** None directly from non-legacy code. Surfaced via `visualization/__init__.py`.

**Contains:** `vis_residual_heatmap()`, `vis_arclength_histogram()`, `vis_per_camera_spline_overlays()`, `write_diagnostic_report()`, `vis_synthetic_*` family, `vis_optimizer_progression()` — extensive diagnostic visualization logic.

**Recommendation:** Keep in place. Contains unique diagnostic visualization logic. Surfaced via `visualization/__init__.py`.

---

#### visualization/plot3d.py — CANONICAL

**Imported by (non-legacy consumers):** None directly from non-legacy code. Surfaced via `visualization/__init__.py`.

**Contains:** `plot_3d_frame()`, `render_3d_animation()` — 3D matplotlib visualization functions.

**Recommendation:** Keep in place. Unique 3D plotting logic surfaced via `visualization/__init__.py`.

---

#### visualization/__init__.py — THIN WRAPPER (re-export aggregator)

**Imported by (non-legacy consumers):** None. Non-legacy code (`engine/` files) imports directly from `visualization.frames`, not from `visualization.__init__`.

**Contains:** Re-exports from `frames.py`, `midline_viz.py`, `overlay.py`, `plot3d.py`, `triangulation_viz.py`.

**Recommendation:** Keep for now. Provides a clean package API.

---

#### visualization/diagnostics.py — DEAD CODE (backward-compat shim, no current importers)

**Imported by:** Nobody. Not imported by any file in `src/`, `tests/`, or `scripts/`.

**Contains:** Pure re-export shim. Docstring explains it was created when the original 2200-LOC `diagnostics.py` was split into `midline_viz.py` and `triangulation_viz.py`. It re-exports all public names from both split modules for "backward compatibility" but no current file imports from it.

**Recommendation:** Delete. This is a backwards-compat shim for a migration that is now complete. No current code uses it. All public names it re-exports are available directly from `visualization/__init__.py`.

---

## Recommended Actions

### Action 1 (LOW RISK): Delete `visualization/diagnostics.py`

**Justification:** Confirmed dead code — zero importers across all of `src/`, `tests/`, and `scripts/`. It is a backwards-compat shim that served its purpose when `diagnostics.py` was split. The migration is complete; the shim can be deleted.

**Impact:** None. No file imports from it.

**Files affected:** `src/aquapose/visualization/diagnostics.py` (delete)

---

### Action 2 (INFORMATIONAL): The legacy directories are load-bearing

**Summary:** `reconstruction/midline.py`, `reconstruction/triangulation.py`, `reconstruction/curve_optimizer.py`, `segmentation/detector.py`, `segmentation/crop.py`, and `tracking/ocsort_wrapper.py` are all actively imported by `core/` code. They are **not** dead code — they are the canonical implementations that `core/` wraps.

**The current architecture is intentional:** `core/` provides Stage-Protocol adapters that delegate to the implementations in the legacy directories. These legacy directories are not legacy in the sense of "obsolete" — they are the implementation layer beneath the pipeline abstraction.

**No migration is recommended at this time.** The `core/` adapters are thin wrappers around these implementations. Migrating the implementations into `core/` would touch 20+ import sites and provide minimal benefit given the current stable state of the codebase.

---

### Action 3 (INFORMATIONAL): `__init__.py` wrappers are safe to delete but provide value

The four package `__init__.py` files (`reconstruction/__init__.py`, `segmentation/__init__.py`, `tracking/__init__.py`, `visualization/__init__.py`) are thin re-export aggregators. No non-legacy code imports from the package level — all non-legacy code imports directly from submodules.

However, deleting them would:
1. Break the public API surface if external scripts or notebooks use `from aquapose.visualization import ...`
2. Break `visualization/diagnostics.py` (if it weren't already being deleted per Action 1)

**Recommendation:** Leave all four `__init__.py` files in place. They provide value as package API surfaces with negligible maintenance cost.

---

## Deferred Items

The following items were noted but are out of scope for this cleanup:

- `segmentation/detector.py` contains `YOLODetector` (v2.x wrapper) — the Stage 1 pipeline now uses `core/detection/backends/yolo.py` and `yolo_obb.py` instead. `YOLODetector` may no longer be used in production but `Detection` (defined in the same file) is essential. Separating `Detection` from `YOLODetector` into distinct files would allow cleaning up the legacy detector class, but requires a separate dedicated plan.

- `segmentation/crop.py` exports `compute_crop_region()`, `extract_crop()`, `paste_mask()` which are v2.x API. The v3.0 backends use `AffineCrop` / `extract_affine_crop()`. The v2.x functions may be dead code within their own file — but since `Detection` and `CropRegion` in the same file are canonical, cleanup requires careful scoping.
