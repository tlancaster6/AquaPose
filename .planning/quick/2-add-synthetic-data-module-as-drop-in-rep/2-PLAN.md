---
phase: quick-2
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/synthetic/__init__.py
  - src/aquapose/synthetic/fish.py
  - src/aquapose/synthetic/rig.py
  - src/aquapose/synthetic/stubs.py
  - tests/unit/synthetic/__init__.py
  - tests/unit/synthetic/test_synthetic.py
  - scripts/diagnose_pipeline.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "generate_synthetic_midline_sets returns list[MidlineSet] compatible with triangulation/curve optimizer input"
    - "Ground truth Midline3D objects are returned alongside MidlineSet for comparison"
    - "diagnose_pipeline.py --synthetic skips stages 1-4 and feeds synthetic data to stage 5"
    - "Straight line and circular arc fish shapes are supported"
    - "Both real calibration and fabricated rig cameras work as projection sources"
  artifacts:
    - path: "src/aquapose/synthetic/__init__.py"
      provides: "Public API for synthetic module"
    - path: "src/aquapose/synthetic/fish.py"
      provides: "Fish shape generation and projection to 2D midlines"
    - path: "src/aquapose/synthetic/rig.py"
      provides: "Fabricated camera rig builder"
    - path: "src/aquapose/synthetic/stubs.py"
      provides: "Stub functions for future synthetic Detection/FishTrack generation"
    - path: "tests/unit/synthetic/test_synthetic.py"
      provides: "Unit tests for synthetic module"
    - path: "scripts/diagnose_pipeline.py"
      provides: "Updated with --synthetic flag"
  key_links:
    - from: "src/aquapose/synthetic/fish.py"
      to: "RefractiveProjectionModel.project"
      via: "Projects 3D ground truth points through refractive cameras"
      pattern: "model\\.project"
    - from: "src/aquapose/synthetic/fish.py"
      to: "Midline2D, MidlineSet, Midline3D"
      via: "Returns same types consumed by triangulate_midlines / CurveOptimizer"
      pattern: "Midline2D|MidlineSet|Midline3D"
    - from: "scripts/diagnose_pipeline.py"
      to: "src/aquapose/synthetic/"
      via: "--synthetic flag imports and calls generate_synthetic_midline_sets"
      pattern: "generate_synthetic_midline_sets"
---

<objective>
Add a synthetic data module (`src/aquapose/synthetic/`) that generates known ground truth 3D fish midlines, projects them through refractive camera models to produce 2D MidlineSets, and wires this into diagnose_pipeline.py as a `--synthetic` flag to bypass stages 1-4.

Purpose: Enable controlled testing of triangulation and curve optimization without real video/detection/segmentation dependencies. Known ground truth allows quantitative accuracy measurement.

Output: New `src/aquapose/synthetic/` package, updated `scripts/diagnose_pipeline.py`, unit tests.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/reconstruction/triangulation.py (Midline2D import, MidlineSet type alias, Midline3D dataclass, SPLINE_K, SPLINE_KNOTS, SPLINE_N_CTRL, N_SAMPLE_POINTS constants)
@src/aquapose/reconstruction/midline.py (Midline2D dataclass definition)
@src/aquapose/calibration/projection.py (RefractiveProjectionModel.project and cast_ray signatures)
@src/aquapose/calibration/uncertainty.py (build_synthetic_rig and build_rig_from_calibration as reference for camera rig construction)
@tests/unit/test_curve_optimizer.py (_make_camera, _build_synthetic_rig, _make_gt_spline_pts, _project_points helpers as patterns)
@scripts/diagnose_pipeline.py (full diagnostic script to modify with --synthetic flag)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create synthetic data module with fish generation and projection</name>
  <files>
    src/aquapose/synthetic/__init__.py
    src/aquapose/synthetic/fish.py
    src/aquapose/synthetic/rig.py
    src/aquapose/synthetic/stubs.py
  </files>
  <action>
Create `src/aquapose/synthetic/` package with three modules:

**rig.py** — Fabricated camera rig builder:
- `build_fabricated_rig(n_cameras_x: int = 3, n_cameras_y: int = 3, spacing_x: float = 0.4, spacing_y: float = 0.4, water_z: float = 0.75, n_air: float = 1.0, n_water: float = 1.333, fx: float = 1400.0, cx: float = 800.0, cy: float = 600.0) -> dict[str, RefractiveProjectionModel]`
  - Creates a rectangular grid of cameras at Z=0 (identity rotation, translation = -cam_pos)
  - Camera IDs: `"syn_00"`, `"syn_01"`, etc.
  - All cameras look straight down (R=I), placed symmetrically around origin
  - Pattern: follow `_make_camera` from test_curve_optimizer.py (K with fx, R=I, t=(-x,-y,0), normal=(0,0,-1))

**fish.py** — Fish shape generation and projection:
- `@dataclass class FishConfig`: position (3-tuple, default (0,0,1.25)), heading_rad (float, default 0.0), curvature (float, default 0.0 for straight), scale (float, default 0.085), n_points (int, default N_SAMPLE_POINTS=15)
- `generate_fish_3d(config: FishConfig) -> np.ndarray`:
  - Returns shape (n_points, 3) float32 body points
  - For curvature == 0: straight line along heading direction, centered at position, total length = scale
  - For curvature != 0: circular arc with radius = 1/abs(curvature), subtending arc_length = scale, centered at position, oriented by heading_rad (rotation around Z axis)
  - Points are evenly spaced in arc-length
- `generate_fish_half_widths(n_points: int = N_SAMPLE_POINTS, max_ratio: float = 0.08) -> np.ndarray`:
  - Returns shape (n_points,) float32 half-widths in world metres
  - Elliptical profile: thickest at 40% from head, tapered at head and tail
  - max_ratio is fraction of scale (default gives ~6.8mm max half-width for 85mm fish)
- `make_ground_truth_midline3d(fish_id: int, frame_index: int, pts_3d: np.ndarray, half_widths: np.ndarray) -> Midline3D`:
  - Fits a B-spline to pts_3d using scipy.interpolate.make_lsq_spline with SPLINE_KNOTS, SPLINE_K
  - Returns Midline3D with control_points, knots, degree, arc_length, half_widths, n_cameras=99, mean_residual=0.0, max_residual=0.0
- `project_fish_to_midline2d(pts_3d: np.ndarray, half_widths_3d: np.ndarray, model: RefractiveProjectionModel, fish_id: int, camera_id: str, frame_index: int) -> Midline2D | None`:
  - Projects pts_3d through model.project(), returns None if <3 points visible
  - Converts half_widths from world metres to pixels using pinhole approximation: hw_px = hw_m * fx / depth (where depth = pt_z - water_z, fx from model.K[0,0])
  - Returns Midline2D with points, half_widths, fish_id, camera_id, frame_index, is_head_to_tail=True
- `generate_synthetic_midline_sets(models: dict[str, RefractiveProjectionModel], fish_configs: list[FishConfig] | None = None, n_frames: int = 1, frame_start: int = 0) -> tuple[list[MidlineSet], list[dict[int, Midline3D]]]`:
  - If fish_configs is None, default to one straight fish at (0, 0, 1.25) heading=0
  - For each frame, for each fish, generate 3D points, project to every camera, build MidlineSet
  - Returns (midline_sets, ground_truths) where ground_truths[frame_idx][fish_id] = Midline3D
  - Fish IDs are 0-indexed integers matching list position

**stubs.py** — Future expansion stubs:
- `def generate_synthetic_detections(...) -> ...: raise NotImplementedError("Stub: synthetic Detection generation for testing midline extraction")`
- `def generate_synthetic_tracks(...) -> ...: raise NotImplementedError("Stub: synthetic FishTrack generation for testing midline extraction with tracking")`
- Include docstrings describing what these would produce when implemented

**__init__.py** — Public API:
- Import and export: `FishConfig`, `generate_fish_3d`, `generate_fish_half_widths`, `make_ground_truth_midline3d`, `project_fish_to_midline2d`, `generate_synthetic_midline_sets`, `build_fabricated_rig`, `generate_synthetic_detections`, `generate_synthetic_tracks`
- Define `__all__`

Use imports from existing codebase:
- `from aquapose.reconstruction.midline import Midline2D`
- `from aquapose.reconstruction.triangulation import Midline3D, MidlineSet, SPLINE_KNOTS, SPLINE_K, SPLINE_N_CTRL, N_SAMPLE_POINTS`
- `from aquapose.calibration.projection import RefractiveProjectionModel`

Follow project conventions: Google docstrings, type hints on all public functions, ruff-compatible formatting.
  </action>
  <verify>
    `hatch run check` passes (lint + typecheck).
    `python -c "from aquapose.synthetic import generate_synthetic_midline_sets, build_fabricated_rig; rig = build_fabricated_rig(); ms, gt = generate_synthetic_midline_sets(rig); print(f'Cameras: {len(rig)}, Frames: {len(ms)}, Fish: {len(gt[0])}')"` prints "Cameras: 9, Frames: 1, Fish: 1".
  </verify>
  <done>
    synthetic package exists with FishConfig, generate_synthetic_midline_sets, build_fabricated_rig, and stubs. Generates MidlineSet and Midline3D ground truth from configurable fish shapes projected through refractive camera models.
  </done>
</task>

<task type="auto">
  <name>Task 2: Add unit tests and wire --synthetic flag into diagnose_pipeline.py</name>
  <files>
    tests/unit/synthetic/__init__.py
    tests/unit/synthetic/test_synthetic.py
    scripts/diagnose_pipeline.py
  </files>
  <action>
**tests/unit/synthetic/__init__.py** — Empty init file.

**tests/unit/synthetic/test_synthetic.py** — Unit tests:
- `test_build_fabricated_rig_default`: 3x3 grid = 9 cameras, all camera IDs start with "syn_", all are RefractiveProjectionModel instances
- `test_build_fabricated_rig_custom`: Custom 2x4 grid with custom spacing returns 8 cameras
- `test_generate_fish_3d_straight`: curvature=0 produces straight line along heading direction with correct number of points and total arc length ~= scale
- `test_generate_fish_3d_arc`: curvature=10 (radius=0.1m) produces curved shape with correct arc length ~= scale
- `test_generate_fish_3d_heading`: heading_rad=pi/2 rotates fish 90 degrees (verify points spread along Y instead of X)
- `test_project_fish_returns_midline2d`: Project straight fish through one camera, verify Midline2D with correct shape (15,2) points, (15,) half_widths, correct fish_id/camera_id/frame_index
- `test_project_fish_returns_none_when_not_visible`: Camera looking away from fish returns None
- `test_generate_synthetic_midline_sets_structure`: Default call returns (list[MidlineSet], list[dict]) with correct lengths; MidlineSet has fish_id 0 with multiple camera entries
- `test_generate_synthetic_midline_sets_multi_fish`: 3 fish configs, verify all 3 fish_ids present in MidlineSet
- `test_ground_truth_midline3d_valid`: Verify ground truth Midline3D has control_points shape (7,3), knots shape (11,), degree=3, arc_length > 0
- `test_round_trip_accuracy`: Generate 3D fish, project to cameras, verify projected 2D points are within 1px of direct model.project() call (sanity check, not testing triangulation)

All tests use `build_fabricated_rig()` with default params. No `@pytest.mark.slow` needed (all tests should be <1s).

**scripts/diagnose_pipeline.py** — Add `--synthetic` flag:
- Add argument: `--synthetic` (store_true, default False), help="Use synthetic fish data instead of real video (bypasses stages 1-4)"
- Add argument: `--n-fish` (int, default 1), help="Number of synthetic fish (only with --synthetic)"
- Add argument: `--n-synthetic-cameras` (int, default 4), help="Number of cameras per axis for fabricated rig (NxN grid, only with --synthetic without --calibration)"
- When `--synthetic` is active:
  1. Skip check_paths() for video/unet/yolo (still check calibration if provided)
  2. Build camera models: if calibration path exists, use `build_rig_from_calibration` + wrap as dict with cam IDs; otherwise use `build_fabricated_rig(n_cameras_x=args.n_synthetic_cameras, n_cameras_y=args.n_synthetic_cameras)`
  3. Build fish configs: create `args.n_fish` FishConfig objects with positions spread in a line along X (spacing 0.1m), alternating straight (curvature=0) and arc (curvature=15) shapes
  4. Call `generate_synthetic_midline_sets(models, fish_configs, n_frames=args.stop_frame)`
  5. Skip stages 1-4 entirely, go straight to Stage 5 with the synthetic midline_sets
  6. After Stage 5, compute and print ground truth comparison: for each fish in each frame, compute mean Euclidean distance between reconstructed control points and ground truth control points (after aligning by fish_id). Print per-fish mean error and overall mean.
  7. Still run applicable diagnostic visualizations: 3d_animation, residual_heatmap, arclength_histogram. Skip detection_grid, confidence_histogram, claiming_overlay, midline_montage, skip_reasons (these need real detection data). For spline_overlays, skip since there's no VideoSet.
  8. Still write HDF5 output and timing summary
  9. Print "=== Synthetic Ground Truth Comparison ===" section with per-fish and overall control point error in mm
  </action>
  <verify>
    `hatch run test tests/unit/synthetic/` passes all tests.
    `hatch run check` passes (lint + typecheck).
    `python scripts/diagnose_pipeline.py --synthetic --stop-frame 3 --method triangulation` runs without error and prints ground truth comparison metrics.
  </verify>
  <done>
    Unit tests validate synthetic module correctness. diagnose_pipeline.py --synthetic bypasses real data pipeline, feeds synthetic midlines to stage 5, and prints ground truth comparison metrics showing reconstruction accuracy.
  </done>
</task>

</tasks>

<verification>
- `hatch run check` passes (lint + typecheck across entire codebase)
- `hatch run test tests/unit/synthetic/` passes all synthetic module tests
- `python scripts/diagnose_pipeline.py --synthetic --stop-frame 3` completes and prints timing + ground truth comparison
- `python scripts/diagnose_pipeline.py --synthetic --stop-frame 3 --method curve` also completes (both reconstruction methods work)
</verification>

<success_criteria>
- `generate_synthetic_midline_sets` produces `list[MidlineSet]` consumable by both `triangulate_midlines` and `CurveOptimizer.optimize_midlines`
- Ground truth `Midline3D` objects enable quantitative accuracy comparison
- `--synthetic` flag in diagnose_pipeline.py enables running the full diagnostic pipeline without any real data files
- Both straight line and circular arc fish shapes produce valid projections in all visible cameras
</success_criteria>

<output>
After completion, create `.planning/quick/2-add-synthetic-data-module-as-drop-in-rep/2-SUMMARY.md`
</output>
