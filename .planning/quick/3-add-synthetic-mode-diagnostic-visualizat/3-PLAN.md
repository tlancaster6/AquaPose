---
phase: quick-3
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/visualization/diagnostics.py
  - src/aquapose/visualization/__init__.py
  - scripts/diagnose_pipeline.py
autonomous: true
requirements: [QUICK-3]
must_haves:
  truths:
    - "Running --synthetic produces GT-vs-reconstructed 3D comparison plot"
    - "Running --synthetic produces per-camera 2D overlay PNGs in a subdirectory"
    - "Running --synthetic produces a multi-panel error distribution figure"
    - "Running --synthetic produces a synthetic_report.md with per-fish and per-camera tables"
  artifacts:
    - path: "src/aquapose/visualization/diagnostics.py"
      provides: "vis_synthetic_3d_comparison, vis_synthetic_camera_overlays, vis_synthetic_error_distribution, write_synthetic_report"
    - path: "src/aquapose/visualization/__init__.py"
      provides: "Public exports for 4 new functions"
    - path: "scripts/diagnose_pipeline.py"
      provides: "Integration of 4 new vis calls in _run_synthetic()"
  key_links:
    - from: "scripts/diagnose_pipeline.py"
      to: "src/aquapose/visualization/diagnostics.py"
      via: "import and call in _run_synthetic vis_funcs_syn list"
      pattern: "vis_synthetic_3d_comparison|vis_synthetic_camera_overlays|vis_synthetic_error_distribution|write_synthetic_report"
---

<objective>
Add 4 synthetic-mode diagnostic visualizations and a synthetic report to diagnose_pipeline.py.

Purpose: The synthetic mode currently produces only 3 generic visualizations and a console-only GT comparison. Adding GT-vs-predicted overlays, per-camera projection overlays, error distribution plots, and a structured markdown report will make synthetic mode a proper diagnostic tool for evaluating reconstruction quality.

Output: 4 new public functions in diagnostics.py, exported via __init__.py, wired into _run_synthetic().
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/visualization/diagnostics.py
@src/aquapose/visualization/overlay.py
@src/aquapose/visualization/plot3d.py
@src/aquapose/visualization/__init__.py
@scripts/diagnose_pipeline.py
@src/aquapose/reconstruction/triangulation.py (Midline3D dataclass, MidlineSet type)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement 4 synthetic diagnostic functions in diagnostics.py</name>
  <files>
    src/aquapose/visualization/diagnostics.py
    src/aquapose/visualization/__init__.py
  </files>
  <action>
Add 4 new functions to the end of `diagnostics.py` (after the existing `write_diagnostic_report` function), in a new section headed `# Synthetic Mode Diagnostics`. Import `FishConfig` under TYPE_CHECKING from `aquapose.synthetic`. Add `import scipy.interpolate` and `import torch` as lazy imports inside the functions that need them (following the pattern in `vis_per_camera_spline_overlays`).

**1. `vis_synthetic_3d_comparison`**

Signature:
```python
def vis_synthetic_3d_comparison(
    midlines_3d: list[dict[int, Midline3D]],
    ground_truths: list[dict[int, Midline3D]],
    output_path: Path,
    *,
    n_eval: int = 30,
) -> None:
```

- Create a matplotlib 3D figure (`fig = plt.figure(figsize=(12, 9))`, `ax = fig.add_subplot(111, projection="3d")`).
- For each fish_id found in ground_truths (use frame 0 or last frame with data), evaluate both GT and reconstructed B-splines at `n_eval` points via `scipy.interpolate.BSpline`.
- Plot GT midline as dashed line (`linestyle="--"`, label=`f"GT Fish {fish_id}"`), reconstructed as solid line, both using `FISH_COLORS[fish_id % len(FISH_COLORS)]` converted to RGB floats (BGR->RGB: `(b/255, g/255, r/255)`).
- Annotate each fish with mean control-point error in mm (compute `np.linalg.norm(recon.control_points - gt.control_points, axis=1).mean() * 1000`).
- Use `_robust_bounds` from `plot3d.py` (import it: `from aquapose.visualization.plot3d import _robust_bounds`) for axis scaling on the combined GT+recon points.
- Set axis labels X/Y/Z (m), title "GT vs Reconstructed 3D Midlines", legend.
- Save with `dpi=150, bbox_inches="tight"`, close fig.
- Log with `logger.info`.

**2. `vis_synthetic_camera_overlays`**

Signature:
```python
def vis_synthetic_camera_overlays(
    midlines_3d: list[dict[int, Midline3D]],
    ground_truths: list[dict[int, Midline3D]],
    models: dict[str, RefractiveProjectionModel],
    output_dir: Path,
    *,
    canvas_size: tuple[int, int] = (720, 1280),
    n_eval: int = 40,
) -> None:
```

- `output_dir.mkdir(parents=True, exist_ok=True)`.
- Use frame 0 (or last non-empty frame) for both GT and reconstructed.
- For each camera in `sorted(models)`:
  - Create a gray canvas: `np.full((canvas_size[0], canvas_size[1], 3), 64, dtype=np.uint8)`.
  - For each fish_id in GT frame:
    - Get BGR color from FISH_COLORS.
    - Evaluate GT B-spline at `n_eval` points, project via `model.project(torch.from_numpy(...))`, draw as green dashed polyline (`color=(0, 200, 0)`, draw with `cv2.polylines` using a dotted pattern — actually since cv2 polylines don't support dash, draw every other segment or use small gaps: iterate pairs and draw alternate segments).
    - If fish_id in reconstructed frame: evaluate recon B-spline, project, draw as solid colored polyline using the fish's FISH_COLORS color with `thickness=2`.
    - Compute per-fish reprojection residual: mean pixel distance between GT projected points and recon projected points. Annotate with `_annotate_label`.
  - Add camera label at top-left, legend text "Dashed=GT, Solid=Reconstructed".
  - `cv2.imwrite(str(output_dir / f"synthetic_overlay_{cam_id}.png"), canvas)`.
- Log count of saved overlays.

**3. `vis_synthetic_error_distribution`**

Signature:
```python
def vis_synthetic_error_distribution(
    midlines_3d: list[dict[int, Midline3D]],
    ground_truths: list[dict[int, Midline3D]],
    output_path: Path,
) -> None:
```

- Collect per-control-point 3D errors: for each frame, each fish_id present in both, compute `np.linalg.norm(recon.control_points - gt.control_points, axis=1)` (shape (7,) in metres, multiply by 1000 for mm). Also store fish_id and control-point index (0-6).
- Create `fig, axes = plt.subplots(1, 3, figsize=(18, 5))`.
- Panel (a) `axes[0]`: histogram of all per-control-point errors in mm. `bins=30, edgecolor="black", alpha=0.7`. Add vertical lines for mean and median. Title "Control-Point Error Distribution". xlabel "Error (mm)", ylabel "Count".
- Panel (b) `axes[1]`: box plot grouped by fish_id. Use `axes[1].boxplot(...)` with data grouped per fish. Title "Per-Fish Error Distribution". xlabel "Fish ID", ylabel "Error (mm)".
- Panel (c) `axes[2]`: scatter of error vs control-point index (0=head, 6=tail). Color by fish_id using FISH_COLORS RGB. Title "Error vs Body Position". xlabel "Control Point Index (head->tail)", ylabel "Error (mm)".
- `plt.tight_layout()`, save `dpi=150, bbox_inches="tight"`, close fig, log.

**4. `write_synthetic_report`**

Signature:
```python
def write_synthetic_report(
    output_path: Path,
    stage_timing: dict[str, float],
    midlines_3d: list[dict[int, Midline3D]],
    ground_truths: list[dict[int, Midline3D]],
    models: dict[str, RefractiveProjectionModel],
    fish_configs: list[FishConfig],
    method: str,
    diag_dir: Path,
) -> None:
```

Build a markdown report with these sections:
- **Header**: "# Synthetic Diagnostic Report", date (UTC), method name.
- **Config Summary**: table with n_fish, n_cameras, n_frames, method. Then a sub-table of fish_configs listing position, heading_rad, curvature, scale per fish.
- **Per-Fish GT Comparison**: table columns: Fish ID, Mean Error (mm), Max Error (mm), Std Error (mm), Arc Length Error (mm), Mean Residual (px). Compute from control-point distances across all frames. Arc length error = `abs(recon.arc_length - gt.arc_length) * 1000`.
- **Per-Camera Mean Reprojection Residual**: table from `per_camera_residuals` on reconstructed Midline3D. Columns: Camera ID, Mean Residual (px). Skip if no per_camera_residuals data.
- **Error Statistics**: percentiles (p5, p25, p50, p75, p95, max) of all per-control-point errors in mm.
- **Stage Timing**: reuse the format from `print_timing` but in markdown table form.
- **Diagnostic Files**: list all .png/.mp4/.gif/.md files in `diag_dir`.
- Write with `output_path.write_text(...)`, log.

**Update `__init__.py`**: Add the 4 new functions to both the import block and `__all__` list. Keep alphabetical order within `__all__`.
  </action>
  <verify>
Run `hatch run check` (lint + typecheck). All 4 new functions should be importable:
```
python -c "from aquapose.visualization.diagnostics import vis_synthetic_3d_comparison, vis_synthetic_camera_overlays, vis_synthetic_error_distribution, write_synthetic_report; print('OK')"
```
  </verify>
  <done>4 new public functions exist in diagnostics.py, exported via __init__.py, pass lint and typecheck.</done>
</task>

<task type="auto">
  <name>Task 2: Wire synthetic visualizations into _run_synthetic()</name>
  <files>scripts/diagnose_pipeline.py</files>
  <action>
In `scripts/diagnose_pipeline.py`, modify `_run_synthetic()`:

1. Update the import block (around line 275) to also import the 4 new functions:
```python
from aquapose.visualization.diagnostics import (
    vis_arclength_histogram,
    vis_residual_heatmap,
    vis_synthetic_3d_comparison,
    vis_synthetic_camera_overlays,
    vis_synthetic_error_distribution,
    write_synthetic_report,
)
```

2. Extend the `vis_funcs_syn` list (around line 412-429) to include the 4 new visualizations. Add them AFTER the existing 3 entries but BEFORE the closing `]`:

```python
(
    "synthetic_3d_comparison.png",
    lambda: vis_synthetic_3d_comparison(
        midlines_3d, ground_truths, diag_dir / "synthetic_3d_comparison.png"
    ),
),
(
    "synthetic_camera_overlays/",
    lambda: vis_synthetic_camera_overlays(
        midlines_3d, ground_truths, models, diag_dir / "synthetic_camera_overlays"
    ),
),
(
    "synthetic_error_distribution.png",
    lambda: vis_synthetic_error_distribution(
        midlines_3d, ground_truths, diag_dir / "synthetic_error_distribution.png"
    ),
),
```

3. After the vis_funcs_syn loop (after the existing GT comparison print, around line 442), add the synthetic report generation in its own try/except block (matching the pattern from `_run_real`):

```python
print("  Generating synthetic_report.md...")
try:
    write_synthetic_report(
        output_path=diag_dir / "synthetic_report.md",
        stage_timing=stage_timing,
        midlines_3d=midlines_3d,
        ground_truths=ground_truths,
        models=models,
        fish_configs=fish_configs,
        method=args.method,
        diag_dir=diag_dir,
    )
except Exception as exc:
    print(f"  [WARN] Failed to generate synthetic_report.md: {exc}")
    logger.exception("Failed to generate synthetic_report.md")
```
  </action>
  <verify>
Run `hatch run check` to verify lint+typecheck pass. Then do a dry-run:
```
python scripts/diagnose_pipeline.py --synthetic --n-fish 2 --stop-frame 3 --output-dir output/test_synth_diag
```
Verify output directory contains:
- `diagnostics/synthetic_3d_comparison.png`
- `diagnostics/synthetic_camera_overlays/` with per-camera PNGs
- `diagnostics/synthetic_error_distribution.png`
- `diagnostics/synthetic_report.md`
  </verify>
  <done>All 4 new synthetic visualizations are generated when running `--synthetic` mode. The existing 3 visualizations continue to work. The synthetic_report.md contains per-fish error tables, per-camera residuals, error statistics, timing, and diagnostic file listing.</done>
</task>

</tasks>

<verification>
1. `hatch run check` passes (lint + typecheck)
2. `python scripts/diagnose_pipeline.py --synthetic --n-fish 2 --stop-frame 3` runs without errors
3. Output diagnostics/ directory contains all 7 visualization outputs (3 existing + 4 new)
4. synthetic_report.md contains meaningful numeric data (not empty tables)
</verification>

<success_criteria>
- 4 new visualization functions added to diagnostics.py with proper type hints and docstrings
- Functions exported in __init__.py and __all__
- _run_synthetic() calls all 4 new functions with correct arguments
- Each visualization handles edge cases (empty data, missing fish IDs) gracefully
- Lint and typecheck pass
</success_criteria>

<output>
After completion, create `.planning/quick/3-add-synthetic-mode-diagnostic-visualizat/3-SUMMARY.md`
</output>
