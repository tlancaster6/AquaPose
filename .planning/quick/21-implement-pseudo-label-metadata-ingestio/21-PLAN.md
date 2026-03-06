---
phase: 21-implement-pseudo-label-metadata-ingestion
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/training/pseudo_labels.py
  - src/aquapose/training/pseudo_label_cli.py
  - src/aquapose/training/data_cli.py
  - src/aquapose/training/run_manager.py
  - src/aquapose/training/__init__.py
  - src/aquapose/training/frame_selection.py
  - tests/unit/training/test_frame_selection.py
autonomous: true
requirements: [QUICK-21]
must_haves:
  truths:
    - "compute_curvature lives in pseudo_labels.py and is exported from training/__init__.py"
    - "frame_selection.py and its test file are deleted; no imports reference them"
    - "data import auto-reads confidence.json sidecar and merges per-image metadata"
    - "data import computes 2D curvature from pose label keypoints for pose store"
    - "pseudo-label generate writes 3D curvature into confidence.json entries"
    - "run_manager next-steps text uses --project instead of --config"
  artifacts:
    - path: "src/aquapose/training/pseudo_labels.py"
      provides: "compute_curvature function"
      contains: "def compute_curvature"
    - path: "src/aquapose/training/data_cli.py"
      provides: "sidecar ingestion and curvature computation at import"
      contains: "confidence.json"
  key_links:
    - from: "src/aquapose/training/__init__.py"
      to: "src/aquapose/training/pseudo_labels.py"
      via: "import compute_curvature"
      pattern: "from .pseudo_labels import.*compute_curvature"
---

<objective>
Implement pseudo-label metadata ingestion improvements: move compute_curvature to pseudo_labels.py, delete dead frame_selection.py, auto-read confidence.json sidecar at import, compute 2D curvature at import time, add 3D curvature to generate sidecar, and fix stale --config references in run_manager.py.

Purpose: Close the metadata gap between pseudo-label generation and store import so that curvature and confidence metadata flow automatically into the training data store without manual --metadata-json flags.
Output: Updated CLI commands with automatic sidecar ingestion and curvature computation.
</objective>

<execution_context>
@/home/tlancaster6/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlancaster6/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/training/pseudo_labels.py
@src/aquapose/training/pseudo_label_cli.py
@src/aquapose/training/data_cli.py
@src/aquapose/training/frame_selection.py
@src/aquapose/training/__init__.py
@src/aquapose/training/run_manager.py
@tests/unit/training/test_frame_selection.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Move compute_curvature, delete frame_selection.py, fix run_manager</name>
  <files>
    src/aquapose/training/pseudo_labels.py
    src/aquapose/training/frame_selection.py
    tests/unit/training/test_frame_selection.py
    src/aquapose/training/__init__.py
    src/aquapose/training/run_manager.py
  </files>
  <action>
1. Copy `compute_curvature` function from `frame_selection.py` to `pseudo_labels.py`. Place it after the existing imports. Update the docstring to note it works on both 2D and 3D points (finite differences of tangent vectors, dimension-agnostic). The function signature stays the same: `control_points: np.ndarray` -> `float`. The only change is the docstring shape hint: remove the "(7, 3)" specificity and say "(N, D)" since it will now also be used on 2D keypoints.

2. Delete `src/aquapose/training/frame_selection.py` entirely.

3. Delete `tests/unit/training/test_frame_selection.py` entirely.

4. Update `src/aquapose/training/__init__.py`:
   - Remove the entire `from .frame_selection import (...)` block (lines 40-46).
   - Add `compute_curvature` to the existing `from .pseudo_labels import (...)` block.
   - Remove from `__all__`: `DiversitySampleResult`, `diversity_sample`, `filter_empty_frames`, `temporal_subsample`.
   - Keep `compute_curvature` in `__all__` (it's still exported, just from a different module).

5. Fix `src/aquapose/training/run_manager.py` lines 331-339: Replace `--config {config_path}` with `--project {config_path.parent}` in the compare command (line 333), and replace `--config ...` with `--project ...` in the retrain command (line 338). The config_path variable computes `run_dir.parent.parent.parent / "config.yaml"` — the project dir is `run_dir.parent.parent.parent` (i.e., `config_path.parent`). Change the variable name from `config_path` to `project_dir` and set it to `run_dir.parent.parent.parent`.
  </action>
  <verify>
    <automated>cd /home/tlancaster6/Projects/AquaPose && hatch run test</automated>
  </verify>
  <done>
    - compute_curvature exists in pseudo_labels.py and is exported from training/__init__.py
    - frame_selection.py and test_frame_selection.py are deleted
    - No remaining imports of frame_selection anywhere in the codebase
    - run_manager.py uses --project instead of --config in next-steps output
    - All tests pass
  </done>
</task>

<task type="auto">
  <name>Task 2: Auto-read confidence.json sidecar and compute 2D curvature at import</name>
  <files>
    src/aquapose/training/data_cli.py
    src/aquapose/training/pseudo_label_cli.py
  </files>
  <action>
1. In `data_cli.py:import_cmd`, after resolving `input_path` and parsing `metadata` (around line 90), add sidecar auto-detection:

```python
# Auto-detect confidence.json sidecar in input directory
sidecar_data: dict = {}
sidecar_path = input_path / "confidence.json"
if sidecar_path.exists():
    sidecar_data = json.loads(sidecar_path.read_text())
    click.echo(f"Found confidence.json sidecar ({len(sidecar_data)} entries)")
```

2. Inside the `for img_path in image_files:` loop, before calling `sample_store.import_sample(...)`, build per-sample metadata by merging three sources (in priority order: --metadata-json overrides sidecar):

```python
# Build per-sample metadata: sidecar + static metadata overlay
sample_meta: dict = {}

# Sidecar metadata (keyed by image stem)
sidecar_entry = sidecar_data.get(img_path.stem, {})
if sidecar_entry:
    # Flatten sidecar: extract confidence, gap_reason, n_source_cameras, source from labels[0]
    labels_list = sidecar_entry.get("labels", [])
    if labels_list:
        first_label = labels_list[0]
        for key in ("confidence", "gap_reason", "n_source_cameras", "raw_metrics", "source"):
            if key in first_label:
                sample_meta[key] = first_label[key]

# 2D curvature computation (pose store only)
if store == "pose" and label_path.exists():
    label_text = label_path.read_text().strip()
    if label_text:
        from .pseudo_labels import compute_curvature
        # Parse first pose label line to extract keypoints
        first_line = label_text.split("\n")[0].split()
        # YOLO pose format: cls cx cy w h x1 y1 v1 x2 y2 v2 ...
        kp_tokens = first_line[5:]
        if len(kp_tokens) >= 6:  # at least 2 keypoints
            pts = []
            for ki in range(0, len(kp_tokens), 3):
                if ki + 2 < len(kp_tokens):
                    vis = float(kp_tokens[ki + 2])
                    if vis > 0:
                        pts.append([float(kp_tokens[ki]), float(kp_tokens[ki + 1])])
            if len(pts) >= 3:  # need at least 3 points for curvature
                import numpy as np
                sample_meta["curvature"] = compute_curvature(np.array(pts))

# Static metadata from --metadata-json overrides sidecar
if metadata:
    sample_meta.update(metadata)
```

Then pass `sample_meta if sample_meta else metadata` as the `metadata` argument to `sample_store.import_sample(...)`. Actually simpler: just pass `sample_meta or None`.

3. In `pseudo_label_cli.py:generate`, add 3D curvature to the confidence.json entries. In the consensus loop (around line 353-360), after building the conf_entry dict, add:
```python
from aquapose.training.pseudo_labels import compute_curvature
# ... (import at top of function, not in loop)
```
Actually, add the import near the top of the `generate` function body (alongside other imports). Then in the consensus conf_entry dict (line 354-360), add `"curvature_3d": compute_curvature(midline.control_points)`. Similarly in the gap conf_entry dict (line 425-432), add `"curvature_3d": compute_curvature(midline.control_points)`.

The import of compute_curvature should be placed alongside the existing imports from pseudo_labels at the top of the file (line 21-25), adding it to the existing import block:
```python
from aquapose.training.pseudo_labels import (
    compute_curvature,
    detect_gaps,
    generate_fish_labels,
    generate_gap_fish_labels,
)
```
  </action>
  <verify>
    <automated>cd /home/tlancaster6/Projects/AquaPose && hatch run test</automated>
  </verify>
  <done>
    - data import auto-detects confidence.json in input directory and merges per-image sidecar metadata into sample metadata
    - data import computes 2D curvature from visible keypoints for pose store samples and stores it as "curvature" in metadata
    - --metadata-json still works as override on top of sidecar data
    - pseudo-label generate writes curvature_3d field into confidence.json entries for both consensus and gap labels
    - All tests pass
  </done>
</task>

</tasks>

<verification>
- `hatch run test` passes
- `hatch run check` passes (lint + typecheck)
- `grep -r "frame_selection" src/ tests/` returns no results
- `grep -rn "compute_curvature" src/aquapose/training/pseudo_labels.py` shows the function
- `grep -rn "confidence.json" src/aquapose/training/data_cli.py` shows sidecar detection
- `grep -rn "curvature_3d" src/aquapose/training/pseudo_label_cli.py` shows 3D curvature in sidecar
- `grep -rn "\-\-project" src/aquapose/training/run_manager.py` shows fixed references
</verification>

<success_criteria>
- compute_curvature relocated from frame_selection.py to pseudo_labels.py
- frame_selection.py and its test deleted with no remaining references
- data import automatically reads confidence.json sidecar and computes 2D curvature
- pseudo-label generate includes 3D curvature in confidence.json
- run_manager uses --project instead of --config
- All tests and checks pass
</success_criteria>

<output>
After completion, create `.planning/quick/21-implement-pseudo-label-metadata-ingestio/21-SUMMARY.md`
</output>
