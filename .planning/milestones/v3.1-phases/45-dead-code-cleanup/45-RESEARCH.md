# Phase 45: Dead Code Cleanup - Research

**Researched:** 2026-03-03
**Domain:** Python codebase dead code removal — file deletion, import fixup, config update
**Confidence:** HIGH

## Summary

Phase 45 is a mechanical cleanup phase with no new functionality. The codebase accumulated two reconstruction backends (`triangulation` and `curve_optimizer`) that have been superseded by the `dlt` backend (Phase 43). The task is to delete those backends and their supporting modules, migrate surviving shared symbols to `reconstruction/utils.py`, update all import sites, simplify `ReconstructionConfig`, and update project config files.

The cleanup is well-scoped: exact file candidates are known, all import sites have been identified, and the surviving shared utilities already live in `reconstruction/utils.py`. The main risk is test files — several test files directly import from modules being deleted (`test_triangulation.py`, `test_curve_optimizer.py`, `test_reconstruction_stage.py`). These tests cover dead code and must be deleted or rewritten to cover surviving code paths.

A secondary investigation is required for `synthetic/fish.py` and `io/midline_writer.py` — both import from `reconstruction/triangulation.py` but may themselves be live or dead code.

**Primary recommendation:** Delete dead modules in dependency order (backends first, then top-level modules), migrate the one surviving constant (`DEFAULT_INLIER_THRESHOLD`) to `utils.py`, update all import sites atomically, then delete or rewrite tests that reference deleted modules.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Shared Utility Handling
- Surviving helpers/constants from `reconstruction/triangulation.py` (e.g. `DEFAULT_INLIER_THRESHOLD`, refraction utilities) move to `reconstruction/utils.py`
- All consumers (`dlt.py`, `stage.py`, `synthetic/fish.py`, `io/midline_writer.py`) update imports to point at `utils.py`
- Researcher should investigate whether `synthetic/fish.py` and `io/midline_writer.py` are themselves dead code — if so, delete them too
- The evaluation harness should use the backend registry (`get_backend('dlt')`) rather than direct `DltBackend` import

#### Config Cleanup
- Default reconstruction backend changes from `'triangulation'` to `'dlt'`
- Remove `'triangulation'` and `'curve_optimizer'` from the backend registry with a standard `ValueError` (no migration hints or deprecation warnings)
- Remove config fields that only the old backends used (e.g. `inlier_threshold`, `snap_threshold`, `max_depth`) — keep only what DLT needs
- Update `~/aquapose/projects/YH/config.yaml` to use `'dlt'` and remove any old-backend-specific fields

#### Backend Naming
- Keep `'dlt'` as the backend kind name — it's accurate and specific
- Do not rename to a generic `'triangulation'`

#### Dead Code Sweep Scope
- Opportunistic: delete exactly what CLEAN-01/02/03 specify, plus any additional dead code found in `reconstruction/` during investigation
- Remove visualization code tied to deleted backends (`triangulation_viz.py` — delete if only serves old backends, keep DLT-relevant parts)
- Update `GUIDEBOOK.md` to remove references to old backends (triangulation, curve_optimizer)
- Keep the backend registry pattern (`get_backend()`) with only `'dlt'` registered — preserves extensibility for future backends

### Claude's Discretion
- Exact ordering of file deletions and import fixups
- Whether to consolidate `reconstruction/utils.py` or keep it minimal
- How to handle any edge cases in test files referencing deleted code

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CLEAN-01 | Old triangulation backend code removed | Identified: `backends/triangulation.py` (167 lines), `reconstruction/triangulation.py` (910 lines). All import sites mapped. |
| CLEAN-02 | Curve optimizer backend code removed | Identified: `backends/curve_optimizer.py` (132 lines), `reconstruction/curve_optimizer.py` (1881 lines). One import site (`test_confidence_weighting.py`) references it. |
| CLEAN-03 | Dead code removed (refine_midline_lm stub, unused orientation/epipolar code paths) | `refine_midline_lm` is at line 893 of `triangulation.py` — deleted with the whole file. Orientation/epipolar helpers (`_align_midline_orientations`, `_refine_correspondences_epipolar`) are used only by `triangulate_midlines` in the same file — also deleted with it. |
</phase_requirements>

---

## Standard Stack

No new libraries required. This phase uses only standard Python file operations and git.

### Core Tools
| Tool | Purpose |
|------|---------|
| `git rm` | Delete tracked files from the repo |
| `hatch run test` | Verify tests pass after changes |
| `hatch run check` | Lint + typecheck verification |

---

## Architecture Patterns

### Backend Registry Pattern (SURVIVE)
The `backends/__init__.py` registry (`get_backend()`) survives with only `'dlt'` registered. Unknown kinds raise `ValueError`. This preserves extensibility for future backends without keeping dead code.

**Current registry (`backends/__init__.py` lines 42-64):**
```python
if kind == "triangulation":   # DELETE these two branches
    ...
if kind == "curve_optimizer": # DELETE this branch
    ...
if kind == "dlt":             # KEEP
    ...
raise ValueError(...)         # Update error message to list only ['dlt']
```

**Post-cleanup:**
```python
def get_backend(kind: str, **kwargs: Any) -> object:
    if kind == "dlt":
        from aquapose.core.reconstruction.backends.dlt import DltBackend
        return DltBackend(**kwargs)
    raise ValueError(
        f"Unknown reconstruction backend kind: {kind!r}. "
        f"Supported kinds: ['dlt']"
    )
```

### `reconstruction/utils.py` — Surviving Shared Symbols
`utils.py` already exists and contains the shared helpers. The old `triangulation.py` re-exports `MIN_BODY_POINTS` from `utils.py` via `noqa: F401`. After deletion of `triangulation.py`, consumers that imported from it must update to import from `utils.py` directly.

**Symbols that survive in `utils.py` (already there):**
- `MIN_BODY_POINTS`
- `SPLINE_K`
- `build_spline_knots`
- `fit_spline`
- `pixel_half_width_to_metres`
- `weighted_triangulate_rays`

**Symbol that needs to MOVE to `utils.py`:**
- `DEFAULT_INLIER_THRESHOLD` — currently defined in `triangulation.py` (line 40 = 50.0), imported by `stage.py` and `backends/triangulation.py`. The DLT backend uses its own `DEFAULT_OUTLIER_THRESHOLD` which is different. Decide: if `stage.py` only needs a default for the old `triangulation` backend init path, that import can be deleted entirely when the stage's `__init__` is simplified. **Investigate at implementation time.**

**Symbols that are dead (only used by deleted code):**
- `SPLINE_N_CTRL` — defined in `triangulation.py`, only used there and in `backends/triangulation.py`
- `SPLINE_KNOTS` — defined in `triangulation.py`, used there, in `backends/triangulation.py` (via `SPLINE_N_CTRL`), in `synthetic/fish.py`, and in `io/midline_writer.py`
- `N_SAMPLE_POINTS` — defined in `triangulation.py` (= 15), used in `synthetic/fish.py` and `io/midline_writer.py`

---

## Files to Delete (Complete Map)

### Confirmed Deletions

| File | Lines | Why |
|------|-------|-----|
| `src/aquapose/core/reconstruction/backends/triangulation.py` | 167 | Old TriangulationBackend (CLEAN-01) |
| `src/aquapose/core/reconstruction/backends/curve_optimizer.py` | 132 | Old CurveOptimizerBackend (CLEAN-02) |
| `src/aquapose/core/reconstruction/triangulation.py` | 910 | Core triangulation logic, refine_midline_lm stub, orientation/epipolar code (CLEAN-01, CLEAN-03) |
| `src/aquapose/core/reconstruction/curve_optimizer.py` | 1881 | Core curve optimizer logic (CLEAN-02) |

### Visualization File Decision
`src/aquapose/visualization/triangulation_viz.py` — **investigation required at implementation time:**
- Has a TYPE_CHECKING import from `core.reconstruction.curve_optimizer` (`OptimizerSnapshot`)
- Functions `vis_optimizer_progression` and `write_synthetic_report` reference curve optimizer-specific types
- Functions `vis_residual_heatmap`, `vis_arclength_histogram`, `vis_per_camera_spline_overlays`, `vis_synthetic_3d_comparison`, `vis_synthetic_camera_overlays`, `vis_synthetic_error_distribution`, `write_diagnostic_report` are general enough to apply to DLT results
- Decision per CONTEXT.md: "delete if only serves old backends, keep DLT-relevant parts"
- **Likely outcome**: Delete the file entirely (no live callers found in production pipeline) OR remove only the optimizer-specific function `vis_optimizer_progression`

### Conditional Deletions (investigate at implementation time)

**`src/aquapose/synthetic/fish.py`:**
- Imports: `N_SAMPLE_POINTS`, `SPLINE_K`, `SPLINE_KNOTS` from `reconstruction/triangulation.py`
- Provides: `FishConfig`, `generate_fish_3d`, `generate_synthetic_midline_sets`, etc.
- Used by: `tests/unit/synthetic/test_synthetic.py` (confirmed via grep), `synthetic/scenarios.py`, evaluation harness tests
- **Verdict: LIVE CODE** — used by tests and evaluation. Must update imports to point to `utils.py` for `SPLINE_K` and create local constants for `SPLINE_KNOTS`/`N_SAMPLE_POINTS`.

**`src/aquapose/io/midline_writer.py`:**
- Imports: `N_SAMPLE_POINTS`, `SPLINE_K`, `SPLINE_KNOTS` from `reconstruction/triangulation.py`
- Provides: `Midline3DWriter`, `read_midline3d_results`
- Used by: `tests/unit/io/test_midline_writer.py` — **imports from `reconstruction/triangulation.py`** (line 11)
- `Midline3DWriter` is a HDF5 writer used in the pipeline
- **Verdict: LIVE CODE** — must update imports to get constants from `utils.py` or define them locally.

---

## All Import Sites That Must Change

### `src/aquapose/core/reconstruction/stage.py`
Line 25: `from aquapose.core.reconstruction.triangulation import DEFAULT_INLIER_THRESHOLD`
- Action: Remove this import. The stage's `__init__` has a branch `if backend == "triangulation"` that passes `inlier_threshold` — that branch is deleted. The default value constant is no longer needed in stage.py.
- Also update docstring/comment referencing "triangulation" as default backend.

### `src/aquapose/engine/config.py`
`ReconstructionConfig`:
- Change `backend: str = "triangulation"` → `backend: str = "dlt"`
- Remove field `inlier_threshold: float = 50.0`
- Remove field `snap_threshold: float = 20.0`
- Remove field `max_depth: float | None = None`
- Keep: `outlier_threshold`, `min_cameras`, `max_interp_gap`, `n_control_points`
- Update docstring to remove references to old backends

### `src/aquapose/evaluation/harness.py`
Lines 19-20: Direct imports of `TriangulationBackend` and `DltBackend`
- Remove: `from aquapose.core.reconstruction.backends.triangulation import TriangulationBackend`
- Remove: `from aquapose.core.reconstruction.backends.dlt import DltBackend`
- Change `run_evaluation` to use `get_backend('dlt', models=models, ...)` via registry
- Update `backend` parameter: only `'dlt'` is valid; remove `'triangulation'` branch
- Update `ValueError` message to list only `['dlt']`

### `src/aquapose/synthetic/fish.py`
Lines 19-22: Imports from `reconstruction/triangulation.py`
```python
from aquapose.core.reconstruction.triangulation import (
    N_SAMPLE_POINTS,
    SPLINE_K,
    SPLINE_KNOTS,
)
```
- `SPLINE_K` → import from `reconstruction/utils.py`
- `SPLINE_KNOTS` → define locally in `synthetic/fish.py` (it's a fixed constant `np.array([0,0,0,0,0.25,0.5,0.75,1,1,1,1])`) or move to `utils.py`
- `N_SAMPLE_POINTS` → define locally (= 15) or move to `utils.py`

### `src/aquapose/io/midline_writer.py`
Lines 15-18: Imports from `reconstruction/triangulation.py`
```python
from aquapose.core.reconstruction.triangulation import (
    N_SAMPLE_POINTS,
    SPLINE_K,
    SPLINE_KNOTS,
)
```
- Same as `synthetic/fish.py` — get `SPLINE_K` from `utils.py`, define `N_SAMPLE_POINTS` and `SPLINE_KNOTS` locally or move to `utils.py`

### `src/aquapose/visualization/triangulation_viz.py` (if kept)
Line 24 (TYPE_CHECKING): `from aquapose.core.reconstruction.curve_optimizer import OptimizerSnapshot`
- Remove this import and any function using `OptimizerSnapshot` (e.g. `vis_optimizer_progression`)

### `src/aquapose/core/reconstruction/backends/__init__.py`
- Remove `triangulation` and `curve_optimizer` branches
- Update docstring and error message

---

## Test Files That Must Change

### Delete (test dead code)
| File | Why |
|------|-----|
| `tests/unit/test_triangulation.py` | Tests `triangulate_midlines`, `refine_midline_lm`, `_align_midline_orientations`, `_refine_correspondences_epipolar` — all in deleted `triangulation.py` |
| `tests/unit/test_curve_optimizer.py` | Tests `CurveOptimizer` — in deleted `curve_optimizer.py` |

### Rewrite / Trim (test live code that references dead imports)
| File | Change Required |
|------|----------------|
| `tests/unit/core/reconstruction/test_reconstruction_stage.py` | Remove `test_triangulation_backend_delegates`, `test_curve_optimizer_backend_delegates`, `test_backend_selection_triangulation`, `test_backend_selection_curve_optimizer` (lines ~463-561). Update `test_import_boundary` to remove module references to deleted backends. |
| `tests/unit/evaluation/test_harness.py` | Patches reference `TriangulationBackend.from_models` (lines 143-195) — update mocks to match new harness code using `get_backend` |
| `tests/unit/engine/test_config.py` | Line 46: `assert config.reconstruction.backend == "triangulation"` → `"dlt"`. Line 220: `assert parsed["reconstruction"]["backend"] == "triangulation"` → `"dlt"`. Also update tests for removed config fields (`inlier_threshold`, `snap_threshold`, `max_depth`) if any exist. |
| `tests/unit/io/test_midline_writer.py` | Line 11: `from aquapose.core.reconstruction.triangulation import (...)` — update to new import source |
| `tests/unit/core/reconstruction/test_confidence_weighting.py` | Lines 24-38: imports from `curve_optimizer` and `triangulation` — remove curve_optimizer imports, update triangulation imports to `utils.py` |

### Unaffected (golden/regression tests reference "triangulation" as a variable name for the 3D reconstruction output, not the backend)
- `tests/golden/` — references to "golden_triangulation" are test fixture names, not backend kinds
- `tests/regression/` — same

---

## Config File Changes

### `src/aquapose/engine/config.py` — `ReconstructionConfig`
Remove fields: `inlier_threshold`, `snap_threshold`, `max_depth`
Change default: `backend = "dlt"` (was `"triangulation"`)

**Post-cleanup `ReconstructionConfig`:**
```python
@dataclass(frozen=True)
class ReconstructionConfig:
    backend: str = "dlt"
    outlier_threshold: float = 10.0
    min_cameras: int = 3
    max_interp_gap: int = 5
    n_control_points: int = 7
```

### `~/aquapose/projects/YH/config.yaml`
Per MEMORY.md, current YH config likely has `backend: triangulation` or none specified. Update to `backend: dlt`. Remove `inlier_threshold`, `snap_threshold`, `max_depth` fields if present.

---

## `reconstruction/stage.py` Cleanup

The `ReconstructionStage.__init__` currently has a special branch for `backend == "triangulation"` that adds `inlier_threshold`, `snap_threshold`, `max_depth` to `combined_kwargs`. After cleanup, this branching is removed — the stage just passes `calibration_path` and DLT-specific kwargs directly.

**Post-cleanup `__init__`:**
```python
def __init__(
    self,
    calibration_path: str | Path,
    backend: str = "dlt",
    outlier_threshold: float = ...,  # from ReconstructionConfig default
    min_cameras: int = _DEFAULT_MIN_CAMERAS,
    max_interp_gap: int = _DEFAULT_MAX_INTERP_GAP,
    n_control_points: int = _DEFAULT_N_CONTROL_POINTS,
) -> None:
    ...
    combined_kwargs = {
        "calibration_path": calibration_path,
        "outlier_threshold": outlier_threshold,
        "n_control_points": n_control_points,
    }
    self._backend = get_backend(backend, **combined_kwargs)
```

---

## GUIDEBOOK.md Changes

Per CONTEXT.md, update `.planning/GUIDEBOOK.md` to remove references to old backends. Specific grep targets:
- Line 50: `reconstruction/ ... backends/ (triangulation.py, curve_optimizer.py)` → update to just `(dlt.py)`
- Line 71: `visualization/ ... triangulation_viz` — remove if file is deleted
- Line 171: `*Swappable backend: triangulation / curve optimizer*` section — update to just `dlt`
- Lines 176, 179: References to orientation/curve optimizer behavior

---

## Common Pitfalls

### Pitfall 1: `DEFAULT_INLIER_THRESHOLD` vs `DEFAULT_OUTLIER_THRESHOLD`
**What goes wrong:** `stage.py` imports `DEFAULT_INLIER_THRESHOLD` from `triangulation.py`. If you simply delete the import without checking whether the constant is still used, `stage.py` will fail to import.
**How to avoid:** The `stage.py` `__init__` no longer needs this constant after removing the `backend == "triangulation"` branch. Verify the import is removed entirely.

### Pitfall 2: `N_SAMPLE_POINTS` and `SPLINE_KNOTS` have multiple consumers
**What goes wrong:** `synthetic/fish.py` and `io/midline_writer.py` both import `N_SAMPLE_POINTS` (= 15), `SPLINE_K`, and `SPLINE_KNOTS` from `triangulation.py`. If you delete `triangulation.py` without updating these files, they will fail at import time.
**How to avoid:** Update both files before or atomically with deleting `triangulation.py`. Options:
- Move these constants to `utils.py` (cleaner, single source)
- Define them locally in each consumer (simpler, avoids `utils.py` scope creep)
Recommendation: move `N_SAMPLE_POINTS` and `SPLINE_KNOTS` to `utils.py` since they're inherently part of the reconstruction math infrastructure.

### Pitfall 3: Test imports from deleted modules will fail before test changes
**What goes wrong:** If `triangulation.py` is deleted before test files are updated, the test suite will fail to import and all tests in those files will error (not just the dead-code tests).
**How to avoid:** Update or delete test files in the same commit/wave as the module deletion, not after.

### Pitfall 4: `test_import_boundary` test lists deleted modules
**What goes wrong:** `test_reconstruction_stage.py` line 583-587 tests import boundaries for `backends.triangulation` and `backends.curve_optimizer` — if those modules are deleted, the test itself will fail on import.
**How to avoid:** Remove those module references from the test's list.

### Pitfall 5: `ReconstructionConfig` field removal breaks existing config YAML files
**What goes wrong:** If a YAML config file (like `YH/config.yaml`) has `inlier_threshold` or `snap_threshold` under `reconstruction:`, `_filter_fields()` will raise `ValueError` with "unknown field" — breaking the pipeline.
**How to avoid:** Update `~/aquapose/projects/YH/config.yaml` as part of this phase. Also check if `_filter_fields()` needs updating for any migration-hint entries in `_RENAME_HINTS`.

### Pitfall 6: `harness.py` test patches reference deleted backend
**What goes wrong:** `test_harness.py` patches `aquapose.evaluation.harness.TriangulationBackend.from_models` — after the harness is updated to use `get_backend`, these patches will fail with `AttributeError`.
**How to avoid:** Update test patches to match the new `get_backend` call in `harness.py`.

---

## Code Examples

### Updated `backends/__init__.py`
```python
"""Backend registry for the Reconstruction stage.

Provides a factory function that resolves reconstruction backend kind strings
to configured backend instances. Supports only "dlt".
"""

from __future__ import annotations
from typing import Any

__all__ = ["get_backend"]


def get_backend(kind: str, **kwargs: Any) -> object:
    """Create a reconstruction backend by kind name.

    Args:
        kind: Backend identifier. Only ``"dlt"`` is supported.
        **kwargs: Forwarded to the backend constructor. Accepted kwargs:
            ``calibration_path``, ``outlier_threshold``,
            ``n_control_points``, ``low_confidence_fraction``.

    Returns:
        A configured backend instance with a
        ``reconstruct_frame(frame_idx, midline_set)`` method.

    Raises:
        ValueError: If *kind* is not a recognized backend identifier.
    """
    if kind == "dlt":
        from aquapose.core.reconstruction.backends.dlt import DltBackend
        return DltBackend(**kwargs)

    raise ValueError(
        f"Unknown reconstruction backend kind: {kind!r}. "
        f"Supported kinds: ['dlt']"
    )
```

### Updated `harness.py` backend selection
```python
# Replace direct backend imports with registry
from aquapose.core.reconstruction.backends import get_backend

# In run_evaluation:
if backend == "dlt":
    if outlier_threshold is not None:
        recon_backend = get_backend("dlt", models=models, outlier_threshold=outlier_threshold)
    else:
        recon_backend = get_backend("dlt", models=models)
else:
    raise ValueError(
        f"Unknown evaluation backend: {backend!r}. "
        f"Supported backends: ['dlt']"
    )
```

---

## Open Questions

1. **`N_SAMPLE_POINTS` and `SPLINE_KNOTS` placement**
   - What we know: Both `synthetic/fish.py` and `io/midline_writer.py` need them; currently sourced from `triangulation.py`
   - What's unclear: Whether they belong in `utils.py` or should be defined locally
   - Recommendation: Move both to `utils.py` — they are reconstruction math constants and the file is the natural home for shared reconstruction utilities. Add to `__all__` in `utils.py`.

2. **`triangulation_viz.py` disposition**
   - What we know: Contains both DLT-relevant and optimizer-specific functions; has a TYPE_CHECKING import from `curve_optimizer.py`
   - What's unclear: Whether any live pipeline code calls the DLT-relevant functions
   - Recommendation: Delete the entire file — no live callers were found in the production pipeline. The functions are diagnostic/research utilities. If needed in future, they can be reintroduced without the curve_optimizer dependency.

3. **`stage.py` `_run_legacy` path**
   - What we know: `_run_legacy` is the fallback path that runs when `tracklet_groups` is None and `annotated_detections` is present
   - What's unclear: Whether this path is still needed after cleanup
   - Recommendation: Keep `_run_legacy` — it is not dead code, it handles a valid pipeline state. It does not depend on the old backends.

---

## Validation Architecture

Nyquist validation is not configured (`workflow.nyquist_validation` absent from `.planning/config.json`). Skipping this section.

Test commands:
- Per-task: `hatch run test` (excludes @slow)
- Full phase gate: `hatch run check` (lint + typecheck) then `hatch run test`

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection — all findings verified by reading actual source files
- Files read: `reconstruction/triangulation.py`, `reconstruction/curve_optimizer.py`, `reconstruction/backends/triangulation.py`, `reconstruction/backends/curve_optimizer.py`, `reconstruction/backends/__init__.py`, `reconstruction/utils.py`, `reconstruction/stage.py`, `engine/config.py`, `evaluation/harness.py`, `synthetic/fish.py`, `io/midline_writer.py`, `visualization/triangulation_viz.py`
- Test files read: `test_triangulation.py`, `test_curve_optimizer.py` (partial), `test_reconstruction_stage.py` (partial), `test_config.py` (partial), `test_harness.py` (via grep)

---

## Metadata

**Confidence breakdown:**
- Files to delete: HIGH — confirmed by reading files and their contents
- Import sites: HIGH — confirmed by grep across full codebase
- Test changes: HIGH — confirmed by reading specific test functions
- `synthetic/fish.py` / `io/midline_writer.py` as live code: HIGH — confirmed by test file presence

**Research date:** 2026-03-03
**Valid until:** Until Phase 44 completes (changes to dlt.py or evaluation harness could affect findings)
