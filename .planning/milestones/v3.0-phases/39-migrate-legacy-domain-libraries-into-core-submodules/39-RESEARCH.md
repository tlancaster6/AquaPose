# Phase 39: Migrate Legacy Domain Libraries into Core Submodules - Research

**Researched:** 2026-03-02
**Domain:** Python package reorganization, import graph surgery, type extraction
**Confidence:** HIGH (pure codebase analysis — no external libraries to verify)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**File placement strategy:**
- Cross-stage types go to a new `core/types/` package (not per-stage `types.py` files)
- `core/types/` is a package with domain-split files: `detection.py`, `crop.py`, `midline.py`, `reconstruction.py`
- Implementation code goes to the consuming stage's submodule
- Types are split from implementations: dataclasses/type aliases → `core/types/`, classes/functions/helpers → stage submodule

**Full placement map:**
- `segmentation/detector.py` → split:
  - `Detection` dataclass → `core/types/detection.py`
  - `YOLODetector` class + `make_detector()` → `core/detection/backends/yolo.py` (already consumed there)
- `segmentation/crop.py` → split:
  - `CropRegion`, `AffineCrop` types → `core/types/crop.py`
  - `extract_affine_crop()`, `invert_affine_point()`, `invert_affine_points()` utilities → `core/midline/crop.py`
  - v2.x functions (`compute_crop_region`, `extract_crop`, `paste_mask`) → drop if unused, otherwise bring along
- `reconstruction/midline.py` → split:
  - `Midline2D` type → `core/types/midline.py`
  - `MidlineExtractor` + private helpers → `core/midline/midline.py`
- `reconstruction/triangulation.py` → split:
  - `Midline3D`, `MidlineSet` types → `core/types/reconstruction.py`
  - `triangulate_midlines()` + helpers + constants → `core/reconstruction/triangulation.py`
- `reconstruction/curve_optimizer.py` → whole file → `core/reconstruction/curve_optimizer.py`
- `tracking/ocsort_wrapper.py` → whole file → `core/tracking/ocsort_wrapper.py`

**Re-export / backwards-compat approach:**
- One-shot rewrite of all import paths across the entire codebase — no shims, no deprecation period
- ALL consumers updated in the same pass: `core/`, `engine/`, `visualization/`, `io/`, `synthetic/`, tests
- `visualization/` is explicitly in scope — this phase is not complete until all existing functionality works
- Old legacy directories (`reconstruction/`, `segmentation/`, `tracking/`) deleted entirely after migration

**Private helper exposure:**
- Private helpers (`_adaptive_smooth`, `_skeleton_and_widths`, `_longest_path_bfs`, `_resample_arc_length`) stay private (underscore-prefixed)
- After moving to `core/midline/midline.py`, the cross-package import smell vanishes since they're now package-internal
- v2.x crop functions (`compute_crop_region`, `extract_crop`, `paste_mask`) — check usage and drop if dead code
- Import direction: `core/types/` is canonical source, implementations import types from there (not reverse)

**Type ownership:**
- Canonical type locations:
  - `Detection` → `core/types/detection.py`
  - `CropRegion`, `AffineCrop` → `core/types/crop.py`
  - `Midline2D` → `core/types/midline.py`
  - `Midline3D`, `MidlineSet` → `core/types/reconstruction.py`
- Stage-specific types stay with implementations:
  - `CurveOptimizerConfig`, `OptimizerSnapshot` → `core/reconstruction/curve_optimizer.py`
- Existing per-stage `types.py` re-export files (e.g., `core/detection/types.py`) are eliminated
- `core/types/__init__.py` re-exports all public types for convenience (`from aquapose.core.types import Detection, Midline2D`)

### Claude's Discretion
- Exact ordering of migration steps (which files move first to minimize breakage)
- Whether `make_detector()` factory merges into existing detection backend code or gets its own file
- How to handle any circular import issues discovered during the split
- GUIDEBOOK.md and docstring updates to reflect new paths

### Deferred Ideas (OUT OF SCOPE)
- visualization/ directory reorganization (moving files closer to engine observers) — separate phase
- Refactoring MidlineExtractor internals or splitting large files — separate concern
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| STAB-04 | All stale docstrings referencing U-Net, no-op stubs, or Phase 37 pending status are updated | After migration, all module-level docstrings in moved files and all consumer files referencing old paths must be updated. GUIDEBOOK.md source layout section requires path updates to remove legacy directories. |
| TBD | Legacy directories reorganized into core submodules (implicit requirement from phase goal) | Full placement map is locked in CONTEXT.md. All 73 import sites updated in one-shot pass. Three legacy top-level packages deleted. |
</phase_requirements>

## Summary

Phase 39 is a pure structural reorganization — no logic changes, only file moves and import rewrites. The three legacy top-level directories (`reconstruction/`, `segmentation/`, `tracking/`) are canonical implementations that have been consumed via cross-package imports since v1.0. The phase extracts shared types into a new `core/types/` package and relocates the implementation files into the stage submodules that own them.

The work divides cleanly into three concerns: (1) creating new file locations with correct content, (2) updating all 73 import sites across the codebase in one atomic pass, and (3) deleting the now-empty legacy directories. The trickiest part is the type/implementation split for `segmentation/detector.py`, `segmentation/crop.py`, `reconstruction/midline.py`, and `reconstruction/triangulation.py` — each file must be split into a types fragment going to `core/types/` and an implementation fragment going to the consuming stage submodule.

A key operational detail: because `core/midline/backends/segmentation.py` currently imports private helpers (`_adaptive_smooth`, `_skeleton_and_widths`, `_longest_path_bfs`, `_resample_arc_length`) from `reconstruction/midline.py` across package boundaries, moving `midline.py` to `core/midline/midline.py` resolves this smell entirely — those helpers become package-internal. Similarly, `visualization/` and tests have 30+ import sites that all need updating.

**Primary recommendation:** Execute the migration in dependency order — types first (`core/types/`), then implementation files, then update all consumers, then delete legacy directories. Each step leaves the codebase in a valid (if temporarily dual-path) state until the final delete.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python importlib | stdlib | No additional libraries needed | Pure file reorganization and import path surgery |
| hatch | project | Test runner (`hatch run test`) | Already established in project |

No new dependencies required. This phase uses only the existing project infrastructure.

### Supporting
Not applicable — this is a reorganization phase with no new technical dependencies.

## Architecture Patterns

### Recommended Project Structure (post-migration)
```
src/aquapose/
├── calibration/        # Unchanged
├── core/
│   ├── types/          # NEW: cross-stage shared types
│   │   ├── __init__.py      # Re-exports all public types
│   │   ├── detection.py     # Detection dataclass (from segmentation/detector.py)
│   │   ├── crop.py          # CropRegion, AffineCrop (from segmentation/crop.py)
│   │   ├── midline.py       # Midline2D (from reconstruction/midline.py)
│   │   └── reconstruction.py # Midline3D, MidlineSet (from reconstruction/triangulation.py)
│   ├── detection/
│   │   ├── backends/
│   │   │   ├── yolo.py     # NOW ALSO OWNS: YOLODetector, make_detector() (from segmentation/detector.py)
│   │   │   └── yolo_obb.py
│   │   ├── types.py         # DELETED (was thin re-export shim)
│   │   └── stage.py
│   ├── midline/
│   │   ├── backends/
│   │   │   ├── segmentation.py
│   │   │   └── pose_estimation.py
│   │   ├── crop.py          # NEW: extract_affine_crop(), invert_affine_point/points() (from segmentation/crop.py)
│   │   ├── midline.py       # NEW: MidlineExtractor + private helpers (from reconstruction/midline.py)
│   │   ├── orientation.py
│   │   ├── types.py         # DELETED (was thin re-export shim)
│   │   └── stage.py
│   ├── reconstruction/
│   │   ├── backends/
│   │   │   ├── triangulation.py   # UPDATED: imports from new locations
│   │   │   └── curve_optimizer.py # UPDATED: imports from new locations
│   │   ├── curve_optimizer.py     # NEW: full CurveOptimizer (from reconstruction/curve_optimizer.py)
│   │   ├── triangulation.py       # NEW: triangulate_midlines() etc (from reconstruction/triangulation.py)
│   │   ├── types.py               # DELETED (was thin re-export shim)
│   │   └── stage.py
│   ├── tracking/
│   │   ├── ocsort_wrapper.py      # NEW: OcSortTracker (from tracking/ocsort_wrapper.py)
│   │   ├── types.py               # KEEP (contains FishTrack, TrackState, TrackHealth, Tracklet2D — not legacy)
│   │   └── stage.py
│   ├── association/
│   ├── context.py
│   └── synthetic.py
├── engine/             # Unchanged except import updates
├── io/                 # Unchanged except import updates
├── synthetic/          # Unchanged except import updates
├── visualization/      # Unchanged except import updates
└── training/           # Unchanged
```
Legacy top-level directories **deleted entirely**:
- `reconstruction/` — GONE
- `segmentation/` — GONE
- `tracking/` — GONE

### Pattern 1: Type Extraction — Splitting a File into Types + Implementation

The four files requiring splitting each follow the same pattern: extract the dataclass/type alias into `core/types/<domain>.py`, leave implementation in the consuming location.

**Example — `reconstruction/midline.py` split:**

`core/types/midline.py` (types fragment):
```python
"""Shared type: 2D midline for a single fish in a single camera."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Midline2D:
    # ... exact same definition ...
```

`core/midline/midline.py` (implementation fragment — imports its own type):
```python
"""2D medial axis extraction and arc-length sampling pipeline."""

from __future__ import annotations
from aquapose.core.types.midline import CropRegion  # types come from core/types/
from aquapose.core.types.midline import Midline2D   # types come from core/types/
# ... MidlineExtractor, private helpers (_adaptive_smooth, etc.) ...
```

**Key constraint:** `core/types/` files MUST NOT import from `core/<stage>/` — types are the base layer. Implementations import types. Never the reverse.

### Pattern 2: Whole-File Move (ocsort_wrapper, curve_optimizer)

For files that move as a unit without splitting, the process is: copy file to new location, update its internal imports, then update all consumers. The file at the old location is deleted (no shim).

**`tracking/ocsort_wrapper.py` → `core/tracking/ocsort_wrapper.py`:**
- File already imports from `aquapose.core.tracking.types` (Tracklet2D) — that import stays unchanged
- Internal comment on line 194 references `aquapose.segmentation.detector` — update to `aquapose.core.types.detection`
- `tracking/__init__.py` currently re-exports `OcSortTracker` — this file is deleted
- `core/tracking/stage.py` currently does a lazy import from `aquapose.tracking.ocsort_wrapper` — update to `aquapose.core.tracking.ocsort_wrapper`

### Pattern 3: Eliminating Per-Stage Re-Export Shims

Existing per-stage `types.py` files are thin wrappers importing from legacy locations. After migration they are deleted:

- `core/detection/types.py`: currently `from aquapose.segmentation.detector import Detection` → DELETE
- `core/midline/types.py`: currently re-exports `Midline2D`, `CropRegion`, `Detection` from legacy + defines `AnnotatedDetection` → `AnnotatedDetection` moves to `core/midline/types.py` still, but imports repoint to `core/types/`... OR `AnnotatedDetection` joins a new home. See Open Questions.
- `core/reconstruction/types.py`: currently re-exports from legacy → DELETE
- `core/midline/__init__.py`: currently re-exports `Midline2D` from legacy → update

### Anti-Patterns to Avoid

- **Shim files:** Do NOT leave `reconstruction/__init__.py` alive as a re-export shim. Delete entire directories.
- **Circular imports:** `core/types/crop.py` defines `CropRegion` and `AffineCrop`. `core/midline/crop.py` imports these types. `core/midline/midline.py` imports `CropRegion` from `core/types/crop.py`. No cycle.
- **Reverse type imports:** Implementation files import from `core/types/`. `core/types/` files import ONLY stdlib and third-party (numpy, dataclasses). Never import from implementations.
- **Partial migration:** Do not update some consumers and leave others pointing at old paths. One-shot is the decision.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Finding all import sites | Custom AST walker | grep/ripgrep (already used in Phase 38-04) | Fast, already proven in this codebase |
| Import validation | Custom import checker | `hatch run test` (existing test suite catches broken imports immediately) | Tests import from all the right places |
| Circular import detection | Manual trace | `python -c "import aquapose"` after each file move | Python's import system reports cycles immediately |

## Common Pitfalls

### Pitfall 1: `core/midline/types.py` — AnnotatedDetection Has No Clear New Home
**What goes wrong:** `core/midline/types.py` currently defines `AnnotatedDetection` (a midline-stage-specific type that wraps `Detection` + `Midline2D`) AND re-exports `Midline2D`, `CropRegion`, `Detection` from legacy. The CONTEXT.md decision says "eliminate per-stage types.py re-export files." But `AnnotatedDetection` is a real type, not a shim.
**Why it happens:** `AnnotatedDetection` is a stage-specific composite type — it belongs to the midline stage. It is NOT a cross-stage shared type (nothing outside midline backends uses it directly).
**How to avoid:** Keep `core/midline/types.py` alive but rewrite it to define only `AnnotatedDetection` (not re-export from legacy). Its imports update from legacy paths to `core/types/` paths. It is NOT deleted — it is cleaned up.
**Warning signs:** If `core/midline/types.py` is deleted without moving `AnnotatedDetection` elsewhere, `core/midline/backends/segmentation.py` and `pose_estimation.py` will break (they import `AnnotatedDetection` from `core.midline.types`).

### Pitfall 2: Private Helper Imports from visualization/ — Still Need Updating
**What goes wrong:** `visualization/midline_viz.py` and `visualization/triangulation_viz.py` both import private helpers (`_adaptive_smooth`, `_skeleton_and_widths`, `_longest_path_bfs`, `_resample_arc_length`) via TYPE_CHECKING-style lazy `from aquapose.reconstruction.midline import (...)` blocks. After moving `midline.py` to `core/midline/midline.py`, these import paths break at runtime.
**Why it happens:** The visualization files are explicitly in-scope per CONTEXT.md ("this phase is not complete until all existing functionality works"), but they use lazy imports at function-level, making them easy to miss in a grep.
**How to avoid:** Check for function-level lazy imports in all visualization files, not just module-level imports.
**Exact locations to update:**
  - `visualization/midline_viz.py` lines ~393 and ~577: `from aquapose.reconstruction.midline import (_adaptive_smooth, ...)` → `from aquapose.core.midline.midline import (...)`
  - `visualization/triangulation_viz.py` lines ~327: `from aquapose.reconstruction.midline import (_adaptive_smooth, ...)` → `from aquapose.core.midline.midline import (...)`

### Pitfall 3: `reconstruction/curve_optimizer.py` Imports Private Functions from `triangulation.py`
**What goes wrong:** `curve_optimizer.py` imports `_pixel_half_width_to_metres` (a private function) from `reconstruction/triangulation.py`. After moving both files, `core/reconstruction/curve_optimizer.py` must import from `core/reconstruction/triangulation.py`. Since they're now co-located in the same package, this import is clean.
**How to avoid:** When moving `curve_optimizer.py` to `core/reconstruction/curve_optimizer.py`, update its internal import from `aquapose.reconstruction.triangulation` to `aquapose.core.reconstruction.triangulation`. The private function import becomes intra-package.

### Pitfall 4: Dead Code Analysis — v2.x Crop Functions
**What goes wrong:** `compute_crop_region()`, `extract_crop()`, `paste_mask()` in `segmentation/crop.py` have zero import sites in `src/` or `tests/` (confirmed by grep). Carrying them into `core/midline/crop.py` would be confusing if they are truly dead.
**Verification:** grep confirms zero usages outside `segmentation/crop.py` itself and `segmentation/__init__.py`. CONTEXT.md decision: "drop if unused." They are unused.
**How to avoid:** Do NOT port `compute_crop_region`, `extract_crop`, `paste_mask` to the new location. `core/midline/crop.py` should contain only: `extract_affine_crop`, `invert_affine_point`, `invert_affine_points`.

### Pitfall 5: `reconstruction/triangulation.py` Imports `Midline2D` from `reconstruction/midline.py`
**What goes wrong:** After splitting, `core/reconstruction/triangulation.py` must import `Midline2D` from `core/types/midline.py` — not from `core/midline/midline.py` (which is an implementation file). Getting the import direction wrong creates a cross-layer import from reconstruction → midline stage implementation.
**How to avoid:** The canonical source for `Midline2D` is `core/types/midline.py`. All reconstruction code imports from there.

### Pitfall 6: `__init__.py` Files Need Updating Throughout
**What goes wrong:** Several `__init__.py` files in the legacy packages and core packages currently import from legacy paths and will break immediately after moving files.
**Files requiring `__init__.py` surgery:**
  - `core/midline/__init__.py`: imports `Midline2D` from `aquapose.reconstruction.midline` → update to `aquapose.core.types.midline`
  - `core/reconstruction/__init__.py`: imports `Midline3D` from `aquapose.reconstruction.triangulation` → update to `aquapose.core.types.reconstruction`
  - Legacy `__init__.py` files (`reconstruction/__init__.py`, `segmentation/__init__.py`, `tracking/__init__.py`) → DELETE (with the entire directories)

### Pitfall 7: STAB-04 Docstring Scope Is Broader Than Just Moved Files
**What goes wrong:** STAB-04 requires updating all stale docstrings. Several files contain docstrings referencing legacy module paths or old terminology. After the migration, moved files need docstrings updated to reference new locations.
**Specific cases:**
  - `reconstruction/midline.py` module docstring: says nothing about U-Net (already clean)
  - `core/detection/backends/yolo.py` docstring: currently refers to `make_detector` being from `segmentation/detector.py`
  - `tracking/ocsort_wrapper.py` line 194 comment: `"List of Detection objects from aquapose.segmentation.detector."` → update to `aquapose.core.types.detection`
  - GUIDEBOOK.md source layout section: currently lists `reconstruction/`, `segmentation/`, `tracking/` as top-level packages → these descriptions must be removed/updated
  - CLAUDE.md Architecture section: lists `reconstruction/`, `segmentation/`, `tracking/` as `src/aquapose/` subdirectories → update

## Code Examples

### Complete Import Site Inventory (73 total)

**src/aquapose/ — 42 sites:**

| File | Current Import | New Import |
|------|---------------|------------|
| `core/detection/backends/yolo.py:15` | `from aquapose.segmentation.detector import Detection, YOLODetector` | `from aquapose.core.types.detection import Detection` + `YOLODetector` defined locally |
| `core/detection/backends/yolo_obb.py:20` | `from aquapose.segmentation.detector import Detection` | `from aquapose.core.types.detection import Detection` |
| `core/detection/types.py` | entire file (re-export shim) | DELETE (replaced by `core/types/detection.py`) |
| `core/midline/__init__.py:19` | `from aquapose.reconstruction.midline import Midline2D` | `from aquapose.core.types.midline import Midline2D` |
| `core/midline/backends/pose_estimation.py:21` | `from aquapose.reconstruction.midline import Midline2D` | `from aquapose.core.types.midline import Midline2D` |
| `core/midline/backends/pose_estimation.py:22-26` | `from aquapose.segmentation.crop import (AffineCrop, extract_affine_crop, invert_affine_points)` | `from aquapose.core.types.crop import AffineCrop` + `from aquapose.core.midline.crop import (extract_affine_crop, invert_affine_points)` |
| `core/midline/backends/pose_estimation.py:27` | `from aquapose.segmentation.detector import Detection` | `from aquapose.core.types.detection import Detection` |
| `core/midline/backends/segmentation.py:20-26` | `from aquapose.reconstruction.midline import (Midline2D, _adaptive_smooth, ...)` | `from aquapose.core.types.midline import Midline2D` + `from aquapose.core.midline.midline import (_adaptive_smooth, ...)` |
| `core/midline/backends/segmentation.py:27-31` | `from aquapose.segmentation.crop import (AffineCrop, extract_affine_crop, invert_affine_points)` | `from aquapose.core.types.crop import AffineCrop` + `from aquapose.core.midline.crop import (extract_affine_crop, invert_affine_points)` |
| `core/midline/backends/segmentation.py:32` | `from aquapose.segmentation.detector import Detection` | `from aquapose.core.types.detection import Detection` |
| `core/midline/types.py:15-17` | imports from legacy paths | repoint to `core/types/` (keep file for `AnnotatedDetection`) |
| `core/reconstruction/__init__.py:9` | `from aquapose.reconstruction.triangulation import Midline3D` | `from aquapose.core.types.reconstruction import Midline3D` |
| `core/reconstruction/backends/curve_optimizer.py:14-15` | imports from `aquapose.reconstruction.*` | `from aquapose.core.reconstruction.curve_optimizer import ...` + `from aquapose.core.types.reconstruction import ...` |
| `core/reconstruction/backends/triangulation.py:14-19` | `from aquapose.reconstruction.triangulation import (...)` | `from aquapose.core.reconstruction.triangulation import (...)` + `from aquapose.core.types.reconstruction import (...)` |
| `core/reconstruction/stage.py:25-26` | `from aquapose.reconstruction.midline import Midline2D` + `from aquapose.reconstruction.triangulation import ...` | `from aquapose.core.types.midline import Midline2D` + `from aquapose.core.types.reconstruction import Midline3D` + `from aquapose.core.reconstruction.triangulation import DEFAULT_INLIER_THRESHOLD` |
| `core/reconstruction/types.py` | entire file (re-export shim) | DELETE (replaced by `core/types/reconstruction.py`) |
| `core/synthetic.py:20-22` | imports from legacy paths | `from aquapose.core.types.midline import Midline2D` + `from aquapose.core.types.crop import CropRegion` + `from aquapose.core.types.detection import Detection` |
| `core/tracking/stage.py:76` | lazy `from aquapose.tracking.ocsort_wrapper import OcSortTracker` | `from aquapose.core.tracking.ocsort_wrapper import OcSortTracker` |
| `io/midline_writer.py:15,22` | `from aquapose.reconstruction.triangulation import (...)` | `from aquapose.core.types.reconstruction import (...)` + `from aquapose.core.reconstruction.triangulation import (...)` |
| `reconstruction/midline.py` (internal) | `from aquapose.segmentation.crop import CropRegion` | becomes `from aquapose.core.types.crop import CropRegion` in new location `core/midline/midline.py` |
| `reconstruction/triangulation.py` (internal) | `from aquapose.reconstruction.midline import Midline2D` | becomes `from aquapose.core.types.midline import Midline2D` in new location |
| `reconstruction/curve_optimizer.py` (internal) | `from aquapose.reconstruction.triangulation import (...)` | becomes `from aquapose.core.reconstruction.triangulation import (...)` in new location |
| `synthetic/detection.py:17` | `from aquapose.segmentation.detector import Detection` | `from aquapose.core.types.detection import Detection` |
| `synthetic/fish.py:18-19` | `from aquapose.reconstruction.midline import Midline2D` + `from aquapose.reconstruction.triangulation import (...)` | `from aquapose.core.types.midline import Midline2D` + `from aquapose.core.types.reconstruction import (Midline3D, MidlineSet)` + `from aquapose.core.reconstruction.triangulation import (triangulate_midlines)` |
| `tracking/__init__.py:13` | `from aquapose.tracking.ocsort_wrapper import OcSortTracker` | DELETE (whole directory deleted) |
| `visualization/midline_viz.py:24-25` | lazy imports from `segmentation.crop`, `segmentation.detector` | `from aquapose.core.types.crop import CropRegion` + `from aquapose.core.types.detection import Detection` |
| `visualization/midline_viz.py:393,577` | lazy `from aquapose.reconstruction.midline import (_adaptive_smooth, ...)` | `from aquapose.core.midline.midline import (_adaptive_smooth, ...)` |
| `visualization/overlay.py:20` | lazy `from aquapose.reconstruction.triangulation import Midline3D` | `from aquapose.core.types.reconstruction import Midline3D` |
| `visualization/plot3d.py:22` | lazy `from aquapose.reconstruction.triangulation import Midline3D` | `from aquapose.core.types.reconstruction import Midline3D` |
| `visualization/triangulation_viz.py:25-28` | lazy imports from multiple legacy paths | update each to `core/types/` or `core/reconstruction/triangulation` |
| `visualization/triangulation_viz.py:327` | lazy `from aquapose.reconstruction.midline import (_adaptive_smooth, ...)` | `from aquapose.core.midline.midline import (_adaptive_smooth, ...)` |

**tests/ — 31 sites:**

All test files referencing legacy paths must be updated to new `core/types/` or `core/<stage>/` paths. Key files:

| Test File | What Changes |
|-----------|-------------|
| `tests/unit/segmentation/test_affine_crop.py` | `from aquapose.segmentation.crop import (...)` → `from aquapose.core.types.crop import (...)` + `from aquapose.core.midline.crop import (...)` |
| `tests/unit/segmentation/test_detector.py` | `from aquapose.segmentation.detector import (...)` → `from aquapose.core.types.detection import Detection` + `from aquapose.core.detection.backends.yolo import (YOLODetector, make_detector)` |
| `tests/unit/tracking/test_ocsort_wrapper.py` | `from aquapose.tracking.ocsort_wrapper import OcSortTracker` → `from aquapose.core.tracking.ocsort_wrapper import OcSortTracker` |
| `tests/unit/test_midline.py` | `from aquapose.reconstruction.midline import (...)` + `from aquapose.segmentation.crop import CropRegion` → `from aquapose.core.midline.midline import (...)` + `from aquapose.core.types.midline import Midline2D` + `from aquapose.core.types.crop import CropRegion` |
| `tests/unit/test_triangulation.py` | `from aquapose.reconstruction.midline import Midline2D` + `from aquapose.reconstruction.triangulation import (...)` → `from aquapose.core.types.midline import Midline2D` + `from aquapose.core.types.reconstruction import (Midline3D, MidlineSet)` + `from aquapose.core.reconstruction.triangulation import (...)` |
| `tests/unit/test_curve_optimizer.py` | `from aquapose.reconstruction.curve_optimizer import (...)` + `from aquapose.reconstruction.midline import Midline2D` + `from aquapose.reconstruction.triangulation import (...)` → updated to `core/reconstruction/curve_optimizer` + `core/types/` |
| `tests/unit/core/reconstruction/test_confidence_weighting.py` | all imports from legacy → `core/reconstruction/curve_optimizer` + `core/types/` |
| `tests/unit/core/reconstruction/test_reconstruction_stage.py` | `from aquapose.reconstruction.{midline,triangulation} import (...)` → `core/types/` |
| `tests/unit/core/midline/test_segmentation_backend.py` | `from aquapose.segmentation.crop import AffineCrop` + `from aquapose.segmentation.detector import Detection` → `from aquapose.core.types.crop import AffineCrop` + `from aquapose.core.types.detection import Detection` |
| `tests/unit/core/midline/test_pose_estimation_backend.py` | same as above |
| `tests/unit/core/midline/test_midline_stage.py` | `from aquapose.segmentation.detector import Detection` → `from aquapose.core.types.detection import Detection` |
| `tests/unit/core/midline/test_direct_pose_backend.py` | `from aquapose.segmentation.detector import Detection` → `from aquapose.core.types.detection import Detection` |
| `tests/unit/synthetic/test_detection_gen.py` | `from aquapose.segmentation.detector import Detection` → `from aquapose.core.types.detection import Detection` |
| `tests/unit/synthetic/test_synthetic.py` | `from aquapose.reconstruction.triangulation import (...)` → `core/types/reconstruction` |
| `tests/unit/core/test_synthetic.py` | lazy `from aquapose.reconstruction.midline import Midline2D` → `from aquapose.core.types.midline import Midline2D` |
| `tests/unit/io/test_midline_writer.py` | `from aquapose.reconstruction.triangulation import (...)` → `core/types/reconstruction` + `core/reconstruction/triangulation` |

### Suggested Migration Order

Execute in this dependency order to minimize broken intermediate states:

**Step 1 — Create `core/types/` package (no deletions yet)**
- Create `core/types/__init__.py`, `detection.py`, `crop.py`, `midline.py`, `reconstruction.py`
- Content copied from legacy source files (types only)
- `core/types/__init__.py` re-exports all public types

**Step 2 — Move implementation files to new core locations**
- `reconstruction/midline.py` → `core/midline/midline.py` (update its import of `CropRegion` to `core/types/crop`)
- `segmentation/crop.py` (affine utilities only) → `core/midline/crop.py`
- `reconstruction/triangulation.py` → `core/reconstruction/triangulation.py` (update its import of `Midline2D` to `core/types/midline`)
- `reconstruction/curve_optimizer.py` → `core/reconstruction/curve_optimizer.py` (update its imports to `core/reconstruction/triangulation`)
- `tracking/ocsort_wrapper.py` → `core/tracking/ocsort_wrapper.py` (no type changes needed — already imports from `core/tracking/types`)

**Step 3 — Merge `YOLODetector` + `make_detector()` into detection backend**
- Add `YOLODetector` class and `make_detector()` to `core/detection/backends/yolo.py`
- Update its exports and `__all__`

**Step 4 — Update all consumers (one-shot pass)**
- Update all 73 import sites across `core/`, `engine/`, `io/`, `synthetic/`, `visualization/`, tests
- Delete per-stage re-export shim files: `core/detection/types.py`, `core/reconstruction/types.py`
- Update (not delete) `core/midline/types.py`: rewrite to import from `core/types/`, keep `AnnotatedDetection`

**Step 5 — Delete legacy directories**
- Delete `reconstruction/` (entire directory)
- Delete `segmentation/` (entire directory)
- Delete `tracking/` (entire directory)

**Step 6 — Update documentation**
- GUIDEBOOK.md source layout section: remove `reconstruction/`, `segmentation/`, `tracking/` entries; add `core/types/`
- CLAUDE.md Architecture section: update the directory tree
- Update module-level docstrings in moved files to reference new paths

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Types defined in legacy packages, imported cross-package | Types in `core/types/`, implementations in stage submodules | Eliminates cross-package private imports; `core/` is self-contained |
| `reconstruction/`, `segmentation/`, `tracking/` as top-level siblings | Relocated to `core/<stage>/` submodules | Matches architectural documentation; import discipline becomes enforceable |
| Per-stage thin `types.py` re-export shims (e.g., `core/detection/types.py`) | Eliminated | Reduces indirection; canonical source is always `core/types/` |

## Open Questions

1. **Where does `AnnotatedDetection` live permanently?**
   - What we know: `AnnotatedDetection` is defined in `core/midline/types.py` and used only by `core/midline/backends/segmentation.py` and `core/midline/backends/pose_estimation.py` (and tests for those backends).
   - What's unclear: CONTEXT.md says "eliminate per-stage types.py re-export files." But `AnnotatedDetection` is not a re-export — it is a real midline-stage-specific type.
   - Recommendation: Keep `core/midline/types.py` alive as a non-shim file that contains only `AnnotatedDetection`. Update its imports to point to `core/types/`. This is consistent with the intent: eliminate SHIMS, not eliminate all per-stage types.

2. **Does `make_detector()` merge into `yolo.py` or get its own file?**
   - What we know: CONTEXT.md assigns this to Claude's discretion. Currently `make_detector()` is in `segmentation/detector.py` alongside `YOLODetector`. The function is a single-dispatch factory that returns `YOLODetector`.
   - What's unclear: Whether a 3-line factory function merits its own file vs. being colocated with `YOLODetector` in `yolo.py`.
   - Recommendation: Merge into `core/detection/backends/yolo.py`. The factory is trivial and `YOLODetector` is its only product. Having a separate `factory.py` would be over-engineering.

3. **How should `core/types/__init__.py` `__all__` be structured?**
   - What we know: CONTEXT.md says it should re-export all public types: `from aquapose.core.types import Detection, Midline2D`.
   - Recommendation: Import and re-export all public types from all four domain files. This gives consumers a single import target while allowing granular imports from submodules when clarity is preferred.

## Validation Architecture

> `workflow.nyquist_validation` is not present in `.planning/config.json` (the file uses a different schema) — treating as disabled. Skipping Validation Architecture section.

The existing test suite is the validation mechanism. Run `hatch run test` after each migration step to catch broken imports immediately. All 73 import sites are covered by existing tests — no new test infrastructure is required.

### Phase Gate Checks

| Check | Command | What It Catches |
|-------|---------|----------------|
| Import smoke test | `python -c "import aquapose"` | Circular imports, missing modules |
| Unit tests | `hatch run test` | All 73 import sites exercised via existing tests |
| Type check | `hatch run typecheck` | Incorrect type annotations after path changes |
| Lint | `hatch run lint` | Code style regressions |

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection — all findings are from reading actual source files
- grep of all import sites confirmed against file contents

### Secondary (MEDIUM confidence)
- None required — this is a pure codebase reorganization with no external library dependencies

## Metadata

**Confidence breakdown:**
- File inventory: HIGH — verified by reading every affected file
- Import site count (73): HIGH — confirmed by grep across src/ and tests/
- Dead code finding (v2.x crop functions): HIGH — zero grep matches outside definition files
- Migration ordering: MEDIUM — based on dependency analysis; circular import issues are possible during the split and may require adjustment
- `AnnotatedDetection` home: MEDIUM — recommendation is a judgment call; planner should confirm

**Research date:** 2026-03-02
**Valid until:** N/A (codebase-internal research, valid until codebase changes)
