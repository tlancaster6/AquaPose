# Phase 39: Migrate Legacy Domain Libraries into Core Submodules - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Legacy top-level domain libraries (`reconstruction/`, `segmentation/`, `tracking/`) are reorganized into `core/` submodules alongside the stages that consume them. Cross-stage types are extracted into a new `core/types/` package. All imports across the codebase (including `visualization/`, `io/`, `synthetic/`, `engine/`, and tests) are updated in a one-shot rewrite. Old directories are deleted entirely.

</domain>

<decisions>
## Implementation Decisions

### File placement strategy
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
  - `extract_affine_crop()`, `invert_affine_points()` utilities → `core/midline/crop.py`
  - v2.x functions (`compute_crop_region`, `extract_crop`, `paste_mask`) → drop if unused, otherwise bring along
- `reconstruction/midline.py` → split:
  - `Midline2D` type → `core/types/midline.py`
  - `MidlineExtractor` + private helpers → `core/midline/midline.py`
- `reconstruction/triangulation.py` → split:
  - `Midline3D`, `MidlineSet` types → `core/types/reconstruction.py`
  - `triangulate_midlines()` + helpers + constants → `core/reconstruction/triangulation.py`
- `reconstruction/curve_optimizer.py` → whole file → `core/reconstruction/curve_optimizer.py`
- `tracking/ocsort_wrapper.py` → whole file → `core/tracking/ocsort_wrapper.py`

### Re-export / backwards-compat approach
- One-shot rewrite of all import paths across the entire codebase — no shims, no deprecation period
- ALL consumers updated in the same pass: `core/`, `engine/`, `visualization/`, `io/`, `synthetic/`, tests
- visualization/ is explicitly in scope — this phase is not complete until all existing functionality works
- Old legacy directories (`reconstruction/`, `segmentation/`, `tracking/`) deleted entirely after migration

### Private helper exposure
- Private helpers (`_adaptive_smooth`, `_skeleton_and_widths`, `_longest_path_bfs`, `_resample_arc_length`) stay private (underscore-prefixed)
- After moving to `core/midline/midline.py`, the cross-package import smell vanishes since they're now package-internal
- v2.x crop functions (`compute_crop_region`, `extract_crop`, `paste_mask`) — check usage and drop if dead code
- Import direction: `core/types/` is canonical source, implementations import types from there (not reverse)

### Type ownership
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

</decisions>

<specifics>
## Specific Ideas

- "This phase is not complete until all existing critical functionality is regained" — visualization/ imports must be updated, not deferred
- The migration is about relocation and type extraction, not restructuring. Files should keep their internal structure (no splitting MidlineExtractor from its helpers, no refactoring implementations)
- v2.x dead code should be dropped during migration rather than carried forward

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `core/types/` pattern is new — currently no shared types package exists
- Per-stage `types.py` files exist in `core/detection/`, `core/midline/`, `core/reconstruction/`, `core/tracking/` — these are thin re-export wrappers that will be eliminated
- `core/midline/backends/segmentation.py` and `core/midline/backends/pose_estimation.py` are the heaviest consumers of cross-package imports

### Established Patterns
- Stage protocol: stages in `core/<stage>/stage.py`, backends in `core/<stage>/backends/`
- Import discipline documented in GUIDEBOOK.md: `core/ → nothing`, `engine/ → core/`, `cli/ → engine/`
- Legacy directories are Layer 1 computation — canonical implementations, not dead code

### Integration Points
- 25+ imports from `core/` into legacy directories (all must be rewired)
- 15+ imports from outside `core/` (engine observers, visualization, io, synthetic)
- Test files import from legacy paths and must be updated
- GUIDEBOOK.md source layout section needs path updates

</code_context>

<deferred>
## Deferred Ideas

- visualization/ directory reorganization (moving files closer to engine observers) — separate phase
- Refactoring MidlineExtractor internals or splitting large files — separate concern

</deferred>

---

*Phase: 39-migrate-legacy-domain-libraries-into-core-submodules*
*Context gathered: 2026-03-02*
