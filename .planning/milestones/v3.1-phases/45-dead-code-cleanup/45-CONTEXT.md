# Phase 45: Dead Code Cleanup - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove the old triangulation backend, curve optimizer backend, and dead reconstruction code. After cleanup, DLT is the sole reconstruction backend. No new functionality is added.

</domain>

<decisions>
## Implementation Decisions

### Shared Utility Handling
- Surviving helpers/constants from `reconstruction/triangulation.py` (e.g. `DEFAULT_INLIER_THRESHOLD`, refraction utilities) move to `reconstruction/utils.py`
- All consumers (`dlt.py`, `stage.py`, `synthetic/fish.py`, `io/midline_writer.py`) update imports to point at `utils.py`
- Researcher should investigate whether `synthetic/fish.py` and `io/midline_writer.py` are themselves dead code — if so, delete them too
- The evaluation harness should use the backend registry (`get_backend('dlt')`) rather than direct `DltBackend` import

### Config Cleanup
- Default reconstruction backend changes from `'triangulation'` to `'dlt'`
- Remove `'triangulation'` and `'curve_optimizer'` from the backend registry with a standard `ValueError` (no migration hints or deprecation warnings)
- Remove config fields that only the old backends used (e.g. `inlier_threshold`, `snap_threshold`, `max_depth`) — keep only what DLT needs
- Update `~/aquapose/projects/YH/config.yaml` to use `'dlt'` and remove any old-backend-specific fields

### Backend Naming
- Keep `'dlt'` as the backend kind name — it's accurate and specific
- Do not rename to a generic `'triangulation'`

### Dead Code Sweep Scope
- Opportunistic: delete exactly what CLEAN-01/02/03 specify, plus any additional dead code found in `reconstruction/` during investigation
- Remove visualization code tied to deleted backends (`triangulation_viz.py` — delete if only serves old backends, keep DLT-relevant parts)
- Update `GUIDEBOOK.md` to remove references to old backends (triangulation, curve_optimizer)
- Keep the backend registry pattern (`get_backend()`) with only `'dlt'` registered — preserves extensibility for future backends

### Claude's Discretion
- Exact ordering of file deletions and import fixups
- Whether to consolidate `reconstruction/utils.py` or keep it minimal
- How to handle any edge cases in test files referencing deleted code

</decisions>

<specifics>
## Specific Ideas

No specific requirements — straightforward mechanical cleanup guided by the decisions above.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `reconstruction/utils.py`: Already exists, natural home for surviving shared helpers
- `reconstruction/backends/__init__.py`: Registry pattern (`get_backend()`) survives with only `'dlt'` entry

### Established Patterns
- Backend registry pattern used by detection, midline, and reconstruction stages — keep consistent
- Import discipline: `core/` imports only stdlib, third-party, and core internals

### Integration Points
- `reconstruction/stage.py`: Imports `DEFAULT_INLIER_THRESHOLD` from old `triangulation.py` — must update
- `synthetic/fish.py`: Imports from `triangulation.py` — investigate if dead code
- `io/midline_writer.py`: Imports from `triangulation.py` — investigate if dead code
- `evaluation/harness.py`: Currently imports both `TriangulationBackend` and `DltBackend` — switch to registry
- `visualization/triangulation_viz.py`: References `curve_optimizer` — likely dead, investigate
- `engine/config.py`: `ReconstructionConfig` references old backend kinds and fields
- `~/aquapose/projects/YH/config.yaml`: Project config may reference old backends

### Files to Delete (candidates)
- `reconstruction/backends/triangulation.py` (167 lines)
- `reconstruction/backends/curve_optimizer.py` (132 lines)
- `reconstruction/triangulation.py` (910 lines)
- `reconstruction/curve_optimizer.py` (1881 lines)
- `visualization/triangulation_viz.py` (if only serves old backends)

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 45-dead-code-cleanup*
*Context gathered: 2026-03-03*
