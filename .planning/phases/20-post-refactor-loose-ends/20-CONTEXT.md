# Phase 20: Post-Refactor Loose Ends - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Remediate all findings from the Phase 19 audit (19-AUDIT.md). Fix the critical IB-003 violations, resolve warning-level items (Stage 3/4 coupling, CLI thinning, camera skip removal, dead modules), address info-level items (large file splitting, duplicated code, stale comments), and fix + run regression tests to confirm v2.0 numerical equivalence. This phase closes out the v2.0 Alpha refactor.

</domain>

<decisions>
## Implementation Decisions

### IB-003 Fix Strategy (AUD-001 — Critical)
- Move `PipelineContext`, `CarryForward`, and `Stage` Protocol to `core/context.py` — they are pure data containers with no engine logic
- Delete `engine/stages.py` entirely — no re-export shim, no backward compat. Update all imports in one pass
- Update the import boundary checker to flag TYPE_CHECKING imports too — zero tolerance for core/ importing engine/ in any form
- This single change resolves all 7 IB-003 violations and both DoD criterion 7 failures

### Stage 3/4 Coupling Fix (AUD-005, AUD-006 — Warning)
- Refactor `FishTracker.update()` to accept pre-associated bundles from Stage 3 output directly
- Remove FishTracker's internal RANSAC association — Stage 3 owns association now
- Delete the old association code inside tracking (no archive, clean cut)
- Stage 3 is a hard dependency for Stage 4 — if association hasn't run, tracking raises a precondition error
- No new backend; refactor the existing Hungarian backend to consume bundles

### Dead Module Cleanup (AUD-008, AUD-019, AUD-020)
- Delete all 5 dead modules and their orphaned tests:
  - `src/aquapose/pipeline/` + `tests/unit/pipeline/`
  - `src/aquapose/initialization/` + `tests/unit/initialization/`
  - `src/aquapose/mesh/` + `tests/unit/mesh/`
  - `src/aquapose/utils/`
  - `src/aquapose/optimization/` (stale `__pycache__/` only)
- Delete legacy scripts that depend on dead modules:
  - `scripts/legacy/diagnose_pipeline.py`
  - `scripts/legacy/diagnose_tracking.py`
  - `scripts/legacy/diagnose_triangulation.py`
  - `scripts/legacy/per_camera_spline_overlay.py`
- Remove `pytorch3d` from optional dependencies if present

### CLI Thinning (AUD-002 — Warning)
- Extract `_build_observers()` from `cli.py` into `engine/` as a factory function (e.g., `build_observers()`)
- CLI should drop to ~100 LOC — parse args, build config, call engine factories, run pipeline

### Camera Skip Removal (AUD-007 — Warning)
- Remove all camera skip logic entirely — no `skip_camera_id` config, no hardcoded constants
- If a user doesn't want a camera analyzed, they don't include its video in the input directory
- Delete all 9+ `_DEFAULT_SKIP_CAMERA_ID` constants from core stage files
- Remove any skip-camera filtering from stage `run()` methods

### Info-Level Cleanup (AUD-009 through AUD-018)
- Extract duplicated camera-video discovery pattern from `detection/stage.py` and `midline/stage.py` into a shared utility
- Split `visualization/diagnostics.py` (2,203 LOC) into focused modules (overlay, midline viz, triangulation viz)
- Remove stale "v1.0 orchestrator" comments from `core/detection/stage.py` and `core/midline/stage.py`
- Delete backward-compat test fixtures that test dead modules (`test_stages.py` importability, `MaskRCNNSegmentor` check)
- Address any remaining info items from the audit

### Regression Test Fix (AUD-004 — Warning)
- Replace hardcoded video/calibration paths with environment variables:
  - `AQUAPOSE_VIDEO_DIR` — path to video directory. No fallback. Tests skip with clear message if not set
  - `AQUAPOSE_CALIBRATION_PATH` — path to calibration file. No fallback. Same skip behavior
- Run the full regression test suite after fixing paths
- If numerical drift is found: Claude triages per case — fix bugs, accept inherent non-determinism (RANSAC), document reasoning

### Claude's Discretion
- Exact file organization within `core/context.py` (whether to split into multiple files if large)
- How to split `visualization/diagnostics.py` (exact module boundaries)
- Whether to keep any backward-compat re-exports during intermediate commits
- Per-case triage of any regression test failures (fix vs accept with documentation)
- Plan ordering and wave grouping for the remediation work

</decisions>

<specifics>
## Specific Ideas

- Camera skip removal is a philosophical shift: the pipeline processes everything in the input directory, period. No filtering logic inside the pipeline
- The IB-003 fix makes `core/` truly standalone — it defines its own types, never references `engine/`
- Stage 3 → Stage 4 coupling fix means the pipeline's data flow diagram is finally honest: each stage consumes the previous stage's output

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 20-post-refactor-loose-ends*
*Context gathered: 2026-02-26*
