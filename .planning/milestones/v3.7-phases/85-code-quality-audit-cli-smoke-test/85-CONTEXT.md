# Phase 85: Code Quality Audit & CLI Smoke Test - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Verify the v3.7 overhaul (segmentation removal, pipeline reorder, custom tracker, association upgrade) left no dead code, broken cross-references, or type errors. Confirm the full pipeline runs cleanly from the CLI with the new stage ordering. This is a quality gate — no new features.

</domain>

<decisions>
## Implementation Decisions

### BoxMot Removal
- Remove BoxMot entirely — delete the dependency from pyproject.toml, the ocsort_wrapper module, and its test file (test_ocsort_wrapper.py)
- Remove all config keys related to OC-SORT/BoxMot (tracker_kind "ocsort" option, iou_threshold for ocsort, boxmot references in docstrings)
- Clean break — no backward-compatible aliases, no fallback
- Verify no orphaned transitive dependencies remain after removal (run dependency check)

### Type Error Triage
- Fix all 25 type errors — not just v3.7 regressions, bring the count to zero
- Fix the LutConfig protocol mismatch properly (align the LutConfigLike protocol with the frozen dataclass pattern)
- Use targeted `# type: ignore[attr]` comments for third-party stub gaps (cv2.VideoWriter_fourcc etc.)
- No CI enforcement gate — just fix the current batch; pre-commit hooks already run typecheck

### CLI Smoke Test
- Use YH project config with chunk-size=100, max-chunks=2 (real data, exercises chunk plumbing)
- Success = exit code 0 AND expected output artifacts exist (run directory, frozen config, stage outputs)
- Verify the new tracker's CLI flags parse correctly (the 6 tunable params from Phase 84.1)
- Tracker params stay as YAML config only — do NOT add individual CLI flags to `aquapose run`
- Tracker config documentation is satisfied by existing TrackingConfig docstrings — no separate doc file

### Audit Reporting
- Fix inline + write audit report to phase directory (85-AUDIT-REPORT.md)
- Full dead code scan across the entire codebase, not just v3.7 remnants
- Fix obviously dead code (unused imports, unreachable functions) inline if safe
- Document ambiguous findings for Phase 86 review
- Report structure: what was found, what was fixed, what remains for Phase 86

### Claude's Discretion
- Dead code scanning approach (manual inspection, vulture, or combination)
- Exact order of audit tasks (type fixes, dead code removal, BoxMot removal, smoke test)
- Report format and level of detail

</decisions>

<specifics>
## Specific Ideas

- User explicitly prefers YAML config over CLI flags for tracker parameters — "yaml config is sufficient, and preferable"
- Smoke test should exercise chunk-based execution to test all key plumbing
- The audit report determines whether Phase 86 is needed or can be skipped

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TrackingConfig` (engine/config.py:173-224): Already has detailed docstrings for all 6 tracker params — satisfies documentation requirement
- `hatch run typecheck` (basedpyright): Current tool for type checking, shows 25 errors

### Established Patterns
- Frozen dataclass config hierarchy (engine/config.py) — LutConfig protocol fix must maintain this pattern
- Backend registration pattern (tracker_kind in valid_kinds set) — needs updating when ocsort removed
- GSD phase artifacts in `.planning/phases/XX-name/` — audit report follows this convention

### Integration Points
- pyproject.toml: BoxMot dependency line to remove
- engine/config.py: TrackingConfig.tracker_kind validation, docstrings referencing boxmot/ocsort
- core/tracking/ocsort_wrapper.py: Entire module to remove
- core/tracking/stage.py: References to ocsort backend to remove
- tests/unit/tracking/test_ocsort_wrapper.py: Test file to remove
- calibration/ LutConfigLike protocol: Structural fix needed for frozen dataclass compatibility

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 85-code-quality-audit-cli-smoke-test*
*Context gathered: 2026-03-11*
