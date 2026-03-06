# Phase 19: Alpha Refactor Audit - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Verify that the completed v2.0 Alpha refactor (Phases 13-18) faithfully implements the architectural vision in `alpha_refactor_guidebook.md`. Identify gaps, violations, and cleanup opportunities. Produce a structured audit report and reusable tooling. This phase does NOT fix findings — it catalogs them for Phase 20.

</domain>

<decisions>
## Implementation Decisions

### Audit scope & depth
- Pragmatic assessment: guidebook is the target, but known compromises are acceptable if documented. Focus on architectural violations, not cosmetic mismatches.
- Three-tier severity: Critical (architectural violation, must fix), Warning (non-ideal but functional, should fix), Info (cosmetic/minor, fix if convenient)
- Definition of Done (guidebook section 16) is the primary gate check. If DoD passes, the refactor is fundamentally sound. Other findings are improvements.

### Verification runs
- Full pipeline verification: run `aquapose run` across (1) each mode preset (production, diagnostic, synthetic, benchmark), (2) each swappable backend (e.g. curve_optimizer vs triangulation), (3) both real and synthetic data
- Not a full matrix — each dimension tested independently with others at defaults
- Stop-frame at 10 to keep timing reasonable
- Reproducibility check: run pipeline twice with identical config, compare outputs at data level (np.allclose on HDF5 arrays, not byte-identical files)
- Package as a reusable smoke test script, flexible to future changes

### Import boundary enforcement
- Automated lint rule as a pre-commit hook enforcing core/ → engine/ → cli/ layering
- No TYPE_CHECKING exemptions — the guidebook explicitly forbids TYPE_CHECKING backdoors
- Also check all structural rules from guidebook section 15: no file I/O in stage run(), no mutable class attrs on stages, observers don't import core/ internals
- Both the lint rule and pre-commit hook are deliverables of this phase

### Audit deliverables
- Structured markdown report: `19-AUDIT.md` in the phase directory
- Sections: DoD Gate Check, Structural Rules, Verification Run Results, Codebase Health, Findings by Severity, Remediation Summary
- Reusable smoke test script (committed as a tool)
- Import boundary pre-commit hook (committed and active)
- Findings inform Phase 20 but do NOT auto-populate it — separate planning after review

### Codebase health audit
- Dead code candidates for deletion
- Repeated patterns candidates for functionalization
- Bloated modules candidates for splitting into submodules
- Legacy scripts that won't work with the new codebase (update or delete)
- Guidebook vs actual structure deviations (flag only, don't propose fixes)
- Unused imports and dependencies in pyproject.toml
- Stale test fixtures referencing old pipeline patterns
- Inconsistent naming (old conventions vs new engine vocabulary)
- TODO/FIXME/HACK comment catalog (actionable vs stale)
- Cleanup findings use the same 3-tier severity in a "Codebase Health" section of AUDIT.md

### Numerical verification
- Golden data regression tests from Phase 16 should be passing — failures are Critical severity
- Missing coverage for a stage is Warning severity
- Both go into the audit report for Phase 20 remediation

### Claude's Discretion
- Exact smoke test script structure and CLI interface
- How to implement the import boundary checker (AST-based vs regex vs ruff plugin)
- Which structural rules are feasible to automate vs require manual review
- Audit report organization within the defined sections

</decisions>

<specifics>
## Specific Ideas

- Smoke test should be packaged for reuse — "a test we'll run rarely but will run again"
- Smoke test must be flexible to future changes (new modes, new backends)
- Guidebook deviations flagged only — user decides in Phase 20 whether to update guidebook or fix code
- Cleanup audit includes things like: stale TODO comments, unused pyproject.toml deps, naming inconsistencies between old and new vocabulary

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 19-alpha-refactor-audit*
*Context gathered: 2026-02-26*
