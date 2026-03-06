# Phase 21: Retrospective, Prospective - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Produce a backward-looking retrospective of the v2.0 Alpha refactor (Phases 13-20) and a forward-looking prospective that seeds the next milestone's requirements. This phase is analytical and documentary — it assesses what was built, identifies gaps, and sets direction. It does not implement new features or fix substantive issues.

</domain>

<decisions>
## Implementation Decisions

### Retrospective Scope
- Broader assessment: DoD gates from the guidebook (Section 16) PLUS architecture quality, code health, testing coverage, and lessons learned
- Fresh analysis independent of the Phase 19 audit report (19-AUDIT.md) — higher-level view, not a rehash of granular findings
- Includes GSD workflow/process lessons learned — what worked, what was friction, what to change for next milestone
- Quantitative metrics (test counts, coverage %, LoC delta, module counts) woven inline alongside qualitative narrative

### Prospective Direction
- Primary thrust: end-to-end accuracy improvement across all pipeline stages
- Stage-by-stage priority assessment — identify where accuracy gains are most impactful, ordered by bottleneck analysis
- Directional goals (not specific numeric targets) — let target-setting happen during next milestone's requirements phase
- Driven by research publication needs — results need to be good enough to publish

### Output Format
- Two separate documents:
  - `21-RETROSPECTIVE.md` — backward look, architecture assessment, metrics, process reflections. Feeds into /gsd:complete-milestone as the completion narrative for archiving v2.0 Alpha
  - `21-PROSPECTIVE.md` — forward-looking direction, structured as a requirements seed for /gsd:new-milestone. Candidate requirements with priority ordering and phase suggestions
- Quantitative metrics inline in retrospective (no separate appendix)

### Gap Reconciliation
- Discovery-based: let the retrospective analysis surface gaps organically rather than pre-listing them
- Fix trivial bookkeeping discrepancies in this phase (e.g., CLI-01 through CLI-05 status in REQUIREMENTS.md — confirmed fully implemented, just missed check-off)
- Substantive gaps cataloged as candidate requirements feeding into the prospective document
- Gap flow: retrospective discovers gaps → gaps become candidate requirements in prospective → prospective seeds /gsd:new-milestone

### Claude's Discretion
- Internal structure and section ordering of both documents
- Which metrics are worth gathering vs. noise
- How to present the GSD process retrospective (inline vs. dedicated section)
- Level of detail in stage-by-stage prospective assessment

</decisions>

<specifics>
## Specific Ideas

- Retrospective should combine with /gsd:complete-milestone — serves as the milestone completion narrative
- Prospective structured so /gsd:new-milestone can consume it directly as requirements input
- CLI requirements (CLI-01 through CLI-05) confirmed fully implemented and verified — fix the Pending status in REQUIREMENTS.md as part of bookkeeping cleanup
- Publication-readiness is the key driver for next milestone priorities

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 21-retrospective-prospective*
*Context gathered: 2026-02-26*
