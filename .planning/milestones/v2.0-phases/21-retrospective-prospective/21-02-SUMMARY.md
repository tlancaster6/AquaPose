---
phase: 21-retrospective-prospective
plan: 02
subsystem: documentation
tags: [prospective, requirements-seed, accuracy, publication-readiness, bottleneck-analysis]

# Dependency graph
requires:
  - phase: 21-retrospective-prospective
    provides: 21-RETROSPECTIVE.md with 7 identified gaps feeding into this prospective
provides:
  - v2.1 prospective document with bottleneck analysis, 11 candidate requirements, and 3-phase suggested structure
  - Requirements seed for /gsd:new-milestone consumption
affects: [gsd:new-milestone, gsd:complete-milestone]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Prospective as requirements seed: candidate requirements with IDs, priorities, and dependencies ready for /gsd:new-milestone"
    - "Evaluation-first ordering: measurement infrastructure precedes accuracy improvements in phase structure"
    - "Bottleneck analysis as phase ordering rationale: upstream-first sequencing documented with dependency arguments"

key-files:
  created:
    - .planning/phases/21-retrospective-prospective/21-PROSPECTIVE.md

key-decisions:
  - "Evaluation infrastructure (CI, regression suite, measurement pipeline) ordered before any accuracy improvement work — cannot validate what you cannot measure"
  - "Segmentation (IoU 0.623) identified as primary bottleneck with highest upstream-to-downstream impact leverage"
  - "11 candidate requirements across 5 categories: evaluation infra, segmentation, tracking, reconstruction, CI/CD"
  - "3-phase structure: Phase 1 foundations (EVAL-01, CI-01, RECON-02), Phase 2 segmentation + benchmarks (SEG-01..03, EVAL-02..03, TRACK-01), Phase 3 downstream optimization (RECON-01, TRACK-02, CI-02)"
  - "Out of scope defined explicitly: no architectural changes, no new stages, no GUI, no real-time, no MOG2 backend validation"
  - "Document structured for direct /gsd:new-milestone consumption: IDs, priorities, dependencies, rationale, phase suggestions all provided"

patterns-established:
  - "Gaps-to-requirements flow: retrospective gaps section feeds directly into prospective candidate requirements"
  - "Bottleneck analysis before requirements: stage-by-stage assessment drives priority ordering"

requirements-completed: [CLI-01, CLI-02, CLI-03, CLI-04, CLI-05]

# Metrics
duration: 2min
completed: 2026-02-27
---

# Phase 21 Plan 02: Prospective Summary

**v2.1 prospective with 11 candidate requirements ordered by bottleneck impact — segmentation primary, evaluation-infrastructure-first phase structure seeding /gsd:new-milestone**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-27T03:12:52Z
- **Completed:** 2026-02-27T03:14:59Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Wrote 330-line prospective document with: milestone vision (publication readiness), stage-by-stage bottleneck analysis ordered by downstream impact, 11 candidate requirements across evaluation infra / segmentation / tracking / reconstruction / CI-CD categories, a 3-phase suggested structure, explicit out-of-scope section, and a How to Use section for `/gsd:new-milestone` consumption
- Incorporated all 7 gaps from 21-RETROSPECTIVE.md as candidate requirements or section rationale
- Established evaluation-infrastructure-first ordering with documented rationale (cannot validate improvements without measurement)

## Task Commits

Each task was committed atomically:

1. **Task 1: Write 21-PROSPECTIVE.md** - `5fa8e1e` (docs)

## Files Created/Modified

- `.planning/phases/21-retrospective-prospective/21-PROSPECTIVE.md` — v2.1 requirements seed document, 330 lines

## Decisions Made

- Evaluation infrastructure ordered before accuracy improvements — measurement precedes optimization as a strict dependency
- Stage bottleneck analysis ordered: Segmentation (primary) > Evaluation (enabling) > Detection (moderate) > Association (moderate, improves from segmentation) > Tracking (lower) > Reconstruction (lowest, depends on upstream)
- 3-phase structure reflects the bottleneck ordering: Phase 1 foundations, Phase 2 primary bottleneck, Phase 3 downstream optimization
- Candidate requirement IDs follow module-based prefixes (EVAL, SEG, TRACK, RECON, CI) for clarity in next milestone's REQUIREMENTS.md
- Out of scope defined explicitly to prevent scope creep at `/gsd:new-milestone` discuss-phase

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 21-PROSPECTIVE.md is complete and structured for direct `/gsd:new-milestone` consumption
- 21-RETROSPECTIVE.md (Plan 01) serves as the v2.0 Alpha completion narrative for `/gsd:complete-milestone`
- Phase 21 is fully complete — both plans executed, all requirements marked complete
- Ready for `/gsd:complete-milestone` to archive v2.0 Alpha and start `/gsd:new-milestone` for v2.1

## Self-Check: PASSED

- `.planning/phases/21-retrospective-prospective/21-PROSPECTIVE.md` — FOUND (330 lines)
- Commit `5fa8e1e` — FOUND

---
*Phase: 21-retrospective-prospective*
*Completed: 2026-02-27*
