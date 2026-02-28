# Phase 29: Guidebook Audit - Context

**Gathered:** 2026-02-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Audit and update GUIDEBOOK.md so it accurately reflects the shipped v2.1 codebase and documents v2.2 planned features. The guidebook is the architectural reference for future Claude sessions. No code changes — documentation only.

</domain>

<decisions>
## Implementation Decisions

### v2.2 Feature Depth
- Architecture-level overview for all v2.2 planned features — where each fits in the pipeline, what it replaces/extends, backend vs model distinction
- Include a concrete example showing how to register a new backend (e.g., adding YOLO-OBB as a detection model)
- Training CLI: just mention it exists as a planned v2.2 feature, details deferred to Phase 31
- Keypoint midline backend: describe the approach in detail:
  - 6 anatomical keypoints, each assigned a fixed `t` value in [0,1] calibrated from full skeletons (mean cumulative arc-length fractions)
  - Output is always `np.linspace(0, 1, N)` — point #k always means the same anatomical location
  - Fit spline using only observed keypoints and their fixed t values
  - Evaluate spline only for output t values within `[t_min_observed, t_max_observed]`; everything outside → NaN (no extrapolation)

### Planned vs Shipped Marking
- Inline parenthetical tags: `(v2.2)` after planned feature names — subtle, scannable
- No header banner or separate planned-features section — inline tags are self-explanatory
- Tags are NOT updated per-phase; batch update at v2.2 milestone end

### Structural Changes
- Source layout tree: reflect v2.1 actual layout only; let individual phase implementers decide new subdirectories
- Fully remove all stale v2.0 references (FishTracker, RANSAC centroid clustering, pre-reorder stage descriptions) — no historical preservation in main sections
- Update stage descriptions (Section 6) for v2.1 reorder AND add a backend registry subsection explaining how backends are registered/resolved from config
- Fold v2.2 material into existing sections — no new top-level sections

### Sections to Remove
- Delete Section 16 (Definition of Done) — roadmap has per-phase success criteria, guidebook shouldn't duplicate
- Delete Section 18 (Discretionary Items) — guidebook isn't the right place for it

### Milestone History
- Mark v2.1 Identity as shipped (2026-02-28) with a factual summary of what was done (pipeline reorder, OC-SORT tracking, Leiden association, refractive LUTs)
- Add v2.2 Backends as current milestone with 6-phase overview

### Claude's Discretion
- Exact wording of v2.1 shipped summary
- How to organize the backend registry subsection within existing sections
- Whether v2.2 milestone entry includes motivation or just scope/phases
- Section renumbering after DoD and Discretionary Items removal

</decisions>

<specifics>
## Specific Ideas

- Partial skeleton handling for keypoint backend was described in detail by the user — this is a core architectural decision, not just a note. Future sessions implementing Phase 33 should treat this as a locked contract.
- Backend registration example should be concrete enough that "a reader following GUIDEBOOK.md can correctly identify where to add a new backend" (success criterion #4)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 29-guidebook-audit*
*Context gathered: 2026-02-28*
