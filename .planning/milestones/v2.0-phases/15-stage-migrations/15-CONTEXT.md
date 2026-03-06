# Phase 15: Stage Migrations - Context

**Gathered:** 2026-02-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Port all 5 computation stages (Detection, Midline, Cross-View Association, Tracking, Reconstruction) as pure Stage Protocol implementors. Each stage lives in `core/`, satisfies the Stage Protocol via structural typing without importing `engine/`, and is wired into PosePipeline. Old v1.0 code is left in place — cleanup is Phase 16.

</domain>

<decisions>
## Implementation Decisions

### Backend scope
- **Detection:** YOLO only as the model-based backend. MOG2 deferred — not worth porting now
- **Midline:** Segment-then-extract (U-Net → skeletonize → BFS) fully implemented. Direct pose estimation backend added as a stub (NotImplementedError) to prove the backend registry pattern
- **Association:** Single backend — RANSAC centroid clustering. Backend registry still available for future alternatives
- **Tracking:** Single backend — Hungarian 3D matching with population constraint. No stubs for alternatives
- **Reconstruction:** Both triangulation AND curve optimizer backends fully implemented. Both have working v1.0 code to port

### V1.0 behavior preservation
- Preserve v1.0 behavior exactly during porting — no bug fixes during migration
- Maintain a centralized bug ledger at `.planning/phases/15-stage-migrations/15-BUG-LEDGER.md` tracking known quirks preserved for later fixing (feeds into Phase 16)
- Full internal refactor: rewrite internals to use typed dataclasses, clean signatures, new conventions — clean-room rebuild per guidebook
- Extract hardcoded thresholds and magic numbers to frozen config parameters during porting

### Code layout
- Stage implementations live in `core/` — they are computation modules that happen to satisfy the Stage Protocol via structural typing
- Protocol definition stays in `engine/stages.py` — stages never import from `engine/`
- Pipeline in `engine/` imports from `core/`, discovers Protocol satisfaction, and orchestrates
- New stage-aligned module paths: `core/detection/`, `core/midline/`, `core/association/`, `core/tracking/`, `core/reconstruction/`
- Old v1.0 module code stays in place until Phase 16 legacy cleanup
- Consistent internal layout per stage module:
  - `__init__.py` — public API
  - `backends/` — backend implementations
  - `types.py` — stage-specific types

### Model and weight loading
- Models load eagerly at stage construction (before `pipeline.run()`). Fail-fast if weights missing
- Stages cache the loaded model — logically stateless but may cache expensive initialization per guidebook
- Weight paths resolved from stage config only — no magic resolution, no environment variable fallbacks. Wrong path = clear error
- Load and trust: confirm file exists and loads without error, no smoke-test inference during construction
- Device placement specified in stage config (`device` field, default: 'cuda' if available, else 'cpu')

### Claude's Discretion
- Exact backend registry pattern implementation
- Internal function decomposition within each stage
- Stage construction parameter naming
- Test fixture design for interface tests
- Error message wording

</decisions>

<specifics>
## Specific Ideas

- "Stage implementations live in core/. The Protocol definition lives in engine/stages.py. A stage implementation is a computation module that happens to satisfy the Stage Protocol via structural typing — it doesn't import anything from engine/." — user's exact architectural description
- Import discipline is strict and one-way: `core/ → nothing`, `engine/ → core/`, `cli/ → engine/`
- The guidebook is the single source of truth for architectural decisions

</specifics>

<deferred>
## Deferred Ideas

- MOG2 detection backend — could be added as a second backend later
- Direct pose estimation midline backend (beyond stub) — needs a trained model first
- Additional association/tracking backend alternatives
- Bug fixes for preserved v1.0 quirks — Phase 16

</deferred>

---

*Phase: 15-stage-migrations*
*Context gathered: 2026-02-25*
