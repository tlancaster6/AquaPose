# Phase 35: Codebase Cleanup - Context

**Gathered:** 2026-03-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove custom model implementations (U-Net, custom _PoseModel, SAM2 pseudo-labeler, MOG2 detector) and their associated training commands. Leave the backend orchestration layer intact — `segment_then_extract` and `direct_pose` survive as approaches, they just lose their custom model code. Stub in no-op YOLO replacements so the pipeline stays runnable. Consolidate surviving shared utilities.

**Critical clarification (from discussion):** CLEAN-03 does NOT mean "delete the midline backends." It means "remove the custom model implementations from within the backends." The segment_then_extract approach (segment → skeletonize → arc-length sample) and the direct_pose approach (keypoints → spline) both survive. Phase 37 wires in YOLO-seg and YOLO-pose as their new model providers. ROADMAP.md success criteria must be updated to reflect this.

</domain>

<decisions>
## Implementation Decisions

### Shared code triage
- Keep utilities if any non-removed code imports them; delete only truly orphaned code
- When preserving, consider whether utilities should be relocated for cleaner imports
- Extract reusable pieces from `segmentation/` package (e.g., crop helpers) to a shared location before deleting the rest
- Keep shared training infrastructure (base datasets, utilities) in `training/`; only remove `train_unet.py` and `train_pose.py`
- Opportunistic cleanup: also remove clearly dead adjacent code discovered during removal
- Verify whether top-level `reconstruction/`, `tracking/`, `visualization/` directories are legacy duplicates of their `core/` counterparts — remove if truly redundant and doable without major churn

### Two-pass structure
- **Pass 1 — Separate and delete**: Extract potentially reusable utilities, then delete removed modules
- **Pass 2 — Audit survivors**: For everything preserved: (1) judge keep-as-shared vs fold-into-callers, (2) delete if actually unneeded, (3) verify correctness. Consolidate survivors into a clear shared location with clean import paths.

### Test disposition
- Tests for removed code: delete with the code
- Integration/E2E tests referencing removed models: update to use remaining backends (not delete)
- Note: segment_then_extract and direct_pose are still valid backends — tests may just need model references updated

### Migration messaging
- Generic "unknown" errors, not migration hints (e.g., "Unknown detector_kind: 'mog2'. Available: yolo, yolo_obb")
- Strict config validation — unknown fields or invalid backend names raise errors at config load time, no silent ignoring
- Update `init-config` templates and example YAML files to remove references to deleted models

### Removal verification
- Tests must pass AND manual `aquapose run` smoke test with valid config
- Stub both `yolo_seg` and `yolo_pose` as no-op backends so the pipeline executes end-to-end without errors (empty/null midline results are acceptable)
- Atomic commits: one commit per CLEAN requirement for easy bisection

### ROADMAP.md corrections
- Update CLEAN-03 success criteria: "custom model code removed from midline backends" instead of "backends removed"
- Update any Phase 37 references that imply segment_then_extract/direct_pose don't exist

### Claude's Discretion
- No-op stub output format (empty Midline2D vs skip entirely) — pick based on what downstream stages handle gracefully
- Exact shared utility relocation target (e.g., `core/utils/` or similar)
- Order of removal within the atomic commit structure
- Whether top-level `reconstruction/`, `tracking/`, `visualization/` are actually legacy duplicates worth removing

</decisions>

<specifics>
## Specific Ideas

- "Every time you asked about delete vs preserve shared utilities, I leaned towards preserve" — bias toward keeping, then audit
- Two-pass approach was user-initiated: extract → delete → audit survivors → consolidate
- Pipeline must remain runnable at every step — no-op stubs ensure this
- User explicitly called out that segment_then_extract and direct_pose are NOT being removed as approaches — only their custom model code goes away

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 35-codebase-cleanup*
*Context gathered: 2026-03-01*
