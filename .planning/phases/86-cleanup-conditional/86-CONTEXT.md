# Phase 86: Cleanup (Conditional) - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Address the 4 issues found during the Phase 85 code quality audit. Scope is fixed by the audit report (`85-AUDIT-REPORT.md` Section 3). No new features — only fixes to existing code quality issues.

</domain>

<decisions>
## Implementation Decisions

### Cross-chunk handoff bug (spline interpolation)
- Strip builders from `get_state()`/`from_state()` serialization entirely
- On restore, initialize empty builders for each active track — only carry KF states, track metadata (`time_since_update`, `hit_streak`, `detected_count`, `state`), OCR state (`pre_coast_x/P`, `obs_history`), and `next_track_id` counter
- Each chunk's `get_tracklets()` returns only that chunk's tracklet segments (downstream already handles per-chunk tracklets independently)
- Add a regression unit test: `get_state()`/`from_state()` roundtrip produces valid state, and a 2-chunk sequence doesn't produce duplicate frame indices
- No CLI smoke rerun needed — unit test is sufficient

### Dead code disposal
- **Delete:** `FishTrack`, `TrackState`, `TrackHealth` from `core/tracking/types.py` and `core/tracking/__init__.py` (~200 lines) — orphaned after OC-SORT removal, no consumers remain
- **Delete:** `_reproject_3d_midline` from `evaluation/viz/overlay.py` — defined but never called, genuinely dead
- **Keep:** `_triangulate_body_point` in `core/reconstruction/backends/dlt.py` — intentionally retained as scalar reference for the vectorized implementation (docstring explicitly references it)
- **Keep:** Synthetic trajectory methods (`loose_school`, `milling`, `streaming`, `FishTrajectoryState`) — valid test/dev tooling, exercised by tests

### Test directory rename
- Split `tests/unit/segmentation/` into correct directories matching the source layout:
  - `test_affine_crop.py` → `tests/unit/pose/`
  - `test_detector.py` → `tests/unit/detection/`
  - `test_dataset.py` → `tests/unit/training/`
- Merge into existing directories if they already exist
- Delete the `tests/unit/segmentation/` directory after move

### augment_count CLI wiring
- Wire `--augment-count` CLI option to `generate_variants()` via a new `n_variants` parameter
- Alternating C/S pattern: C+, S+, C-, S-, C+2, S+2, ... up to n_variants
- Each variant independently samples a random angle from the angle_range (consistent with current behavior)
- Default remains 4 (backward compatible)

### Claude's Discretion
- Internal implementation details of the alternating C/S generator
- Exact structure of the regression unit test for cross-chunk handoff
- Whether to create `__init__.py` files for new test directories

</decisions>

<specifics>
## Specific Ideas

No specific requirements — scope is fully defined by the Phase 85 audit report findings.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_KptTrackletBuilder`: The builder class in `keypoint_tracker.py` that accumulates per-frame data — needs `add_frame()` to work with empty state after restore
- `generate_variants()` in `elastic_deform.py`: Currently hardcoded 4-variant deform_specs list — needs refactoring to support variable count

### Established Patterns
- `get_state()`/`from_state()` pattern on `_SinglePassTracker` — serializes/deserializes tracker state for cross-chunk continuity
- `CarryForward` object carries per-camera track state through the orchestrator's outer loop
- Test directory structure mirrors source layout: `tests/unit/<module>/`

### Integration Points
- `TrackingStage.run()` (stage.py line ~120) uses `enumerate(detections)` for 0-based frame indices per chunk
- `orchestrator.py` calls `get_state()` after each chunk and `from_state()` before the next
- `data_cli.py` import_cmd passes `augment_angle_range` to `generate_variants()` but not `augment_count`
- `core/tracking/__init__.py` and `__all__` export `FishTrack`, `TrackState`, `TrackHealth` — must be cleaned on deletion

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 86-cleanup-conditional*
*Context gathered: 2026-03-11*
