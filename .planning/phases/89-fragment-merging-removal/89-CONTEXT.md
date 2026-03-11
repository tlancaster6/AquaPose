# Phase 89: Fragment Merging Removal - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Delete the `merge_fragments` step from the association pipeline and the `max_merge_gap` config field. The pipeline must still run end-to-end without errors. This is a pure code removal phase — no new functionality is added.

</domain>

<decisions>
## Implementation Decisions

### Config backward compatibility
- Strict error on unknown YAML fields (already the default behavior via `_filter_fields`)
- No YAML config files currently reference `max_merge_gap`, so removal is safe
- Remove `max_merge_gap` from `AssociationConfig` in `engine/config.py`

### "interpolated" status cleanup
- Clean up all references to the `"interpolated"` frame_status value since nothing will produce it after deletion
- Remove from `Tracklet2D` docstrings — drop the fragment-merging example, just state keypoints can be `None` without giving a specific scenario
- Remove from `keypoint_tracker.py` comments
- Remove from `clustering.py` references

### Deletion boundary
- Delete `merge_fragments`, `_merge_cam_fragments`, `_try_merge_pair` from `clustering.py`
- Delete the entire "Fragment merging (SPECSEED Step 4)" section
- Remove `max_merge_gap` from `ClusteringConfigLike` protocol (only `cluster_tracklets` consumes it, and it doesn't use `max_merge_gap`)
- Remove `merge_fragments` from `__init__.py` re-exports and `__all__`
- Remove `merge_fragments` call from `stage.py`
- Update `clustering.py` module docstring to remove fragment merging mention
- Delete fragment-merging tests from `test_clustering.py`

### Claude's Discretion
- Exact wording of updated docstrings
- Whether to keep or simplify the `ClusteringConfigLike` protocol if it becomes trivially small
- Any minor cleanup of surrounding code comments that reference the removed functionality

</decisions>

<specifics>
## Specific Ideas

No specific requirements — this is a clean deletion with well-defined success criteria from the roadmap.

</specifics>

<code_context>
## Existing Code Insights

### Files to Modify
- `src/aquapose/core/association/clustering.py`: Delete ~200 lines (merge_fragments + helpers), update module docstring, update protocol, update `__all__`
- `src/aquapose/core/association/stage.py:102`: Remove `merge_fragments()` call
- `src/aquapose/core/association/__init__.py`: Remove re-export and `__all__` entry
- `src/aquapose/engine/config.py:167`: Remove `max_merge_gap: int = 30`
- `src/aquapose/core/tracking/types.py`: Update Tracklet2D docstrings
- `src/aquapose/core/tracking/keypoint_tracker.py`: Update comments
- `tests/unit/core/association/test_clustering.py`: Delete merge_fragments tests

### Established Patterns
- Config loading already strictly rejects unknown YAML fields via `_filter_fields` — removing a field from the dataclass automatically makes it an error in YAML
- Core protocols (`ClusteringConfigLike`) mirror engine config fields — protocol must stay in sync with what consumers actually use

### Integration Points
- `AssociationStage.run()` in `stage.py` calls `merge_fragments` as a post-clustering step — removing the call is the pipeline integration change
- No downstream stages depend on fragment merging output specifically — they consume `TrackletGroup` which is produced by `cluster_tracklets` before merging

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 89-fragment-merging-removal*
*Context gathered: 2026-03-11*
