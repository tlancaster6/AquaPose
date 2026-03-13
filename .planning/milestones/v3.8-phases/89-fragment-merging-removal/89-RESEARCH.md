# Phase 89: Fragment Merging Removal - Research

**Researched:** 2026-03-11
**Domain:** Python code deletion — association pipeline, config dataclasses, protocol types
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Config backward compatibility:** Strict error on unknown YAML fields (already the default via `_filter_fields`). No YAML config files currently reference `max_merge_gap`, so removal is safe. Remove `max_merge_gap` from `AssociationConfig` in `engine/config.py`.
- **"interpolated" status cleanup:** Clean up all references to the `"interpolated"` frame_status value since nothing will produce it after deletion. Remove from `Tracklet2D` docstrings — drop the fragment-merging example, just state keypoints can be `None` without giving a specific scenario. Remove from `keypoint_tracker.py` comments. Remove from `clustering.py` references.
- **Deletion boundary:** Delete `merge_fragments`, `_merge_cam_fragments`, `_try_merge_pair` from `clustering.py`. Delete the entire "Fragment merging (SPECSEED Step 4)" section. Remove `max_merge_gap` from `ClusteringConfigLike` protocol. Remove `merge_fragments` from `__init__.py` re-exports and `__all__`. Remove `merge_fragments` call from `stage.py`. Update `clustering.py` module docstring to remove fragment merging mention. Delete fragment-merging tests from `test_clustering.py`.

### Claude's Discretion

- Exact wording of updated docstrings
- Whether to keep or simplify the `ClusteringConfigLike` protocol if it becomes trivially small
- Any minor cleanup of surrounding code comments that reference the removed functionality

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CLEAN-01 | Fragment merging code and max_merge_gap config field removed | Complete deletion map documented below; all touch points identified in source |
</phase_requirements>

## Summary

Phase 89 is a pure deletion phase: remove `merge_fragments` and its two private helpers from `clustering.py`, remove the `max_merge_gap` field from `AssociationConfig` and `ClusteringConfigLike`, remove the call from `stage.py`, remove the re-export from `__init__.py`, delete the `TestMergeFragments` test class, and clean up residual references to `"interpolated"` frame_status across three files.

No new functionality is introduced. The pipeline continues to run end-to-end because downstream stages consume `TrackletGroup` objects produced by `cluster_tracklets`, which is not touched. The `merge_fragments` call was a post-clustering step; removing it does not break the data contract with the refinement stage or any later stage.

The only subtlety is the `ClusteringConfigLike` protocol. After removing `max_merge_gap`, the protocol has three remaining fields: `score_min`, `expected_fish_count`, `leiden_resolution`. It remains useful — `cluster_tracklets` consumes all three — so it should be kept but shrunk, not deleted.

**Primary recommendation:** Delete all listed symbols in a single wave; run `hatch run test` to confirm the pipeline-level smoke test passes and no imports break.

## Standard Stack

No new libraries. This phase uses only existing project tooling.

### Tooling
| Tool | Purpose |
|------|---------|
| `hatch run test` | Run unit tests (excludes @slow) — primary verification |
| `hatch run check` | Lint + typecheck — catches dangling imports and protocol mismatches |

## Architecture Patterns

### Files to Modify (complete inventory)

```
src/aquapose/core/association/clustering.py   # primary deletion target
src/aquapose/core/association/stage.py        # remove merge_fragments call
src/aquapose/core/association/__init__.py     # remove re-export
src/aquapose/engine/config.py                 # remove AssociationConfig.max_merge_gap
src/aquapose/core/tracking/types.py           # update Tracklet2D docstring
src/aquapose/core/tracking/keypoint_tracker.py # update comment
tests/unit/core/association/test_clustering.py # delete TestMergeFragments class
```

### Pattern: Protocol Shrinkage

`ClusteringConfigLike` currently declares four fields:
```python
score_min: float
expected_fish_count: int
leiden_resolution: float
max_merge_gap: int   # DELETE THIS
```

After deletion it becomes three fields. `cluster_tracklets` only reads `score_min`, `expected_fish_count`, and `leiden_resolution` from config — confirmed by reading the implementation. The protocol docstring also lists `max_merge_gap` and must be updated.

### Pattern: `__init__.py` Sync

Per project rules, removing a public symbol requires removing it from both the `import` block and `__all__`. Current state in `core/association/__init__.py`:

```python
from aquapose.core.association.clustering import (
    ClusteringConfigLike,
    build_must_not_link,
    cluster_tracklets,
    merge_fragments,          # DELETE
)
# ...
__all__ = [
    # ...
    "merge_fragments",        # DELETE
    # ...
]
```

### Pattern: Test Class Deletion

The entire `TestMergeFragments` class (lines 235-328 in `test_clustering.py`) is deleted. The import of `merge_fragments` at the top of the test module (line 17) and the `max_merge_gap` field on `MockClusteringConfig` (line 32) are also removed.

`MockClusteringConfig` is also used by `TestClusterTracklets`. After removing `max_merge_gap`, the mock must still satisfy `ClusteringConfigLike`. Since the protocol will no longer require `max_merge_gap`, removing it from `MockClusteringConfig` is correct and leaves the mock aligned with the updated protocol.

## Don't Hand-Roll

No custom solutions needed. This is deletion-only.

## Common Pitfalls

### Pitfall 1: Leaving the `build_must_not_link` docstring stale

**What goes wrong:** `build_must_not_link` has a docstring line: "Coasted-only overlap is NOT a constraint (those are fragment merge candidates)." After deletion, fragment merging no longer exists, so calling something a "fragment merge candidate" is misleading.

**How to avoid:** Update that docstring line to simply state that coasted-only overlap does not create a must-not-link constraint, without mentioning fragment merging.

**Warning signs:** Any docstring line containing "fragment merge" that is not inside the deleted functions themselves.

### Pitfall 2: Forgetting the `clustering.py` module docstring

**What goes wrong:** The module-level docstring reads "Leiden-based tracklet clustering with must-not-link constraints and fragment merging." and "SPECSEED Steps 2-4". After deletion, fragment merging (Step 4) is gone.

**How to avoid:** Update module docstring to describe only what remains: Leiden clustering and must-not-link enforcement (SPECSEED Steps 2-3).

### Pitfall 3: `MockClusteringConfig` still satisfying the protocol

**What goes wrong:** `MockClusteringConfig` in `test_clustering.py` must keep satisfying the updated `ClusteringConfigLike`. Since the protocol no longer requires `max_merge_gap`, removing it from the mock is correct. If removed from the protocol but kept on the mock, no error occurs (extra fields are fine for structural protocols). If kept on the protocol but removed from `AssociationConfig`, runtime `isinstance` checks would fail.

**How to avoid:** Verify: after changes, `AssociationConfig` still satisfies `ClusteringConfigLike` (all protocol fields present on the dataclass). Run `hatch run typecheck` to catch protocol mismatches statically.

### Pitfall 4: `stage.py` import cleanup

**What goes wrong:** The `stage.py` local import block reads:
```python
from aquapose.core.association.clustering import (
    build_must_not_link,
    cluster_tracklets,
    merge_fragments,
)
```
Removing the call to `merge_fragments` without removing it from the import causes an unused import lint error (caught by ruff).

**How to avoid:** Remove `merge_fragments` from both the import and the call site in `stage.py`.

### Pitfall 5: `_filter_fields` strict rejection on existing YAML configs

**What goes wrong:** Confirmed by reading `_filter_fields`: any YAML file that sets `association.max_merge_gap` will raise `ValueError` after removal, blocking pipeline runs on configs that have it.

**How to avoid:** This is already verified safe — no `.yaml` files in the repo reference `max_merge_gap`. However, the plan should note that users with custom project YAML configs must remove this field.

## Code Examples

### Current `stage.py` call site (to be deleted)
```python
# Source: src/aquapose/core/association/stage.py:101-102
# Step 4: Merge fragments
groups = merge_fragments(groups, self._config.association)
```

### Updated `ClusteringConfigLike` after removal
```python
@runtime_checkable
class ClusteringConfigLike(Protocol):
    """Structural protocol for clustering configuration.

    Satisfied by ``AssociationConfig`` from ``engine.config`` without import.

    Attributes:
        score_min: Minimum affinity score for an edge.
        expected_fish_count: Expected number of fish clusters.
        leiden_resolution: Leiden resolution parameter.
    """

    score_min: float
    expected_fish_count: int
    leiden_resolution: float
```

### Updated `Tracklet2D.frame_status` docstring (target)
```
frame_status: Per-frame detection status, one per entry in ``frames``.
    Each value is ``"detected"`` (directly observed) or ``"coasted"``
    (position interpolated during a missed detection).
    Type: ``tuple[str, ...]``
```
Remove the reference to `keypoints can be None` when sourced from fragment merging; instead simply state keypoints can be `None` when keypoint data is unavailable.

### Updated `Tracklet2D.keypoints` docstring (target)
```
keypoints: Per-frame keypoint positions, shape ``(T, K, 2)``, float32.
    ``None`` when keypoint data is unavailable.
```
Drop the parenthetical "(e.g. tracklets from fragment merging)".

## State of the Art

| Old | Current after Phase 89 | Impact |
|-----|------------------------|--------|
| `merge_fragments` post-clustering step | Deleted | Pipeline output from `cluster_tracklets` feeds directly to refinement |
| `AssociationConfig.max_merge_gap: int = 30` | Field deleted | YAML configs with this key will raise `ValueError` on load |
| `"interpolated"` as a valid frame_status value | Never produced | Only `"detected"` and `"coasted"` remain as valid statuses |

## Open Questions

None — the deletion boundary is fully specified in CONTEXT.md and verified against the source.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (via hatch) |
| Config file | `pyproject.toml` |
| Quick run command | `hatch run test` |
| Full suite command | `hatch run test` (same; slow tests excluded by marker) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CLEAN-01 | `merge_fragments` absent from `clustering.py` and `__init__.py` | unit (import check) | `hatch run test` | ✅ (existing tests; TestMergeFragments will be deleted) |
| CLEAN-01 | `max_merge_gap` absent from `AssociationConfig` | unit (config load) | `hatch run test` | ✅ (existing config tests) |
| CLEAN-01 | Pipeline runs end-to-end without error | integration smoke | `hatch run test` (TestAssociationStage.test_produces_groups_graceful_degradation) | ✅ |

### Sampling Rate
- **Per task commit:** `hatch run test`
- **Phase gate:** `hatch run check && hatch run test`

### Wave 0 Gaps

None — existing test infrastructure covers all phase requirements. No new test files needed; the plan deletes tests rather than adding them.

## Sources

### Primary (HIGH confidence)
- Direct source inspection: `src/aquapose/core/association/clustering.py` — full function inventory
- Direct source inspection: `src/aquapose/core/association/stage.py` — call site
- Direct source inspection: `src/aquapose/core/association/__init__.py` — export list
- Direct source inspection: `src/aquapose/engine/config.py` — `AssociationConfig` field list
- Direct source inspection: `src/aquapose/core/tracking/types.py` — `Tracklet2D` docstring
- Direct source inspection: `tests/unit/core/association/test_clustering.py` — test class scope
- Grep over all `.yaml` files in repo — confirmed zero references to `max_merge_gap`
- `.planning/phases/89-fragment-merging-removal/89-CONTEXT.md` — user decisions

## Metadata

**Confidence breakdown:**
- Deletion scope: HIGH — all symbols located and verified in source
- Protocol safety: HIGH — `cluster_tracklets` implementation verified not to use `max_merge_gap`
- YAML safety: HIGH — grep confirmed no YAML configs in repo use `max_merge_gap`
- Test impact: HIGH — `TestMergeFragments` class fully mapped, `MockClusteringConfig` field relationship verified

**Research date:** 2026-03-11
**Valid until:** Until source changes — stable codebase, no external dependencies
