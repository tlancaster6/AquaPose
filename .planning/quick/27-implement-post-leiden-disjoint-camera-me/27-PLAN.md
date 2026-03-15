---
phase: 27-implement-post-leiden-disjoint-camera-merge
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/aquapose/core/association/clustering.py
  - tests/unit/core/association/test_clustering.py
autonomous: true
requirements: [QUICK-27]

must_haves:
  truths:
    - "Disjoint-camera cluster pairs from the same connected component are merged when mean cross-cluster score >= score_min"
    - "Merges are blocked when they would create must-not-link violations"
    - "Merges only happen when camera sets are fully disjoint"
    - "Existing clustering behavior is unchanged for non-split cases"
  artifacts:
    - path: "src/aquapose/core/association/clustering.py"
      provides: "_merge_disjoint_clusters helper and integration into cluster_tracklets"
      contains: "_merge_disjoint_clusters"
    - path: "tests/unit/core/association/test_clustering.py"
      provides: "Tests for disjoint-camera merge logic"
      contains: "test_disjoint_camera_merge"
  key_links:
    - from: "cluster_tracklets"
      to: "_merge_disjoint_clusters"
      via: "called after Leiden partition loop, before MNL enforcement"
      pattern: "_merge_disjoint_clusters"
---

<objective>
Add a post-Leiden disjoint-camera merge pass to fix fish splitting when Leiden partitions a high-camera-count fish into two sub-clusters with disjoint camera sets.

Purpose: 17 split events observed in a 32-chunk full run where fish visible in 6+ cameras get split into two groups because cameras form spatial sub-clusters. Cross-cluster pairwise scores are strong (0.3-0.87) but Leiden prefers two smaller communities.

Output: Updated clustering.py with `_merge_disjoint_clusters()` helper called from `cluster_tracklets()`, plus unit tests.
</objective>

<execution_context>
@/home/tlancaster6/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlancaster6/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/core/association/clustering.py
@tests/unit/core/association/test_clustering.py
@src/aquapose/core/association/types.py

<interfaces>
<!-- Key types the executor needs -->

From clustering.py:
```python
TrackletKey = tuple[str, int]  # (camera_id, track_id)

class ClusteringConfigLike(Protocol):
    score_min: float
    expected_fish_count: int
    leiden_resolution: float
```

From types.py:
```python
@dataclass(frozen=True)
class TrackletGroup:
    fish_id: int
    tracklets: tuple
    confidence: float | None = None
```

From test_clustering.py:
```python
@dataclass(frozen=True)
class MockClusteringConfig:
    score_min: float = 0.3
    expected_fish_count: int = 2
    leiden_resolution: float = 1.0

def _make_tracklet(camera_id, track_id, frames, centroid=(100.0, 100.0), status="detected", statuses=None) -> Tracklet2D
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Implement _merge_disjoint_clusters and integrate into cluster_tracklets</name>
  <files>src/aquapose/core/association/clustering.py, tests/unit/core/association/test_clustering.py</files>
  <behavior>
    - Test: Two clusters from same component with fully disjoint cameras and mean cross-score >= score_min are merged into one cluster
    - Test: Two clusters with overlapping cameras are NOT merged (cameras not disjoint)
    - Test: Two clusters with disjoint cameras but mean cross-score < score_min are NOT merged
    - Test: Merge is blocked when it would create a must-not-link violation
    - Test: Iterative merging works (3 clusters that can pairwise merge end up as 1)
    - Test: Existing two-fish test still passes unchanged (no regression)
  </behavior>
  <action>
1. Add `_merge_disjoint_clusters()` helper function in clustering.py (between the MNL section and the Leiden section, around line 88):

```python
def _merge_disjoint_clusters(
    clusters: dict[int, list[int]],
    sub_key_list: list[TrackletKey],
    scores: dict[tuple[TrackletKey, TrackletKey], float],
    must_not_link: set[frozenset[TrackletKey]],
    config: ClusteringConfigLike,
) -> dict[int, list[int]]:
```

Logic:
- Iterate until no more merges possible (converged flag)
- For each pair of cluster IDs (cid_a, cid_b) where cid_a < cid_b:
  - Get camera sets: `cam_a = {sub_key_list[m][0] for m in members_a}`, same for b
  - Skip if `cam_a & cam_b` is non-empty (not disjoint)
  - Check MNL: for each (ki, kj) cross-pair, if `frozenset({ki, kj}) in must_not_link`, skip this merge
  - Compute mean cross-cluster score: collect all `scores.get((ki, kj), 0) + scores.get((kj, ki), 0)` for ki in cluster A keys, kj in cluster B keys. Count nonzero entries. If count == 0, skip. Mean = total / count.
  - If mean >= config.score_min: merge B into A (extend A's member list, delete B), set converged=False, break inner loops to restart
- Log number of merges performed at DEBUG level

2. Integrate in `cluster_tracklets()`: Call `_merge_disjoint_clusters()` on the `clusters` dict right after the Leiden partition builds it (after line 178), before the MNL enforcement loop (line 181). Pass `sub_key_list`, `scores`, `must_not_link`, `config`.

3. Update `__all__` in clustering.py: `_merge_disjoint_clusters` is private (underscore prefix), so do NOT add to `__all__` or `__init__.py`.

4. Add tests in a new `TestDisjointCameraMerge` class in test_clustering.py:
  - `test_disjoint_camera_merge_basic`: 6 tracklets across 6 cameras, Leiden splits into 2 groups of 3 with disjoint cameras. Set up scores so cross-cluster mean >= 0.3. Assert single merged group with all 6 tracklets.
  - `test_disjoint_camera_merge_blocked_by_overlap`: 2 clusters sharing a camera are NOT merged.
  - `test_disjoint_camera_merge_blocked_by_low_score`: Disjoint cameras but mean cross-score < score_min, no merge.
  - `test_disjoint_camera_merge_blocked_by_mnl`: Disjoint cameras, good score, but MNL violation would result. No merge.
  - `test_disjoint_camera_merge_iterative`: 3 clusters with pairwise disjoint cameras and good scores all merge to 1.

To force Leiden to split in tests: use very high leiden_resolution (e.g. 5.0) so it over-partitions, or construct the score graph with two dense sub-cliques connected by weaker (but still >= score_min) cross edges. The simpler approach: test `_merge_disjoint_clusters` directly as a unit function by importing it and passing pre-built clusters dicts, bypassing Leiden entirely. This is cleaner and more reliable.

For the integration test through `cluster_tracklets`, use 6 cameras with 1 tracklet each. Build intra-subclique scores at 0.9 (cam_a/b/c pairwise, cam_d/e/f pairwise) and cross-subclique scores at 0.4. Use leiden_resolution=5.0 to encourage splitting. Assert the final result has 1 multi-tracklet group containing all 6 camera IDs.
  </action>
  <verify>
    <automated>cd /home/tlancaster6/Projects/AquaPose && hatch run test tests/unit/core/association/test_clustering.py -x -v</automated>
  </verify>
  <done>
    - _merge_disjoint_clusters helper exists and is called from cluster_tracklets after Leiden partition
    - All new tests pass: disjoint merge basic, blocked by overlap, blocked by low score, blocked by MNL, iterative
    - All existing clustering tests still pass (no regression)
    - Debug log message emitted when merges occur
  </done>
</task>

</tasks>

<verification>
All tests in test_clustering.py pass: `hatch run test tests/unit/core/association/test_clustering.py -x -v`
Full test suite has no regressions: `hatch run test`
</verification>

<success_criteria>
- cluster_tracklets() merges disjoint-camera Leiden sub-clusters when cross-cluster affinity meets threshold
- Merges respect MNL constraints and camera disjointness requirement
- 6+ new test cases cover merge, no-merge, and edge cases
- No regressions in existing test suite
</success_criteria>

<output>
After completion, create `.planning/quick/27-implement-post-leiden-disjoint-camera-me/27-SUMMARY.md`
</output>
