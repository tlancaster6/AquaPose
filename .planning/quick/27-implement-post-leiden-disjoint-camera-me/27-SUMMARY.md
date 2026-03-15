---
phase: 27-implement-post-leiden-disjoint-camera-merge
plan: "01"
subsystem: association
tags: [clustering, leiden, disjoint-camera, fish-splitting]
dependency_graph:
  requires: []
  provides: [disjoint-camera-merge-pass]
  affects: [cluster_tracklets, association-stage]
tech_stack:
  added: []
  patterns: [iterative-convergence-loop, protocol-based-config]
key_files:
  modified:
    - src/aquapose/core/association/clustering.py
    - tests/unit/core/association/test_clustering.py
key_decisions:
  - "Merge pass runs after Leiden partition, before MNL enforcement — so merges can be immediately checked against MNL constraints"
  - "Iterative convergence (restart after each merge) ensures transitive merges work correctly for 3+ cluster chains"
  - "Score averaging only over cross-cluster pairs with nonzero scores — avoids diluting mean with absent edges"
metrics:
  duration_seconds: 152
  completed_date: "2026-03-15"
  tasks_completed: 1
  files_modified: 2
---

# Quick Task 27: Implement Post-Leiden Disjoint-Camera Merge

**One-liner:** Post-Leiden merge pass using camera-set disjointness + mean cross-cluster affinity threshold to fix fish splitting in high-camera-count scenarios.

## What Was Built

Added `_merge_disjoint_clusters()` helper to `clustering.py` and integrated it into `cluster_tracklets()`. The function runs immediately after the Leiden partition loop builds the `clusters` dict (per connected component), before the must-not-link enforcement pass.

### Algorithm

Iterative convergence loop:
1. For each pair of clusters (cid_a, cid_b):
   - Skip if camera sets overlap (not disjoint)
   - Skip if any cross-pair (ki, kj) is in `must_not_link`
   - Compute mean cross-cluster score over pairs with nonzero affinity; skip if no pairs or mean < `score_min`
   - Merge B into A, delete B, restart loop
2. Log merged count at DEBUG level

### Integration Point

```python
# In cluster_tracklets(), after building clusters dict from partition.membership:
clusters = _merge_disjoint_clusters(
    clusters, sub_key_list, scores, must_not_link, config
)
# MNL enforcement follows immediately after
```

## Tests Added

`TestDisjointCameraMerge` class in `test_clustering.py` — 6 new tests:

| Test | Coverage |
|------|----------|
| `test_disjoint_camera_merge_basic` | 2 clusters, disjoint cameras, good score → merge to 1 |
| `test_disjoint_camera_merge_blocked_by_overlap` | Shared camera → no merge |
| `test_disjoint_camera_merge_blocked_by_low_score` | Disjoint cameras but score < score_min → no merge |
| `test_disjoint_camera_merge_blocked_by_mnl` | MNL constraint present → no merge |
| `test_disjoint_camera_merge_iterative` | 3 clusters → all merge to 1 |
| `test_disjoint_camera_merge_integration_via_cluster_tracklets` | End-to-end: 6-camera fish split by Leiden (resolution=5.0) merged back |

All 1214 non-skipped tests pass (no regressions).

## Deviations from Plan

None — plan executed exactly as written. Pre-commit ruff lint required two minor fixes (RUF015: `next(iter(...))` instead of `list(...)[0]`; B905: `strict=True` on `zip()`).

## Self-Check: PASSED

- `_merge_disjoint_clusters` exists in `clustering.py`: confirmed
- `cluster_tracklets` calls `_merge_disjoint_clusters` after Leiden partition: confirmed
- All new tests present in `test_clustering.py`: confirmed
- Commit `3faaf03` exists: confirmed
