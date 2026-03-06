# Phase 25 Verification: Association Scoring and Clustering

## Requirements Verified

### ASSOC-01: Pairwise Association Scoring
**Status: PASS**

Evidence:
- `score_all_pairs()` in `src/aquapose/core/association/scoring.py` iterates all cross-camera tracklet pairs from adjacent cameras (via `camera_overlap_graph()`)
- `score_tracklet_pair()` computes ray-ray closest-point distance using `ForwardLUT.cast_ray()`, counts inlier frames below `ray_distance_threshold`
- Ghost-point penalty applied via `ghost_point_lookup()` from InverseLUT — penalizes 3D points visible to cameras with no supporting detection
- Early termination when first `early_k` frames have 0 inliers
- Overlap reliability weighting: `w = min(shared_frames, t_saturate) / t_saturate`
- Tests: `test_scoring.py` — 10 tests covering geometry, ghost penalty, adjacency filtering, score_min threshold

### ASSOC-02: Leiden Clustering and Fragment Merging
**Status: PASS**

Evidence:
- `cluster_tracklets()` in `src/aquapose/core/association/clustering.py` builds igraph weighted graph, splits into connected components, runs `leidenalg.find_partition()` with `RBConfigurationVertexPartition`
- `build_must_not_link()` identifies same-camera tracklet pairs with detection-backed temporal overlap — coasted-only overlap is NOT a constraint
- Must-not-link enforcement evicts lower-affinity tracklet to singleton
- `merge_fragments()` merges same-camera non-overlapping fragments within each cluster, with linear interpolation for gap frames tagged "interpolated"
- `AssociationStage` in `src/aquapose/core/association/stage.py` replaces AssociationStubStage, wiring LUT loading → scoring → clustering → fragment merging → `context.tracklet_groups`
- Graceful degradation: produces empty groups when LUTs unavailable
- Tests: `test_clustering.py` — 12 tests covering must-not-link, clustering, merging, gap limits, cross-camera isolation, integration

## Success Criteria Check

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Pairwise affinity from ray-ray distance with ghost penalty | PASS | `score_tracklet_pair()` + `score_all_pairs()` with LUT-based ray casting and ghost penalty |
| 2 | Leiden clustering with must-not-link constraints | PASS | `cluster_tracklets()` with connected components + Leiden + must-not-link enforcement |
| 3 | Same-camera fragment merging | PASS | `merge_fragments()` with gap interpolation and max_merge_gap limit |
| 4 | tracklet_groups in PipelineContext | PASS | `AssociationStage.run()` populates `context.tracklet_groups`; `expected_fish_count=9` configurable |

## Test Results

```
512 passed, 34 deselected, 64 warnings in 23.86s
```

All association tests pass. No regressions in existing test suite.

## Phase Verdict: PASS
