---
status: resolved
trigger: "association produces 45/49 singleton groups instead of linking tracklets across views"
created: 2026-03-02T00:00:00Z
updated: 2026-03-03T00:00:00Z
---

## Current Focus

RESOLVED. The original diagnosis ("thresholds too tight") was wrong. The 2D grid
sweep over ray_distance_threshold x score_min (56 combos) showed that wider
thresholds make things *worse*, not better. The high singleton count was
misinterpreted — it reflects leftover single-camera tracklets, not a failure to
recover fish. With tuned params, 7/9 target fish are recovered as multi-camera
groups (3-5 cameras each).

## Key Correction: Singleton Rate Is Misleading

The original alarm ("45/49 singletons!") conflated tracklet count with fish count.
Actual per-frame counts (from tuned-param fixture, 100 frames):
- 44 tracklets/frame across all cameras
- 27 fish groups/frame (vs 9 actual fish — ~18 extra are singletons)
- Cameras per fish-group: 1 cam 70%, 2 cam 8%, 3 cam 12%, 4 cam 10%
- Mean 1.6 cameras per group

Of the 13 cameras in the rig, only 8 actually see fish:
- e3v83ee, e3v83ef: never detect fish (expected — no view of tank interior)
- e3v82f9, e3v83e9: persistent false-positive on inanimate objects; their
  tracklets correctly remain as singletons (never associate). This is good.
- The remaining 8 cameras produce all real associations.

The ~18 singleton groups per frame are a mix of false-positive tracklets
(correctly isolated) and leftover single-camera views of real fish. A 70%
singleton rate by observation count still corresponds to 7/9 = 78% fish recovery.

Analysis of 15 sampled frames confirmed always-singleton tracklets are NOT short
fragments — 11 persist all 15 frames, 4 for 14 frames, 4 for 13 frames. These
are real fish tracked stably in one camera whose views simply can't find
cross-view partners.

## Why Wider Thresholds Don't Help

The 2D grid result:
```
              0.03    0.05    0.08     0.1    0.15     0.2    0.25     0.3
      0.02      8%     10%      9%      9%      5%      6%     18%     17%
      0.03     16%     22%    22%*     22%     17%     14%     17%     16%
      0.04     16%     16%      7%     21%     16%     16%     11%     13%
      0.06      4%      4%      4%      4%      4%      4%      8%      9%
      0.08      3%      3%      3%      3%      3%      7%      2%      5%
       0.1     10%     10%     10%     11%      5%      7%      8%      9%
      0.15      8%      4%      8%      4%      4%      8%      4%      8%
```

Yield peaks at ray_distance_threshold=0.03 and collapses at wider values. Wider
thresholds admit more false matches (wrong fish paired across cameras), which
contaminate Leiden clusters, trigger must-not-link evictions ("Detection-backed
overlap" flood), and produce bad triangulations. The problem is
**signal-to-noise** — correct and incorrect pairs have similar ray-ray distances
in a densely populated tank — not threshold miscalibration.

## Tuning Results (2026-03-03)

Winning params from tune_association.py sweep:
```yaml
ray_distance_threshold: 0.03
score_min: 0.08
eviction_reproj_threshold: 0.03
leiden_resolution: 1.5
early_k: 20
```

Baseline vs Winner:
- Fish reconstructed: 78 -> 106 (+28)
- Yield: 15.7% -> 23.9% (+8.2%)
- Mean error: 12.32 -> 10.08 px (-2.24)
- Singleton rate: 80.1% -> 72.0% (7/9 fish recovered)

## Symptoms (original)

expected: ~9 fish form multi-view groups (3-7 cameras each) across 7+ cameras
actual: 45/49 groups are singletons; only 4 multi-view groups; 2 have duplicate camera IDs; fish 7 and 15 fail to produce correspondences
errors: no runtime errors
reproduction: run pipeline on YH dataset for 100 frames
started: current state of association algorithm

## Eliminated

- hypothesis: "Clustering/Leiden logic itself is wrong"
  evidence: cluster_tracklets correctly handles connected components. If no edges exist (scores all below score_min), tracklets become isolated nodes in the graph and are returned as singletons. The Leiden algorithm is not the issue.
  timestamp: 2026-03-02T00:10:00Z

- hypothesis: "Ghost penalty is the primary cause of singletons"
  evidence: Ghost penalty is a multiplicative penalty on the score, but it only applies to INLIER frames (frames where ray-ray distance < threshold). If there are no inliers (all frames miss the threshold), ghost_ratios list is empty and ghost penalty is 0.0. The inlier fraction f=0 itself kills the score. Ghost penalty amplifies the problem but ray_distance_threshold is the primary gatekeeper.
  timestamp: 2026-03-02T00:15:00Z

- hypothesis: "Duplicate camera IDs cause refinement to produce corrupted consensus"
  evidence: Same-camera tracklets in a group are temporally non-overlapping (must-not-link prevents detected-frame overlap). So in any given frame, at most ONE tracklet from each camera is active. _compute_frame_consensus only gets one ray per camera per frame. No degenerate parallel rays.
  timestamp: 2026-03-02T00:18:00Z

- hypothesis: "Thresholds too tight — relaxing ray_distance_threshold to 0.05-0.08m will unlock singletons"
  evidence: 2D grid sweep showed yield DECREASES at wider thresholds (0.06-0.15 → 3-10% yield vs 22% at 0.03). False matches overwhelm the clustering at wider thresholds. The discrimination limit is fundamental to bbox-centroid ray-ray geometry, not threshold miscalibration.
  timestamp: 2026-03-03T00:00:00Z

## Evidence

- timestamp: 2026-03-02T00:05:00Z
  checked: scoring.py score_tracklet_pair() - inlier classification logic
  found: Inlier classification: ray-ray closest-point distance < ray_distance_threshold (default 0.03m = 3cm). For a real fish pair: calibration error + LUT interpolation error + centroid extraction noise + refraction model error all contribute to ray-ray distance. If combined error budget exceeds 3cm, ALL frames are "outliers" (inlier_count=0). Early termination then fires at frame 10 (early_k=10): if first 10 shared frames all miss, return 0.0 immediately. This explains rapid failure for most pairs.
  implication: ray_distance_threshold=0.03m is the primary gatekeeper. If true ray-ray distances for this calibration are 0.03-0.08m, nearly all pairs score 0 and trigger early termination.

- timestamp: 2026-03-02T00:06:00Z
  checked: scoring.py score formula: score = f * (1 - mean_ghost) * w
  found: With f=0 (no inliers), score=0 regardless of ghost. Ghost only matters when f>0. For the 4 groups that DID form, f>0, confirming some camera pairs have ray-ray distance < 0.03m. But most camera pairs apparently exceed 0.03m.
  implication: The 4 multi-view groups formed because those specific camera pairs happened to have sub-3cm ray convergence. Other cameras have slightly worse calibration or larger refraction errors.

- timestamp: 2026-03-03T00:00:00Z
  checked: tune_association.py 2D grid sweep results (56 combos)
  found: Yield peaks at ray_distance_threshold=0.03 (22%) and drops dramatically at 0.06+ (3-10%). Wider thresholds admit false matches that contaminate clusters and trigger must-not-link evictions.
  implication: The discrimination problem is fundamental — correct and incorrect pairs have overlapping ray-ray distance distributions in a densely populated tank. No single threshold cleanly separates them.

- timestamp: 2026-03-03T00:00:00Z
  checked: Per-fish singleton analysis across 15 sampled frames
  found: 38 always-singleton fish, but 11 of them are present in all 15 frames and most are 10+ frames. Only 7 singletons are short (<=3 frames). Meanwhile 7 multi-camera fish recovered with 3-5 cameras each, all present 14-15 frames.
  implication: The 72% singleton rate by tracklet count is misleading. It actually represents 7/9 fish recovered (78%). The singletons are leftover views from cameras that can't find partners, not lost fish.

- timestamp: 2026-03-03T00:00:00Z
  checked: Per-camera singleton breakdown
  found: Of 13 cameras, only 8 see fish. e3v83ee/e3v83ef have no view of the tank. e3v82f9/e3v83e9 have persistent false-positive detections on inanimate objects — their tracklets correctly remain as singletons (never associate with real fish). The 8 real cameras (e3v829d, e3v82e0, e3v831e, e3v832e, e3v8334, e3v83eb, e3v83f0, e3v83f1) produce all real associations. e3v82e0 is most reliable (99% multi-cam).
  implication: Some singletons are correctly-isolated false positives, not missed associations.

- timestamp: 2026-03-03T00:00:00Z
  checked: Trail video label occlusion — fish 13 invisible in tracklet_trails
  found: Fish 13 (multi-cam group) and fish 32 (singleton) share the exact same centroid (1160, 983) in e3v83eb at frame 50. Fish 32 is a tracking fragment from the same physical fish — OC-SORT split the fish into two tracklets, only one joined the multi-cam group. The trail renderer draws both labels at the same pixel, so "32" overwrites "13". This means singleton tracklets can obscure multi-cam group labels in the trail visualization when they originate from the same physical fish.
  implication: Some singletons are not distinct physical entities but tracking fragments of already-associated fish. The singleton count overstates the number of unassociated physical fish.

- timestamp: 2026-03-03T00:00:00Z
  checked: DLT threshold sweep (tune_threshold.py, 20 values 5-100px on tuned-association fixture)
  found: |
    Best threshold=10px: 74/403 yield (18%), 2.91px mean error, 26.95px max error.
    DLT at 10px vs baseline triangulation: half the mean error (2.91 vs 6.25px),
    third the max error (27 vs 85px), but 18% vs 22% yield.
    Fish 13 (3-4 cameras, present all 15 frames) is MISSING at threshold=10px
    because its reprojection error (~10.5px) sits just above the cutoff.
    DLT requires only 2 cameras (not 3) — the minimum is in
    _triangulate_body_point line 396. Fish 13 fails because its body points
    all exceed the 10px outlier threshold, not due to insufficient cameras.
    Tier 2 stability dramatically better with DLT: fish 9 displacement drops
    from 1739mm (baseline) to 46mm (DLT threshold=10).
  implication: The outlier threshold controls a precision-recall tradeoff. 10px is optimal for the score metric (yield * mean_error). Fish 13 would require ~15px threshold to recover.

## Resolution

root_cause: |
  The original diagnosis was WRONG. The high singleton count does not indicate
  broken association — it reflects the normal surplus of single-camera tracklets
  in a 13-camera rig where each fish associates with 3-4 cameras.

  The actual performance is 7/9 target fish recovered as multi-camera groups.
  The 2 missing fish are likely only visible in cameras without sufficient
  geometric overlap to discriminate them from neighbors.

  The scoring kernel's discrimination is limited by bbox-centroid ray-ray
  geometry in a densely-packed tank. This is a fundamental geometric constraint,
  not a parameter tuning problem. Wider thresholds make things worse by admitting
  false matches.

fix: |
  Tuned parameters applied (2026-03-03):

  Association (config.yaml + AssociationConfig defaults):
    ray_distance_threshold: 0.03
    score_min: 0.08
    eviction_reproj_threshold: 0.03
    leiden_resolution: 1.5
    early_k: 20

  Reconstruction (config.yaml + ReconstructionConfig + DltBackend defaults):
    outlier_threshold: 10.0  (was 50.0)

  Files changed:
    - src/aquapose/core/reconstruction/backends/dlt.py (DEFAULT_OUTLIER_THRESHOLD 50→10)
    - src/aquapose/engine/config.py (ReconstructionConfig.outlier_threshold 50→10)
    - ~/aquapose/projects/YH/config.yaml (added association + reconstruction sections)

  Further improvements would require better discrimination features (e.g.,
  midline geometry instead of bbox centroids, temporal motion consistency),
  not threshold changes.

verification: verified via tune_association.py sweep (56-combo grid + sequential stages) and tune_threshold.py sweep (20 thresholds, 5-100px)
files_changed:
  - src/aquapose/core/reconstruction/backends/dlt.py
  - src/aquapose/engine/config.py
