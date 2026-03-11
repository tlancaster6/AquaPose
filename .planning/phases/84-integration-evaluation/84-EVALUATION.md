# Phase 84 Tracker Evaluation: OC-SORT vs keypoint_bidi

**Clip:** Camera `e3v83eb`, frames 3300–4500 (1200 frames, ~40s at 30 fps)
**N animals:** 9
**Run date:** 2026-03-11
**Script:** `scripts/evaluate_custom_tracker.py`
**Config:** `~/aquapose/projects/YH/config.yaml`

---

## Phase 80 Baseline (reference — from archived run)

| Metric | Value |
|---|---|
| Track count | 27 |
| Detection coverage | 0.931 |
| Total gaps | 0 |
| Continuity ratio | 1.000 |
| Track births | 18 |
| Track deaths | 17 |

*OC-SORT params at Phase 80: min_hits=1, det_thresh=0.1 (no warm-up penalty)*

---

## Current Evaluation Results

**Tracker params used (both trackers):**
- OC-SORT: max_age=15, min_hits=1, det_thresh=0.1
- KeypointTracker: max_age=15, n_init=1, det_thresh=0.1, base_r=10.0, lambda_ocm=0.2, max_gap_frames=5

| Metric | OC-SORT Baseline (Ph80) | OC-SORT (this run) | Keypoint Bidi |
|---|---|---|---|
| Track count (target=9) | 27 | 30 | **44** |
| Detection coverage | 0.931 | 0.955 | 0.898 |
| Coast frequency | — | 0.045 | 0.102 |
| Length min (frames) | — | 1 | 17 |
| Length max (frames) | — | 1200 | 1200 |
| Length median (frames) | — | 109 | 73 |
| Total gaps | 0 | 0 | 0 |
| Continuity ratio | 1.000 | 1.000 | 1.000 |
| Track births | 18 | 21 | 33 |
| Track deaths | 17 | 21 | 35 |

---

## Interpretation

The custom `keypoint_bidi` tracker is **more fragmented than OC-SORT**, not less. Track count increased from 30 (OC-SORT) to 44 (keypoint_bidi), moving further from the 9-fish target rather than closer. Detection coverage dropped from 95.5% to 89.8%, just below the 90% BYTE trigger threshold.

Neither tracker approaches the 9-track target — both produce 3–5x over-fragmentation. The primary failure mode for both is identity-breaking at occlusion: when two fish pass close together, a track is lost and a new one is born on re-detection. This is a fundamentally different problem from fragmentation (temporal gaps), which both trackers handle perfectly (0 gaps, continuity=1.000).

The `keypoint_bidi` tracker's higher fragmentation likely stems from stricter matching: the OKS+OCM cost function demands keypoint consistency across frames, which rejects matches during occlusion where detections are noisy. OC-SORT's looser IoU matching accepts these imperfect reacquisitions. With `n_init=1` and `max_age=15`, both trackers have equal confirmation/dropping sensitivity — the difference is purely in the cost function.

The higher coast frequency for keypoint_bidi (0.102 vs 0.045) confirms tracks are coasting more often, meaning detections are being passed over as unmatched when they don't meet OKS threshold. This is the proximate cause of the coverage drop to 89.8%.

---

## BYTE-Style Secondary Pass Recommendation (TRACK-10)

**Triggered: YES** — coverage = 0.898 < 0.90 threshold.

A BYTE-style secondary pass would attempt to match unmatched low-confidence detections against coasting tracks, recovering some of the detections currently rejected by the OKS cost. However, given that the primary problem is identity-breaking at occlusion (track death + new track birth on reacquisition), a secondary pass on low-confidence detections is unlikely to address the root cause. The continuity metric is already 1.000 — tracks that exist are continuous. The issue is that too many tracks are being created.

TRACK-10 implementation is deferred. The appropriate next step is parameter tuning (see below) or architectural investigation of the occlusion reacquisition failure.

---

## Parameter Tuning Analysis

The default parameters were selected from `TrackingConfig` defaults. The current results suggest the OKS matching threshold is too strict for this data, causing excessive track fragmentation at occlusion events.

**Suggested tuning directions for future investigation:**
1. Increase `base_r` (e.g., 50–100) to widen the Kalman filter measurement noise tolerance, accepting more uncertain matches
2. Reduce `lambda_ocm` (e.g., 0.05) to rely less on heading similarity during occlusion
3. Increase `max_age` (e.g., 30–60) to keep coasting tracks alive longer, improving reacquisition
4. Reduce cost match threshold (`_MATCH_COST_THRESHOLD` in `keypoint_tracker.py`) to be more permissive

No tuning rounds were performed in this evaluation — the architectural insight (occlusion reacquisition vs. temporal fragmentation) is the more important finding.

---

## Conclusions

- The custom `keypoint_bidi` tracker is functional and integrated into the pipeline
- Both `tracker_kind: ocsort` and `tracker_kind: keypoint_bidi` work end-to-end
- Neither tracker solves over-fragmentation at occlusion — this requires either cross-view association (Phase 85) or deeper keypoint tracking tuning
- OC-SORT remains the better-performing option (30 vs 44 tracks) under current parameters
- `TrackingConfig.tracker_kind` default is `keypoint_bidi` per the phase plan; this can be reverted to `ocsort` if needed via config override
- TRACK-10 (BYTE-style secondary pass) triggered by coverage, deferred pending further investigation

**Annotated video:** `eval_tracker_output/keypoint_bidi_tracking.mp4`
**Metrics JSON:** `eval_tracker_output/comparison_metrics.json`
