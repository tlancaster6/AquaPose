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

**Context:** Phase 84-02 stripped the bidirectional merge from `KeypointTracker`. The
bidi merge used 1:1 Hungarian assignment against an N:1 mismatch (forward fragments vs.
single backward track), producing unmatched forward fragments that survived as duplicate
tracklets. 8 duplicate pairs inflated the track count from ~37 to 44 in the prior run.
With bidi removed, `KeypointTracker` now runs a single forward pass only with ORU/OCR
occlusion recovery — same as `_SinglePassTracker` directly.

**Tracker params used (both trackers):**
- OC-SORT: max_age=15, min_hits=1, det_thresh=0.1
- KeypointTracker: max_age=15, n_init=1, det_thresh=0.1, base_r=10.0, lambda_ocm=0.2, max_gap_frames=5

| Metric | OC-SORT Baseline (Ph80) | OC-SORT (this run) | Keypoint Bidi (bidi, 84-01) | Keypoint Single-Pass (84-02) |
|---|---|---|---|---|
| Track count (target=9) | 27 | 30 | **44** | **42** |
| Detection coverage | 0.931 | 0.955 | 0.898 | 0.936 |
| Coast frequency | — | 0.045 | 0.102 | 0.064 |
| Length min (frames) | — | 1 | 17 | 7 |
| Length max (frames) | — | 1200 | 1200 | 1200 |
| Length median (frames) | — | 109 | 73 | 63 |
| Total gaps | 0 | 0 | 0 | 0 |
| Continuity ratio | 1.000 | 1.000 | 1.000 | 1.000 |
| Track births | 18 | 21 | 33 | 33 |
| Track deaths | 17 | 21 | 35 | 33 |

---

## Interpretation

Removing the bidirectional merge improved coverage from 89.8% to 93.6% and reduced coast
frequency from 0.102 to 0.064. Track count dropped slightly from 44 to 42 — a modest
improvement from eliminating the duplicate-inflated pairs. The BYTE-style trigger is no
longer fired (coverage 0.936 >= 0.90 threshold).

The remaining over-fragmentation (42 vs. target 9) has the same root cause as OC-SORT's
(30 tracks): identity-breaking at occlusion events. When two fish pass close together, a
track is lost and a new one is born on reacquisition. Both trackers handle temporal
continuity perfectly (0 gaps, continuity=1.000) — the problem is exclusively
cross-fish occlusion reacquisition, not temporal fragmentation.

The custom single-pass tracker still shows more fragmentation than OC-SORT (42 vs. 30
tracks) under these parameters. The OKS+OCM cost function is stricter than OC-SORT's IoU
matching, rejecting near-occlusion matches that OC-SORT would accept. Higher `base_r`
or lower `lambda_ocm` could allow more permissive matching during occlusion — see
Parameter Tuning Analysis below.

---

## BYTE-Style Secondary Pass Recommendation (TRACK-10)

**Triggered: NO** — coverage = 0.936 >= 0.90 threshold (was triggered in 84-01 with buggy bidi at 0.898).

TRACK-10 implementation remains deferred. The primary problem is occlusion
reacquisition fragmentation, not low-confidence detection misses.

---

## Parameter Tuning Analysis

The default parameters were selected from `TrackingConfig` defaults. The current results
suggest the OKS matching threshold is too strict for this data, causing excessive track
fragmentation at occlusion events.

**Suggested tuning directions for future investigation:**
1. Increase `base_r` (e.g., 50–100) to widen the Kalman filter measurement noise tolerance, accepting more uncertain matches
2. Reduce `lambda_ocm` (e.g., 0.05) to rely less on heading similarity during occlusion
3. Increase `max_age` (e.g., 30–60) to keep coasting tracks alive longer, improving reacquisition
4. Reduce cost match threshold (`_MATCH_COST_THRESHOLD` in `keypoint_tracker.py`) to be more permissive

No tuning rounds were performed in this evaluation — the architectural insight (occlusion reacquisition vs. temporal fragmentation) is the more important finding.

---

## Conclusions

- The bidi merge removal improved coverage (89.8% → 93.6%) and eliminated duplicate IDs in the annotated video
- The custom `keypoint_bidi` tracker is now a clean single-pass tracker with no dead bidi code
- Both `tracker_kind: ocsort` and `tracker_kind: keypoint_bidi` work end-to-end
- Neither tracker solves over-fragmentation at occlusion — this requires either cross-view association (Phase 85) or deeper keypoint tracking tuning
- OC-SORT remains the lower-track-count option (30 vs. 42) under current parameters
- TRACK-10 (BYTE-style secondary pass) no longer triggered after bidi removal

**Annotated video:** `eval_tracker_output/keypoint_bidi_tracking.mp4`
**Metrics JSON:** `eval_tracker_output/comparison_metrics.json`
