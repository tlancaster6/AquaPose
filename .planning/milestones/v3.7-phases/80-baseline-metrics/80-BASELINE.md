# Phase 80 Baseline: OC-SORT Tracking Metrics

**Established:** 2026-03-10
**Purpose:** Quantitative baseline for Phase 84 custom tracker comparisons.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Camera | e3v83eb |
| Frame range | 3300–4500 (1200 frames) |
| Model | OBB production weights (run_20260310_115419) |
| Tracker | OC-SORT via `OcSortTracker` |
| `min_hits` | 1 (no warm-up penalty — honest baseline) |
| `det_thresh` | 0.1 |
| `conf_threshold` | 0.1 |
| `nms_threshold` | 0.45 (polygon NMS) |
| Expected fish | 9 |

---

## Tracking Metrics

| Metric | Value |
|--------|-------|
| Track count | 27 |
| Length min | 1 frame |
| Length max | 1200 frames |
| Length median | 144.0 frames |
| Length mean | 408.81 frames |
| Coast frequency | 0.069 |
| Detection coverage | 0.931 |

---

## Fragmentation Metrics

| Metric | Value |
|--------|-------|
| Total gaps | 0 |
| Mean gap duration | 0.0 frames |
| Max gap duration | 0 frames |
| Mean continuity ratio | 1.000 |
| Unique fish IDs (tracks) | 27 |
| Track births | 18 |
| Track deaths | 17 |
| Mean track lifespan | 408.81 frames |
| Median track lifespan | 144.0 frames |

---

## Gap-to-Target Analysis

**Target:** 9 tracks, zero fragmentation (one continuous track per fish, zero gaps).

| Dimension | Current | Target | Delta |
|-----------|---------|--------|-------|
| Track count | 27 | 9 | +18 (3x over-fragmented) |
| Total gaps | 0 | 0 | 0 (met) |
| Continuity ratio | 1.000 | 1.000 | 0 (met) |
| Detection coverage | 0.931 | 1.000 | -0.069 (6.9% missed) |

### Interpretation

- **Track fragmentation is the primary failure mode.** OC-SORT produces 27 tracks instead of 9 — a 3x overcount — despite `min_hits=1` and a low detection threshold of 0.1. This means each fish is associated across multiple disjoint track IDs throughout the 1200-frame window.
- **Within-track continuity is perfect (0 gaps, ratio 1.0).** No track has internal frame gaps; every track that exists is continuous for its lifetime. The problem is track identity breaks (old track ends, new track starts for the same fish).
- **Track birth/death counts confirm identity breaks.** 18 births and 17 deaths over 1200 frames against 9 fish indicates the tracker creates roughly 2 tracks per fish on average (27 ÷ 9 ≈ 3, but median lifespan of 144 vs max of 1200 shows one long track per fish plus shorter fragments).
- **Detection coverage at 93.1%** is acceptable but 6.9% of frames have fewer than expected detections, which may contribute to identity breaks when fish temporarily disappear.

### Phase 84 Improvement Targets

The custom tracker should achieve:
1. Track count <= 9 (ideally exactly 9 for the full 1200-frame window)
2. Detection coverage >= 0.93 (match or exceed)
3. Zero gaps maintained (already met by OC-SORT)
4. Births/deaths reduced from 18/17 toward 9/9 (one birth and one death per fish)
