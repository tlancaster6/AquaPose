# Phase 83: Custom Tracker Implementation - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Bidirectional batched keypoint tracker replacing OC-SORT for 2D per-camera tracking (Stage 2). Uses OKS-based association cost, OCM direction consistency, Kalman filter over keypoint positions, ORU/OCR mechanisms, bidirectional forward+backward merge, chunk handoff, and gap interpolation. Target: reduce track fragmentation from 27 → 9 tracks on the benchmark clip (Phase 80 baseline). The tracker is per-camera only — cross-camera association (Stage 3) is unchanged.

</domain>

<decisions>
## Implementation Decisions

### Kalman filter state
- **60-dim state**: Track all 6 keypoints (nose, head, spine1, spine2, spine3, tail) × 2D × (position + velocity)
- **Confidence-scaled measurement noise**: R_i = base_R / max(conf_i, epsilon). Low-confidence keypoints get high measurement noise, causing the KF to rely on prediction instead. Smooth degradation, no hard masking or special cases.
- **From scratch**: NumPy-based constant-velocity KF, no external library (filterpy, etc.). Full control over state layout, confidence-scaled R, and serialization for chunk handoff.
- **Coast prediction**: Predict all 6 keypoints during coast. Use full predicted keypoint state for OKS cost when matching coasting tracks to new detections.

### OKS cost design
- **Empirical per-keypoint sigmas**: Compute sigmas from manual annotation dataset — measure per-keypoint position variance normalized by fish scale. Expect: endpoints (nose, tail) get larger sigmas, mid-body (spine1-3) get smaller sigmas.
- **Confidence-weighted OKS**: Weight each keypoint's OKS contribution by detection confidence: `oks = sum(c_k * exp(-d_k^2 / (2*s^2*sigma_k^2))) / sum(c_k)`. Occluded endpoints (low confidence) contribute near-zero weight.
- **Scale term**: sqrt(OBB area), analogous to COCO's bbox area normalization.
- **OKS + OCM as weighted sum**: `cost = (1 - OKS) + lambda * (1 - OCM)`. OCM uses cosine similarity of spine heading vector (spine1→spine3). Lambda is tunable (0.1–0.3 range). OKS dominates; OCM breaks ties and penalizes head-tail flips.

### Bidirectional merge strategy
- **Temporal overlap + OKS matching**: Find temporal overlap between forward and backward tracklets, compute mean OKS in the overlap region, match pairs via Hungarian assignment on OKS cost matrix.
- **Overlap frame resolution**: Keep detected over coasted. If both detected, keep the one with higher mean keypoint confidence. Both coasted → keep either.
- **Independent passes**: Backward pass runs with its own ID space, no seeding from forward results. Merge step handles identity unification.
- **Unmatched tracklets**: Keep if they meet minimum length threshold (n_init frames). Discard short unmatched fragments as likely false positives.

### Birth/death rules
- **Uniform rules — no spatial edge asymmetry**: Same n_init and max_age everywhere. Bidirectional merge naturally recovers tracks that one pass missed (forward catches entries, backward catches exits). No special treatment for detections near frame borders.
- **max_age = 15**: Lower than current OC-SORT default (30). OKS matching is more discriminative — tracks that can't be matched within 15 frames are likely genuinely lost. Configurable.
- **TRACK-05 (asymmetric birth/death)**: Satisfied by the bidirectional merge design — forward+backward naturally provides asymmetric coverage without explicit spatial-edge rules.
- **TRACK-10 (BYTE-style secondary pass)**: Deferred to Phase 84 evaluation. Build primary tracker with single confidence threshold first. If Phase 84 shows missed detections, add BYTE-style pass then.

### Claude's Discretion
- ORU/OCR mechanism details (TRACK-06) — how observation-centric re-update and recovery are implemented within the custom KF
- Gap interpolation method (TRACK-09) — spline type and maximum gap length for interpolation
- n_init default value and tuning range
- Process noise (Q) tuning for the constant-velocity KF model
- OKS sigma computation methodology (exact script/approach for deriving from annotations)
- Internal data structures for the tracker (track pool, tentative vs confirmed lists)

</decisions>

<specifics>
## Specific Ideas

- Confidence-scaled measurement noise in the KF mirrors the confidence-weighted OKS cost — both use the same principle that per-keypoint confidence is a reliable occlusion signal (validated by Phase 78 findings)
- The tracker should feel like a natural evolution of OC-SORT (same ORU/OCR philosophy) but with keypoint state instead of bbox state
- Phase 80 baseline: 27 tracks, 93.1% detection coverage, 0 gaps, 1.000 continuity ratio — fragmentation (27 vs 9 expected) is the primary failure mode to fix

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `OcSortTracker` wrapper pattern (`core/tracking/ocsort_wrapper.py`): Excellent template for the new tracker class — same interface (`update`, `get_tracklets`, `get_state`, `from_state`)
- `_TrackletBuilder` accumulation class: Already structured to build `Tracklet2D` objects from per-frame data
- `Tracklet2D` dataclass (`core/tracking/types.py`): Output contract — frozen, immutable, used by all downstream stages
- `ChunkHandoff` (`engine/context.py`): Existing cross-chunk state serialization mechanism
- `TrackingStage` (`core/tracking/stage.py`): Per-camera dispatch pattern — each camera gets independent tracker instance
- `evaluate_tracking()` and `evaluate_fragmentation()` in `evaluation/stages/`: Ready-made metrics for comparison against Phase 80 baseline
- Test suite (`tests/unit/core/tracking/test_tracking_stage.py`): 281-line template covering single/multi-camera, chunk handoff, field conformance

### Established Patterns
- Tracker isolation: All third-party imports confined to wrapper module; downstream sees only `Tracklet2D`
- Config-driven construction: Frozen dataclasses with YAML round-trip, new fields get defaults for backward compatibility
- Stage protocol: `run(context, carry) -> (context, carry)` for tracking (special-cased in pipeline.py)
- Backend registration: `tracker_kind` field in `TrackingConfig` selects implementation

### Integration Points
- `Detection.keypoints` and `Detection.keypoint_conf` populated by PoseStage (Phase 81) before tracking runs
- `TrackingConfig` in `engine/config.py` — extend with new tracker-specific fields (OKS params, KF params, bidi merge params)
- Pipeline factory (`engine/pipeline.py` `build_stages()`): Creates TrackingStage, which internally selects tracker by `tracker_kind`
- Association stage reads `tracklet.centroids` — custom tracker must populate centroids identically (keypoint-derived per Phase 82)

</code_context>

<deferred>
## Deferred Ideas

- TRACK-10 (BYTE-style secondary pass for low-confidence detections) — deferred to Phase 84 evaluation. Implement only if Phase 84 metrics show missed detections that a secondary pass would recover.

</deferred>

---

*Phase: 83-custom-tracker-implementation*
*Context gathered: 2026-03-10*
