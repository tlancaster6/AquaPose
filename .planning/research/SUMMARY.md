# Project Research Summary

**Project:** AquaPose v3.8 Improved Association
**Domain:** Multi-keypoint cross-view association scoring, temporal changepoint detection, singleton recovery
**Researched:** 2026-03-11
**Confidence:** HIGH

## Executive Summary

AquaPose v3.8 is a targeted upgrade to the multi-view fish association stage: replacing single-centroid ray-ray scoring with multi-keypoint scoring, replacing the post-clustering refinement step with a group validation pass that uses temporal changepoint detection to split swapped tracklets, and adding a singleton recovery step that reassigns or split-assigns leftover tracklets. The goal is to reduce the 27% singleton rate observed post-v3.7 to approximately 15%. All research confirms this milestone can be executed entirely with the existing dependency set — no new libraries are needed. The critical prerequisite is a data contract change: `Tracklet2D` must carry per-frame keypoint arrays and confidence values, which the `_KptTrackletBuilder` already produces internally but currently discards in `to_tracklet2d()`. Once that pass-through is in place, the rest of the work is scoring algebra, a new module `validation.py`, and config field additions.

The recommended implementation approach is strictly bottom-up following the build order derived from the architecture research: (1) extend `Tracklet2D`, (2) extend scoring, (3) remove fragment merging, (4) add group validation with changepoint detection, (5) add singleton recovery, (6) clean up config and evaluation, (7) run the tuning pass. Aggregation should start with arithmetic mean of confidence-filtered keypoints per frame; trimmed mean and confidence-weighted mean are deferred to v3.8.x after the baseline multi-keypoint result is validated. Custom binary split (O(n) prefix-sum sweep) is sufficient for changepoint detection — do not add `ruptures` or any changepoint library.

The highest risks are a recurrence of the LUT coordinate space mismatch bug (already occurred in v3.7 with 14% focal length divergence causing 86% singleton rate), and changepoint false positives on short tracklets that produce more singletons post-v3.8 than pre. Both are preventable with targeted unit tests: a round-trip 3D→project→cast_ray test verifies coordinate space correctness before any end-to-end evaluation, and a false positive rate measurement (split fragments that rejoin original group in singleton recovery) verifies changepoint threshold calibration before deploying.

## Key Findings

### Recommended Stack

All v3.8 features are implementable with the existing dependency set (NumPy >=1.24, PyTorch >=2.0, Python >=3.11). The only structural change is a type extension — `Tracklet2D` gains two optional tuple fields — not a dependency change. The changepoint algorithm specified by the design document is an O(n) prefix-sum sweep, which is approximately 20 lines of NumPy and requires no library. External options (`ruptures`, `bayesian_changepoint_detection`, `changepy`) were evaluated and rejected: `ruptures` requires penalty calibration that adds a tuning free parameter, while the design document's threshold is expressed in metres (directly comparable to existing `eviction_reproj_threshold`), and the other libraries are unmaintained. `pyproject.toml` does not change.

**Core technologies:**
- NumPy >=1.24: `np.nanmean`, `np.cumsum`, `np.argmax`, `np.where` — all aggregation, masking, and changepoint sweep operations; already in project
- PyTorch >=2.0: `ForwardLUT.cast_ray()` — extended to accept K×T input instead of T; already in `scoring.py`
- Python frozen dataclasses + structural Protocol: `Tracklet2D` type extension and new `ValidationConfigLike` — established codebase patterns

### Expected Features

**Must have (table stakes) — v3.8 core:**
- Tracklet2D keypoint propagation — prerequisite that gates all other work; builder already accumulates keypoints internally
- Multi-keypoint pairwise scoring with binary confidence threshold filtering and arithmetic mean aggregation — core scoring upgrade replacing single-centroid
- Group validation with multi-keypoint residuals, outlier eviction, and temporal changepoint detection — replaces `refinement.py`
- Singleton recovery: simple assignment of singletons to existing groups — primary recovery mechanism for case B singletons
- Singleton recovery: split-and-assign for case A ID swaps — secondary recovery for 50/50 swap tracklets
- Fragment merging removed, `refinement.py` deleted — simplification with no loss
- Updated `AssociationConfig` with new parameters and defaults — enables tuning pass
- Parameter tuning pass against v3.7 baseline — validation that singleton rate improved

**Should have (v3.8.x, after core validated):**
- Trimmed mean aggregation (drop top 1 of 6 distances) — add if single-keypoint outliers are visible in diagnostics
- Confidence-weighted mean — add if binary threshold causes systematic per-frame dropouts
- Min-keypoint-count guard per frame — add if low-kpt frames degrade scoring in practice
- Extended association evaluator metrics (per-group residual, camera count distribution) — richer tuning signal

**Defer (v3.9+):**
- Appearance-based ReID for close-proximity sustained swimming — requires fish-specific training data that does not exist
- 3D trajectory gap-bridging — belongs in post-processing, not association
- `ruptures` PELT changepoint upgrade — quality-of-life only; custom split is correct and sufficient

**Explicit anti-features (do not build):**
- Per-frame 3D position consensus during scoring — architectural regression to the v1.0 failure mode
- Learned/neural affinity scoring — no ground truth exists; geometric signal is sufficient
- Bidirectional changepoint scoring — no measured benefit from v3.7 bidirectional merge experiment (44 vs 42 tracks)
- Exhaustive keypoint permutation matching — O(K!) = 720 per pair-frame; 2D tracker maintains keypoint identity by index

### Architecture Approach

The v3.8 association module replaces steps 4-5 of the current 5-step orchestration (merge fragments, refine clusters) with two new steps (validate groups, recover singletons) while keeping the scoring and clustering steps structurally identical. A single new file `validation.py` replaces `refinement.py`. The `scoring.py` file is extended in-place. The critical data flow addition is `Tracklet2D` growing two optional fields that propagate keypoint data from the tracking stage into association; all downstream consumers of `TrackletGroup` (reconstruction, evaluation) are unchanged. The import boundary IB-003 (core never imports engine) is preserved via `ValidationConfigLike` structural protocol, matching the existing `AssociationConfigLike` pattern.

**Major components and their v3.8 changes:**
1. `tracking/types.py` + `tracking/keypoint_tracker.py` — MODIFY: add `keypoints: tuple | None` and `keypoint_conf: tuple | None` fields to `Tracklet2D`; populate in `to_tracklet2d()`
2. `association/scoring.py` — MODIFY: extend `_batch_score_frames` from 1 ray/frame to K rays/frame with confidence masking and per-frame mean aggregation
3. `association/clustering.py` — MODIFY: delete `merge_fragments` and helpers; clustering logic unchanged
4. `association/validation.py` — NEW: `validate_groups()` (changepoint + eviction) and `recover_singletons()` (simple assign + split-assign)
5. `association/refinement.py` — DELETE: replaced entirely by `validation.py`
6. `association/stage.py` — MODIFY: replace merge+refine steps with validate+recover
7. `engine/config.py` — MODIFY: add 6 new `AssociationConfig` fields; remove `max_merge_gap` and `refinement_enabled`

### Critical Pitfalls

1. **Tracklet2D missing keypoints (P1)** — extend `Tracklet2D` and populate fields in `to_tracklet2d()` as the first implementation step before any scoring code is written; all other work depends on this contract change

2. **LUT coordinate space mismatch recurrence (P2)** — write a round-trip unit test (3D keypoint → refractive projection → `cast_ray` → verify ray passes within 2mm of source) before any end-to-end evaluation; this exact bug caused 86% singleton rate in v3.7

3. **Broadcasting shape mismatch with K-keypoint extension (P3)** — decide on either flat `(N*K_valid, 2)` with index tracking or dense `(N, K, 2)` with NaN masking before writing code; write a loop-based reference implementation as a correctness oracle, then vectorize and compare

4. **Changepoint false positives on short tracklets (P5)** — calibrate significance threshold against confirmed-correct tracklets from the v3.7 benchmark run; measure false positive rate (split fragments that rejoin original group) and target < 30%; enforce minimum segment length of at least 10 frames

5. **Refinement removal breaks downstream confidence fields (P7)** — grep all reads of `TrackletGroup.per_frame_confidence` and `consensus_centroids` before deleting `refinement.py`; `validate_groups()` must populate equivalent values or all consumers must handle `None` correctly

## Implications for Roadmap

Based on research, the build dependency graph is strict and bottom-up. The architecture research provides an explicit 7-step build order that maps directly to roadmap phases.

### Phase 1: Tracklet2D Keypoint Propagation
**Rationale:** Hard prerequisite for all other work. `_KptTrackletBuilder` already accumulates the data; this is a pass-through fix plus frozen dataclass extension. Smallest possible change that unblocks everything downstream.
**Delivers:** `Tracklet2D` with optional `keypoints` and `keypoint_conf` fields populated by tracking stage; all existing consumers unaffected due to `None` defaults
**Addresses:** Table-stakes prerequisite from FEATURES.md
**Avoids:** P1 (silent data loss), establishes foundation to avoid P3 (vectorization design requires knowing field layout first)

### Phase 2: Multi-Keypoint Pairwise Scoring
**Rationale:** Scoring extension is logically independent of clustering and validation. Can be validated in isolation by comparing pair scores against single-centroid baseline on known tracklet pairs. Must be working before group validation, which reuses the same ray casting infrastructure.
**Delivers:** `_batch_score_frames` extended to K rays/frame with confidence filtering and mean-per-frame aggregation; fallback to centroid when keypoints absent; `AssociationConfig` gains `scoring_keypoint_confidence_floor`
**Uses:** NumPy `np.nanmean` and `np.where`, `ForwardLUT.cast_ray()` — no new dependencies
**Avoids:** P2 (round-trip unit test required before end-to-end), P3 (vectorization correctness oracle), P4 (aggregation choice documented and justified before implementation)

### Phase 3: Fragment Merging Removal
**Rationale:** Clean deletion with no new logic. Touches different files from Phase 2 (only `clustering.py` and orchestration in `stage.py`). Can be parallelized with Phase 2 in execution since there are no shared file edits. Reduces code surface before adding new validation logic.
**Delivers:** `merge_fragments` and all helpers deleted; `max_merge_gap` removed from config; `stage.py` updated; pipeline still runs end-to-end
**Implements:** Architectural simplification prescribed by design document

### Phase 4: Group Validation with Changepoint Detection
**Rationale:** Requires Phase 1 (keypoint data in tracklets) and Phase 2 (LUT interaction patterns validated) to be complete. The changepoint algorithm is fresh implementation; must be built on a foundation of validated ray casting. This phase replaces `refinement.py` as the post-clustering quality gate.
**Delivers:** `validation.py` with `validate_groups()`: per-frame multi-keypoint residuals against group consensus, O(n) binary split changepoint detection, tracklet eviction or splitting; `refinement.py` deleted; `stage.py` updated
**Addresses:** Temporal changepoint detection and outlier eviction from FEATURES.md table stakes
**Avoids:** P5 (false positive rate measurement required before deploying), P7 (downstream field audit before deletion)

### Phase 5: Singleton Recovery
**Rationale:** Requires Phase 4 output (finalized groups with evicted singletons) as input. The simple assignment case dominates; split-and-assign can be validated only after simple assignment confirms the residual infrastructure works end-to-end.
**Delivers:** `recover_singletons()` in `validation.py`: simple assignment (singleton to best-match group) and split-and-assign (case A swap recovery); `stage.py` wired; group validity assertion as post-condition
**Addresses:** Singleton recovery (simple + split) from FEATURES.md table stakes
**Avoids:** P6 (wrong-group assignment — margin gap criterion required), P8 (must-not-link constraint inheritance for fragments)

### Phase 6: Config and Evaluation Cleanup
**Rationale:** Final integration pass. All implementation is complete; this phase ensures the parameter surface is clean, the evaluation grid is accurate, and no dead config fields remain from deleted components.
**Delivers:** `engine/config.py` with complete final field set (6 additions, 2 removals); `evaluation/stages/association.py` with updated `DEFAULT_GRID`; no dead code remaining

### Phase 7: Parameter Tuning Pass
**Rationale:** The new config parameters have reasonable defaults but require empirical calibration against real data. The significance threshold, minimum segment length, and singleton assignment threshold all interact. The existing `aquapose tune` infrastructure handles this as a grid search over cached tracking outputs.
**Delivers:** Tuned `AssociationConfig` defaults validated against v3.7 baseline (27% singleton rate); measured improvement toward ~15% target; tuning results documented
**Addresses:** Parameter tuning pass from FEATURES.md MVP definition

### Phase Ordering Rationale

- Phase 1 must come first: no other work is possible without the `Tracklet2D` data contract change
- Phase 3 can run concurrently with Phase 2 in execution (different files, no shared edits)
- Phases 4 and 5 both require Phase 2 to be validated first — the LUT interaction and confidence filtering patterns established in Phase 2 are the foundation for validation logic
- Phase 6 is deferred until all implementation is complete to avoid chasing moving config surfaces
- Phase 7 is necessarily last — calibration requires a working end-to-end pipeline

### Research Flags

Phases likely needing deeper research or careful design review during planning:
- **Phase 2 (Multi-keypoint scoring):** Aggregation choice (mean vs. trimmed vs. weighted), confidence filtering edge cases, and the `(N*K, 2)` vectorization design require explicit documentation before coding begins. A loop-based reference implementation should be written first as a correctness oracle.
- **Phase 4 (Group validation):** Changepoint significance threshold must be calibrated against confirmed-correct tracklets from the v3.7 benchmark run before deploying. The false-positive measurement protocol needs to be defined in the phase plan.
- **Phase 5 (Singleton recovery):** The margin gap criterion for avoiding wrong-group assignment (P6) requires a decision on the specific margin percentage, which depends on observed residual distributions from Phase 4.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Tracklet2D extension):** Frozen dataclass field addition with `None` defaults — well-understood pattern already used in the codebase
- **Phase 3 (Fragment merging removal):** Pure deletion — no research needed
- **Phase 6 (Config/eval cleanup):** Mechanical follow-up to implementation phases
- **Phase 7 (Tuning pass):** Existing `aquapose tune` infrastructure handles this; only threshold ranges need to be determined from Phase 4 and 5 outcomes

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Direct codebase inspection confirms all operations are in existing stack; `pyproject.toml` unchanged; `ruptures` rejection documented with explicit comparison table |
| Features | HIGH | Design document is the primary source (HIGH confidence); table-stakes/differentiator/anti-feature classification derived directly from design doc and codebase constraints; MEDIUM only for empirical tradeoffs on very short sequences |
| Architecture | HIGH | Built entirely from direct codebase inspection; build order is derived from hard code dependencies, not heuristics; 7-step sequence is deterministic |
| Pitfalls | HIGH | P1 and P2 confirmed from codebase inspection and v3.7 post-mortem; P3-P8 derived from structural analysis of vectorized broadcasting patterns and design document gaps |

**Overall confidence:** HIGH

### Gaps to Address

- **Changepoint threshold calibration:** The design document proposes `changepoint_delta_threshold=0.015` (1.5cm) as an initial value. This needs validation against confirmed-correct tracklets from the v3.7 benchmark run before Phase 4 is complete. Resolution: Phase 4 plan should include the false positive rate measurement protocol.
- **Aggregation method A/B validation:** Start with arithmetic mean; compare to single-centroid baseline before committing to the approach. Resolution: Phase 2 plan should include an explicit A/B comparison against the v3.7 centroid baseline as a gate condition.
- **`per_frame_confidence` and `consensus_centroids` consumer audit:** Must be done before `refinement.py` is deleted. Resolution: first task in Phase 4 plan is a grep audit of all reads of these fields across the codebase.
- **Singleton split-and-assign vs. simple assignment contribution ratio:** Case B (simple assignment) is expected to dominate, but the actual split between case A and case B is unknown until Phase 5 runs on real data. Resolution: Phase 5 plan should measure both modes separately to determine whether split-and-assign adds measurable value.

## Sources

### Primary (HIGH confidence)
- `.planning/inbox/association_multikey_rework.md` — primary design document specifying multi-keypoint scoring, changepoint detection, and singleton recovery
- AquaPose codebase direct inspection: `core/association/scoring.py`, `core/association/refinement.py`, `core/association/clustering.py`, `core/association/stage.py`, `core/tracking/types.py`, `core/tracking/keypoint_tracker.py`, `engine/config.py`, `evaluation/stages/association.py`
- `.planning/PROJECT.md` — v3.8 milestone definition, IB-003 import boundary constraint
- AquaPose project memory: ForwardLUT coordinate space mismatch post-mortem (2026-03-04)

### Secondary (MEDIUM confidence)
- `ruptures` PyPI page (version 1.1.10, Sept 2025) and documentation — PELT and Binary Segmentation algorithm characteristics; used to confirm rejection decision
- [SelfPose3d CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Srivastav_SelfPose3d_Self-Supervised_Multi-Person_Multi-View_3d_Pose_Estimation_CVPR_2024_paper.pdf) — confidence-weighted aggregation in multi-view pose estimation
- [Multi-animal DeepLabCut, Nature Methods 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9007739/) — keypoint affinity scoring and tracklet graph matching

### Tertiary (LOW confidence)
- [Ma et al. 2024, Information Fusion](https://arxiv.org/abs/2405.18606) — track re-identification in 3D multi-view tracking (abstract only)
- [Fish Tracking Challenge 2024](https://arxiv.org/html/2409.00339v1) — evaluation methodology for fish multi-object tracking

---
*Research completed: 2026-03-11*
*Ready for roadmap: yes*
