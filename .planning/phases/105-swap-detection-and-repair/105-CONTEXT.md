# Phase 105: Swap Detection and Repair - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Detect identity swap events using embedding cosine similarity and repair them, producing a corrected `midlines_reid.h5`. Two detection modes: body-length seeded (fast) and independent embedding scan (thorough). Phase 106 handles CLI integration.

</domain>

<decisions>
## Implementation Decisions

### Detection strategy
- Two detection modes: **body-length seeded** and **independent embedding scan**
- Body-length seeded mode reads persisted `/swap_events/` from `midlines_stitched.h5` as candidate events (stitch and reid are independent modular steps)
- Independent scan mode combines two signals automatically: **proximity-triggered** (primary, cheap) then **sliding-window** (fallback for uncovered segments)
- Proximity is defined by **3D centroid distance threshold** between fish pairs per frame

### Margin & thresholds
- Fixed cosine margin threshold (default 0.15), configurable via parameter
- Comparison uses **mean embedding in temporal window** (average across cameras and across configurable window, default 10 frames before/after event)
- Swap confirmation requires the **cross pattern**: fish A's post-event embedding matches fish B's pre-event embedding AND vice versa. Self-similarity drop alone is not sufficient.

### Repair semantics
- `midlines_reid.h5` is a **full copy** of `midlines_stitched.h5` with corrected fish_id values
- ID correction propagates from the swap frame **until the next confirmed event or end of video**
- Full provenance in `/reid_events/` dataset: frame, fish_a, fish_b, cosine_margin, detection_mode (seeded vs scan), action (confirmed/rejected/repaired)
- Original `/swap_events/` from body-length detector is **preserved** in `midlines_reid.h5` alongside `/reid_events/`

### Validation approach
- Ground truth from existing YH run data:
  - **Known MF swap**: frame ~2665, fish 0 <-> 5 (already caught by body-length detector — confirmation test)
  - **Known FF swap**: frame ~600, fish 2 <-> 4 (NOT caught by current methods — the key test for embedding approach)
  - **All other IDs**: clean across full video — ground truth for false positive measurement
- Reprojection error check **dropped** — repair only relabels fish_id on entire 3D track segments, so 3D points and 2D projections are mathematically identical before/after. Document this invariance in code.
- Validation report: console summary (events detected, confirmed/rejected, known swaps hit/missed, FP rate on clean segments) + H5 provenance

### Targeted usage strategy (cost control)
- Body-length seeded mode computes embeddings **on-the-fly** only for temporal windows around candidate events (no dependency on prior full embed run)
- Independent scan mode: primary speed knob is **frame stride** (embed every Nth frame, default stride=1)
- Independent scan scope: embed **densely around proximity events**, then at **coarse stride for remaining gaps**
- Scan mode **opportunistically reuses** `embeddings.npz` if a prior full embed run exists, falling back to on-the-fly for missing frames

### Claude's Discretion
- Proximity distance threshold default value
- Sliding window size for the scan mode's gap-filling pass
- Internal data structures for the swap detector
- How to efficiently index into existing embeddings.npz

</decisions>

<specifics>
## Specific Ideas

- The FF swap (fish 2 <-> 4 at ~frame 600) is the motivating case — body-length method can't catch same-size fish swaps, embeddings should
- The MF swap (fish 0 <-> 5 at ~frame 2665) is an easy confirmation that the new method works at least as well as body-length
- User wants an easily tunable accuracy-speed tradeoff — frame stride is the primary knob

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `FishEmbedder` (`core/reid/embedder.py`): MegaDescriptor-T backbone, accepts BGR crops, returns L2-normalized 768-dim embeddings
- `EmbedRunner` (`core/reid/runner.py`): Batch embedding from chunk caches, writes `embeddings.npz` — logic for crop extraction, prestitch ID mapping, and frame source management
- `SwapEvent` dataclass (`core/stitching.py`): frame, fish_a, fish_b, score_a, score_b, auto_corrected
- `detect_and_repair_swaps()` (`core/stitching.py`): Body-length changepoint swap detection, writes `/swap_events/` to H5
- `apply_swap_repairs()` (`core/stitching.py`): Swaps fish_id values in H5 from swap frame forward — propagation logic to reuse

### Established Patterns
- H5 structure: `/midlines/` group with `frame_index`, `fish_id`, `points`, `swap_events` datasets
- Chunk cache loading via `load_chunk_cache()` for accessing per-frame detections and tracklet groups
- Prestitch-to-stitched ID mapping via centroid matching in `EmbedRunner._build_prestitch_to_stitched_map()`
- OBB-aligned crop extraction via `extract_affine_crop()` for consistent embedding input

### Integration Points
- Reads: `midlines_stitched.h5` (for swap_events, 3D centroids, fish_id arrays), chunk caches (for crop extraction), video frames
- Writes: `midlines_reid.h5` (corrected copy), optionally reuses/extends `embeddings.npz`
- The swap detector module should live in `core/reid/` alongside existing embedding infrastructure

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 105-swap-detection-and-repair*
*Context gathered: 2026-03-25*
