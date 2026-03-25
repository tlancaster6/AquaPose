# Phase 105: Swap Detection and Repair - Research

**Researched:** 2026-03-25
**Domain:** Appearance-based fish identity swap detection via cosine similarity; H5 provenance; embedding reuse
**Confidence:** HIGH — all findings grounded in existing codebase

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Detection strategy:**
- Two detection modes: **body-length seeded** and **independent embedding scan**
- Body-length seeded mode reads persisted `/swap_events/` from `midlines_stitched.h5` as candidate events (stitch and reid are independent modular steps)
- Independent scan mode combines two signals automatically: **proximity-triggered** (primary, cheap) then **sliding-window** (fallback for uncovered segments)
- Proximity is defined by **3D centroid distance threshold** between fish pairs per frame

**Margin & thresholds:**
- Fixed cosine margin threshold (default 0.15), configurable via parameter
- Comparison uses **mean embedding in temporal window** (average across cameras and across configurable window, default 10 frames before/after event)
- Swap confirmation requires the **cross pattern**: fish A's post-event embedding matches fish B's pre-event embedding AND vice versa. Self-similarity drop alone is not sufficient.

**Repair semantics:**
- `midlines_reid.h5` is a **full copy** of `midlines_stitched.h5` with corrected fish_id values
- ID correction propagates from the swap frame **until the next confirmed event or end of video**
- Full provenance in `/reid_events/` dataset: frame, fish_a, fish_b, cosine_margin, detection_mode (seeded vs scan), action (confirmed/rejected/repaired)
- Original `/swap_events/` from body-length detector is **preserved** in `midlines_reid.h5` alongside `/reid_events/`

**Validation approach:**
- Ground truth from existing YH run data:
  - **Known MF swap**: frame ~2665, fish 0 <-> 5 (confirmation test for seeded mode)
  - **Known FF swap**: frame ~600, fish 2 <-> 4 (key test for embedding scan mode)
  - **All other IDs**: clean across full video — ground truth for false positive measurement
- Reprojection error check **dropped** — repair only relabels fish_id on entire 3D track segments, so 3D points and 2D projections are mathematically identical before/after. Document this invariance in code.
- Validation report: console summary (events detected, confirmed/rejected, known swaps hit/missed, FP rate on clean segments) + H5 provenance

**Targeted usage strategy (cost control):**
- Body-length seeded mode computes embeddings **on-the-fly** only for temporal windows around candidate events
- Independent scan mode: primary speed knob is **frame stride** (embed every Nth frame, default stride=1)
- Independent scan scope: embed **densely around proximity events**, then at **coarse stride for remaining gaps**
- Scan mode **opportunistically reuses** `embeddings.npz` if a prior full embed run exists, falling back to on-the-fly for missing frames

### Claude's Discretion
- Proximity distance threshold default value
- Sliding window size for the scan mode's gap-filling pass
- Internal data structures for the swap detector
- How to efficiently index into existing embeddings.npz

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SWAP-01 | Swap detector identifies ID swap candidates at occlusion events by comparing pre/post-event mean embeddings via cosine similarity | `FishEmbedder.embed_batch()` provides L2-normalized embeddings; `SwapEvent` from stitching.py seeds the seeded mode; 3D centroid data from `points` dataset enables proximity scan |
| SWAP-02 | Margin-gated repair re-assigns fish IDs only when cosine margin exceeds threshold, writes corrected output to `midlines_reid.h5` | `apply_swap_repairs()` propagation logic in stitching.py is directly reusable; `write_remapped_h5()` pattern for H5 copy; new `/reid_events/` dataset for provenance |
</phase_requirements>

---

## Summary

Phase 105 implements a two-mode appearance-based swap detector and repair engine in `core/reid/`. The phase builds on substantial existing infrastructure: `FishEmbedder` for crop embedding, `EmbedRunner` for batch embedding with crop extraction from chunk caches, `SwapEvent` for the seeded event source, and `apply_swap_repairs()` for the fish_id propagation logic. The new module (`core/reid/swap_detector.py`) will read `midlines_stitched.h5`, optionally reuse `embeddings.npz`, and write `midlines_reid.h5` with corrected IDs and a provenance `/reid_events/` dataset.

The key algorithmic challenge is the cross-pattern confirmation check: a swap is only confirmed when fish A's post-event mean embedding is closer to fish B's pre-event mean embedding than to fish A's own pre-event mean embedding, and vice versa (cosine margin > 0.15 for both directions). This is stronger than just measuring self-similarity drop. The repair step is mathematically trivial — it only relabels fish_id values; all 3D points and 2D projections are unchanged, so reprojection error is invariant by construction.

The independent scan mode introduces the more complex logic: proximity-triggered dense embedding around 3D close-approach events, with sliding-window fallback for coverage gaps. The embeddings.npz opportunistic reuse requires efficient indexing (frame + fish_id lookup into the flat NPZ arrays). A standalone script approach (per workflow preference) is appropriate before CLI integration in Phase 106.

**Primary recommendation:** Implement as `core/reid/swap_detector.py` with a `SwapDetector` class and a companion `scripts/detect_swaps.py` standalone driver. Reuse `apply_swap_repairs()` propagation logic directly, adapting it to write to a new output file rather than in-place.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| h5py | project dep | Read `midlines_stitched.h5`, write `midlines_reid.h5`, read/write datasets | Already used throughout codebase for all H5 I/O |
| numpy | project dep | Cosine similarity, mean embedding, centroid distance, array indexing | All embedding math is float32 numpy |
| torch / FishEmbedder | project dep | On-the-fly crop embedding | Existing backbone wrapper, L2-normalized output |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dataclasses | stdlib | `ReidEvent` dataclass for provenance records | Matches `SwapEvent` pattern in stitching.py |
| shutil | stdlib | H5 file copy (`shutil.copy2`) | For producing `midlines_reid.h5` as a copy of `midlines_stitched.h5` |
| pathlib | stdlib | File path handling | Project standard |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| shutil.copy2 for H5 copy | h5py deep copy | shutil is simpler and preserves exact bytes; h5py copy is needed only if selective dataset exclusion is required |
| numpy cosine similarity | scipy.spatial.distance | numpy is already a dep; scipy adds no value here for 1-vs-1 comparisons |

---

## Architecture Patterns

### Recommended Module Structure

```
src/aquapose/core/reid/
├── embedder.py          # FishEmbedder (existing)
├── runner.py            # EmbedRunner (existing)
├── miner.py             # TrainingDataMiner (existing)
├── eval.py              # compute_reid_metrics (existing)
├── swap_detector.py     # NEW: SwapDetector class
└── __init__.py          # update with new exports

scripts/
└── detect_swaps.py      # NEW: standalone driver script

tests/unit/core/reid/
├── __init__.py          # exists
├── test_miner.py        # exists
└── test_swap_detector.py  # NEW: pure function unit tests
```

### Pattern 1: SwapDetector Class

The swap detector should be a class that accepts config at construction and exposes `run()` returning a list of `ReidEvent` records. This matches the `EmbedRunner` pattern.

```python
@dataclass
class SwapDetectorConfig:
    cosine_margin_threshold: float = 0.15
    window_frames: int = 10          # pre/post event window
    proximity_threshold_m: float = 0.15  # 3D centroid distance (meters)
    scan_frame_stride: int = 1
    scan_dense_window: int = 20      # frames around proximity events
    scan_gap_stride: int = 10        # coarse stride for gap-filling

class SwapDetector:
    def __init__(self, run_dir: Path, config: SwapDetectorConfig) -> None: ...
    def run(self, mode: Literal["seeded", "scan"]) -> list[ReidEvent]: ...
```

### Pattern 2: ReidEvent Dataclass

Mirrors `SwapEvent` but with additional provenance fields for `/reid_events/` H5 dataset:

```python
@dataclass
class ReidEvent:
    frame: int
    fish_a: int
    fish_b: int
    cosine_margin: float        # min(cross_sim_a, cross_sim_b) - max(self_sim_a, self_sim_b)
    detection_mode: str         # "seeded" | "scan_proximity" | "scan_window"
    action: str                 # "confirmed" | "rejected" | "repaired"
```

### Pattern 3: H5 Output Structure

`midlines_reid.h5` is a byte-identical copy of `midlines_stitched.h5` with:
1. `/midlines/fish_id` dataset modified in-place (same propagation logic as `apply_swap_repairs()`)
2. `/midlines/swap_events` preserved (copied from source)
3. `/reid_events/` — new top-level group containing a structured dataset of `ReidEvent` records

The new `/reid_events/` should be a **top-level group** (`f["/reid_events"]`), not under `/midlines/`, to match the CONTEXT.md spec and keep it distinct from the body-length swap events.

```python
dt = np.dtype([
    ("frame", np.int32),
    ("fish_a", np.int32),
    ("fish_b", np.int32),
    ("cosine_margin", np.float32),
    ("detection_mode", h5py.string_dtype()),
    ("action", h5py.string_dtype()),
])
```

### Pattern 4: Embeddings.npz Indexing

`embeddings.npz` stores parallel flat arrays: `frame_index`, `fish_id`, `camera_id`, `embeddings`. To look up embeddings for a specific (frame_range, fish_id) tuple efficiently:

```python
# Build index once
npz = np.load(embeddings_path, allow_pickle=True)
frames = npz["frame_index"]   # shape (N,)
fish_ids = npz["fish_id"]     # shape (N,)
embs = npz["embeddings"]      # shape (N, 768)

# Lookup: mean embedding for fish_id=fid in frame range [f_start, f_end]
mask = (fish_ids == fid) & (frames >= f_start) & (frames <= f_end)
if mask.any():
    mean_emb = embs[mask].mean(axis=0)
    mean_emb /= np.linalg.norm(mean_emb)  # re-normalize after mean
```

Re-normalization after averaging is important — the mean of unit vectors is not unit.

### Pattern 5: Cross-Pattern Confirmation

The swap confirmation logic (not self-similarity drop):

```python
def _confirm_swap(
    emb_a_pre: np.ndarray,   # mean pre-event embedding for fish A
    emb_a_post: np.ndarray,  # mean post-event embedding for fish A
    emb_b_pre: np.ndarray,   # mean pre-event embedding for fish B
    emb_b_post: np.ndarray,  # mean post-event embedding for fish B
    threshold: float,
) -> tuple[bool, float]:
    """Returns (confirmed, cosine_margin)."""
    # Cross similarities: does A-post match B-pre, and B-post match A-pre?
    cross_a = float(np.dot(emb_a_post, emb_b_pre))   # should be high if swapped
    cross_b = float(np.dot(emb_b_post, emb_a_pre))   # should be high if swapped
    # Self similarities: does A-post still match A-pre?
    self_a = float(np.dot(emb_a_post, emb_a_pre))
    self_b = float(np.dot(emb_b_post, emb_b_pre))

    margin_a = cross_a - self_a   # positive = A-post looks more like B-pre
    margin_b = cross_b - self_b   # positive = B-post looks more like A-pre
    cosine_margin = min(margin_a, margin_b)
    return cosine_margin > threshold, cosine_margin
```

### Pattern 6: Repair Propagation

Adapt `apply_swap_repairs()` from `stitching.py` for multi-event chained propagation. Key insight from existing code: apply events in chronological order, each swap propagates from its frame until the next swap event (or end of array). The existing implementation applies all confirmed swaps to the entire tail — for chained events, process them sequentially in order.

```python
# Existing pattern (from apply_swap_repairs):
for swap in sorted_swaps:
    start_row = np.searchsorted(frame_index, swap.frame)
    # swap fish_a <-> fish_b in all rows from start_row onward
```

For Phase 105, the same logic applies but:
- Source is `midlines_stitched.h5` (read-only)
- Destination is `midlines_reid.h5` (new copy, modify in-place)
- Events are `ReidEvent` with `action == "repaired"`

### Anti-Patterns to Avoid
- **In-place modification of `midlines_stitched.h5`:** Phase 105 must write a new `midlines_reid.h5`. Never modify the source.
- **Re-normalizing before mean but not after:** Take mean, then normalize. Normalizing individual embeddings before mean doesn't help if they point in similar but not identical directions.
- **Using self-similarity drop alone:** A fish moving through a structurally similar environment can cause self-similarity drop. The cross-pattern confirmation is mandatory.
- **Ignoring camera averaging in pre/post windows:** The mean embedding should average across all cameras AND all frames in the window — this is more robust than single-camera comparisons.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Structured H5 dtype | Custom serialization | `np.dtype` with named fields + `h5py.string_dtype()` for string columns | Already demonstrated in `_write_swap_events()` |
| Fish ID propagation | Custom swap relay | Adapt `apply_swap_repairs()` from `stitching.py` (lines 679-737) | Tested, handles multi-slot rows, sequential ordering |
| Crop extraction for on-the-fly embedding | New extraction logic | Reuse `EmbedRunner` crop extraction pattern (runner.py lines 283-364) | Already handles OBB alignment, coasted frame skipping, chunk cache lookup |
| H5 file copy | Streaming H5 copy | `shutil.copy2(src, dst)` then open dst in `r+` | Preserves all datasets/attributes including `/midlines/swap_events` |

---

## Common Pitfalls

### Pitfall 1: H5 swap_events path confusion
**What goes wrong:** CONTEXT.md says `/swap_events/` but the actual H5 stores body-length events at `/midlines/swap_events` (a dataset under the `midlines` group, not a top-level group).
**Why it happens:** CONTEXT.md uses shorthand notation.
**How to avoid:** Read via `f["midlines"]["swap_events"]` for the body-length seeded events. Write new `/reid_events/` as `f.create_group("reid_events")` or `f.create_dataset("reid_events", data=arr)` at top level.

### Pitfall 2: Empty embedding window
**What goes wrong:** A swap event is near the start or end of the video; the pre- or post-event window has fewer than expected frames, or zero crops.
**Why it happens:** Window extends beyond video bounds or no fish visible in that window.
**How to avoid:** Fall back to whatever frames are available; if fewer than e.g. 3 frames are found, skip confirmation (log a warning, mark action as "skipped_insufficient_data").

### Pitfall 3: embeddings.npz frame coverage gaps
**What goes wrong:** On-the-fly mode builds embeddings only for window frames; opportunistic reuse assumes frame exists in npz but it may not (e.g., fish not detected in that frame).
**Why it happens:** Some frames have no valid detections for a given fish_id.
**How to avoid:** Always check mask.any() before computing mean. Fall back gracefully.

### Pitfall 4: Fish ID remapping after repair
**What goes wrong:** After applying a repair, a second swap event involving one of the repaired fish IDs may reference the pre-repair ID.
**Why it happens:** Events are detected before repair; the swap-then-swap-back scenario needs careful ordering.
**How to avoid:** Apply events strictly in chronological order (matching existing `apply_swap_repairs` pattern). The caller is responsible for ordering.

### Pitfall 5: Cross-pattern margin sign confusion
**What goes wrong:** Computing margin as (cross - self) where cross similarity is low and self is high gives a large negative margin, which would be correctly rejected — but if the formula is inverted, everything gets confirmed.
**Why it happens:** Cosine similarity is bounded [-1, 1]; higher = more similar. Margin > 0 means cross > self, i.e., swapped embedding is more similar to the other fish than to itself.
**How to avoid:** Write a unit test with synthetic embeddings that represent a clear swap (orthogonal pre/post for each fish) and verify the margin sign is positive and above threshold.

### Pitfall 6: string dtype in H5 structured array
**What goes wrong:** Storing string columns (`detection_mode`, `action`) in a structured numpy dtype fails with a plain `np.dtype` using `str`.
**Why it happens:** numpy uses fixed-width bytes for string dtypes by default; variable-length strings need `h5py.string_dtype()` or `h5py.special_dtype(vlen=str)`.
**How to avoid:** Use `h5py.string_dtype()` for string columns in the structured dtype, or use separate datasets for string fields.

### Pitfall 7: Z-axis anisotropy in proximity threshold
**What goes wrong:** 3D centroid distance for proximity detection uses raw Euclidean distance, but Z is ~2.9x less reliable than XY.
**Why it happens:** Z/XY anisotropy documented in memory: ~2.9x mean.
**How to avoid:** Apply the Z_WEIGHT = 1.0/2.9 factor (already defined in stitching.py) when computing centroid-to-centroid distance for proximity detection. Or use the `_weighted_dist()` function already in stitching.py.

---

## Code Examples

### Reading body-length swap events from midlines_stitched.h5

```python
# Source: src/aquapose/core/stitching.py (pattern used throughout)
import h5py
import numpy as np

with h5py.File(h5_path, "r") as f:
    grp = f["midlines"]
    if "swap_events" in grp:
        se = grp["swap_events"][()]
        # se.dtype has fields: frame, fish_a, fish_b, score_a, score_b, auto_corrected
        seeded_candidates = [
            SwapEvent(
                frame=int(row["frame"]),
                fish_a=int(row["fish_a"]),
                fish_b=int(row["fish_b"]),
                score_a=float(row["score_a"]),
                score_b=float(row["score_b"]),
                auto_corrected=bool(row["auto_corrected"]),
            )
            for row in se
        ]
```

### Writing /reid_events/ to midlines_reid.h5

```python
# Adapted from _write_swap_events in stitching.py
import h5py
import numpy as np

def _write_reid_events(h5_path: Path, events: list[ReidEvent]) -> None:
    arr_data = [
        (e.frame, e.fish_a, e.fish_b, e.cosine_margin,
         e.detection_mode.encode(), e.action.encode())
        for e in events
    ]
    # Use separate arrays to avoid structured dtype string issues
    with h5py.File(h5_path, "r+") as f:
        if "reid_events" in f:
            del f["reid_events"]
        grp = f.create_group("reid_events")
        grp.create_dataset("frame", data=np.array([e.frame for e in events], dtype=np.int32))
        grp.create_dataset("fish_a", data=np.array([e.fish_a for e in events], dtype=np.int32))
        grp.create_dataset("fish_b", data=np.array([e.fish_b for e in events], dtype=np.int32))
        grp.create_dataset("cosine_margin", data=np.array([e.cosine_margin for e in events], dtype=np.float32))
        grp.create_dataset("detection_mode", data=np.array([e.detection_mode for e in events], dtype=h5py.string_dtype()))
        grp.create_dataset("action", data=np.array([e.action for e in events], dtype=h5py.string_dtype()))
```

### Producing midlines_reid.h5 as a copy

```python
import shutil

def _make_reid_copy(src: Path, dst: Path) -> None:
    """Copy midlines_stitched.h5 to midlines_reid.h5 preserving all data."""
    shutil.copy2(src, dst)
```

Then open `dst` in `r+` mode to apply fish_id corrections and write `/reid_events/`.

### Proximity-based candidate event detection

```python
# Read 3D centroids from H5 (points dataset, shape (N, max_fish, n_kpts, 3))
# Compute per-frame pairwise fish distances
def _find_proximity_events(
    frame_index: np.ndarray,   # (N,)
    fish_id: np.ndarray,       # (N, max_fish)
    points: np.ndarray,        # (N, max_fish, n_kpts, 3)
    threshold_m: float,
) -> list[tuple[int, int, int]]:  # (frame, fish_a, fish_b)
    """Find frames where two fish centroids are within threshold_m (meters)."""
    from aquapose.core.stitching import Z_WEIGHT
    events = []
    n_frames, max_fish = fish_id.shape
    for row in range(n_frames):
        fish_slots = [(fish_id[row, s], s) for s in range(max_fish) if fish_id[row, s] >= 0]
        for i, (fid_a, slot_a) in enumerate(fish_slots):
            for fid_b, slot_b in fish_slots[i+1:]:
                pts_a = points[row, slot_a]
                pts_b = points[row, slot_b]
                valid_a = ~np.isnan(pts_a).any(axis=1)
                valid_b = ~np.isnan(pts_b).any(axis=1)
                if not valid_a.any() or not valid_b.any():
                    continue
                c_a = pts_a[valid_a].mean(axis=0)
                c_b = pts_b[valid_b].mean(axis=0)
                diff = c_a - c_b
                diff[2] *= Z_WEIGHT
                dist = float(np.linalg.norm(diff))
                if dist < threshold_m:
                    events.append((int(frame_index[row]), int(fid_a), int(fid_b)))
    return events
```

---

## Implementation Plan

### Recommended Plan Split

**Plan 105-01: Core swap detector module**
- `core/reid/swap_detector.py`: `ReidEvent` dataclass, `SwapDetectorConfig`, `SwapDetector` class
- Seeded mode: reads `/midlines/swap_events`, extracts embedding windows, applies cross-pattern check
- Scan mode: proximity detection, dense embedding around events, sliding-window gap fill, embeddings.npz opportunistic reuse
- Repair: writes `midlines_reid.h5` (copy + fish_id corrections + `/reid_events/`)
- Update `core/reid/__init__.py`
- Unit tests: `tests/unit/core/reid/test_swap_detector.py` covering pure functions (cross-pattern confirm, proximity detection, margin computation)

**Plan 105-02: Standalone validation script**
- `scripts/detect_swaps.py`: hardcoded YH paths, runs both modes, compares against known events (MF at ~2665, FF at ~600), computes FP rate on clean segments, prints console validation report

### Key Design Decisions for Planner

1. **Where does `ReidEvent` live?** In `core/reid/swap_detector.py`. No need to move to `core/stitching.py`.

2. **Crop extraction for on-the-fly embedding in swap detector:** The swap detector needs to extract crops for specific (frame, fish_id, camera) tuples. The `EmbedRunner` private methods do this but are tightly coupled to chunk cache iteration. Best approach: factor out the crop extraction logic into a small helper (or call `FishEmbedder` directly with crops extracted via the same pattern as `EmbedRunner`).

3. **Proximity threshold default:** 0.15 meters (15 cm) is a reasonable default — fish body length is ~10-15 cm, so fish that are within one body-length of each other can realistically occlude and swap. Z_WEIGHT should be applied.

4. **Sliding window size for gap-fill:** 50 frames (≈1.7 seconds at 30 fps) is a reasonable default for scan mode gap-filling — covers brief inter-proximity intervals.

5. **embeddings.npz indexing:** Load the full file once into memory at `SwapDetector.__init__` time if opportunistic reuse is enabled. The file is typically a few hundred MB at most for a 9450-frame run.

---

## Open Questions

1. **Does the fine-tuned projection head affect embeddings for Phase 105?**
   - What we know: Phase 104 trained a projection head on top of the 768-dim backbone features. The backbone `embeddings.npz` was written by Phase 102's `EmbedRunner` (raw MegaDescriptor-T, 768-dim).
   - What's unclear: Was `embeddings.npz` re-embedded through the projection head after Phase 104? The `train_reid_head.py` script mentions re-embedding via the projection head.
   - Recommendation: Phase 105 should accept an optional `projection_head_path` parameter and, if provided, apply the projection head to raw embeddings before comparison. Default to raw 768-dim embeddings (zero-shot mode). The standalone script can hardcode the best projection head path from Phase 104.

2. **What frame does the FF swap (fish 2 <-> 4) actually occur?**
   - What we know: CONTEXT.md says "frame ~600". The Phase 102 eval report (mentioned in STATE.md) shows Fish 2↔8 as most confusable, not Fish 2↔4. The comment in `train_reid_head.py` says "Known swap: fish 2 <-> fish 4 around frame 600".
   - What's unclear: Whether this is the swap frame or the detection frame, and whether the FF swap is pre- or post-stitch.
   - Recommendation: The standalone script should scan a range of frames (~550-650) when validating FF swap detection rather than requiring an exact match.

3. **Is there a `VideoFrameSource` available for on-the-fly crop extraction outside of `EmbedRunner.run()`?**
   - What we know: `VideoFrameSource` is imported in `EmbedRunner.run()` from `core.types.frame_source`. It requires `video_dir` and `calibration_path` from `run_config`.
   - Recommendation: The `SwapDetector` should read `config.yaml` from `run_dir` (same pattern as `EmbedRunner._load_run_config()`) to get video paths for on-the-fly embedding.

---

## Sources

### Primary (HIGH confidence)
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/reid/embedder.py` — FishEmbedder interface, L2-normalized 768-dim output
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/reid/runner.py` — EmbedRunner: crop extraction pattern, embeddings.npz structure, frame_source usage
- `/home/tlancaster6/Projects/AquaPose/src/aquapose/core/stitching.py` — SwapEvent, detect_and_repair_swaps, apply_swap_repairs, _write_swap_events, _weighted_dist (Z_WEIGHT)
- `/home/tlancaster6/Projects/AquaPose/scripts/train_reid_head.py` — confirms re-embedding logic, known swap frame reference
- `/home/tlancaster6/Projects/AquaPose/.planning/phases/105-swap-detection-and-repair/105-CONTEXT.md` — all locked decisions

### Secondary (MEDIUM confidence)
- Memory notes: Z/XY anisotropy (~2.9x), fish sex labels (FEMALE_IDS = {0,1,2,3,4,8})

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in use
- Architecture: HIGH — patterns are direct adaptations of existing code
- Pitfalls: HIGH — grounded in code review, not speculation
- Open questions: MEDIUM — require running code or Phase 104 output inspection to fully resolve

**Research date:** 2026-03-25
**Valid until:** 2026-04-25 (stable codebase, no fast-moving deps)
