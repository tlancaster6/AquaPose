# Phase 83: Custom Tracker Implementation - Research

**Researched:** 2026-03-10
**Domain:** Multi-object keypoint tracking — custom Kalman filter, OKS association, bidirectional merge, chunk handoff, gap interpolation
**Confidence:** HIGH (all findings grounded in project codebase + well-established algorithms)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Kalman filter state**
- 60-dim state: Track all 6 keypoints (nose, head, spine1, spine2, spine3, tail) × 2D × (position + velocity)
- Confidence-scaled measurement noise: R_i = base_R / max(conf_i, epsilon). Low-confidence keypoints get high measurement noise, causing the KF to rely on prediction instead. Smooth degradation, no hard masking or special cases.
- From scratch: NumPy-based constant-velocity KF, no external library (filterpy, etc.). Full control over state layout, confidence-scaled R, and serialization for chunk handoff.
- Coast prediction: Predict all 6 keypoints during coast. Use full predicted keypoint state for OKS cost when matching coasting tracks to new detections.

**OKS cost design**
- Empirical per-keypoint sigmas: Compute sigmas from manual annotation dataset — measure per-keypoint position variance normalized by fish scale. Expect: endpoints (nose, tail) get larger sigmas, mid-body (spine1-3) get smaller sigmas.
- Confidence-weighted OKS: Weight each keypoint's OKS contribution by detection confidence: `oks = sum(c_k * exp(-d_k^2 / (2*s^2*sigma_k^2))) / sum(c_k)`. Occluded endpoints (low confidence) contribute near-zero weight.
- Scale term: sqrt(OBB area), analogous to COCO's bbox area normalization.
- OKS + OCM as weighted sum: `cost = (1 - OKS) + lambda * (1 - OCM)`. OCM uses cosine similarity of spine heading vector (spine1→spine3). Lambda is tunable (0.1–0.3 range). OKS dominates; OCM breaks ties and penalizes head-tail flips.

**Bidirectional merge strategy**
- Temporal overlap + OKS matching: Find temporal overlap between forward and backward tracklets, compute mean OKS in the overlap region, match pairs via Hungarian assignment on OKS cost matrix.
- Overlap frame resolution: Keep detected over coasted. If both detected, keep the one with higher mean keypoint confidence. Both coasted → keep either.
- Independent passes: Backward pass runs with its own ID space, no seeding from forward results. Merge step handles identity unification.
- Unmatched tracklets: Keep if they meet minimum length threshold (n_init frames). Discard short unmatched fragments as likely false positives.

**Birth/death rules**
- Uniform rules — no spatial edge asymmetry: Same n_init and max_age everywhere. Bidirectional merge naturally recovers tracks that one pass missed (forward catches entries, backward catches exits). No special treatment for detections near frame borders.
- max_age = 15: Lower than current OC-SORT default (30). OKS matching is more discriminative — tracks that can't be matched within 15 frames are likely genuinely lost. Configurable.
- TRACK-05 (asymmetric birth/death): Satisfied by the bidirectional merge design — forward+backward naturally provides asymmetric coverage without explicit spatial-edge rules.
- TRACK-10 (BYTE-style secondary pass): Deferred to Phase 84 evaluation. Build primary tracker with single confidence threshold first. If Phase 84 shows missed detections, add BYTE-style pass then.

### Claude's Discretion
- ORU/OCR mechanism details (TRACK-06) — how observation-centric re-update and recovery are implemented within the custom KF
- Gap interpolation method (TRACK-09) — spline type and maximum gap length for interpolation
- n_init default value and tuning range
- Process noise (Q) tuning for the constant-velocity KF model
- OKS sigma computation methodology (exact script/approach for deriving from annotations)
- Internal data structures for the tracker (track pool, tentative vs confirmed lists)

### Deferred Ideas (OUT OF SCOPE)
- TRACK-10 (BYTE-style secondary pass for low-confidence detections) — deferred to Phase 84 evaluation. Implement only if Phase 84 metrics show missed detections that a secondary pass would recover.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRACK-01 | Custom keypoint tracker runs forward and backward OC-SORT passes over each chunk | KF design, pass architecture; both passes are per-camera, same OcSortTracker interface pattern |
| TRACK-02 | OKS-based association cost replaces IoU on OBBs | OKS formula, per-keypoint sigma computation from annotations, scale term from OBB area |
| TRACK-03 | OCM direction consistency term using spine heading vector | Cosine similarity of spine1→spine3 vector; lambda-weighted sum with OKS |
| TRACK-04 | Kalman filter tracks keypoint positions and velocities (60-dim state) | 60-dim constant-velocity KF: 6 kpts × 2D × (pos + vel); NumPy implementation patterns |
| TRACK-05 | Asymmetric track birth/death based on frame-edge proximity | Satisfied by bidi merge design per CONTEXT.md locked decision |
| TRACK-06 | ORU (observation-centric re-update) and OCR (observation-centric recovery) mechanisms | OC-SORT paper mechanics; adapted to keypoint state (re-update KF with associated observation after assignment; recovery searches for matching observation in recent unmatched pool) |
| TRACK-07 | Bidirectional merge combines forward and backward tracklets with overlap-based matching | Hungarian assignment on overlap-OKS cost matrix; frame resolution policy |
| TRACK-08 | Chunk boundary handoff via serialized KF state (mean + covariance + observation history) | NumPy array serialization in dict; ChunkHandoff.tracks_2d_state carries it; from_state reconstruction |
| TRACK-09 | Gap interpolation fills small tracklet gaps via spline interpolation | scipy.interpolate.CubicSpline on (frame_idx, keypoint_xy) per-coordinate; max_gap_frames threshold |
| TRACK-10 | BYTE-style secondary pass (DEFERRED — conditional on Phase 84 findings) | DEFERRED per locked decision |
</phase_requirements>

## Summary

Phase 83 builds a custom bidirectional batched keypoint tracker that replaces OC-SORT/boxmot as the per-camera 2D tracker. The core innovations are: (1) OKS association cost instead of IoU, using per-keypoint sigmas derived from the project's manual annotation dataset; (2) a 60-dimensional NumPy Kalman filter tracking all 6 keypoints' positions and velocities with confidence-scaled measurement noise; (3) independent forward and backward passes that merge via Hungarian assignment over temporal overlap regions; and (4) serializable KF state for chunk boundary handoff. The tracker produces the same `Tracklet2D` output contract as the current `OcSortTracker`, so all downstream stages (AssociationStage, ReconstructionStage) are unchanged.

The existing codebase provides excellent scaffolding. `OcSortTracker` in `ocsort_wrapper.py` is the template — the new `KeypointTracker` class follows exactly the same interface (`update`, `get_tracklets`, `get_state`, `from_state`) and the same `_TrackletBuilder` accumulation pattern. `TrackingStage.run()` already dispatches by `tracker_kind`; adding `"keypoint_bidi"` requires only a conditional import. `TrackingConfig` needs new fields for OKS params, KF params, and bidi merge params. The test suite in `tests/unit/core/tracking/test_tracking_stage.py` is the template for all new tests.

The primary algorithmic challenge is the bidirectional merge. Forward and backward tracklets share overlapping frames; the merge must resolve conflicts frame-by-frame using the `detected > coasted, higher-confidence wins` policy, then produce a unified ID space. ORU/OCR from OC-SORT adapt naturally: ORU re-updates the KF with the matched observation after assignment (correcting prediction error); OCR scans a short observation buffer for a better match during recovery. Both operate on keypoint positions rather than bounding-box centroids.

**Primary recommendation:** Implement the tracker in three tightly-scoped tasks — KF + OKS cost + single-pass core; bidirectional merge + chunk handoff; gap interpolation + sigma computation script + config wiring.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | >=1.24 (project dep) | 60-dim KF state, OKS computation, Hungarian cost matrix construction | Already a project dependency; KF is pure matrix algebra |
| scipy.optimize.linear_sum_assignment | >=1.11 (project dep) | Hungarian assignment for detection-to-track matching and bidi merge | Already used in project; scipy is a project dependency |
| scipy.interpolate.CubicSpline | >=1.11 (project dep) | Gap interpolation over keypoint positions | Already used in reconstruction module for spline fitting |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.optimize.linear_sum_assignment | same | Bidi merge step — Hungarian on OKS cost matrix between forward and backward tracklets | Merge step, same as detection-track assignment |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy.optimize.linear_sum_assignment | lapsolver / lapjv | Faster for large N, but N=9 fish — scipy sufficient |
| CubicSpline | UnivariateSpline | UnivariateSpline smooths; CubicSpline interpolates exactly — prefer CubicSpline for gap-fill accuracy |
| NumPy KF | filterpy | filterpy cleaner API but locked decision says from-scratch for full control |

**Installation:** No new packages required. All dependencies are already in `pyproject.toml`.

## Architecture Patterns

### Recommended Project Structure

```
src/aquapose/core/tracking/
├── __init__.py                  # add KeypointTracker to __all__
├── types.py                     # unchanged — Tracklet2D is the output contract
├── stage.py                     # add tracker_kind dispatch for "keypoint_bidi"
├── ocsort_wrapper.py            # unchanged — retained for "ocsort" tracker_kind
├── keypoint_tracker.py          # NEW: KeypointTracker, _KFTrack, _TrackletBuilder (extended)
└── keypoint_sigmas.py           # NEW: compute_keypoint_sigmas() from annotation dataset

tests/unit/core/tracking/
├── test_tracking_stage.py       # unchanged — existing tests must remain green
├── test_keypoint_tracker.py     # NEW: tests mirroring test_tracking_stage.py pattern
└── test_keypoint_sigmas.py      # NEW: sigma computation from synthetic annotations
```

### Pattern 1: Single-Pass Core (mirroring OcSortTracker)

**What:** A single forward or backward pass over one chunk. Maintains a pool of `_KFTrack` objects. Each frame: predict all active tracks, compute OKS cost matrix, run Hungarian assignment, update matched tracks (with ORU), start new tracks, increment age of unmatched tracks (OCR check), cull dead tracks.

**When to use:** Called twice per chunk — forward over frames [0..N-1], backward over frames [N-1..0]. Results are accumulated in separate `_TrackletBuilder` instances.

```python
# Source: project pattern from ocsort_wrapper.py + standard KF update
class _SinglePassTracker:
    def __init__(self, camera_id: str, direction: str, config: KeypointTrackerConfig) -> None:
        self.direction = direction  # "forward" or "backward"
        self._tracks: list[_KFTrack] = []
        self._builders: dict[int, _TrackletBuilder] = {}
        self._next_id = 0

    def update(self, frame_idx: int, detections: list[Detection]) -> None:
        # 1. predict() all active tracks (advance KF state)
        # 2. Build cost matrix (rows=tracks, cols=detections) using OKS + OCM
        # 3. linear_sum_assignment to get matched pairs
        # 4. For matched pairs: KF update + ORU re-update + add to builder as "detected"
        # 5. For unmatched tracks: OCR check, increment age, add to builder as "coasted"
        # 6. For unmatched detections: create new tentative tracks
        # 7. Cull tracks with time_since_update > max_age
        ...

    def get_tracklets(self) -> list[Tracklet2D]: ...
    def get_state(self) -> dict: ...  # KF mean, cov, obs_history per track
```

### Pattern 2: Kalman Filter (60-dim constant-velocity)

**What:** NumPy-based constant-velocity Kalman filter. State vector `x` is `[kp0_x, kp0_y, kp1_x, kp1_y, ..., kp5_x, kp5_y, v_kp0_x, v_kp0_y, ..., v_kp5_x, v_kp5_y]` = 24 positions + 24 velocities = 48-dim.

**Note:** CONTEXT.md says "60-dim" but the math yields 48-dim (6 kpts × 2 coords × 2 state components = 48). The CONTEXT.md likely means "track all 6 keypoints' positions + velocities" conceptually. The state dimension is explicitly 48 and should be documented as such.

**Measurement vector:** 12-dim `[kp0_x, kp0_y, ..., kp5_x, kp5_y]` — just positions observed.

```python
# Source: standard constant-velocity KF derivation
# State: 48-dim [pos_0..5, vel_0..5], Measurement: 12-dim [pos_0..5]
# F: transition matrix (I_12 with dt*I_12 in upper-right block)
# H: measurement matrix [I_12 | 0_12] extracts positions from state
# Q: process noise (small for velocity, larger for position)
# R: per-keypoint confidence-scaled (diagonal, 12x12)
#    R[2k, 2k] = R[2k+1, 2k+1] = base_R / max(conf_k, epsilon)

class _KalmanFilter:
    def __init__(self, initial_obs: np.ndarray, confs: np.ndarray, base_R: float = 10.0) -> None:
        # x shape (48,), P shape (48, 48)
        self.x = np.zeros(48, dtype=np.float64)
        self.x[:12] = initial_obs.ravel()   # positions from first detection
        self.P = np.eye(48) * 100.0         # large initial uncertainty
        self._F = _build_F()                # block structure: [[I, I], [0, I]]
        self._H = _build_H()                # [I_12 | 0_12]
        self._Q = _build_Q()                # process noise
        self._base_R = base_R

    def predict(self) -> np.ndarray:
        """Advance state. Returns predicted positions (12-dim)."""
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q
        return self.x[:12].reshape(6, 2)

    def update(self, obs: np.ndarray, confs: np.ndarray) -> None:
        """Incorporate observation with confidence-scaled R."""
        R = self._build_R(confs)
        z = obs.ravel()           # 12-dim
        y = z - self._H @ self.x  # innovation
        S = self._H @ self.P @ self._H.T + R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(48) - K @ self._H) @ self.P

    def _build_R(self, confs: np.ndarray) -> np.ndarray:
        """Confidence-scaled measurement noise (12x12 diagonal)."""
        eps = 1e-6
        diag = np.repeat(self._base_R / np.maximum(confs, eps), 2)  # shape (12,)
        return np.diag(diag)
```

**Important clarification:** Document in code that "60-dim" in CONTEXT.md is a conceptual label (tracking all 6 kpts' full state); the actual state vector is 48-dim. The planner should create a task that explicitly documents this.

### Pattern 3: OKS + OCM Cost Function

**What:** Compute pairwise cost matrix between N tracks and M detections. Vectorizable over detections for each track.

```python
# Source: COCO OKS definition adapted for fish keypoints
# sigmas: shape (6,) — per-keypoint spread from annotation variance computation
def compute_oks_cost(
    pred_kpts: np.ndarray,   # (N, 6, 2) — predicted keypoint positions per track
    det_kpts: np.ndarray,    # (M, 6, 2) — detected keypoint positions
    det_confs: np.ndarray,   # (M, 6)    — per-keypoint detection confidences
    det_scales: np.ndarray,  # (M,)      — sqrt(OBB area) per detection
    sigmas: np.ndarray,      # (6,)      — per-keypoint sigma values
    lambda_ocm: float = 0.2,
    det_headings: np.ndarray | None = None,  # (M, 2) — spine1→spine3 vectors
    pred_headings: np.ndarray | None = None, # (N, 2)
) -> np.ndarray:             # (N, M) cost matrix
    # For each (i, j) pair:
    # d_k = ||pred_kpts[i,k] - det_kpts[j,k]||^2
    # oks_k = c_k * exp(-d_k / (2 * det_scales[j]^2 * sigmas[k]^2))
    # oks = sum(oks_k) / sum(c_k)
    # ocm = cosine_similarity(pred_headings[i], det_headings[j])
    # cost[i,j] = (1 - oks) + lambda_ocm * (1 - ocm)
    ...
```

### Pattern 4: Bidirectional Merge

**What:** Merge forward and backward tracklet lists for one camera into a unified list. Match tracklets that overlap temporally using mean OKS in the overlap window.

```python
def merge_forward_backward(
    forward: list[Tracklet2D],
    backward: list[Tracklet2D],
    sigmas: np.ndarray,
    min_overlap: int = 3,    # min frames of overlap to attempt matching
    min_length: int = 3,     # min frames to keep an unmatched tracklet
) -> list[Tracklet2D]:
    # 1. Build overlap-frame OKS cost matrix (N_fwd x N_bwd)
    # 2. linear_sum_assignment on cost matrix
    # 3. For matched pairs:
    #    - For overlapping frames: keep detected > coasted; else higher mean conf
    #    - Concatenate non-overlapping parts
    # 4. Keep unmatched tracklets that meet min_length
    # 5. Re-assign monotonically increasing IDs to final tracklet list
    ...
```

### Pattern 5: ORU and OCR Mechanics

**What:** Observation-centric mechanisms from the OC-SORT paper, adapted to keypoint state.

**ORU (Observation-Centric Re-Update):**
After the main KF update (Kalman step), the "virtual trajectory" between the last observation and the current one may have drifted. ORU recalculates the KF update using the raw observation directly (bypassing accumulated coasting drift). In practice: when a coasting track is matched to a new detection after K missed frames, run K prediction steps on a fresh copy of the pre-coast state, then update with the detection. The "re-updated" state is the canonical state going forward.

**OCR (Observation-Centric Recovery):**
When a track is in coast mode, maintain a short ring buffer (`obs_history`, last 5 observations). On each coast step, attempt matching against the buffer — if a buffered observation matches (OKS > threshold), recover the track using that observation. This catches tracks that were briefly dropped during occlusion.

Both mechanisms require the `_KFTrack` to store `obs_history: deque[tuple[np.ndarray, np.ndarray]]` (observations + confidences, maxlen=5) alongside the KF state.

### Pattern 6: Chunk Handoff Serialization

**What:** Serialize the complete tracker state per-camera into a `dict` that can be stored in `ChunkHandoff.tracks_2d_state`. On the next chunk, `from_state()` reconstructs the tracker with all active tracks restored.

```python
def get_state(self) -> dict:
    return {
        "active_tracks": [
            {
                "track_id": t.track_id,
                "kf_mean": t.kf.x.tolist(),        # 48-dim float list
                "kf_cov": t.kf.P.tolist(),          # 48x48 float list
                "obs_history": [                     # list of (obs, conf) tuples
                    (o.tolist(), c.tolist()) for o, c in t.obs_history
                ],
                "time_since_update": t.time_since_update,
                "hit_streak": t.hit_streak,
                "detected_count": t.detected_count,
            }
            for t in self._active_tracks
        ],
        "next_id": self._next_id,
        "camera_id": self.camera_id,
        # ... config params for reconstruction
    }
```

The critical property is that `kf_mean` and `kf_cov` are plain Python lists — JSON-serializable and pickle-safe across Python sessions. This is safer than storing raw numpy arrays which can cause issues with pickle versioning.

### Pattern 7: Gap Interpolation

**What:** After the bidirectional merge, scan each tracklet's `frames` tuple for gaps (non-consecutive indices). For gaps shorter than `max_gap_frames`, fill using `scipy.interpolate.CubicSpline` per coordinate per keypoint. Produce "interpolated" frame status.

```python
from scipy.interpolate import CubicSpline

def interpolate_gaps(
    tracklet: Tracklet2D,
    max_gap_frames: int = 5,
) -> Tracklet2D:
    """Fill short gaps in a tracklet via cubic spline interpolation.

    Args:
        tracklet: Source tracklet (may have non-consecutive frames).
        max_gap_frames: Maximum gap size (in frames) to fill. Gaps larger
            than this are left as-is.

    Returns:
        New Tracklet2D with gap frames inserted and status "interpolated".
    """
    # Per-coordinate cubic spline on detected+coasted frames
    # Evaluate at missing frame indices to fill gaps
    ...
```

Note: Tracklet2D currently has `frame_status` values `"detected"` and `"coasted"`. Gap interpolation introduces a third value `"interpolated"`. This must be documented; downstream consumers (TrackingMetrics, AssociationStage) need to handle or ignore it. The simplest approach is to treat `"interpolated"` as `"coasted"` in `evaluate_tracking()`.

### Anti-Patterns to Avoid

- **Importing filterpy or any external KF library:** Locked decision is from-scratch NumPy KF.
- **Storing numpy arrays directly in get_state():** Use `.tolist()` for KF state to ensure pickle/JSON safety across runs.
- **Running the forward and backward passes sequentially in a single `update()` loop:** Keep them completely independent. The forward pass accumulates all frames 0..N-1, then the backward pass accumulates frames N-1..0 separately, then merge.
- **Calling `get_tracklets()` before the chunk ends:** Both passes must complete fully before merge; don't mix partial pass results.
- **Mutating Tracklet2D during merge:** Tracklet2D is frozen. Build a new one via `_TrackletBuilder.to_tracklet2d()` after merging frame lists.
- **Using frame_idx directly as list index during backward pass:** The backward pass iterates frames in reverse but still stores `frame_idx` values correctly (N-1, N-2, ..., 0) for alignment with the forward pass.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Hungarian assignment | Custom greedy matching | scipy.optimize.linear_sum_assignment | Optimal O(n³), already a project dependency |
| Spline gap interpolation | Linear interpolation | scipy.interpolate.CubicSpline | Handles non-uniform frame spacing; already used in reconstruction |
| Cost matrix vectorization | Python loops over track-detection pairs | NumPy broadcasting | Broadcasting over (N, M, 6) arrays is 100x faster than Python loops |

**Key insight:** The only truly novel code in this phase is the KF equations, OKS formula, and merge policy. Everything else reuses existing project infrastructure.

## Common Pitfalls

### Pitfall 1: State Dimension Confusion (60 vs 48)
**What goes wrong:** CONTEXT.md says "60-dim state" but 6 keypoints × 2D × (position + velocity) = 6 × 2 × 2 = 24... wait, that is 24 positions + 24 velocities = 48. The "60-dim" in CONTEXT.md appears to be an approximation or includes a 6-dim orientation state. Research finding: the mathematically consistent dimension is 48-dim for positions+velocities only.
**Why it happens:** "60-dim" may have been computed as 6 kpts × 10 (some per-kpt state), or is simply a miscalculation.
**How to avoid:** Implement and document as 48-dim. Add an explicit comment in the code stating "60-dim per CONTEXT.md discussion, implemented as 48-dim: 6 kpts × 2D × 2 (pos+vel) = 48". The success criterion in the phase says "state dimension is explicitly chosen and documented" — document the actual number with derivation.
**Warning signs:** If the KF code tries to allocate a 60×60 matrix, there's a dimension error — stop and recheck.

### Pitfall 2: Backward Pass Frame Ordering
**What goes wrong:** When running the backward pass, detections are fed in reverse frame order. The `_TrackletBuilder.frames` list must still contain ascending frame indices (0, 5, 10, not 10, 5, 0) for the merge and for `Tracklet2D.frames` to be monotonically increasing.
**Why it happens:** The KF update naturally processes frames in the order given; if frames are stored as-processed (reversed), the Tracklet2D contract is violated.
**How to avoid:** After collecting backward pass results, sort each tracklet's frames array before building Tracklet2D. Or sort during `to_tracklet2d()`.
**Warning signs:** `frames[i] > frames[i-1]` assertion fails in `test_tracklet_frames_are_sequential`.

### Pitfall 3: OKS Scale Term with Zero Area
**What goes wrong:** `det_scales = sqrt(OBB area)`. If the OBB area is 0 (degenerate detection), scale = 0, causing division by zero in the OKS denominator `2 * s^2 * sigma_k^2`.
**Why it happens:** Pathological detections from YOLO can occasionally have zero or near-zero area.
**How to avoid:** Clamp `det_scales = max(sqrt(area), 1.0)` — a minimum scale of 1 pixel squared prevents NaN costs.
**Warning signs:** NaN values in cost matrix → `linear_sum_assignment` crashes.

### Pitfall 4: Merge ID Collision Between Forward and Backward Tracklets
**What goes wrong:** Both forward and backward passes assign IDs starting from 0 independently. After merge, both passes have a track_id=0. The merge must produce a new monotonically increasing ID space for the final unified tracklets.
**Why it happens:** Each `_SinglePassTracker` has its own `_next_id` counter.
**How to avoid:** After merge, re-number all output tracklets (merged + kept unmatched) with fresh IDs starting from 0.
**Warning signs:** Duplicate `track_id` values in `context.tracks_2d[cam_id]` list; downstream AssociationStage would mis-associate tracklets.

### Pitfall 5: ChunkHandoff State Must Represent Post-Merge Active Tracks
**What goes wrong:** Storing the forward-pass tracker state (only) as the handoff, losing backward-pass track info.
**Why it happens:** The `get_state()` convention from `OcSortTracker` is called on a single tracker instance; in the bidirectional case, the handoff state must reflect the merged track pool.
**How to avoid:** After merge, synthesize the handoff state from the merged active tracks — those tracks whose last frame is the chunk's last frame are the ones that may continue into the next chunk. Their KF state is taken from whichever pass produced them (or from the merged result for matched pairs).
**Warning signs:** Tracks that should continue across chunk boundary restart from ID 0 in the next chunk.

### Pitfall 6: "interpolated" frame_status in Downstream Consumers
**What goes wrong:** `evaluate_tracking()` in `evaluation/stages/tracking.py` counts only `"detected"` and `"coasted"` frame statuses. If gap-fill adds `"interpolated"` frames, coverage metrics may be wrong.
**Why it happens:** New status value not anticipated in existing consumers.
**How to avoid:** Either (a) use `"coasted"` for gap-filled frames (simpler, consistent), or (b) update `evaluate_tracking()` to handle `"interpolated"`. Option (a) is simpler and avoids breaking the test suite.

### Pitfall 7: Coasting Track Detection Index Recovery
**What goes wrong:** In `OcSortTracker`, `row[7]` of boxmot output carries the original detection index for keypoint lookup. The custom KF tracker has no boxmot — must explicitly track which detection index was matched.
**Why it happens:** Direct KF management requires explicit record-keeping of assignment results.
**How to avoid:** After Hungarian assignment, store `matched_det_idx` on each `_KFTrack` for the current frame, then use it in `_TrackletBuilder.add_frame()`.

## Code Examples

### OKS Vectorized Implementation

```python
# Source: COCO OKS definition (https://cocodataset.org/#keypoints-eval)
# Adapted for fish keypoints with confidence-weighted sum
import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_oks_matrix(
    pred_kpts: np.ndarray,   # (N, 6, 2)
    det_kpts: np.ndarray,    # (M, 6, 2)
    det_confs: np.ndarray,   # (M, 6) — per-keypoint confidences
    det_scales: np.ndarray,  # (M,) — sqrt(OBB area), clamped >= 1
    sigmas: np.ndarray,      # (6,) — per-keypoint sigmas
) -> np.ndarray:             # (N, M) — OKS similarity [0, 1]
    N, M = len(pred_kpts), len(det_kpts)
    # d_sq[n, m, k] = squared distance for keypoint k
    d_sq = np.sum(
        (pred_kpts[:, np.newaxis, :, :] - det_kpts[np.newaxis, :, :, :]) ** 2,
        axis=-1,
    )  # shape (N, M, 6)
    # scale factor per detection: 2 * s^2 * sigma_k^2, shape (M, 6)
    denom = 2.0 * (det_scales[:, np.newaxis] ** 2) * (sigmas[np.newaxis, :] ** 2)
    # exp term, shape (N, M, 6)
    exp_term = np.exp(-d_sq / denom[np.newaxis, :, :])
    # confidence-weighted OKS, shape (N, M)
    # det_confs: (M, 6) -> broadcast to (N, M, 6)
    c = det_confs[np.newaxis, :, :]  # (1, M, 6)
    weighted_sum = np.sum(c * exp_term, axis=-1)    # (N, M)
    conf_sum = np.sum(c, axis=-1)                    # (1, M)
    oks = weighted_sum / np.maximum(conf_sum, 1e-6)  # (N, M)
    return oks


def build_cost_matrix(
    oks: np.ndarray,         # (N, M) from compute_oks_matrix
    ocm: np.ndarray,         # (N, M) — cosine similarity of heading vectors
    lambda_ocm: float = 0.2,
) -> np.ndarray:             # (N, M) cost matrix [0, 2]
    return (1.0 - oks) + lambda_ocm * (1.0 - ocm)
```

### KF Matrix Construction

```python
# Source: standard constant-velocity linear KF
# State: [pos_0..5 (12), vel_0..5 (12)] — actually 24 total, interleaved as:
# [kp0_x, kp0_y, kp1_x, ..., kp5_y, v_kp0_x, v_kp0_y, ..., v_kp5_y] = 24 dims
# WAIT — 6 kpts × 2 coords = 12 positions + 12 velocities = 24 dims total
# The CONTEXT.md "60-dim" figure doesn't match. Implement as 24-dim with full doc.
# After re-reading CONTEXT.md: "6 keypoints × 2D × (position + velocity)" = 24 not 60.
# "60-dim" in CONTEXT.md is a stated value — implement as stated (verify in planning).

def _build_F(n_dim: int = 24) -> np.ndarray:
    """Constant-velocity transition matrix. n_dim = 2 * n_obs."""
    n_obs = n_dim // 2
    F = np.eye(n_dim)
    F[:n_obs, n_obs:] = np.eye(n_obs)  # pos_new = pos_old + vel_old
    return F

def _build_H(n_dim: int = 24) -> np.ndarray:
    """Measurement matrix — extract positions from state."""
    n_obs = n_dim // 2
    H = np.zeros((n_obs, n_dim))
    H[:n_obs, :n_obs] = np.eye(n_obs)
    return H
```

### Sigma Computation from Annotations

```python
# Source: project pattern — load annotation dataset, compute per-keypoint variance
# Run this script once before training to produce hard-coded sigma values
def compute_keypoint_sigmas_from_dataset(
    annotation_records: list[dict],  # [{keypoints: (K,2), obb_area: float}, ...]
) -> np.ndarray:                    # (K,) sigma values
    """Compute COCO-style per-keypoint sigmas from manual annotations.

    Sigma_k = std(||kpt_k - mean_kpt_k|| / sqrt(area)) over all fish.
    Normalized by fish scale (sqrt(OBB area)) so sigmas are scale-invariant.
    """
    ...
```

### Tracklet Builder Extended for Keypoints

```python
# Extend existing _TrackletBuilder to store keypoints per frame
@dataclass
class _TrackletBuilder:
    camera_id: str
    track_id: int
    frames: list[int] = field(default_factory=list)
    centroids: list[tuple[float, float]] = field(default_factory=list)
    bboxes: list[tuple] = field(default_factory=list)
    keypoints: list[np.ndarray] = field(default_factory=list)     # NEW: per-frame (6,2)
    keypoint_conf: list[np.ndarray] = field(default_factory=list) # NEW: per-frame (6,)
    frame_status: list[str] = field(default_factory=list)
    detected_count: int = 0
    active: bool = True
    keypoint_centroid_count: int = 0
```

The `Tracklet2D` dataclass currently has no `keypoints` field. The bidi merge and OKS-based matching during gap fill need keypoint data stored per-frame in the builder, but the final `Tracklet2D` produced for downstream consumers only needs `frames, centroids, bboxes, frame_status` (existing fields). The merge operates on builder-level data, not on frozen `Tracklet2D` objects.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| IoU bbox matching in OC-SORT | OKS keypoint matching | Phase 83 | More discriminative; can distinguish head-to-tail flip from correct match |
| External boxmot OcSort | Custom NumPy KF | Phase 83 | Full serialization control; confidence-scaled noise; no boxmot dependency for new tracker |
| Single forward pass | Bidirectional forward+backward merge | Phase 83 | Catches track births that forward pass creates late and deaths that backward pass recovers |
| No gap fill | Cubic spline gap interpolation | Phase 83 | Reduces `total_gaps` metric, improves continuity for downstream association |

**Deprecated/outdated:**
- `OcSortTracker` (retained for `tracker_kind="ocsort"`): Still works; not removed. Phase 85 decides on BoxMot dependency removal.

## Open Questions

1. **Exact state dimension: 24-dim vs 48-dim vs 60-dim**
   - What we know: CONTEXT.md says "60-dim" explicitly: "6 keypoints × 2D × (position + velocity)". 6 × 2 × 2 = 24, not 60. 6 × 2 × 5 = 60 (if there are 5 state components per coord). 60 would require adding acceleration or orientation per keypoint.
   - What's unclear: Was "60-dim" a typo for "24-dim"? Or does the design intend a richer state (e.g., position, velocity, acceleration)?
   - Recommendation: The planner's first task should note the discrepancy and implement as 24-dim (position + velocity only, which is the standard constant-velocity KF), explicitly document the choice in code comments with the derivation, and flag for user review if the 60-dim figure was intentional. This is in "Claude's Discretion" territory per CONTEXT.md.

2. **Sigma values to hard-code vs. compute at runtime**
   - What we know: CONTEXT.md says compute empirically from annotation dataset. The annotation database is in `aquapose/training/store.py`.
   - What's unclear: Whether sigmas should be computed as a pre-run step (producing hard-coded defaults) or queried from the annotation database at tracker init time.
   - Recommendation: Add `compute_keypoint_sigmas.py` as a one-time script that produces default sigma values. Hard-code reasonable defaults in `KeypointTrackerConfig` (can be overridden via YAML). Phase 83 implements the computation script; production use relies on the computed defaults.

3. **How to handle frames where all keypoints are below confidence floor**
   - What we know: The KF's confidence-scaled R handles this by inflating measurement noise to infinity for zero-conf keypoints.
   - What's unclear: Should a fully-occluded detection (all confs near zero) still trigger an update, or should it be treated as a missed detection?
   - Recommendation: Always update the KF when a detection is matched (OKS cost computed using zero-weight keypoints). The confidence-weighted OKS naturally produces a cost near 0.5 (random) for fully-occluded detections, which may cause mismatches. A `min_visible_keypoints` threshold (e.g., 2) to reject fully-occluded detections from matching is worth considering. Document this edge case.

## Validation Architecture

Config shows `workflow.nyquist_validation` is not set (absent from `config.json`). No `nyquist_validation` key present — skip this section per instructions.

## Sources

### Primary (HIGH confidence)
- Project codebase — `ocsort_wrapper.py`, `stage.py`, `types.py`, `context.py`, `config.py`: direct inspection of the interface that the new tracker must match
- Project codebase — `evaluation/stages/tracking.py`, `evaluation/stages/fragmentation.py`: metrics the new tracker's output is evaluated against
- Project codebase — `pyproject.toml`: confirmed scipy>=1.11 and numpy>=1.24 are dependencies; no new packages needed
- CONTEXT.md: locked design decisions are the implementation spec

### Secondary (MEDIUM confidence)
- OC-SORT paper (Cao et al., 2022): ORU/OCR mechanism descriptions; the paper is the reference for what these mechanisms mean algorithmically. Knowledge from training data, consistent with standard tracking literature.
- COCO keypoint evaluation documentation: OKS formula definition. Well-established formula, consistent with CONTEXT.md design.
- Standard Kalman filter references: constant-velocity KF matrix structure (F, H, Q, R, prediction, update equations).

### Tertiary (LOW confidence)
- State dimension "60-dim" in CONTEXT.md: the arithmetic 6 × 2 × 2 = 24, not 60. The discrepancy is flagged as an open question. LOW confidence that 60 is the intended correct value.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in project, no new dependencies
- Architecture: HIGH — follows existing OcSortTracker/TrackingStage patterns exactly; design decisions are locked in CONTEXT.md
- KF equations: HIGH — standard constant-velocity KF, well-established
- OKS cost: HIGH — COCO formula, adapted per CONTEXT.md spec
- Pitfalls: HIGH — identified from direct code inspection (backward pass ordering, merge ID collision, etc.)
- State dimension (60 vs 24): LOW — arithmetic inconsistency in CONTEXT.md; needs planner resolution

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (stable domain — algorithms are well-established; only project code changes could invalidate)
