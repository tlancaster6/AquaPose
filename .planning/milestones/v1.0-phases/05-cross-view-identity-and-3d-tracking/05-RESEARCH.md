# Phase 5: Cross-View Identity and 3D Tracking - Research

**Researched:** 2026-02-21
**Domain:** Multi-view cross-camera fish identity association, 3D centroid triangulation, temporal tracking
**Confidence:** HIGH (core algorithms well-established; RANSAC centroid clustering pattern verified from literature; no existing library perfectly matches the custom refractive ray case)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Association strategy:**
- RANSAC centroid ray clustering for cross-view association
- Require minimum 2 cameras for a valid association; single-view detections passed through as flagged low-confidence entries (for track continuity, not reconstruction)
- Prior-guided association: use previous frame's 3D centroids as seed points to bias RANSAC clustering — critical because fish often overlap in many views
- `expected_count` is a configurable parameter (default 9), used as a soft global expectation — not enforced per-camera, not a hard constraint

**Track lifecycle:**
- Short grace period (5-10 frames) before killing lost tracks — produces high-confidence tracklets rather than long uncertain tracks; trajectory stitching deferred to a later phase with more signal from Phase 6
- 2-3 frame birth confirmation required before a track is considered valid — filters spurious detections
- Constant velocity motion model to predict next-frame position for Hungarian assignment
- First-frame initialization: Claude's discretion (batch vs gradual)

**Quality & thresholds:**
- Reprojection residual threshold: configurable parameter with a sensible default (informed by Phase 1 Z-uncertainty characterization)
- Low-confidence associations: flag but include in output with confidence score — downstream decides whether to use them
- No per-frame diagnostic logging — adds loop time, not useful at scale for hours-long video; info is recoverable from output
- No warnings on low fish count — silent operation, output speaks for itself

**Output interface:**
- Primary format: HDF5 serialization (matches existing io module patterns)
- Dataclass API kept accessible for future in-memory pipeline use, but serialization is the focus
- Chunked processing: configurable chunk size for hours-long videos, track state carried across chunks
- Per-fish per-frame data includes: 3D centroid, confidence score, (camera_id, detection_id) → fish_id mapping, AND per-camera bounding boxes organized by fish ID (saves Phase 6 from re-joining detection + identity outputs)

### Claude's Discretion

- First-frame initialization strategy (batch vs gradual ramp)
- Exact RANSAC parameters (iterations, inlier threshold)
- Chunk size default
- HDF5 dataset layout and compression
- Exact constant-velocity model implementation details

### Deferred Ideas (OUT OF SCOPE)

- Trajectory stitching (merging tracklets into continuous tracks) — future phase after Phase 6 provides midline signal
- Per-frame diagnostic summaries — could add later behind a debug flag if needed
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRACK-01 | System associates detections across cameras to physical fish via RANSAC-based centroid ray clustering — casting refractive rays from 2D centroids, triangulating minimal camera subsets, and scoring consensus against remaining cameras | RANSAC centroid ray algorithm must be custom-built using existing `triangulate_rays` + `cast_ray` from Phase 1; see Architecture Patterns for the algorithm steps |
| TRACK-02 | System produces a 3D centroid per fish per frame with reprojection residual; high-residual associations are flagged for downstream quality checks | Reprojection via `RefractiveProjectionModel.project()` already available; residual = mean pixel distance of all inlier cameras to projected centroid |
| TRACK-03 | System assigns persistent fish IDs across frames via Hungarian algorithm on 3D centroid distances | `scipy.optimize.linear_sum_assignment` (confirmed in pyproject.toml dependencies) is the standard, proven tool; see Standard Stack section |
| TRACK-04 | System enforces population constraint (exactly 9 fish at all times). If a track is lost and a new detection appears in the same frame window, they are linked | Expected-count soft constraint as configurable parameter; track lifecycle (birth/death) designed around maintaining close-to-9 fish count |
</phase_requirements>

---

## Summary

Phase 5 builds a cross-view identity and temporal tracking layer between per-camera detections (Phase 2) and per-fish reconstruction (Phase 6+). The core problem is: given detections from up to 13 cameras in a frame, cluster them into groups where each group corresponds to one physical fish, then maintain those identity assignments across frames.

The algorithm has two orthogonal concerns: **spatial association** (which detections from which cameras belong to the same fish) and **temporal association** (which fish in frame T+1 is the same as which fish in frame T). Spatial association is solved via RANSAC centroid ray clustering — a custom algorithm built on the existing `triangulate_rays` and `cast_ray` infrastructure from Phase 1. Temporal association is solved via the standard Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) applied to Euclidean distances between predicted and observed 3D centroids.

All required tools already exist in the project's dependencies (scipy, h5py, numpy, torch) and the Phase 1 refractive projection module. No new third-party dependencies are needed. The main implementation challenge is the RANSAC centroid clustering algorithm itself — this must be designed from scratch (no existing library implements the refractive-aware multi-view fish centroid clustering use case), but it is straightforward to build from the available primitives.

**Primary recommendation:** Build `src/aquapose/tracking/` as a new module with `associate.py` (RANSAC centroid clustering), `tracker.py` (track lifecycle + Hungarian assignment), and `writer.py` (HDF5 serialization). Use existing `triangulate_rays` and `cast_ray` from Phase 1 directly — no re-implementation needed.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy | >=1.11 (already in pyproject.toml) | `linear_sum_assignment` for Hungarian algorithm temporal association | Standard tool for optimal bipartite matching; uses Jonker-Volgenant algorithm (faster than classic Hungarian for large matrices) |
| numpy | >=1.24 (already in pyproject.toml) | Cost matrix construction, centroid arithmetic, constant-velocity prediction | All Phase 1 triangulation outputs are numpy/torch compatible |
| torch | >=2.0 (already in pyproject.toml) | Refractive ray casting and reprojection via `cast_ray` / `project` | Phase 1 projection model (`RefractiveProjectionModel`) is PyTorch-based |
| h5py | >=3.9 (already in pyproject.toml) | HDF5 output serialization with chunked resizable datasets | Matches existing io module pattern; `maxshape=(None, ...)` enables streaming writes |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dataclasses (stdlib) | Python 3.11+ | `FishTrack`, `FrameAssociation`, `TrackletResult` data containers | All data structures internal to the tracking module |
| collections.OrderedDict or deque (stdlib) | Python 3.11+ | Track history buffer for constant-velocity model | Rolling position history, fixed window for velocity estimation |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `scipy.optimize.linear_sum_assignment` | `lapjv` (LAPJV algorithm, external) | lapjv is faster for large matrices (>500x500) but not in current dependencies; fish count is 9 so the 9x9 cost matrix makes this irrelevant |
| Custom RANSAC centroid clustering | Open3D RANSAC | Open3D RANSAC fits geometric primitives (planes/spheres), not camera-ray clustering; not suitable |
| Custom constant-velocity model | filterpy Kalman filter | filterpy adds a dependency and Kalman's full covariance tracking is overkill when fish move slowly and the constant-velocity prediction is sufficient for 9 fish in 3D |

**Installation:** No new dependencies required — all tools already declared in `pyproject.toml`.

---

## Architecture Patterns

### Recommended Project Structure

```
src/aquapose/tracking/
├── __init__.py          # Public API: FishTrack, TrackingResult, FishTracker
├── associate.py         # RANSAC centroid ray clustering (TRACK-01, TRACK-02)
├── tracker.py           # Track lifecycle + Hungarian temporal assignment (TRACK-03, TRACK-04)
└── writer.py            # HDF5 serialization of tracklets (output interface)

tests/unit/tracking/
├── __init__.py
├── test_associate.py    # RANSAC centroid clustering unit tests
├── test_tracker.py      # Track lifecycle unit tests
└── test_writer.py       # HDF5 write/read round-trip tests
```

This follows the existing module pattern: `calibration/`, `segmentation/`, `initialization/`, `mesh/` — each is a focused sub-package with `__init__.py` exposing its public interface.

### Pattern 1: RANSAC Centroid Ray Clustering Algorithm

**What:** Custom algorithm using Phase 1 ray infrastructure to group per-camera detections into per-fish clusters.

**When to use:** Every frame, before temporal assignment.

**Algorithm steps (TRACK-01, TRACK-02):**

```python
# Source: custom algorithm built on triangulate_rays / cast_ray from Phase 1

def ransac_centroid_cluster(
    detections_per_camera: dict[str, list[Detection]],
    models: dict[str, RefractiveProjectionModel],
    expected_count: int = 9,
    n_iter: int = 200,
    reprojection_threshold: float = 15.0,  # pixels; configurable
    seed_points: list[np.ndarray] | None = None,  # prior frame centroids
) -> list[AssociationResult]:
    """
    For each RANSAC iteration:
      1. Sample a minimal camera subset (2 cameras, 1 detection per camera)
      2. Compute detection centroid pixels → cast refractive rays
      3. Triangulate rays → candidate 3D centroid
      4. Score consensus: project candidate back to all cameras, find
         detections within reprojection_threshold pixels
      5. Keep candidate with maximum inlier camera count (and min reprojection residual)

    If seed_points provided (prior-guided):
      - Weight sampling toward detection pairs consistent with prior centroids
      - Effectively: for each seed point, find the assignment that minimizes
        distance to seed, then use those detections as initial sampling bias

    After finding N best candidates (N ≈ expected_count):
      - Greedily assign each detection to at most one fish (no double-counting)
      - Any detection unassigned after all fish found → single-view low-confidence entry
    """
```

**Key design notes:**
- The `triangulate_rays` function from `aquapose.calibration.projection` already implements the least-squares DLT solver — call it directly on the 2-camera subset
- Reprojection residual (TRACK-02) = mean pixel distance from projected candidate centroid to nearest unassigned detection centroid in each inlier camera
- Prior-guided means: before random iteration, run one deterministic pass using previous-frame centroids as candidate points, find their inlier sets — this seeds the solution and dramatically reduces iterations needed for overlapping fish

**Reprojection residual computation:**
```python
# Source: RefractiveProjectionModel.project() from Phase 1
pixels_proj, valid = model.project(candidate_centroid.unsqueeze(0))
# residual for this camera = L2 distance to nearest detection centroid
residual = min(torch.linalg.norm(pixels_proj[0] - det_centroid_px)
               for det_centroid in camera_detections)
# mean across all inlier cameras = association confidence score
```

### Pattern 2: Constant-Velocity Temporal Prediction

**What:** Simple dead-reckoning motion model using last 2 frames to predict next-frame 3D centroid position.

**When to use:** Before building Hungarian cost matrix each frame.

**Example:**
```python
# No Kalman filter needed — constant velocity in 3D is sufficient
# State: position history (deque of length 2)
from collections import deque
import numpy as np

class FishTrack:
    def __init__(self, fish_id: int, centroid_3d: np.ndarray):
        self.fish_id = fish_id
        self.positions: deque[np.ndarray] = deque(maxlen=2)
        self.positions.append(centroid_3d)
        self.age: int = 1          # frames since creation
        self.hit_streak: int = 1   # consecutive frames matched
        self.frames_since_update: int = 0
        self.is_confirmed: bool = False  # True after min_hits confirmed

    def predict(self) -> np.ndarray:
        """Predict next position using constant velocity."""
        if len(self.positions) == 2:
            velocity = self.positions[-1] - self.positions[-2]
            return self.positions[-1] + velocity
        return self.positions[-1].copy()  # no velocity estimate yet

    def update(self, centroid_3d: np.ndarray) -> None:
        self.positions.append(centroid_3d)
        self.frames_since_update = 0
        self.hit_streak += 1
        self.age += 1
        if self.hit_streak >= MIN_HITS:
            self.is_confirmed = True
```

### Pattern 3: Hungarian Temporal Assignment

**What:** Optimal bipartite matching of predicted track positions to newly-observed 3D centroids.

**When to use:** After RANSAC clustering produces per-frame associations, before updating track state.

**Example:**
```python
# Source: scipy.optimize.linear_sum_assignment docs
from scipy.optimize import linear_sum_assignment
import numpy as np

def build_cost_matrix(
    tracks: list[FishTrack],
    observations: list[np.ndarray],  # list of 3D centroids from RANSAC
    max_distance: float = 0.3,       # meters; fish rarely jump this far between frames
) -> np.ndarray:
    """Cost = Euclidean 3D distance between predicted and observed centroid."""
    n_tracks = len(tracks)
    n_obs = len(observations)
    cost = np.full((n_tracks, n_obs), fill_value=max_distance + 1.0)
    for i, track in enumerate(tracks):
        pred = track.predict()
        for j, obs in enumerate(observations):
            dist = np.linalg.norm(pred - obs)
            cost[i, j] = dist
    return cost

def assign(
    tracks: list[FishTrack],
    observations: list[np.ndarray],
    max_distance: float = 0.3,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Returns:
        matched: list of (track_idx, obs_idx) pairs
        unmatched_tracks: track indices with no assignment
        unmatched_obs: observation indices with no assignment (new fish)
    """
    cost = build_cost_matrix(tracks, observations, max_distance)
    row_ind, col_ind = linear_sum_assignment(cost)

    matched, unmatched_tracks, unmatched_obs = [], [], []
    assigned_obs = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= max_distance:
            matched.append((r, c))
            assigned_obs.add(c)
        else:
            unmatched_tracks.append(r)

    unmatched_tracks += [i for i in range(len(tracks)) if i not in {r for r, _ in matched}]
    unmatched_obs = [j for j in range(len(observations)) if j not in assigned_obs]
    return matched, unmatched_tracks, unmatched_obs
```

Note: `linear_sum_assignment` handles rectangular matrices natively — if there are more tracks than observations (fish temporarily hidden) or more observations than tracks (new fish), unmatched rows/columns are handled by the threshold gate (`cost[r, c] <= max_distance` filter).

### Pattern 4: Track Lifecycle (SORT-derived, simplified for 3D)

**What:** State machine for track birth, confirmation, and death.

**Parameters (informed by SORT paper and CONTEXT.md decisions):**
```python
MIN_HITS = 2       # frames before a track is "confirmed" (CONTEXT: 2-3 frames)
MAX_AGE = 7        # frames before a lost track is killed (CONTEXT: 5-10 frames)
```

**Lifecycle:**
1. New detection with no matching track → create tentative track
2. Tentative track: confirmed when `hit_streak >= MIN_HITS`
3. Active track: updated each frame it matches a detection
4. Lost track: `frames_since_update > 0` — still predicted, included in assignment
5. Dead track: `frames_since_update > MAX_AGE` — removed from active set
6. TRACK-04 (population constraint): if a track dies and a new observation appears within same frame window, link them (treat the new observation as a re-observation of the lost track)

### Pattern 5: HDF5 Output Schema

**What:** Chunked HDF5 layout for tracking results, optimized for per-fish sequential access.

**Layout:**
```
/tracking/
    /frame_index          shape=(N_frames,), dtype=int64
    /fish_id              shape=(N_frames, MAX_FISH), dtype=int32, fill=-1 for absent
    /centroid_3d          shape=(N_frames, MAX_FISH, 3), dtype=float32
    /confidence           shape=(N_frames, MAX_FISH), dtype=float32, -1.0 for absent
    /reprojection_residual shape=(N_frames, MAX_FISH), dtype=float32, -1.0 for absent
    /n_cameras            shape=(N_frames, MAX_FISH), dtype=int32, 0 for absent
    /is_confirmed         shape=(N_frames, MAX_FISH), dtype=bool
    /camera_assignments/  # sparse: (camera_id, detection_id) → fish_id per frame
        /<camera_name>    shape=(N_frames, MAX_FISH), dtype=int32, -1 for absent
    /bboxes/              # per-camera bboxes organized by fish_id (for Phase 6)
        /<camera_name>    shape=(N_frames, MAX_FISH, 4), dtype=int32, -1 for absent
                          # [x, y, w, h] or [-1, -1, -1, -1] if fish not visible
```

**Creation pattern with chunking:**
```python
# Source: h5py docs (docs.h5py.org/en/stable/high/dataset.html)
import h5py
import numpy as np

CHUNK_FRAMES = 1000   # configurable; 1000 frames ≈ ~33 seconds at 30fps

with h5py.File(output_path, "w") as f:
    grp = f.create_group("tracking")
    # Unlimited growth on frame axis; fixed fish axis
    grp.create_dataset(
        "centroid_3d",
        shape=(0, MAX_FISH, 3),
        maxshape=(None, MAX_FISH, 3),
        chunks=(CHUNK_FRAMES, MAX_FISH, 3),
        dtype=np.float32,
        compression="gzip",
        compression_opts=4,
        fillvalue=-1.0,
    )
    # Resize and write each chunk:
    # dset.resize(current_n_frames + chunk_size, axis=0)
    # dset[current_n_frames:current_n_frames + chunk_size] = chunk_data
```

**Why this layout over per-fish groups:** Columnar arrays (one dataset per field) are faster to read for downstream analysis (select all centroids for all fish at once), avoid the overhead of creating thousands of groups per frame, and compress better with gzip since values within a field are more homogeneous.

### Anti-Patterns to Avoid

- **Kalman filter for constant velocity:** Overkill for 9 slow-moving fish in 3D — adds code complexity and dependency without accuracy benefit. A 2-frame deque for velocity estimation is sufficient.
- **Per-frame HDF5 groups:** Creating `/frame_0000/fish_0/centroid` etc. results in thousands of group objects and poor read performance. Use columnar arrays instead.
- **Greedy nearest-neighbor matching instead of Hungarian:** Greedy fails when fish are close together — it can give a suboptimal global assignment that swaps IDs. Hungarian guarantees globally-optimal 1-to-1 matching.
- **Enforcing expected_count as a hard constraint in RANSAC:** If you stop after finding exactly N fish, you miss incomplete associations (e.g., 8 confident + 1 low-confidence). Let the algorithm find as many as it can; the count is a soft guide.
- **Re-implementing `triangulate_rays`:** This already exists in `aquapose.calibration.projection` and is tested. Call it directly.
- **Re-implementing `cast_ray`:** Same — `RefractiveProjectionModel.cast_ray()` is already implemented and tested.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Optimal bipartite matching (track-to-detection) | Custom greedy or min-cost matching | `scipy.optimize.linear_sum_assignment` | Proven, optimal, handles rectangular matrices; already a dependency |
| Ray casting through air-water interface | Custom Snell's law ray tracer | `RefractiveProjectionModel.cast_ray()` from Phase 1 | Already implemented, tested, handles edge cases |
| 3D point from ray intersections | Custom least-squares solver | `triangulate_rays()` from Phase 1 | Already implemented with SVD, handles degenerate cases |
| 3D-to-pixel reprojection for residual | Custom Newton-Raphson iteration | `RefractiveProjectionModel.project()` from Phase 1 | Already implemented with 10-step Newton-Raphson, handles validity |

**Key insight:** The heavy geometric lifting (ray casting, triangulation, reprojection) was all built in Phase 1 and is directly reusable here. Phase 5 adds the clustering logic and lifecycle management on top of those primitives.

---

## Common Pitfalls

### Pitfall 1: RANSAC Convergence Failure Under Dense Overlap

**What goes wrong:** When 9 fish overlap in many views, random 2-camera sampling repeatedly picks pairs from merged detections, generating poor candidate centroids that don't converge to any real fish.

**Why it happens:** With 9 fish in 13 cameras, many camera pairs may show only 3-4 detection blobs (merged fish). Naive random sampling doesn't know which blobs contain multiple fish.

**How to avoid:** Prior-guided seeding (locked decision from CONTEXT.md). Run one deterministic initialization pass using previous-frame 3D centroids before random iterations. This ensures that even if RANSAC doesn't fully converge for new fish, the carried-over fish from the prior frame are correctly located. The random iterations then fill in the gaps.

**Warning signs:** Track IDs swap between frames even when fish are spatially separated; RANSAC produces fewer than expected_count candidates consistently.

### Pitfall 2: Hungarian Cost Matrix Max-Distance Gate

**What goes wrong:** Without a max-distance gate, the Hungarian algorithm will happily match a track to a detection 2 meters away (if it's the "closest" option), causing phantom long-range ID swaps when a fish is temporarily occluded.

**Why it happens:** `linear_sum_assignment` minimizes total cost regardless of individual cost values. It will produce an assignment even if all costs are large.

**How to avoid:** Apply a distance threshold: after solving the assignment, reject any matched pair where `cost[r, c] > max_distance`. The rejected tracks go to `unmatched_tracks` (grace period logic) and rejected detections go to `unmatched_obs` (new track creation).

**Warning signs:** Tracks "teleport" when fish reappear after occlusion.

### Pitfall 3: Detection Centroid vs. Mask Centroid

**What goes wrong:** Using bounding box center as the centroid pixel for ray casting instead of the actual mask centroid (center of mass of foreground pixels). For non-centered fish detections this can be off by 10-40 pixels.

**Why it happens:** Bounding boxes from YOLO/MOG2 are padded and rectangular — they don't center on the fish. The fish might be off-center in its bbox.

**How to avoid:** Compute mask centroid as the mean of foreground pixel coordinates (same as `extract_keypoints()` center in Phase 3). The `Detection.mask` field stores the full-frame binary mask — use `np.where(mask > 0)` → `np.mean(coords, axis=0)`.

**Warning signs:** Reprojection residuals are consistently high (>20px) even for well-separated fish.

### Pitfall 4: HDF5 Chunk Write Performance

**What goes wrong:** Writing one frame at a time to a chunked HDF5 dataset results in extremely slow I/O due to chunk cache thrashing — each single-frame write forces a partial chunk to disk.

**Why it happens:** HDF5 chunks are the atomic I/O unit. Writing less than a full chunk triggers read-modify-write cycles.

**How to avoid:** Buffer `CHUNK_FRAMES` frames in memory (numpy arrays), then write the full chunk at once via `dset[start:end] = buffer`. The `CHUNK_FRAMES` default should match the chunk size used at dataset creation.

**Warning signs:** Writing 10,000 frames takes 10x longer than expected; disk write patterns show many small partial writes.

### Pitfall 5: First-Frame Bootstrap Without Prior

**What goes wrong:** The prior-guided RANSAC requires previous-frame centroids, but frame 0 has none. Without a proper bootstrap strategy, frame 0 produces poor initial assignments that cascade into bad early track IDs.

**Why it happens:** Pure random RANSAC without priors can find multiple solutions — the specific ordering it happens to find on frame 0 determines which detection gets fish_id 0, 1, 2, etc. This ordering may not be consistent with what's intuitive later.

**How to avoid (Claude's Discretion — recommended batch initialization):** On frame 0, run full random RANSAC with many iterations (e.g., 500 instead of 200) without any prior. Sort the resulting N fish by XY position (e.g., spatial sort) to assign deterministic initial IDs. From frame 1 onward, use the frame-0 centroids as priors. This "batch initialization" approach is simpler than gradual ramp and produces stable results quickly.

**Warning signs:** Fish IDs oscillate during the first 5-10 frames; tracks die and are re-born repeatedly at the start.

### Pitfall 6: Z-Axis Weight in Cost Matrix

**What goes wrong:** Using raw 3D Euclidean distance in the Hungarian cost matrix gives too much weight to Z (depth), which has 132x more uncertainty than XY for this top-down 13-camera rig (documented in Phase 1 uncertainty report).

**Why it happens:** The Z reconstruction error can be large even when XY is accurate. Using Euclidean distance treats all axes equally, causing fish that are spatially close in XY to be "far" in the cost matrix due to Z noise.

**How to avoid:** Compute cost as XY-only Euclidean distance (ignore Z) or use a weighted norm: `cost = sqrt(dx^2 + dy^2 + (dz/100)^2)` with the 132x Z:XY anisotropy factor from Phase 1. XY-only is simpler and may be sufficient since fish rarely occupy the same XY position simultaneously.

**Warning signs:** Track swaps when fish are at different depths but similar XY positions.

---

## Code Examples

Verified patterns from official sources and existing codebase:

### Linear Sum Assignment (Temporal Association)

```python
# Source: scipy.optimize.linear_sum_assignment docs
# (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)
from scipy.optimize import linear_sum_assignment
import numpy as np

cost_matrix = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=float)
row_ind, col_ind = linear_sum_assignment(cost_matrix)
# row_ind: [0, 1, 2], col_ind: optimal assignment
# Total cost: cost_matrix[row_ind, col_ind].sum()
```

### Refractive Ray Casting from Detection Centroid

```python
# Source: aquapose.calibration.projection.RefractiveProjectionModel (Phase 1)
import torch
import numpy as np
from aquapose.calibration.projection import RefractiveProjectionModel

def centroid_ray(
    mask: np.ndarray,
    model: RefractiveProjectionModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast a refractive ray from the mask centroid pixel."""
    rows, cols = np.where(mask > 0)
    u = float(np.mean(cols))
    v = float(np.mean(rows))
    pixel = torch.tensor([[u, v]], dtype=torch.float32)
    origins, directions = model.cast_ray(pixel)  # shapes (1, 3), (1, 3)
    return origins[0], directions[0]
```

### Triangulate 3D Centroid from 2 Camera Rays

```python
# Source: aquapose.calibration.projection.triangulate_rays (Phase 1)
from aquapose.calibration.projection import triangulate_rays

origins = torch.stack([origin_cam0, origin_cam1], dim=0)   # (2, 3)
directions = torch.stack([dir_cam0, dir_cam1], dim=0)       # (2, 3)
candidate_centroid = triangulate_rays(origins, directions)  # (3,)
```

### Reprojection Residual Computation

```python
# Source: aquapose.calibration.projection.RefractiveProjectionModel.project() (Phase 1)
def reprojection_residual(
    centroid_3d: torch.Tensor,
    detection_centroid_px: np.ndarray,  # (2,) in (u, v) format
    model: RefractiveProjectionModel,
) -> float:
    """Return pixel reprojection residual for this camera."""
    pixels, valid = model.project(centroid_3d.unsqueeze(0))
    if not valid[0]:
        return float("inf")
    proj_uv = pixels[0].numpy()
    return float(np.linalg.norm(proj_uv - detection_centroid_px))
```

### H5py Resizable Chunked Dataset Creation

```python
# Source: h5py documentation (docs.h5py.org/en/stable/high/dataset.html)
import h5py
import numpy as np

MAX_FISH = 9
CHUNK_FRAMES = 1000

with h5py.File("tracking_output.h5", "w") as f:
    grp = f.create_group("tracking")
    grp.create_dataset(
        "centroid_3d",
        shape=(0, MAX_FISH, 3),
        maxshape=(None, MAX_FISH, 3),
        chunks=(CHUNK_FRAMES, MAX_FISH, 3),
        dtype=np.float32,
        compression="gzip",
        compression_opts=4,
        fillvalue=np.nan,
    )

# Streaming write pattern:
with h5py.File("tracking_output.h5", "a") as f:
    dset = f["tracking/centroid_3d"]
    current_len = dset.shape[0]
    new_len = current_len + len(chunk_data)
    dset.resize(new_len, axis=0)
    dset[current_len:new_len] = chunk_data
```

### SORT-style Track Lifecycle (Adapted for 3D Centroids)

```python
# Source: SORT paper (Bewley et al. 2016), adapted from abewley/sort
# (github.com/abewley/sort/blob/master/sort.py) — simplified for 3D centroid tracking
from collections import deque
from dataclasses import dataclass, field
import numpy as np

MIN_HITS = 2    # birth confirmation (CONTEXT: 2-3 frames)
MAX_AGE = 7     # grace period (CONTEXT: 5-10 frames)

@dataclass
class FishTrack:
    fish_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=2))
    age: int = 0
    hit_streak: int = 0
    frames_since_update: int = 0
    is_confirmed: bool = False

    def predict(self) -> np.ndarray:
        if len(self.positions) == 2:
            vel = self.positions[-1] - self.positions[-2]
            return self.positions[-1] + vel
        return self.positions[-1].copy()

    def update(self, centroid: np.ndarray) -> None:
        self.positions.append(centroid)
        self.frames_since_update = 0
        self.hit_streak += 1
        self.age += 1
        self.is_confirmed = self.hit_streak >= MIN_HITS

    def mark_missed(self) -> None:
        self.frames_since_update += 1
        self.hit_streak = 0
        self.age += 1

    @property
    def is_dead(self) -> bool:
        return self.frames_since_update > MAX_AGE
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Kalman filter (full covariance) for 2D tracking | Kalman or simple constant-velocity for 3D tracking | Ongoing — for small-N known-count tracking, simple models suffice | Phase 5 can use simpler deque-based velocity; no Kalman needed |
| Epipolar matching for cross-view identity | RANSAC centroid ray clustering | Specific to this project's refractive geometry | Epipolar lines don't account for refraction; ray clustering does |
| Appearance-based re-ID for identity | Geometry-only (3D centroid distance) | Locked decision — cichlids are visually identical | No appearance feature extraction needed |
| IoU-based Hungarian cost (2D) | 3D Euclidean distance Hungarian cost | N/A — this is a 3D tracking problem | More robust to viewpoint changes than 2D IoU |

**Deprecated/outdated for this project:**
- Appearance features (ReID): Explicitly out of scope — cichlids are visually indistinguishable; geometry is the only reliable signal
- Full Kalman filter: Overkill for 9 fish in 3D; deque-based constant velocity is sufficient and simpler to debug
- 2D-per-camera tracking then cross-view fusion: The multi-view framework paper (arXiv 2505.17201) shows this approach works but requires stereo matching; our refractive rig's ray clustering is more principled

---

## Open Questions

1. **RANSAC inlier threshold default value**
   - What we know: Phase 1 uncertainty report shows XY triangulation error ~0.3-2px for typical depths (from test suite); reprojection through water adds some error
   - What's unclear: What pixel threshold separates "detection belongs to this fish" vs "unrelated detection"? Fish bounding boxes are ~50-150px wide; centroid error from mask noise is likely 2-10px
   - Recommendation: Start with 15px (conservative); make it configurable so the user can tune per-rig. Log mean residuals during first few frames to calibrate.

2. **Hungarian max-distance gate value**
   - What we know: Fish move ~5-15cm/sec at 30fps → 0.5-1.5cm per frame in 3D world coordinates; tank is ~1m diameter
   - What's unclear: Whether to use XY-only or full 3D Euclidean, and what the right threshold is
   - Recommendation: Use XY-only distance (avoids Z noise), 0.1m max gate (10cm in XY per frame = fast fish). Make configurable.

3. **First-frame initialization ordering**
   - What we know: Claude's discretion. Batch initialization (run RANSAC once on frame 0 with many iterations) is simpler. Gradual ramp (treat first MIN_HITS frames as tentative) is what SORT does.
   - What's unclear: Whether spatial sorting of initial IDs matters for downstream analysis
   - Recommendation: Use batch initialization with spatial sort (e.g., by X coordinate of initial 3D centroid) for deterministic ID assignment. This is Claude's discretion, so implement it and note it's configurable.

4. **Expected_count constraint enforcement**
   - What we know: Configurable, default 9, soft constraint. Fish in 13-camera rig should always produce >=9 detections total (one per camera per fish = 117 detections maximum).
   - What's unclear: What happens if RANSAC finds only 7 fish in a frame — should we report 7 confirmed + 2 missing, or try harder?
   - Recommendation: Report what's found. Track lifecycle handles missing fish via grace period. Don't force the count to 9 — just let the tracker manage it.

---

## Sources

### Primary (HIGH confidence)
- `aquapose.calibration.projection` — Phase 1 implementation, confirmed reusable: `triangulate_rays`, `cast_ray`, `project`
- `aquapose.initialization.triangulator` — Phase 3 triangulation pipeline, confirms `triangulate_keypoint` pattern
- `aquapose.segmentation.detector` — `Detection` dataclass with `.mask` (full-frame binary) field
- scipy.optimize.linear_sum_assignment docs (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) — confirmed rectangular matrix support and `maximize` parameter
- h5py dataset docs (https://docs.h5py.org/en/stable/high/dataset.html) — confirmed `maxshape`, `chunks`, `compression`, `resize` API
- `pyproject.toml` — confirmed scipy>=1.11, h5py>=3.9, numpy>=1.24, torch>=2.0 already declared as dependencies

### Secondary (MEDIUM confidence)
- SORT paper / abewley/sort implementation (https://github.com/abewley/sort/blob/master/sort.py) — `min_hits`, `max_age`, track lifecycle state machine; confirmed `linear_sum_assignment` usage pattern
- AquaPose Phase 1 STATE.md decision: "Z error is 132x larger than XY for top-down 13-camera rig" — informs cost matrix design to downweight Z

### Tertiary (LOW confidence)
- Feature point based 3D fish tracking (PLOS One, 2017, https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180254) — confirms multi-view fish tracking using epipolar geometry and trajectory association; specific algorithm differs (2D track fusion vs. 3D ray clustering)
- Multi-view fish tracking framework (arXiv 2505.17201) — confirms ByteTrack+epipolar approach; our approach is more appropriate for refractive geometry

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all tools already in project dependencies; APIs confirmed from official docs
- Architecture: HIGH — follows existing project module patterns; RANSAC + Hungarian pattern is well-established
- Pitfalls: MEDIUM — most derived from codebase analysis and algorithm knowledge; some (Z-weight pitfall) are project-specific insights from Phase 1

**Research date:** 2026-02-21
**Valid until:** 2026-08-21 (6 months; scipy/h5py APIs are stable)
