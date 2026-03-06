# Phase 6: 2D Medial Axis and Arc-Length Sampling - Research

**Researched:** 2026-02-21
**Domain:** Morphological skeletonization, arc-length parameterization, crop-to-frame coordinate transforms
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Midline point count & format:**
- 15 points per fish per camera (head=0, tail=14), evenly spaced by normalized arc-length
- Half-widths included always — each midline point carries (x, y, half_width)
- Arc-length fractions are implicit from index (index/14 gives normalized position)
- Final coordinates in full-frame pixel space — transform from crop-space using detection bounding box (scale + translate)

**Head-to-tail orientation:**
- Anatomical ordering: point 0 = snout, point 14 = tail tip
- Use the track's 3D velocity vector (from Phase 5's `FishTrack`) to determine which skeleton endpoint is the head — the leading edge of motion is the head
- Ambiguous frames (velocity < 0.5 body-lengths/second): inherit orientation from previous frame
- First frame of tracklet: arbitrary assignment, then back-correct early frames once velocity establishes direction — capped at 30 frames or 1 second (whichever is less). After the cap, early frames keep whatever orientation they had.
- Cross-view consistency enforced: all cameras orient the same way for a given fish on a given frame, driven by the shared 3D velocity

**Skeletonization approach:**
- Aggressive morphological smoothing before skeletonization — closing then opening with larger kernels, appropriate for U-Net masks at ~0.62 IoU
- Adaptive kernel radius proportional to mask minor axis width (per pivot doc: `max(3, minor_axis_width // 8)`)
- Use `skimage.morphology.skeletonize` (morphological thinning), NOT medial axis transform
- Separate distance transform (`scipy.ndimage.distance_transform_edt`) on smoothed mask for half-width extraction — sampled at skeleton pixel locations
- Longest-path BFS pruning (two-pass): find farthest endpoint from any endpoint, then farthest from that — the path between is the midline. All branches discarded (single path only)

**Edge case handling:**
- Masks too small: skip silently — no midline output for that camera, other cameras compensate
- Degenerate skeletons (round mask, skeleton shorter than N points): skip this view entirely
- Boundary-clipped masks: if mask has nonzero pixels touching any edge of the crop, skip this view — truncated skeletons produce endpoints at arbitrary cutoff points, breaking arc-length parameterization
- Single-camera fish (only 1 camera sees fish this frame): skip entirely — need ≥2 cameras for triangulation
- No diagnostic output from this module — diagnostics via separate scripts if needed

### Claude's Discretion
- Minimum mask area threshold for the "too small" skip
- Exact smoothing kernel sizes and morphological operation sequence
- Implementation of the coordinate transform (scale + translate from crop to frame)
- Internal data structures for midline representation

### Deferred Ideas (OUT OF SCOPE)
- Merged-mask splitting: overlapping fish producing Y-shaped skeletons — handle upstream in segmentation/instance separation, not in skeletonization. Defer to future phase if it turns out to be common in practice.
- Epipolar-guided correspondence refinement: use arc-length as initial guess, refine by finding closest point on each camera's 2D midline to epipolar line from another camera. More robust for highly curved fish but more involved to implement. (Noted in pivot doc as future upgrade.)
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| RECON-01 | System extracts 2D medial axis from binary masks via morphological smoothing + skeletonization + longest-path BFS pruning, producing an ordered head-to-tail midline with local half-widths from the distance transform | `skimage.morphology.skeletonize` + `scipy.ndimage.distance_transform_edt` verified working in hatch env (scikit-image 0.26.0); two-pass BFS longest-path algorithm verified correct on branched skeleton (finds 16-pixel path from 20-pixel branched graph) |
| RECON-02 | System resamples 2D midlines at N fixed normalized arc-length positions (head=0, tail=1), producing consistent cross-view correspondences with coordinate transform from crop space to full-frame pixels | `scipy.interpolate.interp1d` arc-length resampling verified: 15-point output from skeleton pixel sequence; crop-to-frame transform is scale+translate using `CropRegion` fields already present in codebase |
</phase_requirements>

---

## Summary

Phase 6 implements the transition from segmentation masks to structured midline point correspondences ready for multi-view triangulation. The technical domain is classical morphological image processing — no machine learning or differentiable rendering. The core tools are `skimage.morphology.skeletonize` and `scipy.ndimage.distance_transform_edt`, both of which are standard, stable, and verified working in the project's hatch environment (scikit-image 0.26.0; scipy already in pyproject.toml). The arc-length resampling uses `scipy.interpolate.interp1d`, also already available.

The most algorithmically novel piece is the two-pass BFS longest-path pruning. This is not a known library function — it must be hand-implemented using Python's `collections.deque`. It was verified correct on a branched skeleton during research: from a T-shaped graph with 3 endpoints, the algorithm correctly identifies the 16-pixel spine and ignores the 7-pixel branch. The adjacency traversal uses 8-connectivity on the skeleton's boolean pixel array.

The critical integration decision is that `scikit-image` is not yet in `pyproject.toml` and must be added. All other dependencies (scipy, opencv-python, numpy) are already declared. The module fits cleanly into a new `src/aquapose/reconstruction/` package (or `src/aquapose/midline/`), taking `FishTrack` objects from Phase 5's tracking module and binary masks from Phase 2's segmentation module as inputs.

**Primary recommendation:** Implement as `src/aquapose/reconstruction/midline.py` with a single public function `extract_midlines(tracks, masks_per_camera, crop_regions) -> dict[int, dict[str, Midline2D]]`. Keep all pipeline stages (smooth → skeletonize → prune → orient → resample → transform) in private helpers within the same file.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `skimage.morphology.skeletonize` | 0.26.0 (installed) | Morphological thinning to 1-pixel-wide skeleton | Pivot doc specifies this explicitly over `medial_axis`; produces fewer branches on noisy masks |
| `scipy.ndimage.distance_transform_edt` | ≥1.11 (in pyproject) | Euclidean distance transform for half-width extraction | Decouples width from skeleton extraction per pivot doc; scipy already declared |
| `scipy.interpolate.interp1d` | ≥1.11 (in pyproject) | Arc-length resampling via 1D interpolation | Standard numpy/scipy; handles irregular skeleton spacing cleanly |
| `opencv-python (cv2)` | ≥4.8 (in pyproject) | Morphological closing/opening for mask smoothing | Already used throughout codebase (detector.py, crop.py); `cv2.getStructuringElement` + `cv2.morphologyEx` |
| `numpy` | ≥1.24 (in pyproject) | Array operations throughout | Already project-wide dependency |
| `collections.deque` | stdlib | BFS queue for longest-path pruning | No external dep; deque already used in tracker.py |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `scipy.ndimage.convolve` | ≥1.11 | Count skeleton pixel neighbors for endpoint detection | One-line neighbor count: `convolve(skel.astype(uint8), kernel_3x3_no_center)` gives degree of each pixel |
| `skimage.measure.regionprops` | 0.26.0 | Compute mask minor-axis width for adaptive kernel sizing | `regionprops(label_image)[0].minor_axis_length` gives the shorter axis of the binary mask |

### Installation Required

`scikit-image` is NOT currently in `pyproject.toml`. Must be added.

```toml
# In pyproject.toml [project] dependencies:
"scikit-image>=0.21",
```

Verified: `hatch run pip install scikit-image` installs 0.26.0 successfully. No conflicts with existing deps.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `skimage.morphology.skeletonize` | `skimage.morphology.medial_axis` | `medial_axis` returns distance transform simultaneously but produces noisier results on imperfect masks; pivot doc explicitly rejects it |
| `scipy.interpolate.interp1d` | `numpy.interp` | `numpy.interp` works for 1D; `interp1d` cleaner for x/y separately on same arc-length param |
| Two-pass BFS (hand-coded) | `networkx.dag_longest_path` | NetworkX adds a heavy dep for graph construction overhead; BFS is 30 lines and O(N) |
| `cv2.morphologyEx` | `skimage.morphology.binary_closing` | Both work; `cv2` already used in project; `cv2.getStructuringElement` gives ellipse kernels easily |

---

## Architecture Patterns

### Recommended Module Location

```
src/aquapose/
├── reconstruction/          # NEW package for Phase 6+
│   ├── __init__.py          # exports Midline2D, extract_midlines
│   └── midline.py           # Phase 6: skeletonize + resample + orient
```

Rationale: The pivot document defines Stages 1–4 as a "reconstruction pipeline." Grouping Phase 6 under `reconstruction/` makes Phase 7 (triangulation + spline fitting) a natural sibling module. The existing `initialization/` module is analogous precedent for a sub-package boundary.

### Key Data Structure

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Midline2D:
    """Ordered 2D midline for one fish in one camera view.

    Attributes:
        points: Full-frame pixel coordinates, shape (N, 2), dtype float32.
            Row i is (x, y) at normalized arc-length i/(N-1).
        half_widths: Local half-widths in pixels, shape (N,), dtype float32.
        fish_id: Track ID from Phase 5.
        camera_id: Camera identifier string.
        frame_index: Frame number.
        is_head_to_tail: True if point 0 is snout, point N-1 is tail.
            False if orientation is not yet established (first frames of tracklet).
    """
    points: np.ndarray       # (N, 2) float32, full-frame pixels
    half_widths: np.ndarray  # (N,) float32
    fish_id: int
    camera_id: str
    frame_index: int
    is_head_to_tail: bool = True
```

### Pattern 1: Mask Smoothing with Adaptive Kernel

```python
import cv2
import numpy as np
from skimage.measure import regionprops
from skimage.measure import label as skimage_label

def _adaptive_smooth(mask: np.ndarray) -> np.ndarray:
    """Apply morphological closing then opening with adaptive kernel.

    Kernel radius = max(3, minor_axis_width // 8) per pivot doc.
    """
    labeled = skimage_label(mask > 0)
    props = regionprops(labeled)
    if not props:
        return mask
    minor = props[0].minor_axis_length
    r = max(3, int(minor) // 8)
    kernel_size = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened
```

### Pattern 2: Skeletonize + Distance Transform

```python
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

def _skeleton_and_widths(smooth_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (skeleton_bool, distance_transform) on the smoothed mask."""
    bool_mask = smooth_mask > 0
    skel = skeletonize(bool_mask)          # bool array, 1px wide
    dt = distance_transform_edt(bool_mask) # float64, half-width in px
    return skel, dt
```

### Pattern 3: Two-Pass BFS Longest-Path Pruning

```python
from collections import deque

def _longest_path_bfs(skel: np.ndarray) -> list[tuple[int, int]]:
    """Extract the longest path from a skeleton via two-pass BFS.

    Returns ordered list of (row, col) pixel coords, or [] if skeleton empty.
    """
    pts = set(map(tuple, map(list, np.argwhere(skel).tolist())))
    if not pts:
        return []

    def neighbors(r: int, c: int) -> list[tuple[int, int]]:
        return [(r+dr, c+dc) for dr in (-1,0,1) for dc in (-1,0,1)
                if (dr, dc) != (0, 0) and (r+dr, c+dc) in pts]

    def bfs_farthest(start: tuple[int, int]) -> tuple[tuple[int, int], dict]:
        prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        q: deque[tuple[int, int]] = deque([start])
        last = start
        while q:
            node = q.popleft()
            last = node
            for nb in neighbors(*node):
                if nb not in prev:
                    prev[nb] = node
                    q.append(nb)
        return last, prev

    # Find any endpoint (degree-1 node) or fall back to arbitrary start
    from scipy.ndimage import convolve
    kernel = np.ones((3, 3), dtype=np.uint8); kernel[1, 1] = 0
    nc = convolve(skel.astype(np.uint8), kernel, mode='constant')
    endpoints = list(map(tuple, np.argwhere(skel & (nc == 1)).tolist()))
    start = endpoints[0] if endpoints else next(iter(pts))

    ep1, _ = bfs_farthest(start)        # type: ignore[arg-type]
    ep2, prev = bfs_farthest(ep1)       # type: ignore[arg-type]

    # Reconstruct path from ep2 back to ep1
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = ep2
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path  # ordered ep1 → ep2
```

### Pattern 4: Arc-Length Resampling

```python
import numpy as np
from scipy.interpolate import interp1d

def _resample_arc_length(
    path_yx: list[tuple[int, int]],
    dt: np.ndarray,
    n_points: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a pixel path at n_points evenly-spaced arc-length positions.

    Args:
        path_yx: Ordered (row, col) pixel coordinates.
        dt: Distance transform array (same shape as mask) for half-widths.
        n_points: Number of output samples.

    Returns:
        Tuple of (xy_crop, half_widths) each shape (n_points,) or (n_points, 2).
        xy_crop is in (col, row) = (x, y) pixel coordinates, crop-space.
    """
    yx = np.array(path_yx, dtype=float)  # (K, 2)
    xy = yx[:, ::-1]                     # convert to (x, y)

    diffs = np.diff(yx, axis=0)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    arc_norm = arc / arc[-1]

    half_widths_raw = dt[yx[:, 0].astype(int), yx[:, 1].astype(int)]

    t = np.linspace(0.0, 1.0, n_points)
    fx = interp1d(arc_norm, xy[:, 0], kind='linear')
    fy = interp1d(arc_norm, xy[:, 1], kind='linear')
    fw = interp1d(arc_norm, half_widths_raw, kind='linear')

    return np.stack([fx(t), fy(t)], axis=1).astype(np.float32), fw(t).astype(np.float32)
```

### Pattern 5: Crop-to-Frame Coordinate Transform

The `CropRegion` dataclass in `segmentation/crop.py` already provides the fields needed. The transform is simple scale + translate:

```python
from aquapose.segmentation.crop import CropRegion

def _crop_to_frame(
    xy_crop: np.ndarray,   # (N, 2) in crop-pixel coords
    crop_region: CropRegion,
    crop_h: int,
    crop_w: int,
) -> np.ndarray:
    """Map crop-space (x, y) to full-frame pixel (x, y).

    The mask is crop_h x crop_w; the CropRegion defines where it maps in the frame.
    Scale factors handle cases where the mask was resized relative to the crop.

    Args:
        xy_crop: Points in crop image coordinates, shape (N, 2).
        crop_region: The CropRegion from Phase 2 detection.
        crop_h: Height of the mask array (may differ from region.height if resized).
        crop_w: Width of the mask array (may differ from region.width if resized).

    Returns:
        Full-frame pixel coordinates, shape (N, 2), float32.
    """
    scale_x = crop_region.width / crop_w
    scale_y = crop_region.height / crop_h
    xy_frame = xy_crop.copy()
    xy_frame[:, 0] = xy_crop[:, 0] * scale_x + crop_region.x1
    xy_frame[:, 1] = xy_crop[:, 1] * scale_y + crop_region.y1
    return xy_frame.astype(np.float32)
```

Note: `CropRegion` does not store `crop_h`/`crop_w` — caller must pass these from the actual mask array shape.

### Pattern 6: Head-to-Tail Orientation via 3D Velocity

```python
import numpy as np
from aquapose.tracking.tracker import FishTrack
from aquapose.calibration.projection import RefractiveProjectionModel

def _orient_midline(
    xy_frame: np.ndarray,     # (N, 2) full-frame pixels, arbitrary endpoint order
    track: FishTrack,
    model: RefractiveProjectionModel,
    velocity_threshold_ms: float = 0.5,  # body-lengths/sec threshold (converted to m/frame)
    body_length_m: float = 0.15,         # typical fish body length in metres
) -> tuple[np.ndarray, bool]:
    """Orient midline so point 0 is the snout (leading motion direction).

    Returns (oriented_xy, is_established) where is_established=False when
    velocity is below threshold (caller should inherit previous orientation).
    """
    vel = track.velocity   # shape (3,) in world metres per frame
    speed = float(np.linalg.norm(vel))

    # Convert threshold: body-lengths/sec → metres/frame (assume 30fps)
    threshold_m_per_frame = velocity_threshold_ms * body_length_m / 30.0
    if speed < threshold_m_per_frame:
        return xy_frame, False   # ambiguous — caller inherits previous frame

    # Project both endpoints into this camera; the one closer to the
    # leading edge (position + velocity direction) is the head
    ep0 = xy_frame[0]
    ep1 = xy_frame[-1]

    # Project 3D predicted head position (centroid + velocity unit * half-body)
    head_3d = list(track.positions)[-1] + (vel / speed) * (body_length_m / 2.0)
    head_2d = model.project(head_3d)   # (2,) pixel coords

    dist0 = float(np.linalg.norm(ep0 - head_2d))
    dist1 = float(np.linalg.norm(ep1 - head_2d))

    if dist1 < dist0:
        return xy_frame[::-1].copy(), True   # flip so ep0 is head
    return xy_frame, True
```

### Pattern 7: Edge Case Detection (Skip Conditions)

```python
def _check_skip_mask(
    mask: np.ndarray,       # uint8, crop-space
    min_area: int = 300,    # minimum pixel area
) -> str | None:
    """Returns a skip reason string if this mask should be skipped, else None."""
    area = int((mask > 0).sum())
    if area < min_area:
        return f"mask too small: {area} < {min_area}"

    # Boundary-clipped: nonzero pixels touching any edge of the crop
    h, w = mask.shape[:2]
    if (mask[0, :].any() or mask[-1, :].any() or
            mask[:, 0].any() or mask[:, -1].any()):
        return "mask touches crop boundary (boundary-clipped)"

    return None
```

Minimum area threshold is at Claude's discretion. Research recommendation: **300 pixels** — at 128×128 training resolution, this corresponds to roughly a 17×18 pixel blob, sufficient to produce a skeleton with ≥15 path pixels. Adjust based on observed mask sizes in practice.

### Anti-Patterns to Avoid

- **Using `skimage.morphology.medial_axis`**: Returns the distance transform alongside the skeleton, which seems convenient, but produces noisier, branchier skeletons on imperfect U-Net masks. The pivot doc explicitly specifies `skeletonize`. Confidence: HIGH (pivot doc is the authoritative design reference).
- **Sorting skeleton pixels by column/row instead of BFS ordering**: `np.argwhere(skel)` returns pixels in raster-scan order, not path order. Must use BFS traversal (or similar graph walk) to get a connected, ordered sequence. Otherwise arc-length is computed on a scrambled path.
- **Using `scipy.ndimage.convolve` with `mode='wrap'`**: Endpoint detection uses neighbor counts at the image boundary; `mode='constant'` (default fill=0) correctly treats outside-image as empty. `mode='wrap'` would create false neighbors across image edges.
- **Applying distance transform to the original (unsmoothed) mask**: Half-widths should come from the smoothed mask's EDT, not the raw mask. Using the raw mask's EDT would give noisy, jaggy width estimates.
- **Resampling half-widths in crop space then scaling to frame space**: Scale both xy coordinates AND half-widths by the crop-to-frame scale factor (use average of x and y scale, or the geometric mean). Omitting width scaling produces systematically wrong widths downstream.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Morphological thinning to 1px skeleton | Custom iterative erosion | `skimage.morphology.skeletonize` | Zhang-Suen thinning with topological preservation; hand-rolled versions break connectivity on curved shapes |
| Euclidean distance transform | Manhattan approximation or loop | `scipy.ndimage.distance_transform_edt` | Exact Euclidean EDT in C; half-widths require accuracy, not just approximation |
| 1D arc-length interpolation | Manual lerp with index search | `scipy.interpolate.interp1d` | Handles edge cases (duplicate x values, endpoint extrapolation) correctly |
| Elliptical structuring element | Hand-drawn circular disk | `cv2.getStructuringElement(MORPH_ELLIPSE, ...)` | Already used in codebase; correct pixel-fill for small radii |

**Key insight:** All the image processing primitives here are classical computer vision algorithms with 30+ years of optimized implementations. The only custom code needed is the two-pass BFS (which is a graph algorithm, not image processing), the head-tail orientation logic (domain-specific), and the coordinate transform (trivial arithmetic).

---

## Common Pitfalls

### Pitfall 1: Skeleton Returns Zero or Single Pixel

**What goes wrong:** `skeletonize` returns an all-zero or nearly-empty result on a mask that passed area check.
**Why it happens:** The morphological opening step can over-erode thin masks (caudal peduncle, fins) at the aggressive kernel sizes needed for smoothing. Or the mask is circular (fish head-on) with no dominant axis.
**How to avoid:** After skeletonize, check `skel.sum() >= n_points` (where n_points=15). If not, skip this view. The CONTEXT.md labels this "degenerate skeleton" and says to skip entirely.
**Warning signs:** skeleton pixel count < 15 after processing a mask that visually appears valid.

### Pitfall 2: BFS Produces Wrong Longest Path When Skeleton Has Cycles

**What goes wrong:** `skeletonize` occasionally produces skeleton loops (junction pixels connected to themselves via multiple paths). BFS from an endpoint can get trapped in a cycle.
**Why it happens:** Junction pixels in the skeleton (degree ≥ 3) can form small loops near branch points. Two-pass BFS assumes a tree structure.
**How to avoid:** After finding the BFS path, verify that the number of unique pixels in the result is approximately equal to the path length (no repeated visits). If the path visits fewer pixels than expected, the skeleton has a cycle — skip this view and mark as degenerate. Alternatively, check for junction pixels (degree ≥ 3) before BFS; if any exist after pruning expectations, flag as suspicious.
**Warning signs:** `len(path) < skel.sum() * 0.7` (path covers less than 70% of skeleton pixels).

### Pitfall 3: Head-Tail Orientation Flip at Low Speed

**What goes wrong:** Orientation flips back and forth between frames when fish is nearly stationary, causing downstream midline correspondences to swap head/tail on alternate frames.
**Why it happens:** Velocity vector is near zero; small noise in centroid estimate alternates which projection is closer to the "leading" endpoint.
**How to avoid:** The CONTEXT.md decision is clear: inherit from previous frame when speed < threshold. Store the last-established orientation per `(fish_id)` in the caller. Never re-derive orientation from current velocity if `is_established=False`.
**Warning signs:** Arc-length point 0 (snout) appears near the tail in reprojections.

### Pitfall 4: Back-Correction Exceeds Frame Buffer

**What goes wrong:** The back-correction of early tracklet frames (retroactively fixing orientation once velocity establishes) stores all early-frame midlines and tries to flip them, but the buffer grows unbounded for fish that never establish clear velocity.
**Why it happens:** If velocity stays below threshold for more than 30 frames (the cap), the buffer was supposed to be frozen, but without a hard stop, it keeps growing.
**How to avoid:** Enforce the 30-frame cap strictly in the back-correction buffer. After 30 frames or 1 second (whichever is fewer at video fps), commit whatever orientation the early frames have and discard the buffer.
**Warning signs:** Memory usage growing per fish track over time.

### Pitfall 5: Distance Transform Half-Width Scaling to Frame Space

**What goes wrong:** Half-widths are measured in crop-pixel units. When transformed to frame space, only `xy` coordinates are scaled, but `half_width` stays in crop-pixel units.
**Why it happens:** The coordinate transform code only handles `xy`; half-width scaling is an easy omission.
**How to avoid:** Scale half-widths by `(scale_x + scale_y) / 2.0` (or `sqrt(scale_x * scale_y)` for geometric mean). Add an explicit unit test that verifies half-widths scale correctly when `crop_region.width != crop_w`.
**Warning signs:** Half-widths at frame scale appear much smaller than visible fish body width.

### Pitfall 6: scikit-image Not in pyproject.toml

**What goes wrong:** `from skimage.morphology import skeletonize` raises `ModuleNotFoundError` in a fresh hatch env.
**Why it happens:** `scikit-image` is not currently in `pyproject.toml`. It was manually installed for research verification but is not declared as a dependency.
**How to avoid:** Add `"scikit-image>=0.21"` to `[project] dependencies` in `pyproject.toml` as the FIRST task in Phase 6 planning.
**Warning signs:** Import fails in CI or fresh clone.

---

## Code Examples

### Full Single-Mask Pipeline (Verified)

```python
# Verified working in hatch env (scikit-image 0.26.0, scipy 1.11+)
import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, convolve
from scipy.interpolate import interp1d
from collections import deque

# --- 1. Morphological smoothing ---
mask_u8 = mask.astype(np.uint8) * 255  # input: binary uint8 crop mask
r = max(3, int(minor_axis_width) // 8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
smooth_bool = opened > 0

# --- 2. Skeletonize + distance transform ---
skel = skeletonize(smooth_bool)                        # bool array
dt = distance_transform_edt(smooth_bool)               # float64 half-widths

# --- 3. Two-pass BFS longest path ---
pts = set(map(tuple, np.argwhere(skel).tolist()))
conn_k = np.ones((3,3), dtype=np.uint8); conn_k[1,1] = 0
nc = convolve(skel.astype(np.uint8), conn_k, mode='constant')
endpoints = list(map(tuple, np.argwhere(skel & (nc == 1)).tolist()))

def bfs(start):
    prev = {start: None}; q = deque([start]); last = start
    while q:
        node = q.popleft(); last = node
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if (dr,dc)==(0,0): continue
                nb = (node[0]+dr, node[1]+dc)
                if nb in pts and nb not in prev:
                    prev[nb] = node; q.append(nb)
    return last, prev

start = endpoints[0] if endpoints else next(iter(pts))
ep1, _ = bfs(start)
ep2, prev = bfs(ep1)
path = []
cur = ep2
while cur is not None:
    path.append(cur); cur = prev[cur]
# path is ordered (row, col) from ep1 to ep2

# --- 4. Arc-length resampling ---
yx = np.array(path, dtype=float)
xy_crop = yx[:, ::-1]   # convert to (x, y)
diffs = np.diff(yx, axis=0)
arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
arc_norm = arc / arc[-1]
hw_raw = dt[yx[:,0].astype(int), yx[:,1].astype(int)]
t = np.linspace(0.0, 1.0, 15)
fx = interp1d(arc_norm, xy_crop[:,0]); fy = interp1d(arc_norm, xy_crop[:,1])
fw = interp1d(arc_norm, hw_raw)
xy_resampled = np.stack([fx(t), fy(t)], axis=1).astype(np.float32)  # (15, 2)
hw_resampled = fw(t).astype(np.float32)                              # (15,)

# --- 5. Crop-to-frame transform ---
scale_x = crop_region.width / mask.shape[1]
scale_y = crop_region.height / mask.shape[0]
xy_frame = xy_resampled.copy()
xy_frame[:, 0] = xy_resampled[:, 0] * scale_x + crop_region.x1
xy_frame[:, 1] = xy_resampled[:, 1] * scale_y + crop_region.y1
hw_frame = hw_resampled * ((scale_x + scale_y) / 2.0)
```

### FishTrack Interface for Velocity Access

```python
# FishTrack.velocity is np.ndarray shape (3,), metres per frame
# FishTrack.positions is deque of shape-(3,) arrays (last 2 frames)
# FishTrack.bboxes is dict[str, tuple[int,int,int,int]] — NOTE: currently set to {} in update_from_claim
#   Phase 6 will need to populate bboxes when calling tracker, or derive crop_region separately

from aquapose.tracking.tracker import FishTrack
import numpy as np

track: FishTrack = ...
speed = float(np.linalg.norm(track.velocity))  # metres/frame
last_pos = list(track.positions)[-1]            # shape (3,)
```

**Critical finding:** `FishTrack.bboxes` is reset to `{}` in `update_from_claim` and `update_position_only` (tracker.py lines 195, 234). Phase 6 needs bounding boxes per camera to locate the crop region for each detection. Either: (a) populate `bboxes` from the `Detection.bbox` at claim time in the tracker (requires tracker modification), or (b) pass crop regions alongside masks into Phase 6 separately without relying on `FishTrack.bboxes`. Option (b) avoids modifying Phase 5 code and is recommended.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `skimage.morphology.medial_axis` | `skimage.morphology.skeletonize` | Project pivot 2026-02-21 | `skeletonize` (Zhang-Suen thinning) produces fewer branches on noisy binary masks than `medial_axis` (ridge-based) |
| Epipolar-constrained keypoint extraction (PCA-based) | Skeleton → arc-length sampling | Project pivot 2026-02-21 | Direct arc-length sampling avoids Adam optimizer; much faster |

**scikit-image 0.26.0 note:** `skimage.morphology.skeletonize` signature is stable. The `method` parameter (default `'lee'` since 0.19) uses Lee's 3D thinning algorithm, which preserves topology. No breaking API changes between 0.21 and 0.26. Confidence: HIGH (verified in env).

---

## Integration Notes

### Input Interface (what Phase 6 receives)

Phase 6 receives from Phase 5:
- `list[FishTrack]` — confirmed tracks with `.velocity`, `.positions`, `.camera_detections` (camera_id → detection_index)
- `dict[str, list[Detection]]` — detections per camera (for bbox → crop region lookup)
- `dict[str, RefractiveProjectionModel]` — camera models (for 3D head position projection into 2D)

Phase 6 receives from Phase 2:
- Binary masks in crop space — one mask per `(camera_id, detection_index)` pair
- `CropRegion` for each mask (for coordinate transform)

**Important:** The `FishTrack.camera_detections` dict maps `camera_id → detection_index`. This is sufficient to look up both the mask and the bbox for that camera's detection.

### Output Interface (what Phase 7 triangulation receives)

```python
# Recommended output type alias
MidlineSet = dict[int, dict[str, Midline2D]]
# MidlineSet[fish_id][camera_id] = Midline2D
# Only cameras with valid (non-skipped) midlines are present
```

---

## Open Questions

1. **`FishTrack.bboxes` is always empty** — confirmed by code inspection. Phase 6 planning must decide whether to (a) pass `detections_per_camera` alongside tracks so Phase 6 can look up `Detection.bbox` via `track.camera_detections`, or (b) modify the tracker to store bboxes at claim time. Option (a) is lower-risk (no Phase 5 changes). **Recommendation: (a) — pass detections_per_camera into Phase 6's public API.**

2. **Minor axis width for adaptive kernel** — requires `skimage.measure.regionprops` on the labeled mask. This adds one import and ~1ms per mask. Alternative: use a fixed rule like `max(3, mask_minor_axis // 8)` where `mask_minor_axis` is approximated as `min(mask.shape) // 4`. Recommendation: use `regionprops` for accuracy — it's already available with scikit-image.

3. **Back-correction buffer implementation** — the CONTEXT.md specifies storing early frames (up to 30) then retroactively flipping. The implementation needs a frame-indexed buffer per fish. This is stateful across frames. Phase 6 should expose a stateful `MidlineExtractor` class (not a pure function) to hold this state, similar to `FishTracker`.

4. **fps for velocity threshold** — the 0.5 body-lengths/second threshold needs the video fps to convert to metres/frame. The project hasn't standardized a fps constant. Recommendation: accept `fps: float` as a constructor parameter to `MidlineExtractor` with a default of 30.0.

---

## Sources

### Primary (HIGH confidence)
- **Direct code inspection** — `FishTrack.velocity` (tracker.py:119), `FishTrack.bboxes` reset behavior (tracker.py:195, 234), `CropRegion` fields (crop.py:17-44), `pyproject.toml` dependencies (confirmed scikit-image absent)
- **Live environment verification** — `skimage.morphology.skeletonize` (0.26.0), `scipy.ndimage.distance_transform_edt`, `scipy.interpolate.interp1d` — all tested in hatch env
- **Algorithm verification** — two-pass BFS on branched skeleton (3 endpoints → 16-pixel longest path); full pipeline (smooth → skeleton → BFS → resample) returns (15, 2) array as expected
- **Pivot document** `.planning/inbox/fish-reconstruction-pivot.md` — authoritative design reference for Stages 1 and 2

### Secondary (MEDIUM confidence)
- **CONTEXT.md** — locked decisions (kernel sizing formula, skip conditions, back-correction cap) — source of truth for all user decisions; no independent verification needed

### Tertiary (LOW confidence)
- **scikit-image 0.26.0 changelog** — not directly checked; inferred from installed version and stable API history. Risk: low (skeletonize API stable since 0.19).

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified installed and functional in hatch env
- Architecture: HIGH — module location follows established codebase patterns; data structures are straightforward
- Pitfalls: HIGH for code-inspection-verified issues (bboxes always empty, scale transform omission); MEDIUM for runtime pitfalls (skeleton cycles, orientation flipping) based on algorithm analysis

**Research date:** 2026-02-21
**Valid until:** 2026-05-21 (stable libraries, 90-day validity)
