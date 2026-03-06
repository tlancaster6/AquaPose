# AquaPose Pipeline Flowcharts

Two pipeline architectures for flowchart generation. Each step lists its operation, inputs, outputs, and decision points.

---

## Pipeline A: v2.0 (Current)

```
Detection → Midline → Association → Tracking → Reconstruction
```

### Step 1: Frame Loading (Pre-pipeline)

- **Operation:** Load video frames from 12 cameras, apply undistortion from calibration intrinsics
- **In:** Video files (12 cameras, mp4/avi), calibration JSON
- **Out:** Undistorted frames grouped into batches

### Step 2: Detection (Stage 1)

- **Operation:** Run YOLO object detector independently on each camera's frames
- **In:** Undistorted frames (all cameras)
- **Out:** Per-camera, per-frame bounding boxes with confidence scores
- **Note:** Each camera processed independently. Most cameras see 0–3 fish per frame (partial overlap is normal).

### Step 3: Midline Extraction (Stage 2)

- **Operation:** For every detection, crop the frame around the bounding box, run through segmentation model, extract 2D midline from the mask
- **In:** Bounding boxes (Stage 1), raw video frames
- **Sub-steps:**
  1. Crop frame region around bounding box
  2. Run U-Net segmentation model on crop → binary mask
  3. Skeletonize mask → 1-pixel-wide skeleton
  4. BFS traversal of skeleton to find longest path
  5. Arc-length resample path to exactly 15 evenly-spaced points
  6. Measure half-widths perpendicular to midline at each point
- **Out:** Annotated detections — each detection now carries a 15-point 2D midline + half-widths
- **Failure case:** If mask is degenerate (too small, disconnected), detection gets a flagged empty midline (not an exception)

### Step 4: Cross-View Association (Stage 3)

- **Operation:** Group detections across cameras that correspond to the same physical fish, using RANSAC on 3D centroid triangulation
- **In:** Annotated detections from all cameras (Stage 2), refractive projection models from calibration
- **Sub-steps:**
  1. For each detection, cast a refractive ray from the camera through the bbox centroid into 3D space
  2. Sample random 2-camera subsets, triangulate candidate 3D points (closest point on two rays)
  3. For each candidate 3D point, reproject into all cameras, count inliers (detections within reprojection threshold)
  4. Rank candidates by inlier count (greedy, highest first)
  5. Assign detections to best candidate; each detection assigned to at most one group
  6. Merge nearby candidates (within 5 cm XY distance)
  7. Single-view detections assigned as low-confidence fallback groups
- **Out:** Association bundles — each bundle is one physical fish with: 3D centroid, list of (camera_id → detection_index) assignments, camera count, reprojection residual
- **Decision point:** Is reprojection residual below threshold? → Accept / Reject candidate

### Step 5: Tracking (Stage 4)

- **Operation:** Assign persistent fish IDs to association bundles across frames using 3D position matching with velocity prediction
- **In:** Association bundles (Stage 3), carry-forward state from previous batch (3D positions, velocities, track lifecycles)
- **Sub-steps — Phase A (Claim):**
  1. For each existing track, predict 3D position using constant-velocity model
  2. Compute distance from each predicted position to each bundle's 3D centroid
  3. Sort candidates by (distance, priority) — confirmed tracks get priority over probationary
  4. Greedy assign: closest valid pair first, each track and bundle assigned at most once
  5. Reject assignments where residual exceeds mean + 3σ (anomaly gate)
  6. Update matched tracks: new position, velocity (windowed mean of recent deltas)
  7. Mark unmatched tracks as missed
- **Sub-steps — Phase B (Birth):**
  1. Collect unmatched bundles
  2. **Decision point:** Birth gating — allow birth if: (frame 0) OR (active tracks < expected count) OR (periodic birth interval reached)
  3. For each unmatched bundle passing birth gate and minimum camera count:
     - **Decision point:** Proximity check — is bundle > 8 cm from all existing tracks? → Birth / Reject as ghost
     - Create new track in PROBATIONARY state
     - Assign fish ID from dead-ID recycling pool if available, else fresh ID
- **Sub-steps — Phase C (Lifecycle):**
  1. Probationary tracks with ≥ 5 consecutive hits → promote to CONFIRMED
  2. Confirmed tracks with missed frames → transition to COASTING (velocity-based prediction continues)
  3. Probationary tracks missing > 2 frames → DEAD
  4. Confirmed/coasting tracks missing > 7 frames → DEAD
  5. Remove dead tracks; recycle their fish IDs
- **Out:** Per-frame list of confirmed fish tracks, each with: persistent fish_id, 3D position, velocity, camera_detections mapping, lifecycle state
- **Carry-forward out:** Track state (positions, velocities, lifecycles) for next batch

### Step 6: Reconstruction (Stage 5)

- **Operation:** Triangulate 2D midlines from multiple cameras into 3D splines for each tracked fish
- **In:** Fish tracks (Stage 4), annotated detections (Stage 2), refractive projection models
- **Sub-steps:**
  1. For each fish track in each frame, look up annotated detections via track's camera_detections mapping
  2. Collect 2D midlines from all cameras observing this fish
  3. **Decision point:** Single-view fish? → Cannot reconstruct, add to dropped list with reason
  4. For each of the 15 midline points, triangulate across cameras using refractive ray casting
  5. RANSAC on midline point correspondence across views (handle outlier cameras)
  6. Fit 7-control-point B-spline through triangulated 3D points
- **Out:** Per-frame dict of fish_id → Spline3D (3D B-spline midlines), plus dropped dict with failure reasons

### Step 7: Output (Post-pipeline)

- **Operation:** Observers write results to disk
- **In:** All pipeline context fields
- **Out:** HDF5 file with 3D midlines, diagnostic visualizations, timing logs

---

## Pipeline B: v2.1 (Proposed)

```
Detection → 2D Tracking → Cross-Camera Association → Midline → Reconstruction
```

### Step 1: Frame Loading (Pre-pipeline)

- **Operation:** Load video frames from 12 cameras, apply undistortion from calibration intrinsics
- **In:** Video files (12 cameras, mp4/avi), calibration JSON
- **Out:** Undistorted frames grouped into batches
- *(Unchanged from v2.0)*

### Step 2: Detection (Stage 1)

- **Operation:** Run object detector independently on each camera's frames
- **In:** Undistorted frames (all cameras)
- **Out:** Per-camera, per-frame bounding boxes with confidence scores
- *(Unchanged from v2.0. Future: OBB detector replaces axis-aligned YOLO for tighter crops.)*

### Step 3: Per-Camera 2D Tracking (Stage 2) — NEW

- **Operation:** Within each camera independently, link detections across frames into 2D tracklets using Hungarian matching
- **In:** Per-camera detections (Stage 1)
- **Per camera, per frame:**
  1. Predict each existing tracklet's next position (constant-velocity on 2D centroid)
  2. Compute cost matrix: predicted position vs. new detection centroids (Euclidean distance or IoU)
  3. Hungarian assignment: optimal 1-to-1 matching minimizing total cost
  4. **Decision point:** Assignment cost below threshold? → Match / Leave unmatched
  5. Matched detections extend their tracklet
  6. Unmatched detections start new tracklets (local ID, PROBATIONARY)
  7. Unmatched tracklets increment miss counter
  8. **Decision point:** Miss count exceeded? → Kill tracklet / Continue coasting
- **Out:** `tracks_2d` — per-camera list of tracklets. Each tracklet: local_track_id (camera-scoped), list of (frame_index, detection) pairs, lifecycle state
- **Carry-forward out:** Per-camera tracklet state (positions, velocities, lifecycles) for next batch
- **Key property:** No cross-camera awareness. Each camera's tracker is fully independent.

### Step 4: Cross-Camera Tracklet Association (Stage 3) — NEW

- **Operation:** Determine which tracklets across cameras correspond to the same physical fish by testing 3D geometric consistency across many frames
- **In:** `tracks_2d` from all cameras (Stage 2), refractive projection models from calibration
- **Sub-steps:**
  1. **Prune camera pairs:** Skip pairs with no overlapping field of view (precomputed from calibration geometry). Reduces the N² camera-pair space.
  2. **For each viable camera pair (cam_i, cam_j):**
     a. For each tracklet pair (one from cam_i, one from cam_j) with temporal overlap:
        - Sample shared frames (not all — subsample for efficiency)
        - For each sampled frame: cast refractive ray from each tracklet's centroid, triangulate (closest point on two rays), record 3D residual (ray-to-ray distance)
        - Compute median residual across sampled frames
     b. **Decision point:** Median residual below association threshold? → Candidate match / Discard
  3. **Build affinity graph:** Nodes = tracklets, edges = candidate matches weighted by median residual (lower = stronger)
  4. **Cluster into groups:** Connected components with consistency enforcement — if tracklet A matches B and B matches C, verify A–C consistency. Inconsistent edges pruned.
  5. **Assign global fish IDs:** Each cluster gets a unique global ID. All tracklets in the cluster are tagged with this ID.
  6. **Decision point:** Number of groups vs. expected fish count (9) — flag if significantly different (diagnostic, not a hard gate)
- **Out:** `tracklet_groups` — list of groups. Each group: global_fish_id, dict of camera_id → tracklet(s), triangulated 3D centroid trajectory
- **Handles fragmentation:** Multiple short tracklets from one camera can join the same group (e.g., fish occluded briefly, producing two tracklets in that camera)

### Step 5: Midline Extraction (Stage 4) — MOVED (was Stage 2)

- **Operation:** Extract 2D midlines only for detections belonging to confirmed tracklet-groups
- **In:** `tracklet_groups` (Stage 3), raw video frames, detection bounding boxes
- **Decision point:** Which backend?

#### Path A: Segment-then-extract (existing)
  1. Crop frame region around bounding box
  2. Run U-Net segmentation model on crop → binary mask
  3. Skeletonize → BFS longest path → arc-length resample to N points
  4. Measure half-widths at each point
  - **Out per detection:** Midline2D with N points + half-widths

#### Path B: Direct keypoint estimation (new)
  1. Crop frame region around bounding box
  2. Run shared MobileNetV3-Small encoder on crop
  3. Run keypoint head → 6 normalized (x, y) coordinates
  4. Transform crop-relative coordinates to full-frame via CropRegion
  - **Out per detection:** Midline2D with 6 points, half_widths = None

- **Head-tail consistency:** Cross-camera group membership (from Stage 3) provides a consistency signal — if most cameras agree on head direction, flip outliers
- **Out:** `annotated_detections` — midlines for grouped detections only (ungrouped detections skipped entirely)

### Step 6: Reconstruction (Stage 5)

- **Operation:** Triangulate 2D midlines from multiple cameras into 3D splines for each fish
- **In:** `tracklet_groups` (Stage 3), `annotated_detections` (Stage 4), refractive projection models
- **Sub-steps:**
  1. For each global fish ID, for each frame:
     - Look up which cameras observe this fish (from tracklet_group)
     - Retrieve annotated midlines from those cameras
  2. **Decision point:** Fewer than 2 cameras with midlines this frame? → Drop frame for this fish (record reason)
  3. For each of N midline points, triangulate across cameras using refractive ray casting
     - Correspondence is known (point i in cam A = point i in cam B) — no RANSAC needed for cross-view matching
     - Outlier cameras still filtered by reprojection error
  4. Fit B-spline through triangulated 3D points (control point count from config)
- **Out:** Per-frame dict of fish_id → Spline3D, plus dropped dict with reasons
- **Key difference from v2.0:** No RANSAC for cross-view identity — correspondence is pre-established. Triangulation is per-fish with known camera sets.

### Step 7: Output (Post-pipeline)

- **Operation:** Observers write results to disk
- **In:** All pipeline context fields
- **Out:** HDF5 file with 3D midlines, diagnostic visualizations (including tracklet trails and association group coloring), timing logs
