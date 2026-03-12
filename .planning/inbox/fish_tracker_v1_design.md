# Bidirectional Batched Keypoint Tracker for Fish — V1 Design

## Problem Statement

We need to track individual fish across video sequences processed in 300-frame batches. Each fish is detected with an oriented bounding box (OBB) and a pose model that outputs 6 spine keypoints, which are interpolated to 15 evenly-spaced keypoints for downstream use. Fish swim in and out of the frame, occlude one another, and follow non-linear trajectories. The tracker must produce consistent identity assignments across frames and across chunk boundaries, working offline on batched data rather than live streams.

## Solution Overview

We extend OC-SORT to operate on keypoint poses in a bidirectional batch setting. The tracker runs a **forward pass** and a **backward pass** over each 300-frame chunk, then **merges** the resulting tracklets to exploit the temporal context available in offline processing. The Kalman filter state is extended from bounding boxes to 15 spine keypoints, and the association cost function is replaced with Object Keypoint Similarity (OKS) plus an observation-centric momentum (OCM) term derived from the spine heading vector. At chunk boundaries, full Kalman filter state (mean + covariance) is handed off to the next chunk.

### Why OC-SORT as the Foundation

OC-SORT addresses the primary failure mode of Kalman filter-based trackers: error accumulation during occlusion. Its three mechanisms — Observation-Centric Re-Update (ORU), Observation-Centric Momentum (OCM), and Observation-Centric Recovery (OCR) — are each directly relevant to fish tracking, and each becomes more powerful in a batch setting where future observations are available.

### Why Keypoints Over Bounding Boxes

Spine keypoints provide three advantages over oriented bounding boxes for association: (1) they resolve heading ambiguity without the 180° flip problem inherent in OBB angle representations, (2) OKS is a more discriminative similarity metric than IoU when fish are similarly sized and swimming near each other, and (3) the keypoint configuration encodes body curvature, acting as a lightweight appearance descriptor.

---

## Key Challenges

### 1. Non-Linear Fish Motion

**Problem:** Fish trajectories involve sudden turns, accelerations, and schooling dynamics that violate the linear motion assumption underlying the Kalman filter.

**How we address it:** The Kalman filter's constant-velocity model is a reasonable local approximation at typical frame rates. OC-SORT's OCM term adds a direction consistency cost based on observed heading (derived from the spine keypoint vector from head to tail), penalizing associations that would imply implausible direction reversals. The bidirectional pass further mitigates this: when a fish makes a sharp turn during occlusion, the backward pass may successfully track through the turn from the other temporal direction, providing observations that the forward-only pass missed.

### 2. Frequent Occlusion from Overlapping Fish

**Problem:** Fish swimming in schools frequently overlap, causing detections to be missed for one or both individuals. During occlusion, the Kalman filter predicts forward with no observation updates, and its estimates drift.

**How we address it:** OC-SORT's ORU mechanism fires when a lost track is re-associated after occlusion: it generates a virtual trajectory between the last pre-occlusion observation and the re-association observation, then replays KF updates along that trajectory to correct accumulated drift. In our batch setting, the backward pass may have tracked through the occluded region from the other direction, providing actual observations to use as the virtual trajectory instead of linear interpolation — a strictly better signal.

### 3. Variable Fish Count (Entries and Exits)

**Problem:** Fish continuously enter and leave the frame. The tracker must initialize new tracks promptly and terminate lost tracks without premature deletion or excessive lingering.

**How we address it:** Asymmetric birth/death logic tuned to frame-edge proximity. Detections appearing near the frame boundary immediately initialize tracks (no confirmation delay), since they likely represent entering fish. Tracks whose last observation was near the frame edge are terminated quickly after going unmatched — they likely left the frame. Tracks that vanish mid-frame are kept alive longer (higher `max_age`), as they are likely occluded and will reappear. The backward pass also seeds tracks for fish that enter mid-chunk: a fish first detected at frame 150 in the forward pass gets tracked backward to earlier frames where it may have been present but not yet associated.

### 4. Similar Appearance Across Individuals

**Problem:** Fish of the same species are visually near-identical, making appearance-based re-identification unreliable without a dedicated ReID model.

**How we address it (V1):** We rely on keypoint configuration (body curvature, heading) and motion consistency rather than visual appearance. OKS over 15 spine keypoints is discriminative enough when fish are not perfectly overlapping. Appearance embeddings are deferred to V2, where they can be batch-extracted on GPU across all detections in the chunk.

### 5. Chunk Boundary Continuity

**Problem:** Processing in 300-frame batches means tracks must be handed off between chunks without identity loss.

**How we address it:** At chunk end, we serialize the full Kalman filter state (mean and covariance) for every active track, along with recent observation history needed for OCM direction computation. The covariance is critical — it encodes how uncertain the track's predicted position is, which determines the matching gate width at the start of the next chunk. The next chunk initializes its track set from these carried-over states and matches new detections against KF-predicted positions. No chunk overlap is required; the filter state is sufficient.

### 6. Measurement Noise Correlation in Interpolated Keypoints

**Problem:** The 15 tracked keypoints are interpolated from 6 raw model outputs. Adjacent interpolated keypoints have correlated measurement errors, violating the diagonal measurement noise assumption.

**How we address it (V1):** We use a diagonal measurement noise covariance matrix (R) with inflated noise values for interpolated keypoints relative to the 6 raw keypoints. This is an approximation. If the filter proves over-confident in interpolated keypoint positions (visible as jittery corrections when observations arrive), the R values for those keypoints should be increased, or a block-diagonal R with off-diagonal terms for adjacent interpolated keypoints can be introduced.

---

## Core Algorithm: Pseudocode

### Data Structures

```
Detection:
    frame_idx    : int
    keypoints    : array[15, 2]       # (x, y) for each spine keypoint
    scores       : array[15]          # confidence per keypoint
    obb          : array[5] | None    # (cx, cy, w, h, theta), optional
    edge_flags   : (bool, bool, bool, bool)  # near (top, bottom, left, right) edge

TrackState:
    track_id         : int
    kf_mean          : array[60]      # [x1,y1,...,x15,y15, vx1,vy1,...,vx15,vy15]
    kf_covariance    : array[60, 60]
    observations     : deque of (frame_idx, keypoints)  # last N observations
    hits             : int            # consecutive successful matches
    misses           : int            # consecutive frames without match
    age              : int            # total frames since track creation
    state            : enum {TENTATIVE, ACTIVE, LOST}
    birth_near_edge  : bool           # was the track initialized near a frame edge?

ChunkHandoff:
    active_tracks    : list[TrackState]
    next_track_id    : int            # counter to ensure globally unique IDs
    chunk_end_frame  : int

Tracklet:
    track_id         : int
    detections       : dict[frame_idx -> Detection]
    interpolated     : dict[frame_idx -> keypoints]  # gap-filled frames
```

### Top-Level Pipeline

```
function track_chunk(detections_by_frame, handoff_in) -> (tracklets, handoff_out):
    """
    Main entry point. Processes a 300-frame chunk.

    Args:
        detections_by_frame: dict mapping frame_idx -> list[Detection]
        handoff_in: ChunkHandoff | None (None for first chunk)

    Returns:
        tracklets: list[Tracklet] — finalized tracks for this chunk
        handoff_out: ChunkHandoff — state to pass to the next chunk
    """

    # Initialize track set from previous chunk or empty
    if handoff_in is not None:
        initial_tracks = handoff_in.active_tracks
        next_id = handoff_in.next_track_id
    else:
        initial_tracks = []
        next_id = 0

    # Stage 1: Forward pass
    forward_tracklets, forward_end_tracks, next_id = run_ocsort_pass(
        detections_by_frame,
        initial_tracks,
        next_id,
        direction="forward"
    )

    # Stage 2: Backward pass (no carried-over tracks; discovers from scratch)
    backward_tracklets, _, next_id = run_ocsort_pass(
        detections_by_frame,
        initial_tracks=[],
        next_id=next_id,
        direction="backward"
    )

    # Stage 3: Merge forward and backward tracklets
    merged_tracklets = merge_bidirectional(
        forward_tracklets,
        backward_tracklets,
        detections_by_frame
    )

    # Stage 4: Interpolate small gaps
    final_tracklets = interpolate_gaps(merged_tracklets, max_gap=5)

    # Package handoff
    handoff_out = ChunkHandoff(
        active_tracks = forward_end_tracks,  # use forward pass's KF states
        next_track_id = next_id,
        chunk_end_frame = max(detections_by_frame.keys())
    )

    return final_tracklets, handoff_out
```

### OC-SORT Pass (Forward or Backward)

```
function run_ocsort_pass(detections_by_frame, initial_tracks, next_id, direction):
    """
    Runs a single OC-SORT pass over the chunk.

    Returns:
        tracklets: list[Tracklet]
        active_tracks: list[TrackState] at end of pass
        next_id: updated ID counter
    """

    tracks = deep_copy(initial_tracks)
    tracklets = {t.track_id: Tracklet(t.track_id) for t in tracks}

    frames = sorted(detections_by_frame.keys())
    if direction == "backward":
        frames = reversed(frames)

    for frame_idx in frames:
        detections = detections_by_frame[frame_idx]

        # --- KF Predict ---
        for track in tracks:
            kf_predict(track)

        # --- Build cost matrix ---
        active_and_lost = [t for t in tracks if t.state in {ACTIVE, LOST}]
        cost_matrix = build_cost_matrix(active_and_lost, detections)

        # --- Primary association (Hungarian) ---
        matched, unmatched_tracks, unmatched_dets = hungarian_assign(
            cost_matrix,
            active_and_lost,
            detections,
            threshold=MATCH_THRESHOLD
        )

        # --- Update matched tracks ---
        for track, det in matched:
            # If track was LOST, trigger ORU before updating
            if track.state == LOST:
                virtual_traj = generate_virtual_trajectory(
                    track.observations[-1],  # last seen observation
                    (frame_idx, det.keypoints),  # re-association observation
                    track.misses  # number of missed frames
                )
                kf_replay_updates(track, virtual_traj)

            kf_update(track, det.keypoints)
            track.observations.append((frame_idx, det.keypoints))
            track.hits += 1
            track.misses = 0
            track.state = ACTIVE if track.hits >= CONFIRM_HITS else TENTATIVE
            tracklets[track.track_id].detections[frame_idx] = det

        # --- OCR: secondary association for lost tracks ---
        remaining_lost = [t for t in unmatched_tracks if t.state == LOST]
        if remaining_lost and unmatched_dets:
            ocr_matches, unmatched_tracks_2, unmatched_dets = ocr_associate(
                remaining_lost,
                unmatched_dets
            )
            for track, det in ocr_matches:
                # Same ORU + update logic as above
                trigger_oru_and_update(track, det, frame_idx, tracklets)
                # Remove from unmatched
                unmatched_tracks.remove(track)

        # --- Handle unmatched tracks ---
        for track in unmatched_tracks:
            track.misses += 1
            if should_terminate(track):
                track.state = DEAD
            else:
                track.state = LOST

        # --- Initialize new tracks from unmatched detections ---
        for det in unmatched_dets:
            new_track = initialize_track(det, frame_idx, next_id)
            next_id += 1
            tracks.append(new_track)
            tracklets[new_track.track_id] = Tracklet(new_track.track_id)
            tracklets[new_track.track_id].detections[frame_idx] = det

        # --- Prune dead tracks ---
        tracks = [t for t in tracks if t.state != DEAD]

    # Collect all tracklets and final active track states
    all_tracklets = [t for t in tracklets.values() if len(t.detections) >= MIN_TRACK_LENGTH]
    active_tracks = [t for t in tracks if t.state in {ACTIVE, LOST}]

    return all_tracklets, active_tracks, next_id
```

### Cost Matrix Construction

```
function build_cost_matrix(tracks, detections) -> array[N_tracks, N_dets]:
    """
    Builds the association cost matrix combining OKS and OCM direction
    consistency.
    """

    N_t = len(tracks)
    N_d = len(detections)
    cost = full((N_t, N_d), fill=INF)

    for i, track in enumerate(tracks):
        predicted_kps = extract_predicted_keypoints(track.kf_mean)  # shape [15, 2]

        for j, det in enumerate(detections):
            # --- OKS distance ---
            oks = compute_oks(predicted_kps, det.keypoints, det.scores)
            oks_cost = 1.0 - oks  # OKS is similarity in [0,1]; invert to cost

            # --- OCM: direction consistency ---
            if len(track.observations) >= 2:
                ocm_cost = compute_ocm(track.observations, det.keypoints)
            else:
                ocm_cost = 0.0

            # --- Combined cost ---
            cost[i, j] = WEIGHT_OKS * oks_cost + WEIGHT_OCM * ocm_cost

    return cost


function compute_oks(predicted_kps, detected_kps, scores) -> float:
    """
    Object Keypoint Similarity between predicted and detected keypoints.

    For each keypoint k:
        ks_k = exp(-d_k^2 / (2 * s^2 * sigma_k^2))
    where d_k is Euclidean distance, s is the object scale (OBB area),
    and sigma_k is a per-keypoint constant controlling tolerance.

    Returns the weighted average of per-keypoint similarities.
    """

    dists_sq = sum_sq(predicted_kps - detected_kps, axis=1)  # shape [15]
    scale_sq = estimate_scale(detected_kps) ** 2  # e.g., from OBB area or kp span
    sigmas = get_keypoint_sigmas()  # shape [15], tuned per keypoint position

    ks = exp(-dists_sq / (2.0 * scale_sq * sigmas ** 2))

    # Weight by detection confidence if available
    if scores is not None:
        valid = scores > SCORE_THRESHOLD
        return mean(ks[valid]) if any(valid) else 0.0
    return mean(ks)


function compute_ocm(observations, detected_kps) -> float:
    """
    Observation-Centric Momentum: direction consistency cost.

    Computes the heading direction from the two most recent observations,
    then checks whether the new detection is consistent with that direction.
    """

    (_, kps_prev), (_, kps_curr) = observations[-2], observations[-1]

    # Heading = head keypoint (index 0) direction of travel
    observed_velocity = kps_curr[0] - kps_prev[0]
    predicted_velocity = detected_kps[0] - kps_curr[0]

    # Cosine similarity; invert to cost
    cos_sim = dot(observed_velocity, predicted_velocity) / (
        norm(observed_velocity) * norm(predicted_velocity) + EPS
    )

    return 1.0 - (cos_sim + 1.0) / 2.0  # map [-1,1] -> [1,0] -> cost [0,1]
```

### Bidirectional Merge

```
function merge_bidirectional(forward_tracklets, backward_tracklets, detections_by_frame):
    """
    Merges forward and backward tracklets into a unified set.

    Strategy:
    1. Build a detection-to-tracklet mapping for both directions.
    2. Match forward and backward tracklets that share detections.
    3. For matched pairs, take the union of their frame coverage,
       preferring the assignment with lower cost at each frame.
    4. Adopt unmatched backward tracklets that cover frames no
       forward tracklet reached.
    """

    # Map (frame_idx, det_idx) -> tracklet for each direction
    fwd_assignment = {}  # (frame, det_idx) -> (tracklet_id, cost)
    bwd_assignment = {}

    for t in forward_tracklets:
        for frame_idx, det in t.detections.items():
            det_idx = find_detection_index(det, detections_by_frame[frame_idx])
            fwd_assignment[(frame_idx, det_idx)] = t.track_id

    for t in backward_tracklets:
        for frame_idx, det in t.detections.items():
            det_idx = find_detection_index(det, detections_by_frame[frame_idx])
            bwd_assignment[(frame_idx, det_idx)] = t.track_id

    # Find backward tracklets that overlap significantly with forward ones
    overlap_counts = Counter()  # (fwd_id, bwd_id) -> count of shared detections
    for key in fwd_assignment:
        if key in bwd_assignment:
            overlap_counts[(fwd_assignment[key], bwd_assignment[key])] += 1

    # Greedily match forward <-> backward by overlap count
    matched_bwd_ids = set()
    merged = {}

    for (fwd_id, bwd_id), count in overlap_counts.most_common():
        fwd_t = get_tracklet(forward_tracklets, fwd_id)
        bwd_t = get_tracklet(backward_tracklets, bwd_id)
        min_overlap = 0.5 * min(len(fwd_t.detections), len(bwd_t.detections))

        if count >= min_overlap and bwd_id not in matched_bwd_ids:
            # Merge: take union of frames, forward gets priority
            merged_t = Tracklet(track_id=fwd_id)
            all_frames = set(fwd_t.detections.keys()) | set(bwd_t.detections.keys())
            for f in all_frames:
                if f in fwd_t.detections:
                    merged_t.detections[f] = fwd_t.detections[f]
                else:
                    merged_t.detections[f] = bwd_t.detections[f]
            merged[fwd_id] = merged_t
            matched_bwd_ids.add(bwd_id)

    # Keep unmerged forward tracklets as-is
    for t in forward_tracklets:
        if t.track_id not in merged:
            merged[t.track_id] = t

    # Adopt unmatched backward tracklets that fill gaps
    for t in backward_tracklets:
        if t.track_id not in matched_bwd_ids:
            # Check this doesn't conflict with existing merged tracklets
            if not conflicts_with(t, merged):
                merged[t.track_id] = t

    return list(merged.values())
```

### Track Lifecycle

```
function initialize_track(det, frame_idx, track_id) -> TrackState:
    """
    Creates a new TrackState from a first detection.
    """

    kf_mean = zeros(60)
    kf_mean[0:30] = det.keypoints.flatten()  # positions
    kf_mean[30:60] = 0.0  # zero initial velocity

    kf_covariance = build_initial_covariance(det)
    # High uncertainty on velocities, moderate on positions
    # Inflated position noise for interpolated keypoints (indices 6–14)

    return TrackState(
        track_id = track_id,
        kf_mean = kf_mean,
        kf_covariance = kf_covariance,
        observations = deque([(frame_idx, det.keypoints)], maxlen=OBS_HISTORY),
        hits = 1,
        misses = 0,
        age = 0,
        state = TENTATIVE,
        birth_near_edge = any(det.edge_flags)
    )


function should_terminate(track) -> bool:
    """
    Asymmetric termination based on where the track was last seen.
    """

    last_obs_frame, last_obs_kps = track.observations[-1]
    near_edge = is_near_frame_edge(last_obs_kps)

    if near_edge:
        return track.misses > MAX_EDGE_MISS      # e.g., 3 frames
    else:
        return track.misses > MAX_INTERIOR_MISS   # e.g., 15 frames
```

### Gap Interpolation

```
function interpolate_gaps(tracklets, max_gap) -> list[Tracklet]:
    """
    Fills small gaps in tracklets using cubic spline interpolation
    over the observed keypoint positions.
    """

    for tracklet in tracklets:
        frames = sorted(tracklet.detections.keys())

        for i in range(len(frames) - 1):
            gap_start = frames[i]
            gap_end = frames[i + 1]
            gap_size = gap_end - gap_start - 1

            if 0 < gap_size <= max_gap:
                kps_start = tracklet.detections[gap_start].keypoints
                kps_end = tracklet.detections[gap_end].keypoints

                for g in range(1, gap_size + 1):
                    t = g / (gap_size + 1)  # interpolation parameter
                    # Cubic spline using surrounding context if available;
                    # fall back to linear if at tracklet boundary
                    interp_kps = spline_interpolate(
                        tracklet, gap_start, gap_end, t
                    )
                    interp_frame = gap_start + g
                    tracklet.interpolated[interp_frame] = interp_kps

    return tracklets
```

### Chunk Boundary Handoff

```
function package_handoff(active_tracks, next_id, chunk_end_frame) -> ChunkHandoff:
    return ChunkHandoff(
        active_tracks = [
            TrackState(
                track_id = t.track_id,
                kf_mean = t.kf_mean,
                kf_covariance = t.kf_covariance,
                observations = t.observations,  # last N, for OCM
                hits = t.hits,
                misses = t.misses,
                age = t.age,
                state = t.state,
                birth_near_edge = t.birth_near_edge
            )
            for t in active_tracks
        ],
        next_track_id = next_id,
        chunk_end_frame = chunk_end_frame
    )


function restore_from_handoff(handoff) -> (list[TrackState], int):
    """
    Restores track states at the beginning of a new chunk.
    The KF covariance already encodes uncertainty from the time gap
    between chunks (if any). No overlap frames needed.
    """

    tracks = deep_copy(handoff.active_tracks)
    return tracks, handoff.next_track_id
```

---

## Configuration Parameters

| Parameter | Suggested V1 Value | Notes |
|---|---|---|
| `MATCH_THRESHOLD` | 0.5 | OKS+OCM cost above which matches are rejected |
| `WEIGHT_OKS` | 0.85 | Relative weight of OKS in cost matrix |
| `WEIGHT_OCM` | 0.15 | Relative weight of direction consistency |
| `CONFIRM_HITS` | 2 | Frames to confirm tentative track (1 if near edge) |
| `MAX_EDGE_MISS` | 3 | Max miss frames before kill, last seen near edge |
| `MAX_INTERIOR_MISS` | 15 | Max miss frames before kill, last seen mid-frame |
| `MIN_TRACK_LENGTH` | 3 | Minimum detections to output a tracklet |
| `OBS_HISTORY` | 10 | Observation history kept per track for OCM |
| `SCORE_THRESHOLD` | 0.3 | Minimum keypoint confidence to include in OKS |

---

## Future Extensions (V2+)

- **Appearance embeddings:** Batch-extract ReID features on GPU for all detections in the chunk. Add cosine distance as a third term in the cost matrix. Use embeddings for long-range re-identification across extended occlusions.
- **Correlated measurement noise:** Replace diagonal R matrix with block structure reflecting the interpolation dependency among the 15 keypoints.
- **Global optimization:** Replace greedy forward/backward merge with minimum-cost network flow over the full chunk for globally optimal assignment.
- **Chunk overlap (if needed):** If boundary ID switches become a problem, add a small overlap window and reconcile assignments in the shared frames.
