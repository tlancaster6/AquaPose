# Phase 24: Per-Camera 2D Tracking - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Independent per-camera 2D tracking of fish detections using OC-SORT, producing structured tracklets with frame-level status tags. Each camera is tracked independently — no cross-camera awareness. Cross-camera association is Phase 25.

</domain>

<decisions>
## Implementation Decisions

### OC-SORT Sourcing
- Use **boxmot** third-party package (pip-installable, includes OC-SORT + alternatives)
- Tracker choice (OC-SORT, ByteTrack, plain SORT) is a **configurable model** within a single tracking backend, not a swappable backend — all share the same data flow (detections in → tracklets out)
- Third-party tracker objects are **fully wrapped** behind our own `Tracklet2D` dataclass — downstream never sees boxmot internals
- The tracking module isolates the boxmot dependency entirely

### Coasting Behavior
- Default coast window: **30 frames** (~1 sec at 30fps), configurable via `max_coast_frames` config parameter
- Frame status is **binary**: `"detected"` or `"coasted"` — no additional quality signal (coast count, etc.)
- All tracklets active during the batch are included in output (both completed and still-active), per the chunk-aware architecture in MS3-SPECSEED
- CarryForward preserves internal tracker state for continuity into the next batch

### Tracklet Lifecycle
- Tracks use a **probationary period** (`n_init`, default 3) — must accrue N matched detections before confirmation
- **Only confirmed tracklets** appear in `tracks_2d` output — tentative and dropped tracks are internal to the tracker
- Track IDs are **simple incrementing integers** per camera; `camera_id` is a separate field — the `(camera_id, track_id)` pair is unique
- Tracklet2D fields match MS3-SPECSEED exactly: `camera_id`, `track_id`, `frames`, `centroids`, `bboxes`, `frame_status`

### CarryForward Design
- CarryForward is a **generic typed container** with optional per-stage slots (e.g. `tracking_state`, `association_state`) — extensible for future stages
- Tracking state stored as **opaque blob** — only the tracking stage knows how to serialize/deserialize it
- **Memory-only** between batches within a single run — no disk serialization (runs restart from beginning if interrupted)
- First batch receives `carry=None` — stages check for None and initialize fresh

### Claude's Discretion
- Exact boxmot API integration details and version pinning
- TrackingStage config dataclass structure
- How Tracklet2D stores per-frame arrays (lists vs numpy arrays)
- Test strategy (unit tests for wrapper, integration with detection stage)

</decisions>

<specifics>
## Specific Ideas

- Configurable model pattern per GUIDEBOOK §8: tracker choice is config, not a backend swap
- n_init probationary period is standard SORT-family — use boxmot's built-in support
- CarryForward generic container should be defined in engine/ (used by pipeline outer loop) with typed slots that stages populate

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 24-per-camera-2d-tracking*
*Context gathered: 2026-02-27*
