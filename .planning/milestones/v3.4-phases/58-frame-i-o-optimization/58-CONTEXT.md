# Phase 58: Frame I/O Optimization - Context

**Gathered:** 2026-03-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Add background-thread prefetch to ChunkFrameSource so frame decoding overlaps with GPU inference, eliminating seek overhead and GPU idle time between frames. Drop-in replacement — all stage code unaffected, `aquapose eval` produces identical results.

</domain>

<decisions>
## Implementation Decisions

### Buffer sizing
- Fixed prefetch depth of 2 frames (hardcoded default, no config field)
- ~144 MB footprint for 12 cameras at ~6 MB/frame
- Undistortion happens in the background decode thread (not main thread) — main thread receives ready-to-use undistorted frames

### Integration path
- Replace ChunkFrameSource internals with background prefetch, **keep the class name ChunkFrameSource**
- No new class — same API surface, same orchestrator usage
- `read_frame()` (random access) stays as a direct seek — not prefetch-aware; only sequential iteration (`__iter__`) uses prefetch
- Prefetch thread starts lazily on first `__iter__` call, not on construction. Thread joins on iteration completion or `__exit__`

### Sequential read fix
- Prefetch thread reads frames sequentially with `cap.read()` instead of seeking every frame
- Initial seek to `start_frame` only if captures aren't already positioned there (check `CAP_PROP_POS_FRAMES` first) — avoids seeks entirely for sequential chunk processing
- One background thread reads all 12 cameras sequentially per frame, then enqueues the complete frame dict
- Initial seek (if needed) happens in the main thread before spawning the prefetch thread — seek errors raise directly

### Error propagation
- Exception sentinel pattern: on decode error, thread puts the exception into the queue and exits; main thread raises it on dequeue
- Daemon thread with a stop event checked between frames — abandoned iteration doesn't block shutdown
- Single camera decode failure: skip that camera for the frame (log warning), don't kill the entire frame
- Must verify downstream stages (detection, tracking, association) handle a missing camera gracefully — if not, fix as a prerequisite within this phase

### Claude's Discretion
- Queue implementation details (stdlib `queue.Queue` vs alternatives)
- Thread naming and logging patterns
- Exact position-check logic for avoiding initial seek
- Test strategy and mock patterns

</decisions>

<specifics>
## Specific Ideas

- User concerned about OpenCV seek inaccuracy with compressed codecs — sequential reads are both faster and more reliable than per-frame seeking
- Tolerance for partial camera failures: "If one camera out of 12 has a corrupt frame, killing the 11 other cameras worth of good data seems excessive"

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `FrameSource` protocol (`core/types/frame_source.py:34`): defines `__iter__`, `read_frame`, `__len__`, `camera_ids`, context manager
- `VideoFrameSource` (`core/types/frame_source.py:81`): concrete multi-camera reader with undistortion; owns the `cv2.VideoCapture` handles
- `ChunkFrameSource` (`core/types/frame_source.py:251`): chunk-windowed view over VideoFrameSource — this is the class being modified
- `undistort_image()` from `calibration.loader`: called per-camera per-frame during reads

### Established Patterns
- Context manager protocol: `VideoFrameSource` opens/closes captures; `ChunkFrameSource` is a no-op context manager (orchestrator owns the lifecycle)
- Frozen config hierarchy: stage configs are frozen dataclasses; no new config field needed (hardcoded prefetch depth)
- Strict/tolerant error modes: `strict` raises, `tolerant` logs and continues

### Integration Points
- `engine/orchestrator.py`: creates `ChunkFrameSource` views per chunk — no changes needed if class name and API are preserved
- `core/detection/stage.py` and `core/midline/stage.py`: consume frames from the source via iteration
- `engine/pipeline.py`: wraps stage calls; unaware of frame source internals

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 58-frame-i-o-optimization*
*Context gathered: 2026-03-04*
