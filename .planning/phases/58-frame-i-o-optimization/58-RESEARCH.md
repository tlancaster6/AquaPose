# Phase 58: Frame I/O Optimization - Research

**Researched:** 2026-03-04
**Domain:** Python threading, producer-consumer queues, OpenCV VideoCapture thread safety
**Confidence:** HIGH

## Summary

Phase 58 replaces `ChunkFrameSource.__iter__` internals with a background-thread prefetch that reads and undistorts frames sequentially (via `cap.read()`) instead of seeking each frame. The stdlib `threading.Thread` + `queue.Queue` pattern is the standard approach and is well-suited here: the GIL is released during both OpenCV decode (`cap.read()`) and undistortion (`cv2.remap()`), so the background thread genuinely overlaps I/O with GPU inference in the main thread.

The current `ChunkFrameSource.__iter__` calls `self._source.read_frame(global_idx)` for every frame, which seeks all 12 captures per frame. This is both slow (seek overhead) and unreliable (compressed codecs may decode to wrong frames after seek). Sequential `cap.read()` eliminates both problems.

**Primary recommendation:** Use `threading.Thread` (daemon) + `queue.Queue(maxsize=2)` with a sentinel-based shutdown protocol. Access `VideoFrameSource._captures` and `._undist_maps` directly from the background thread -- these are read-only after `__enter__` and safe to share.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Fixed prefetch depth of 2 frames (hardcoded default, no config field)
- ~144 MB footprint for 12 cameras at ~6 MB/frame
- Undistortion happens in the background decode thread (not main thread) -- main thread receives ready-to-use undistorted frames
- Replace ChunkFrameSource internals with background prefetch, **keep the class name ChunkFrameSource**
- No new class -- same API surface, same orchestrator usage
- `read_frame()` (random access) stays as a direct seek -- not prefetch-aware; only sequential iteration (`__iter__`) uses prefetch
- Prefetch thread starts lazily on first `__iter__` call, not on construction. Thread joins on iteration completion or `__exit__`
- Prefetch thread reads frames sequentially with `cap.read()` instead of seeking every frame
- Initial seek to `start_frame` only if captures aren't already positioned there (check `CAP_PROP_POS_FRAMES` first) -- avoids seeks entirely for sequential chunk processing
- One background thread reads all 12 cameras sequentially per frame, then enqueues the complete frame dict
- Initial seek (if needed) happens in the main thread before spawning the prefetch thread -- seek errors raise directly
- Exception sentinel pattern: on decode error, thread puts the exception into the queue and exits; main thread raises it on dequeue
- Daemon thread with a stop event checked between frames -- abandoned iteration doesn't block shutdown
- Single camera decode failure: skip that camera for the frame (log warning), don't kill the entire frame
- Must verify downstream stages handle a missing camera gracefully -- if not, fix as a prerequisite within this phase

### Claude's Discretion
- Queue implementation details (stdlib `queue.Queue` vs alternatives)
- Thread naming and logging patterns
- Exact position-check logic for avoiding initial seek
- Test strategy and mock patterns

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FIO-01 | Frame source prefetches frames in a background thread via producer-consumer queue | Core implementation: `threading.Thread` + `queue.Queue(maxsize=2)`, sequential `cap.read()` in background thread with undistortion, sentinel-based shutdown |
| FIO-02 | Prefetch source satisfies existing FrameSource protocol (drop-in replacement) | Same class name `ChunkFrameSource`, same API surface, `__iter__` returns identical `(local_idx, {cam_id: frame})` tuples |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `threading` | stdlib | Background decode thread | GIL released during OpenCV C extensions; daemon threads for clean shutdown |
| `queue` | stdlib | Bounded producer-consumer queue | Thread-safe, blocking put/get, maxsize for backpressure |
| `cv2` (OpenCV) | existing dep | Video decode + undistortion | Already used; `cap.read()` releases GIL |

### Recommendation: `queue.Queue`
Use stdlib `queue.Queue(maxsize=2)`. No need for `collections.deque` (not thread-safe for blocking), `multiprocessing.Queue` (IPC overhead), or third-party alternatives. `queue.Queue` is the standard, battle-tested choice for this exact pattern.

## Architecture Patterns

### Pattern 1: Prefetch Thread with Sentinel Shutdown

**What:** Background thread reads frames sequentially, enqueues complete frame dicts. Main thread consumes from queue in `__iter__`. Thread exits on stop event or EOF.

**When to use:** Always during `__iter__` -- this IS the iteration mechanism.

```python
import queue
import threading
from typing import Any

_SENTINEL = object()  # End-of-stream marker

class ChunkFrameSource:
    def __init__(self, source, start_frame, end_frame):
        self._source = source
        self.start_frame = start_frame
        self.end_frame = end_frame
        self._prefetch_queue: queue.Queue | None = None
        self._prefetch_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

    def __iter__(self):
        # 1. Seek captures to start_frame in main thread (if needed)
        # 2. Create queue, stop_event, start thread
        # 3. Yield from queue until sentinel or exception
        # 4. Join thread on completion
        ...

    def _prefetch_worker(self):
        """Background thread: sequential read + undistort + enqueue."""
        try:
            for local_idx in range(self.end_frame - self.start_frame):
                if self._stop_event.is_set():
                    return
                frames = {}
                for cam_id in self._source._camera_ids:
                    ret, frame = self._source._captures[cam_id].read()
                    if not ret:
                        logger.warning("Camera %s: decode failed at frame %d", cam_id, ...)
                        continue  # skip camera, don't kill frame
                    frames[cam_id] = undistort_image(frame, self._source._undist_maps[cam_id])
                self._prefetch_queue.put((local_idx, frames))
        except Exception as exc:
            self._prefetch_queue.put(exc)  # exception sentinel
            return
        self._prefetch_queue.put(_SENTINEL)  # normal EOF
```

### Pattern 2: Exception Sentinel Protocol

**What:** If the background thread encounters an unrecoverable error, it puts the exception object into the queue and exits. The main thread checks each dequeued item: if it's an `Exception`, re-raise it.

```python
def __iter__(self):
    # ... setup ...
    while True:
        item = self._prefetch_queue.get()
        if item is _SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item
```

**Why:** Clean error propagation across threads without shared mutable state. The queue itself IS the communication channel for both data and errors.

### Pattern 3: Lazy Thread Start + Cleanup in `__exit__`

**What:** Thread is not created in `__init__` or `__enter__`. It starts on first `__iter__` call. `__exit__` sets stop event and joins the thread to prevent resource leaks if iteration is abandoned.

```python
def __exit__(self, *exc):
    if self._stop_event is not None:
        self._stop_event.set()
    if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
        # Drain queue to unblock put() in worker
        while True:
            try:
                self._prefetch_queue.get_nowait()
            except queue.Empty:
                break
        self._prefetch_thread.join(timeout=5.0)
```

### Pattern 4: Initial Seek Logic

**What:** Before starting the prefetch thread, check if captures are already positioned at `start_frame`. Only seek if position is wrong.

```python
def _ensure_captures_positioned(self):
    """Seek all captures to start_frame if not already there. Main thread only."""
    for cam_id in self._source._camera_ids:
        cap = self._source._captures[cam_id]
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_pos != self.start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            # Verify seek succeeded
            actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if actual != self.start_frame:
                raise RuntimeError(
                    f"Camera {cam_id}: seek to frame {self.start_frame} failed "
                    f"(at frame {actual})"
                )
```

**Why:** The orchestrator processes chunks sequentially. After chunk 0 finishes reading frames 0-199, captures are at position 200 -- exactly where chunk 1 needs to start. No seek needed.

### Anti-Patterns to Avoid

- **Accessing captures from both threads simultaneously:** `read_frame()` seeks captures. If called while prefetch thread is running, it corrupts position. Document that `read_frame()` must NOT be called during active iteration. In practice this never happens (visualization uses `read_frame` post-pipeline, not during iteration).
- **Not draining the queue on early exit:** If `__exit__` just sets `stop_event` without draining the queue, the worker thread may block on `queue.put()` forever (queue is full). Must drain before joining.
- **Using `threading.Lock` around captures:** Unnecessary complexity. The design ensures only one thread accesses captures at a time (prefetch thread during iteration, main thread otherwise).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Thread-safe bounded queue | Ring buffer with locks | `queue.Queue(maxsize=2)` | Handles blocking, backpressure, thread safety correctly |
| Thread lifecycle management | Manual start/join tracking | `threading.Thread(daemon=True)` + `Event` | Daemon threads auto-exit on interpreter shutdown |

## Common Pitfalls

### Pitfall 1: Queue Deadlock on Abandoned Iteration
**What goes wrong:** Main thread stops consuming from the queue (exception, early break). Worker thread blocks on `queue.put()` because queue is full. Thread never exits, resources leak.
**Why it happens:** Bounded queue with `maxsize=2` blocks `put()` when full.
**How to avoid:** In `__exit__`, set `stop_event` then drain the queue before calling `thread.join()`. Use `put(item, timeout=1.0)` in the worker with periodic `stop_event` checks.
**Warning signs:** Thread join timeouts, growing memory, "thread still alive" warnings.

### Pitfall 2: OpenCV CAP_PROP_POS_FRAMES Unreliability
**What goes wrong:** `cap.get(CAP_PROP_POS_FRAMES)` may not reflect the actual decode position accurately with some codecs (H.264 B-frames, variable frame rate).
**Why it happens:** OpenCV position is based on internal packet counters, not actual decoded frame content.
**How to avoid:** Use the position check as a heuristic optimization only. If it says position matches, skip seek. If it says mismatch, seek. The sequential `cap.read()` path is always reliable regardless of the initial position check.
**Warning signs:** First frame of chunk N+1 looks identical to first frame of chunk N.

### Pitfall 3: Missing Camera KeyError in DetectionStage
**What goes wrong:** When prefetch skips a camera due to decode failure, the frames dict is missing that camera ID. `DetectionStage.run()` line 83 does `frames[cam_id]` which raises `KeyError`.
**Why it happens:** DetectionStage iterates `camera_ids` (from the source) and indexes into frames dict without checking membership.
**How to avoid:** Either (a) guard in DetectionStage: `if cam_id not in frames: continue`, or (b) in ChunkFrameSource, if a camera fails, insert a black frame of the correct shape as placeholder. Option (a) is cleaner -- log a warning, emit an empty detection list for that camera.
**Warning signs:** `KeyError` exceptions during pipeline run when a single camera has corrupt frames.

### Pitfall 4: Thread Safety of cv2.VideoCapture
**What goes wrong:** `cv2.VideoCapture.read()` is not documented as thread-safe. Multiple threads calling `read()` on different capture objects is fine, but the same capture object must only be accessed from one thread at a time.
**Why it happens:** OpenCV's C++ backend may share internal state.
**How to avoid:** Design ensures this: captures are only accessed by the prefetch thread during iteration, and only by the main thread when prefetch is not running. Document this invariant. Never call `read_frame()` during active `__iter__`.

### Pitfall 5: Re-entrancy of `__iter__`
**What goes wrong:** If `__iter__` is called twice (e.g., `for ... in source:` followed by another `for ... in source:`), the second call finds a stale thread/queue.
**Why it happens:** No guard against re-entrant iteration.
**How to avoid:** In `__iter__`, check if a thread is already running and raise `RuntimeError("iteration already in progress")`, or clean up the previous iteration first. Since chunks are single-use, raising is simpler and catches bugs.

## Code Examples

### Complete `__iter__` Flow
```python
def __iter__(self) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
        raise RuntimeError("ChunkFrameSource: concurrent iteration not supported")

    self._ensure_captures_positioned()

    self._prefetch_queue = queue.Queue(maxsize=2)
    self._stop_event = threading.Event()
    self._prefetch_thread = threading.Thread(
        target=self._prefetch_worker,
        name=f"prefetch-{self.start_frame}-{self.end_frame}",
        daemon=True,
    )
    self._prefetch_thread.start()

    try:
        while True:
            item = self._prefetch_queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        self._stop_event.set()
        # Drain queue to unblock worker's put()
        while True:
            try:
                self._prefetch_queue.get_nowait()
            except queue.Empty:
                break
        self._prefetch_thread.join(timeout=5.0)
        if self._prefetch_thread.is_alive():
            logger.warning("Prefetch thread did not exit within timeout")
```

### Worker Thread
```python
def _prefetch_worker(self) -> None:
    try:
        for local_idx in range(self.end_frame - self.start_frame):
            if self._stop_event.is_set():
                return
            frames: dict[str, np.ndarray] = {}
            for cam_id in self._source._camera_ids:
                cap = self._source._captures[cam_id]
                ret, raw = cap.read()
                if not ret:
                    logger.warning(
                        "Camera %s: decode failed at global frame %d",
                        cam_id, self.start_frame + local_idx,
                    )
                    continue
                if cam_id in self._source._undist_maps:
                    raw = undistort_image(raw, self._source._undist_maps[cam_id])
                frames[cam_id] = raw
            # Use timeout to allow stop_event checks
            while not self._stop_event.is_set():
                try:
                    self._prefetch_queue.put((local_idx, frames), timeout=0.5)
                    break
                except queue.Full:
                    continue
    except Exception as exc:
        try:
            self._prefetch_queue.put(exc, timeout=1.0)
        except queue.Full:
            pass
        return
    # Signal end of stream
    try:
        self._prefetch_queue.put(_SENTINEL, timeout=5.0)
    except queue.Full:
        pass
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-frame seek via `read_frame()` | Sequential `cap.read()` in background thread | This phase | Eliminates seek overhead (~12% of pipeline time), overlaps I/O with GPU inference |

**Current codebase state:**
- `ChunkFrameSource.__iter__` calls `self._source.read_frame(global_idx)` per frame -- this seeks 12 captures every frame
- `VideoFrameSource.__iter__` already uses sequential `cap.read()` but ChunkFrameSource does NOT delegate to it
- No threading exists anywhere in the codebase currently

## Open Questions

1. **Missing camera graceful handling in DetectionStage**
   - What we know: `DetectionStage.run()` line 83 does `frames[cam_id]` without checking membership -- will `KeyError` if camera missing
   - What's unclear: Whether MidlineStage has the same issue (it accesses `frames` similarly)
   - Recommendation: Add `if cam_id not in frames: continue` guards in both DetectionStage and MidlineStage. Emit empty detection/midline lists for missing cameras. This is a small prerequisite fix within this phase.

2. **VideoCapture internal state after sequential reads across chunks**
   - What we know: After chunk N finishes iterating, captures should be positioned at `start_frame + chunk_size`. Chunk N+1's `start_frame` should equal that position.
   - What's unclear: Whether OpenCV's internal buffer/state is perfectly aligned after a long sequence of `cap.read()` calls
   - Recommendation: The `CAP_PROP_POS_FRAMES` check before each chunk is sufficient insurance. If position matches, skip seek. This handles both the common case (sequential chunks) and edge cases (skipped chunks due to errors).

3. **`__exit__` responsibility**
   - What we know: ChunkFrameSource `__exit__` is currently a no-op. It needs to become the cleanup path for the prefetch thread.
   - What's unclear: Whether DetectionStage/MidlineStage always exhaust the iterator (which would trigger the `finally` block in `__iter__`)
   - Recommendation: Both `__iter__`'s `finally` block AND `__exit__` should handle cleanup. `__exit__` is the safety net for abandoned iteration.

## Sources

### Primary (HIGH confidence)
- Project source code: `src/aquapose/core/types/frame_source.py` -- full implementation of ChunkFrameSource and VideoFrameSource
- Project source code: `src/aquapose/engine/orchestrator.py` -- ChunkOrchestrator usage of ChunkFrameSource
- Project source code: `src/aquapose/core/detection/stage.py` -- frame consumption pattern
- Python stdlib docs: `threading.Thread`, `queue.Queue` -- well-established patterns

### Secondary (MEDIUM confidence)
- OpenCV `CAP_PROP_POS_FRAMES` reliability -- known limitation with compressed codecs, documented in OpenCV community

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- stdlib threading + queue, no external dependencies
- Architecture: HIGH -- producer-consumer is a textbook pattern; code structure is well-constrained by user decisions
- Pitfalls: HIGH -- threading pitfalls are well-known; OpenCV seek issues confirmed by user concern in CONTEXT.md
- Missing camera handling: MEDIUM -- need to verify MidlineStage frame access pattern during implementation

**Research date:** 2026-03-04
**Valid until:** 2026-04-04 (stable domain, no moving parts)
