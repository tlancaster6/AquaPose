---
phase: 51-frame-source-refactor
verified: 2026-03-03T23:30:00Z
status: passed
score: 9/9 must-haves verified
---

# Phase 51: Frame Source Refactor Verification Report

**Phase Goal:** Stages receive frames from an injectable source instead of opening VideoSet directly
**Verified:** 2026-03-03T23:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                                           | Status     | Evidence                                                                                          |
|----|-----------------------------------------------------------------------------------------------------------------|------------|---------------------------------------------------------------------------------------------------|
| 1  | FrameSource protocol exists in core/types/ and yields (frame_idx, dict[str, ndarray])                          | VERIFIED   | `frame_source.py` defines `@runtime_checkable class FrameSource(Protocol)` with correct signature |
| 2  | VideoFrameSource handles video discovery, calibration loading, undistortion, and max_frames                     | VERIFIED   | `VideoFrameSource.__init__` calls `discover_camera_videos`, `load_calibration_data`, `compute_undistortion_maps`; `max_frames` applied in `__iter__` and `__enter__` |
| 3  | DetectionStage and MidlineStage accept a FrameSource via constructor and never open VideoSet internally         | VERIFIED   | Both constructors take `frame_source: FrameSource`; no `VideoSet` import in either file           |
| 4  | build_stages() creates a VideoFrameSource and passes it to both stages                                          | VERIFIED   | `pipeline.py` constructs `VideoFrameSource(video_dir=config.video_dir, calibration_path=config.calibration_path)` and passes it to both `DetectionStage` and `MidlineStage` |
| 5  | Overlay2DObserver and TrackletTrailObserver receive a FrameSource instead of video_dir + calibration_path       | VERIFIED   | Both constructors accept `frame_source: FrameSource | None = None`; no `VideoSet` imports         |
| 6  | stop_frame is absent from PipelineConfig; _RENAME_HINTS provides migration hint                                 | VERIFIED   | `stop_frame` not a field on `PipelineConfig`; `_RENAME_HINTS["stop_frame"]` = `"max_frames on frame source..."` |
| 7  | VideoSet class is deleted — no code imports from io/video.py                                                    | VERIFIED   | `src/aquapose/io/video.py` does not exist; zero `VideoSet` import matches across `src/`           |
| 8  | observer_factory passes frame_source to observers that need frame access                                        | VERIFIED   | `build_observers()` accepts `frame_source: FrameSource | None = None` and passes it to `Overlay2DObserver` and `TrackletTrailObserver` in all modes |
| 9  | CLI creates VideoFrameSource once and shares it with both build_stages and build_observers                      | VERIFIED   | `cli.py` line 127: creates `VideoFrameSource`; line 132: passes to `build_stages`; line 155: passes to `build_observers` |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact                                              | Expected                                                    | Status     | Details                                                                          |
|-------------------------------------------------------|-------------------------------------------------------------|------------|----------------------------------------------------------------------------------|
| `src/aquapose/core/types/frame_source.py`             | FrameSource protocol + VideoFrameSource                     | VERIFIED   | 249 lines; protocol runtime_checkable; VideoFrameSource fully implemented         |
| `src/aquapose/core/types/__init__.py`                 | Exports FrameSource and VideoFrameSource                    | VERIFIED   | Both in `__all__` and imported from `frame_source`                                |
| `src/aquapose/core/detection/stage.py`                | DetectionStage accepting FrameSource via constructor        | VERIFIED   | Constructor: `frame_source: FrameSource`; `run()` uses `with self._frame_source` |
| `src/aquapose/core/midline/stage.py`                  | MidlineStage accepting FrameSource via constructor          | VERIFIED   | Constructor: `frame_source: FrameSource`; `run()` uses `with self._frame_source` |
| `src/aquapose/engine/pipeline.py`                     | build_stages() creates VideoFrameSource and injects it      | VERIFIED   | Lines 379-404 show VideoFrameSource creation and injection to both stages         |
| `src/aquapose/engine/config.py`                       | PipelineConfig without stop_frame; _RENAME_HINTS updated    | VERIFIED   | `stop_frame` not a dataclass field; `_RENAME_HINTS` entry points to `max_frames` |
| `src/aquapose/engine/observer_factory.py`             | build_observers accepting frame_source parameter            | VERIFIED   | Signature: `frame_source: FrameSource | None = None`; passed to both observers   |
| `src/aquapose/engine/overlay_observer.py`             | Overlay2DObserver accepting optional FrameSource            | VERIFIED   | Constructor: `frame_source: FrameSource | None = None`; no VideoSet import        |
| `src/aquapose/engine/tracklet_trail_observer.py`      | TrackletTrailObserver accepting optional FrameSource, no stop_frame | VERIFIED | Constructor: `frame_source: FrameSource | None = None`; no stop_frame param; no VideoSet import |
| `src/aquapose/io/video.py`                            | DELETED                                                     | VERIFIED   | File does not exist                                                               |
| `tests/unit/core/types/test_frame_source.py`          | Protocol conformance, max_frames, camera_ids, error tests   | VERIFIED   | 219 lines; tests protocol isinstance, max_frames truncation, camera_ids sort, FileNotFoundError, k_new |

### Key Link Verification

| From                                      | To                                          | Via                                              | Status   | Details                                                                                              |
|-------------------------------------------|---------------------------------------------|--------------------------------------------------|----------|------------------------------------------------------------------------------------------------------|
| `engine/pipeline.py`                      | `core/types/frame_source.py`                | build_stages() constructs VideoFrameSource       | WIRED    | `VideoFrameSource(video_dir=..., calibration_path=...)` at line 380; from `from aquapose.core.types import VideoFrameSource` |
| `core/detection/stage.py`                 | `core/types/frame_source.py`                | DetectionStage.__init__ accepts FrameSource      | WIRED    | `frame_source: FrameSource` param; `TYPE_CHECKING` guard import; iterated in `run()` via `with self._frame_source` |
| `engine/observer_factory.py`              | `engine/overlay_observer.py`                | build_observers passes frame_source              | WIRED    | `frame_source=frame_source` passed to `Overlay2DObserver(...)` in synthetic, diagnostic, and additive paths |
| `engine/observer_factory.py`              | `engine/tracklet_trail_observer.py`         | build_observers passes frame_source              | WIRED    | `frame_source=frame_source` passed to `TrackletTrailObserver(...)` in synthetic, diagnostic, and additive paths |
| `cli.py`                                  | `engine/pipeline.py` + `observer_factory.py`| Shared VideoFrameSource                          | WIRED    | `frame_source` created at line 127; passed to `build_stages` line 132 and `build_observers` line 155 |

### Requirements Coverage

| Requirement | Source Plan | Description                                                                                          | Status    | Evidence                                                                                          |
|-------------|-------------|------------------------------------------------------------------------------------------------------|-----------|---------------------------------------------------------------------------------------------------|
| FRAME-01    | 51-01, 51-02 | DetectionStage and MidlineStage receive frames from an injectable frame source instead of opening VideoSet directly | SATISFIED | Both stages accept `frame_source: FrameSource`; no VideoSet imports; build_stages injects shared VideoFrameSource |
| FRAME-02    | 51-01       | Frame source yields `(frame_idx, dict[str, ndarray])` — local frame index plus per-camera undistorted frames | SATISFIED | `VideoFrameSource.__iter__` yields `(frame_idx, {cam_id: undistorted_frame})`; protocol signature matches |
| FRAME-03    | 51-02       | `stop_frame` removed from pipeline config — frame windowing is a frame-source concern managed by the orchestrator | SATISFIED | `stop_frame` absent from `PipelineConfig` dataclass; `_RENAME_HINTS` provides migration hint; `max_frames` lives on `VideoFrameSource` constructor |

All three FRAME requirements are satisfied. No orphaned requirements found for Phase 51.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `core/midline/stage.py` | 492 | `return {}` | Info | Legitimate guard return in `_filter_frame_detections` when no index entry exists for frame |
| `engine/overlay_observer.py` | 309, 321, 337, 340 | `return None` | Info | Legitimate early-exit guards in helper method checking for missing spline data |

No blocker anti-patterns found. All `return {}` and `return None` occurrences are legitimate guard returns inside helper functions, not stub implementations.

### Human Verification Required

None — all goal truths are verifiable through static code inspection for this refactor phase. The frame source injection is structural and fully traceable without running the pipeline.

### Gaps Summary

No gaps found. All nine must-haves are verified:

- FrameSource protocol is properly defined as `@runtime_checkable` in `core/types/frame_source.py` with the correct interface.
- VideoFrameSource is a complete, non-stub implementation absorbing video discovery, calibration loading, undistortion, and max_frames windowing.
- DetectionStage and MidlineStage constructors no longer accept `video_dir` for frame access and have no `VideoSet` imports.
- `build_stages()` constructs one `VideoFrameSource` and passes it to both stages in production mode.
- Overlay2DObserver and TrackletTrailObserver accept injected `FrameSource`; no `VideoSet` imports remain in either file.
- `stop_frame` is not a field on `PipelineConfig`; `_RENAME_HINTS` provides a clear migration hint.
- `src/aquapose/io/video.py` (VideoSet) is confirmed deleted; zero `VideoSet` imports found in `src/`.
- `observer_factory.build_observers()` has `frame_source` parameter and passes it to all observer constructors that need frame access.
- CLI creates one `VideoFrameSource` and shares it with both `build_stages` and `build_observers`.

---

_Verified: 2026-03-03T23:30:00Z_
_Verifier: Claude (gsd-verifier)_
