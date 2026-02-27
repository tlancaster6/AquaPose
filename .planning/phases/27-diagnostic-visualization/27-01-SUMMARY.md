---
phase: 27-diagnostic-visualization
plan: 01
subsystem: engine
tags: [opencv, visualization, observer, tracklet, mosaic, diagnostics]

# Dependency graph
requires:
  - phase: 24-per-camera-2d-tracking
    provides: Tracklet2D type and tracks_2d context field
  - phase: 25-association
    provides: TrackletGroup type and tracklet_groups context field
  - phase: 22-pipeline-scaffolding
    provides: Observer protocol, EventBus, PipelineComplete event, observer_factory

provides:
  - TrackletTrailObserver class implementing Observer protocol
  - Per-camera centroid trail MP4 videos with fading tails and detected/coasted color distinction
  - Cross-camera association mosaic MP4 with consistent global fish ID color coding
  - tracklet_trail registered in observer factory for diagnostic mode and --add-observer

affects: [28-e2e-testing, diagnostic-mode, cli]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Paul Tol 22-color BGR palette for fish ID color coding (FISH_COLORS_BGR constant)"
    - "Fading trail via per-segment cv2.addWeighted alpha blend (oldest=0.3, newest=1.0)"
    - "coasted_color: blend base toward 128 at 50% for visual distinction without hue change"
    - "Deferred VideoSet/calibration imports inside generation methods (same as Overlay2DObserver)"
    - "_build_frame_lookup: pre-built O(1) per-camera per-frame index to avoid O(N*T) scan"
    - "Mosaic with tile_scale=0.35 downsampling; _mosaic_dims/_build_mosaic as static helpers"

key-files:
  created:
    - src/aquapose/engine/tracklet_trail_observer.py
    - tests/unit/engine/test_tracklet_trail_observer.py
  modified:
    - src/aquapose/engine/observer_factory.py
    - src/aquapose/engine/__init__.py

key-decisions:
  - "FISH_COLORS_BGR hardcoded in tracklet_trail_observer.py (not imported from elsewhere) to avoid tight coupling"
  - "calib_data typed as object in generation method signatures (ENG-07 boundary); cast at VideoSet callsite with type: ignore[arg-type]"
  - "VideoWriter_fourcc type: ignore not added — pre-existing pattern in overlay_observer.py; typecheck error count unchanged"
  - "_draw_trail_scaled is a separate method (not unified with _draw_trail) to keep scale_x/scale_y out of the per-camera path"

patterns-established:
  - "Diagnostic observer: guard on both tracks_2d and tracklet_groups; skip silently with logger.warning if either is None"
  - "Ungrouped tracklets (not in any TrackletGroup) get _GRAY_BGR color and fish_id=-1"

requirements-completed: [DIAG-01]

# Metrics
duration: 7min
completed: 2026-02-27
---

# Phase 27 Plan 01: Diagnostic Visualization Summary

**TrackletTrailObserver: fading BGR centroid trail videos per camera and tiled cross-camera association mosaic, color-coded by global fish ID via Paul Tol 22-color palette**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-27T21:50:07Z
- **Completed:** 2026-02-27T21:56:43Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- TrackletTrailObserver implementing Observer protocol — responds to PipelineComplete, passive and fault-tolerant
- Per-camera trail videos: fading alpha-blended polyline trails with detected (full color) vs coasted (blended-to-gray) distinction; global fish ID label at trail head
- Association mosaic video: all cameras tiled at tile_scale=0.35 downsampling with consistent fish ID colors for cross-camera association quality inspection
- Observer registered in diagnostic mode and --add-observer via observer_factory; TrackletTrailObserver exported from aquapose.engine

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement TrackletTrailObserver** - `8354744` (feat)
2. **Task 2: Wire factory, exports, unit tests** - `b00d7ac` (feat)

## Files Created/Modified

- `src/aquapose/engine/tracklet_trail_observer.py` - TrackletTrailObserver class: color maps, frame lookup, trail drawing, per-camera and mosaic video generation
- `tests/unit/engine/test_tracklet_trail_observer.py` - 9 unit tests (all passing)
- `src/aquapose/engine/observer_factory.py` - TrackletTrailObserver import, _OBSERVER_MAP entry, diagnostic mode block, extra_observers elif branch
- `src/aquapose/engine/__init__.py` - TrackletTrailObserver import and __all__ entry

## Decisions Made

- FISH_COLORS_BGR hardcoded in module (not imported) to avoid tight coupling with Overlay2DObserver
- calib_data passes as `object` to maintain ENG-07 boundary; `type: ignore[arg-type]` at VideoSet callsite
- VideoWriter_fourcc typecheck error not suppressed (pre-existing pattern in overlay_observer.py; no new net errors introduced)
- Separate `_draw_trail_scaled` method for mosaic tile rendering keeps scale factors isolated from per-camera path

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- TrackletTrailObserver is complete and wired; diagnostic mode produces trail and mosaic videos on PipelineComplete
- Phase 28 (e2e testing) can consume the full diagnostic observer stack

---
*Phase: 27-diagnostic-visualization*
*Completed: 2026-02-27*
