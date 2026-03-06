---
phase: 17-observers
plan: 01
subsystem: engine
tags: [timing, profiling, observer]

requires:
  - phase: 13-engine-core
    provides: Observer protocol, EventBus, pipeline lifecycle events
provides:
  - TimingObserver for per-stage wall-clock profiling
affects: [18-cli]

tech-stack:
  added: []
  patterns: [observer-as-pure-event-consumer]

key-files:
  created:
    - src/aquapose/engine/timing.py
    - tests/unit/engine/test_timing.py
  modified:
    - src/aquapose/engine/__init__.py

key-decisions:
  - "TimingObserver is always-on — attached to every pipeline run regardless of mode"

patterns-established:
  - "Observer pattern: subscribe to Event base, filter by isinstance in on_event"

requirements-completed: [OBS-01]

duration: 5min
completed: 2026-02-26
---

# Plan 17-01: Timing Observer Summary

**TimingObserver records per-stage and total pipeline wall-clock time with formatted percentage breakdown report and optional file output**

## Performance

- **Duration:** 5 min
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- TimingObserver class with on_event handler for all lifecycle events
- Formatted report with stage names, elapsed seconds, percentages, and total
- Optional file output on pipeline completion
- 6 unit tests covering capture, formatting, file output, and failure handling

## Task Commits

1. **Task 1+2: Implement TimingObserver + tests** - `c833fae`

## Files Created/Modified
- `src/aquapose/engine/timing.py` - TimingObserver class
- `tests/unit/engine/test_timing.py` - 6 unit tests
- `src/aquapose/engine/__init__.py` - Added TimingObserver export

## Decisions Made
None - followed plan as specified

## Deviations from Plan
None - plan executed exactly as written

## Issues Encountered
None

## Next Phase Readiness
- TimingObserver ready for CLI integration in Phase 18

---
*Phase: 17-observers*
*Completed: 2026-02-26*
