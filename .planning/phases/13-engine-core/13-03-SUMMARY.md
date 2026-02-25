---
phase: 13-engine-core
plan: "03"
subsystem: engine
tags: [events, observers, eventbus, protocol, dataclasses, synchronous-dispatch]

# Dependency graph
requires:
  - phase: 13-engine-core (plan 01)
    provides: engine package __init__.py needed to export event symbols
provides:
  - Typed event dataclasses in 3-tier taxonomy (pipeline, stage, frame lifecycle)
  - Observer Protocol with structural typing (no inheritance required)
  - EventBus with typed subscription, MRO-aware synchronous dispatch, and fault tolerance
  - 9 unit tests covering all event/observer behaviors
affects: [13-engine-core plan 04 (PosePipeline orchestrator emits these events)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Frozen dataclasses as event types — all event fields immutable after construction"
    - "typing.Protocol with @runtime_checkable for structural Observer typing"
    - "MRO-aware EventBus dispatch — subscribing to base Event type receives all subclass events"
    - "Fault-tolerant observer dispatch — logging.warning on exception, delivery continues"

key-files:
  created:
    - src/aquapose/engine/events.py
    - src/aquapose/engine/observers.py
    - tests/unit/engine/test_events.py
  modified:
    - src/aquapose/engine/__init__.py

key-decisions:
  - "Observer uses structural typing (Protocol) not ABC inheritance — any class with on_event satisfies it"
  - "EventBus walks __mro__ for dispatch — subscribing to Event base receives all subtypes without special handling"
  - "Fault-tolerant dispatch logs warnings but never re-raises — pipeline determinism preserved"
  - "Events use frozen=True dataclasses — immutability prevents observers from accidentally modifying event state"
  - "Synchronous delivery — pipeline blocks on each observer call to guarantee determinism"

patterns-established:
  - "Event taxonomy pattern: base Event class + typed subclasses grouped by tier (pipeline/stage/frame)"
  - "EventBus.emit() iterates event_type.__mro__ to deliver to both exact-type and base-type subscribers"

requirements-completed: [ENG-03, ENG-04]

# Metrics
duration: 5min
completed: "2026-02-25"
---

# Phase 13 Plan 03: Event System and Observer Protocol Summary

**Typed event dataclasses (6 events, 3-tier taxonomy) with Observer Protocol and MRO-aware EventBus providing synchronous, fault-tolerant dispatch**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-25T21:05:04Z
- **Completed:** 2026-02-25T21:10:47Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Implemented 6 frozen event dataclasses covering pipeline lifecycle (PipelineStart, PipelineComplete, PipelineFailed), stage lifecycle (StageStart, StageComplete), and frame-level (FrameProcessed) — all with auto-populated timestamp fields
- Implemented Observer as a `@runtime_checkable` Protocol and EventBus with typed subscription, MRO-aware synchronous dispatch (subscribing to Event base receives all subtypes), and fault-tolerant error handling
- Wrote 9 unit tests covering frozen immutability, timestamp auto-population, structural typing, filtered dispatch, synchronous ordering, base-type subscription, fault tolerance, and unsubscribe behavior — all passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Create typed event dataclasses and Observer protocol with EventBus** - `14245bf` (feat) — Note: committed by previous agent session as part of 13-02's batch commit
2. **Task 2: Write tests for event creation, subscription, synchronous dispatch, and fault tolerance** - `a17f3ed` (test)

**Plan metadata:** TBD (docs: complete plan)

_Note: events.py and observers.py were committed in the prior agent session's 13-02 commit (`14245bf`). test_events.py was committed fresh in this session._

## Files Created/Modified

- `src/aquapose/engine/events.py` — 6 frozen event dataclasses with 3-tier taxonomy and auto-timestamp
- `src/aquapose/engine/observers.py` — Observer Protocol and EventBus with MRO-aware dispatch and fault tolerance
- `src/aquapose/engine/__init__.py` — Updated to export Event, EventBus, Observer, and all 6 event classes
- `tests/unit/engine/test_events.py` — 9 unit tests covering all event/observer behaviors

## Decisions Made

- EventBus dispatch uses `event_type.__mro__` iteration to deliver to both exact-type and ancestor-type subscribers. This enables the "subscribe to Event base to receive everything" pattern without special-casing.
- Observer fault tolerance uses `logging.warning(..., exc_info=True)` and continues delivery — pipeline determinism is preserved even if one observer misbehaves.
- Events use `frozen=True` to prevent observers from accidentally mutating shared event state between sequential observer calls.

## Deviations from Plan

None — plan executed exactly as written. Events and observers were already committed (by a previous session executing plans 01/02), so Task 1 verification passed immediately. Task 2 (test file) was the primary work of this session.

## Issues Encountered

- Pre-existing test failure in `tests/unit/tracking/test_tracker.py::test_near_claim_penalty_suppresses_ghost` — unrelated to plan 03, deferred per scope boundary rules. This failure existed before this plan and is not caused by any engine changes.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Event system complete and tested — plan 13-04 (PosePipeline orchestrator) can import and emit all 6 event types
- EventBus is ready to receive observer registrations from the orchestrator
- No blockers

---
*Phase: 13-engine-core*
*Completed: 2026-02-25*
