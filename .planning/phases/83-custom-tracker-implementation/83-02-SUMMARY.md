---
phase: 83-custom-tracker-implementation
plan: "02"
subsystem: tracking
tags: [keypoint-tracker, bidirectional-merge, gap-interpolation, chunk-handoff, config]
dependency_graph:
  requires: [83-01]
  provides: [KeypointTracker, merge_forward_backward, interpolate_gaps, TrackingConfig.keypoint_bidi]
  affects: [TrackingStage, TrackingConfig, tracking/__init__.py]
tech_stack:
  added: [scipy.interpolate.CubicSpline]
  patterns: [bidirectional-KF-tracker, Hungarian-OKS-merge, spline-gap-fill, chunk-handoff-JSON]
key_files:
  created: []
  modified:
    - src/aquapose/core/tracking/keypoint_tracker.py
    - src/aquapose/core/tracking/stage.py
    - src/aquapose/core/tracking/__init__.py
    - src/aquapose/engine/config.py
    - tests/unit/core/tracking/test_keypoint_tracker.py
decisions:
  - "KeypointTracker.get_tracklets() uses _collect_merged_builders() (returns raw _KptTrackletBuilder) so that gap interpolation can operate on mutable builders before conversion to Tracklet2D"
  - "TRACK-05 documented as satisfied by bidi design (no spatial-edge heuristic needed)"
  - "TRACK-10 explicitly deferred to Phase 84 per user decision"
  - "oks_sigmas not in TrackingConfig — loaded from DEFAULT_SIGMAS at construction time to avoid config/sigma coupling"
  - "TrackingConfig.__post_init__ validates tracker_kind strictly; rejects unknown values"
metrics:
  duration_seconds: 501
  completed_date: "2026-03-11"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 5
  tests_added: 28
  tests_total: 1179
---

# Phase 83 Plan 02: Bidirectional Merge, Gap Interpolation, Config Wiring Summary

**One-liner:** Bidirectional merge via Hungarian OKS assignment wrapping Plan 01's _SinglePassTracker, with cubic-spline gap filling, JSON chunk handoff, and TrackingStage dispatch via `tracker_kind='keypoint_bidi'`.

## What Was Built

### Task 1: Bidirectional merge, gap interpolation, KeypointTracker wrapper

Added to `src/aquapose/core/tracking/keypoint_tracker.py`:

**`merge_forward_backward()`** — Takes forward and backward builder lists, builds an (N_fwd x N_bwd) OKS cost matrix by computing mean OKS in temporal overlap regions, runs `scipy.optimize.linear_sum_assignment`, and merges matched pairs. Frame overlap resolution policy: detected > coasted; if both detected, higher mean keypoint confidence wins; if both coasted, keep forward. Unmatched builders kept if length >= `min_length`. Output track IDs are fresh monotonic sequence from 0.

**`interpolate_gaps()`** — Scans for gaps <= `max_gap_frames` between consecutive frames. For each fillable gap, builds `CubicSpline` per-keypoint per-axis and per-bbox dimension on all known frames, then inserts interpolated frames with status "coasted". Gaps exceeding `max_gap_frames` are left unfilled. Uses `strict=False` in zip for Python 3.12 compatibility.

**`KeypointTracker`** — Public bidirectional wrapper implementing the same interface as `OcSortTracker` (`update`, `get_tracklets`, `get_state`, `from_state`). Stores detection frames for backward pass replay. `get_tracklets()` triggers backward pass then calls `_collect_merged_builders()` (internal helper returning raw builders for mutable gap interpolation), applies `interpolate_gaps()` per builder, then assigns fresh monotonic track IDs. `get_state()` returns JSON-safe dict; `from_state()` reconstructs forward tracker from saved KF states.

**`_collect_merged_builders()`** — Internal helper identical in logic to `merge_forward_backward` but returns `_KptTrackletBuilder` objects instead of `Tracklet2D`, enabling mutable gap interpolation before Tracklet2D conversion.

### Task 2: Config extension and TrackingStage wiring

**`TrackingConfig`** extended with three new backward-compatible fields:
- `base_r: float = 10.0` — KF measurement noise base
- `lambda_ocm: float = 0.2` — OCM weight in cost function
- `max_gap_frames: int = 5` — max gap size for spline interpolation

Added `__post_init__` validation raising `ValueError` for unknown `tracker_kind` values. Valid kinds: `{"ocsort", "keypoint_bidi"}`.

**`TrackingStage.run()`** now dispatches by `tracker_kind`:
- `"keypoint_bidi"` — imports and constructs `KeypointTracker` with all new config fields; `from_state()` path also dispatches correctly
- `"ocsort"` — existing `OcSortTracker` path unchanged

**`tracking/__init__.py`** updated to export `KeypointTracker`, `interpolate_gaps`, `merge_forward_backward` in `__all__`.

## TRACK Requirements Status

| Req | Status | Notes |
|-----|--------|-------|
| TRACK-01 | Implemented (Plan 01) | 24-dim KF |
| TRACK-02 | Implemented (Plan 01) | OKS cost matrix |
| TRACK-03 | Implemented (Plan 01) | OCM direction consistency |
| TRACK-04 | Implemented (Plan 01) | ORU recovery |
| TRACK-05 | Satisfied by design | Bidi merge — forward catches entries, backward catches exits |
| TRACK-06 | Implemented (Plan 01) | OCR history scan |
| TRACK-07 | Implemented this plan | Gap interpolation via CubicSpline |
| TRACK-08 | Implemented this plan | Chunk handoff: KF state JSON serialization |
| TRACK-09 | Implemented this plan | `tracker_kind='keypoint_bidi'` in config + stage |
| TRACK-10 | Deferred to Phase 84 | BYTE-style secondary pass; single threshold only |

## Deviations from Plan

None — plan executed exactly as written.

## Verification

- `hatch run test`: 1179 passed, 3 skipped, 14 deselected (zero regressions)
- `hatch run lint`: All checks passed
- `hatch run typecheck`: No new errors in modified files (25 pre-existing errors in unrelated files)

## Self-Check: PASSED

- FOUND: `.planning/phases/83-custom-tracker-implementation/83-02-SUMMARY.md`
- FOUND: `src/aquapose/core/tracking/keypoint_tracker.py`
- FOUND: `src/aquapose/core/tracking/stage.py`
- FOUND: `src/aquapose/engine/config.py`
- FOUND: commit 5e2007e (feat(83-02): bidirectional merge, gap interpolation, KeypointTracker wrapper)
- FOUND: commit 2f8b4a2 (feat(83-02): extend TrackingConfig and wire KeypointTracker into TrackingStage)
