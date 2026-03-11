# Phase 80: Baseline Metrics - Research

**Researched:** 2026-03-10
**Domain:** 2D tracking evaluation, standalone script, metric extraction
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Metric scope
- Use existing `TrackingMetrics` (track count, length stats, coast frequency) for 2D tracking
- Adapt `FragmentationMetrics` gap/continuity analysis for 2D tracklets (gaps, births, deaths, continuity ratio within the single camera)
- Global aggregates only — no per-camera breakdown (there's only one camera)
- No new metrics (no ID switches, MOTA, etc.) — existing evaluators cover the success criteria

#### Execution method
- Standalone script in `scripts/` (similar pattern to `investigate_occlusion.py`)
- Use Phase 78.1 production models (OBB + pose) — matches what the rest of the milestone will use
- Single run, no variance analysis
- Script outputs both metrics (to console/file) and an annotated video with track IDs overlaid on each detection

#### Document format
- Baseline metrics document at `.planning/phases/80-baseline-metrics/80-BASELINE.md`
- Structured metrics table followed by gap analysis section stating delta to 9-track zero-fragmentation target
- Text and numbers only — no embedded screenshots (annotated video exists separately)

#### Target definition
- 9 fish are visible in e3v83eb during frames 3300-4500
- Qualitative target: "9 tracks with zero fragmentation" — no numeric thresholds defined
- No minimum improvement thresholds for Phase 84; Phase 84 compares freely against the baseline numbers
- Phase 84 success criteria already require "measurable improvement on at least one primary metric"

### Claude's Discretion
- Exact script structure and argument handling
- How to adapt FragmentationMetrics gap analysis for 2D tracklets (may reuse logic or write new)
- Video annotation style (colors, font, overlay positioning)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INV-03 | Baseline tracking metrics (track count, duration distribution, fragmentation, coverage) measured on perfect-tracking target clip with current OC-SORT | Script runs OC-SORT on e3v83eb frames 3300-4500, feeds Tracklet2D output into `evaluate_tracking()` + adapted fragmentation logic; 80-BASELINE.md records all required metrics and gap-to-target |
</phase_requirements>

## Summary

Phase 80 is a measurement-and-documentation phase: write a standalone script that runs the current OC-SORT tracker on one camera's target clip (e3v83eb, frames 3300-4500), extract metrics using existing evaluators, produce an annotated video, and record results in 80-BASELINE.md as a permanent reference for Phase 84 to compare against.

The existing codebase already provides all required building blocks. `evaluate_tracking()` in `evaluation/stages/tracking.py` consumes a `list[Tracklet2D]` and returns a frozen `TrackingMetrics` covering track count, length stats, and coast frequency. `evaluate_fragmentation()` in `evaluation/stages/fragmentation.py` provides gap counts, continuity ratios, births/deaths, and lifespan stats — but it currently operates on 3D `Midline3D` lists. For 2D use, the gap/continuity logic must be adapted to consume `Tracklet2D` objects directly. The structural template for the script is `scripts/investigate_occlusion.py`, which already handles config loading, OC-SORT initialization, frame-by-frame detection+tracking, and annotated video output.

**Primary recommendation:** Write `scripts/measure_baseline_tracking.py` modeled on `investigate_occlusion.py`. Reuse `evaluate_tracking()` unchanged. Add a `evaluate_fragmentation_2d()` function (either inline in the script or as a thin wrapper) that converts `Tracklet2D.frames` to the sparse-frame-index representation `evaluate_fragmentation()` already uses internally — avoiding code duplication by just calling the 3D function's logic directly with the 2D frame sets. Record results in 80-BASELINE.md.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `aquapose.core.tracking.ocsort_wrapper.OcSortTracker` | project | Per-camera OC-SORT tracker | Only tracked tracker in project; already used in `investigate_occlusion.py` |
| `aquapose.evaluation.stages.tracking.evaluate_tracking` | project | Computes `TrackingMetrics` from `Tracklet2D` list | Frozen dataclass, pure function, exact match for required metrics |
| `aquapose.evaluation.stages.fragmentation.evaluate_fragmentation` / adapted 2D version | project | Gap/continuity analysis | Provides all required fragmentation metrics; needs minor adaptation for 2D |
| `ultralytics.YOLO` | installed | OBB detection | Production model interface |
| `cv2` (OpenCV) | installed | Video I/O and annotation | Already used throughout |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `yaml` | stdlib | Config loading | Loading `config.yaml` |
| `json` | stdlib | Metrics JSON output | Saving metrics alongside video |
| `numpy` | installed | Array math for metrics | Aggregating frame stats |

## Architecture Patterns

### Recommended Project Structure
```
scripts/
├── investigate_occlusion.py    # existing template
└── measure_baseline_tracking.py  # new script (Phase 80)

.planning/phases/80-baseline-metrics/
├── 80-CONTEXT.md               # existing
├── 80-RESEARCH.md              # this file
└── 80-BASELINE.md              # output: metrics document (written manually after script runs)
```

### Pattern 1: Config-Driven Script (from `investigate_occlusion.py`)
**What:** `_load_config(config_path)` reads `config.yaml`, resolves relative paths to absolute, and returns a flat dict with `_obb_weights`, `_pose_weights`, `_video_dir`. The script consumes this dict throughout.
**When to use:** Whenever a standalone script needs to reference project models or data.
**Example:** `scripts/investigate_occlusion.py` lines 58-90 — copy this pattern verbatim.

### Pattern 2: OcSortTracker Initialization
**What:** `OcSortTracker(camera_id=camera_id, min_hits=1, det_thresh=0.1)` creates a fresh tracker per run. For baseline measurement, `min_hits=1` is appropriate (count all tracks that got even one match); callers of the production `TrackingStage` use configurable params.
**When to use:** `investigate_occlusion.py` uses `min_hits=1` — follow the same choice for baseline to avoid filtering too-short tracks.
**Note:** `max_age` defaults to 30; this controls how long OC-SORT coasts before dropping a track.

### Pattern 3: `evaluate_tracking()` Usage
**What:** After calling `tracker.get_tracklets()`, pass the returned `list[Tracklet2D]` directly to `evaluate_tracking(tracklets)`. Returns a frozen `TrackingMetrics` dataclass. Call `.to_dict()` for JSON serialization.
**When to use:** Always — this is the canonical evaluator; don't reimplement.

### Pattern 4: Adapting `evaluate_fragmentation` for 2D Tracklets
**What:** `evaluate_fragmentation()` currently expects `list[dict[int, Midline3D] | None]` and uses `m3d.frame_index` to build `fish_id -> set[frame_index]`. The same logic works for `Tracklet2D` — each tracklet's `.frames` tuple already is the set of frame indices, and `.track_id` is the fish_id analog.

Two implementation options:
1. **Inline adapter in script** — build a `fish_frames: dict[int, set[int]]` directly from `Tracklet2D.frames` and call the gap-analysis logic manually (< 20 lines). This keeps the script self-contained.
2. **New `evaluate_fragmentation_2d(tracklets, n_animals)` function** in `evaluation/stages/fragmentation.py` — cleaner, testable, matches project style.

The CONTEXT.md says "may reuse logic or write new" — recommend **Option 2** (new public function in `fragmentation.py`) to keep the script thin and get a testable unit. The function body is essentially the same algorithm but takes `list[Tracklet2D]` instead of `list[dict | None]`.

### Pattern 5: Annotated Video Output
**What:** Identical to `investigate_occlusion.py` Pass 2 render loop (lines 467-518). Color each detection by `track_id` using `_PALETTE_BGR`, overlay `track_id` text near the OBB center or corner, write each frame with `cv2.VideoWriter`.
**When to use:** Drop the pose-keypoint overlay from `investigate_occlusion.py`; this script focuses on tracking identity, so OBBs + track ID text labels are sufficient.

### Anti-Patterns to Avoid
- **Importing from `aquapose.engine`** — evaluation stages must not import from the engine. Verification tests (`test_stage_tracking.py`, `test_stage_fragmentation.py`) enforce this with AST checks.
- **Re-implementing metric logic** — `evaluate_tracking()` already computes everything in TrackingMetrics. Do not duplicate.
- **Using `min_hits` > 1 without understanding the effect** — `min_hits=3` (the production default) will suppress tracks that only appear for 1-2 frames. For baseline measurement, `min_hits=1` captures all tracks the tracker ever confirmed, giving a more honest fragmentation count.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Track count + length stats + coast frequency | Custom aggregation loop | `evaluate_tracking(tracklets)` | Already tested, frozen, correct |
| Gap detection within a track | Manual consecutive-diff logic | Logic pattern from `evaluate_fragmentation()` lines 142-146 | Handles edge cases (single-frame tracks, sparse indices) |
| Config path resolution | Custom path handling | `_load_config()` from `investigate_occlusion.py` | Handles relative/absolute, project_dir expansion |
| Video I/O | Custom reader/writer | `cv2.VideoCapture` / `cv2.VideoWriter` pattern from `investigate_occlusion.py` | Established working pattern |

**Key insight:** The only genuinely new code in this phase is (a) the thin adapter from Tracklet2D to fragmentation input and (b) the track-ID text overlay in the video. Everything else is existing code connected together.

## Common Pitfalls

### Pitfall 1: `from_state` Clears Builder History
**What goes wrong:** `OcSortTracker.from_state()` intentionally clears `_builders` (lines 326-329 of `ocsort_wrapper.py`). If the script calls `from_state` expecting to accumulate across chunks, the builders from the prior chunk are gone.
**Why it happens:** Cross-chunk design: each chunk has local frame indices 0..N-1.
**How to avoid:** For this script, run the entire 1200-frame range in a single `OcSortTracker` instance with one `update()` call per frame. No chunking needed.

### Pitfall 2: Local vs Global Frame Indices
**What goes wrong:** `OcSortTracker.update(frame_idx, ...)` stores whatever `frame_idx` you pass into `Tracklet2D.frames`. If you call `update(0, ...)`, `update(1, ...)`, etc. from `start_frame=3300`, the resulting Tracklet2D will have frames [0, 1, 2, ...] not [3300, 3301, ...].
**Why it happens:** Tracker is frame-index-agnostic; it records whatever index is passed.
**How to avoid:** Pass the absolute frame index: `tracker.update(fidx, dets)` where `fidx = start_frame + loop_i`. This matches what `investigate_occlusion.py` already does.

### Pitfall 3: `evaluate_fragmentation` Expects `Midline3D` with `.frame_index`
**What goes wrong:** Calling `evaluate_fragmentation()` directly with Tracklet2D objects will fail — it expects `m3d.frame_index` attribute.
**Why it happens:** Function is typed for 3D reconstruction output.
**How to avoid:** Either write the `evaluate_fragmentation_2d()` adapter (Option 2 above) or build the `fish_frames` dict manually from `Tracklet2D.frames` and run the gap analysis inline.

### Pitfall 4: OBB Corner Order (CLAUDE.md warning)
**What goes wrong:** `obb.xyxyxyxy` returns `[RB, RT, LT, LB]` not `[TL, TR, BR, BL]`. Affine crop extraction must use `src = [pts[2], pts[1], pts[3]]` (TL, TR, BL).
**Why it happens:** Ultralytics OBB output convention differs from training data.
**How to avoid:** This script does NOT need pose estimation (tracking-only). If pose is omitted, the crop extraction code is unnecessary and this pitfall is avoided.

### Pitfall 5: Coast Count Includes Gaps vs. OC-SORT "Coast" Semantics
**What goes wrong:** `TrackingMetrics.coast_frequency` counts frames where status is "coasted" — this is the Kalman-predicted frames while the track is alive. This is different from fragmentation gaps (periods where the track had no output at all).
**Why it happens:** OC-SORT keeps coasting tracks active (they appear in Tracklet2D with "coasted" status). If a track is permanently lost, it will not have "coasted" entries — it simply ends.
**How to avoid:** Report both metrics clearly in 80-BASELINE.md. Coast frequency measures within-track interpolation quality; fragmentation gaps (from the adapted evaluate_fragmentation_2d) measure track splits — these answer different questions.

### Pitfall 6: `min_hits=1` vs Production `min_hits=3`
**What goes wrong:** Using `min_hits=3` (the production default) may suppress brief false-positive tracks but also misses genuine 1-2 frame fragments, artificially making the baseline look better.
**Why it happens:** `investigate_occlusion.py` uses `min_hits=1`; production config uses `n_init` which defaults to 3.
**How to avoid:** Use `min_hits=1` for baseline measurement and document this choice in 80-BASELINE.md. The Phase 84 comparison should use the same value.

## Code Examples

### evaluate_tracking usage
```python
# Source: src/aquapose/evaluation/stages/tracking.py
from aquapose.evaluation.stages.tracking import evaluate_tracking

tracklets = tracker.get_tracklets()
metrics = evaluate_tracking(tracklets)
print(f"Track count: {metrics.track_count}")
print(f"Length median: {metrics.length_median:.1f} frames")
print(f"Coast frequency: {metrics.coast_frequency:.3f}")
print(f"Detection coverage: {metrics.detection_coverage:.3f}")
```

### Tracklet2D → fragmentation adapter pattern
```python
# Build fish_frames from Tracklet2D objects (same logic evaluate_fragmentation uses internally)
fish_frames: dict[int, set[int]] = {}
for t in tracklets:
    fish_frames[t.track_id] = set(t.frames)

# Then run gap analysis — see evaluate_fragmentation() lines 125-158 for the loop
```

### OcSortTracker initialization pattern (from investigate_occlusion.py)
```python
from aquapose.core.tracking.ocsort_wrapper import OcSortTracker

tracker = OcSortTracker(camera_id=camera_id, min_hits=1, det_thresh=0.1)
for fidx in range(start_frame, end_frame):
    ret, frame = cap.read()
    # ... detect ...
    tracker.update(fidx, det_objects)

tracklets = tracker.get_tracklets()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| n/a — new phase | New standalone script | Phase 80 | First quantitative OC-SORT baseline |

## Open Questions

1. **Should `evaluate_fragmentation_2d` be added as a public function in `fragmentation.py` or stay in the script?**
   - What we know: CONTEXT says "may reuse logic or write new" — Claude's discretion
   - What's unclear: Whether Phase 84 will want to call the same function for comparison
   - Recommendation: Add `evaluate_fragmentation_2d(tracklets, n_animals)` to `fragmentation.py` as a public function. Phase 84 will need the same metric computation to compare fairly, so a shared, tested function is clearly better than script-local logic.

2. **Does the script need to also run pose estimation?**
   - What we know: CONTEXT says "annotated video with track IDs overlaid on each detection" — no mention of pose overlay
   - What's unclear: Whether tracking-only (no pose) is sufficient for the video
   - Recommendation: Skip pose estimation. The phase goal is to measure tracking metrics, not pose quality. Simpler = faster run, less complexity.

## Validation Architecture

`workflow.nyquist_validation` is not present in config.json — treating as disabled. Skipping Validation Architecture section.

> Note: `config.json` has no `nyquist_validation` field; per instructions, skip this section.

## Sources

### Primary (HIGH confidence)
- `src/aquapose/evaluation/stages/tracking.py` — exact `TrackingMetrics` fields and `evaluate_tracking()` signature
- `src/aquapose/evaluation/stages/fragmentation.py` — exact `FragmentationMetrics` fields, `evaluate_fragmentation()` logic (lines 93-172)
- `src/aquapose/core/tracking/ocsort_wrapper.py` — `OcSortTracker` constructor params, `update()`, `get_tracklets()`, `from_state()` semantics
- `src/aquapose/core/tracking/types.py` — `Tracklet2D` dataclass fields
- `scripts/investigate_occlusion.py` — complete reference implementation for script structure, config loading, OcSortTracker usage, video annotation
- `~/aquapose/projects/YH/config.yaml` — production model paths confirmed (OBB: `run_20260310_115419`, Pose: `run_20260310_171543`)
- `~/aquapose/projects/YH/videos/core_videos/e3v83eb-20260218T145915-150429.mp4` — target video confirmed present

### Secondary (MEDIUM confidence)
- `tests/unit/evaluation/test_stage_tracking.py` — test patterns for `evaluate_tracking()` (verified behavior and edge cases)
- `tests/unit/evaluation/test_stage_fragmentation.py` — test patterns for `evaluate_fragmentation()` (frozen dataclass, JSON serialization requirements)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries directly read from project source
- Architecture: HIGH — script template (`investigate_occlusion.py`) is complete and directly applicable
- Pitfalls: HIGH — identified from direct code inspection, not inference

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (stable codebase, no external dependencies)
