---
phase: 65-frame-selection-and-dataset-assembly
verified: 2026-03-05T20:55:00Z
status: passed
score: 10/10 must-haves verified
re_verification:
  previous_status: gaps_found
  previous_score: 8/10
  gaps_closed:
    - "Frame selection filters pseudo-labels to selected frames during assembly"
    - "Pseudo-label val metadata sidecar includes gap_reason per image"
  gaps_remaining: []
  regressions: []
---

# Phase 65: Frame Selection and Dataset Assembly Verification Report

**Phase Goal:** Users can build a training dataset from manual annotations plus filtered pseudo-labels with controlled diversity and validation splits
**Verified:** 2026-03-05T20:55:00Z
**Status:** passed
**Re-verification:** Yes -- after gap closure (plan 65-03)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Temporal subsampling selects every Kth frame | VERIFIED | `temporal_subsample` in frame_selection.py (148 lines); 14 tests pass |
| 2 | Frames with zero reconstructions removed before diversity sampling | VERIFIED | `filter_empty_frames` checks midlines_3d non-empty; tests pass |
| 3 | Diversity sampling clusters curvatures and samples per bin | VERIFIED | `diversity_sample` uses scipy kmeans2; tests pass |
| 4 | Per-fish curvature computed from 3D spline control points | VERIFIED | `compute_curvature` finite differences on control_points; tests pass |
| 5 | Pool manual + Source A + Source B with independent confidence thresholds | VERIFIED | `assemble_dataset` filters consensus/gap at independent thresholds; `test_gap_threshold_independent` confirms |
| 6 | Manual annotations always included in full (bypass confidence filtering) | VERIFIED | Manual data never passes through `filter_by_confidence`; `test_manual_bypasses_confidence_filter` confirms |
| 7 | Assembled dataset has separate manual val set and pseudo-label val set | VERIFIED | Manual val -> images/val, pseudo val tracked via sidecar in images/train |
| 8 | Manual val set is official val in dataset.yaml | VERIFIED | `dataset.yaml` writes `val: images/val` pointing to manual val; test confirms |
| 9 | Pseudo-label val has JSON metadata sidecar recording source, gap_reason, confidence | VERIFIED | `_extract_dominant_gap_reason` (line 256) extracts gap_reason from label metadata; `test_gap_reason_in_pseudo_val_metadata` and `test_pseudo_val_metadata_sidecar` both verify gap_reason presence |
| 10 | Frame selection filters pseudo-labels to selected frames during assembly | VERIFIED | `_filter_by_frames` (line 229) filters by run_id and frame index; CLI builds `selected_frames` dict (line 793-822) and passes to `assemble_dataset` (line 835); `test_selected_frames_filters_pseudo_labels` and `test_selected_frames_none_includes_all` confirm |

**Score:** 10/10 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/training/frame_selection.py` | Frame selection functions | VERIFIED | 148 lines, 4 public functions |
| `tests/unit/training/test_frame_selection.py` | Unit tests (min 80 lines) | VERIFIED | 179 lines, 14 tests, all pass |
| `src/aquapose/training/dataset_assembly.py` | Dataset assembly with selected_frames param and gap_reason | VERIFIED | 444 lines, `selected_frames` param (line 296), `_filter_by_frames` (line 229), `_extract_dominant_gap_reason` (line 256) |
| `tests/unit/training/test_dataset_assembly.py` | Unit tests including frame filtering and gap_reason | VERIFIED | 623 lines, includes `test_selected_frames_filters_pseudo_labels`, `test_selected_frames_none_includes_all`, `test_gap_reason_in_pseudo_val_metadata` |
| `src/aquapose/training/pseudo_label_cli.py` | CLI assemble command with frame selection wiring | VERIFIED | `selected_frames` dict built (line 793-822), passed to `assemble_dataset` (line 835), no TODO comments |
| `src/aquapose/training/__init__.py` | Exports for all new functions | VERIFIED | `assemble_dataset`, `compute_curvature`, `diversity_sample` in `__all__` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `frame_selection.py` | `Midline3D.control_points` | curvature computation | WIRED | `midline.control_points` accessed in compute_curvature |
| `dataset_assembly.py` | `confidence.json` | reads from run directories | WIRED | `json.loads(conf_path.read_text())` in collect_pseudo_labels |
| `pseudo_label_cli.py` | `dataset_assembly.py` | CLI calls assemble_dataset() | WIRED | Import and call on lines 824-836 |
| `pseudo_label_cli.py` | `frame_selection.py` | CLI uses frame selection | WIRED | Functions imported (line 784-788), results stored in `selected_frames` dict (line 822), passed to assemble_dataset (line 835) |
| `dataset_assembly.py` | `_filter_by_frames` | frame-level filtering | WIRED | Called on lines 377-378 when `selected_frames is not None` |
| `dataset_assembly.py` | `_extract_dominant_gap_reason` | gap_reason in sidecar | WIRED | Called on line 409, output written to pseudo_val_metadata.json (line 432) |
| `dataset_assembly.py` | `dataset.yaml` | writes YOLO config | WIRED | YAML written on lines 425-428 with correct val path |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FRAME-01 | 65-01 | Temporal subsampling (every Kth frame) | SATISFIED | `temporal_subsample` implemented and tested |
| FRAME-02 | 65-01 | Zero-reconstruction frames filtered | SATISFIED | `filter_empty_frames` implemented and tested |
| FRAME-03 | 65-01 | Pose-diversity sampling via curvature clustering | SATISFIED | `diversity_sample` with kmeans2 implemented and tested |
| DATA-01 | 65-02, 65-03 | Pool manual + Source A + Source B with independent thresholds, frame filtering | SATISFIED | Assembly works with independent thresholds; frame selection wired via selected_frames parameter |
| DATA-02 | 65-02 | Separate manual val + pseudo-label val splits | SATISFIED | Manual val in images/val, pseudo val tracked via sidecar |
| DATA-03 | 65-02, 65-03 | Pseudo-label val broken down by source and gap reason | SATISFIED | Sidecar includes source, confidence, run_id, and gap_reason fields |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No TODO/FIXME/PLACEHOLDER comments found in any modified files |

### Human Verification Required

### 1. End-to-end CLI invocation with real data

**Test:** Run `aquapose pseudo-label assemble --run-dir <real_run> --manual-dir <manual> --output-dir <out> --model-type obb --temporal-step 5 --diversity-max-per-bin 10`
**Expected:** Output directory contains YOLO-standard structure with filtered pseudo-labels (fewer than without frame selection), dataset.yaml is valid, pseudo_val_metadata.json includes gap_reason entries
**Why human:** Requires real pipeline run directories with pseudo-labels, manual annotations, and diagnostic caches

### Gaps Summary

No gaps. All 10 observable truths verified. Both previously-identified gaps have been closed by plan 65-03:

1. **Frame selection wired to assembly (CLOSED):** `_filter_by_frames` added to dataset_assembly.py, CLI builds `selected_frames` dict and passes it to `assemble_dataset`. The `--temporal-step` and `--diversity-max-per-bin` flags now produce filtered datasets.

2. **gap_reason in sidecar (CLOSED):** `_extract_dominant_gap_reason` extracts the most common gap_reason from per-fish label metadata and writes it to `pseudo_val_metadata.json`. Both new and updated tests verify its presence.

All 961 tests pass with no regressions. No anti-patterns detected.

---

_Verified: 2026-03-05T20:55:00Z_
_Verifier: Claude (gsd-verifier)_
