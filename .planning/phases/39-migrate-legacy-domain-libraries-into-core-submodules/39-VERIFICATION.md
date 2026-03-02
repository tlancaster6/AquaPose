---
phase: 39-migrate-legacy-domain-libraries-into-core-submodules
verified: 2026-03-02T19:00:00Z
status: passed
score: 12/12 must-haves verified
re_verification: false
---

# Phase 39: Migrate Legacy Domain Libraries into Core Submodules — Verification Report

**Phase Goal:** Legacy top-level domain libraries (reconstruction/, segmentation/, tracking/) are reorganized into core/ submodules alongside the stages that consume them, eliminating cross-package private-helper imports and misleading directory names. Stale docstrings and GUIDEBOOK.md updated to match new paths.
**Verified:** 2026-03-02T19:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | `core/types/` package exists with Detection, CropRegion, AffineCrop, Midline2D, Midline3D, MidlineSet types | VERIFIED | `ls src/aquapose/core/types/` shows all 5 files; `from aquapose.core.types import Detection, CropRegion, AffineCrop, Midline2D, Midline3D, MidlineSet` succeeds |
| 2  | Implementation files exist at new core locations with correct internal imports | VERIFIED | All 5 files present at `core/midline/midline.py`, `core/midline/crop.py`, `core/reconstruction/triangulation.py`, `core/reconstruction/curve_optimizer.py`, `core/tracking/ocsort_wrapper.py`; each imports types from `core/types/` |
| 3  | YOLODetector and make_detector() are defined in core/detection/backends/yolo.py | VERIFIED | grep confirms `class YOLODetector`, `def make_detector`, `class YOLOBackend` at lines 21, 89, 111; `__all__` exports all three |
| 4  | All src/ imports point to core/types/ or core/<stage>/ paths — zero references to legacy packages | VERIFIED | `grep -rn "from aquapose.reconstruction.\|from aquapose.segmentation.\|from aquapose.tracking." src/aquapose/ --include="*.py"` returns 0 matches |
| 5  | Legacy directories reconstruction/, segmentation/, tracking/ are deleted entirely | VERIFIED | `ls src/aquapose/reconstruction/` and siblings all return "No such file or directory" |
| 6  | Re-export shim files core/detection/types.py and core/reconstruction/types.py are deleted | VERIFIED | Both files absent; `ls` returns nothing |
| 7  | core/midline/types.py retains only AnnotatedDetection with imports from core/types/ | VERIFIED | File imports `Detection`, `Midline2D`, `CropRegion` from `core/types/`; `__all__ = ["AnnotatedDetection"]` |
| 8  | All test files import from core/types/ or core/<stage>/ paths — zero legacy references | VERIFIED | `grep -rn "from aquapose.reconstruction.\|from aquapose.segmentation.\|from aquapose.tracking." tests/ --include="*.py"` returns 0 matches |
| 9  | hatch run test passes with all tests green | VERIFIED | 656 passed, 3 skipped, 31 deselected, 0 failures |
| 10 | GUIDEBOOK.md source layout section reflects actual directory structure | VERIFIED | Section 4 shows `types/` under `core/`; no legacy `reconstruction/`, `segmentation/`, `tracking/` top-level lines; `core/<stage>/` descriptions include actual module names |
| 11 | CLAUDE.md Architecture section shows correct directory tree | VERIFIED | `core/` subtree shows `types/`, `detection/`, `midline/`, `reconstruction/`, `tracking/`, `association/`; legacy top-level dirs absent |
| 12 | No module-level docstring references legacy paths, U-Net, no-op stubs, or Phase 37 pending status | VERIFIED | grep across all moved files (`core/midline/midline.py`, `core/midline/crop.py`, `core/reconstruction/triangulation.py`, `core/reconstruction/curve_optimizer.py`, `core/tracking/ocsort_wrapper.py`, `core/detection/backends/yolo.py`) returns 0 matches for stale references |

**Score:** 12/12 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/aquapose/core/types/__init__.py` | Re-exports all public cross-stage types | VERIFIED | Exports 6 types; imports from 4 sub-modules |
| `src/aquapose/core/types/detection.py` | Detection dataclass | VERIFIED | File present; only stdlib + numpy imports |
| `src/aquapose/core/types/crop.py` | CropRegion and AffineCrop types | VERIFIED | File present; only stdlib + numpy imports |
| `src/aquapose/core/types/midline.py` | Midline2D type | VERIFIED | File present; only stdlib + numpy imports |
| `src/aquapose/core/types/reconstruction.py` | Midline3D and MidlineSet types | VERIFIED | Imports Midline2D from `core/types/midline` (same layer, not implementation) — deliberate design decision documented in 39-01-SUMMARY |
| `src/aquapose/core/midline/midline.py` | MidlineExtractor + private helpers | VERIFIED | MidlineExtractor present; imports CropRegion and Midline2D from core/types/ |
| `src/aquapose/core/midline/crop.py` | extract_affine_crop, invert_affine_point, invert_affine_points | VERIFIED | File present; imports types from core/types/crop |
| `src/aquapose/core/reconstruction/triangulation.py` | triangulate_midlines() + helpers + constants | VERIFIED | File present; imports Midline2D, Midline3D, MidlineSet from core/types/ |
| `src/aquapose/core/reconstruction/curve_optimizer.py` | CurveOptimizer + CurveOptimizerConfig + OptimizerSnapshot | VERIFIED | File present; imports from core/reconstruction/triangulation |
| `src/aquapose/core/tracking/ocsort_wrapper.py` | OcSortTracker | VERIFIED | File present; docstring comment references core/types/detection |
| `src/aquapose/core/detection/backends/yolo.py` | YOLODetector and make_detector() | VERIFIED | Class YOLODetector, def make_detector, class YOLOBackend present; __all__ correct |
| `src/aquapose/core/midline/types.py` | AnnotatedDetection only (not a shim) | VERIFIED | Contains only AnnotatedDetection; imports from core/types/ not legacy paths |
| `src/aquapose/reconstruction/` | DELETED | VERIFIED | Directory absent |
| `src/aquapose/segmentation/` | DELETED | VERIFIED | Directory absent |
| `src/aquapose/tracking/` | DELETED | VERIFIED | Directory absent |
| `src/aquapose/core/detection/types.py` | DELETED | VERIFIED | File absent |
| `src/aquapose/core/reconstruction/types.py` | DELETED | VERIFIED | File absent |
| `.planning/GUIDEBOOK.md` | Updated source layout reflecting post-migration structure; contains core/types/ | VERIFIED | Section 4 has `types/` under core/; no legacy top-level dirs |
| `CLAUDE.md` | Updated Architecture section | VERIFIED | Shows full core/ subtree with types/ present |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `core/types/` | stdlib + numpy only | types are base layer | VERIFIED | No implementation imports in any core/types/ file (except reconstruction.py importing from core/types/midline — same layer, by design) |
| `core/midline/midline.py` | `core/types/midline.py`, `core/types/crop.py` | imports Midline2D, CropRegion from types | VERIFIED | Lines 22-23 confirm `from aquapose.core.types.crop import CropRegion` and `from aquapose.core.types.midline import Midline2D` |
| `core/reconstruction/triangulation.py` | `core/types/midline.py`, `core/types/reconstruction.py` | imports Midline2D, Midline3D, MidlineSet from types | VERIFIED | Lines 19-20 confirm both imports |
| `core/reconstruction/curve_optimizer.py` | `core/reconstruction/triangulation.py` | imports _pixel_half_width_to_metres | VERIFIED | Line 24 confirms `from aquapose.core.reconstruction.triangulation import (...)` |
| `core/midline/backends/segmentation.py` | `core/types/midline.py` + `core/midline/midline.py` | Midline2D from types, private helpers from midline.py | VERIFIED | Lines 19-29 show correct split imports |
| `core/midline/backends/pose_estimation.py` | `core/types/` + `core/midline/crop.py` | Detection, AffineCrop from types; extract_affine_crop from crop | VERIFIED | Lines 20-24 confirm split imports |
| `visualization/midline_viz.py` | `core/types/` + `core/midline/midline.py` | function-level lazy imports | VERIFIED | Lines 24-25, 393, 577 show updated lazy imports to core paths |

---

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| REORG-01 | 39-01, 39-02, 39-03 | Legacy top-level domain libraries (reconstruction/, segmentation/, tracking/) reorganized into core/ submodules with shared types extracted to core/types/ | SATISFIED | Legacy dirs deleted; core/types/ package exists; all imports rewired; 656 tests pass |
| STAB-04 | 39-04 | All stale docstrings referencing U-Net, no-op stubs, or Phase 37 pending status are updated | SATISFIED | grep across src/ returns 0 matches for "U-Net", "no-op stub", "Phase 37 pending"; lint passes |

Both requirement IDs from REQUIREMENTS.md for Phase 39 are accounted for and verified satisfied.

---

### Anti-Patterns Found

None. All checks clean:
- Zero TODO/FIXME/PLACEHOLDER in moved files
- Zero legacy path references in docstrings
- Lint passes (`ruff check` — all checks passed)
- Test suite: 656 passed, 0 failures

---

### Human Verification Required

None. All phase goals are verifiable programmatically via import checks, grep, and the test suite.

---

### Gaps Summary

No gaps. All 12 observable truths verified, all 19 artifacts in expected state, all 7 key links confirmed wired.

**Notable design decision (not a gap):** `core/types/reconstruction.py` imports `Midline2D` from `core/types/midline` — this is a within-types-layer dependency (MidlineSet is a type alias over Midline3D which references Midline2D). This was explicitly documented in the 39-01-SUMMARY as a deliberate design decision, and does not violate the constraint that "types files must not import from implementation modules."

**Notable naming difference (not a gap):** `YOLODetectionBackend` referenced in the 39-01-PLAN was actually implemented as `YOLOBackend` in the final code. The class is exported correctly in `__all__` and the plan's core intent (merge YOLODetector into the detection backend file) is fully achieved.

---

_Verified: 2026-03-02T19:00:00Z_
_Verifier: Claude (gsd-verifier)_
