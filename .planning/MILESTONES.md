# Milestones

## v1.0 MVP (Shipped: 2026-02-25)

**Phases completed:** 12 phases, 28 plans
**Timeline:** 11 days (2026-02-14 → 2026-02-25)
**Codebase:** 50,802 LOC Python, 300 files modified
**Git range:** e9eddd1..590d068

**Key accomplishments:**
1. Differentiable refractive projection layer validated across 13 cameras (sub-pixel accuracy, Z-uncertainty 132x XY quantified)
2. Full segmentation pipeline: MOG2/YOLO detection → SAM2 pseudo-labels → U-Net inference (best val IoU 0.623)
3. Parametric fish mesh with differentiable spine + cross-sections in PyTorch (watertight, batch-first)
4. RANSAC cross-view identity association + Hungarian 3D tracking with persistent fish IDs (9 fish, population-constrained)
5. End-to-end direct triangulation pipeline: 2D midlines → arc-length sampling → RANSAC triangulation → B-spline fitting → HDF5 output + visualization
6. Correspondence-free B-spline curve optimizer as alternative reconstruction method (chamfer distance + L-BFGS)

**Delivered:** Complete 3D fish midline reconstruction system from multi-view silhouettes via refractive multi-view triangulation, with two reconstruction methods (direct triangulation and curve optimization), diagnostic tooling, and synthetic test infrastructure.

**Known gaps:**
- Phase 8 plan 03 (E2E integration test on real data): SUMMARY.md missing from disk, though E2E test code was committed (feat(08-03))
- No milestone audit performed — completed without pre-audit

**Architecture pivot:** Analysis-by-synthesis pipeline (Phases 3-4) shelved mid-milestone due to 30+ min/sec runtime. Replaced by direct triangulation pipeline (Phases 5-9) achieving orders-of-magnitude faster reconstruction.

---

