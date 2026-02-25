# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — MVP

**Shipped:** 2026-02-25
**Phases:** 12 | **Plans:** 28 | **Timeline:** 11 days

### What Was Built
- Complete 3D fish midline reconstruction from multi-view silhouettes
- Two reconstruction methods: direct triangulation (fast) and curve optimization (experimental)
- Full detection-to-output pipeline: YOLO/MOG2 → SAM2 pseudo-labels → U-Net → skeletonization → triangulation → B-spline → HDF5
- Synthetic data generation system for controlled testing
- Diagnostic tooling: CLEAR MOT metrics, 2D overlays, 3D animations, markdown reports

### What Worked
- **Architecture pivot was decisive:** Shelving analysis-by-synthesis (Phase 4) mid-milestone and switching to direct triangulation was the right call — saved weeks of optimization dead-end
- **Decimal phase insertion:** Urgent work (02.1, 02.1.1, 04.1) was cleanly inserted without disrupting the main roadmap
- **Quick tasks for targeted fixes:** 8 quick tasks handled bug fixes and enhancements without full phase overhead
- **Batch-first APIs from day one:** All modules accepted lists, making multi-fish extension straightforward
- **GSD workflow:** Planning → research → execution cycle kept each phase focused and dependency-ordered

### What Was Inefficient
- **U-Net IoU 0.623 accepted too early:** Should have invested more in segmentation quality before building 6 downstream phases — noisy 2D midlines are now the primary bottleneck
- **Phase 2 was planned and then entirely replanned:** Original 4 plans were superseded by troubleshooting phases (02.1, 02.1.1), then the module was rewritten again. Better upfront real-data testing would have caught issues earlier
- **Phase 8 plan 03 summary gap:** E2E integration test was committed but summary not written, creating an audit gap

### Patterns Established
- `.cpu().numpy()` always (never bare `.numpy()`) for GPU tensors
- Differential LR for encoder/decoder in U-Net training
- XY-only cost matrices for tracking (Z uncertainty too high)
- Population-constrained tracking with fish ID recycling
- Exhaustive pairwise triangulation for ≤7 cameras
- Chamfer distance (not index-aligned residual) for curve optimization

### Key Lessons
1. **Segmentation quality gates reconstruction quality** — investing in better masks earlier would have prevented weeks of debugging noisy midlines downstream
2. **Architecture pivots are cheaper than sunk-cost optimization** — the ABS-to-triangulation pivot saved the project from a 30-min/sec dead end
3. **Z-reconstruction is fundamentally limited by top-down geometry** — 132x anisotropy means XY-only approaches are often superior to full 3D
4. **Synthetic data is essential for debugging** — real data has too many confounds; synthetic data isolated individual failure modes

### Cost Observations
- Model mix: ~70% sonnet, ~25% opus, ~5% haiku (balanced profile)
- Sessions: ~15-20 across 11 days
- Notable: Quick tasks (8 total) handled ~30% of functional improvements with minimal overhead

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Timeline | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | 11 days | 12 | Initial development; architecture pivot mid-milestone |

### Cumulative Quality

| Milestone | LOC | Files | Quick Tasks |
|-----------|-----|-------|-------------|
| v1.0 | 50,802 | 300 | 8 |

### Top Lessons (Verified Across Milestones)

1. Invest in input quality (segmentation) before building downstream pipelines
2. Pivot early when runtime measurements invalidate an approach
