# Requirements: AquaPose

**Defined:** 2026-03-13
**Core Value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation

## v3.9 Requirements

Requirements for Reconstruction Modernization milestone. Each maps to roadmap phases.

### Config Plumbing

- [x] **CFG-01**: `n_sample_points` propagated from ReconstructionConfig through pipeline.py to ReconstructionStage
- [x] **CFG-02**: Default `n_sample_points` changed from 15 to 6 (identity mapping: 6 keypoints → 6 body points)

### Spline Refactoring

- [ ] **SPL-01**: B-spline fitting moved out of core reconstruction path into optional post-processing utility
- [ ] **SPL-02**: Reconstruction produces raw triangulated keypoints as primary Midline3D output when spline is disabled
- [ ] **SPL-03**: Midline3D type updated to support both spline-based and raw-keypoint representations

### Dead Code Removal

- [ ] **CLEAN-01**: Dead `_triangulate_body_point()` scalar fallback removed from dlt.py
- [ ] **CLEAN-02**: Stale comments referencing scalar fallback path removed

### Z-Denoising

- [ ] **ZDEN-01**: Z-denoising (centroid flatten + temporal smooth) operates correctly on raw keypoint arrays (n_sample_points=6)

### Documentation & Types

- [ ] **DOC-01**: stage.py module docstring updated to reflect keypoint-native N-point output
- [ ] **DOC-02**: Midline2D and Midline3D type documentation updated for variable point counts

## Future Requirements

Deferred to future release. Tracked but not in current roadmap.

### Width Estimation

- **WIDTH-01**: Per-keypoint width estimation from segmentation data (requires seg backend reintroduction)

### Advanced Z-Denoising

- **ZDEN-02**: Per-keypoint independent z-smoothing as alternative to centroid flattening
- **ZDEN-03**: Adaptive smoothing kernel width based on reconstruction confidence

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Segmentation backend reintroduction | Removed in v3.7; keypoint-only pipeline is the direction |
| Alternative triangulation backends | DLT is sole backend per v3.1 decision; curve optimizer removed |
| Spline degree/control-point auto-tuning | Current 7-control-point cubic is sufficient; parameterize only |
| Real-time reconstruction | Batch-only per project constraints |
| Width profile from keypoints | No width signal in pose-only pipeline |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CFG-01 | Phase 93 | Complete |
| CFG-02 | Phase 93 | Complete |
| SPL-01 | Phase 95 | Pending |
| SPL-02 | Phase 95 | Pending |
| SPL-03 | Phase 95 | Pending |
| CLEAN-01 | Phase 94 | Pending |
| CLEAN-02 | Phase 94 | Pending |
| ZDEN-01 | Phase 96 | Pending |
| DOC-01 | Phase 96 | Pending |
| DOC-02 | Phase 96 | Pending |

**Coverage:**
- v3.9 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-13*
*Last updated: 2026-03-13 after roadmap creation*
