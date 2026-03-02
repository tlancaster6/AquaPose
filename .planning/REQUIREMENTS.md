# Requirements: AquaPose

**Defined:** 2026-03-02
**Core Value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation

## v3.1 Requirements

Requirements for reconstruction rebuild milestone. Each maps to roadmap phases.

### Diagnostics

- [x] **DIAG-01**: Diagnostic observer captures and serializes MidlineSet data from pipeline runs
- [x] **DIAG-02**: Serialized MidlineSet fixtures can be loaded independently of the pipeline for offline evaluation

### Evaluation

- [x] **EVAL-01**: Evaluation harness loads MidlineSet fixtures + calibration data and runs metrics without the full pipeline
- [x] **EVAL-02**: Frame selection produces 15-20 frames from ~300 frame window via uniform temporal sampling
- [x] **EVAL-03**: Tier 1 metric: per-fish, per-camera reprojection error (mean and max) with overall aggregates
- [x] **EVAL-04**: Tier 2 metric: leave-one-out camera stability (max control-point displacement across dropout runs)
- [x] **EVAL-05**: Evaluation outputs human-readable summary table and machine-diffable regression data
- [x] **EVAL-06**: Baseline evaluation run against current reconstruction backend establishes reference metrics

### Reconstruction

- [x] **RECON-01**: Per-body-point triangulation via confidence-weighted DLT (single strategy regardless of camera count)
- [ ] **RECON-02**: Outlier camera rejection via reprojection residual threshold (empirically tuned)
- [ ] **RECON-03**: Re-triangulation with inlier cameras after outlier rejection
- [x] **RECON-04**: B-spline fitting via `make_lsq_spline` with 7 control points and minimum valid-point threshold
- [ ] **RECON-05**: Water surface rejection (Z ≤ water_z)
- [ ] **RECON-06**: Low-confidence flagging when configurable fraction of body points had <3 inlier cameras
- [ ] **RECON-07**: Half-widths passed through from upstream (defaults if absent), not used in reconstruction logic
- [ ] **RECON-08**: New triangulation backend outperforms or matches current backend on Tier 1 and Tier 2 metrics

### Cleanup

- [ ] **CLEAN-01**: Old triangulation backend code removed
- [ ] **CLEAN-02**: Curve optimizer backend code removed
- [ ] **CLEAN-03**: Dead code removed (refine_midline_lm stub, unused orientation/epipolar code paths)

## Future Requirements

### Deferred Evaluation

- **EVAL-T3-01**: Tier 3 synthetic ground-truth evaluation using existing synthetic data kit
- **EVAL-T3-02**: Synthetic inputs precomputed and checked in for CI-portable unit tests

### Deferred Reconstruction

- **RECON-CURVE-01**: Curve optimizer rebuild (must beat triangulation baseline on eval harness)
- **RECON-SEG-01**: Segmentation midline backend support in reconstruction
- **RECON-EPI-01**: Epipolar refinement (only if eval shows correspondence errors matter)
- **RECON-ORI-01**: Orientation logic in reconstruction (only if upstream orientation proves unreliable)
- **RECON-HW-01**: Half-width reconstruction logic (if downstream consumers need accurate 3D half-widths)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Chunk processing for long videos | Separate milestone — v3.1 works with ~300 frames |
| Curve optimizer rebuild | Must beat triangulation baseline to justify reintroduction |
| Segmentation midline backend in reconstruction | Pose estimation backend only — ordered keypoints eliminate correspondence machinery |
| Epipolar refinement | Expensive, fragile; ordered keypoints already solve the problem it addresses |
| Real-time processing | Batch only |
| Orientation logic in reconstruction | Lives in midline stage; reintroduce only if evaluation demonstrates need |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DIAG-01 | Phase 40 | Complete |
| DIAG-02 | Phase 40 | Complete |
| EVAL-01 | Phase 41 | Complete |
| EVAL-02 | Phase 41 | Complete |
| EVAL-03 | Phase 41 | Complete |
| EVAL-04 | Phase 41 | Complete |
| EVAL-05 | Phase 41 | Complete |
| EVAL-06 | Phase 42 | Complete |
| RECON-01 | Phase 43 | Complete |
| RECON-02 | Phase 43 | Pending |
| RECON-03 | Phase 43 | Pending |
| RECON-04 | Phase 43 | Complete |
| RECON-05 | Phase 43 | Pending |
| RECON-06 | Phase 43 | Pending |
| RECON-07 | Phase 43 | Pending |
| RECON-08 | Phase 44 | Pending |
| CLEAN-01 | Phase 45 | Pending |
| CLEAN-02 | Phase 45 | Pending |
| CLEAN-03 | Phase 45 | Pending |

**Coverage:**
- v3.1 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-02*
*Last updated: 2026-03-02 after roadmap creation (v3.1)*
