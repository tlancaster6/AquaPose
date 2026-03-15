# Requirements: AquaPose v3.10 Publication Metrics

**Defined:** 2026-03-14
**Core Value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation

## v3.10 Requirements

Requirements for publication-ready metrics. Each maps to roadmap phases.

### Pipeline Run

- [x] **RUN-01**: Full 5-minute diagnostic pipeline run (9,000 frames) completes with production models
- [x] **RUN-02**: Per-stage timing breakdown recorded (detection, pose, tracking, association, reconstruction)
- [x] **RUN-03**: End-to-end throughput measured (frames/sec, wall-time)

### Reconstruction Quality

- [x] **RECON-01**: Reprojection error distribution reported (mean, p50, p90, p99) on full run
- [x] **RECON-02**: Per-keypoint reprojection error breakdown on full run
- [x] **RECON-03**: Camera visibility statistics (mean cameras per fish, distribution) on full run

### Tracking Quality

- [ ] **TRACK-01**: Track count and fragmentation metrics on full run
- [ ] **TRACK-02**: Identity consistency across chunk boundaries on full run
- [ ] **TRACK-03**: Detection coverage (% frames with detections per camera) on full run

### Association Quality

- [ ] **ASSOC-01**: Singleton rate measured on full run
- [ ] **ASSOC-02**: Association wall-time measured on full run

### Results Document

- [ ] **DOC-01**: performance-accuracy.md updated with all new full-run metrics and CSVs
- [ ] **DOC-02**: Stale results section cleared (replaced by current measurements)

## Future Requirements

None — this is a metrics-only milestone.

## Out of Scope

| Feature | Reason |
|---------|--------|
| New pipeline features or algorithms | This milestone is evaluation-only |
| Biological analysis of 3D trajectories | Application paper scope, not methods paper |
| Comparison to other multi-view pose systems | No standardized benchmarks exist for refractive aquatic setups |
| Figure generation / LaTeX formatting | Downstream of data collection; separate task |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| RUN-01 | Phase 97 | Complete |
| RUN-02 | Phase 98 | Complete |
| RUN-03 | Phase 98 | Complete |
| RECON-01 | Phase 99 | Complete |
| RECON-02 | Phase 99 | Complete |
| RECON-03 | Phase 99 | Complete |
| TRACK-01 | Phase 100 | Pending |
| TRACK-02 | Phase 100 | Pending |
| TRACK-03 | Phase 100 | Pending |
| ASSOC-01 | Phase 100 | Pending |
| ASSOC-02 | Phase 100 | Pending |
| DOC-01 | Phase 101 | Pending |
| DOC-02 | Phase 101 | Pending |

**Coverage:**
- v3.10 requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-14*
*Last updated: 2026-03-14 after roadmap creation*
