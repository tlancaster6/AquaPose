# Requirements: AquaPose

**Defined:** 2026-03-10
**Core Value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation

## v3.7 Requirements

Requirements for the Improved Tracking milestone. Each maps to roadmap phases.

### Investigation

- [x] **INV-01**: Occlusion investigation script generates per-detection pose overlay video with confidence visualization for a configurable camera/frame range
- [x] **INV-02**: Written summary characterizing OBB detector and pose model behavior under fish occlusion with go/no-go recommendation
- [x] **INV-03**: Baseline tracking metrics (track count, duration distribution, fragmentation, coverage) measured on perfect-tracking target clip with current OC-SORT
- [x] **INV-04**: Detection confidence threshold determined empirically — characterize the quality vs false-positive tradeoff across confidence levels

### Remediation

- [x] **REM-01**: Occlusion-related failure modes identified in INV-02 are addressed before tracker implementation (conditional — skip if INV-02 is go) — **SKIPPED: INV-02 yielded GO**

### Production Retrain

- [x] **RETRAIN-01**: OBB dataset assembled with all-source stratified val split (manual + corrected pseudo-labels) for production training
- [x] **RETRAIN-02**: Pose dataset assembled with all-source stratified val split and elastic augmentation for production training
- [x] **RETRAIN-03**: Production OBB and Pose models trained with 300 epochs/patience=50 and evaluated against Round 1 winners
- [x] **RETRAIN-04**: Project config updated with production model weights; white-wall recall visually verified via investigation script

### Pipeline Architecture

- [x] **PIPE-01**: Pose estimation runs immediately after detection (Stage 2), before tracking
- [x] **PIPE-02**: Segmentation midline backend removed (backends/segmentation.py, skeletonization code, orientation resolution)
- [x] **PIPE-03**: PipelineContext and stage interfaces updated for new stage ordering

### Tracking

- [x] **TRACK-01**: Custom keypoint tracker runs forward and backward OC-SORT passes over each chunk
- [x] **TRACK-02**: OKS-based association cost replaces IoU on OBBs
- [x] **TRACK-03**: OCM direction consistency term using spine heading vector
- [x] **TRACK-04**: Kalman filter tracks keypoint positions and velocities (60-dim or 24-dim state)
- [x] **TRACK-05**: Asymmetric track birth/death based on frame-edge proximity
- [x] **TRACK-06**: ORU (observation-centric re-update) and OCR (observation-centric recovery) mechanisms
- [x] **TRACK-07**: Bidirectional merge combines forward and backward tracklets with overlap-based matching
- [x] **TRACK-08**: Chunk boundary handoff via serialized KF state (mean + covariance + observation history)
- [x] **TRACK-09**: Gap interpolation fills small tracklet gaps via spline interpolation
- [x] **TRACK-10**: Secondary BYTE-style association pass for low-confidence detections (conditional — implement if INV-04 reveals significant low-confidence valid detections)

### Association

- [x] **ASSOC-01**: Cross-view association uses mid-body keypoint centroid instead of OBB centroid for ray-based matching

### Integration

- [ ] **INTEG-01**: New tracker evaluated against INV-03 baseline metrics on perfect-tracking target
- [ ] **INTEG-02**: Full pipeline runs end-to-end from CLI with new stage ordering
- [ ] **INTEG-03**: Code quality audit: no dead code, broken cross-references, or type errors from the overhaul
- [ ] **INTEG-04**: BoxMot dependency removal decision documented

## Future Requirements

### Tracking V2

- **TRACK-V2-01**: Appearance/ReID embeddings for long-range re-identification
- **TRACK-V2-02**: Global optimization (min-cost flow) for tracklet merge
- **TRACK-V2-03**: Chunk overlap for boundary reconciliation (if needed)
- **TRACK-V2-04**: Full 15-point pose-aware cross-view association (beyond keypoint centroid swap)

## Out of Scope

| Feature | Reason |
|---------|--------|
| ReID appearance embeddings | V2 — requires training dedicated model, premature for V1 |
| Global optimization (min-cost flow) | V2 — forward/backward merge is sufficient for V1 |
| Chunk overlap for boundary reconciliation | V2 — KF state handoff should suffice; add only if needed |
| Full pose-aware cross-view association | V2 — keypoint centroid swap is minimally invasive first step |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INV-01 | Phase 78 | Complete |
| INV-02 | Phase 78 | Complete |
| INV-03 | Phase 80 | Complete |
| INV-04 | Phase 78 | Complete |
| REM-01 | Phase 79 | Skipped (GO) |
| RETRAIN-01 | Phase 78.1 | Complete |
| RETRAIN-02 | Phase 78.1 | Complete |
| RETRAIN-03 | Phase 78.1 | Complete |
| RETRAIN-04 | Phase 78.1 | Complete |
| PIPE-01 | Phase 81 | Complete |
| PIPE-02 | Phase 81 | Complete |
| PIPE-03 | Phase 81 | Complete |
| TRACK-01 | Phase 83 | Complete |
| TRACK-02 | Phase 83 | Complete |
| TRACK-03 | Phase 83 | Complete |
| TRACK-04 | Phase 83 | Complete |
| TRACK-05 | Phase 83 | Complete |
| TRACK-06 | Phase 83 | Complete |
| TRACK-07 | Phase 83 | Complete |
| TRACK-08 | Phase 83 | Complete |
| TRACK-09 | Phase 83 | Complete |
| TRACK-10 | Phase 83 | Complete |
| ASSOC-01 | Phase 82 | Complete |
| INTEG-01 | Phase 84 | Pending |
| INTEG-02 | Phase 84 | Pending |
| INTEG-03 | Phase 85 | Pending |
| INTEG-04 | Phase 85 | Pending |

**Coverage:**
- v3.7 requirements: 27 total
- Mapped to phases: 27
- Unmapped: 0

---
*Requirements defined: 2026-03-10*
*Last updated: 2026-03-10 after Phase 78.1 planning*
