# Requirements: AquaPose

**Defined:** 2026-03-05
**Core Value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation

## v3.4 Requirements

Requirements for performance optimization milestone. Each maps to roadmap phases.

### Inference Batching

- [ ] **BATCH-01**: Detection stage batches all camera frames per timestep into a single `predict()` call
- [ ] **BATCH-02**: Midline stage batches all crops per frame into a single `predict()` call
- [ ] **BATCH-03**: Batch sizes are configurable via pipeline config fields
- [ ] **BATCH-04**: Inference gracefully retries with halved batch size on CUDA OOM

### Frame I/O

- [x] **FIO-01**: Frame source prefetches frames in a background thread via producer-consumer queue
- [x] **FIO-02**: Prefetch source satisfies existing FrameSource protocol (drop-in replacement)

### Reconstruction

- [x] **RECON-01**: DLT first-pass triangulation vectorized across body points via batched SVD
- [x] **RECON-02**: Vectorized reconstruction produces numerically equivalent results to per-point loop

### Association

- [ ] **ASSOC-01**: Pairwise ray scoring vectorized via NumPy broadcasting
- [ ] **ASSOC-02**: Vectorized scoring produces identical results to per-pair loop

## Future Requirements

None — this is a focused optimization milestone.

## Out of Scope

| Feature | Reason |
|---------|--------|
| TensorRT model export | Premature — batch inference gains should be confirmed first |
| GPU video decoding (NVDEC/decord) | CPU-side undistortion negates GPU decode gains |
| Multiprocessing decode | IPC overhead for large frame arrays likely negative ROI |
| asyncio integration | Wrong primitive for synchronous C extensions |
| Numba JIT for DLT | NumPy batched SVD is simpler and sufficient |
| Benchmark infrastructure | Ad-hoc timing via existing stage logs and py-spy sufficient |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ASSOC-01 | Phase 56 | Pending |
| ASSOC-02 | Phase 56 | Pending |
| RECON-01 | Phase 57 | Complete |
| RECON-02 | Phase 57 | Complete |
| FIO-01 | Phase 58 | Complete |
| FIO-02 | Phase 58 | Complete |
| BATCH-01 | Phase 59 | Pending |
| BATCH-02 | Phase 59 | Pending |
| BATCH-03 | Phase 59 | Pending |
| BATCH-04 | Phase 59 | Pending |

**Coverage:**
- v3.4 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-05*
*Last updated: 2026-03-05 after roadmap creation — all 10 requirements mapped to phases 56-59*
