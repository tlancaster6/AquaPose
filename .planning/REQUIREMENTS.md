# Requirements: AquaPose

**Defined:** 2026-03-25
**Core Value:** Accurate 3D fish midline reconstruction from multi-view silhouettes via refractive multi-view triangulation

## v3.11 Requirements

Requirements for appearance-based re-identification milestone. Each maps to roadmap phases.

### Embedding Infrastructure

- [ ] **EMBED-01**: Crop extractor extracts affine-warped fish crops from video frames using OBB detections (reuses `extract_affine_crop()`)
- [ ] **EMBED-02**: Backbone wrapper loads MegaDescriptor-T via timm and produces L2-normalized embedding vectors from crops
- [ ] **EMBED-03**: Batch embed runner iterates all (frame, fish_id, camera) tuples in a completed run, batches crops through backbone, and writes embeddings to disk

### Training

- [ ] **TRAIN-01**: Training data extractor mines crops from high-confidence 3D trajectory segments (configurable quality filters: min cameras, min duration, max residual)
- [ ] **TRAIN-02**: Contamination filter excludes crops within 150-frame buffer around detected changepoints/swap events
- [ ] **TRAIN-03**: Fine-tuning loop trains backbone with metric learning loss (triplet or SubcenterArcFace) and BatchHardMiner on extracted fish crops

### Unfrozen Backbone Fine-Tuning

- [ ] **UNFREEZE-01**: Configurable partial backbone unfreezing (last N Swin transformer blocks) with differential learning rates (backbone LR 10-100x lower than head LR)
- [ ] **UNFREEZE-02**: End-to-end training path loads raw crop images through backbone+head each epoch (no stale cached features) and saves combined checkpoint with backbone_state_dict + head_state_dict
- [ ] **UNFREEZE-03**: CLI `reid fine-tune --unfreeze-blocks N` transparently selects end-to-end training and re-embeds using the fine-tuned backbone, producing embeddings_finetuned.npz compatible with SwapDetector

### Swap Repair

- [x] **SWAP-01**: Swap detector identifies ID swap candidates at occlusion events by comparing pre/post-event mean embeddings via cosine similarity
- [x] **SWAP-02**: Margin-gated repair re-assigns fish IDs only when cosine margin exceeds threshold, writes corrected output to `midlines_reid.h5`

### CLI

- [x] **CLI-01**: `aquapose reid` command group with subcommands (embed, repair, train-data) following existing CLI patterns

## Future Requirements

Deferred to future release. Tracked but not in current roadmap.

### Data Quality

- **QUAL-01**: Cross-camera positive pair enforcement (>= 50% cross-camera pairs in training batches)
- **QUAL-02**: HDF5 embedding storage with camera_id and detection confidence metadata
- **QUAL-03**: Reprojection error regression check on repaired segments

### Production

- **PROD-01**: Embed runner resume support (skip already-computed embeddings)
- **PROD-02**: Confidence-gated prototype update (gate on OBB confidence + camera count)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Online ReID inside PipelineStage | Adds GPU overhead to hot path; post-hoc repair is correct design |
| Training from scratch | Infeasible with 9 fish identities; transfer from MegaDescriptor is mandatory |
| Sex classification as auxiliary signal | Male-female already handled geometrically; doesn't help female-female |
| Global (unconditional) swap repair | Will introduce false repairs in correctly-assigned segments |
| Cross-session fish identity persistence | Hard open problem, out of scope |
| Cross-chunk ReID replacing body-length stitching | Separate milestone; current stitching works well |
| Keypoint-guided crop normalization | Only if OBB crops show residual roll variance; defer until measured |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| EMBED-01 | Phase 102 | Pending |
| EMBED-02 | Phase 102 | Pending |
| EMBED-03 | Phase 102 | Pending |
| TRAIN-01 | Phase 103 | Pending |
| TRAIN-02 | Phase 103 | Pending |
| TRAIN-03 | Phase 104 | Pending |
| UNFREEZE-01 | Phase 107 | Pending |
| UNFREEZE-02 | Phase 107 | Pending |
| UNFREEZE-03 | Phase 107 | Pending |
| SWAP-01 | Phase 105 | Complete |
| SWAP-02 | Phase 105 | Complete |
| CLI-01 | Phase 106 | Complete |

**Coverage:**
- v3.11 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0

---
*Requirements defined: 2026-03-25*
*Last updated: 2026-03-25 after Phase 107 planning*
