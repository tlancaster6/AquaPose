# Phase 104: Backbone Fine-Tuning - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Fine-tune MegaDescriptor-T to discriminate female cichlids well enough to gate swap repair. The deliverable is a trained projection head that produces 128-dim embeddings with female-female pair AUC >= 0.75 on a temporal holdout set. Re-embedding all detections with the fine-tuned model is included. Swap detection and repair are Phase 105.

</domain>

<decisions>
## Implementation Decisions

### Loss function
- SubcenterArcFace with 3 sub-centers per identity (9 identities total)
- Use pytorch-metric-learning library (new dependency)
- Monitor ArcFace loss + validation AUC only — no additional triplet metrics

### Training data handling
- Group-level identity trust boundary: fish_id is only reliable within a temporal group. Same fish_id across groups does NOT necessarily mean the same physical fish — the short temporal windows exist specifically to protect against the persistent ID swaps that ReID is designed to fix
- Positive pairs: cross-camera, any frame within the same group (not restricted to same-frame). Maximizes appearance diversity (viewpoint + pose/curvature variation)
- Batch sampling: uniform from all valid cross-camera crops within a group; BatchHardMiner selects hard positives/negatives naturally
- Train/val split: temporal holdout at group level (last 20% of groups by time)
- Overlapping windows from the miner (50% stride) are accepted as-is — no deduplication needed since the temporal split prevents leakage

### Fine-tuning strategy
- Freeze entire Swin-Tiny backbone (28M params), train only the new projection head + ArcFace classification head
- Cache backbone features to disk (768-dim vectors per crop) — run backbone once, then training reads cached vectors with no GPU backbone cost. Re-embedding = run small head on cached vectors (~seconds)
- Projection head architecture: Linear(768, 256) → BN → ReLU → Linear(256, 128) → L2-normalize
- Output embedding dimension: 128
- ArcFace head: SubcenterArcFace(128, 9, sub_centers=3) — discarded after training
- Epochs: 50, early stopping patience: 10 on holdout AUC
- Save projection head weights only (~200KB), not the frozen backbone

### Evaluation and gating
- Standalone evaluation script (not integrated into pipeline), hardcoded sex labels
- Males: fish 5, 6, 7. Females: fish 0, 1, 2, 3, 4, 8
- Known swap in test data: fish 2 ↔ fish 4 (both female) around frame 600
- Gate metric: all-female-pair AUC >= 0.75 on temporal holdout groups
- Report includes per-pair breakdown to identify bottleneck pairs
- If gate passes: automatically re-embed all detections using cached backbone features + trained head, write embeddings_finetuned.npz, print before/after within-ID cosine similarity comparison
- If gate fails: print diagnostic with achieved AUC, per-pair breakdown, suggested next steps, exit(1)

### Claude's Discretion
- Learning rate and schedule for the projection head
- Batch size and samples-per-identity-per-batch
- ArcFace margin and scale hyperparameters
- Exact early stopping metric (loss vs AUC plateau)
- Feature cache file format (NPZ vs H5)

</decisions>

<specifics>
## Specific Ideas

- The backbone caching approach was specifically chosen to make training iteration fast — epochs should take seconds, not minutes
- The group trust boundary is a critical design constraint: the miner's temporal windows exist to protect against the exact swap errors that ReID is meant to fix. Never assume cross-group fish_id consistency.
- For the YH test data: one remaining true swap (fish 2 ↔ 4 around frame 600) was verified by manual video inspection. The per-pair AUC breakdown should reveal whether this known-hard pair is the discriminability bottleneck.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `FishEmbedder` (core/reid/embedder.py): MegaDescriptor-T wrapper via timm, BGR crop → L2-normalized 768-dim vectors. Can be reused for the backbone feature caching step.
- `EmbedRunner` (core/reid/runner.py): Iterates (frame, fish_id, camera) tuples from H5 + chunk caches, batches through embedder. Iteration logic reusable for caching all backbone features.
- `TrainingDataMiner` (core/reid/miner.py): Outputs `reid_crops/group_NNN/fish_N/*.jpg` with per-group manifest JSON. Group structure and manifests are the training data source.
- `compute_reid_metrics` / `print_reid_report` (core/reid/eval.py): Zero-shot cosine similarity, rank-1, mAP. Can be extended or called for before/after comparison.
- `ReidConfig` (engine/config.py): Frozen dataclass with model_name, batch_size, crop_size, device, embedding_dim.
- `extract_affine_crop` (core/pose/crop.py): OBB-aligned crop extraction — already used by miner and embedder.

### Established Patterns
- Training code lives in `src/aquapose/training/` with standalone function pattern (see `train_yolo()`)
- The training subsystem must not import from `engine/` (enforced by pre-commit AST import boundary)
- Frozen dataclass configs in `engine/config.py`
- Standalone scripts preferred for new algorithms before CLI integration

### Integration Points
- Reads: `reid_crops/` directory with group manifests from Phase 103 miner output
- Reads: existing `embeddings.npz` from Phase 102 embed runner (for zero-shot baseline comparison)
- Writes: `best_reid_model.pt` (projection head weights)
- Writes: `embeddings_finetuned.npz` (re-embedded detections with fine-tuned model)
- Writes: evaluation report with AUC metrics and per-pair breakdown
- Phase 105 consumes: fine-tuned embeddings for swap detection

</code_context>

<deferred>
## Deferred Ideas

- Progressive backbone unfreezing if head-only doesn't reach the AUC gate — try as manual intervention if gate fails
- Image-level augmentation during training — ruled out by feature caching approach, revisit only if head-only underperforms
- Cross-session fish identity persistence — explicitly out of scope per REQUIREMENTS.md

</deferred>

---

*Phase: 104-backbone-fine-tuning*
*Context gathered: 2026-03-25*
