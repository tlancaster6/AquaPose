# Phase 107: Unfrozen Backbone Fine-Tuning - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Frozen MegaDescriptor-T backbone hits a hard discrimination ceiling on YH female cichlids (fish 2v4 at 0.495 AUC, overall female AUC 0.64). Projection head training adds ~0.04 AUC max — the backbone features don't encode enough appearance difference. This phase adds support for partially unfreezing the backbone during fine-tuning so the model can learn fish-specific discriminative features.

This changes the existing training architecture significantly:
- **Before:** Cache backbone features once, train only projection head, apply head to cached features
- **After:** Run crops through backbone end-to-end during training, re-embed using fine-tuned backbone

The existing frozen-head path should remain as a fast baseline option.

</domain>

<decisions>
## Implementation Decisions

### Training Architecture
- Support configurable unfreeze depth: last N transformer blocks of the ViT backbone
- Use differential learning rates: backbone LR should be 10-100x lower than head LR to avoid catastrophic forgetting
- Keep the warmup + cosine annealing scheduler from Phase 106
- MultiSimilarityLoss + MultiSimilarityMiner (current) is fine as the loss function
- MPerClassSampler with samples_per_class=16 (current) is fine
- The cached-features fast path (frozen backbone) should still work when unfreeze_blocks=0

### Re-embedding
- When backbone is fine-tuned, re-embedding must run crops through the full fine-tuned model (backbone + head), not just apply a head to stale zero-shot features
- Save the full fine-tuned model (backbone state + head state) as the checkpoint, not just head weights
- The CLI `reid fine-tune` command should handle this transparently — if backbone was unfrozen, re-embed uses the fine-tuned backbone

### Swap Detector Integration
- SwapDetector currently loads a ProjectionHead and applies it to cached zero-shot embeddings
- With a fine-tuned backbone, it should either: (a) use pre-computed fine-tuned embeddings from embeddings_finetuned.npz, or (b) load the full model for on-the-fly embedding
- Option (a) is simpler and preferred — the fine-tune step already saves embeddings_finetuned.npz

### Claude's Discretion
- Whether to save backbone + head as a single checkpoint or separate files
- How to structure the unfreezing (by block index, by layer name pattern, etc.)
- Whether to add gradient clipping for stability
- Image augmentation during end-to-end training (random crops, flips, color jitter)

</decisions>

<specifics>
## Specific Ideas

- MegaDescriptor-T is a ViT-Small with 12 transformer blocks — unfreezing last 2-3 blocks is a good starting point
- The backbone_cache.npz optimization no longer applies when backbone is unfrozen — training must load and forward-pass actual crop images each epoch
- With 3056 crops and batch size 144, each epoch is ~21 batches — should be fast even with backbone forward pass on GPU
- Consider saving training curves (loss + AUC per epoch) to a JSON/CSV for analysis

</specifics>

<deferred>
## Deferred Ideas

- Full backbone fine-tuning (all blocks) — start with partial and see if it helps
- Knowledge distillation from a larger model
- Temporal/trajectory-based ReID methods (non-appearance)

</deferred>

---

*Phase: 107-unfrozen-backbone-fine-tuning*
*Context gathered: 2026-03-25 from production run analysis*
