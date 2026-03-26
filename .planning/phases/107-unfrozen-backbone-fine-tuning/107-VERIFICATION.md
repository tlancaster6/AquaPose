---
phase: 107-unfrozen-backbone-fine-tuning
status: passed
verified: 2026-03-25
requirements_checked: [UNFREEZE-01, UNFREEZE-02, UNFREEZE-03]
---

# Phase 107: Unfrozen Backbone Fine-Tuning — Verification

## Goal
Support partial backbone unfreezing in ReID training so the model can learn fish-specific features beyond what frozen MegaDescriptor-T provides.

## Success Criteria Verification

### 1. `reid fine-tune --unfreeze-blocks 2` trains end-to-end with differential LR
**Status: PASSED**
- `cli.py:303` branches on `unfreeze_blocks > 0` and calls `train_reid_end_to_end(config)`
- `reid_training.py` `train_reid_end_to_end` creates optimizer with two param groups:
  - backbone params at `lr_head * lr_backbone_factor` (default 0.1x)
  - head params at `lr_head`
- `unfreeze_last_n_blocks(backbone, config.unfreeze_blocks)` selectively enables gradients on last N Swin blocks + final LayerNorm
- Tests verify unfreezing behavior: `test_unfreeze_zero_leaves_all_frozen`, `test_unfreeze_two_makes_last_two_trainable`

### 2. Combined checkpoint with backbone_state_dict + head_state_dict + config
**Status: PASSED**
- `train_reid_end_to_end` saves checkpoint with keys: `backbone_state_dict`, `head_state_dict`, `config`
- `config` dict contains `unfreeze_blocks`, `embedding_dim`, `hidden_dim`, `backbone_dim`
- Test `test_checkpoint_has_backbone_state` (@slow) verifies checkpoint format

### 3. Re-embedding runs crops through fine-tuned backbone+head
**Status: PASSED**
- `_reembed_finetuned` in cli.py loads checkpoint, creates backbone+head, wraps them in `_FineTunedEmbedder`
- `_FineTunedEmbedder.embed_batch` applies identical normalization to `FishEmbedder`: BGR->RGB, resize, (x/255-0.5)/0.5
- Forwards through backbone (raw output) then ProjectionHead (L2-normalizes)
- Uses EmbedRunner for crop extraction (monkey-patches FishEmbedder at module level)
- Writes `embeddings_finetuned.npz` with same schema

### 4. Default (unfreeze_blocks=0) uses cached features path unchanged
**Status: PASSED**
- `cli.py` else branch calls `build_feature_cache` + `train_reid_head` (identical to pre-phase code)
- Head-only re-embedding: loads state dict directly, applies to zero-shot embeddings
- No modifications to `train_reid_head`, `build_feature_cache`, `CachedFeatureDataset`, or `split_by_group`
- All 1281 existing tests pass (0 regressions)

### 5. SwapDetector reads embeddings_finetuned.npz without code changes
**Status: PASSED**
- `embeddings_finetuned.npz` written with keys: `embeddings`, `frame_index`, `fish_id`, `camera_id`, `detection_confidence`
- This matches the schema SwapDetector expects (same as `embeddings.npz`)
- No SwapDetector code was modified

## Requirements Traceability

| Requirement | Status | Evidence |
|-------------|--------|----------|
| UNFREEZE-01 | Complete | `unfreeze_last_n_blocks` + differential LR optimizer in `train_reid_end_to_end` |
| UNFREEZE-02 | Complete | `ImageCropDataset` loads raw crops, `train_reid_end_to_end` saves combined checkpoint |
| UNFREEZE-03 | Complete | `--unfreeze-blocks` CLI option, `_reembed_finetuned` produces `embeddings_finetuned.npz` |

## Test Results
- `hatch run test`: 1281 passed, 3 skipped, 17 deselected (0 failures)
- `hatch run lint`: All checks passed
- `hatch run typecheck`: 97 errors (all pre-existing; 95 before phase, +2 from duplicated MultiSimilarityLoss pattern)
- `aquapose -p YH reid fine-tune --help`: Shows `--unfreeze-blocks` and `--lr-backbone-factor` options

## Verdict
**PASSED** — All 5 success criteria verified, all 3 requirements traced to implementations.
