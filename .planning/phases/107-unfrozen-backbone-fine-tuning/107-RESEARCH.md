# Phase 107: Unfrozen Backbone Fine-Tuning - Research

**Researched:** 2026-03-25
**Domain:** PyTorch transfer learning — partial ViT/Swin unfreezing, metric learning end-to-end
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Support configurable unfreeze depth: last N transformer blocks of the ViT backbone
- Use differential learning rates: backbone LR should be 10-100x lower than head LR
- Keep the warmup + cosine annealing scheduler from Phase 106
- MultiSimilarityLoss + MultiSimilarityMiner (current) is fine as the loss function
- MPerClassSampler with samples_per_class=16 (current) is fine
- The cached-features fast path (frozen backbone) should still work when unfreeze_blocks=0
- When backbone is fine-tuned, re-embedding must run crops through the full fine-tuned model (backbone + head), not just apply a head to stale zero-shot features
- Save the full fine-tuned model (backbone state + head state) as the checkpoint, not just head weights
- The CLI `reid fine-tune` command should handle this transparently
- SwapDetector should use pre-computed fine-tuned embeddings from embeddings_finetuned.npz (Option a, simpler and preferred)

### Claude's Discretion
- Whether to save backbone + head as a single checkpoint or separate files
- How to structure the unfreezing (by block index, by layer name pattern, etc.)
- Whether to add gradient clipping for stability
- Image augmentation during end-to-end training (random crops, flips, color jitter)

### Deferred Ideas (OUT OF SCOPE)
- Full backbone fine-tuning (all blocks) — start with partial and see if it helps
- Knowledge distillation from a larger model
- Temporal/trajectory-based ReID methods (non-appearance)
</user_constraints>

---

## Summary

MegaDescriptor-T is a **SwinTransformer** (not ViT), loaded via timm. It has 4 stages with 2, 2, 6, 2 SwinTransformerBlock instances respectively — 12 blocks total — matching the CONTEXT.md description. Stages are accessed via `model.layers[i].blocks[j]`. The final LayerNorm is at `model.norm`.

Unfreezing the last N blocks works cleanly: freeze the entire backbone with `model.requires_grad_(False)`, then selectively re-enable grad on the last N blocks (enumerated in stage order) and on `model.norm`. When `unfreeze_blocks=0`, the existing cached-feature fast path applies without change. With `unfreeze_blocks=2`, approximately 51.5% of backbone parameters are trainable (14.2M of 27.5M). The last 2 blocks are in stage 3 and have 7.1M params each — these are the highest-level semantic feature blocks.

End-to-end training is straightforward and verified working: backbone + head forward pass, metric loss, backward, differential-LR Adam optimizer step. Measured per-batch time on GPU with batch=144, unfreeze_blocks=2: ~1.0s/batch, ~21.6s/epoch, ~18 minutes for 50 epochs. Training requires a new image-loading Dataset replacing `CachedFeatureDataset`; the new dataset must apply the same normalization as `FishEmbedder.embed_batch` (`(x/255 - 0.5) / 0.5`).

**Primary recommendation:** Implement end-to-end training as a parallel path alongside the existing cached-feature path, controlled by `unfreeze_blocks` in `ReidTrainingConfig`. When `unfreeze_blocks > 0`, save a combined checkpoint `{backbone_state_dict, head_state_dict, config}` (~111 MB); re-embed using the fine-tuned backbone directly; pass `embeddings_finetuned.npz` to SwapDetector unchanged (no SwapDetector code changes needed).

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| timm | existing | SwinTransformer backbone loading | Already in use; `model.layers` API confirmed |
| torch | existing | Partial freeze, differential LR, checkpointing | Native PyTorch pattern |
| pytorch-metric-learning | existing | MultiSimilarityLoss + Miner | Already wired in training loop |

### No New Dependencies Required

All needed functionality is available in the existing stack. No new packages needed.

---

## Architecture Patterns

### Swin Block Structure (VERIFIED)

```
model.layers[0]  (SwinTransformerStage)  — 2 blocks, 112K params each
model.layers[1]  (SwinTransformerStage)  — 2 blocks, 446K params each
model.layers[2]  (SwinTransformerStage)  — 6 blocks, 1.78M params each
model.layers[3]  (SwinTransformerStage)  — 2 blocks, 7.09M params each  ← last N=2 default
model.norm       (LayerNorm)                                             ← always unfreeze with blocks
```

Total backbone: 27.5M parameters. Unfreezing last 2 blocks + norm: ~14.2M trainable (51.5%).

### Pattern 1: Selective Backbone Unfreezing

```python
# Source: verified via live model inspection
def unfreeze_last_n_blocks(backbone: nn.Module, n: int) -> None:
    """Freeze all backbone params, then unfreeze last N Swin blocks + final norm.

    Args:
        backbone: SwinTransformer from timm (MegaDescriptor-T).
        n: Number of blocks to unfreeze from the end. 0 = fully frozen.
    """
    backbone.requires_grad_(False)
    if n <= 0:
        return
    # Collect all blocks in stage order
    all_blocks: list[nn.Module] = []
    for layer in backbone.layers:
        if hasattr(layer, "blocks"):
            all_blocks.extend(layer.blocks)
    # Unfreeze last N blocks
    for blk in all_blocks[-n:]:
        blk.requires_grad_(True)
    # Always unfreeze final LayerNorm with unfrozen blocks
    backbone.norm.requires_grad_(True)
```

### Pattern 2: Differential Learning Rate Optimizer

```python
# Source: verified via live training step test
def build_optimizer(
    backbone: nn.Module,
    head: nn.Module,
    lr_head: float,
    lr_backbone_factor: float = 0.1,
) -> torch.optim.Optimizer:
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    return torch.optim.Adam([
        {"params": backbone_params, "lr": lr_head * lr_backbone_factor},
        {"params": head_params, "lr": lr_head},
    ])
```

### Pattern 3: Image Crop Dataset (replaces CachedFeatureDataset for end-to-end training)

The existing `CachedFeatureDataset` returns pre-computed backbone features. For end-to-end training, we need a dataset that loads raw crop images. Key requirement: use **identical normalization** to `FishEmbedder.embed_batch`:

```python
# Normalization in FishEmbedder.embed_batch (must match exactly):
arr = resized.astype(np.float32) / 255.0
arr = (arr - 0.5) / 0.5  # maps [0,1] -> [-1, 1]
tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # (H,W,C) -> (C,H,W)
```

The `ImageCropDataset` must replicate this preprocessing (BGR->RGB convert + resize + normalize).

### Pattern 4: Combined Checkpoint Save/Load

```python
# Save (full fine-tuned model, ~111 MB)
checkpoint = {
    "backbone_state_dict": backbone.state_dict(),
    "head_state_dict": head.state_dict(),
    "config": {
        "unfreeze_blocks": config.unfreeze_blocks,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
    },
}
torch.save(checkpoint, output_dir / "best_reid_model.pt")

# Load for re-embedding
ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
backbone.load_state_dict(ckpt["backbone_state_dict"])
head.load_state_dict(ckpt["head_state_dict"])
```

### Pattern 5: Re-embedding with Fine-Tuned Backbone

When `unfreeze_blocks > 0`, the `fine-tune` CLI step must run crops through the full backbone+head stack instead of applying a projection head to stale zero-shot features. The `EmbedRunner` infrastructure already exists for extracting crops from the run directory — re-use it with the fine-tuned backbone loaded:

```python
# In fine_tune_cmd, after training gate passes:
if unfreeze_blocks > 0:
    # Re-embed using fine-tuned backbone + head (full forward pass)
    # Use EmbedRunner with a custom embedder that wraps backbone+head
    # Write embeddings_finetuned.npz with same schema as embeddings.npz
    pass
else:
    # Existing path: apply projection head to zero-shot embeddings
    # zs_embeddings -> head(zs_embeddings)
    pass
```

### Pattern 6: Frozen Path Backward Compatibility (unfreeze_blocks=0)

When `unfreeze_blocks=0`, no changes to the existing training flow:
- `build_feature_cache` builds/loads `backbone_cache.npz`
- `CachedFeatureDataset` loads features from cache
- `train_reid_head` trains projection head on cached features
- `best_reid_model.pt` saves **only** `head_state_dict` (existing format)
- Re-embedding applies projection head to stale zero-shot embeddings (existing behavior)

This backward compatibility is essential — the planner should ensure the frozen path continues to work unchanged.

### Recommended Project Structure Changes

```
src/aquapose/training/reid_training.py    # Add: ReidTrainingConfig.unfreeze_blocks
                                          # Add: ImageCropDataset
                                          # Add: unfreeze_last_n_blocks()
                                          # Add: train_reid_end_to_end()
                                          # Keep: all existing functions unchanged
src/aquapose/core/reid/cli.py             # Modify: fine_tune_cmd for backbone path
                                          # Modify: embed_cmd to support finetuned backbone
```

### Anti-Patterns to Avoid

- **Changing normalization:** Do NOT use timm's default ImageNet normalization (`mean=[0.485, 0.456, 0.406]`) during training. It differs from `FishEmbedder`'s `(x-0.5)/0.5`. Mixing them would make the trained backbone incompatible with inference.
- **Rebuilding backbone in EmbedRunner:** Do NOT create a new `timm.create_model()` call to load the fine-tuned model from scratch and re-apply weights. Instead, load the checkpoint and restore state_dict onto a fresh model instance.
- **Using backbone_cache.npz for end-to-end path:** When `unfreeze_blocks > 0`, the backbone changes every epoch. The feature cache is stale after each backward pass and must NOT be used.
- **Saving only head weights when backbone was unfrozen:** The `best_reid_model.pt` must include `backbone_state_dict` when `unfreeze_blocks > 0`. The re-embedding step in the CLI will break if backbone state is missing.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Swin block enumeration | Custom layer traversal | `for layer in model.layers: all_blocks.extend(layer.blocks)` | Verified pattern, handles stage structure correctly |
| Differential LR | Separate optimizers | PyTorch `param_groups` in one Adam | Cleaner, scheduler applies correctly |
| Image transforms | Custom pipeline | Replicate existing `FishEmbedder.embed_batch` normalization exactly | Consistency with inference is critical |
| Crop loading | New image loader | `cv2.imread` + existing manifest.json format | Already used in `build_feature_cache` |

---

## Common Pitfalls

### Pitfall 1: Normalization Mismatch Between Training and Inference
**What goes wrong:** Training dataset uses different pixel normalization than `FishEmbedder.embed_batch`. The fine-tuned backbone learns features for the wrong input distribution. Embeddings produced at inference time are garbage.
**Why it happens:** timm's `data_config` reports ImageNet per-channel normalization, which is different from the `(x/255 - 0.5) / 0.5` currently used in `FishEmbedder`. Naively using `timm.data.create_transform()` would apply the wrong normalization.
**How to avoid:** In `ImageCropDataset.__getitem__`, replicate the exact preprocessing from `FishEmbedder.embed_batch`: BGR->RGB, resize to 224x224, cast to float32/255, then `(arr - 0.5) / 0.5`.
**Warning signs:** Val AUC is worse than zero-shot baseline after end-to-end training.

### Pitfall 2: backbone_cache.npz Staleness with Unfrozen Backbone
**What goes wrong:** The training loop uses cached features (built once before training) but the backbone parameters are being updated. After epoch 1, the cached features are stale — the backbone has changed but the cache hasn't. The loss is computed against stale features, not current ones.
**Why it happens:** The frozen path uses a pre-built cache as an optimization. This optimization is invalid when the backbone is trainable.
**How to avoid:** When `unfreeze_blocks > 0`, use `ImageCropDataset` (loads raw images) instead of `CachedFeatureDataset`. Never call `build_feature_cache` for the unfrozen training path.

### Pitfall 3: Re-embedding Stale Zero-Shot Features When Backbone Was Fine-Tuned
**What goes wrong:** The CLI `fine-tune` command's re-embedding step applies the projection head to `embeddings.npz` (zero-shot backbone features). But if the backbone was fine-tuned, the projection head was trained on backbone outputs that are different from those zero-shot features. The resulting `embeddings_finetuned.npz` is incorrect.
**Why it happens:** The existing re-embedding code in `fine_tune_cmd` loads `embeddings.npz` and applies the head. This is correct for `unfreeze_blocks=0` (frozen backbone) but wrong otherwise.
**How to avoid:** Branch on `config.unfreeze_blocks > 0`. For the unfrozen path, re-embed using `EmbedRunner` with the fine-tuned backbone loaded. For the frozen path, use the existing projection-head-on-cached-features approach.

### Pitfall 4: BatchNorm in eval() During Training
**What goes wrong:** The `ProjectionHead` contains `BatchNorm1d`. If `head.eval()` is called before the training loop starts and never switched back to `head.train()`, BatchNorm uses running statistics instead of batch statistics, degrading training convergence.
**Why it happens:** Common copy-paste error from eval code.
**How to avoid:** Call `backbone.train()` (partial — only unfrozen blocks benefit) and `head.train()` at the start of each training epoch. Call `backbone.eval()` + `head.eval()` only during validation.

### Pitfall 5: Gradient Accumulation in MPerClassSampler Batches
**What goes wrong:** With `unfreeze_blocks=2` and batch_size=144, each batch is 1.0s on GPU. The training loop is 18 minutes for 50 epochs. If gradient clipping is not applied, the large parameter count (14M unfrozen backbone params) can cause loss spikes.
**Why it happens:** End-to-end training with large pretrained models has different gradient scale than head-only training.
**How to avoid:** Add `torch.nn.utils.clip_grad_norm_` with max_norm=1.0 before `optimizer.step()`. This is Claude's discretion per CONTEXT.md.

---

## Code Examples

### Complete Unfreezing + Training Step

```python
# Source: verified via live execution on MegaDescriptor-T-224
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", num_classes=0, pretrained=True)

# Freeze all backbone
backbone.requires_grad_(False)

# Unfreeze last N=2 blocks + norm
all_blocks = []
for layer in backbone.layers:
    if hasattr(layer, "blocks"):
        all_blocks.extend(layer.blocks)
for blk in all_blocks[-2:]:
    blk.requires_grad_(True)
backbone.norm.requires_grad_(True)

# Verified: trainable=14,185,392 / 27,519,354 (51.5%)
# Frozen params confirmed to have no grad after backward

# Differential LR optimizer
backbone_params = [p for p in backbone.parameters() if p.requires_grad]
head_params = list(head.parameters())
optimizer = torch.optim.Adam([
    {"params": backbone_params, "lr": 3e-5},  # 10x lower than head
    {"params": head_params, "lr": 3e-4},
])
```

### ImageCropDataset Normalization (must match FishEmbedder.embed_batch)

```python
# Source: FishEmbedder.embed_batch in src/aquapose/core/reid/embedder.py
import cv2
import numpy as np
import torch

class ImageCropDataset(Dataset):
    def __init__(self, crop_paths, labels, group_ids, crop_size=224):
        self._paths = crop_paths
        self._labels = torch.from_numpy(labels).long()
        self._group_ids = group_ids
        self._crop_size = crop_size

    def __getitem__(self, idx):
        img = cv2.imread(str(self._paths[idx]))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._crop_size, self._crop_size),
                             interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5   # MUST match FishEmbedder normalization
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # (C, H, W)
        return tensor, self._labels[idx]

    def get_labels(self):
        return self._labels.tolist()
```

### Combined Checkpoint Save

```python
# Source: verified pattern (~111 MB output)
checkpoint = {
    "backbone_state_dict": backbone.state_dict(),
    "head_state_dict": head.state_dict(),
    "config": {
        "unfreeze_blocks": config.unfreeze_blocks,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "backbone_dim": config.backbone_dim,
    },
}
torch.save(checkpoint, output_dir / "best_reid_model.pt")
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Cached backbone features only | End-to-end with partial unfreezing | Phase 107 | Enables backbone to learn fish-specific features |
| Head-only checkpoint (`head_state_dict`) | Combined checkpoint (`backbone_state_dict` + `head_state_dict`) | Phase 107 | Re-embedding uses fine-tuned backbone |
| Apply head to zero-shot embeddings | Run crops through fine-tuned backbone+head | Phase 107 | Embeddings reflect fine-tuned features |

**Backward compatibility:** The frozen path (`unfreeze_blocks=0`) must continue producing `best_reid_model.pt` with only `head_state_dict` to avoid breaking existing checkpoint consumers. OR: always save the full format but detect at load time whether `backbone_state_dict` is present.

---

## Open Questions

1. **Checkpoint format: single file vs separate files**
   - What we know: Single combined `.pt` is 111 MB, cleanly loadable, includes backbone+head+config
   - What's unclear: Whether existing consumers of `best_reid_model.pt` (SwapDetector, `fine_tune_cmd` re-embedding) need to detect frozen vs unfrozen format
   - Recommendation (Claude's discretion): Use a single file always, with `unfreeze_blocks` key in the config dict. Load code checks: if `"backbone_state_dict"` present, load backbone; otherwise use default timm weights. This handles both frozen and unfrozen paths uniformly.

2. **Gradient clipping**
   - What we know: End-to-end training with 14M+ unfrozen params; MultiSimilarityLoss; loss magnitude was stable in smoke tests
   - What's unclear: Whether gradients spike in practice with real fish crop data
   - Recommendation (Claude's discretion): Add `clip_grad_norm_` with max_norm=1.0 as a safety measure, controlled by an optional `gradient_clip_val: float | None = 1.0` in `ReidTrainingConfig`

3. **Data augmentation during end-to-end training**
   - What we know: Cached-feature path has no augmentation; end-to-end training loads raw images each epoch
   - What's unclear: Whether random horizontal flips or color jitter help or hurt (fish have bilateral symmetry; color consistency is part of identity)
   - Recommendation (Claude's discretion): Start with no augmentation in Phase 107. The backbone is already trained on diverse data; too much augmentation may destroy the fine-grained color/texture cues that distinguish individual fish. If AUC doesn't improve, augmentation can be tried in a follow-up.

4. **Scheduler: warmup_epochs and cosine schedule with two param groups**
   - What we know: Current cosine scheduler is on `head_optimizer` only; with differential LR, we need schedulers for both param groups
   - What's unclear: Whether a single Adam with two param groups needs one scheduler or two
   - Recommendation: One `CosineAnnealingLR` on the combined optimizer affects all param groups proportionally. The warmup scaling (`warmup_factor * config.lr_head`) must also scale the backbone LR by the same factor (`warmup_factor * config.lr_backbone`). Iterate over all param groups in the warmup loop.

---

## Validation Architecture

> workflow.nyquist_validation not set in config.json — skipping this section.

---

## Sources

### Primary (HIGH confidence)
- Live model inspection: `timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", ...)` — confirmed SwinTransformer, 4 stages, 12 blocks, `model.layers` API
- Live training verification: end-to-end forward+backward pass confirmed working, frozen params confirmed grad-free after backward
- Live timing measurement: 1.0s/batch on GPU with batch=144, unfreeze_blocks=2
- Live checkpoint sizing: combined checkpoint = 111 MB
- Source: `src/aquapose/core/reid/embedder.py` — FishEmbedder normalization convention `(x/255-0.5)/0.5`
- Source: `src/aquapose/training/reid_training.py` — existing training architecture (CachedFeatureDataset, train_reid_head, warmup+cosine scheduler)
- Source: `src/aquapose/core/reid/cli.py` — fine_tune_cmd structure and re-embedding flow

### Secondary (MEDIUM confidence)
- PyTorch docs: `requires_grad_(False)` freeze pattern is standard; `param_groups` differential LR is documented
- timm docs: `model.layers` for SwinTransformer stage access

### Tertiary (LOW confidence)
- None — all critical claims verified by direct code execution

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all verified by live execution
- Architecture: HIGH — Swin block structure confirmed, patterns tested end-to-end
- Pitfalls: HIGH — normalization mismatch is a real risk (verified different conventions exist), cache staleness is architectural certainty

**Research date:** 2026-03-25
**Valid until:** 2026-04-25 (stable libraries — timm Swin API, PyTorch optimizer API)
