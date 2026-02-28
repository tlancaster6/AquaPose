# Phase 31: Training Infrastructure - Context

**Gathered:** 2026-02-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a unified `aquapose train` CLI group and `src/aquapose/training/` package that centralizes all model training (U-Net segmentation, YOLO-OBB detection, pose/keypoint regression) behind consistent conventions. Replaces disconnected training scripts. Built early so model training can begin while pipeline integration proceeds in later phases.

</domain>

<decisions>
## Implementation Decisions

### CLI design
- CLI flags only — no YAML config files for training. All options as explicit flags (--lr, --batch-size, --epochs, --data-dir, --output-dir, --device, etc.)
- Epoch summary lines for progress: one line per epoch with epoch number, train loss, val loss, best metric, time elapsed. No progress bars. CI-friendly and grep-able.
- Shared `--val-split` flag across all subcommands for consistent validation split handling
- Metrics logged to both console AND a metrics file (CSV or JSON) in --output-dir for later analysis and plotting

### Package layout
- One module per model: `training/unet.py`, `training/yolo_obb.py`, `training/pose.py` + `training/common.py` for shared utilities
- Subcommand name is `pose` (not `keypoint` or `direct-pose`) — matches animal pose estimation convention and the `direct_pose` backend name
- Dataset classes (BinaryMaskDataset, CropDataset) move from `segmentation/` into `training/` (e.g. `training/datasets.py`)
- Rewrite training code from scratch using shared utilities, with old `segmentation/training.py` as reference — not a line-by-line move

### Checkpoints & weights
- Save best and last weights only — no full training state checkpoints
- No `--resume` flag — networks train fast enough that full resume is overkill. **Note: this diverges from TRAIN-02 and success criteria #2 in ROADMAP.md; requirements should be updated to remove --resume**
- `--backbone-weights` flag for pose training: loads U-Net encoder weights and freezes backbone. `--unfreeze` flag enables end-to-end fine-tuning
- Fixed best-metric per model type: U-Net = best val IoU, YOLO-OBB = best val mAP, Pose = best val keypoint error. Not configurable.

### Migration
- Breaking changes to weight format are allowed — no backward compatibility with existing saved weights required
- Delete `segmentation/training.py` at end of Phase 31 once `aquapose train unet` produces equivalent results
- Clean delete of dataset classes from `segmentation/` — no re-exports or backward compat shims
- Import boundary: `training/` must not import from `engine/` (enforced by pre-commit). Importing from `segmentation/` (for model architecture classes like UNet) and `calibration/` is allowed.

### Claude's Discretion
- Exact shared utility design in `training/common.py` (early stopping, metric tracking, etc.)
- Metrics file format choice (CSV vs JSON)
- Internal organization of dataset classes
- YOLO-OBB training integration approach (likely wrapping ultralytics)

</decisions>

<specifics>
## Specific Ideas

- "pose" is the canonical name for the keypoint regression network (U-Net encoder + regression head for anatomical landmarks)
- Three subcommands: `aquapose train unet`, `aquapose train yolo-obb`, `aquapose train pose`
- Use old `segmentation/training.py` as behavioral reference, then delete it

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 31-training-infrastructure*
*Context gathered: 2026-02-28*
