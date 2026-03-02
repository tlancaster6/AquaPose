# Phase 36: Training Wrappers - Context

**Gathered:** 2026-03-01
**Status:** Ready for planning

<domain>
## Phase Boundary

COCO-to-NDJSON segmentation data converter and YOLO-seg/pose training wrappers callable from CLI, following the existing `yolo_obb.py` pattern. No pipeline integration (Phase 37), no model architecture changes, no full training runs.

</domain>

<decisions>
## Implementation Decisions

### Seg data conversion
- Add `--mode seg` to existing `scripts/build_yolo_training_data.py` (not a new CLI subcommand — roadmap defers CLI formalization)
- Source format: standard COCO polygon JSON (`segmentation` field with polygon arrays)
- Training on **per-detection crops** (not full frames) — crops extracted around each OBB detection
- **All visible fish** in each crop get labeled (target + intruders), not just the target fish — enables instance separation at inference
- Multi-ring COCO polygons (one fish, multiple disjoint rings): **keep largest ring only**, drop small fragments
- Data converter produces NDJSON files + dataset YAML; training wrapper reads the YAML — **fully decoupled**, YAML is the contract

### Training wrappers
- **YOLO26 models** (not YOLO11 as roadmap originally stated) — `yolo26n-seg` and `yolo26n-pose` as defaults
- `--model` flag for variant selection (e.g., `yolo26s-seg`, `yolo26m-pose`)
- `--weights` flag for pretrained weight loading (transfer learning); no `--resume` support
- Ultralytics default augmentation — no custom augmentation flags exposed
- Keypoint definition (names, count, skeleton) read from dataset YAML, **not hardcoded** in wrapper
- Random 80/20 train/val split via `--val-split` flag (same as existing yolo_obb pattern)

### File structure
- Separate files: `yolo_seg.py` and `yolo_pose.py` alongside existing `yolo_obb.py`
- Existing `pose.py` (old custom trainer) removed in Phase 35; new file is `yolo_pose.py` to avoid confusion
- Fully decoupled from data converter — no shared utility code

### Output and validation
- Default training output directory: `~/aquapose/yolo/` (keeps artifacts out of repo), overridable via `--output-dir`
- Quality thresholds for "good enough": seg mAP > 0.5, pose mAP > 0.4
- Phase 36 scope is **smoke test only** (1-2 epochs to verify wrappers work); full training is a user activity

### Existing pose data
- `--mode pose` in `build_yolo_training_data.py` already produces NDJSON from COCO keypoint JSON — verify compatibility with YOLO26-pose format and use as-is

### Claude's Discretion
- Exact Ultralytics `.train()` parameter mapping
- Dataset YAML structure details (follows Ultralytics conventions)
- Error handling for malformed COCO annotations
- Test structure for smoke tests

</decisions>

<specifics>
## Specific Ideas

- "YOLO11 is pretty outdated — we should be using YOLO26 models by default"
- Training output at `~/aquapose/yolo/` — keeps training runs out of the code directory
- Per-detection crops with all visible fish labeled enables IoU-based instance matching in Phase 37

</specifics>

<deferred>
## Deferred Ideas

- CLI formalization of training data prep (`aquapose data convert-seg`) — deferred per roadmap
- Full training runs with quality tuning — user activity, not phase scope
- Resume from checkpoint (`--resume`) — not needed for initial wrappers

</deferred>

---

*Phase: 36-training-wrappers*
*Context gathered: 2026-03-01*
