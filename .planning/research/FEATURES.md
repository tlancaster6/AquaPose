# Feature Research

**Domain:** 3D fish pose estimation — swappable detection/midline backends, training infrastructure, config cleanup (v2.2 Backends milestone)
**Researched:** 2026-02-28
**Confidence:** HIGH (existing codebase fully understood; OBB/keypoint approaches verified against official docs and primary literature)

---

## Context: What's Already Built

This is a subsequent-milestone research document. The following are fully shipped and NOT re-researched here:

- YOLO v8n standard bbox detection (`YOLODetector` in `segmentation/detector.py`)
- U-Net segmentation (MobileNetV3-Small encoder, IoU 0.623) + skeletonize-based midline
- `Midline2D` struct: 15 arc-length samples, (N,2) points, half-widths, no confidence field
- Multi-view RANSAC triangulation + view-angle weighting in `reconstruction/triangulation.py`
- B-spline fitting (7 control points, cubic, knots hardcoded)
- `N_SAMPLE_POINTS = 15` as module-level constant in `triangulation.py`
- Frozen dataclass config + YAML + CLI overrides
- Click-based CLI (`aquapose run`, `aquapose init-config`) in `cli.py`
- 4 training scripts in `scripts/` (build_training_data, generate_golden_data, organize_yolo_dataset, train_yolo)

The milestone adds: YOLO-OBB detection backend, keypoint-based midline backend (U-Net encoder + regression head), unified training CLI infrastructure, and config system cleanup (N_SAMPLE_POINTS, device, init-config UX).

---

## Context: Who Are the "Users"?

AquaPose users for this milestone are the researchers themselves running the pipeline and training models. "Table stakes" means: missing this makes the new backend unusable or incompatible with the existing pipeline. "Differentiators" are the features that make the new backends qualitatively better than the current approach.

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features that anyone using the new backends will assume exist. Missing these makes the milestone feel incomplete or broken.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| OBB detection outputs angle that flows into downstream | Users expect detected rotation angle to be extractable and usable without manual conversion | LOW | `result.obb.xywhr` gives (cx, cy, w, h, angle_radians); `result.obb.xyxyxyxy` gives 4 corner points; both are native ultralytics outputs |
| Affine-crop extraction from OBB | Any OBB pipeline requires rotating the image patch to be axis-aligned before feeding to U-Net or keypoint head | MEDIUM | `cv2.getRotationMatrix2D(center, angle_deg, 1.0)` + `cv2.warpAffine`; canvas must be expanded to avoid cutoff at image edges; transform is invertible for back-projecting keypoints |
| YOLO-OBB integrates with existing `make_detector()` factory | Users expect to change detector via config (`detector_kind: yolo_obb`), not code changes | LOW | Add `YOLOOBBDetector` class and `"yolo_obb"` branch to `make_detector()`; `Detection` dataclass gains optional `angle: float \| None = None` |
| Keypoint head reuses existing U-Net encoder | Users expect the new midline backend to benefit from trained encoder weights, not start from scratch | MEDIUM | Shared MobileNetV3-Small encoder → two heads: existing segmentation decoder (unchanged) + new keypoint regression head; head weights trained separately |
| Per-point confidence from keypoint head | Downstream reconstruction expects confidences; missing them means falling back to uniform weights silently | MEDIUM | Keypoint head outputs (N_points, 3): x, y, confidence; x/y use sigmoid × crop_size; confidence uses sigmoid |
| `Midline2D` carries optional confidence array | Reconstruction backend and observers need a consistent interface; ad-hoc confidence passing is fragile | LOW | Add `confidences: np.ndarray \| None = None` to existing `Midline2D` dataclass; `None` = uniform weights; no downstream breakage since existing code ignores new field |
| `aquapose train` CLI subgroup replaces ad-hoc scripts | Users expect installable, discoverable commands with `--help`; bare `python scripts/train_yolo.py` is not ergonomic | MEDIUM | Click group under existing `cli` group: `aquapose train unet`, `aquapose train yolo-obb`, `aquapose train keypoint`; registered via existing `aquapose` entry point in pyproject.toml |
| `N_SAMPLE_POINTS` configurable | Currently hardcoded to 15 in `triangulation.py`; users cannot tune it without code edits | LOW | Move to `ReconstructionConfig` or top-level `PipelineConfig` dataclass field with default 15; update all references |
| Device propagation through config | Users set `device: cuda` once in YAML and expect it to flow everywhere; currently some sub-modules require explicit device kwarg | LOW | Single `device: str = "cuda"` field at top-level config, passed through `build_stages()` to all sub-components |

### Differentiators (Competitive Advantage)

Features that make the new backends qualitatively better than the current pipeline.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| YOLO-OBB provides rotation angle directly | OBB detection eliminates the orientation ambiguity that currently requires BFS heuristics; tighter rotation-aligned crops for slender fish reduce U-Net and keypoint head confusion | MEDIUM | OBB angle is constrained to [0°, 90°) by ultralytics — the 180° head/tail ambiguity still requires downstream orientation logic (tracking continuity), but the basic "which way is the fish oriented in the image" is resolved |
| Keypoint regression head bypasses skeletonization | Direct regression from encoder features produces ordered, smooth midline points; skeletonization on IoU 0.623 masks produces noisy, sometimes-disconnected skeletons that require BFS pruning and still produce artifacts | HIGH | At 128×128 input, direct regression is preferred over heatmaps (see Technical Notes); outputs 15 ordered points directly in crop coordinates; transforms back to full-frame via affine inverse |
| Confidence-weighted triangulation | Per-point confidence from keypoint head flows into DLT as per-observation weights; noisy keypoints are suppressed rather than discarding entire views; qualitatively improves 3D midlines on partially-occluded fish | MEDIUM | Weighted DLT: each view's rows in the A matrix are scaled by sqrt(confidence) before SVD; existing RANSAC loop structure unchanged; fallback to uniform weights when `Midline2D.confidences is None` |
| Partial midline handling via confidence gating | Fish occluded at head or tail produces a partial midline; with confidence gating, the high-confidence portion contributes to 3D reconstruction rather than causing full-view rejection | MEDIUM | Gate: exclude per-point entries with `conf < threshold` from DLT rows; require at least `MIN_BODY_POINTS` valid rows across all views; confidence threshold configurable |
| OBB visualization overlay in diagnostic mode | Rotated polygon overlaid on diagnostic frames makes detection quality immediately auditable; rotated boxes visually encode fish orientation in a way axis-aligned boxes cannot | LOW | Observer enhancement: `xyxyxyxy` 4-point polygon → `cv2.polylines` in overlay observer; angle label optional |
| Unified `aquapose train` CLI with consistent conventions | Replaces 4 disconnected scripts with a documented, discoverable interface; enforces consistent `--output-dir`, `--epochs`, `--device`, `--resume` conventions across all training commands | MEDIUM | `src/aquapose/training/` module; shared `common.py` for checkpoint saving, logging, LR scheduling; subcommands are thin wrappers |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Heatmap-based keypoint regression | Standard approach in human pose estimation (HRNet, ViTPose, SimpleBaseline); very familiar to the ML community | At 128×128 input the heatmap is 32×32 after the standard 1/4 downscale — quantization error is ~4 px per point; a 10cm cichlid at pipeline crop size spans ~50px, so 4px quantization is ~8% body-length error, which is unacceptable; heatmap approaches are significantly better than regression only at higher resolutions (≥256×256) | Direct regression: Linear(encoder_channels, N_points × 3) → reshape to (N, 3); or integral regression if direct regression accuracy is insufficient |
| Separate keypoint encoder trained from scratch | Clean separation of concerns; avoids "touching" the segmentation model | Discards the U-Net encoder weights already trained on fish crops (IoU 0.623); requires more labeled keypoint data and more compute; the encoder already knows what fish look like | Shared encoder → freeze encoder weights; train only regression head; optionally fine-tune top encoder layers with lower LR |
| Joint end-to-end segmentation + keypoint loss | Theoretically optimal multi-task learning | Couples two tasks with different training data and quality ceilings: segmentation uses SAM2 pseudo-labels (abundant), keypoints need separate annotation (scarce); joint training requires balancing loss scales and can degrade the segmentation head that the existing pipeline depends on | Stage-wise: train segmentation first (already done), then freeze encoder and train keypoint head; optionally add multi-task fine-tuning as P3 enhancement |
| Replacing frozen dataclasses with Pydantic for config | Pydantic provides runtime validation and IDE type narrowing | Explicitly out of scope per PROJECT.md; frozen dataclasses already shipped in v2.0 and all existing code uses them; migration would touch every config path in the codebase with no scientific benefit | Validate in `load_config()` with explicit `assert` or `ValueError` checks; document valid ranges in docstrings |
| Separate `aquapose-train` binary entry point | Feels cleaner as a standalone tool | Splits help system; users can't discover training via `aquapose --help`; requires an additional entry point declaration in pyproject.toml | Subgroup: `aquapose train yolo-obb` — stays under the unified `aquapose` CLI group, fully discoverable |
| Real-time OBB inference on all 13 cameras | Seems like a natural next step once OBB works | PROJECT.md is explicit: batch mode only, 5-30 min clips; 13-camera real-time at 30fps exceeds single-GPU memory and introduces synchronization complexity that is orthogonal to the research goal | Batch mode only; document constraint in CLI help text |

---

## Feature Dependencies

```
YOLO-OBB Detection
    └──requires──> Detection.angle field (optional float) in Detection dataclass
    └──requires──> make_detector("yolo_obb", ...) factory branch
    └──enables──>  Affine crop extraction
                       └──enables──> Rotation-aligned crops for keypoint head training
                       └──enables──> OBB overlay visualization

Keypoint Regression Head
    └──requires──> Shared U-Net encoder (existing, already trained on fish crops)
    └──requires──> Affine crop extraction (rotation-aligned input)
    └──outputs──>  (N, 3) tensor: (x, y, confidence) in crop coordinates
                       └──requires──> Back-projection via affine inverse → full-frame coords
                       └──feeds──>    Midline2D with confidences field populated

Midline2D.confidences field
    └──requires──> Midline2D dataclass updated (backward-compatible: None = uniform)
    └──required by──> Confidence-weighted triangulation
    └──required by──> Partial midline handling

Confidence-Weighted Triangulation
    └──requires──> Midline2D.confidences field
    └──enhances──> Existing RANSAC triangulation (does not replace; adds weighted DLT inner path)
    └──enables──>  Partial midline handling (low-conf points excluded from DLT rows)

Training CLI (aquapose train)
    └──requires──> src/aquapose/training/ module with unet.py, yolo_obb.py, keypoint.py, common.py
    └──requires──> aquapose CLI group registration (already in pyproject.toml [project.scripts])
    └──subsumes──> scripts/train_yolo.py logic → training/yolo_obb.py
    └──subsumes──> segmentation/training.py train() → training/unet.py
    └──independent of──> all detection/midline backend features (can land in any phase)

Config Cleanup
    └──N_SAMPLE_POINTS──> remove module constant in triangulation.py, add to ReconstructionConfig
    └──device field──> add to top-level PipelineConfig, propagate in build_stages()
    └──independent of──> all other v2.2 features (no ordering constraint)
```

### Dependency Notes

- **Keypoint head requires affine crops:** The existing pipeline feeds axis-aligned bbox crops to U-Net. The keypoint head needs rotation-aligned crops so that point ordering is consistent (head at left, tail at right). Without affine crops, the 15 output keypoints have ambiguous orientation depending on fish angle in frame.
- **OBB detection is independent of keypoint head:** YOLO-OBB can be used with the existing skeletonize-based midline backend — the rotation angle simply improves crop alignment. Keypoint head can also be trained without OBB (using existing bbox crops with no rotation). They work well together but are not coupled.
- **Confidence-weighted triangulation requires Midline2D.confidences:** The field must exist before the triangulation path can use it. This is the only hard cross-feature dependency in the milestone.
- **Training CLI is fully independent:** It does not depend on any other v2.2 feature and can be developed or landed in any phase.
- **Config cleanup is fully independent:** No other v2.2 feature depends on it (existing code uses the hardcoded constant), but it should land early to avoid config debt accumulating.

---

## MVP Definition (for v2.2 Backends milestone)

### Launch With

- [ ] `YOLOOBBDetector` class + `make_detector("yolo_obb", ...)` — drop-in OBB backend; `Detection.angle` field added
- [ ] `extract_obb_crop(frame, xywhr)` utility in `segmentation/crop.py` — returns `(crop, M_inv)` where M_inv back-projects crop keypoints to full-frame
- [ ] `UNetKeypointHead` — attaches to frozen/fine-tuned MobileNetV3-Small encoder; outputs (N_points, 3) in crop coordinates
- [ ] `Midline2D.confidences: np.ndarray | None = None` — backward-compatible field; None = uniform weights
- [ ] Confidence-weighted DLT path in `triangulation.py` — weighted when confidences present; falls back to existing uniform-weight path when None
- [ ] `aquapose train` Click group with `unet`, `yolo-obb`, `keypoint` subcommands — replaces scripts/; consistent CLI conventions
- [ ] `src/aquapose/training/` module — shared `common.py`, `unet.py`, `yolo_obb.py`, `keypoint.py`
- [ ] `N_SAMPLE_POINTS` moved to config field — remove module-level constant, add `n_sample_points: int = 15` to `ReconstructionConfig`
- [ ] `device: str = "cuda"` at top-level config, propagated through `build_stages()`
- [ ] OBB polygon overlay in diagnostic observer

### Add After Validation

- [ ] Partial midline handling with confidence gating — defer until keypoint head quality is known; requires threshold tuning
- [ ] Fine-tune encoder top layers with keypoint head (currently freeze encoder entirely) — only if regression accuracy is insufficient with frozen encoder
- [ ] `init-config` UX improvements (YAML schema comments, interactive prompts) — cosmetic, low priority

### Future Consideration (v2.3+)

- [ ] Keypoint pseudo-label generation — auto-label from skeletonizer output to bootstrap keypoint training data without manual annotation
- [ ] OBB fine-tuning on fish-specific dataset (currently DOTA pretrained weights via ultralytics)
- [ ] Confidence calibration (temperature scaling on keypoint head outputs)
- [ ] Integral regression alternative to direct regression — viable if direct regression accuracy is insufficient at 128×128
- [ ] Multi-task joint loss (segmentation + keypoint) — only after stage-wise training is validated

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| `Midline2D.confidences` field | HIGH | LOW | P1 |
| `N_SAMPLE_POINTS` in config | MEDIUM | LOW | P1 |
| Device propagation via config | MEDIUM | LOW | P1 |
| `YOLOOBBDetector` + `make_detector` | HIGH | LOW | P1 |
| `extract_obb_crop()` affine utility | HIGH | MEDIUM | P1 |
| Confidence-weighted DLT | HIGH | MEDIUM | P1 |
| `aquapose train` CLI group | MEDIUM | MEDIUM | P1 |
| `src/aquapose/training/` module | MEDIUM | MEDIUM | P1 |
| `UNetKeypointHead` arch + training | HIGH | HIGH | P1 |
| OBB polygon overlay | LOW | LOW | P2 |
| Partial midline confidence gating | MEDIUM | MEDIUM | P2 |
| Fine-tune encoder top layers | LOW | MEDIUM | P3 |
| `init-config` UX improvements | LOW | LOW | P3 |
| Multi-task joint loss | LOW | HIGH | P3 |

**Priority key:**
- P1: Ship in v2.2
- P2: Add when core is validated
- P3: Future milestone

---

## Technical Implementation Notes

### YOLO-OBB: API and Angle Convention

```python
# ultralytics result.obb fields (HIGH confidence — verified against official docs):
xywhr = result.obb.xywhr      # Tensor (N, 5): cx, cy, w, h, angle_radians
xyxyxyxy = result.obb.xyxyxyxy  # Tensor (N, 4, 2): 4 corner points in pixel coords
conf = result.obb.conf         # Tensor (N,): per-detection confidence
```

- Angle is in radians, constrained to [0, pi/2) by ultralytics. Angles of 90° or greater are not supported.
- The 180° head/tail ambiguity is NOT resolved by OBB — a fish at heading 45° and one at heading 225° look identical to the OBB head. Downstream orientation logic (tracking continuity or BFS direction) is still required.
- OBB models use the `-obb` suffix: `yolov8n-obb.pt`, pretrained on DOTA v1.

### Affine Crop Extraction Pattern

Standard pattern (OpenCV-based):

```python
cx, cy, w, h, angle_rad = xywhr  # from result.obb.xywhr[i]
angle_deg = np.degrees(angle_rad)
# Build 2x3 rotation matrix centered on OBB center
M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale=1.0)
# Warp entire frame (canvas stays same size; objects near edges may clip)
rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
# Axis-aligned crop of (w, h) centered at (cx, cy) in the rotated frame
x1 = int(cx - w / 2)
y1 = int(cy - h / 2)
crop = rotated_frame[y1:y1+int(h), x1:x1+int(w)]
```

To back-project keypoints from crop coordinates to full-frame pixel coordinates:
```python
# M is 2x3; extend to 3x3 for invertibility
M_full = np.vstack([M, [0, 0, 1]])
M_inv = np.linalg.inv(M_full)[:2]  # back to 2x3
# Apply to keypoints (N, 2):
kps_homogeneous = np.hstack([kps + crop_origin, np.ones((N, 1))])  # add crop offset first
kps_full_frame = (M_inv @ kps_homogeneous.T).T
```

### Keypoint Regression Head: Direct Regression (Recommended)

Three approaches exist; direct regression is recommended for this project:

| Approach | Output | Advantage | Disadvantage |
|----------|--------|-----------|--------------|
| **Direct regression** (recommended) | `Linear(C, N×3)` → (N, 3) | Simple, fast, no quantization error | Requires ordering constraint in training (head-first) |
| Heatmap regression | (N, H/4, W/4) per keypoint | Strong spatial context | 32×32 heatmap at 128px input → 4px quantization; unacceptable for ~50px fish |
| Integral regression | Softmax over feature map, expectation = coord | No quantization, differentiable, uses spatial context | More complex; multi-modal heatmaps cause coordinate averaging artifacts |

For direct regression at 128×128 input:
```python
class KeypointHead(nn.Module):
    def __init__(self, in_channels: int, n_points: int = 15):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, n_points * 3)
        self.n_points = n_points

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, C, H, W) — encoder output
        x = self.pool(features).flatten(1)  # (B, C)
        x = self.fc(x)                      # (B, N*3)
        x = x.view(-1, self.n_points, 3)    # (B, N, 3)
        # x[:, :, :2] = sigmoid * crop_size → x, y in [0, crop_size]
        # x[:, :, 2] = sigmoid → confidence in [0, 1]
        return x
```

### Confidence-Weighted DLT

In the existing `triangulation.py`, each pair of views contributes two rows to the DLT matrix A. The weighted DLT pattern scales rows by per-point confidence before SVD:

```python
# Pseudocode — weighted DLT for one point across views:
# For view i, point j with confidence c_ij:
#   rows contributed by view i get multiplied by sqrt(c_ij)
# This is the maximum-likelihood DLT when errors are iid Gaussian with sigma ~ 1/c
A_weighted = A * np.sqrt(confidences[:, np.newaxis])  # broadcast over row dim
_, _, Vt = np.linalg.svd(A_weighted)
X = Vt[-1]  # last right singular vector = triangulated point
```

The RANSAC loop structure, inlier threshold, and view-angle weighting are unchanged. Confidence weighting is an inner-loop change only.

### Training CLI Architecture

Recommended structure:
```
src/aquapose/training/
    __init__.py       # exports: train_unet, train_yolo_obb, train_keypoint
    common.py         # shared: save_checkpoint(), load_checkpoint(), get_lr_scheduler()
    unet.py           # train_unet() — migrated from segmentation/training.py
    yolo_obb.py       # train_yolo_obb() — replaces scripts/train_yolo.py logic
    keypoint.py       # train_keypoint_head() — new
```

CLI registration — no new entry point needed; add a subgroup to the existing `cli` group in `cli.py`:

```python
@cli.group()
def train() -> None:
    """Train AquaPose model components."""

@train.command("unet")
@click.option("--data-dir", required=True, ...)
@click.option("--output-dir", required=True, ...)
@click.option("--epochs", default=100, ...)
@click.option("--device", default="cuda", ...)
def train_unet_cmd(...) -> None:
    """Train the U-Net fish segmentation model."""
    ...

@train.command("yolo-obb")
...

@train.command("keypoint")
...
```

The existing `aquapose` entry point in `pyproject.toml [project.scripts]` already covers all subcommands — no new entry point needed.

---

## Competitor Feature Analysis

| Feature | DeepLabCut + Anipose | SLEAP + 3D | AquaPose v2.1 | AquaPose v2.2 Target |
|---------|---------------------|-----------|---------------|----------------------|
| Detection | Manual labeling or YOLO bbox | Manual | YOLO bbox, MOG2 | + YOLO-OBB |
| Midline extraction | Keypoint heatmaps (fixed landmarks) | Keypoint heatmaps | Skeletonize → BFS | + Direct keypoint regression |
| Per-point confidence | Yes (heatmap peak value) | Yes | No | Yes (regression head) |
| Multi-view triangulation | DLT, uniform weighted | DLT | RANSAC + angle-weighted | + Confidence-weighted |
| Training CLI | Yes (entry points) | Yes | 4 scripts in scripts/ | `aquapose train` group |
| Refractive optics | No | No | Yes | Unchanged |
| Config system | YAML | YAML | Frozen dataclasses + YAML + CLI | + N_SAMPLE_POINTS, device |

---

## Sources

- [Ultralytics OBB Task Documentation](https://docs.ultralytics.com/tasks/obb/) — OBB output format, xywhr convention, angle constraints (HIGH confidence — official docs)
- [OpenCV Affine Transformations Tutorial](https://docs.opencv.org/4.x/d4/d61/tutorial_warp_affine.html) — getRotationMatrix2D + warpAffine affine crop pattern (HIGH confidence — official docs)
- [Integral Human Pose Regression — ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xiao_Sun_Integral_Human_Pose_ECCV_2018_paper.pdf) — integral regression alternative (HIGH confidence — peer-reviewed)
- [Boosting integral-based pose estimation — ScienceDirect 2024](https://www.sciencedirect.com/science/article/abs/pii/S0893608024004489) — comparison of heatmap vs integral vs direct at varying resolutions (MEDIUM confidence)
- [Learnable Triangulation of Human Pose](https://saic-violet.github.io/learnable-triangulation/) — confidence-weighted algebraic triangulation pattern (MEDIUM confidence — primary source for the weighted DLT approach)
- [DLT Triangulation: Why Optimize? (arXiv)](https://arxiv.org/pdf/1907.11917) — weighted vs unweighted DLT tradeoffs and ML optimality (MEDIUM confidence)
- [Click Entry Points Documentation](https://click.palletsprojects.com/en/latest/entry-points/) — CLI group/subgroup patterns (HIGH confidence — official docs)
- Existing codebase: `src/aquapose/segmentation/detector.py`, `reconstruction/triangulation.py`, `reconstruction/midline.py`, `cli.py` — authoritative ground truth for all existing interfaces (HIGH confidence)

---

*Feature research for: AquaPose v2.2 Backends — YOLO-OBB, keypoint midline, training CLI, config cleanup*
*Researched: 2026-02-28*
