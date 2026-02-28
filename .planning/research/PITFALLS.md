# Pitfalls Research

**Domain:** Adding YOLO-OBB, keypoint midline backend, training infrastructure, and config cleanup to an existing multi-view 3D fish pose estimation pipeline (v2.2 Backends milestone)
**Researched:** 2026-02-28
**Confidence:** HIGH — codebase directly inspected for all integration points; OBB angle convention verified against ultralytics GitHub issues #13003, #16235 and official docs; existing pitfalls from prior research retained below.

> **Scope note:** This file covers two tiers of pitfalls. The first section ("v2.2 Integration Pitfalls") is new research specific to adding OBB detection, keypoint regression, and training infrastructure to the existing system. The second section ("Foundation Pitfalls") preserves the original project-wide pitfalls from v1.0/v2.0/v2.1 research that remain relevant.

---

## v2.2 Integration Pitfalls

These pitfalls are specific to adding the v2.2 feature set to the existing pipeline. They involve coordinate system mismatches, contract changes, and config system fragility when integrating new components with existing consumers.

---

### Pitfall A1: OBB Angle Convention Mismatch Between Extraction and Affine Crop

**What goes wrong:**
Ultralytics YOLO-OBB outputs angles in **radians** in the range `[-pi/4, 3pi/4)`, using a **clockwise** convention (angle=0 means no rotation, positive angle rotates clockwise). Code that assumes degrees, counter-clockwise, or the `[0, pi/2)` range produced by OpenCV `minAreaRect` will generate affine crops rotated by the wrong amount — often 90 degrees off — silently producing valid-looking but misoriented crops. The fish body appears horizontal when it should be diagonal. The keypoint model trained on correctly oriented crops receives garbage input at inference time.

**Why it happens:**
Three separate angle representations coexist in this ecosystem. OpenCV `minAreaRect()` returns angles in degrees in `(-90, 0]`. Ultralytics OBB model outputs radians in `[-pi/4, 3pi/4)` (clockwise). Label format (`xyxyxyxy` corners) is angle-free; conversion via `minAreaRect` introduces the OpenCV degree convention. When training labels are prepared with OpenCV and the model is queried via ultralytics, developers assume both use the same convention. There is a documented inconsistency between the `[0, pi/2)` range used in label conversion and the `[-pi/4, 3pi/4)` range of model predictions (ultralytics issues #13003 and #16235).

**How to avoid:**
- Always extract angle from `result.obb.xywhr` (not re-derived from `xyxyxyxy`) and confirm units are radians before passing to `cv2.getRotationMatrix2D`.
- Convert: `angle_deg = float(box.xywhr[0, 4]) * 180 / math.pi` — this is clockwise from horizontal.
- Add an explicit smoke test before any keypoint model training: given a synthetic fish at a known angle, verify the affine crop has the fish axis aligned with the crop horizontal.
- Never mix `minAreaRect` angles with ultralytics OBB angles in the same pipeline step without an explicit conversion guard.

**Warning signs:**
- Affine-cropped images consistently look sideways or upside-down relative to the OBB overlay drawn from the same detection.
- Keypoint model outputs confidences near 0.5 uniformly across all body points (random-looking, indicating random orientation input).
- OBB overlay on the full frame looks correct but the crop-based visualization looks wrong.

**Phase to address:**
YOLO-OBB detection backend phase. Add a crop-orientation smoke test before any keypoint model training begins.

---

### Pitfall A2: Keypoint Coordinates Returned in Crop-Local Space, Consumed as Frame-Global

**What goes wrong:**
A keypoint regression head operating on a 128x128 crop returns points in crop coordinates `(0..crop_w, 0..crop_h)`. If those coordinates are stored directly into `Midline2D.points` without applying the `CropRegion` inverse transform (scale + translate), all downstream consumers receive midline points clustered near the image origin rather than at the actual fish position. Reconstruction will fail silently: triangulation attempts to triangulate near `(0, 0)` in each view, producing 3D points near the camera, not the fish. The `AnnotatedDetection` and HDF5 writer accept any `Midline2D` without checking coordinate plausibility.

**Why it happens:**
The existing `segment_then_extract` backend explicitly calls `_crop_to_frame()` before returning a `Midline2D`. A keypoint backend author implementing inference-then-return may return points immediately after argmax or soft-argmax without noticing the coordinate system obligation. `Midline2D` has no field that records whether points are in frame or crop space — the contract is implicit in the docstring ("Full-frame pixel coordinates") and is never enforced at runtime.

**How to avoid:**
- The keypoint backend **must** call `_crop_to_frame()` (or an equivalent) before constructing `Midline2D`. This function in `reconstruction/midline.py` handles the resize scale from model input size (128x128) to actual crop dimensions, then translates by `crop_region.x1, crop_region.y1`.
- Add an integration test: `midline.points[:, 0].min() > crop_region.x1 - 10` and `midline.points[:, 1].min() > crop_region.y1 - 10` for every non-None midline.
- Consider a runtime assertion in `MidlineStage.run()` after each backend call: verify points fall within detection bbox expanded by some margin.

**Warning signs:**
- Midline visualization overlays show all points clustered at `(0,0)` or top-left of the frame.
- Triangulation produces 3D points with X and Y near zero regardless of fish position.
- `AnnotatedDetection.midline.points.mean(axis=0)` is far from `detection.bbox` centroid.

**Phase to address:**
Keypoint midline backend phase, at the coordinate transform step. Verify with a frame-space coordinate assertion before integration testing.

---

### Pitfall A3: Changing Midline2D Point Count Breaks HDF5 Writer, Curve Optimizer, and Visualization

**What goes wrong:**
`N_SAMPLE_POINTS = 15` is hardcoded in `reconstruction/triangulation.py` and imported by `io/midline_writer.py`, which pre-allocates HDF5 datasets with shape `(N, max_fish, 15)`. The curve optimizer (`reconstruction/curve_optimizer.py`) also imports `N_SAMPLE_POINTS` directly. Visualization code in `visualization/triangulation_viz.py:514` and `visualization/midline_viz.py:615` has bare `if n_skel < 15:` guards. If a keypoint backend produces a different point count, the HDF5 writer silently truncates with `n_hw = min(len(hw), N_SAMPLE_POINTS)`. If a user sets `midline.n_points = 20` via YAML, only `MidlineStage` and its backends respect it — the HDF5 writer, curve optimizer, and visualization stay at 15.

**Why it happens:**
`n_points=15` is a config parameter in `MidlineConfig` that flows through `MidlineStage.__init__()` → `get_backend()` → backends. But the HDF5 writer, curve optimizer, and visualization modules import the constant directly rather than receiving it from config. They do not observe `MidlineConfig.n_points`. This is a pre-existing partial wiring that becomes a landmine when v2.2 adds a keypoint backend.

**How to avoid:**
- Do not change `n_points` from 15 in v2.2 unless all consumers are audited first.
- If configurable point count is needed, add `n_sample_points` to `ReconstructionConfig` and thread it through `Midline3DWriter` and the curve optimizer constructor.
- Remove all bare `15` literals from visualization code — use the `N_SAMPLE_POINTS` constant.
- Add a CI check: `grep -rn "< 15\b\|== 15\b" src/aquapose/visualization/` should return zero matches after the config cleanup phase.
- The keypoint backend should use the same `n_points` parameter passed to it by `MidlineStage` — no independent hardcoding inside the backend.

**Warning signs:**
- HDF5 `half_widths` dataset has trailing NaN columns when n_points differs from 15.
- Visualization skeleton-length check fires at a threshold that disagrees with the backend minimum.
- `Midline3D.half_widths.shape[0]` is not 15 but downstream analysis indexes `[:15]` silently.

**Phase to address:**
Config cleanup phase (scatter audit). Keypoint backend phase (ensure backend receives and uses the same n_points). The HDF5 writer fix should precede reconstruction integration.

---

### Pitfall A4: OBB NMS Suppresses Overlapping Fish Differently from AABB NMS

**What goes wrong:**
OBB NMS uses rotated IoU rather than axis-aligned IoU. For fish that are nearly parallel and close (common schooling behavior), rotated IoU is significantly higher than AABB IoU even when the fish are distinct individuals — because elongated boxes sharing the same orientation have large overlap. OBB NMS with the same `iou_threshold=0.45` used for AABB will suppress one of two nearby parallel fish, silently reducing detection count. Downstream tracking loses a fish, Association stage gets fewer tracklets, and reconstruction drops a fish from the 3D output without error.

**Why it happens:**
The existing `YOLODetector` uses `iou_threshold=0.45` tuned for axis-aligned boxes. Developers test with solo or spaced fish and see correct results, but schooling cases only appear in full recordings. The AABB threshold is not the right starting point for OBB.

**How to avoid:**
- Use a higher OBB `iou_threshold` (0.60–0.70) as the starting point for elongated fish; 0.45 is too aggressive for aligned orientations.
- Test with frames that have at least 2 parallel fish within 100px of each other — this is the stress case.
- After deploying OBB detection, compare per-frame detection counts against the existing YOLO AABB baseline on the same video segment.

**Warning signs:**
- Per-frame detection count is lower than expected, especially in schooling frames.
- Fish pairs that are visually distinct in the frame are missing one member.
- OC-SORT shows coasting tracks in frames where fish are grouped.

**Phase to address:**
YOLO-OBB detection backend phase. Include a detection count regression test against the existing AABB backend on a reference frame set.

---

### Pitfall A5: Affine Crop Produces Black Border Artifacts That Confuse Segmentor and Keypoint Model

**What goes wrong:**
An affine rotation crop (used to de-rotate the fish to horizontal) fills areas outside the original image with `borderValue=0` (black) when using `cv2.warpAffine`. If the fish is near a frame edge, up to 30–40% of the crop may be black border. The U-Net segmentor, trained on natural crops without large black regions, may predict foreground probability in the black area (treating it as dark water or a fish body). The keypoint model may regress keypoints into the black padding region. `_check_skip_mask` may also incorrectly fire "boundary-clipped" on artificial black borders.

**Why it happens:**
`extract_crop()` in `segmentation/crop.py` is a simple rectangle slice with no border artifacts. Affine-rotated crops using `cv2.warpAffine` introduce hard black edges at rotation boundaries. Developers testing on fish in the tank center (far from edges) never observe the artifact.

**How to avoid:**
- Use `cv2.BORDER_REPLICATE` or `cv2.BORDER_REFLECT` instead of `cv2.BORDER_CONSTANT` in all `warpAffine` calls. Replicated borders are semantically neutral (background-like) rather than black.
- Test with a detection whose bbox is within 50px of the frame edge. Verify the affine crop has no all-zero rows or columns inside the fish region.

**Warning signs:**
- Masks for fish near frame edges include large rectangular black regions.
- `_check_skip_mask` reports "boundary-clipped" on affine crops that aren't near actual frame boundaries.
- Keypoint confidence is systematically lower for fish observed by cameras that see the tank wall.

**Phase to address:**
YOLO-OBB affine crop implementation within the midline backend phase.

---

### Pitfall A6: Training Augmentation Breaks Spatial Consistency Between Image and Keypoint Labels

**What goes wrong:**
Standard image augmentation (horizontal flip, random crop, perspective warp) applied to the image without applying the exact same transform to keypoint coordinates produces mismatched labels. The keypoint model learns from image patches where the fish is flipped left-right but the label says "head is at the left end." Training loss decreases normally (the model memorizes a random mapping) but inference orientation is systematically wrong. This is particularly insidious because the loss looks healthy.

**Why it happens:**
Image augmentation libraries have two modes: image-only and image+annotations. Albumentations, torchvision, and imgaug all support keypoint-aware transforms but require explicitly registering keypoints as `KeypointParams` or similar. If a developer extends the existing `BinaryMaskDataset` pattern by adding augmentation at the image level without registering keypoints, the geometric transforms are applied to images only.

**How to avoid:**
- Use Albumentations with `KeypointParams(format="xy", remove_invisible=False)` so all geometric transforms (flip, rotate, crop, warp) apply to both image and keypoint labels simultaneously.
- Non-geometric augmentations (brightness, contrast, blur, noise) are safe to apply to the image only.
- Add a visual validation step: augment a batch, overlay keypoints on the augmented image, and confirm alignment before training for more than a few epochs.

**Warning signs:**
- Training loss decreases to a low value but validation keypoint metrics are poor.
- Head-end prediction accuracy is near 50% (random) despite good total loss.
- Flipped fish in augmented frames have keypoints that do not match the flip.

**Phase to address:**
Training infrastructure phase. Validate augmentation pipeline visually before any model training begins.

---

### Pitfall A7: Train/Val Split Leakage With Temporally Correlated Pseudo-Label Frames

**What goes wrong:**
Pseudo-labels are generated from consecutive video frames. If the train/val split is done by randomly shuffling individual frames, consecutive frames from the same sequence appear in both train and val. The model memorizes fish trajectories rather than generalizing. Validation loss looks excellent from early epochs, the model appears well-trained, but it fails on held-out clips because it has memorized specific fish positions and orientations rather than learned general appearance.

**Why it happens:**
The existing `BinaryMaskDataset` draws from a list of `(image, mask)` pairs without temporal structure awareness. A random 80/20 frame-level split leaks: frames t=100 and t=101 are nearly identical, so the model sees t=100 in train and t=101 in val, which is functionally train data.

**How to avoid:**
- Split by contiguous temporal segment, not by individual frame. Hold out a full contiguous clip (different recording session or a non-overlapping temporal window) as the val set.
- If only one recording is available, split by non-overlapping temporal windows with a gap: e.g., frames 0–200 train, frames 200–250 val (with a 50-frame buffer, not random selection from the pool).

**Warning signs:**
- Val loss matches train loss almost exactly from early epochs onward.
- Model accuracy degrades significantly when tested on a different recording.
- Keypoint metrics on val are much better than on a held-out clip.

**Phase to address:**
Training infrastructure phase, before pseudo-label dataset construction.

---

### Pitfall A8: Device Propagation Failure When Multiple New Backends Each Default to "cuda"

**What goes wrong:**
`MidlineStage` defaults `device="cuda"`. `DetectionConfig` defaults `device="cuda"`. A new OBB backend will add another device parameter. If each component has its own device default rather than inheriting from a single source, two failure modes arise: (1) CPU-only machines fail at construction with "CUDA is not available" unless the user knows to set `device=cpu` in YAML; (2) if the OBB detector (ultralytics auto-selects device) ends up on CPU while the U-Net stays on GPU, tensors from different stages end up on different devices and inference fails mid-frame with a confusing error.

**Why it happens:**
The current `PipelineConfig` has no single top-level `device` field — each sub-config (`DetectionConfig.device`, `MidlineConfig` via `MidlineStage`) has its own default. A `device` set at `detection.device=cpu` does not propagate to `midline` device.

**How to avoid:**
- Add a single `device: str = "cuda"` field at the top level of `PipelineConfig`.
- In `load_config()`, propagate `top_kwargs["device"]` to `det_kwargs` and `mid_kwargs` if those sub-configs do not explicitly override device.
- In the training CLI, default to `"cuda" if torch.cuda.is_available() else "cpu"` rather than hardcoding "cuda".
- Add a test: construct the full pipeline with `device="cpu"` and verify no model tensor ends up on CUDA.

**Warning signs:**
- `RuntimeError: Tensors are on different devices` appearing in the midline backend after adding an OBB crop step.
- CI failures with "CUDA is not available" even though tests are expected to be CPU-only.
- The OBB detector works (ultralytics auto-selects CPU) but the U-Net fails (explicit device="cuda").

**Phase to address:**
Config cleanup phase. Top-level device propagation should precede any new backend addition so new backends inherit it correctly.

---

### Pitfall A9: Adding Fields to Midline2D Without Defaults Breaks All Construction Sites

**What goes wrong:**
`Midline2D` currently has no `confidence` or `per_point_confidence` field. Adding a field to this dataclass without a default value causes `TypeError` at all construction sites. `Midline2D` is constructed in at least 4 locations: `SegmentThenExtractBackend._extract_midline_from_mask()`, `MidlineExtractor.extract_midlines()` (legacy), `core/synthetic.py`, and any new keypoint backend. The HDF5 writer already expects `is_low_confidence: bool` on `Midline3D` (`midline_writer.py:168`) — if the equivalent is not added to `Midline2D` and threaded through `AnnotatedDetection`, the writer always writes `False`.

**Why it happens:**
`Midline2D` is defined in `reconstruction/midline.py` and re-exported from two `core/` type modules. Its construction is scattered. The field addition looks trivial but has wide blast radius.

**How to avoid:**
- Add `per_point_confidence: np.ndarray | None = None` with a default (keyword-only, not positional) so existing construction sites work without modification.
- Audit all construction sites before the field addition: `grep -rn "Midline2D(" src/aquapose/`.
- Update the reconstruction backend to check `per_point_confidence is not None` and apply confidence weighting only when present (backward compatible).

**Warning signs:**
- `TypeError: __init__() missing 1 required positional argument` at any `Midline2D()` call site after adding a field without a default.
- The HDF5 `is_low_confidence` dataset is all-False even when the keypoint backend flags low confidence.
- Triangulation uses uniform weights even when per-point confidence data is available.

**Phase to address:**
Keypoint midline backend phase, as a prerequisite to confidence-weighted reconstruction.

---

### Pitfall A10: Config Backward Compatibility — New Fields Without Defaults Break Existing YAML Files

**What goes wrong:**
When new fields are added to `DetectionConfig`, `MidlineConfig`, or `ReconstructionConfig` without default values, existing YAML config files that do not include those fields cause `TypeError` at load time. The existing `load_config()` has a `_filter_fields()` helper that strips unknown keys for `AssociationConfig` and `TrackingConfig`, but it is **not** applied to `DetectionConfig`, `MidlineConfig`, or `ReconstructionConfig`. Adding new required fields to those dataclasses — or removing old fields — silently breaks all existing YAML configs.

**Why it happens:**
The `_filter_fields()` safety net is partial by accident — it was added to fix an issue with `AssociationConfig` and `TrackingConfig` during v2.1 refactoring but was never applied universally. `DetectionConfig(**det_kwargs)` and `MidlineConfig(**mid_kwargs)` receive raw dicts from YAML without filtering.

**How to avoid:**
- Apply `_filter_fields()` to ALL stage config dataclasses in `load_config()`, not just association and tracking.
- All new config fields must have defaults — never add a required field to an existing production dataclass.
- If a field is renamed, keep the old name as a deprecated alias in the YAML loading layer for at least one milestone.
- After any config schema change, test `load_config(yaml_path=pinned_v21_yaml)` in CI against a representative saved YAML file.

**Warning signs:**
- `TypeError: __init__() missing 1 required positional argument` when loading an existing YAML after a dataclass change.
- Researchers who saved working YAML configs from v2.1 find they no longer work after upgrading.
- CI tests pass with hardcoded test configs but user-generated configs fail.

**Phase to address:**
Config cleanup phase. Apply `_filter_fields()` universally and add a pinned YAML regression test before any other config schema changes.

---

### Pitfall A11: Reconstruction Assumes Midlines Have Consistent Point Count — Keypoint Backend Must Not Vary It

**What goes wrong:**
The triangulation backend uses arc-length position `t[i]` as the correspondence key across cameras: body point at arc-length index `i` in camera A is matched to body point at index `i` in camera B. This works only when both cameras produce midlines with the same number of points sampled at the same arc-length positions (`t = linspace(0, 1, n_points)`). A keypoint backend that returns a variable number of points (e.g., 13 high-confidence points for an occluded fish) produces a mismatched arc-length mapping. Point `t[i]` on the 13-point midline corresponds to a different body position than `t[i]` on the 15-point midline, corrupting triangulation silently.

**Why it happens:**
The segment-then-extract backend always produces exactly `n_points` points (or `None`). A keypoint backend author may decide to omit low-confidence points to "improve quality," not realizing that the correspondence constraint requires a fixed point count.

**How to avoid:**
- The keypoint backend must always produce exactly `n_points` output points, regardless of per-point confidence. Store confidence per point but never omit points.
- If partial midlines are desired (e.g., only head-side points visible), pad to `n_points` with the last known position and mark the padded points as low-confidence in `per_point_confidence`.
- Add a contract assertion in `MidlineStage.run()`: `assert midline.points.shape == (n_points, 2)` for every non-None midline.

**Warning signs:**
- Triangulation produces 3D midlines with incorrect curvature (S-shaped when the fish is straight).
- `midline.points.shape[0]` varies frame-to-frame for the same fish.
- Reconstruction succeeds with 2 cameras but fails with 4+ cameras (more cameras expose the mismatch).

**Phase to address:**
Keypoint midline backend phase, during output format definition. The n_points contract must be locked before reconstruction integration.

---

### Pitfall A12: Import Boundary Violation — New training/ Module Must Not Import from engine/

**What goes wrong:**
The AST-based import boundary checker enforces that `core/` never imports from `engine/`. A new `src/aquapose/training/` module that imports training utilities from `engine/config.py` (which is in engine/) will violate the import boundary and fail the pre-commit hook. If developers work around this by adding `training/` to the boundary checker's allowlist without thought, they may inadvertently permit circular imports.

**Why it happens:**
Training infrastructure naturally wants access to config (for hyperparameters) and possibly to pipeline stages (for data generation). The easy path is to import from wherever the class lives. The correct path is to either have `training/` depend only on `core/` and stdlib, or to explicitly declare that `training/` is allowed to depend on `engine/` as a top-level module.

**How to avoid:**
- Decide before implementation: is `training/` a peer of `engine/` (allowed to import from engine/) or a consumer of `core/` only? Document this in the import boundary checker config.
- If `training/` needs `PipelineConfig`, import it from `engine/config.py` explicitly and declare the allowance in the boundary checker.
- Run `hatch run pre-commit run --all-files` after adding any new import in `training/` — do not wait until the module is complete.

**Warning signs:**
- Pre-commit fails with "import boundary violation: training imports from engine".
- Workarounds like `TYPE_CHECKING` guards proliferating in `training/` to avoid import errors.
- Circular import at runtime when `training/` is imported.

**Phase to address:**
Training infrastructure phase, before writing any training module code.

---

## Technical Debt Patterns (v2.2 Specific)

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Leaving bare `15` literals in visualization code | No refactor needed now | Any n_points change silently breaks visualization | Never — replace with `N_SAMPLE_POINTS` constant in v2.2 cleanup |
| OBB backend config in `DetectionConfig.extra` dict | No new config fields | OBB params undocumented, not type-checked, not validated | Only as a temporary shim during development; add proper fields before shipping |
| Confidence as `None` in segment-then-extract backend after adding `per_point_confidence` field | No change to existing backend | Reconstruction cannot confidence-weight existing backend output | Acceptable for v2.2 if reconstruction checks for None |
| No `_filter_fields()` on `DetectionConfig`/`MidlineConfig` | Less code in load_config | Any YAML from v2.1 breaks on new field additions | Never — apply filter universally in v2.2 config cleanup |
| Separate `device` fields per sub-config | Each stage independently configurable | Device mismatch errors mid-pipeline; no single override point | Acceptable only if top-level propagation is implemented |
| Training CLI invents its own config loading | Faster initial implementation | Two config systems diverge; YAML files not interchangeable | Never — reuse `load_config()` from engine/config.py |

---

## Integration Gotchas (v2.2 Specific)

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| OBB → CropRegion | Deriving CropRegion from OBB corners using their axis-aligned bounding box | Use OBB `xywhr` to compute affine rotation matrix; CropRegion describes the axis-aligned bounding box of the rotated crop in frame space |
| OBB angle → affine crop | Using `cv2.minAreaRect` on OBB corners to re-derive angle (introduces degree/OpenCV convention) | Use `result.obb.xywhr[..., 4]` (radians, clockwise) directly |
| Keypoint → Midline2D | Storing crop-space coordinates directly in `Midline2D.points` | Always call `_crop_to_frame()` from `reconstruction/midline.py` before constructing `Midline2D` |
| Training CLI → config system | Adding a new Click command that re-implements config loading from scratch | Reuse `load_config()` from `engine/config.py`; add training-specific fields to an extension of `PipelineConfig` if needed |
| New midline backend → MidlineStage | Implementing OBB crop logic directly in the stage | Follow the `SegmentThenExtractBackend` pattern: backend class implements `process_frame()`, registered in `core/midline/backends/__init__.py:get_backend()`, stage stays thin |
| Per-point confidence → HDF5 | Adding a new HDF5 dataset without updating `Midline3DWriter` | The writer pre-allocates datasets at `open()` — new fields require updating `_make()` calls in the constructor; existing files cannot be appended to with new schema |

---

## "Looks Done But Isn't" Checklist (v2.2 Specific)

- [ ] **OBB backend registered:** `make_detector("yolo-obb", ...)` in `segmentation/detector.py` returns a `YOLOOBBBackend` — verify the factory handles the new kind string.
- [ ] **Affine crop orientation tested:** Given a known-angle detection, verify the affine-cropped image has the fish axis aligned to horizontal — visual check, not just "no exception raised."
- [ ] **Keypoint coordinate transform verified:** `midline.points[:, 0].min() > crop_region.x1 - 20` for every non-None midline produced by the keypoint backend.
- [ ] **Config filter universal:** `_filter_fields()` applied to `DetectionConfig`, `MidlineConfig`, `ReconstructionConfig` — not just `AssociationConfig` and `TrackingConfig`.
- [ ] **n_points contract enforced:** `assert midline.points.shape == (config.midline.n_points, 2)` in `MidlineStage.run()` after each backend call.
- [ ] **HDF5 schema versioned if changed:** If `Midline3DWriter` gains new datasets, the HDF5 file has a schema version attribute so downstream analysis tools can detect format changes.
- [ ] **Training CLI uses shared config:** `aquapose train` loads via `load_config()` — not a separate config class invented for training.
- [ ] **Import boundary clean:** After adding `src/aquapose/training/`, `hatch run pre-commit run --all-files` reports 0 import boundary violations.
- [ ] **OBB NMS threshold tested on parallel fish:** Run on at least one frame with 2 adjacent parallel fish and verify both are detected with the chosen `iou_threshold`.
- [ ] **Pinned YAML regression test:** Loading a saved v2.1 YAML with the new config schema raises no `TypeError`.

---

## Recovery Strategies (v2.2 Specific)

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| OBB angle convention mismatch discovered after training keypoint model | HIGH | Re-generate all affine crops with correct angle convention, retrain from scratch — no post-hoc correction exists |
| Keypoint coordinates in crop space discovered in production | MEDIUM | Add `_crop_to_frame()` call in backend and re-run inference — no retraining needed |
| n_points mismatch in HDF5 file | MEDIUM | Write migration script that reads old HDF5, pads/truncates half_widths to 15, writes new file |
| Config backward compat broken by new required field | LOW | Add default value to the new field and release a patch — existing YAMLs work again without user changes |
| Train/val temporal leakage discovered after training | HIGH | Re-split by temporal segment and retrain — metrics from leaky split are unreliable |
| Device mismatch error mid-pipeline | LOW | Add device propagation to `load_config()`, no model changes needed |

---

## Pitfall-to-Phase Mapping (v2.2 Specific)

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| A1: OBB angle convention mismatch | YOLO-OBB detection backend | Smoke test: known-angle detection → affine crop → axis-aligned fish |
| A2: Keypoint coordinates in crop space | Keypoint midline backend | Integration test: assert points within detection bbox |
| A3: N_SAMPLE_POINTS scatter | Config cleanup | Grep for bare `15` literals in visualization; assert zero |
| A4: OBB NMS suppressing parallel fish | YOLO-OBB detection backend | Detection count regression test vs AABB baseline on schooling frames |
| A5: Affine crop border artifacts | YOLO-OBB affine crop step | Test: detection within 50px of frame edge produces valid crop |
| A6: Augmentation spatial inconsistency | Training infrastructure | Visual overlay check: augmented image + transformed keypoints aligned |
| A7: Train/val temporal leakage | Training infrastructure | Hold out a full contiguous clip as val; val loss diverges from train |
| A8: Device propagation failure | Config cleanup | Test: construct pipeline with `device="cpu"`, no CUDA tensors anywhere |
| A9: Midline2D contract change | Keypoint midline backend | All construction sites updated; no positional-arg `TypeError` |
| A10: Config backward compat breakage | Config cleanup | Load pinned v2.1 YAML after any schema change — no `TypeError` |
| A11: Arc-length mismatch (variable n_points) | Keypoint midline backend | Assert `midline.points.shape == (n_points, 2)` in `MidlineStage.run()` |
| A12: Import boundary violation | Training infrastructure | `hatch run pre-commit run --all-files` = 0 violations after each new import |

---

---

## Foundation Pitfalls (Retained from v1.0–v2.1 Research)

*These pitfalls from prior research remain relevant to the overall system.*

---

### Pitfall 1: Treating Refractive Distortion as Depth-Independent

**What goes wrong:** Refractive projection through a flat air-water port is depth-dependent. Systems that model refraction as a fixed pixel-wise correction produce systematic 3D errors that grow with distance from calibration depth. **Status:** Addressed in v1.0 via `RefractiveProjectionModel` with per-ray Snell's law tracing.

**Phase to address:** Camera model and calibration phase (resolved).

---

### Pitfall 2: All-Top-Down Camera Configuration Creates Weak Z-Reconstruction

**What goes wrong:** 13 cameras all looking straight down share nearly parallel optical axes. Z-reconstruction uncertainty is 132x larger than XY. **Status:** Quantified in v1.0; XY-only tracking cost matrix applied in early versions, superseded by OC-SORT per-camera in v2.1.

**Phase to address:** Geometry validation phase (resolved; 132x Z/XY anisotropy documented).

---

### Pitfall 3: MOG2 Background Subtraction Fails on Low-Contrast Female Fish

**What goes wrong:** Female cichlids have lower visual contrast against tank substrate. MOG2 absorbs slow/stationary fish into the background model. **Status:** YOLO added as primary detector; MOG2 retained as fallback. Detection recall for females remains a known limitation.

**Phase to address:** Detection module phase (mitigated via YOLO).

---

### Pitfall 4: Arc-Length Correspondence Errors on Curved Fish

**What goes wrong:** Arc-length normalization assumes the midline projection preserves parameterization across views. For significantly curved fish viewed from different angles, foreshortening compresses the arc-length mapping unevenly, creating triangulation errors at body points away from the head/tail endpoints.

**How to avoid:** RANSAC per body point during triangulation; view-angle weighting. **Status:** Partially mitigated by RANSAC + view-angle weighting in `triangulate_midlines()`.

**Phase to address:** Triangulation (active concern for reconstruction quality).

---

### Pitfall 5: Medial Axis Instability on Noisy Masks (IoU ~0.62)

**What goes wrong:** `skeletonize` on masks with boundary noise produces unstable, branchy skeletons that wobble frame-to-frame. **Status:** Mitigated by `_adaptive_smooth()` with morphological closing/opening and adaptive kernel radius. The keypoint backend, if implemented correctly, bypasses this entirely.

**Phase to address:** Midline extraction (partially mitigated; keypoint backend is the intended long-term fix).

---

### Pitfall 6: Head-Tail Ambiguity in Arc-Length Parameterization

**What goes wrong:** Without orientation information, the skeleton may be ordered tail-to-head in some cameras and head-to-tail in others, corrupting arc-length correspondence. **Status:** Addressed in v2.1 via `resolve_orientation()` using cross-camera geometry, velocity, and temporal prior signals.

**Phase to address:** Midline extraction (resolved in v2.1).

---

## Sources

- Ultralytics YOLO OBB documentation: [Oriented Bounding Boxes Object Detection](https://docs.ultralytics.com/tasks/obb/)
- Ultralytics issue #13003 "Is the angle value given by OBB correct?": [GitHub](https://github.com/ultralytics/ultralytics/issues/13003)
- Ultralytics issue #16235 "YOLOv8-OBB angle conversion": [GitHub](https://github.com/ultralytics/ultralytics/issues/16235)
- PyTorch mixed precision training: [What Every User Should Know About Mixed Precision Training in PyTorch](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
- Direct codebase inspection: `src/aquapose/reconstruction/midline.py`, `core/midline/types.py`, `core/midline/stage.py`, `core/midline/backends/segment_then_extract.py`, `engine/config.py`, `io/midline_writer.py`, `reconstruction/triangulation.py`, `segmentation/crop.py`, `segmentation/detector.py`, `core/context.py`
- Prior project pitfalls research: 2026-02-19 / 2026-02-21 (v1.0–v2.1)

---
*Pitfalls research for: v2.2 Backends — YOLO-OBB, keypoint midline, training infrastructure, config cleanup*
*Researched: 2026-02-28*
