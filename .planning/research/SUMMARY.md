# Project Research Summary

**Project:** AquaPose v2.2 Backends
**Domain:** 3D fish pose estimation — swappable ML backends, training infrastructure, config cleanup
**Researched:** 2026-02-28
**Confidence:** HIGH

## Executive Summary

AquaPose v2.2 adds three capabilities to an already-shipped, production-quality pipeline: a YOLO-OBB detection backend that provides fish orientation angles directly from the detector, a keypoint regression head that replaces noisy skeletonization with direct ordered-coordinate prediction, and a unified training CLI that consolidates four disconnected scripts into one discoverable `aquapose train` command group. The existing architecture — a 3-layer event-driven system (Core → PosePipeline → Observers) with 5 pipeline stages and a frozen-dataclass config — accommodates all of these additions cleanly via its backend registry pattern and optional-field dataclass extension pattern. No new runtime dependencies are required; the only version change is bumping `ultralytics>=8.0` to `>=8.1` for OBB support.

The recommended build order follows a strict dependency hierarchy driven by the codebase's own contracts. Foundation work comes first: config cleanup (device propagation, configurable `n_sample_points`, universal `_filter_fields()`) and backward-compatible extensions to the `Detection` and `Midline2D` dataclasses must land before any new backends are written. Detection backend and keypoint model development can then proceed in parallel. Confidence-weighted triangulation depends only on the `Midline2D.point_confidence` field added in the foundation phase, so it too can proceed in parallel with the backend work. Training infrastructure is fully independent and can be developed alongside or after the backend work.

The highest-risk area is coordinate system handling: the OBB angle convention (radians, clockwise, range `[-pi/4, 3pi/4)`) differs from OpenCV's `minAreaRect` output (degrees, `(-90, 0]`), and keypoint coordinates must be transformed from crop space back to frame space before being stored in `Midline2D.points`. Both errors are silent — pipeline execution succeeds but 3D reconstruction produces garbage. The mitigation is mandatory smoke tests at each coordinate boundary before any model training begins.

## Key Findings

### Recommended Stack

The v2.2 milestone requires no new top-level dependencies. All new capabilities use the existing dependency set. The only change to `pyproject.toml` is a version bump for `ultralytics` to `>=8.1` to guarantee OBB model availability. `tqdm`, useful for training progress display, is available as a transitive dependency of `ultralytics` and should be imported without being added as an explicit dependency.

**Core technologies:**
- PyTorch (latest stable): U-Net inference, keypoint regression head, YOLO — no longer pinned to 2.4.1; primary pipeline has no PyTorch3D coupling
- ultralytics >=8.1: YOLO standard bbox and YOLO-OBB detection; OBB support added in 8.1.0
- scikit-image >=0.22: skeletonization for pseudo-label generation, midline arc-length resampling
- scipy >=1.13: spline fitting, Levenberg-Marquardt refinement, Hungarian assignment
- opencv-python >=4.8: `cv2.getRotationMatrix2D` + `cv2.warpAffine` for affine crop extraction
- click >=8.1: all CLI surface including new `aquapose train` subgroup
- boxmot >=11.0 + leidenalg >=0.10: OC-SORT and Leiden (unchanged from v2.1)

**What NOT to add:** albumentations (use torchvision.transforms.v2), timm (torchvision MobileNetV3-Small is sufficient), mmdet/mmpose (complex install), segmentation_models_pytorch (custom U-Net already exists), PyTorch Lightning (simple loops don't justify trainer abstraction), Pydantic (frozen dataclasses decided in v2.0).

### Expected Features

**Must have (table stakes — ship in v2.2):**
- `YOLOOBBDetector` + `make_detector("yolo_obb", ...)` — OBB backend drop-in with `Detection.angle` field
- `extract_obb_crop(frame, xywhr)` — returns `(crop, M_inv)` for keypoint back-projection
- `UNetKeypointHead` — attaches to frozen MobileNetV3-Small encoder; outputs `(N_points, 3)` in crop coordinates
- `Midline2D.point_confidence: np.ndarray | None = None` — backward-compatible; None = uniform weights
- Confidence-weighted DLT path in `triangulation.py` — weighted when confidences present, falls back gracefully
- `aquapose train` Click group with `unet`, `yolo-obb`, `keypoint` subcommands replacing scripts/
- `src/aquapose/training/` module with `common.py`, `unet.py`, `yolo_obb.py`, `keypoint.py`
- `N_SAMPLE_POINTS` moved to `ReconstructionConfig.n_points` (default 15)
- `device: str = "cuda"` at top-level `PipelineConfig`, propagated through `build_stages()`

**Should have (add after core validated):**
- Partial midline confidence gating (defer until keypoint head quality is known)
- OBB polygon overlay in diagnostic observer
- `init-config` CLI UX improvements (YAML schema comments)

**Defer (v2.3+):**
- Keypoint pseudo-label auto-generation pipeline
- OBB fine-tuning on fish-specific dataset
- Confidence calibration (temperature scaling)
- Multi-task joint segmentation + keypoint loss
- Encoder fine-tuning with keypoint head (freeze encoder entirely in v2.2)

**Anti-features to avoid:**
- Heatmap-based keypoint regression — 4px quantization at 128x128 is unacceptable for ~50px fish
- `OBBDetection` subclass — breaks all isinstance checks; extend `Detection` with optional fields instead
- Real-time 13-camera OBB inference — batch mode only per project constraints
- Separate `aquapose-train` entrypoint — keep under unified `aquapose` CLI

### Architecture Approach

The v2.2 additions slot into the existing 3-layer architecture without structural changes. The backend registry pattern (`get_backend(kind, **kwargs)`) handles the new OBB detection backend and already has a stub for the direct-pose midline backend. The `training/` package is a new peer module at the `segmentation/` level — it is explicitly forbidden from importing `engine/` to avoid circular initialization. All new ML model classes live in `segmentation/` (e.g., `keypoint_model.py`), geometric utilities live in `segmentation/crop.py`, and backend orchestration lives in the appropriate `core/` backend directory.

**Major components (new and modified):**
1. `segmentation/keypoint_model.py` (NEW) — `KeypointModel`: frozen MobileNetV3-Small encoder + direct regression head outputting `(B, N, 3)` in crop coordinates
2. `segmentation/crop.py` (EXTENDED) — `compute_obb_crop_region()`, `extract_affine_crop()`, `untransform_points_from_obb()` shared by both detection overlay and midline backend
3. `core/detection/backends/yolo_obb.py` (NEW) — `YOLOOBBBackend` wrapping `YOLOOBBDetector`; produces `Detection` with `angle` set
4. `core/midline/backends/direct_pose.py` (MODIFIED) — replaces `NotImplementedError` stub with full `DirectPoseBackend`
5. `reconstruction/triangulation.py` (MODIFIED) — adds weighted DLT path scaling rows by `sqrt(point_confidence)` before SVD
6. `engine/config.py` (MODIFIED) — `device` propagation, `n_points` in `ReconstructionConfig`, universal `_filter_fields()`
7. `src/aquapose/training/` (NEW PACKAGE) — `TrainingConfig`, `KeypointDataset`, `train_unet`, `train_keypoint`, `train_yolo_obb`
8. `cli.py` (EXTENDED) — `aquapose train` group, `aquapose init-config` command

**Import boundary (enforced by AST pre-commit hook):**
- `training/` → `core/`, `segmentation/`, `reconstruction/` (NOT `engine/`)
- `cli.py` → `engine/` AND `training/` (the single fan-in point)

### Critical Pitfalls

1. **OBB angle convention mismatch** — ultralytics outputs radians clockwise in `[-pi/4, 3pi/4)`, OpenCV `minAreaRect` outputs degrees counter-clockwise in `(-90, 0]`. Always extract angle from `result.obb.xywhr[..., 4]` and convert via `angle_deg = radians * 180 / pi`. Add a crop-orientation smoke test (known-angle detection → affine crop → visually axis-aligned fish) before any keypoint model training begins.

2. **Keypoint coordinates returned in crop-local space** — the keypoint head outputs `(x, y)` in `[0, crop_w] x [0, crop_h]`. Without calling `_crop_to_frame()` (scale + translate) before constructing `Midline2D`, all midline points cluster near frame origin and triangulation silently produces 3D points near the camera. Add assertion: `midline.points[:, 0].min() > crop_region.x1 - 20` for every non-None midline.

3. **`N_SAMPLE_POINTS` scatter across consumers** — `io/midline_writer.py`, `visualization/triangulation_viz.py`, and `reconstruction/curve_optimizer.py` all hardcode or import `N_SAMPLE_POINTS = 15`. A config change that updates `MidlineConfig.n_points` but not these modules silently produces truncated HDF5 datasets and misfiring visualization guards. Audit and replace all bare `15` literals with the constant before any point-count changes.

4. **Config backward compatibility breakage** — `_filter_fields()` is applied to `AssociationConfig` and `TrackingConfig` but NOT to `DetectionConfig`, `MidlineConfig`, or `ReconstructionConfig`. Adding any new field without a default to the latter three will break all existing user YAML files at load time. Apply `_filter_fields()` universally and add a pinned v2.1 YAML regression test before any other config schema changes.

5. **Arc-length mismatch from variable-length keypoint output** — triangulation uses arc-length index `i` as the cross-camera correspondence key. A keypoint backend that omits low-confidence points produces a different-length midline, shifting the arc-length mapping and silently corrupting 3D reconstruction. The keypoint backend must always produce exactly `n_points` outputs, padding with the last known position and marking padded points as low-confidence. Enforce with `assert midline.points.shape == (n_points, 2)` in `MidlineStage.run()`.

## Implications for Roadmap

Based on research, the architecture's dependency graph dictates the following phase structure. Foundation work gates everything else. Detection backend and keypoint model development can overlap. Training infrastructure is fully independent.

### Phase 1: Config and Contract Foundation

**Rationale:** Config cleanup (`device` propagation, `n_sample_points` centralization, universal `_filter_fields()`) and backward-compatible dataclass extensions (`Detection.angle`, `Midline2D.point_confidence`) have no dependencies and unblock every subsequent phase. If device propagation is not in place before new backends are added, each backend invents its own device default and tensor-device mismatches appear mid-pipeline. If `_filter_fields()` is not applied universally, any new config field added in a later phase will break user YAML files with no graceful error.

**Delivers:** A stable, backward-compatible contract layer that all v2.2 features build on.

**Addresses:**
- `N_SAMPLE_POINTS` moved to `ReconstructionConfig.n_points` (default 15)
- `device: str = "cuda"` at top-level `PipelineConfig`, propagated in `load_config()`
- `_filter_fields()` applied universally to all stage configs
- `Detection.angle: float | None = None`, `Detection.obb_points: np.ndarray | None = None`
- `Midline2D.point_confidence: np.ndarray | None = None`
- `aquapose init-config` CLI command

**Avoids:** Pitfalls A8 (device propagation), A9 (Midline2D contract change), A10 (config backward compat), A3 (N_SAMPLE_POINTS scatter).

**Research flag:** Standard patterns — no additional research needed. Frozen dataclass config system is well-understood from live codebase.

---

### Phase 2: YOLO-OBB Detection Backend

**Rationale:** OBB detection is the first visible capability. It enables affine crop extraction, which the keypoint backend (Phase 3) depends on. OBB NMS threshold tuning (parallel fish suppression) must be validated on real video before any keypoint training data is collected. Building detection first allows auditing detection quality through the existing skeletonize midline backend before the keypoint head is added.

**Delivers:** `YOLOOBBDetector`, `YOLOOBBBackend`, affine crop utilities, OBB polygon overlay. The existing pipeline runs end-to-end with OBB crops and the skeletonize midline backend — detection quality is auditable before the keypoint head is built.

**Addresses:**
- `YOLOOBBDetector` + `make_detector("yolo_obb", ...)` factory branch
- `compute_obb_crop_region()`, `extract_affine_crop()`, `untransform_points_from_obb()` in `segmentation/crop.py`
- `YOLOOBBBackend` in `core/detection/backends/yolo_obb.py`
- OBB polygon overlay in `engine/overlay_observer.py`

**Uses:** ultralytics >=8.1, OpenCV `cv2.warpAffine` with `cv2.BORDER_REPLICATE`, existing backend registry pattern.

**Avoids:** Pitfalls A1 (OBB angle convention — crop-orientation smoke test before training), A4 (OBB NMS parallel fish — detection count regression test on schooling frames), A5 (affine crop border artifacts — use `cv2.BORDER_REPLICATE` not `cv2.BORDER_CONSTANT`).

**Research flag:** Standard patterns — ultralytics OBB API is HIGH confidence from official docs; OpenCV affine crop is canonical. No additional research needed.

---

### Phase 3: Keypoint Regression Midline Backend

**Rationale:** Depends on affine crop utilities from Phase 2 and the `Midline2D.point_confidence` field from Phase 1. The `DirectPoseBackend` stub already exists but raises `NotImplementedError` — this phase implements it. Keypoint model training data (pseudo-labels) is generated from the existing skeletonizer output, so no new annotation tooling is needed.

**Delivers:** `KeypointModel` (`segmentation/keypoint_model.py`), `DirectPoseBackend` (full implementation replacing stub), confidence-weighted DLT in `triangulation.py`. The pipeline can run fully with the keypoint midline backend end-to-end.

**Addresses:**
- `UNetKeypointHead` — direct coordinate regression, `(B, N, 3)` output with sigmoid-activated x/y/confidence
- `DirectPoseBackend.process_frame()` — produces `AnnotatedDetection` with `Midline2D(point_confidence=...)`
- Confidence-weighted DLT: scale RANSAC rows by `sqrt(point_confidence)` before SVD
- `CurveOptimizerBackend` confidence weighting (weighted Chamfer distance)

**Uses:** PyTorch `nn.Module`, existing MobileNetV3-Small encoder weights, scikit-image skeletonizer for pseudo-label generation.

**Avoids:** Pitfalls A2 (crop-to-frame coordinate transform — assertion after every backend call), A11 (arc-length mismatch — enforce fixed `n_points` output contract with `assert midline.points.shape == (n_points, 2)`), A5 (border artifacts — inherited fix from Phase 2).

**Research flag:** Direct regression vs heatmap decision is resolved (direct regression wins at 128x128, 4px heatmap quantization is unacceptable). Wing Loss vs MSE: MSE is sufficient for a first pass. No additional research needed beyond STACK.md content.

---

### Phase 4: Training Infrastructure

**Rationale:** The `aquapose train` CLI and `training/` package are fully independent of all other v2.2 features. They can be developed in parallel with Phases 2 and 3 if capacity allows. Deferring to Phase 4 ensures the ML architecture (Phase 3) is locked before the training CLI is built around it, avoiding rework if the keypoint head API changes.

**Delivers:** `src/aquapose/training/` package (`unet.py`, `keypoint.py`, `yolo_obb.py`, `dataset.py`, `config.py`), `aquapose train unet / keypoint / yolo-obb` subcommands. Replaces four disconnected scripts in `scripts/` with a documented, discoverable interface.

**Addresses:**
- `KeypointDataset` — OBB crop + keypoint annotation loading with temporal-segment train/val split
- `train_unet()` — migrated from `segmentation/training.py`
- `train_keypoint_head()` — new training loop with MSE loss, confidence gating, checkpoint saving
- `aquapose train` Click group registered under existing CLI

**Uses:** click (existing), PyTorch bare training loop (no Lightning), tqdm via ultralytics transitive dep.

**Avoids:** Pitfalls A6 (augmentation spatial consistency — use torchvision keypoint-aware transforms for all geometric augmentations), A7 (train/val temporal leakage — split by contiguous clip with gap, not random frames), A12 (import boundary violation — `training/` must not import from `engine/`).

**Research flag:** Training CLI patterns are standard Click subcommands — no additional research needed. The train/val split strategy (temporal segments) is resolved in PITFALLS.md. Verify torchvision v2 keypoint-aware transform API during implementation.

---

### Phase Ordering Rationale

- **Config and dataclass contracts first:** `Midline2D.point_confidence` and `Detection.angle` are prerequisites for Phases 2 and 3. Doing this in Phase 1 means subsequent phases all build on stable interfaces rather than simultaneously modifying shared types.
- **OBB detection before keypoint backend:** Affine crop utilities (`segmentation/crop.py`) are written once in Phase 2 and reused by Phase 3. Building the keypoint backend first would require implementing the same utilities speculatively.
- **Keypoint backend before training infrastructure:** The `KeypointModel` API (input/output shapes, `n_points` parameter) must be stable before `training/keypoint.py` and `training/dataset.py` are written. If the model API changes, the training loop changes.
- **Training infrastructure last (or parallel):** It is fully decoupled and can be developed in parallel with Phases 2 and 3 if capacity allows, or deferred to Phase 4 without blocking the pipeline.

### Research Flags

Phases needing deeper research during planning:

- **Phase 3 (Keypoint Backend) — partial midline handling:** The correct threshold for `point_confidence` gating and the `MIN_BODY_POINTS` floor value are empirically determined. Set conservatively in v2.2 and tune based on real reconstruction quality. This is deferred from the v2.2 MVP per FEATURES.md.
- **Phase 4 (Training) — OBB label conversion:** The existing fish bbox labels are axis-aligned rectangles. Converting to OBB label format requires either re-annotation or automated conversion via per-detection skeletonization. This conversion step is not detailed in any research file and should be scoped during Phase 2 or 4 planning.

Phases with standard patterns (skip additional research):

- **Phase 1 (Config/Contracts):** Frozen dataclass extension and `load_config()` patterns are fully documented in the live codebase.
- **Phase 2 (OBB Detection):** ultralytics OBB API is HIGH confidence from official docs; OpenCV affine transforms are canonical.
- **Phase 3 (Keypoint Model):** Direct regression at 128x128 is the resolved recommendation; no alternative approaches need investigation.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All stack decisions use existing or well-documented libraries; only version bump is ultralytics >=8.1, verified against official OBB release notes and GitHub discussions |
| Features | HIGH | Existing codebase fully inspected; all interface contracts (Detection, Midline2D, backend registry) are known ground truth; OBB/keypoint approaches verified against official docs and literature |
| Architecture | HIGH | Based on live codebase analysis, not speculation; all integration points, import boundaries, and build order derived from actual code |
| Pitfalls | HIGH | OBB angle convention verified against ultralytics GitHub issues #13003 and #16235; all other pitfalls derived from direct codebase inspection of actual construction sites and consumers |

**Overall confidence:** HIGH

### Gaps to Address

- **Keypoint head quality floor:** The target accuracy for the keypoint regression head is not specified. The current U-Net achieves IoU 0.623 (accepted below target to unblock downstream). Plans should include a validation gate: if mean keypoint error exceeds a threshold, fall back to the skeletonize backend for the affected camera-frame.
- **OBB training label conversion:** Existing fish bbox labels are axis-aligned. Converting to OBB 4-corner format for fine-tuning requires a defined process (re-annotation or automated conversion). Scope during Phase 2 or 4 planning.
- **`segmentation/training.py` disposition:** The research recommends `training/unet.py` either wraps or replaces `segmentation/training.py`. The deprecation/migration plan should be decided at the start of Phase 4 to avoid silent dual-maintenance.

## Sources

### Primary (HIGH confidence)
- Ultralytics OBB Task Documentation: https://docs.ultralytics.com/tasks/obb/ — OBB output format, xywhr convention, angle range
- Ultralytics OBB Datasets Overview: https://docs.ultralytics.com/datasets/obb/ — label format (4-corner normalized)
- Ultralytics issue #13003 "Is the angle value given by OBB correct?": https://github.com/ultralytics/ultralytics/issues/13003 — angle convention mismatch
- Ultralytics issue #16235 "YOLOv8-OBB angle conversion": https://github.com/ultralytics/ultralytics/issues/16235 — angle convention mismatch
- OpenCV Affine Transformations Tutorial: https://docs.opencv.org/4.x/d4/d61/tutorial_warp_affine.html — getRotationMatrix2D + warpAffine
- Click Entry Points Documentation: https://click.palletsprojects.com/en/latest/entry-points/ — CLI group/subgroup patterns
- Live codebase (all modules): authoritative ground truth for all interfaces, dataclasses, config structure, import boundaries

### Secondary (MEDIUM confidence)
- Learnable Triangulation of Human Pose (SAIC-violet): https://saic-violet.github.io/learnable-triangulation/ — confidence-weighted algebraic triangulation pattern for weighted DLT
- DLT Triangulation: Why Optimize? (arXiv 1907.11917): https://arxiv.org/pdf/1907.11917 — weighted vs unweighted DLT tradeoffs
- Boosting integral-based pose estimation (ScienceDirect 2024): https://www.sciencedirect.com/science/article/abs/pii/S0893608024004489 — comparison of heatmap vs integral vs direct regression at varying resolutions
- Integral Human Pose Regression (ECCV 2018): https://openaccess.thecvf.com/content_ECCV_2018/papers/Xiao_Sun_Integral_Human_Pose_ECCV_2018_paper.pdf — integral regression alternative

### Tertiary (LOW confidence)
- SimCC paper (ECCV 2022): https://github.com/leeyegy/SimCC — coordinate classification; rejected for this use case at 128x128 input size

---
*Research completed: 2026-02-28*
*Ready for roadmap: yes*
