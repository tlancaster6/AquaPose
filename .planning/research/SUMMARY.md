# Project Research Summary

**Project:** AquaPose v3.0 — Ultralytics Unification
**Domain:** Multi-view 3D fish pose estimation — replacing custom segmentation and keypoint models with Ultralytics-native YOLO backends
**Researched:** 2026-03-01
**Confidence:** HIGH

## Executive Summary

AquaPose v3.0 replaces two custom-trained models — a U-Net segmentor with IoU 0.623 and a custom keypoint regression head that underperformed even with augmentation — with Ultralytics-native YOLO11n-seg and YOLO11n-pose. The case for unification is compelling: `ultralytics>=8.1` is already in the dependency stack from v2.2, the training API is identical across detect/seg/pose tasks, no new pyproject.toml dependencies are required, and pretrained COCO backbone weights dramatically reduce the labeled data requirement for fine-tuning on fish. The primary deliverable is not a new capability — it is replacing fragile custom model infrastructure with a battle-tested unified training loop while improving model quality.

The recommended approach is a five-phase build order driven by data availability and inference pipeline dependencies. Data preparation tooling (SAM2 masks to YOLO polygon and pose label formats) must come first so model training can run as a long background process while pipeline integration proceeds in parallel. The architectural change is narrowly scoped: only `core/midline/backends/` and `segmentation/` are touched, with `Midline2D` and `AnnotatedDetection` contracts unchanged, meaning Stages 1, 2, 3, and 5 of the pipeline are entirely unaffected.

The primary risks are data quality issues, not architecture. SAM2 masks produce multi-region polygons that must be cleaned before YOLO seg training. YOLO pose label coordinates must be in full-frame-normalized space, not crop space — a silent but catastrophic error if missed. The small dataset (~150 annotated frames) requires conservative training hyperparameters and a proper temporal train/val split to avoid memorization. All three of these risks have clear, implementable prevention strategies documented in PITFALLS.md.

---

## Key Findings

### Recommended Stack

The stack requires no new dependencies. `ultralytics>=8.1` (current: 8.4.19) covers YOLO11 seg, YOLO11 pose, YOLO-OBB, and standard YOLO detection under a unified training API. YOLO11 (released September 2024) is the current recommended generation — it outperforms YOLOv8 at equal model size and uses identical annotation formats and training API calls.

**Core technologies:**
- `ultralytics>=8.1`: All three model types (detect, seg, pose) — one library, one training API, no additional dependencies
- `YOLO11n-seg.pt` (2.7M params): Instance segmentation replacing custom U-Net; pretrained COCO backbone; start nano and scale up only if val IoU < 0.80
- `YOLO11n-pose.pt` (2.9M params): Keypoint midline backend replacing custom regression head; use `kpt_shape: [6, 3]` with visibility flags; start from detect backbone (`yolo11n.pt`) since kpt_shape differs from COCO pretrained pose head
- `opencv-python>=4.8`: Mask-to-polygon contour extraction for annotation preparation (`cv2.findContours`, connected component filtering)
- `scikit-image>=0.22`: Skeletonization pipeline reused unchanged in the YOLO-seg backend path; also generates keypoint pseudo-labels from masks via arc-length sampling
- `scipy>=1.13`: Spline fitting and LM refinement in the reconstruction stage — completely unchanged

**What does NOT change:** PyTorch version constraints (no longer pinned to 2.4.1), SAM2 pseudo-label pipeline (generates source masks converted to YOLO format), all reconstruction and tracking libraries, all other `pyproject.toml` entries.

See `STACK.md` for full annotation format specifications, training configuration parameters, and model variant selection guidance.

### Expected Features

The milestone delivers swappable Ultralytics backends in MidlineStage (Stage 4) with a data preparation pipeline that converts existing SAM2 pseudo-labels to YOLO-native formats.

**Must have (table stakes):**
- `YOLOSegMidlineBackend` — YOLO seg inference on crop, binary mask output, feeds existing unchanged skeletonization path to `Midline2D`
- `YOLOPoseMidlineBackend` — YOLO pose inference on crop, keypoints mapped to `Midline2D(points, confidences)` with native per-keypoint confidence
- `convert_sam_masks_to_yolo_seg()` — batch conversion of SAM2 binary masks to YOLO polygon label files
- `generate_pose_labels_from_masks()` — SAM2 mask → skeletonize → arc-length sample 6 points → YOLO pose label with visibility=2
- `aquapose train yolo-seg` and `aquapose train yolo-pose` CLI subcommands using Ultralytics `model.train()` API
- Dataset YAML files for both tasks with correct `nc`, `names`, and `kpt_shape: [6, 3]` for pose
- `make_midline_backend()` factory updated with `"yolo_seg"` and `"yolo_pose"` branches
- Instance matching utility: `match_detections_to_tracks()` using IoU between YOLO result boxes and tracked bounding boxes

**Should have (differentiators):**
- Instance-aware masks from YOLO-seg: separates overlapping fish that confused the U-Net crop approach
- Native per-keypoint confidence from YOLO-pose: flows into existing confidence-weighted DLT triangulation without custom sigmoid head architecture
- Pretrained COCO backbone weights: dramatically reduces labeled data requirement for fish fine-tuning
- Single unified Ultralytics training API: eliminates custom training loop maintenance

**Defer to v3.1+:**
- YOLO11 upgrade from YOLOv8 (trivial API-compatible swap; validate v3.0 first; STACK.md recommends YOLO11 directly)
- Joint seg+pose combined inference pass (not officially supported by Ultralytics)
- Width profile extraction from YOLO-seg masks (useful but not blocking)
- Pose label quality filtering by skeletonizer branch count (improves training data; add after label pipeline works)
- Human annotation refinement of a keypoint subset

**Anti-features to avoid:**
- Running YOLO-seg on pre-cropped patches per-fish (defeats instance separation; run on crop but handle letterbox correctly)
- Using COCO pretrained pose weights directly with custom kpt_shape (head architecture mismatch — use detect backbone as base)
- Combining seg+pose into a single joint model (not supported by Ultralytics standard API)
- Including camera e3v8250 (wide-angle overhead) in training data (established skip from v1.0)

### Architecture Approach

The 5-stage pipeline structure, PipelineContext accumulator, and all inter-stage data contracts (`Midline2D`, `AnnotatedDetection`, `Detection`) are completely unchanged. All changes are confined to `core/midline/backends/` (two new backend classes replacing two deleted ones), `segmentation/` (new `YOLOSegInferencer` class), `training/` (new prep and training wrapper modules), and three new fields in `MidlineConfig`. Stages 1, 2, 3, and 5 are unaffected.

**Major components:**

1. `segmentation/yolo_seg.py` → `YOLOSegInferencer` — wraps Ultralytics YOLO-seg API; lives in `segmentation/` layer (same as `YOLODetector`) to respect AST-enforced import boundaries; loads model once at `__init__`; handles letterbox correction via `result.masks.xy` polygon output rather than raw `masks.data` (which is in padded inference space, not crop dimensions)

2. `core/midline/backends/yolo_seg.py` → `YOLOSegBackend` — invokes `YOLOSegInferencer.segment_batch()`, then feeds mask to unchanged skeletonization pipeline (`_skeleton_and_widths`, `_longest_path_bfs`, `_resample_arc_length`, `_crop_to_frame`); replaces `SegmentThenExtractBackend`

3. `core/midline/backends/yolo_pose.py` → `YOLOPoseBackend` — runs YOLO pose model on affine crop, extracts `(6, 3)` keypoint tensor (pixel coords in crop space), applies back-projection via existing `invert_affine_points()`, then feeds existing CubicSpline path; replaces `DirectPoseBackend`

4. `training/prep_seg.py` and `training/prep_pose.py` — annotation conversion scripts; SAM2 binary masks → YOLO polygon labels (seg), SAM2 masks → skeletonize → 6-keypoint YOLO pose labels (pose)

5. `training/yolo_seg.py` and `training/yolo_pose.py` — thin wrappers calling `ultralytics.YOLO(...).train()`; follow exact pattern of existing `training/yolo_obb.py`

**Deletion targets (only after integration tests pass):** `_UNet`, `UNetSegmentor`, `BinaryMaskDataset`, `train_unet()`, `_PoseModel`, `_KeypointHead`, `train_pose()`, `SegmentThenExtractBackend`, `DirectPoseBackend` — and legacy `MidlineConfig` fields `weights_path`, `keypoint_weights_path`, `n_keypoints`, `keypoint_t_values`, `keypoint_confidence_floor`, `min_observed_keypoints`.

**Critical architectural constraint:** The AST-based import boundary checker (enforced via pre-commit hook) prohibits `core/` from importing `engine/`. Backend classes must receive config values as primitives from `MidlineStage.__init__`, never importing `MidlineConfig` directly.

### Critical Pitfalls

The pitfall file covers three tiers: v3.0-specific (B-series), v2.2-era integration (A-series), and v1.0/v2.0/v2.1 foundation pitfalls. The top pitfalls for this milestone are:

1. **SAM2 multi-region masks produce invalid YOLO seg annotations (B1)** — SAM2 frequently generates disconnected mask regions for a single fish (especially low-contrast females). Naive `cv2.findContours` produces multiple polygons, which Ultralytics treats as multiple fish instances. Prevention: apply morphological closing before contour extraction; if still multi-region, keep only largest contour and log discarded area fraction; run Ultralytics dataset checker before any training begins.

2. **YOLO pose keypoints must be normalized to FULL IMAGE, not crop space (B2)** — The existing codebase is crop-centric; writing keypoints relative to the crop image causes the model to learn that all fish have keypoints in the top-left corner. Prevention: always back-project crop-space coordinates to frame space before normalization (`frame_x = crop_x1 + kp_x_in_crop`); add sanity check that all normalized coordinates are in [0, 1]; run training for 2–3 epochs and visualize predictions.

3. **YOLO-seg masks.data is in letterboxed space for non-square crops (B4)** — `result.masks.data` shape is `(N, 640, 640)` when inference runs on any crop at default `imgsz=640`; naive resize to crop dimensions produces the wrong mask placement. Prevention: use `result.masks.xy` (polygon coordinates already scaled to original image space) then rasterize with `cv2.fillPoly` to get binary mask in crop dimensions.

4. **Small dataset overfitting with Ultralytics default hyperparameters (B6)** — Default Ultralytics settings are tuned for COCO-scale datasets. With ~150 annotated frames, mosaic creates memorized image pairs; closing mosaic in the last 10 epochs removes the primary augmentation at convergence. Prevention: `close_mosaic=0`, `lr0=0.001`, `freeze=10` for first epochs, temporal train/val split (not random), start from pretrained COCO backbone.

5. **Breaking the existing working YOLO detection model (B7)** — Adding new training runs may overwrite existing detection weights via `project/name` collision, or an `ultralytics` version upgrade may break the detection inference API. Prevention: use distinct `project/name` for seg and pose training runs; pin Ultralytics version before adding new training code; run detection pipeline smoke test after any new Ultralytics code is added.

---

## Implications for Roadmap

Based on combined research, a five-phase build order is recommended, driven by data availability constraints and dependency ordering. The overarching principle: start training data preparation and kick off model training as early as possible (training takes hours to days) while pipeline integration proceeds in parallel.

### Phase 1: Annotation Conversion Tooling

**Rationale:** Training cannot begin without data in YOLO format. These scripts have zero dependency on pipeline code — they operate only on SAM2 masks and reconstruction utilities already in the codebase. Completing this phase and immediately kicking off model training maximizes the overlap between background training time and pipeline integration work.

**Delivers:** `training/prep_seg.py` (masks → YOLO polygon labels), `training/prep_pose.py` (masks → skeletonize → YOLO pose labels), dataset YAML files for both tasks, `aquapose prep-seg` and `aquapose prep-pose` CLI subcommands, visual spot-check of 20+ converted labels.

**Addresses features:** Seg polygon label generation (P1), Pose keypoint label generation (P1), Dataset YAMLs (P1)

**Avoids pitfalls:** B1 (SAM2 multi-region polygon cleaning), B2 (full-frame normalization for pose labels), B3 (kpt_shape in YAML), B8 (polygon point count validation)

**Research flag:** LOW — patterns are well-documented; implementation risk is data quality, not API uncertainty.

### Phase 2: Training Wrappers and Model Training

**Rationale:** Thin wrappers over `ultralytics.YOLO(...).train()` following the established `training/yolo_obb.py` precedent. Can be built and training kicked off while Phase 3 (inference backends) proceeds. Background training runs for hours; all subsequent phases can proceed without waiting.

**Delivers:** `training/yolo_seg.py` (`train_yolo_seg()`), `training/yolo_pose.py` (`train_yolo_pose()`), `aquapose train yolo-seg` and `aquapose train yolo-pose` CLI subcommands, first `best.pt` weights for seg and pose.

**Uses:** `ultralytics.YOLO.train()` unified API, `training/yolo_obb.py` as the direct pattern template

**Avoids pitfalls:** B6 (small dataset hyperparameters: `close_mosaic=0`, `lr0=0.001`, `freeze=10`), B7 (distinct project/name for each model type; detection smoke test after adding any training code)

**Research flag:** LOW — direct precedent in codebase (`yolo_obb.py`); well-documented Ultralytics API.

### Phase 3: Inference Backends

**Rationale:** Backends can be developed and unit-tested against pre-trained COCO weights (`yolo11n-seg.pt`, `yolo11n-pose.pt`) before fish-specific model training completes. Does not block on Phases 1 or 2 completion. The most architecturally sensitive phase because it touches import boundaries.

**Delivers:** `segmentation/yolo_seg.py` (`YOLOSegInferencer`), `core/midline/backends/yolo_seg.py` (`YOLOSegBackend`), `core/midline/backends/yolo_pose.py` (`YOLOPoseBackend`), `match_detections_to_tracks()` utility (shared between both backends), updated `get_backend()` registry, unit tests for both backends with mock models.

**Implements:** `YOLOSegInferencer` in `segmentation/` layer, both backends in `core/midline/backends/`, factory registry pattern, lazy import pattern for Ultralytics (inside `__init__`), eager model loading at init.

**Avoids pitfalls:** B4 (letterbox mask space — use `masks.xy` not `masks.data`), B5 (YOLO pose keypoints are in crop space — apply `invert_affine_points()` before `Midline2D` construction), AP0 / CLAUDE.md (CUDA tensor → always `.cpu().numpy()` immediately after Ultralytics result access)

**Research flag:** MEDIUM — the letterbox mask issue (B4) is a non-obvious Ultralytics behavior that must be explicitly tested; instance matching threshold tuning is empirical; import boundary constraints are project-specific and require care. Recommend verifying `result.masks.xy` behavior against current Ultralytics version before finalizing backend implementation.

### Phase 4: Config Wiring and Integration Test

**Rationale:** Backends exist; now wire them into the config system and validate end-to-end. Integration test must pass before any deletion begins. Config changes follow the existing `weights_path` pattern exactly.

**Delivers:** Three new fields in `MidlineConfig` (`yolo_seg_model_path`, `yolo_pose_model_path`, `yolo_imgsz`), updated `build_stages()` in `engine/pipeline.py`, updated `load_config()` path resolution, integration test running the full pipeline with `backend: "yolo_seg"` using COCO weights on a synthetic frame.

**Avoids pitfalls:** AP0 (import boundary — `MidlineStage` passes primitives to backends, never passes `MidlineConfig` object itself)

**Research flag:** LOW — config wiring follows exact existing pattern for `weights_path` field; integration test pattern established from prior phases.

### Phase 5: Deletion Pass

**Rationale:** Only after Phases 1–4 pass integration tests. Both old and new backends coexist during validation (selected via `backend` field in config); deletion removes the fallback. Deleting before validation leaves the pipeline broken with no fallback if issues arise.

**Delivers:** Deleted custom model code (`_UNet`, `UNetSegmentor`, `BinaryMaskDataset`, `train_unet()`, `_PoseModel`, `_KeypointHead`, `train_pose()`), deleted old backends (`SegmentThenExtractBackend`, `DirectPoseBackend`), removed legacy `MidlineConfig` fields, full test suite passing.

**Research flag:** LOW — deletion is mechanical; the only risk is incomplete test coverage of deleted code, which should be resolved in Phase 3 unit tests.

### Phase Ordering Rationale

- Data preparation comes first because model training takes hours and must run in the background — delaying it by even one phase adds direct clock time to the milestone.
- Training wrappers (Phase 2) and inference backends (Phase 3) are independent of each other; they can proceed concurrently if capacity allows.
- Integration test (Phase 4) gates the deletion pass (Phase 5) — no deletion before end-to-end validation.
- Old backends remain selectable via config throughout Phases 1–4, which means a failing Phase 3 or 4 does not break production runs.

### Research Flags

Phases likely needing deeper research during planning:

- **Phase 3 (Inference Backends):** The YOLO-seg letterbox mask correction (B4) and YOLO-pose crop-to-frame back-projection (B5) are implementation-sensitive; the exact API behavior of `result.masks.xy` vs `result.masks.data` should be verified against the current Ultralytics version during implementation.
- **Phase 3 (Instance Matching):** `match_detections_to_tracks()` is a new utility with no existing precedent in the codebase; IoU threshold (default 0.3) may need adjustment for fish in crowded schools.

Phases with standard patterns (skip research-phase):

- **Phase 1 (Annotation Conversion):** `cv2.findContours` → polygon normalization is well-understood; the SAM2 multi-region handling adds complexity but has a clear implementation.
- **Phase 2 (Training Wrappers):** Direct precedent in `training/yolo_obb.py`; Ultralytics training API is stable and documented.
- **Phase 4 (Config Wiring):** Follows exact existing pattern for `weights_path` field.
- **Phase 5 (Deletion):** Mechanical cleanup after validation.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Ultralytics official docs verified; no new dependencies; version constraint already satisfied; YOLO11 API confirmed identical to YOLOv8 |
| Features | HIGH | Ultralytics official docs verified; codebase read directly for interface contracts; fish-domain keypoint studies cited |
| Architecture | HIGH | Existing codebase read directly; import boundaries verified against AST hook; all data contracts confirmed unchanged |
| Pitfalls | HIGH | Codebase inspected for all integration points; Ultralytics GitHub issues cross-referenced for letterbox behavior, kpt_shape handling, and version upgrade risks |

**Overall confidence:** HIGH

### Gaps to Address

- **Instance matching threshold calibration:** The `match_detections_to_tracks()` IoU threshold (default 0.3) is a reasonable starting point but is unvalidated for the 9-fish cichlid school scenario where fish bounding boxes may overlap. Tune empirically after Phase 3 integration test.

- **Pose pseudo-label quality ceiling:** YOLO-pose model accuracy is bounded by the quality of skeletonizer-derived pseudo-labels. The skeletonizer on noisy SAM2 masks has known failure modes (multi-branch skeletons, disconnected arcs). A post-Phase-2 quality audit (filter labels by skeleton branch count) is recommended before committing to a final training run.

- **`yolo_imgsz` tuning for crop inference:** Default `imgsz=640` may be unnecessarily large for ~256x128 fish crops; `imgsz=256` or `imgsz=320` likely sufficient and halves inference time. Tune empirically after backends are integrated.

- **Female fish mask quality:** Low-contrast females were already a known challenge for U-Net. YOLO-seg with pretrained COCO backbone may handle this better, but is unvalidated. If YOLO-seg fails on females, increase `hsv_v=0.6` in training augmentation config (no albumentations — Ultralytics built-in augmentation covers this).

---

## Sources

### Primary (HIGH confidence)

- Ultralytics official documentation — seg task, pose task, dataset formats, training config, model variant comparison: https://docs.ultralytics.com/
- Ultralytics GitHub releases — version 8.4.19 (2026-02-28): https://github.com/ultralytics/ultralytics/releases
- Ultralytics GitHub issues — letterbox mask behavior (#4796), kpt_shape requirements (#1970), SAM2 mask conversion (#15380), OBB angle conventions (#13003, #16235)
- Existing codebase (read directly): `core/midline/stage.py`, `core/midline/backends/`, `segmentation/model.py`, `training/`, `engine/config.py`, `core/detection/backends/`

### Secondary (MEDIUM confidence)

- LearnOpenCV — animal pose estimation with YOLOv8: custom kpt_shape for non-human subjects, training workflow
- Roboflow Blog — custom YOLOv8 pose training workflow with annotated examples
- MDPI Marine Science 2024 — fish-domain YOLOv8-pose validation (albacore tuna, head/jaw/tail keypoints)
- Ultralytics community discussions — SAM2 mask → YOLO seg format conversion patterns

### Tertiary (LOW confidence)

- DmitryCS/yolov8_segment_pose (GitHub) — community joint seg+pose implementation; cited as anti-feature rationale only; not a recommended approach

---

*Research completed: 2026-03-01*
*Ready for roadmap: yes*
