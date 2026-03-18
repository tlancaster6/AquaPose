"""Evaluate ablation models: OBB wall-fish holdout + pose curvature-stratified OKS."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# --- Paths ---
PROJECT = Path.home() / "aquapose/projects/YH"
RESULTS_DIR = Path(__file__).resolve().parent.parent / ".planning/results"
DATA_DIR = RESULTS_DIR / "data"

WALL_VAL_YAML = PROJECT / "training_data/obb/wall_augmented_val/dataset.yaml"

POSE_STORE_DB = PROJECT / "training_data/pose/store.db"
POSE_VAL_IMG_DIR = (
    PROJECT / "training_data/pose/datasets/pose_ablation_a_manual/images/val"
)
POSE_VAL_LBL_DIR = (
    PROJECT / "training_data/pose/datasets/pose_ablation_a_manual/labels/val"
)

OBB_MODELS: dict[str, Path] = {
    "A": PROJECT / "training/obb/run_20260317_232932/best_model.pt",
    "B": PROJECT / "training/obb/run_20260317_233530/best_model.pt",
    "C": PROJECT / "training/obb/run_20260317_234103/best_model.pt",
    "D": PROJECT / "training/obb/run_20260317_234549/best_model.pt",
    "E": PROJECT / "training/obb/run_20260317_235534/best_model.pt",
}

POSE_MODELS: dict[str, Path] = {
    "A": PROJECT / "training/pose/run_20260318_002353/best_model.pt",
    "B": PROJECT / "training/pose/run_20260318_003828/best_model.pt",
    "C": PROJECT / "training/pose/run_20260318_010419/best_model.pt",
    "D": PROJECT / "training/pose/run_20260318_013005/best_model.pt",
}

OBB_MODEL_LABELS = {
    "A": "Manual only",
    "B": "+ raw pseudo-labels",
    "C": "+ corrected pseudo-labels",
    "D": "+ hard cases",
    "E": "+ wall augmentation",
}

POSE_MODEL_LABELS = {
    "A": "Manual only",
    "B": "+ raw pseudo-labels",
    "C": "+ corrected pseudo-labels",
    "D": "+ elastic augmentation",
}


# --- OBB wall-fish holdout ---


def evaluate_obb_wall_holdout() -> dict[str, dict[str, float]]:
    """Run 5 OBB models on wall-fish val set, return metrics."""
    results: dict[str, dict[str, float]] = {}
    for name, weights in OBB_MODELS.items():
        print(f"[OBB] Evaluating model {name}: {weights.name}")
        model = YOLO(str(weights))
        metrics = model.val(data=str(WALL_VAL_YAML), imgsz=640, verbose=False)
        results[name] = {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        print(
            f"  mAP50={results[name]['mAP50']:.3f}  "
            f"mAP50-95={results[name]['mAP50-95']:.3f}"
        )
    return results


# --- Pose curvature-stratified OKS ---


def get_val_curvatures() -> dict[str, float]:
    """Query pose store for val sample curvatures. Returns {filename: curvature}."""
    conn = sqlite3.connect(str(POSE_STORE_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT image_path, metadata FROM samples
           WHERE EXISTS (SELECT 1 FROM json_each(tags) WHERE json_each.value = 'val')
           AND NOT EXISTS (SELECT 1 FROM json_each(tags) WHERE json_each.value = 'excluded')"""
    ).fetchall()
    conn.close()

    curvatures: dict[str, float] = {}
    for r in rows:
        meta = json.loads(r["metadata"])
        filename = Path(r["image_path"]).name
        curvatures[filename] = meta["curvature"]
    return curvatures


def parse_yolo_pose_label(
    label_path: Path, img_w: int, img_h: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Parse YOLO pose label file.

    Returns:
        List of (bbox_xyxy, keypoints_xy, visibility) tuples.
        bbox_xyxy is in pixels, keypoints_xy is in pixels, visibility is int array.
    """
    instances = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # cls cx cy w h (x y v)*6
            cx, cy, w, h = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
            # Convert to pixel bbox
            px_cx, px_cy = cx * img_w, cy * img_h
            px_w, px_h = w * img_w, h * img_h
            bbox = np.array(
                [
                    px_cx - px_w / 2,
                    px_cy - px_h / 2,
                    px_cx + px_w / 2,
                    px_cy + px_h / 2,
                ]
            )

            kpt_data = parts[5:]
            n_kpts = len(kpt_data) // 3
            kpts_xy = np.zeros((n_kpts, 2))
            vis = np.zeros(n_kpts, dtype=int)
            for k in range(n_kpts):
                kpts_xy[k, 0] = float(kpt_data[k * 3]) * img_w
                kpts_xy[k, 1] = float(kpt_data[k * 3 + 1]) * img_h
                vis[k] = int(float(kpt_data[k * 3 + 2]))
            instances.append((bbox, kpts_xy, vis))
    return instances


def compute_oks(
    gt_kpts: np.ndarray,
    pred_kpts: np.ndarray,
    gt_vis: np.ndarray,
    scale: float,
    sigma: float = 0.1,
) -> float:
    """Compute OKS for a single instance.

    Args:
        gt_kpts: (K, 2) ground-truth keypoints.
        pred_kpts: (K, 2) predicted keypoints.
        gt_vis: (K,) visibility flags (>0 means visible).
        scale: Object scale (sqrt of bbox area).
        sigma: Per-keypoint constant.

    Returns:
        OKS score in [0, 1].
    """
    visible = gt_vis > 0
    if not np.any(visible):
        return 0.0

    d2 = np.sum((gt_kpts[visible] - pred_kpts[visible]) ** 2, axis=1)
    oks_per_kpt = np.exp(-d2 / (2.0 * scale**2 * sigma**2))
    return float(np.mean(oks_per_kpt))


def _match_predictions_to_gt(
    gt_bboxes: list[np.ndarray],
    pred_bboxes: np.ndarray,
) -> list[int | None]:
    """Greedy match GT instances to predictions by IoU of their bboxes.

    Returns:
        List of pred indices (or None) for each GT instance.
    """
    from itertools import product

    n_gt = len(gt_bboxes)
    n_pred = len(pred_bboxes)
    if n_pred == 0:
        return [None] * n_gt

    # Compute IoU matrix
    iou_matrix = np.zeros((n_gt, n_pred))
    for gi, pi in product(range(n_gt), range(n_pred)):
        g = gt_bboxes[gi]
        p = pred_bboxes[pi]
        x1 = max(g[0], p[0])
        y1 = max(g[1], p[1])
        x2 = min(g[2], p[2])
        y2 = min(g[3], p[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_g = (g[2] - g[0]) * (g[3] - g[1])
        area_p = (p[2] - p[0]) * (p[3] - p[1])
        union = area_g + area_p - inter
        iou_matrix[gi, pi] = inter / union if union > 0 else 0

    # Greedy assignment
    matches: list[int | None] = [None] * n_gt
    used_preds: set[int] = set()
    # Sort by descending IoU
    indices = np.argwhere(iou_matrix > 0.3)
    if len(indices) == 0:
        return matches
    ious = iou_matrix[indices[:, 0], indices[:, 1]]
    order = np.argsort(-ious)
    for idx in order:
        gi, pi = int(indices[idx, 0]), int(indices[idx, 1])
        if matches[gi] is None and pi not in used_preds:
            matches[gi] = pi
            used_preds.add(pi)
    return matches


def evaluate_pose_curvature() -> tuple[
    dict[str, dict[str, float]], dict[str, list[tuple[str, float, float]]]
]:
    """Run 4 pose models on val set, compute per-instance OKS, bin by curvature tercile.

    Returns:
        Tuple of:
        - Dict[model_name, Dict[tercile_name, mean_oks]] (aggregated)
        - Dict[model_name, list[(filename, curvature, oks)]] (per-instance)
    """
    curvatures = get_val_curvatures()
    print(f"[Pose] Loaded curvatures for {len(curvatures)} val samples")

    # Get val image files and their GT labels
    val_images = sorted(POSE_VAL_IMG_DIR.glob("*.jpg"))
    print(f"[Pose] Found {len(val_images)} val images")

    # Build per-image curvature list
    img_curvatures = []
    for img_path in val_images:
        fname = img_path.name
        if fname not in curvatures:
            raise ValueError(f"No curvature for {fname}")
        img_curvatures.append(curvatures[fname])
    img_curvatures_arr = np.array(img_curvatures)

    # Compute tercile boundaries
    t1 = np.percentile(img_curvatures_arr, 33.33)
    t2 = np.percentile(img_curvatures_arr, 66.67)
    print(
        f"[Pose] Curvature tercile boundaries: low<{t1:.4f}, mid<{t2:.4f}, high>={t2:.4f}"
    )

    def get_tercile(c: float) -> str:
        if c < t1:
            return "low"
        if c < t2:
            return "mid"
        return "high"

    # Read image dimensions (assume all same size, read first)
    from PIL import Image

    with Image.open(val_images[0]) as im:
        img_w, img_h = im.size

    # Parse all GT labels
    gt_per_image: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for img_path in val_images:
        lbl_path = POSE_VAL_LBL_DIR / (img_path.stem + ".txt")
        gt_per_image.append(parse_yolo_pose_label(lbl_path, img_w, img_h))

    results: dict[str, dict[str, float]] = {}
    all_per_instance: dict[str, list[tuple[str, float, float]]] = {}

    for model_name, weights in POSE_MODELS.items():
        print(f"[Pose] Evaluating model {model_name}: {weights.name}")
        model = YOLO(str(weights))

        # Collect per-instance OKS with curvature
        instance_data: list[tuple[str, float]] = []  # (tercile, oks)
        per_instance: list[tuple[str, float, float]] = []  # (filename, curvature, oks)

        for i, img_path in enumerate(val_images):
            preds = model.predict(str(img_path), imgsz=320, verbose=False, conf=0.25)
            pred_result = preds[0]
            gt_instances = gt_per_image[i]
            curv = img_curvatures[i]
            tercile = get_tercile(curv)

            if len(gt_instances) == 0:
                continue

            # Get pred boxes and keypoints
            if pred_result.keypoints is not None and len(pred_result.boxes) > 0:
                pred_boxes = pred_result.boxes.xyxy.cpu().numpy()
                pred_kpts = pred_result.keypoints.xy.cpu().numpy()  # (N, K, 2)
            else:
                pred_boxes = np.empty((0, 4))
                pred_kpts = np.empty((0, 6, 2))

            gt_bboxes = [inst[0] for inst in gt_instances]
            matches = _match_predictions_to_gt(gt_bboxes, pred_boxes)

            for gi, pi in enumerate(matches):
                _, gt_kp, gt_vis = gt_instances[gi]
                bbox = gt_instances[gi][0]
                scale = np.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

                if pi is None:
                    oks = 0.0
                else:
                    oks = compute_oks(gt_kp, pred_kpts[pi], gt_vis, scale)

                instance_data.append((tercile, oks))
                per_instance.append((img_path.stem, curv, oks))

        all_per_instance[model_name] = per_instance

        # Aggregate by tercile
        tercile_oks: dict[str, list[float]] = {"low": [], "mid": [], "high": []}
        for terc, oks in instance_data:
            tercile_oks[terc].append(oks)

        model_results: dict[str, float] = {}
        for terc in ["low", "mid", "high"]:
            vals = tercile_oks[terc]
            mean_oks = float(np.mean(vals)) if vals else 0.0
            model_results[terc] = mean_oks
            print(f"  {terc}: n={len(vals)}, mean_OKS={mean_oks:.3f}")
        results[model_name] = model_results

    return results, all_per_instance


# --- Output ---


def write_obb_csv(results: dict[str, dict[str, float]]) -> None:
    """Write OBB wall-fish holdout results CSV."""
    csv_path = DATA_DIR / "obb_ablation_wall_holdout.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "data_added", "mAP50", "mAP50-95", "precision", "recall"]
        )
        for name in sorted(results):
            r = results[name]
            writer.writerow(
                [
                    name,
                    OBB_MODEL_LABELS[name],
                    f"{r['mAP50']:.3f}",
                    f"{r['mAP50-95']:.3f}",
                    f"{r['precision']:.3f}",
                    f"{r['recall']:.3f}",
                ]
            )
    print(f"[CSV] Wrote {csv_path}")


def write_pose_csv(results: dict[str, dict[str, float]]) -> None:
    """Write pose curvature-stratified OKS results CSV."""
    csv_path = DATA_DIR / "pose_ablation_curvature_oks.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "data_added", "low_oks", "mid_oks", "high_oks"])
        for name in sorted(results):
            r = results[name]
            writer.writerow(
                [
                    name,
                    POSE_MODEL_LABELS[name],
                    f"{r['low']:.3f}",
                    f"{r['mid']:.3f}",
                    f"{r['high']:.3f}",
                ]
            )
    print(f"[CSV] Wrote {csv_path}")


def write_pose_per_instance_csv(
    per_instance: dict[str, list[tuple[str, float, float]]],
) -> None:
    """Write per-instance curvature + OKS CSV for scatter plotting."""
    csv_path = DATA_DIR / "pose_ablation_oks_per_instance.csv"
    # One row per instance, with OKS columns for each model
    # First, collect all (stem, curvature) pairs in order from model A
    ref_model = "A"
    ref_data = per_instance[ref_model]
    model_names = sorted(per_instance)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["stem", "curvature"] + [f"oks_{m}" for m in model_names]
        writer.writerow(header)

        # Build lookup: model -> {stem: oks}
        lookups: dict[str, dict[str, float]] = {}
        for m in model_names:
            lookups[m] = {stem: oks for stem, _curv, oks in per_instance[m]}

        for stem, curv, _oks in ref_data:
            row = [stem, f"{curv:.6f}"]
            for m in model_names:
                row.append(f"{lookups[m].get(stem, 0.0):.4f}")
            writer.writerow(row)

    print(f"[CSV] Wrote {csv_path}")


def update_performance_accuracy(
    obb_results: dict[str, dict[str, float]],
    pose_results: dict[str, dict[str, float]],
) -> None:
    """Append results to Sections 13 and 14 of performance-accuracy.md."""
    md_path = RESULTS_DIR / "performance-accuracy.md"
    content = md_path.read_text()

    # --- Add wall-fish holdout CSV ref to Section 13 (idempotent) ---
    csv_ref_13 = "**CSV (wall-fish holdout)**: `data/obb_ablation_wall_holdout.csv`"
    if csv_ref_13 not in content:
        marker_13 = "**Assembly script**: `scripts/assemble_ablation_datasets.py`"
        if marker_13 in content:
            content = content.replace(marker_13, csv_ref_13 + "\n" + marker_13)

    # --- Add curvature-stratified OKS to Section 14 (idempotent) ---
    curv_marker = "#### Curvature-Stratified OKS (Val Set, 128 images)"
    if curv_marker not in content:
        curvatures = get_val_curvatures()
        curv_arr = np.array(sorted(curvatures.values()))
        t1 = np.percentile(curv_arr, 33.33)
        t2 = np.percentile(curv_arr, 66.67)
        n_low = int(np.sum(curv_arr < t1))
        n_mid = int(np.sum((curv_arr >= t1) & (curv_arr < t2)))
        n_high = int(np.sum(curv_arr >= t2))

        curv_section = f"""
{curv_marker}

Per-instance OKS computed for each model, binned by ground-truth midline curvature tercile (sigma=0.1).

| Model | Data Added | Low (n={n_low}) | Mid (n={n_mid}) | High (n={n_high}) |
|-------|-----------|{"------|" * 3}"""

        for name in sorted(pose_results):
            r = pose_results[name]
            best_markers = {}
            for terc in ["low", "mid", "high"]:
                best_val = max(pose_results[m][terc] for m in pose_results)
                best_markers[terc] = "**" if r[terc] == best_val else ""
            curv_section += (
                f"\n| {name} | {POSE_MODEL_LABELS[name]} | "
                f"{best_markers['low']}{r['low']:.3f}{best_markers['low']} | "
                f"{best_markers['mid']}{r['mid']:.3f}{best_markers['mid']} | "
                f"{best_markers['high']}{r['high']:.3f}{best_markers['high']} |"
            )

        curv_section += (
            "\n\n**CSV**: `data/pose_ablation_curvature_oks.csv`\n"
            "**CSV (per-instance)**: `data/pose_ablation_oks_per_instance.csv`\n"
        )

        # Insert before run details in Section 14
        marker_14 = "### Run Details\n\n| Model | Run ID | Weights |\n|-------|--------|---------|"
        sec14_start = content.find("## 14. Pose Training Data Ablation")
        if sec14_start >= 0:
            run_details_pos = content.find(marker_14, sec14_start)
            if run_details_pos >= 0:
                content = (
                    content[:run_details_pos]
                    + curv_section
                    + "\n"
                    + content[run_details_pos:]
                )

    md_path.write_text(content)
    print(f"[MD] Updated {md_path}")


def print_obb_table(results: dict[str, dict[str, float]]) -> None:
    """Print OBB wall-fish holdout results table."""
    print("\n=== OBB Wall-Fish Holdout Results ===")
    print(
        f"{'Model':<8} {'Data Added':<30} {'mAP50':>8} {'mAP50-95':>10} {'Prec':>8} {'Recall':>8}"
    )
    print("-" * 78)
    for name in sorted(results):
        r = results[name]
        print(
            f"{name:<8} {OBB_MODEL_LABELS[name]:<30} "
            f"{r['mAP50']:>8.3f} {r['mAP50-95']:>10.3f} "
            f"{r['precision']:>8.3f} {r['recall']:>8.3f}"
        )


def print_pose_table(results: dict[str, dict[str, float]]) -> None:
    """Print pose curvature-stratified OKS results table."""
    print("\n=== Pose Curvature-Stratified OKS ===")
    print(f"{'Model':<8} {'Data Added':<30} {'Low':>8} {'Mid':>8} {'High':>8}")
    print("-" * 68)
    for name in sorted(results):
        r = results[name]
        print(
            f"{name:<8} {POSE_MODEL_LABELS[name]:<30} "
            f"{r['low']:>8.3f} {r['mid']:>8.3f} {r['high']:>8.3f}"
        )


def main() -> None:
    """Run both ablation evaluations."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Part 1: OBB Wall-Fish Holdout Evaluation")
    print("=" * 60)
    obb_results = evaluate_obb_wall_holdout()
    print_obb_table(obb_results)
    write_obb_csv(obb_results)

    print("\n" + "=" * 60)
    print("Part 2: Pose Curvature-Stratified OKS")
    print("=" * 60)
    pose_results, pose_per_instance = evaluate_pose_curvature()
    print_pose_table(pose_results)
    write_pose_csv(pose_results)
    write_pose_per_instance_csv(pose_per_instance)

    print("\n" + "=" * 60)
    print("Updating performance-accuracy.md")
    print("=" * 60)
    update_performance_accuracy(obb_results, pose_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
