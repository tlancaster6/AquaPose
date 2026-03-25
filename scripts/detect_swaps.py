"""Standalone swap detection validation script for the YH dataset.

Runs both seeded and scan detection modes on the Phase 72 baseline run,
validates against known ground-truth swap events, and produces a corrected
midlines_reid.h5 with /reid_events/ provenance.

Usage:
    hatch run python scripts/detect_swaps.py
"""

from __future__ import annotations

from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────
RUN_DIR = Path("~/aquapose/projects/YH/runs/run_20260307_140127/").expanduser()
PROJECTION_HEAD_PATH = RUN_DIR / "reid" / "fine_tune" / "best_reid_model.pt"

# Known ground-truth swap events
KNOWN_MF_SWAP = {
    "fish": {0, 5},
    "frame_range": (2600, 2700),
    "label": "MF (fish 0<->5)",
}
KNOWN_FF_SWAP = {"fish": {2, 4}, "frame_range": (550, 650), "label": "FF (fish 2<->4)"}

# Fish IDs involved in known swaps (for FP measurement)
SWAP_FISH_IDS = {0, 2, 4, 5}


def _event_matches_known(
    event_frame: int,
    event_fish: set[int],
    known: dict,
) -> bool:
    """Check if a detected event matches a known ground-truth swap."""
    fmin, fmax = known["frame_range"]
    return event_fish == known["fish"] and fmin <= event_frame <= fmax


def _print_diagnostic(
    label: str,
    fish_a: int,
    fish_b: int,
    frame_range: tuple[int, int],
    detector: object,
) -> None:
    """Print diagnostic info when a known swap is missed."""
    from aquapose.core.reid.swap_detector import _compute_mean_embedding

    print(f"\n  DIAGNOSTIC for {label}:")
    fmin, _fmax = frame_range
    for fid in [fish_a, fish_b]:
        pre = detector._get_embeddings(fid, fmin - 10, fmin - 1)
        post = detector._get_embeddings(fid, fmin, fmin + 10)
        n_pre = 0 if pre is None else pre.shape[0]
        n_post = 0 if post is None else post.shape[0]
        print(f"    Fish {fid}: {n_pre} pre-embeddings, {n_post} post-embeddings")

        if (
            pre is not None
            and post is not None
            and pre.shape[0] > 0
            and post.shape[0] > 0
        ):
            mean_pre = _compute_mean_embedding(pre)
            mean_post = _compute_mean_embedding(post)
            if mean_pre is not None and mean_post is not None:
                import numpy as np

                self_sim = float(np.dot(mean_pre, mean_post))
                print(f"    Fish {fid} self-similarity: {self_sim:.4f}")

    # Cross-similarities
    for fa, fb in [(fish_a, fish_b), (fish_b, fish_a)]:
        pre_a = detector._get_embeddings(fa, fmin - 10, fmin - 1)
        post_b = detector._get_embeddings(fb, fmin, fmin + 10)
        if pre_a is not None and post_b is not None:
            mean_pre_a = _compute_mean_embedding(pre_a)
            mean_post_b = _compute_mean_embedding(post_b)
            if mean_pre_a is not None and mean_post_b is not None:
                import numpy as np

                cross = float(np.dot(mean_pre_a, mean_post_b))
                print(f"    Cross {fa}-pre x {fb}-post: {cross:.4f}")


def main() -> None:
    """Run swap detection validation on YH data."""
    from aquapose.core.reid.swap_detector import SwapDetector, SwapDetectorConfig

    print("=" * 60)
    print("Swap Detection Validation Script")
    print("=" * 60)
    print(f"  Run dir: {RUN_DIR}")
    print(f"  Projection head: {PROJECTION_HEAD_PATH}")
    print(f"  Head exists: {PROJECTION_HEAD_PATH.exists()}")
    print()

    # Configure detector
    config = SwapDetectorConfig(
        cosine_margin_threshold=0.15,
        window_frames=10,
        proximity_threshold_m=0.15,
        scan_frame_stride=1,
        min_window_frames=3,
    )

    # Use projection head if available
    proj_path = PROJECTION_HEAD_PATH if PROJECTION_HEAD_PATH.exists() else None
    if proj_path is None:
        print("WARNING: Projection head not found, using raw 768-dim embeddings")
    print()

    detector = SwapDetector(RUN_DIR, config=config, projection_head_path=proj_path)

    # ── Run Seeded Mode ────────────────────────────────────────────
    print("--- Seeded Mode ---")
    seeded_events = detector.run(mode="seeded")
    n_confirmed_seeded = sum(1 for e in seeded_events if e.action == "confirmed")
    n_rejected_seeded = sum(1 for e in seeded_events if e.action == "rejected")
    n_skipped_seeded = sum(
        1 for e in seeded_events if e.action == "skipped_insufficient_data"
    )
    print(
        f"Seeded: {len(seeded_events)} events "
        f"({n_confirmed_seeded} confirmed, {n_rejected_seeded} rejected, "
        f"{n_skipped_seeded} skipped)"
    )
    for e in seeded_events:
        print(
            f"  frame={e.frame}, fish {e.fish_a}<->{e.fish_b}, "
            f"margin={e.cosine_margin:.4f}, action={e.action}"
        )
    print()

    # ── Run Scan Mode ──────────────────────────────────────────────
    print("--- Scan Mode ---")
    scan_events = detector.run(mode="scan")
    n_confirmed_scan = sum(1 for e in scan_events if e.action == "confirmed")
    n_rejected_scan = sum(1 for e in scan_events if e.action == "rejected")
    n_skipped_scan = sum(
        1 for e in scan_events if e.action == "skipped_insufficient_data"
    )
    print(
        f"Scan: {len(scan_events)} events "
        f"({n_confirmed_scan} confirmed, {n_rejected_scan} rejected, "
        f"{n_skipped_scan} skipped)"
    )
    for e in scan_events:
        print(
            f"  frame={e.frame}, fish {e.fish_a}<->{e.fish_b}, "
            f"margin={e.cosine_margin:.4f}, mode={e.detection_mode}, "
            f"action={e.action}"
        )
    print()

    # ── Validation Report ──────────────────────────────────────────
    print("=" * 60)
    print("=== Swap Detection Validation Report ===")
    print("=" * 60)
    print(
        f"Seeded mode: {len(seeded_events)} events "
        f"({n_confirmed_seeded} confirmed, {n_rejected_seeded} rejected, "
        f"{n_skipped_seeded} skipped)"
    )
    print(
        f"Scan mode: {len(scan_events)} events "
        f"({n_confirmed_scan} confirmed, {n_rejected_scan} rejected, "
        f"{n_skipped_scan} skipped)"
    )
    print()

    # Check known MF swap (seeded mode)
    mf_hit = False
    mf_margin = 0.0
    for e in seeded_events:
        if e.action in ("confirmed", "repaired") and _event_matches_known(
            e.frame, {e.fish_a, e.fish_b}, KNOWN_MF_SWAP
        ):
            mf_hit = True
            mf_margin = e.cosine_margin
            break
    # Also check scan events as fallback
    if not mf_hit:
        for e in scan_events:
            if e.action in ("confirmed", "repaired") and _event_matches_known(
                e.frame, {e.fish_a, e.fish_b}, KNOWN_MF_SWAP
            ):
                mf_hit = True
                mf_margin = e.cosine_margin
                break

    print("Known swap detection:")
    print(
        f"  MF (fish 0<->5, ~frame 2665): "
        f"{'HIT' if mf_hit else 'MISS'}"
        f"{f' (margin={mf_margin:.4f})' if mf_hit else ''}"
    )
    if not mf_hit:
        _print_diagnostic("MF swap", 0, 5, (2600, 2700), detector)

    # Check known FF swap (scan mode)
    ff_hit = False
    ff_margin = 0.0
    for e in scan_events:
        if e.action in ("confirmed", "repaired") and _event_matches_known(
            e.frame, {e.fish_a, e.fish_b}, KNOWN_FF_SWAP
        ):
            ff_hit = True
            ff_margin = e.cosine_margin
            break
    # Also check seeded events as fallback
    if not ff_hit:
        for e in seeded_events:
            if e.action in ("confirmed", "repaired") and _event_matches_known(
                e.frame, {e.fish_a, e.fish_b}, KNOWN_FF_SWAP
            ):
                ff_hit = True
                ff_margin = e.cosine_margin
                break

    print(
        f"  FF (fish 2<->4, ~frame 600):  "
        f"{'HIT' if ff_hit else 'MISS'}"
        f"{f' (margin={ff_margin:.4f})' if ff_hit else ''}"
    )
    if not ff_hit:
        _print_diagnostic("FF swap", 2, 4, (550, 650), detector)
    print()

    # False positive rate on clean segments
    all_events = seeded_events + scan_events
    confirmed_events = [e for e in all_events if e.action == "confirmed"]
    fp_events = [
        e
        for e in confirmed_events
        if not (
            _event_matches_known(e.frame, {e.fish_a, e.fish_b}, KNOWN_MF_SWAP)
            or _event_matches_known(e.frame, {e.fish_a, e.fish_b}, KNOWN_FF_SWAP)
        )
    ]

    # Total clean pair-tests = all confirmed events not matching known swaps
    total_confirmed = len(confirmed_events)
    n_fp = len(fp_events)
    fp_rate = n_fp / max(total_confirmed, 1)

    print(f"False positive rate: {n_fp}/{total_confirmed} = {fp_rate:.1%}")
    fp_pass = fp_rate < 0.05
    print(f"  {'PASS' if fp_pass else 'FAIL'} (threshold 5%)")
    if fp_events:
        print("  FP events:")
        for e in fp_events:
            print(
                f"    frame={e.frame}, fish {e.fish_a}<->{e.fish_b}, "
                f"margin={e.cosine_margin:.4f}, mode={e.detection_mode}"
            )
    print()

    # ── Merge and Repair ───────────────────────────────────────────
    # Combine events, deduplicate by (frame, fish_a, fish_b)
    seen: set[tuple[int, int, int]] = set()
    merged: list = []
    # Prefer scan events over seeded for duplicates
    for e in scan_events + seeded_events:
        key = (e.frame, min(e.fish_a, e.fish_b), max(e.fish_a, e.fish_b))
        if key not in seen:
            seen.add(key)
            merged.append(e)

    # Mark confirmed events as repaired
    from aquapose.core.reid.swap_detector import ReidEvent

    repaired_events = []
    for e in merged:
        if e.action == "confirmed":
            repaired_events.append(
                ReidEvent(
                    frame=e.frame,
                    fish_a=e.fish_a,
                    fish_b=e.fish_b,
                    cosine_margin=e.cosine_margin,
                    detection_mode=e.detection_mode,
                    action="repaired",
                )
            )
        else:
            repaired_events.append(e)

    output_path = detector.repair(repaired_events)
    n_repaired = sum(1 for e in repaired_events if e.action == "repaired")
    print(f"Repairs applied: {n_repaired}")
    print(f"Output: {output_path}")
    print()

    # ── Final Summary ──────────────────────────────────────────────
    print("=" * 60)
    print("Summary:")
    print(f"  MF swap detected: {'YES' if mf_hit else 'NO'}")
    print(f"  FF swap detected: {'YES' if ff_hit else 'NO'}")
    print(f"  FP rate: {fp_rate:.1%} ({'PASS' if fp_pass else 'FAIL'})")
    print(f"  Repairs applied: {n_repaired}")
    print(f"  Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
