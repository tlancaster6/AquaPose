---
created: 2026-03-11T13:45:08.889Z
title: Adapt pseudo-label pipeline for hard case mining
area: training
files:
  - src/aquapose/training/pseudo_label_cli.py
  - src/aquapose/training/frame_selection.py
  - src/aquapose/training/dataset_assembly.py
---

## Problem

The pseudo-label pipeline (`pseudo-label generate/select/assemble`) is designed around generating pseudo-labels for retraining — selecting high-confidence predictions as ground truth for the next training round. But the bigger value may be in **hard case mining**: identifying frames/crops where the model struggles (low confidence, high curvature, occlusion, unusual poses) and routing those to manual annotation or focused review.

Currently the pipeline filters *for* quality (high OKS, good detections) when selecting pseudo-labels. There's no mechanism to invert that logic — find the cases where the model is *worst* — and surface them for human attention.

## Solution

Refactor the pseudo-label pipeline to support a "hard case mining" mode:

1. **Invert selection criteria**: Add a `--hard-cases` flag (or separate subcommand) that selects frames where the model has low confidence, high pose uncertainty, detection mismatches across views, or association failures
2. **Scoring metrics for difficulty**: Use existing diagnostics (OKS scores, detection confidence, association singleton rate, centroid reprojection error) to rank frames by difficulty
3. **Export for review**: Output hard cases in a format suitable for Label Studio import (JSON with pre-filled predictions as suggestions) or as a simple image gallery with diagnostic overlays
4. **Integration with active learning loop**: Frame the pipeline as "run model → find hard cases → annotate → retrain" rather than "run model → trust model → retrain"

This could share most of the existing infrastructure (diagnostic caches, frame selection, dataset assembly) but with inverted selection logic and different output formatting.
