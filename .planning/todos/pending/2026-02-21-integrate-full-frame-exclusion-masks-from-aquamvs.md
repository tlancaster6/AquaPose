---
created: 2026-02-21T15:10:32.680Z
title: Integrate full-frame exclusion masks from AquaMVS
area: calibration
files: []
---

## Problem

AquaMVS generates per-camera binary masks that mark regions of each camera view as excluded from analysis (e.g., tank edges, reflections, mounting hardware). AquaPose currently has no mechanism to incorporate these masks, meaning detection, segmentation, and triangulation can operate on invalid image regions — producing spurious detections or degraded mask quality near excluded areas.

These masks are static per camera (generated once from the rig geometry) and should be loaded alongside calibration data at pipeline startup.

## Solution

- Load optional per-camera exclusion masks early in the pipeline (alongside calibration data or as a separate config)
- Apply masks before detection: zero out excluded regions so YOLO/MOG2 never fires on them
- Apply masks before segmentation: ensure U-Net crops that overlap excluded regions have those pixels masked out
- Apply masks before triangulation: rays from excluded regions should not contribute to RANSAC consensus
- Masks should be strictly optional — pipeline works without them, uses them when available
- The machinery for *generating* these masks lives in AquaMVS, not AquaPose — AquaPose only loads and applies them
