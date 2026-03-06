---
created: 2026-02-24T05:28:56.570Z
title: Active calibration refinement
area: calibration
files:
  - src/aquapose/calibration/
---

## Problem

The AquaCal calibration is treated as fixed ground truth, but calibration parameters (intrinsics, extrinsics, refractive interface geometry) may drift or have residual errors that accumulate in multi-view triangulation. Small calibration inaccuracies are amplified by refraction — a slight error in the air-water interface position or camera extrinsics can shift projected rays significantly underwater.

Currently there is no mechanism to detect or correct calibration drift using the reconstruction pipeline's own observations.

## Solution

- **Bundle adjustment on fish observations**: Use consistently triangulated 3D points across multiple frames to refine extrinsic parameters, treating fish keypoints as sparse feature correspondences
- **Reprojection residual monitoring**: Track per-camera reprojection error over time to detect cameras with degraded calibration
- **Interface refinement**: Jointly optimize the refractive interface height/normal alongside camera extrinsics using multi-view consistency as the objective
- **Self-calibration loop**: After initial reconstruction, use high-confidence 3D reconstructions to refine calibration, then re-reconstruct — iterate until convergence
- Consider starting with extrinsics-only refinement (simplest) before attempting full refractive parameter optimization
