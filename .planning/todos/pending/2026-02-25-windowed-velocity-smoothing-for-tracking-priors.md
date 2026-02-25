---
created: 2026-02-25T02:23:44.880Z
title: Windowed velocity smoothing for tracking priors
area: tracking
files:
  - src/aquapose/tracking/tracker.py
  - src/aquapose/tracking/associate.py
---

## Problem

The tracker uses instantaneous velocity estimates as priors for prediction and association. Single-frame velocity is noisy — especially after occlusions, view transitions, or coasting — and contributes to jerk-freeze-die failure modes. Any process that consumes velocity priors (Kalman prediction, gating, cost matrices) inherits this noise.

A windowed (rolling) velocity estimate would provide smoother, more robust motion priors and reduce sensitivity to single-frame outliers.

## Solution

- Replace instantaneous velocity with exponentially-weighted or sliding-window average over the last N frames (e.g., 5-10)
- Apply to all consumers of velocity priors: prediction step, association gating, and any cost matrix that uses predicted position
- Handle edge cases: tracks with fewer frames than window size, tracks resuming after coasting
- Consider separate smoothing parameters for position vs heading velocity if applicable
