---
created: 2026-02-28T22:00:00.000Z
title: Extract frame status strings to constants or enum
area: tracking
files:
  - src/aquapose/engine/tracklet_trail_observer.py
  - src/aquapose/core/tracking/
---

## Problem

The frame status tags `"detected"` and `"coasted"` are used as raw string literals across multiple files — tracklet trail observer, tracking backends, and association logic. These are part of the tracking contract (tracklets carry per-frame status tags consumed by association for must-not-link constraints), but they're stringly-typed with no central definition.

## Solution

Define status constants (or a string enum) in `core/tracking/types.py` and use them everywhere. This prevents typos, enables IDE navigation, and documents the valid status values in one place.
