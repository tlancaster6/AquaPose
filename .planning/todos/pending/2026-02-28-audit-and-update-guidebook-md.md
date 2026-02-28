---
created: 2026-02-28T13:03:12.289Z
title: Audit and update GUIDEBOOK.md
area: planning
files:
  - .planning/GUIDEBOOK.md
---

## Problem

The GUIDEBOOK.md was written during the v2.0→v2.1 transition to document architectural decisions, refactoring context, and design rationale. Now that v2.1 Identity milestone is complete, the guidebook likely contains outdated references, stale TODOs, and sections that no longer reflect the current codebase state. It needs a thorough audit to ensure it remains accurate and useful for future development sessions.

## Solution

1. Read through GUIDEBOOK.md and cross-reference with current codebase state
2. Remove or update stale references to old code (FishTracker, ransac_centroid_cluster, etc.)
3. Update architectural descriptions to match the final v2.1 implementation
4. Ensure domain conventions and design decisions are still accurate
5. Add any new patterns or conventions discovered during v2.1 development
