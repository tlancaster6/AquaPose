---
created: 2026-03-10T18:00:00.000Z
title: Generate augmentations on-the-fly at assembly time instead of storing in DB
area: training
files:
  - src/aquapose/training/store.py
  - src/aquapose/training/data_cli.py
  - src/aquapose/training/elastic_augment.py
---

## Problem

Storing augmented samples as static rows in the SampleStore is a footgun. Augmented children are indistinguishable from base samples in queries, which caused train/val leakage in the production retrain — augmented samples ended up in val, and parents in val had augmented children in train. The `--split-mode random` path is still vulnerable to this.

## Proposal

Generate elastic augmentations at assembly time rather than storing them in the database. `assemble()` would split base-only samples into train/val, then generate augmented variants on-the-fly for train samples only, writing them directly to the assembled dataset directory.

This eliminates the leakage class of bugs entirely — augmented data never enters the store, so it can't pollute val splits. Replaces `--max-aug-per-parent` with `--augment-count` and `--augment-angle-range`.
