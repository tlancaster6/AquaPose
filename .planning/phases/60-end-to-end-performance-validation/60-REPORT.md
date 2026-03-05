# v3.4 Performance Validation Report

**Date:** 2026-03-05
**Baseline run:** run_20260304_180748 (pre-optimization)
**Post-optimization run:** run_20260304_221326

## Timing Comparison

| Stage | Before (s) | After (s) | Speedup |
|-------|-----------|----------|---------|
| Detection | 303.45 | 26.48 | 11.5x |
| Tracking | 3.47 | 3.27 | 1.1x |
| Association | 35.91 | 9.41 | 3.8x |
| Midline | 451.50 | 55.60 | 8.1x |
| Reconstruction | 119.89 | 17.10 | 7.0x |
| **TOTAL** | **914.22** | **111.86** | **8.2x** |

## Correctness Validation

**Result: FAIL**

The following eval metrics exceeded their tolerance thresholds:

| Stage | Metric | Baseline | Post | Tolerance | Delta |
|-------|--------|----------|------|-----------|-------|
| detection | total_detections | 14880 | 14879 | 0 | 1.0000 |
| detection | mean_confidence | 0.7891436143868392 | 0.7891379109841616 | 0.0 | 0.0000 |
| tracking | track_count | 133 | 131 | 0 | 2.0000 |
| tracking | detection_coverage | 0.7526411450641827 | 0.7545604364380292 | 0.0 | 0.0019 |
| association | fish_yield_ratio | 1.0572222222222223 | 0.9744444444444444 | 0.02 | 0.0828 |
| midline | total_midlines | 2410 | 2169 | 0 | 241.0000 |
| midline | mean_confidence | 0.8337252523105999 | 0.8316979044169275 | 0.001 | 0.0020 |
| reconstruction | fish_reconstructed | 1654 | 1746 | 2 | 92.0000 |
