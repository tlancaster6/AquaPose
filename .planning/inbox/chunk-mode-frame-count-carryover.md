# Chunk Mode: Carry Forward Cumulative Frame Count to Association Scoring

## Context

The association scoring formula uses an overlap reliability weight:

```
w = min(t_shared, effective_saturate) / effective_saturate
```

Where `effective_saturate = min(t_saturate, frame_count)`. This was fixed in
`2e825c8` to adapt to short runs — without it, runs shorter than `t_saturate`
(default 100) can never produce scores above `score_min`.

## Problem for Chunk Mode

When running in chunk mode, each chunk is processed as a short run. If
`frame_count` is set to the current chunk size (e.g. 30 frames), the fix works
for that chunk in isolation. But tracklet pairs that span multiple chunks
accumulate more temporal evidence than any single chunk reflects.

If chunk mode naively passes `frame_count = chunk_size` each time, the
reliability weight resets every chunk — a pair with 200 frames of evidence
across 7 chunks would be scored as if it only had 30.

## Required Handover

The chunk orchestrator must pass **cumulative frames processed so far** (across
all chunks) as `frame_count` to `score_all_pairs()`, not just the current
chunk's length. This ensures `w` reflects the full temporal evidence for each
tracklet pair.

Concretely: `frame_count = chunk_index * chunk_size + current_chunk_size`
(or track it as running state in the orchestrator).
