# exp/weight_prefetch — software prefetch in the Q4 decode loop

## Hypothesis

Decode GEMV streams weights once per token (~4 GB). Sequential access is
normally handled by the HW prefetcher, but our loop interleaves three
streams (nibbles, scales/zeros, activation reuse) plus a costly unpack, so
the core may stall waiting on DRAM. Explicit prefetch 2-4 rows ahead should
hide latency.

## Change

`q4_row_dot_i8` is per-row; prefetching must happen at the GEMV level where
the next row is known. Add `__builtin_prefetch(row_ptr + dist, 0, 0)` for
the packed data and scales of rows r+2..r+4 inside `gemv_q4`'s parallel body
(each thread prefetches its own upcoming rows).

## Result (measured Oct 2026, 6-core Zen4)

| config | bench 16k×4k Q4 | end-to-end tok/s |
|---|---|---|
| off | 27.9 GB/s (noisy ±25%) | 12.1–12.2 |
| dist=2 | 40.3 once, 0.8–1.0 typical | 11.6 |
| dist=4/8 | worse or neutral | — |

## Verdict: REJECTED

The micro-benchmark win didn't transfer: real model shapes stream smaller
rows where prefetch overhead dominates, and the 32 MB bench working set sits
exactly at L3 size (measurement noise, not signal). End-to-end is neutral to
slightly negative. Hook left in place (env-gated, default off) for future
hardware tuning; not part of the performance story.

## Follow-up (not this experiment)

The bench noise itself is a finding: size decode benchmarks clearly above L3
(≥128 MB working set) or report end-to-end tok/s instead.
