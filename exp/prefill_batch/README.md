# exp/prefill_batch — one weight-streaming pass per layer for the whole prompt

## Hypothesis

Today the prompt is fed token-by-token through the decode path: weights are
re-streamed M times (~4 GB × M). A batched prefill streams weights ONCE and
reuses them across M tokens. Same FLOPs, ~M× less traffic. GEMMs go from
memory-bound (decode) to compute-bound (prefill); VNNI int8 throughput
(~1 TOPS here) decides the ceiling.

Estimate (M=150, 42 layers): sequential ~20 s (600 GB @ ~50 GB/s effective);
batched ~3–6 s (4 GB traffic + 2.4 TFLOP @ VNNI throughput). Expected 3–6×
faster prefill. Decode unchanged.

## Design

- `gemm_q4_batch` / `gemm_q8_batch` (in `gemm.h`, permanent home — they are
  general kernels, not a fork): quantize each activation row once (same
  per-32 roundf path as decode, so numerics track), then for each output row
  dot all M tokens. Weight row streamed once.
- `attn_prefill_gqa`: causal (+SWA window) attention over the filled cache,
  parallel over (head, position). O(M²·d) — trivial next to GEMMs for M ≤ 4k.
- `Runner::forward_prefill`: per-layer pipeline reusing the decode sub-steps;
  norms/elementwise/RoPE stay per-token loops (their traffic is tiny).
- Correctness: batched prefill must agree with sequential decode logits to
  gate tolerance (same math, different summation order). Checked by the gate
  (`MOLOT_SEQ_PREFILL=1` forces the old sequential loop for A/B).

## Status: MERGED ✓ (gate PASS 4/5, no degeneracy)

Measured Oct 2026, 6-core Zen4, gemma-4B, 302-token prompt:

| path | prefill time | top-1 vs sequential |
|---|---|---|
| sequential decode loop | 27.2 s | — |
| batched prefill | 13.3 s (**2.05×**) | agrees (cos 0.997, same argmax) |

Short prompts (≤10 tokens): overhead-dominated, no measurable difference
(0.2–0.3 s either way). Wins grow with prompt length; compute-bound past
~100 tokens (VNNI throughput is the ceiling, not traffic).

Bugs found during development (both fixed, both would have shipped silently
without the prefill-vs-sequential equivalence check):
1. Per-branch `out_scale` applied twice in prefill (reference scales the
   whole block once at the end) + a duplicated attention block from a bad
   debug edit.
2. Missing V plain-norm in prefill (present in decode).

## Follow-ups (not this experiment)

- Share activation quantization across QKV (same input normed once) instead
  of re-quantizing per GEMM — ~10% of prefill time is redundant quant work.
- Scalar exp in prefill attention → vector path (decode already has it).
- Push VNNI throughput (currently ~360 effective GOPS vs ~1 TOPS ceiling).
