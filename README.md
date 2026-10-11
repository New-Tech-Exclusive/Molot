# Molot — CPU ML runtime (x86_64) that competes with GPUs where it counts

Batch=1 LLM decode is **bandwidth-bound on every platform**. A 6-core Zen4
streams ~55 GB/s; an H100 streams ~3000 GB/s but costs 15× more and sits idle
at low concurrency. With 4-bit weights, fused decode, and GQA/paged KV, the
CPU's share is shippable — and this repo measures it instead of claiming it.

## Build & run

```sh
make        # g++ -O3 -march=native -fopenmp, header-only lib in include/
./build/bench
```

No dependencies beyond GCC ≥ 12 and AVX512 (VNNI + BW for the fast paths;
scalar fallbacks compile everywhere, slower).

## What it proves (measured on 6-core Ryzen 7600, all DRAM-resident)

| kernel | result | note |
|---|---|---|
| GEMV FP32 16k×4k | 53.6 GB/s, 1.2× on 6 cores | DRAM-bound: threads can't beat the bus |
| GEMV Q8 (VNNI) | **≈50–60 GB/s, ~4× FP32** | ≈ bus max; activation quantized once/token |
| GEMV Q4 GS=32 | **≈26 GB/s, ~4× FP32** | i8-VNNI ≈ float (shuffle≈cvt); exact 2e-8; GGUF Q4_0 (zero-free) favors i8 |
| GQA attn decode | ~230 GB/s L3-hot, 59 GB/s cold 16k | O(N·d); hot = continuous-batching steady state |
| Prefill blocked | bit-exact, +1.1–1.4× at 256×4096² | Bt traffic ÷Mc; gap grows with Bt/L3 |
| Vector exp | relerr 1.5e-6 | scalar exp would own the profile past 8k ctx |
| Prefill GEMM | ~277 GFLOPS FP32 | transposed-B contiguous dots (strided-B was 6) |
| RMSNorm / AdamW | 84 / 50 GB/s | fused, no temps |
| **8B-Q4 extrap.** | **~6 tok/s batch=1** | 4.4 GB/token ÷ 28 GB/s; 70B ≈ 1 tok/s/socket |

Correctness is checked in the same binary (FP32 vs scalar, Q8/Q4 NRMSE
1%/8%, attention vs reference to 1e-8, AdamW bit-exact) — it exits nonzero
on failure.

## Layout

```
DESIGN.md            # full architecture: where CPU wins, fusion, memory, serving
include/molot/
  thread_pool.h      # persistent pinned pool, caller-participates parallel_for
  quant.h            # Q8 (per-row + per-32 block sums), Q4 GS=32, BF16 helpers
  gemm.h             # VNNI Q8 GEMV, vector-unpack Q4 GEMV, transposed prefill GEMM
  attn.h             # fused RoPE+GQA decode attention, online softmax
  norm.h             # RMSNorm + fused GELU-mul + clamped vector exp
  adam.h             # fused AdamW (LoRA training path)
  gguf.h             # GGUF v1/v2/v3 mmap reader (no dependencies)
  repack.h           # GGUF blocks -> runtime layouts (Q4_0 exact incl. nibble order)
  model.h            # arch-generic loader (GGUF + .molot pack), tolerates absence
  forward.h          # single-token forward: GQA/SWA/shared-KV, QK-norms, pe path
  bpe.h              # byte-level BPE from GGUF tokenizer arrays
  sample.h           # greedy / temperature / top-k / top-p
  pack.h             # .molot pack format (mmap-ready derived cache)
bench/bench.cc       # correctness + DRAM-resident benchmarks + model estimate
tools/
  gguf-info          # GGUF integration checklist (arch, dtypes, block presence)
  gguf2molot         # GGUF -> .molot converter with bit-exact --verify
  molot-run          # single-prompt runner (GGUF or .molot, auto-detected)
  molot-chat         # interactive chat REPL with streaming output
```

## Chat

```sh
./build/gguf2molot model.gguf            # -> model.molot (verified)
./build/molot-chat model.molot -t 0.7    # or .gguf directly
```

Chat ships with a default system prompt (`You are a helpful AI assistant.`,
override with `--sys`, disable with `--sys ""`) — without role grounding
an instruct model just continues text instead of assisting. Turn framing
follows the file's chat template literally (`<|turn>role / content /
<turn|>`, marker ids resolved from vocab with a raw fallback). Generation
stops at EOS or the turn-close marker. Sampling defaults (`-t 0.7`,
repetition penalty 1.1) are load-bearing: greedy decoding amplifies small
quant noise into loops on knife-edge calls.

## GGUF vs .molot

GGUF is the interchange artifact (portable, every runner reads it, 30+
quant types, full tokenizer + template). `.molot` is a derived runtime
cache: tensors pre-repacked into exactly the layouts our kernels consume
(sequential nibbles + scales/zeros/sums, per-row Q8 + block sums, baked
rope tables, requantized head), 64B-aligned, mmap-ready. Load = map +
memcpy (≈1.3s vs 2.3s here, mostly embedding widen); no repack compute.
The pack records source hash + size for provenance; the loader trusts it
(documented cache semantics — GGUF remains source of truth).

## Model status (gemma-4B Q4_0)

End-to-end verified against a reference oracle (logit + per-block probes):
exact paths track token-for-token. The fast int path is mathematically
faithful (kernels match exact quantized math to ~1e-8) but greedy decoding
can loop on knife-edge calls — use sampling (`-t 0.7`, the chat default).
known scope limits: no Jinja template rendering (raw turn concatenation).
GGUF Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K tensors are dequantized and repacked
on load. Standard expert tensors (`ffn_*_exps.weight` plus
`ffn_gate_inp.weight`) are routed with top-k softmax; MoE prefill currently
uses the correct decode path sequentially. `.molot` packs remain a dense
cache format, so MoE models should be loaded directly from GGUF.
Qwen3.5 hybrid `qwen35moe` models, including Ornith 35B-A3B, are not yet
supported because they require SSM block execution and fused `attn_qkv`
handling.

## Honest limits

* Q4 unpack is ALU-capped at ~28 GB/s on Zen4 (scalar was 3.3). AMX or
  LUT-dequant closes the gap to the ~55 GB/s bus.
* This is a decode-first runtime. Dense large-batch training still belongs
  on GPUs; the training story here is QLoRA (frozen Q4 base, fused AdamW on
  MB-sized adapters — measured 50 GB/s).
* `tok/s ≈ eff_BW / bytes_per_token` is a roofline, not a full engine:
  sampling, scheduling, and continuous batching are specified in DESIGN.md
  §4, not yet implemented.
