# Molot — CPU ML Runtime to Compete with GPUs

## 1. Where CPU can win (and where it can't)

Raw roofline on a 6-core Zen4 (this machine) vs H100:

* CPU: ~60 GB/s DRAM, ~30 MB L3, ~300 GFLOPS BF16, 128–512 GB RAM, $2k box.
* H100: ~3000 GB/s HBM, ~1000 TFLOPS FP16, 80 GB RAM, $30k.

CPU **never** wins dense large-batch FP16 GEMM (50× bandwidth gap, 3000×
compute gap). It wins where the GPU is also bandwidth-bound, under-utilized,
or capacity-constrained:

1. **Batch=1–8 LLM decode.** Arithmetic intensity ≈ 1–2 FLOP/byte
   (stream weights once per token). Both platforms are memory-bound, so the
   gap is ~bandwidth ratio, not FLOP ratio. 4-bit weights shrink it 4× more.
2. **Models bigger than one GPU.** 70B Q4 ≈ 40 GB fits in CPU RAM for free;
   on GPU it needs 2× cards or offload with PCIe stalls.
3. **MoE / GQA / long-context / sparse.** Only a slice is active per token.
   CPU keeps all experts in cheap DDR; GPU must fit or swap.
4. **$/token at low concurrency.** Chat, agents, RAG, on-prem: 5–15 tok/s
   on CPU is shippable; GPU sits idle.
5. **LoRA / QLoRA fine-tune.** Frozen Q4 base + tiny BF16 adapters.
   Backward is bandwidth-bound on a few MB of adapters, no NVLink needed.

Non-goals: FP16 4k-batch pre-training, dense 4k×4k×4k GEMM shootout.

## 2. Architecture

```
Model (HF/safetensors) → Graph IR → Fuse → Plan (shard, block, paginate)
  → Arena (NUMA, huge pages) → ThreadPool → Microkernels (AVX512/AMX)
```

* **Graph IR:** static DAG: MatMul, RMSNorm, RoPE, SDPA-decode, SiLU-mul,
  AllReduce. Fusion happens here, not in Python.
* **Fusion (the whole win):** per-token decode is one weight-streaming
  pipeline per layer:
  `RMSNorm → QKV-proj → RoPE → GQA-attn → O-proj → RMSNorm → Gate/Up → SiLU-mul → Down`
  Weights are read once from DRAM into L3/shared, never re-streamed.
  No intermediate tensors leave registers/L1.
* **Blocked layout:** weights stored `[N_blk][K_blk]` row-quantized
  (Q8_0: int8 + fp32 scale + precomputed per-64 weight block-sums for the
  VNNI bias correction; Q4_0: int4 nibbles + per-32-group fp32 scale/zero —
  coarse per-row groups collapse on outliers, GS=32 keeps NRMSE < 8%).
  Activations are quantized **once per token** and shared across rows
  (re-quantizing per row repeats the work rows-times). KV-cache is paged
  (vLLM-style, 16-token pages), GQA-shared, optionally Q8 KV.
* **Cache policy (the idea that works):** cache the *reused*, stream the
  rest. Decode weights are used once per token — streaming them with NTA
  prefetch and *not* retaining them is faster than caching (no pollution).
  What stays pinned: Q (L1, ~512 B), per-group scales (L1), scores (L2),
  the C tile across Kc passes (L2/L3, written back once), hot KV heads
  under continuous batching (L3 serves them at ~230 GB/s effective).
  Measured: L3-hot attention ≈ 230 GB/s effective vs cold-DRAM 59 GB/s.
* **Threading:** one persistent pool (1 thread/core, pinned, no
  oversubscription). `parallel_for` splits N (output rows) for GEMV and
  KV-chunks for attention with online-softmax reduction. NUMA: shard Q/KV
  heads, interleave weight pages. No `malloc`/`fork` in hot loop.
* **Prefill vs decode split:** prefill = compute-bound blocked BF16 GEMM
  (L2-tiled, AMX/AVX512); decode = bandwidth-bound quantized GEMV + fused
  attention. Different kernels, same IR.
* **Training path:** frozen Q4 base. Only LoRA A/B (rank 8–64, BF16) get
  grads. Fused AdamW (m/v/fp32 master in one stream), activation
  checkpointing, BF16 grad accumulation. Avoids the optimizer-memory wall
  that forces GPUs to multi-card.

## 3. Kernels (x86_64, AVX512 + AMX paths)

* `qgemv_q8`: INT8 rows × once-quantized INT8 activations. `vdpbusd` (VNNI,
  unsigned×signed) accumulates int32; the +128 activation bias is corrected
  with offline per-64 weight block-sums, so the hot loop is 1 VNNI instr +
  1 rescale per 64 elems. Measured 62 GB/s weight-stream (≈ bus max), 4.6×
  FP32 GEMV — the win is bandwidth, not FLOPs.
* `qgemv_q4`: int8 VNNI path (Q4_0 × Q8_0 style): unpack nibbles to int8,
  dot once-quantized activations via `dpbusd`, activations stored even/odd-
  split per 32-group so unpack is and/shift only. Exact vs quantized math
  (2e-8). Measured ≈ float path (~26 GB/s both — shuffle cost ≈ cvt cost);
  kept as default because GGUF Q4_0 is zero-free, which drops two vector
  instructions from this path and will pull it ahead.
* `bgemm_bf16`: BF16 `vdpbf16ps` 32×64×32 micro-tile, L2-resident B,
  streaming A. Used for prefill + LoRA backward. Prototype prefill GEMM with
  transposed-B contiguous dots: ~277 GFLOPS FP32 on 6 cores (strided-B left
  10× on the table at 6 GFLOPS — layout is the interface).
* `rmsnorm`: single-pass `sqr→rsqrt→scale`, fused into GEMV epilogue.
* `attn_decode`: fused RoPE+GQA: `Q·Kᵀ → softmax → ·V`, online softmax over
  256-elem KV blocks, heads parallel, no N×N materialization. KV in Q8 cuts
  long-context bandwidth 4×. Softmax exp is vectorized (AVX512 range-reduce +
  Taylor-5 + `scalef`, relerr 1.5e-6) — scalar libm exp would own the profile
  past 8k context. Scores buffer is thread-local, grown once, uncapped n.
* `adamw_fused`: one stream over param/m/v/grad, AVX512, no temp buffers.
* `gemm_blocked`: Mc×Kc-tiled prefill — C tile cached across rank passes and
  written back once, Bt panel shared across Mc rows (Bt traffic ÷Mc, A read
  once). Bit-exact vs naive; +1.1–1.4× at 256×4096², decisive as Bt/L3 grows.

Intel AMX (`tile_dpbf16ps`) is a compile-time swap for the `bgemm` inner
loop; AVX512-BF16 is the portable baseline (this repo prototypes AVX512).

## 4. Memory + serving

* Arena allocator, 2 MB huge pages, NUMA-interleave weights, pin KV pages
  local to the worker that owns the head.
* Paged KV (16tok/page), continuous batching, chunked prefill (512tok
  chunks piggy-backed on decode), speculative draft (160M Q4 drafter,
  accept 2–3 tok/step → 1.6–2× tok/s).
* Cost model: `tok/s ≈ eff_BW / bytes_per_token`. 8B Q4 ≈ 4.4 GB/token →
  28 GB/s / 4.4 GB ≈ 6 tok/s batch=1 on this 6-core box (measured basis,
  §5). 70B Q4 ≈ 40 GB → ~1 tok/s/socket, 8-socket ~6 tok/s — no GPU needed.

## 5. What the prototype proves

`bench/bench` (this repo, 6-core Zen4, AVX512-BF16/VNNI, all DRAM-resident):

* FP32 GEMV is DRAM-bound (1.2× on 6 cores — threading can't beat the bus);
* Q8 GEMV 62 GB/s ≈ bus max, 4.6× FP32; Q4 GEMV 28 GB/s, 4.2× FP32;
* fused GQA decode-attention: L3-hot KV ≈ 230 GB/s effective, cold 16k-ctx
  stream at bus max (59 GB/s), O(N·d) traffic, uncapped n;
* prefill GEMM ~277 GFLOPS, cache-blocked variant bit-exact and +1.1–1.4× at
  256×4096² (traffic ÷50×, decisive as Bt/L3 grows);
* RMSNorm 84 GB/s; fused AdamW 50 GB/s; vector exp relerr 1.5e-6;
* 8B-Q4 ≈ 4.4 GB/token → ~6 tok/s batch=1 — GPU-competitive on perf/$
  (a $2k box vs a $30k H100 for low-concurrency chat/agents/RAG).

See `README.md` for build/run, `bench/bench.cc` for numbers.
