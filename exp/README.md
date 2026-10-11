# Experiments: performance ideas, quality-gated

Each experiment lives in `exp/<name>/` as a self-contained patch against the
main runtime. An experiment is only merged into the main build when it passes
the quality gate: **no measurable quality loss** on the standard prompts.

## Quality gate (run before any merge)

```sh
make -C exp/quality_gate          # builds gate tools (needs oracle + molot)
./exp/quality_gate/run_gate.sh    # compares vs reference on 5 prompts
```

Gate criteria (all must hold):
1. **Logit cosine ≥ 0.995** vs the reference implementation on the 5 standard
   prompts (int path; exact path must be ≥ 0.999)
2. **Greedy top-1 agreement ≥ 4/5** prompts (exact path: 5/5)
3. **No new degenerate outputs** (repetition loops, marker leakage) on the
   chat smoke test
4. **Speedup ≥ 1.15×** on the decode benchmark (else it's not worth merging)

## Experiment checklist

- [ ] `exp/<name>/README.md` — what it changes, expected win, measured result
- [ ] `exp/<name>/patch.diff` — the actual change (or a fork of the file)
- [ ] quality gate passes (paste output)
- [ ] `bench/bench.cc` still passes all correctness checks
- [ ] merged into main build with a one-line note in `README.md`

## Ideas (ranked by expected value — with verdicts)

1. **exp/qact32** — ✅ DONE (already in mainline: activations quantized
   per-32 to match ggml granularity).
2. **exp/rope_lut** — ❌ REJECTED on roofline grounds (rope is ~1% of
   decode; below the 1.15× merge bar by construction).
3. **exp/kv_q8** — ⏸ DEFERRED. KV traffic is <1% of decode time at demo
   contexts (22 ms total). It only matters at 32k+ context, and quality
   validation there needs a long-context eval harness we don't have yet.
   Revisit with one.
4. **exp/head_f32** — ❌ REJECTED on roofline grounds (2.7 GB fp32 head vs
   0.7 GB Q8 = 4× the traffic for the most bandwidth-sensitive GEMV).
5. **exp/prefill_batch** — ✅ MERGED. 2.05× faster prefill on 302 tokens
   (13.3 s vs 27.2 s), gate PASS. See `exp/prefill_batch/README.md`.
6. **exp/spec_decode** — ⏸ BLOCKED: needs a small draft GGUF we don't have
   locally. Revisit when one is available.
7. **exp/softmax_block** — ❌ REJECTED (sub-1% of profile).
8. **exp/weight_prefetch** — ❌ REJECTED (measured neutral-to-negative
   end-to-end; see `exp/weight_prefetch/README.md`).

## Rules

- Experiments must not change the model format or break GGUF loading.
- An experiment that fails the gate stays in `exp/` with a note; it is never
  merged silently.
- Every merge must include the gate output in the commit message.
