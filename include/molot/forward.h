#pragma once
// Single-token forward (decode path; prompts are fed token-by-token in v1).
// Block math mirrors the reference implementation exactly:
//
//   x = emb[tok] * sqrt(E)
//   plan = ModelProj(Q8) @ x / sqrt(E)                    # once per token
//   e_b = (RMSNorm(plan_b) + tokcode_b * 16) / sqrt(2)    # per-layer code
//   per block:
//     u = RMSNorm(x, attn_norm); q = Q(u) [, k = K(u), v = V(u)]
//     per-head QK-norms; V unit-norm; RoPE; cache k/v (rotated)
//     a = GQA(q, cache[src], window) with scale 1.0       # q_norm ⇒ no scaling
//     x += RMSNorm(O(a), post_attn_norm)
//     f = RMSNorm(x, ffn_norm); h = GELU(GATE(f)) * UP(f)
//     x += RMSNorm(DOWN(h), post_ffw_norm)
//     g = GELU(inp_gate @ x); m = g * e_b                 # per-layer path
//     x += RMSNorm(proj @ m, per_layer_post_norm)
//     x *= out_scale
//   logits = softcap(Head(RMSNorm(x, out_norm)))
//
// Shared-KV layers (no attn_k) read the reference-rule source cache
// (first_nokv-2 for SWA, first_nokv-1 for global).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "molot/attn.h"
#include "molot/gemm.h"
#include "molot/model.h"
#include "molot/norm.h"
#include "molot/thread_pool.h"

namespace molot::fwd {

inline bool is_bad(float x) {
  uint32_t u;  // -ffast-math folds isfinite() to true; check bits instead
  static_assert(sizeof u == sizeof x);
  std::memcpy(&u, &x, 4);
  return (u & 0x7F800000u) == 0x7F800000u;
}

struct Runner {
  model::Model m;
  ThreadPool pool;
  struct KVC {
    int n = 0, cap = 0, nkv = 0, hd = 0;
    std::vector<float> k, v;  // [cap][nkv*hd] each
  };
  std::vector<KVC> kv;  // per layer (only has_kv entries used)
  // scratch (max dims)
  std::vector<float> x, u, q, k, v, ao, gg, uu, h, d, logits, inj, fq, layer_code, plan;
  std::vector<float> moe_logits, moe_acc;
  std::vector<float> ecode;  // per-layer codes [n_layer*256]
  struct SSMState {
    int kh = 0, vh = 0, nh = 0, conv = 0;
    std::vector<float> recurrent, history;
  };
  std::vector<SSMState> ssm;

  Runner(model::Model mm, int ctx_cap = 4096) : m(std::move(mm)) {
    int E = m.hp.n_embd;
    int maxq = 0, maxkv = 0, maxf = 0;
    for (auto& L : m.layers) {
      maxq = std::max(maxq, L.is_ssm ? L.ssm_qkv.rows : L.q.rows);
      if (L.has_kv) maxkv = std::max(maxkv, L.k.rows);
      maxf = std::max(maxf, L.gate.rows);
      if (L.expert_count) maxf = std::max(maxf, L.moe_gate_exps[0].rows);
    }
    x.resize(E);
    u.resize(E);
    q.resize(maxq);
    k.resize(maxkv);
    v.resize(maxkv);
    ao.resize(maxq);
    gg.resize(maxf);
    uu.resize(maxf);
    h.resize(maxf);
    d.resize(E);
    inj.resize(E);
    logits.resize(m.hp.vocab);
    fq.resize(512);
    if (!m.layers.empty()) {
      int ne = 0;
      for (auto& L : m.layers) ne = std::max(ne, L.expert_count);
      moe_logits.resize(ne);
      moe_acc.resize(E);
    }
    ecode.resize(m.layers.size() * 256);
    kv.resize(m.layers.size());
    ssm.resize(m.layers.size());
    for (size_t b = 0; b < m.layers.size(); ++b) {
      auto& L = m.layers[b];
      if (L.is_ssm) {
        SSMState& z = ssm[b];
        z.kh = L.ssm_key_dim; z.vh = L.ssm_value_dim;
        z.nh = L.ssm_value_heads; z.conv = 4;
        z.recurrent.assign((size_t)z.nh * z.kh * z.vh, 0.0f);
        z.history.assign((size_t)L.ssm_qkv.rows * z.conv, 0.0f);
      }
      if (!L.has_kv) continue;
      KVC& c = kv[b];
      c.nkv = m.hp.n_kv;
      c.hd = L.khd;
      c.cap = ctx_cap;
      c.k.assign((size_t)ctx_cap * c.nkv * c.hd, 0);
      c.v.assign((size_t)ctx_cap * c.nkv * c.hd, 0);
    }
  }

  bool exact = false;
  // section timers (MOLOT_PROFILE=1 prints ms by category at end of run)
  double t_g4 = 0, t_g8 = 0, t_attn = 0, t_other = 0;
  static double now_ms() {
    return 1e-6 *
           (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
               .count();
  }
  void g4(const Q4Matrix& W, const float* x, float* y) {
    double t = (std::getenv("MOLOT_PROFILE") != nullptr) ? now_ms() : 0;
    if (exact)
      gemv_q4_exact(W, x, y, pool);
    else
      gemv_q4(W, x, y, pool);
    if (t) t_g4 += now_ms() - t;
  }
  void g8(const Q8Matrix& W, const float* x, float* y) {
    double t = (std::getenv("MOLOT_PROFILE") != nullptr) ? now_ms() : 0;
    if (exact && !std::getenv("MOLOT_INT8"))
      gemv_q8_exact(W, x, y, pool);
    else
      gemv_q8(W, x, y, pool);
    if (t) t_g8 += now_ms() - t;
  }
  void rms_per_head(float* x, const float* w, int nheads, int hd, float eps) {
    for (int hh = 0; hh < nheads; ++hh) rmsnorm(x + (size_t)hh * hd, w, x + (size_t)hh * hd, hd, eps);
  }

  // Embedding gather + init: sqrt-scaled raw (gemma4) or plain RMSNorm
  // (glimmer embd_norm). Quantized rows dequantize on the fly.
  void embed_row(int tok, float* out) {
    int E = m.hp.n_embd;
    if (m.emb_kind == 0) {
      std::memcpy(out, m.emb.data() + (size_t)tok * E, (size_t)E * 4);
    } else if (m.emb_kind == gguf::Q8_0) {
      const uint8_t* d = m.emb_raw.data();
      repack::dequant_q8_0_row(d + (size_t)tok * (E / 32) * 34, E, out);
    } else if (m.emb_kind == gguf::Q2_K || m.emb_kind == gguf::Q3_K ||
               m.emb_kind == gguf::Q4_K || m.emb_kind == gguf::Q5_K ||
               m.emb_kind == gguf::Q6_K || m.emb_kind == gguf::Q8_K) {
      const uint8_t* d = m.emb_raw.data();
      repack::dequant_k_row((uint32_t)m.emb_kind,
                            d + (size_t)tok * (E / 256) *
                                    repack::kquant_block_bytes((uint32_t)m.emb_kind),
                            E, out);
    } else {
      throw std::runtime_error("fwd: emb kind unsupported");
    }
    if (m.hp.emb_plain_norm) {
      rmsnorm_plain(out, out, E, m.hp.eps);
    } else if (m.hp.emb_scale) {
      float es = std::sqrt((float)E);
      for (int i = 0; i < E; ++i) out[i] *= es;
    }
  }

  float attn_scale(const model::Layer& L) const {
    if (m.hp.attn_scale_one) return 1.0f;
    return 1.0f / std::sqrt((float)L.qhd);
  }

  void ffn_act(const float* g, const float* u, float* y, int n) const {
    if (m.hp.ffn_silu)
      silu_mul(g, u, y, n);
    else
      gelu_mul(g, u, y, n);
  }

  void moe_ffn(const model::Layer& L, const float* in, float* out) {
    g4(L.moe_gate_inp, in, moe_logits.data());
    int k = std::min(L.expert_used_count, L.expert_count);
    std::vector<int> ids(k);
    std::vector<float> scores(moe_logits.begin(), moe_logits.begin() + L.expert_count);
    for (int i = 0; i < k; ++i) {
      ids[i] = 0;
      for (int e = 1; e < L.expert_count; ++e)
        if (scores[e] > scores[ids[i]]) ids[i] = e;
      scores[ids[i]] = -INFINITY;
    }
    float mx = -INFINITY, den = 0;
    std::vector<float> w(k);
    for (int i = 0; i < k; ++i) mx = std::max(mx, scores[ids[i]]);
    for (int i = 0; i < k; ++i) den += (w[i] = std::exp(scores[ids[i]] - mx));
    for (float& z : w) z /= den;
    std::fill(moe_acc.begin(), moe_acc.end(), 0.0f);
    for (int i = 0; i < k; ++i) {
      const auto& eg = L.moe_gate_exps[ids[i]];
      const auto& eu = L.moe_up_exps[ids[i]];
      const auto& ed = L.moe_down_exps[ids[i]];
      g4(eg, in, gg.data());
      g4(eu, in, uu.data());
      ffn_act(gg.data(), uu.data(), h.data(), eg.rows);
      g4(ed, h.data(), d.data());
      for (int j = 0; j < ed.rows; ++j) moe_acc[j] += w[i] * d[j];
    }
    if (!L.moe_shared_gate.data.empty()) {
      float sg = 0;
      for (int j = 0; j < (int)L.moe_shared_router.size(); ++j) sg += in[j] * L.moe_shared_router[j];
      g4(L.moe_shared_gate, in, gg.data());
      g4(L.moe_shared_up, in, uu.data());
      ffn_act(gg.data(), uu.data(), h.data(), L.moe_shared_gate.rows);
      g4(L.moe_shared_down, h.data(), d.data());
      float sw = 1.0f / (1.0f + std::exp(-sg));
      for (int j = 0; j < L.moe_shared_down.rows; ++j) moe_acc[j] += sw * d[j];
    }
    std::memcpy(out, moe_acc.data(), (size_t)L.moe_down_exps[0].rows * sizeof(float));
  }

  void ssm_step(const model::Layer& L, const float* in, float* out, int b) {
    SSMState& st = ssm[b];
    const int C = L.ssm_qkv.rows, K = st.conv;
    g4(L.ssm_qkv, in, q.data());
    g4(L.ssm_z, in, ao.data());
    // depthwise causal convolution, with the current projection at history tail.
    for (int c = 0; c < C; ++c) {
      float v0 = 0;
      for (int j = 0; j < K - 1; ++j) v0 += L.ssm_conv[(size_t)c * K + j] *
                                                   st.history[(size_t)c * K + j + 1];
      v0 += L.ssm_conv[(size_t)c * K + K - 1] * q[c];
      for (int j = 0; j < K - 1; ++j) st.history[(size_t)c * K + j] =
          st.history[(size_t)c * K + j + 1];
      st.history[(size_t)c * K + K - 1] = q[c];
      q[c] = v0 > 0 ? v0 / (1.0f + std::exp(-v0)) : v0 * std::exp(v0) /
                                                     (1.0f + std::exp(v0));
    }
    const int qd = L.ssm_key_dim, vd = L.ssm_value_dim, nk = L.ssm_key_heads;
    const float* qq = q.data(), *kk = q.data() + nk * qd;
    const float* vv = q.data() + 2 * nk * qd;
    std::vector<float> qn((size_t)L.ssm_value_heads * qd), kn(qn.size());
    for (int h0 = 0; h0 < nk; ++h0) {
      float nq = 0, nk0 = 0;
      for (int j = 0; j < qd; ++j) { nq += qq[h0 * qd + j] * qq[h0 * qd + j];
        nk0 += kk[h0 * qd + j] * kk[h0 * qd + j]; }
      nq = 1.0f / std::sqrt(nq + 1e-6f); nk0 = 1.0f / std::sqrt(nk0 + 1e-6f);
      for (int j = 0; j < qd; ++j) {
        qn[(size_t)(2*h0) * qd + j] = qq[h0*qd+j] * nq;
        kn[(size_t)(2*h0) * qd + j] = kk[h0*qd+j] * nk0;
        qn[(size_t)(2*h0+1) * qd + j] = qn[(size_t)(2*h0) * qd + j];
        kn[(size_t)(2*h0+1) * qd + j] = kn[(size_t)(2*h0) * qd + j];
      }
    }
    std::vector<float> core((size_t)L.ssm_value_heads * vd);
    for (int h0 = 0; h0 < L.ssm_value_heads; ++h0) {
      float aa = 0, bb = 0;
      for (int i = 0; i < m.hp.n_embd; ++i) {
        aa += L.ssm_alpha[(size_t)h0 * m.hp.n_embd + i] * in[i];
        bb += L.ssm_beta[(size_t)h0 * m.hp.n_embd + i] * in[i];
      }
      float a = -std::exp(L.ssm_a[h0]) * std::log1p(std::exp(aa + L.ssm_dt[h0]));
      float beta = 1.0f / (1.0f + std::exp(-bb));
      float* S = st.recurrent.data() + (size_t)h0 * qd * vd;
      const float* qh = qn.data() + (size_t)h0 * qd;
      const float* kh = kn.data() + (size_t)h0 * qd;
      for (int i = 0; i < qd * vd; ++i) S[i] *= std::exp(a);
      for (int j = 0; j < vd; ++j) {
        float pred = 0;
        for (int i = 0; i < qd; ++i) pred += S[(size_t)i * vd + j] * qh[i];
        for (int i = 0; i < qd; ++i) S[(size_t)i * vd + j] += beta * kh[i] * (vv[(size_t)h0 * vd + j] - pred);
        float y = 0; for (int i = 0; i < qd; ++i) y += S[(size_t)i * vd + j] * qh[i];
        core[(size_t)h0 * vd + j] = y;
      }
      float ss = 0; for (int j = 0; j < vd; ++j) ss += core[(size_t)h0*vd+j]*core[(size_t)h0*vd+j];
      float ri = 1.0f / std::sqrt(ss / vd + m.hp.eps);
      for (int j = 0; j < vd; ++j) core[(size_t)h0*vd+j] *= ri * L.ssm_norm[j] *
          (ao[(size_t)h0*vd+j] > 0 ? ao[(size_t)h0*vd+j] / (1 + std::exp(-ao[(size_t)h0*vd+j])) :
                                     ao[(size_t)h0*vd+j] * std::exp(ao[(size_t)h0*vd+j]) /
                                     (1 + std::exp(ao[(size_t)h0*vd+j])));
    }
    g4(L.ssm_out, core.data(), out);
  }

  // Batched prefill: M prompt tokens starting at pos0 in one weight-streaming
  // pass per layer. Fills KV caches, returns last-token logits. Numerics
  // track sequential decode (same sub-steps, batched GEMMs with identical
  // per-row activation quantization). Chunked at 512 tokens to bound scratch.
  // MOLOT_SEQ_PREFILL=1 forces the old sequential loop (A/B + gate checks).
  void forward_prefill(const int* toks, int M, int pos0, float* out_logits) {
    for (const auto& L : m.layers)
      if (L.expert_count) {
        for (int i = 0; i < M; ++i) forward_token(toks[i], pos0 + i, out_logits);
        return;
      }
    if (std::getenv("MOLOT_SEQ_PREFILL")) {
      for (int i = 0; i < M; ++i) forward_token(toks[i], pos0 + i, out_logits);
      return;
    }
    exact = std::getenv("MOLOT_FP32") != nullptr;
    if (exact) {  // exact mode stays sequential (diagnostic path, not product)
      for (int i = 0; i < M; ++i) forward_token(toks[i], pos0 + i, out_logits);
      return;
    }
    for (int c = 0; c < M; c += 512) {
      int m0 = c, m1 = c + 512 < M ? c + 512 : M;
      prefill_chunk(toks + m0, m1 - m0, pos0 + m0, m1 == M ? out_logits : nullptr);
    }
  }

  void prefill_chunk(const int* toks, int M, int pos0, float* out_logits) {
    const model::HParams& hp = m.hp;
    int E = hp.n_embd;
    const char* dumpdir = std::getenv("MOLOT_DUMP_MINE");
    auto pdump = [&](size_t b, const char* stage, const float* p, int w, int m) {
      if (!dumpdir) return;
      char path[512];
      snprintf(path, sizeof path, "%s/mypre_%zu_%s_pos%d.bin", dumpdir, b, stage, pos0 + m);
      FILE* f = fopen(path, "wb");
      fwrite(p + (size_t)m * w, 4, w, f);
      fclose(f);
    };
    int maxq = 0, maxkv = 0, maxf = 0, D = 0;
    for (auto& L : m.layers) {
      maxq = std::max(maxq, L.q.rows);
      if (L.has_kv) maxkv = std::max(maxkv, L.k.rows);
      maxf = std::max(maxf, L.gate.rows);
    }
    if (m.has_layer_emb) D = m.layer_emb_dim * (int)m.layers.size();
    // scratch
    std::vector<float> X((size_t)M * E), Xn((size_t)M * E), U((size_t)M * E);
    std::vector<float> Q((size_t)M * maxq), K((size_t)M * maxkv), V((size_t)M * maxkv);
    std::vector<float> AO((size_t)M * maxq), G((size_t)M * maxf), UP((size_t)M * maxf);
    std::vector<float> H((size_t)M * maxf), Dd((size_t)M * E);
    std::vector<float> plan, tokc;
    if (m.has_layer_emb) {
      plan.assign((size_t)M * D, 0);
      tokc.assign((size_t)M * D, 0);
      // gather + scale embs, plan batched GEMM, tokcode per token
      for (int mm = 0; mm < M; ++mm) embed_row(toks[mm], X.data() + (size_t)mm * E);
      gemm_q8_batch(m.layer_plan, X.data(), plan.data(), M, pool);
      const uint8_t* base = m.layer_emb_raw.data();
      size_t stride = (size_t)(D / 32) * 18;
      std::vector<float> tmp(D);
      for (int mm = 0; mm < M; ++mm) {
        repack::dequant_q4_0_row(base + (size_t)toks[mm] * stride, D, tmp.data());
        for (int i = 0; i < D; ++i) tokc[(size_t)mm * D + i] = tmp[i] * 16.0f;
      }
    } else {
      for (int mm = 0; mm < M; ++mm) embed_row(toks[mm], X.data() + (size_t)mm * E);
    }

    float ps = 1.0f / std::sqrt((float)E);
    float hs = 1.0f / std::sqrt(2.0f);
    std::vector<float> ecode((size_t)m.layers.size() * 256);
    for (size_t b = 0; b < m.layers.size(); ++b) {
      const model::Layer& L = m.layers[b];
      int nq = hp.n_head, nkv = hp.n_kv;
      for (int mm = 0; mm < M; ++mm) pdump(b, "inp", X.data(), E, mm);
      // --- attention projections (batched) ---
      for (int mm = 0; mm < M; ++mm)
        rmsnorm(X.data() + (size_t)mm * E, L.attn_norm.data(), U.data() + (size_t)mm * E,
                E, hp.eps);
      gemm_q4_batch(L.q, U.data(), Q.data(), M, pool);
      if (L.has_kv) {
        gemm_q4_batch(L.k, U.data(), K.data(), M, pool);
        gemm_q4_batch(L.v, U.data(), V.data(), M, pool);
      }
      // qnorm/rope per token-head; k norm/rope + cache append
      for (int mm = 0; mm < M; ++mm) {
        float* qq = Q.data() + (size_t)mm * L.q.rows;
        rms_per_head(qq, L.q_norm.data(), nq, L.qhd, hp.eps);
        if (!L.freq.empty())
          for (int hh = 0; hh < nq; ++hh)
            rope_apply(qq + (size_t)hh * L.qhd, L.qhd, pos0 + mm, L.freq.data());
        if (L.has_kv) {
          float* kk = K.data() + (size_t)mm * L.k.rows;
          float* vv = V.data() + (size_t)mm * L.k.rows;
          rms_per_head(kk, L.k_norm.data(), nkv, L.khd, hp.eps);
          if (m.hp.v_norm) rmsnorm_plain(vv, vv, nkv * L.khd, hp.eps);
          if (!L.freq.empty())
            for (int hh = 0; hh < nkv; ++hh)
              rope_apply(kk + (size_t)hh * L.khd, L.khd, pos0 + mm, L.freq.data());
          KVC& c = kv[b];
          if (c.n + M > c.cap) throw std::runtime_error("fwd: KV cache full");
          std::memcpy(c.k.data() + (size_t)(c.n + mm) * nkv * L.khd, kk,
                      (size_t)nkv * L.khd * 4);
          std::memcpy(c.v.data() + (size_t)(c.n + mm) * nkv * L.khd, vv,
                      (size_t)nkv * L.khd * 4);
        }
      }
      if (L.has_kv) kv[b].n += M;
      // attention prefill over the filled cache
      {
        KVC& src = L.has_kv ? kv[b] : kv[L.kv_src];
        attn_prefill_gqa(Q.data(), src.k.data(), src.v.data(), AO.data(), nq,
                         (int)src.nkv, M, src.n, L.qhd, attn_scale(L), pos0, L.window,
                         pool);
        for (int mm = 0; mm < M; ++mm) pdump(b, "kqv_out", AO.data(), L.q.rows, mm);
      }
      if (!L.attn_gate.data.empty()) {
        gemm_q4_batch(L.attn_gate, U.data(), G.data(), M, pool);
        for (int mm = 0; mm < M; ++mm) {
          float* gg = G.data() + (size_t)mm * L.q.rows;
          float* aa = AO.data() + (size_t)mm * L.q.rows;
          for (int i = 0; i < L.q.rows; ++i) {
            float g = gg[i];
            aa[i] *= 1.0f / (1.0f + std::exp(-g));
          }
        }
      }
      gemm_q4_batch(L.o, AO.data(), Dd.data(), M, pool);
      for (int mm = 0; mm < M; ++mm) pdump(b, "o_out", Dd.data(), E, mm);
      for (int mm = 0; mm < M; ++mm) {
        float* dd = Dd.data() + (size_t)mm * E;
        float* xx = X.data() + (size_t)mm * E;
        if (!L.post_attn_norm.empty()) rmsnorm(dd, L.post_attn_norm.data(), dd, E, hp.post_eps);
        for (int i = 0; i < E; ++i) xx[i] += dd[i];
      }
      for (int mm = 0; mm < M; ++mm) pdump(b, "attn_post_norm", Dd.data(), E, mm);
      // --- FFN (batched GEMMs, per-token elementwise) ---
      for (int mm = 0; mm < M; ++mm) pdump(b, "ffn_inp", X.data(), E, mm);
      for (int mm = 0; mm < M; ++mm)
        rmsnorm(X.data() + (size_t)mm * E, L.ffn_norm.data(), U.data() + (size_t)mm * E, E,
                hp.eps);
      for (int mm = 0; mm < M; ++mm) pdump(b, "ffn_norm", U.data(), E, mm);
      gemm_q4_batch(L.gate, U.data(), G.data(), M, pool);
      gemm_q4_batch(L.up, U.data(), UP.data(), M, pool);
      for (int mm = 0; mm < M; ++mm) pdump(b, "ffn_gate", G.data(), L.gate.rows, mm);
      for (int mm = 0; mm < M; ++mm)
        ffn_act(G.data() + (size_t)mm * L.gate.rows, UP.data() + (size_t)mm * L.gate.rows,
                 H.data() + (size_t)mm * L.gate.rows, L.gate.rows);
      for (int mm = 0; mm < M; ++mm) pdump(b, "ffn_geglu", H.data(), L.gate.rows, mm);
      gemm_q4_batch(L.down, H.data(), Dd.data(), M, pool);
      for (int mm = 0; mm < M; ++mm) pdump(b, "ffn_out", Dd.data(), E, mm);
      for (int mm = 0; mm < M; ++mm) {
        float* dd = Dd.data() + (size_t)mm * E;
        float* xx = X.data() + (size_t)mm * E;
        if (!L.post_ffw_norm.empty()) rmsnorm(dd, L.post_ffw_norm.data(), dd, E, hp.post_eps);
        for (int i = 0; i < E; ++i) xx[i] += dd[i];
      }
      // --- per-layer path ---
      if (m.has_layer_emb && !L.inp_gate.data.empty()) {
        gemm_q4_batch(L.inp_gate, X.data(), G.data(), M, pool);
        for (int mm = 0; mm < M; ++mm) {
          float* gg = G.data() + (size_t)mm * 256;
          float* eb = ecode.data() + b * 256;
          // rebuild this token's code (plan slice + tokcode slice)
          {
            float* pp = plan.data() + (size_t)mm * D + (size_t)b * m.layer_emb_dim;
            float nrm[256];
            rmsnorm(pp, m.layer_proj_norm.data(), nrm, m.layer_emb_dim, hp.eps);
            const float* tc = tokc.data() + (size_t)mm * D + (size_t)b * m.layer_emb_dim;
            for (int j = 0; j < 256; ++j) eb[j] = (nrm[j] * ps + tc[j]) * hs;
          }
          for (int j = 0; j < 256; ++j) {
            float g = gg[j];
            gg[j] = (0.5f * g * (1.0f + std::tanh(0.79788456f * g * (1.0f + 0.044715f * g * g)))) *
                    eb[j];
          }
        }
        gemm_q4_batch(L.proj, G.data(), Dd.data(), M, pool);
        for (int mm = 0; mm < M; ++mm) pdump(b, "per_layer_embd_out", Dd.data(), E, mm);
        for (int mm = 0; mm < M; ++mm) {
          float* dd = Dd.data() + (size_t)mm * E;
          float* xx = X.data() + (size_t)mm * E;
          rmsnorm(dd, L.post_norm.data(), dd, E, hp.eps);
          for (int i = 0; i < E; ++i) xx[i] += dd[i];
        }
      }
      // --- output scale ---
      for (int mm = 0; mm < M; ++mm) {
        float* xx = X.data() + (size_t)mm * E;
        for (int i = 0; i < E; ++i) xx[i] *= L.out_scale;
        pdump(b, "l_out", X.data(), E, mm);
      }
    }
    if (out_logits) {
      float* last = X.data() + (size_t)(M - 1) * E;
      rmsnorm(last, m.out_norm.data(), u.data(), E, hp.eps);
      g8(m.head, u.data(), out_logits);
      if (hp.logit_scale != 1.0f)
        for (int i = 0; i < hp.vocab; ++i) out_logits[i] *= hp.logit_scale;
      if (hp.logit_cap > 0)
        for (int i = 0; i < hp.vocab; ++i)
          out_logits[i] = hp.logit_cap * std::tanh(out_logits[i] / hp.logit_cap);
    }
  }

  void forward_token(int tok, int pos, float* out_logits) {
    const model::HParams& hp = m.hp;
    exact = std::getenv("MOLOT_FP32") != nullptr;
    int E = hp.n_embd;
    const char* dumpdir = std::getenv("MOLOT_DUMP_MINE");
    auto dump = [&](size_t b, const char* stage, const float* p, int n) {
      if (!dumpdir) return;
      char path[512];
      snprintf(path, sizeof path, "%s/myblk_%zu_%s_pos%d.bin", dumpdir, b, stage, pos);
      FILE* f = fopen(path, "wb");
      fwrite(p, 4, n, f);
      fclose(f);
    };
    embed_row(tok, x.data());

    // ---- per-layer codes (once per token) ----
    if (m.has_layer_emb) {
      int D = m.layer_emb_dim * (int)m.layers.size();
      if (layer_code.size() != (size_t)D) layer_code.resize(D);
      if (plan.size() != (size_t)D) plan.resize(D);
      const uint8_t* base = m.layer_emb_raw.data();
      size_t stride = (size_t)(D / 32) * 18;
      repack::dequant_q4_0_row(base + (size_t)tok * stride, D, layer_code.data());
      for (int i = 0; i < D; ++i) layer_code[i] *= 16.0f;  // sqrt(256)
      if (exact) {
        // exact plan path: widened fp32 (no requant noise)
        gemv_fp32_mt(m.plan_fp32.data(), x.data(), plan.data(), m.layer_plan.rows,
                     m.layer_plan.cols, pool);
      } else {
        g8(m.layer_plan, x.data(), plan.data());
      }
      float ps = 1.0f / std::sqrt((float)E);
      float hs = 1.0f / std::sqrt(2.0f);
      for (size_t b = 0; b < m.layers.size(); ++b) {
        float* eb = ecode.data() + b * 256;
        rmsnorm(plan.data() + b * 256, m.layer_proj_norm.data(), eb, 256, hp.eps);
        for (int j = 0; j < 256; ++j)
          eb[j] = (eb[j] * ps + layer_code[b * 256 + j]) * hs;
        (void)ps;
      }
    }

    for (size_t b = 0; b < m.layers.size(); ++b) {
      const model::Layer& L = m.layers[b];
      dump(b, "inp", x.data(), E);
      // --- attention ---
      rmsnorm(x.data(), L.attn_norm.data(), u.data(), E, hp.eps);
      if (L.is_ssm) {
        ssm_step(L, u.data(), d.data(), (int)b);
        for (int i = 0; i < E; ++i) x[i] += d[i];
        dump(b, "ssm_out", d.data(), E);
      } else {
      dump(b, "attn_norm", u.data(), E);
      g4(L.q, u.data(), q.data());
      dump(b, "Qcur", q.data(), L.q.rows);
      const float* kvk;
      const float* kvv;
      int nn_src = 0;
      KVC* own = nullptr;
      if (L.has_kv) {
        g4(L.k, u.data(), k.data());
        g4(L.v, u.data(), v.data());
        own = &kv[b];
      }
      rms_per_head(q.data(), L.q_norm.data(), hp.n_head, L.qhd, hp.eps);
      dump(b, "Qcur_normed", q.data(), L.q.rows);
      if (!L.freq.empty())
        for (int hh = 0; hh < hp.n_head; ++hh)
          rope_apply(q.data() + (size_t)hh * L.qhd, L.qhd, pos, L.freq.data());
      dump(b, "Qcur_pos", q.data(), L.q.rows);
      int nkv_use = hp.n_kv;
      if (L.has_kv) {
        rms_per_head(k.data(), L.k_norm.data(), hp.n_kv, L.khd, hp.eps);
        if (m.hp.v_norm) rmsnorm_plain(v.data(), v.data(), hp.n_kv * L.khd, hp.eps);
        dump(b, "Kcur_normed", k.data(), L.k.rows);
        dump(b, "Vcur_normed", v.data(), L.k.rows);
        if (!L.freq.empty())
          for (int hh = 0; hh < hp.n_kv; ++hh)
            rope_apply(k.data() + (size_t)hh * L.khd, L.khd, pos, L.freq.data());
        dump(b, "Kcur_pos", k.data(), L.k.rows);
        if (own->n >= own->cap) throw std::runtime_error("fwd: KV cache full");
        std::memcpy(own->k.data() + (size_t)own->n * nkv_use * L.khd, k.data(),
                    (size_t)nkv_use * L.khd * 4);
        std::memcpy(own->v.data() + (size_t)own->n * nkv_use * L.khd, v.data(),
                    (size_t)nkv_use * L.khd * 4);
        own->n++;
        kvk = own->k.data();
        kvv = own->v.data();
      } else {
        const KVC& srcc = kv[L.kv_src];
        if (srcc.n == 0) throw std::runtime_error("fwd: shared KV empty");
        nkv_use = srcc.nkv;
        kvk = srcc.k.data();
        kvv = srcc.v.data();
        nn_src = srcc.n;
      }
      int nn = L.has_kv ? kv[b].n : nn_src;
      // attention scale per arch (gemma4: 1.0 via QK-norms; else 1/sqrt(head))
      double ta = (std::getenv("MOLOT_PROFILE") != nullptr) ? now_ms() : 0;
      attn_decode_gqa_cache(q.data(), kvk, kvv, ao.data(), hp.n_head, nkv_use, nn, L.qhd,
                            attn_scale(L), pos, L.window, pool);
      if (ta) t_attn += now_ms() - ta;
      dump(b, "kqv_out", ao.data(), L.q.rows);
      if (!L.attn_gate.data.empty()) {
        g4(L.attn_gate, u.data(), gg.data());
        for (int i = 0; i < L.q.rows; ++i) {
          float g = gg[i];
          ao[i] *= 1.0f / (1.0f + std::exp(-g));
        }
      }
      g4(L.o, ao.data(), d.data());
      dump(b, "o_out", d.data(), E);
      if (!L.post_attn_norm.empty()) {
        rmsnorm(d.data(), L.post_attn_norm.data(), d.data(), E, hp.post_eps);
      }
      dump(b, "attn_post_norm", d.data(), E);
      for (int i = 0; i < E; ++i) x[i] += d[i];
      dump(b, "attn_out", x.data(), E);
      }
      // --- FFN (GELU) ---
      dump(b, "ffn_inp", x.data(), E);
      rmsnorm(x.data(), L.ffn_norm.data(), u.data(), E, hp.eps);
      dump(b, "ffn_norm", u.data(), E);
      if (L.expert_count) {
        moe_ffn(L, u.data(), d.data());
      } else {
        g4(L.gate, u.data(), gg.data());
        g4(L.up, u.data(), uu.data());
        dump(b, "ffn_gate", gg.data(), L.gate.rows);
        ffn_act(gg.data(), uu.data(), h.data(), L.gate.rows);
        dump(b, "ffn_geglu", h.data(), L.gate.rows);
        g4(L.down, h.data(), d.data());
      }
      dump(b, "ffn_out", d.data(), E);
      if (!L.post_ffw_norm.empty()) {
        rmsnorm(d.data(), L.post_ffw_norm.data(), d.data(), E, hp.post_eps);
      }
      dump(b, "ffn_post_norm", d.data(), E);
      for (int i = 0; i < E; ++i) x[i] += d[i];
      // --- per-layer path: g=GELU(inp_gate@x); x += post_norm(proj@(g*e_b)) ---
      if (m.has_layer_emb && !L.inp_gate.data.empty()) {
        g4(L.inp_gate, x.data(), gg.data());
        const float* eb = ecode.data() + b * 256;
        for (int j = 0; j < 256; ++j) {
          float g = gg[j];
          gg[j] = 0.5f * g * (1.0f + std::erf(g * 0.7071067811865475f));
        }
        for (int j = 0; j < 256; ++j) gg[j] *= eb[j];
        g4(L.proj, gg.data(), d.data());
        rmsnorm(d.data(), L.post_norm.data(), d.data(), E, hp.eps);
        dump(b, "per_layer_embd_out", d.data(), E);
        for (int i = 0; i < E; ++i) x[i] += d[i];
      }
      // --- output scale (whole block) ---
      for (int i = 0; i < E; ++i) x[i] *= L.out_scale;
      dump(b, "l_out", x.data(), E);
      if (std::getenv("MOLOT_TRACE")) {
        double mx = 0, mn = 0;
        int bad = 0;
        for (int i = 0; i < E; ++i) {
          if (is_bad(x[i])) bad++;
          mx = std::max(mx, (double)std::fabs(x[i]));
          mn += x[i];
        }
        std::fprintf(stderr, "[trace] blk.%2zu x_maxabs=%.3f x_mean=%+.4f bad=%d nkv=%d\n",
                     b, mx, mn / E, bad,
                     L.is_ssm ? 0 : (L.has_kv ? kv[b].n : kv[L.kv_src].n));
      }
    }
    rmsnorm(x.data(), m.out_norm.data(), u.data(), E, hp.eps);
    dump(999, "final", u.data(), E);
    if (std::getenv("MOLOT_F32HEAD")) {
      gemv_fp32_mt(m.emb.data(), u.data(), out_logits, hp.vocab, E, pool);
    } else if (!m.head_q80_raw.empty() && !exact) {
      // default fast path: native Q8_0 blocks, no requant noise
      gemv_q8_0(m.head_q80_raw.data(), hp.vocab, m.head_q80_cols, u.data(), out_logits,
                pool);
    } else if (exact && !m.head_q80_raw.empty()) {
      gemv_q8_0_exact(m.head_q80_raw.data(), hp.vocab, m.head_q80_cols, u.data(),
                      out_logits, pool);
    } else {
      g8(m.head, u.data(), out_logits);
    }
    if (hp.logit_scale != 1.0f)
      for (int i = 0; i < hp.vocab; ++i) out_logits[i] *= hp.logit_scale;
    if (hp.logit_cap > 0) {
      for (int i = 0; i < hp.vocab; ++i)
        out_logits[i] = hp.logit_cap * std::tanh(out_logits[i] / hp.logit_cap);
    }
  }
};

}  // namespace molot::fwd
