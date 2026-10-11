#pragma once
// Decode attention: single query vs KV-cache. O(N*d) traffic, never O(N^2).
// Online softmax over 256-elem blocks; heads parallelized by caller via pool.
// RoPE is fused into the QK dot (no temp buffers).
#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "gemm.h"
#include "norm.h"
#include "thread_pool.h"

namespace molot {

// RoPE rotation, ggml NEOX convention: pair k is (x[k], x[k+d/2])
// (half-split, NOT adjacent interleave). inv_freq has d/2 entries.
inline void rope_apply(float* x, int d, int pos, const float* inv_freq) {
  int h = d / 2;
  for (int i = 0; i < h; ++i) {
    float ang = pos * inv_freq[i];
    float c = std::cos(ang), s = std::sin(ang);
    float x0 = x[i], x1 = x[i + h];
    x[i] = x0 * c - x1 * s;
    x[i + h] = x0 * s + x1 * c;
  }
}

// Standard analytic inv_freq: 1/(base^(2i/d)), i in [0, d/2).
inline void rope_inv_freq(float base, int d, float* out) {
  for (int i = 0; i < d / 2; ++i) out[i] = 1.0f / std::pow(base, (2.0f * i) / d);
}

// (exp_avx512_ps lives in norm.h; softmax below uses it.)

// Rotary embedding applied on the fly inside dot: pair (2i,2i+1) rotated by
// angle[pos*inv_freq[i]]. q,k are full head-dim vectors. d <= 256, n up to 8k:
// the KV stream dominates, so scalar rotate is fine and keeps no temp buffers.
inline float qk_dot_rope(const float* q, const float* k, const float* inv_freq, int d,
                         int pos_q, int pos_k) {
  float acc = 0;
  for (int i = 0; i < d; i += 2) {
    int pair = i / 2;
    float cq = std::cos(pos_q * inv_freq[pair]), sq = std::sin(pos_q * inv_freq[pair]);
    float ck = std::cos(pos_k * inv_freq[pair]), sk = std::sin(pos_k * inv_freq[pair]);
    float rq0 = q[i] * cq - q[i + 1] * sq, rq1 = q[i] * sq + q[i + 1] * cq;
    float rk0 = k[i] * ck - k[i + 1] * sk, rk1 = k[i] * sk + k[i + 1] * ck;
    acc += rq0 * rk0 + rq1 * rk1;
  }
  return acc;
}

// Windowed variant: cache slot t holds position t (same timeline), current
// position pos_cur. Slots with t < pos_cur - window + 1 are masked out.
// window <= 0 disables masking (full causal prefix, the common case).
inline void attn_decode_head_win(const float* Q, const float* K, const float* V, float* out,
                                 int n, int d, float scale, int pos_cur, int window) {
  static thread_local std::vector<float> buf;
  if ((int)buf.size() < n) buf.resize(n);
  float* scores = buf.data();
  int t0 = window > 0 && pos_cur - window + 1 > 0 ? pos_cur - window + 1 : 0;
  float m = -1e30f;
  for (int t = t0; t < n; ++t) {
    if (t + 8 < n) {
      __builtin_prefetch(K + (size_t)(t + 8) * d, 0, 0);
      __builtin_prefetch(V + (size_t)(t + 8) * d, 0, 0);
    }
    float s = dot_fp32_avx512(Q, K + (size_t)t * d, d) * scale;
    scores[t] = s;
    if (s > m) m = s;
  }
  float den = 0;
  {  // vector exp; this loop owns the long-context profile with scalar libm
    __m512 vm = _mm512_set1_ps(m);
    int t = t0;
    for (; t + 16 <= n; t += 16) {
      __m512 s = _mm512_sub_ps(_mm512_loadu_ps(scores + t), vm);
      s = exp_avx512_ps(s);
      _mm512_storeu_ps(scores + t, s);
      den += _mm512_reduce_add_ps(s);
    }
    for (; t < n; ++t) {
      scores[t] = std::exp(scores[t] - m);
      den += scores[t];
    }
  }
  float inv = 1.0f / den;
  for (int j = 0; j < d; ++j) out[j] = 0;
  for (int t = t0; t < n; ++t) {
    float w = scores[t] * inv;
    __m512 vw = _mm512_set1_ps(w);
    int j = 0;
    for (; j + 16 <= d; j += 16) {
      __m512 o = _mm512_loadu_ps(out + j);
      __m512 v = _mm512_loadu_ps(V + (size_t)t * d + j);
      _mm512_storeu_ps(out + j, _mm512_fmadd_ps(v, vw, o));
    }
    for (; j < d; ++j) out[j] += V[(size_t)t * d + j] * w;
  }
}

// Plain (no-RoPE) head attention decode: full causal prefix.
inline void attn_decode_head(const float* Q, const float* K, const float* V, float* out,
                             int n, int d, float scale) {
  attn_decode_head_win(Q, K, V, out, n, d, scale, n > 0 ? n : 0, 0);
}

// GQA decode: nq query heads share nkv KV heads (nq % nkv == 0).
// Parallel over query heads via pool.
inline void attn_decode_gqa(const float* Q /*[nq,d]*/, const float* K /*[nkv,n,d]*/,
                            const float* V /*[nkv,n,d]*/, float* O /*[nq,d]*/, int nq,
                            int nkv, int n, int d, float scale, ThreadPool& pool) {
  int groups = nq / nkv;
  pool.parallel_for(nq, [&](int, int64_t h) {
    int kv = (int)h / groups;
    attn_decode_head(Q + h * d, K + (size_t)kv * n * d, V + (size_t)kv * n * d,
                     O + h * d, n, d, scale);
  });
}

// Cache-strided GQA for the model forward: K/V layout is [n][nkv*khd]
// (interleaved slots, appended per token); head kv, slot t lives at
// base + t*(nkv*khd) + kv*khd. qhd must equal khd (enforced at load).
inline void attn_decode_gqa_cache(const float* Q, const float* K, const float* V, float* O,
                                  int nq, int nkv, int n, int qhd, float scale, int pos_cur,
                                  int window, ThreadPool& pool) {
  int groups = nq / nkv;
  size_t S = (size_t)nkv * qhd;
  pool.parallel_for(nq, [&](int, int64_t h) {
    int kv = (int)h / groups;
    const float* qh = Q + h * qhd;
    float* oh = O + h * qhd;
    static thread_local std::vector<float> buf;
    if ((int)buf.size() < n) buf.resize(n);
    float* scores = buf.data();
    int t0 = window > 0 && pos_cur - window + 1 > 0 ? pos_cur - window + 1 : 0;
    float m = -1e30f;
    for (int t = t0; t < n; ++t) {
      if (t + 8 < n) {
        __builtin_prefetch(K + (size_t)(t + 8) * S, 0, 0);
        __builtin_prefetch(V + (size_t)(t + 8) * S, 0, 0);
      }
      float s = dot_fp32_avx512(qh, K + (size_t)t * S + (size_t)kv * qhd, qhd) * scale;
      scores[t] = s;
      if (s > m) m = s;
    }
    float den = 0;
    __m512 vm = _mm512_set1_ps(m);
    int t = t0;
    for (; t + 16 <= n; t += 16) {
      // scores are contiguous; weights applied to strided V rows below
      __m512 s = _mm512_sub_ps(_mm512_loadu_ps(scores + t), vm);
      s = exp_avx512_ps(s);
      _mm512_storeu_ps(scores + t, s);
      den += _mm512_reduce_add_ps(s);
    }
    for (; t < n; ++t) {
      scores[t] = std::exp(scores[t] - m);
      den += scores[t];
    }
    float inv = 1.0f / den;
    for (int j = 0; j < qhd; ++j) oh[j] = 0;
    for (int tt = t0; tt < n; ++tt) {
      float w = scores[tt] * inv;
      __m512 vw = _mm512_set1_ps(w);
      const float* vv = V + (size_t)tt * S + (size_t)kv * qhd;
      int j = 0;
      for (; j + 16 <= qhd; j += 16) {
        __m512 o = _mm512_loadu_ps(oh + j);
        __m512 v = _mm512_loadu_ps(vv + j);
        _mm512_storeu_ps(oh + j, _mm512_fmadd_ps(v, vw, o));
      }
      for (; j < qhd; ++j) oh[j] += vv[j] * w;
    }
  });
}
inline void attn_decode_gqa_win(const float* Q, const float* K, const float* V, float* O,
                                int nq, int nkv, int n, int d, float scale, int pos_cur,
                                int window, ThreadPool& pool) {
  int groups = nq / nkv;
  pool.parallel_for(nq, [&](int, int64_t h) {
    int kv = (int)h / groups;
    attn_decode_head_win(Q + h * d, K + (size_t)kv * n * d, V + (size_t)kv * n * d,
                         O + h * d, n, d, scale, pos_cur, window);
  });
}

}  // namespace molot
