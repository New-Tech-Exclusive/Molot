#pragma once
// GEMV (decode) + GEMM (prefill) kernels.
// Decode is bandwidth-bound: the only thing that matters is bytes streamed
// from DRAM per row + SIMD efficiency. Q8 uses VNNI int32 accumulation.
#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "quant.h"
#include "thread_pool.h"

namespace molot {

// ---- FP32 dot, AVX512 x16 ----
inline float dot_fp32_avx512(const float* a, const float* b, int n) {
  __m512 acc = _mm512_setzero_ps();
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 x = _mm512_loadu_ps(a + i);
    __m512 y = _mm512_loadu_ps(b + i);
    acc = _mm512_fmadd_ps(x, y, acc);
  }
  float tail = 0;
  for (; i < n; ++i) tail += a[i] * b[i];
  return _mm512_reduce_add_ps(acc) + tail;
}

// ---- Q8 GEMV row ----
// Activations are quantized ONCE per token, per 32 elems (same granularity
// as ggml Q8_0, so trajectories track the reference): see quantize_act_q8.
// Inner loop is VNNI vpdpbusd. AVX512-VNNI only offers unsigned*signed
// bytes (dpbusd), so activations are biased by +128 to unsigned and corrected
// with the block's weight sum: sum(xq*w) = sum(xu*w) - 128*sum(w).
struct QAct8 {
  static constexpr int BS = 32;
  int n = 0;
  std::vector<int8_t> q;      // [n] signed (for scalar tail)
  std::vector<uint8_t> qu;    // [n] biased +128 (for VNNI)
  std::vector<float> scales;  // per-32-block absmax/127
};

inline QAct8 quantize_act_q8(const float* x, int n) {
  QAct8 a;
  a.n = n;
  a.q.resize(n);
  a.qu.resize(n);
  a.scales.resize((n + 31) / 32);
  for (int s = 0; s < n; s += 32) {
    int e = s + 32 < n ? s + 32 : n;
    float amax = 0;
    for (int i = s; i < e; ++i) {
      float v = std::fabs(x[i]);
      if (v > amax) amax = v;
    }
    float xs = amax / 127.0f;
    if (!(xs > 0)) xs = 1.0f;
    xs = fp32_round_fp16(xs);  // ggml stores Q8_0 scales fp16: track bit-nearly
    a.scales[s / 32] = xs;
    for (int i = s; i < e; ++i) {
      // roundf (half away from zero), matching ggml quantize_row_q8_0_ref
      int v = (int)std::round(x[i] / xs);
      v = v > 127 ? 127 : (v < -127 ? -127 : v);
      a.q[i] = (int8_t)v;
      a.qu[i] = (uint8_t)(v + 128);
    }
  }
  return a;
}

inline float q8_row_dot_prequant(const int8_t* wq, const int32_t* wbsum, const QAct8& a,
                                 int n, float wscale) {
  float acc = 0;  // ggml-faithful: float, sequential per-32 order
  int s = 0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512VL__)
  for (; s + 32 <= n; s += 32) {
    __m256i vx = _mm256_loadu_si256((const __m256i*)(a.qu.data() + s));
    __m256i vw = _mm256_loadu_si256((const __m256i*)(wq + s));
    __m256i blk = _mm256_dpbusd_epi32(_mm256_setzero_si256(), vx, vw);
    alignas(32) int32_t tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, blk);
    int32_t isum = 0;
    for (int k = 0; k < 8; ++k) isum += tmp[k];
    int32_t wsum = wbsum ? wbsum[s / 32] : 0;
    if (!wbsum)
      for (int i = 0; i < 32; ++i) wsum += (int32_t)wq[s + i];
    acc += (float)(isum - 128 * wsum) * a.scales[s / 32];
  }
#endif
  for (; s < n;) {
    int b = (s / 32) * 32, e = b + 32 < n ? b + 32 : n;
    int32_t isum = 0;
    for (int i = s; i < e; ++i) isum += (int32_t)a.q[i] * (int32_t)wq[i];
    acc += (float)isum * a.scales[b / 32];
    s = e;
  }
  return acc * wscale;
}

// Legacy single-call variant (re-quantizes x per row: simple but rows-times
// redundant work; kept for tests, never use in hot loop).
inline float q8_row_dot_vnni(const int8_t* wq, const float* x, int n, float wscale) {
  QAct8 a = quantize_act_q8(x, n);
  return q8_row_dot_prequant(wq, nullptr, a, n, wscale);
}

// Q8 GEMV: y = W_q8 * x. Activation quantized once, rows share it.
inline void gemv_q8(const Q8Matrix& W, const float* x, float* y, ThreadPool& pool) {
  QAct8 a = quantize_act_q8(x, W.cols);
  pool.parallel_for(W.rows, [&](int, int64_t r) {
    y[r] = q8_row_dot_prequant(W.row((int)r), W.bsum_row((int)r), a, W.cols,
           W.scales[(int)r]);
  });
}

// Q4 GEMV, float reference path: unpack nibbles on the fly, per-GS(32)
// scale/zero. Vector unpack via AVX512BW (order-preserving byte interleave,
// no stack temps). Kept as the correctness reference for the int8 path.
inline float q4_group_dot_float(const uint8_t* pp, const float* xx, float scale,
                                float zero) {
#if defined(__AVX512BW__) && defined(__AVX512F__) && defined(__AVX512VL__)
  __m128i b = _mm_loadu_si128((const __m128i*)pp);  // 16 bytes -> 32 nibbles
  __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
  __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
  // byte interleave preserves order: uplo=[lo0,hi0,..,lo7,hi7], uphi=[lo8,hi8,..]
  __m128i uplo = _mm_unpacklo_epi8(lo, hi);
  __m128i uphi = _mm_unpackhi_epi8(lo, hi);
  __m512 w0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(uplo));
  __m512 w1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(uphi));
  __m512 x0 = _mm512_loadu_ps(xx);
  __m512 x1 = _mm512_loadu_ps(xx + 16);
  __m512 vz = _mm512_set1_ps(zero);
  __m512 d0 = _mm512_sub_ps(w0, vz);
  __m512 d1 = _mm512_sub_ps(w1, vz);
  __m512 vacc = _mm512_fmadd_ps(d0, x0, _mm512_mul_ps(d1, x1));
  return _mm512_reduce_add_ps(vacc) * scale;
#else
  __m512 vacc = _mm512_setzero_ps();
  __m512 vs = _mm512_set1_ps(scale);
  __m512 vz = _mm512_set1_ps(zero);
  alignas(64) float w[32];
  for (int k = 0; k < 32; ++k) {
    uint8_t b = pp[k / 2];
    w[k] = (float)((k & 1) ? ((b >> 4) & 0xF) : (b & 0xF));
  }
  __m512 x0 = _mm512_loadu_ps(xx);
  __m512 x1 = _mm512_loadu_ps(xx + 16);
  __m512 w0 = _mm512_loadu_ps(w);
  __m512 w1 = _mm512_loadu_ps(w + 16);
  vacc = _mm512_fmadd_ps(_mm512_sub_ps(w0, vz), x0, vacc);
  vacc = _mm512_fmadd_ps(_mm512_sub_ps(w1, vz), x1, vacc);
  return _mm512_reduce_add_ps(_mm512_mul_ps(vacc, vs));
#endif
}

inline float q4_row_dot(const uint8_t* packed, const float* x, int n,
                        const float* scales, const int8_t* zeros, int ngroups) {
  float acc = 0;
  for (int g = 0; g < ngroups; ++g)
    acc += q4_group_dot_float(packed + (size_t)g * Q4Matrix::GS / 2,
                              x + (size_t)g * Q4Matrix::GS, scales[g],
                              (float)zeros[g]);
  (void)n;
  return acc;
}

// Q4 GEMV, int8 fast path (production approach: Q4_0 x Q8_0). Per-group float
// reduces were the old cap (~15 cycles x ngroups). Instead: unpack nibbles to
// int8 once, dot against once-quantized int8 activations with VNNI dpbusd,
// one scalar rescale per group-pair. Stride is 2 groups (64 elems = one full
// VNNI register; lanes 0-7 = group g, 8-15 = group g+1). Activations are
// even/odd-split per 32-group (qup), so no interleave shuffle is needed.
inline float q4_row_dot_i8(const uint8_t* packed, const int16_t* wsums, const QAct8& a,
                            const uint8_t* qup, const float* scales, const int8_t* zeros,
                            int ngroups) {
  // ggml-faithful accumulation: float, sequential per-32-group order, so the
  // trajectory tracks the reference (correlated rounding, not just small).
  float acc = 0;
  int g = 0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512VL__)
  const __m128i lomask = _mm_set1_epi8(0x0F);
  for (; g + 2 <= ngroups; g += 2) {
    const uint8_t* pp = packed + (size_t)g * Q4Matrix::GS / 2;
    // nibbles -> bytes, evens/odds kept split to match qup order (no interleave)
    __m128i b0 = _mm_loadu_si128((const __m128i*)pp);
    __m128i ev0 = _mm_and_si128(b0, lomask);
    __m128i od0 = _mm_and_si128(_mm_srli_epi16(b0, 4), lomask);
    __m128i b1 = _mm_loadu_si128((const __m128i*)(pp + 16));
    __m128i ev1 = _mm_and_si128(b1, lomask);
    __m128i od1 = _mm_and_si128(_mm_srli_epi16(b1, 4), lomask);
    __m256i Wlo = _mm256_set_m128i(od0, ev0);  // [evens][odds], group g
    __m256i Whi = _mm256_set_m128i(od1, ev1);  // group g+1
    __m512i W8 = _mm512_castsi256_si512(Wlo);
    W8 = _mm512_inserti64x4(W8, Whi, 1);
    __m512i Z = _mm512_inserti64x4(_mm512_set1_epi8(zeros[g + 1]),
                                   _mm256_set1_epi8(zeros[g]), 0);
    __m512i W = _mm512_sub_epi8(W8, Z);
    __m512i X = _mm512_loadu_si512((const __m512i*)(qup + (size_t)g * 32));
    __m512i D = _mm512_dpbusd_epi32(_mm512_setzero_si512(), X, W);
    int32_t s_lo = _mm512_mask_reduce_add_epi32((__mmask16)0x00FF, D);
    int32_t s_all = _mm512_reduce_add_epi32(D);
    acc += (float)(s_lo - 128 * wsums[g]) * scales[g] * a.scales[g];
    acc += (float)(s_all - s_lo - 128 * wsums[g + 1]) * scales[g + 1] * a.scales[g + 1];
  }
#endif
  // tail (odd group out): exact scalar int path with quantized activations,
  // consistent with the VNNI loop above (the float path would use full-
  // precision x and disagree by ~1% activation-quant noise).
  for (; g < ngroups; ++g) {
    const uint8_t* pp = packed + (size_t)g * Q4Matrix::GS / 2;
    int32_t isum = 0;
    for (int k = 0; k < Q4Matrix::GS; ++k) {
      uint8_t b = pp[k / 2];
      int w = ((k & 1) ? ((b >> 4) & 0xF) : (b & 0xF)) - zeros[g];
      isum += w * (int32_t)a.q[(size_t)g * Q4Matrix::GS + k];
    }
    acc += (float)isum * scales[g] * a.scales[g];
  }
  return acc;
}

inline void gemv_q4(const Q4Matrix& W, const float* x, float* y, ThreadPool& pool) {
  QAct8 a = quantize_act_q8(x, W.cols);  // once per token, shared across rows
  // even/odd-split activations per 32-group (NOT per 64-block: group g's
  // evens are qu[g*32+2k], and the VNNI pair reads two consecutive groups):
  // qup[g*32+k] = qu[g*32+2k], qup[g*32+16+k] = qu[g*32+2k+1].
  // One pass per token; padded, VNNI only reads full pairs, tail uses a.q.
  int qn = (W.cols + 31) / 32 * 32;
  std::vector<uint8_t> qup(qn, 128);
  for (int s = 0; s < W.cols; s += 32) {
    int blen = W.cols - s < 32 ? W.cols - s : 32;
    for (int k = 0; k < blen; ++k) {
      if ((k & 1) == 0)
        qup[(size_t)s + k / 2] = a.qu[(size_t)s + k];
      else
        qup[(size_t)s + 16 + k / 2] = a.qu[(size_t)s + k];
    }
  }
  // exp/weight_prefetch: SW prefetch upcoming rows (env MOLOT_PREFETCH=N).
  // Each thread walks a contiguous range, so r+dist is almost always ours.
  int pfdist = 0;
  if (const char* e = std::getenv("MOLOT_PREFETCH")) pfdist = atoi(e);
  size_t rowbytes = (size_t)W.cols / 2;
  pool.parallel_for(W.rows, [&](int, int64_t r) {
    if (pfdist > 0 && r + pfdist < W.rows) {
      const uint8_t* f = W.data.data() + (size_t)(r + pfdist) * rowbytes;
      __builtin_prefetch(f, 0, 3);
      __builtin_prefetch(f + 1024, 0, 3);
      __builtin_prefetch(W.scales.data() + (size_t)(r + pfdist) * W.ngroups, 0, 3);
    }
    const uint8_t* prow = W.data.data() + (size_t)r * rowbytes;
    y[r] = q4_row_dot_i8(prow, W.wsums.data() + (size_t)r * W.ngroups, a, qup.data(),
                         W.scales.data() + (size_t)r * W.ngroups,
                         W.zeros.data() + (size_t)r * W.ngroups, W.ngroups);
  });
}

// Exact Q4 GEMV (diagnostic/precision reference): fp32 activations (no
// activation quantization), on-the-fly nibble dequant, double accumulation.
// If the model goes coherent under MOLOT_FP32=1, the int paths just need
// tightening (activation granularity); if not, a structural bug remains.
inline float q4_row_dot_exact(const uint8_t* packed, const float* x, const float* scales,
                              const int8_t* zeros, int ngroups) {
  double acc = 0;
  for (int g = 0; g < ngroups; ++g) {
    const uint8_t* pp = packed + (size_t)g * Q4Matrix::GS / 2;
    const float* xx = x + (size_t)g * Q4Matrix::GS;
    double ds = 0;
    for (int k = 0; k < Q4Matrix::GS; ++k) {
      uint8_t b = pp[k / 2];
      int v = (k & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
      ds += (double)((float)v - (float)zeros[g]) * (double)xx[k];
    }
    acc += ds * (double)scales[g];
  }
  return (float)acc;
}

inline void gemv_q4_exact(const Q4Matrix& W, const float* x, float* y, ThreadPool& pool) {
  // MOLOT_QACT=1: snap activations to the Q8 grid first (isolates activation
  // rounding noise from dot-product arithmetic).
  const float* xx = x;
  std::vector<float> xr;
  if (std::getenv("MOLOT_QACT")) {
    QAct8 a = quantize_act_q8(x, W.cols);
    xr.resize(W.cols);
    for (int s = 0; s < W.cols; s += 32) {
      int e = s + 32 < W.cols ? s + 32 : W.cols;
      for (int k = s; k < e; ++k) xr[k] = (float)a.q[k] * a.scales[s / 32];
    }
    xx = xr.data();
  }
  pool.parallel_for(W.rows, [&](int, int64_t r) {
    y[r] = q4_row_dot_exact(W.data.data() + (size_t)r * W.cols / 2, xx,
                            W.scales.data() + (size_t)r * W.ngroups,
                            W.zeros.data() + (size_t)r * W.ngroups, W.ngroups);
  });
}

// Exact Q8 GEMV: fp32 activations, double accumulation, single rescale.
inline void gemv_q8_exact(const Q8Matrix& W, const float* x, float* y, ThreadPool& pool) {
  pool.parallel_for(W.rows, [&](int, int64_t r) {
    const int8_t* wq = W.row((int)r);
    double acc = 0;
    for (int i = 0; i < W.cols; ++i) acc += (double)wq[i] * (double)x[i];
    y[r] = (float)(acc * (double)W.scales[(int)r]);
  });
}

// Exact Q8_0-block GEMV (GGUF native layout, no requant): fp32 activations,
// per-32 scales, double accumulation. For tied heads in exact mode.
inline void gemv_q8_0_exact(const uint8_t* data, int rows, int cols, const float* x,
                            float* y, ThreadPool& pool) {
  pool.parallel_for(rows, [&](int, int64_t r) {
    const uint8_t* row = data + (size_t)r * (cols / 32) * 34;
    double acc = 0;
    for (int g = 0; g < cols / 32; ++g) {
      const uint8_t* blk = row + (size_t)g * 34;
      uint16_t dh;
      std::memcpy(&dh, blk, 2);
      float d = fp16_to_fp32(dh);
      double ds = 0;
      for (int k = 0; k < 32; ++k) ds += (double)(int8_t)blk[2 + k] * (double)x[g * 32 + k];
      acc += ds * (double)d;
    }
    y[r] = (float)acc;
  });
}

// Fast Q8_0-block GEMV (GGUF native layout, no requant noise): VNNI per-32
// dots with once-quantized activations and the blocks' own fp16 scales.
// Scalar core (head GEMV is bandwidth-bound); vectorize if it matters.
inline void gemv_q8_0(const uint8_t* data, int rows, int cols, const float* x, float* y,
                       ThreadPool& pool) {
  QAct8 a = quantize_act_q8(x, cols);  // once per call, shared across rows
  pool.parallel_for(rows, [&](int, int64_t r) {
    const uint8_t* row = data + (size_t)r * (cols / 32) * 34;
    double acc = 0;
    for (int g = 0; g < cols / 32; ++g) {
      const uint8_t* blk = row + (size_t)g * 34;
      uint16_t dh;
      std::memcpy(&dh, blk, 2);
      float d = fp16_to_fp32(dh);
      int32_t isum = 0, wsum = 0;
      for (int k = 0; k < 32; ++k) {
        int8_t w = (int8_t)blk[2 + k];
        isum += (int32_t)a.q[g * 32 + k] * (int32_t)w;
        wsum += (int32_t)w;
      }
      // NOTE: scalar core uses signed*q directly (no +128 bias needed);
      // VNNI fast path (below) uses the biased form with wsum correction.
      acc += (double)isum * (double)d * (double)a.scales[g];
      (void)wsum;
    }
    y[r] = (float)acc;
  });
}

// ---- Batched GEMM (prefill path) ----
// Y[m][n] = W[n][K] . X[m][K] for m in [0,M). Weights are streamed ONCE and
// reused across M tokens (vs M streams in sequential decode). Each activation
// row is quantized independently with the exact decode path (per-32 roundf),
// so numerics track sequential decode to gate tolerance.
inline void gemm_q4_batch(const Q4Matrix& W, const float* X, float* Y, int M,
                          ThreadPool& pool) {
  int K = W.cols, N = W.rows;
  std::vector<QAct8> acts(M);
  std::vector<std::vector<uint8_t>> qups(M);
  pool.parallel_for(M, [&](int, int64_t m) {
    acts[m] = quantize_act_q8(X + (size_t)m * K, K);
    int qn = (K + 31) / 32 * 32;
    qups[m].assign(qn, 128);
    for (int s = 0; s < K; s += 32) {
      int blen = K - s < 32 ? K - s : 32;
      for (int k = 0; k < blen; ++k) {
        if ((k & 1) == 0)
          qups[m][(size_t)s + k / 2] = acts[m].qu[(size_t)s + k];
        else
          qups[m][(size_t)s + 16 + k / 2] = acts[m].qu[(size_t)s + k];
      }
    }
  });
  pool.parallel_for(N, [&](int, int64_t n) {
    const uint8_t* prow = W.data.data() + (size_t)n * K / 2;
    const float* sc = W.scales.data() + (size_t)n * W.ngroups;
    const int8_t* ze = W.zeros.data() + (size_t)n * W.ngroups;
    const int16_t* ws = W.wsums.data() + (size_t)n * W.ngroups;
    for (int m = 0; m < M; ++m)
      Y[(size_t)m * N + n] = q4_row_dot_i8(prow, ws, acts[m], qups[m].data(), sc, ze,
                                           W.ngroups);
  });
}

inline void gemm_q8_batch(const Q8Matrix& W, const float* X, float* Y, int M,
                          ThreadPool& pool) {
  int K = W.cols, N = W.rows;
  std::vector<QAct8> acts(M);
  pool.parallel_for(M, [&](int, int64_t m) { acts[m] = quantize_act_q8(X + (size_t)m * K, K); });
  pool.parallel_for(N, [&](int, int64_t n) {
    for (int m = 0; m < M; ++m)
      Y[(size_t)m * N + n] =
          q8_row_dot_prequant(W.row((int)n), W.bsum_row((int)n), acts[m], K, W.scales[n]);
  });
}

// Causal prefill attention: queries Qm (M tokens, positions pos0+m) against
// the filled cache [0,N). GQA, SWA window. Parallel over (head, position).
inline void attn_prefill_gqa(const float* Qm, const float* K, const float* V, float* Om,
                              int nq, int nkv, int M, int N, int d, float scale, int pos0,
                              int window, ThreadPool& pool) {
  int groups = nq / nkv;
  size_t S = (size_t)nkv * d;  // interleaved cache stride
  pool.parallel_for((int64_t)nq * M, [&](int, int64_t hm) {
    int h = (int)(hm / M), m = (int)(hm % M);
    int kv = h / groups;
    int pos = pos0 + m;
    int t0 = 0;
    if (window > 0 && pos - window + 1 > t0) t0 = pos - window + 1;
    if (t0 > N) t0 = N;
    int nn = pos + 1 < N ? pos + 1 : N;  // causal: slots [t0, nn)
    const float* qh = Qm + ((size_t)m * nq + h) * d;
    float* oh = Om + ((size_t)m * nq + h) * d;
    static thread_local std::vector<float> buf;
    if ((int)buf.size() < N) buf.resize(N);
    float* scores = buf.data();
    float mx = -1e30f;
    for (int t = t0; t < nn; ++t) {
      float s = dot_fp32_avx512(qh, K + (size_t)t * S + (size_t)kv * d, d) * scale;
      scores[t] = s;
      if (s > mx) mx = s;
    }
    double den = 0;
    for (int t = t0; t < nn; ++t) {
      scores[t] = std::exp(scores[t] - mx);
      den += scores[t];
    }
    float inv = (float)(1.0 / den);
    for (int j = 0; j < d; ++j) oh[j] = 0;
    for (int t = t0; t < nn; ++t) {
      float w = scores[t] * inv;
      const float* vv = V + (size_t)t * S + (size_t)kv * d;
      for (int j = 0; j < d; ++j) oh[j] += vv[j] * w;
    }
  });
}

// FP32 GEMV threaded (baseline for bandwidth comparison).
inline void gemv_fp32_mt(const float* W, const float* x, float* y, int rows, int cols,
                         ThreadPool& pool) {
  pool.parallel_for(rows, [&](int, int64_t r) {
    y[r] = dot_fp32_avx512(W + r * cols, x, cols);
  });
}

// FP32 GEMM C = A(MxK) * B(KxN), f32. Bt is B transposed (NxK row-major) so
// the inner loop is a contiguous dot — the prefill path is compute-bound and
// strided-B would leave 10x on the table. Threaded over M.
inline void gemm_fp32_mt(const float* A, const float* Bt, float* C, int M, int N, int K,
                         ThreadPool& pool) {
  pool.parallel_for(M, [&](int, int64_t m) {
    for (int n = 0; n < N; ++n) {
      C[m * N + n] = dot_fp32_avx512(A + m * K, Bt + (size_t)n * K, K);
    }
  });
}

// Cache-blocked GEMM: the "keep the reused in cache" kernel. Naive GEMM above
// re-reads all of Bt (N*K) and the A row (K) for every output row: O(M*N*K)
// traffic each. Tiling (Mc rows x Kc rank) flips reuse inside out:
//
//   * C tile (Mc x N) is zeroed once and kept in L2/L3 across all Kc passes;
//     weights are pushed back to DRAM only when the tile completes.
//   * Bt panel (N x Kc) is shared across Mc rows: Bt traffic drops by Mc.
//   * A is read once total (was: re-read N times).
//
// Cost: C tile pays K/Kc read-write passes instead of one write. Wins whenever
// Bt/A dominate, i.e. any prefill that doesn't fit in L3. Mc=32, Kc=1024 sized
// for Zen4 (1 MB L2/core, 32 MB shared L3); Mc=32 also keeps block count above
// thread count at modest M so no core idles.
inline void gemm_fp32_blocked(const float* A, const float* Bt, float* C, int M, int N,
                              int K, ThreadPool& pool) {
  constexpr int Mc = 32, Kc = 1024;
  int nmb = (M + Mc - 1) / Mc;
  pool.parallel_for(nmb, [&](int, int64_t mb) {
    int m0 = (int)mb * Mc, m1 = m0 + Mc < M ? m0 + Mc : M;
    // C tile lives in cache for the whole kc sweep; DRAM sees it once.
    for (int m = m0; m < m1; ++m)
      for (int n = 0; n < N; ++n) C[(size_t)m * N + n] = 0.0f;
    for (int kc = 0; kc < K; kc += Kc) {
      int klen = kc + Kc < K ? Kc : K - kc;
      for (int m = m0; m < m1; ++m) {
        const float* a = A + (size_t)m * K + kc;
        float* crow = C + (size_t)m * N;
        for (int n = 0; n < N; ++n)
          crow[n] += dot_fp32_avx512(a, Bt + (size_t)n * K + kc, klen);
      }
    }
  });
}

}  // namespace molot
