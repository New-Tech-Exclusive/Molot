#pragma once
// RMSNorm + fused SiLU-gated MLP tail. Single streaming pass, AVX512.
#include <immintrin.h>

#include <cmath>

namespace molot {

inline void rmsnorm(const float* x, const float* w, float* y, int n, float eps = 1e-6f) {
  __m512 sq = _mm512_setzero_ps();
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 v = _mm512_loadu_ps(x + i);
    sq = _mm512_fmadd_ps(v, v, sq);
  }
  float tail = 0;
  for (; i < n; ++i) tail += x[i] * x[i];
  float mean = (_mm512_reduce_add_ps(sq) + tail) / n;
  float inv = 1.0f / std::sqrt(mean + eps);
  __m512 vinv = _mm512_set1_ps(inv);
  i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 v = _mm512_loadu_ps(x + i);
    __m512 g = _mm512_loadu_ps(w + i);
    _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_mul_ps(v, vinv), g));
  }
  for (; i < n; ++i) y[i] = x[i] * inv * w[i];
}

// Vector exp (AVX512F): range-reduce x = n*ln2 + r, Taylor-5 poly on r,
// 2^n via scalef. Max err ~3e-6 on [-87, 0] — plenty for softmax (which only
// ever evaluates exp(s - max) <= 0). ~12 instr vs ~50+ for scalar libm exp,
// and this loop dominates long-context attention. Clamps to +-88 so float
// can never overflow (exp(88) = 1.6e38 < 3.4e38).
inline __m512 exp_avx512_ps(__m512 x) {
  x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-88.0f)), _mm512_set1_ps(88.0f));
  __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.4426950409f)),
                                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.6931471806f), x);
  __m512 p = _mm512_set1_ps(1.0f / 120);
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 24));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 6));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
  return _mm512_scalef_ps(p, n);
}

// y = silu(gate) * up, elementwise. Clamped vector exp (vexp saturates at
// +-88) so float can never overflow: the denominator stays in [1, 1.7e38].
// (An earlier libm-exp version produced NaNs under -ffast-math codegen.)
inline void silu_mul(const float* gate, const float* up, float* y, int n) {
  __m512 one = _mm512_set1_ps(1.0f);
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 g = _mm512_loadu_ps(gate + i);
    __m512 u = _mm512_loadu_ps(up + i);
    __m512 e = exp_avx512_ps(_mm512_sub_ps(_mm512_setzero_ps(), g));
    __m512 s = _mm512_div_ps(g, _mm512_add_ps(one, e));
    _mm512_storeu_ps(y + i, _mm512_mul_ps(s, u));
  }
  for (; i < n; ++i) {
    float g = gate[i];
    float s = g > 88.0f ? g : (g < -88.0f ? 0.0f : g / (1.0f + std::exp(-g)));
    y[i] = s * up[i];
  }
}

// y = gelu(gate) * up, elementwise. tanh approximation matching ggml_gelu_f32
// exactly (0.5x(1+tanh(0.79788456x(1+0.044715x^2)))). Scalar loop:
// L1-resident elementwise is negligible next to GEMVs, and libmvec/reciprocal
// codegen under -ffast-math produced NaNs here before (see history).
inline void gelu_mul(const float* gate, const float* up, float* y, int n) {
  for (int i = 0; i < n; ++i) {
    float g = gate[i];
    float t = 0.79788456080286536f * g * (1.0f + 0.044715f * g * g);
    y[i] = (0.5f * g * (1.0f + std::tanh(t))) * up[i];
  }
}

// plain RMSNorm without weights (unit norm; used for V and code paths)
inline void rmsnorm_plain(const float* x, float* y, int n, float eps = 1e-6f) {
  __m512 sq = _mm512_setzero_ps();
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 v = _mm512_loadu_ps(x + i);
    sq = _mm512_fmadd_ps(v, v, sq);
  }
  float tail = 0;
  for (; i < n; ++i) tail += x[i] * x[i];
  float inv = 1.0f / std::sqrt((_mm512_reduce_add_ps(sq) + tail) / n + eps);
  __m512 vinv = _mm512_set1_ps(inv);
  i = 0;
  for (; i + 16 <= n; i += 16) _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), vinv));
  for (; i < n; ++i) y[i] = x[i] * inv;
}

}  // namespace molot
