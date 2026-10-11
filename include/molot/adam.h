#pragma once
// Fused AdamW step (training path: LoRA adapters only, BF16 params).
// One streaming pass over p/m/v/g: no temp buffers, AVX512.
#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace molot {

inline void adamw_fused(float* p, float* m, float* v, const float* g, int n, float lr,
                        float b1, float b2, float eps, float wd, float bias1,
                        float bias2) {
  __m512 vb1 = _mm512_set1_ps(b1), vb2 = _mm512_set1_ps(b2);
  __m512 vc1 = _mm512_set1_ps(bias1), vc2 = _mm512_set1_ps(bias2);
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 pp = _mm512_loadu_ps(p + i);
    __m512 mm = _mm512_loadu_ps(m + i);
    __m512 vv = _mm512_loadu_ps(v + i);
    __m512 gg = _mm512_loadu_ps(g + i);
    __m512 one = _mm512_set1_ps(1.0f);
    mm = _mm512_fmadd_ps(_mm512_sub_ps(one, vb1), gg, _mm512_mul_ps(vb1, mm));
    vv = _mm512_fmadd_ps(_mm512_sub_ps(one, vb2), _mm512_mul_ps(gg, gg),
                         _mm512_mul_ps(vb2, vv));
    _mm512_storeu_ps(m + i, mm);
    _mm512_storeu_ps(v + i, vv);
    // bias-corrected update needs sqrt+div per lane: do via L1 arrays
    // (optimizer is DRAM-bound; this stays off the critical path).
    alignas(64) float pa[16], ma[16], va[16];
    _mm512_storeu_ps(pa, pp);
    _mm512_storeu_ps(ma, _mm512_div_ps(mm, vc1));
    _mm512_storeu_ps(va, _mm512_div_ps(vv, vc2));
    for (int k = 0; k < 16; ++k)
      pa[k] -= lr * (ma[k] / (std::sqrt(va[k]) + eps) + wd * pa[k]);
    _mm512_storeu_ps(p + i, _mm512_loadu_ps(pa));
  }
  for (; i < n; ++i) {
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    float mh = m[i] / bias1, vh = v[i] / bias2;
    p[i] -= lr * (mh / (std::sqrt(vh) + eps) + wd * p[i]);
  }
}

}  // namespace molot
