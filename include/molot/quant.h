#pragma once
// Row-wise quantization: Q8_0 (int8 + fp32 scale) and Q4_0 (nibble + fp16 scale).
// Layout is the interface: weights are stored quantized, streamed once per token.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <immintrin.h>  // _cvtss_sh (F16C) for fp16 scale rounding

namespace molot {

struct Q8Row {
  float scale = 1.0f;  // absmax/127
  int8_t* q = nullptr;
};

struct Q8Matrix {
  int rows = 0, cols = 0;
  static constexpr int BS = 32;  // VNNI block = activation block (matches ggml)
  int nblocks = 0;               // (cols+31)/32
  std::vector<float> scales;   // [rows]
  std::vector<int8_t> data;    // [rows*cols]
  std::vector<int32_t> bsums;  // [rows*nblocks]: per-32 weight sums for the
                               // dpbusd +128-bias correction (computed once,
                               // offline — never in the hot loop)
  const int8_t* row(int r) const { return data.data() + (size_t)r * cols; }
  const int32_t* bsum_row(int r) const { return bsums.data() + (size_t)r * nblocks; }
};

// absmax quantize; cols can be any size.
inline void quantize_q8_row(const float* Wrow, int cols, int nblocks, float* scale,
                            int8_t* data, int32_t* bsums) {
  float amax = 0;
  for (int c = 0; c < cols; ++c) amax = std::max(amax, std::fabs(Wrow[c]));
  float s = amax / 127.0f;
  if (!(s > 0)) s = 1.0f;
  *scale = s;
  for (int c = 0; c < cols; ++c) {
    int v = (int)std::nearbyint(Wrow[c] / s);
    data[c] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
  }
  for (int b = 0; b < nblocks; ++b) {
    int32_t sum = 0;
    int e = b * 32 + 32 < cols ? b * 32 + 32 : cols;
    for (int c = b * 32; c < e; ++c) sum += (int32_t)data[c];
    bsums[b] = sum;
  }
}

inline Q8Matrix quantize_q8(const float* W, int rows, int cols) {
  Q8Matrix m;
  m.rows = rows;
  m.cols = cols;
  m.nblocks = (cols + 31) / 32;
  m.scales.resize(rows);
  m.data.resize((size_t)rows * cols);
  m.bsums.assign((size_t)rows * m.nblocks, 0);
  for (int r = 0; r < rows; ++r)
    quantize_q8_row(W + (size_t)r * cols, cols, m.nblocks, &m.scales[r],
                    m.data.data() + (size_t)r * cols,
                    m.bsums.data() + (size_t)r * m.nblocks);
  return m;
}

// Q4_0 with fine groups (GS=32, like production runtimes): 4-bit nibbles with
// per-group fp32 scale + int8 zero. Bandwidth ≈ 0.5 B/elem + 5 B/32 elems.
// Coarse per-row groups collapse on outliers; GS=32 keeps <3% error.
struct Q4Matrix {
  static constexpr int GS = 32;
  int rows = 0, cols = 0;  // cols must be a multiple of GS (and even)
  int ngroups = 0;         // cols / GS
  std::vector<float> scales;   // [rows*ngroups]
  std::vector<int8_t> zeros;   // [rows*ngroups]
  std::vector<uint8_t> data;   // [rows*cols/2]
  std::vector<int16_t> wsums;  // [rows*ngroups]: Σ(nibble - zero) per group,
                               // for the VNNI +128-bias correction (offline).
                               // |sum| <= 32*15 = 480, int16 is exact.
};

// Single-row Q4 quantization into preallocated parts (for streaming repack of
// K-quants: dequant one row, requant, no full-matrix fp32 transient).
inline void quantize_q4_row(const float* Wrow, int cols, int ngroups, float* scales,
                            int8_t* zeros, uint8_t* data, int16_t* wsums) {
  for (int g = 0; g < ngroups; ++g) {
    const float* grp = Wrow + (size_t)g * Q4Matrix::GS;
    float mn = grp[0], mx = grp[0];
    for (int k = 0; k < Q4Matrix::GS; ++k) {
      mn = std::min(mn, grp[k]);
      mx = std::max(mx, grp[k]);
    }
    float s = (mx - mn) / 15.0f;
    if (!(s > 0)) s = 1.0f;
    int z = (int)std::nearbyint(-mn / s);
    z = z < 0 ? 0 : (z > 15 ? 15 : z);
    scales[g] = s;
    zeros[g] = (int8_t)z;
    int wsum = 0;
    for (int k = 0; k < Q4Matrix::GS; k += 2) {
      int v0 = (int)std::nearbyint(grp[k] / s) + z;
      int v1 = (int)std::nearbyint(grp[k + 1] / s) + z;
      v0 = v0 < 0 ? 0 : (v0 > 15 ? 15 : v0);
      v1 = v1 < 0 ? 0 : (v1 > 15 ? 15 : v1);
      data[(size_t)g * Q4Matrix::GS / 2 + k / 2] = (uint8_t)(v0 | (v1 << 4));
      wsum += (v0 - z) + (v1 - z);
    }
    wsums[g] = (int16_t)wsum;
  }
}

inline Q4Matrix quantize_q4(const float* W, int rows, int cols) {
  Q4Matrix m;
  m.rows = rows;
  m.cols = cols;
  m.ngroups = cols / Q4Matrix::GS;
  m.scales.resize((size_t)rows * m.ngroups);
  m.zeros.resize((size_t)rows * m.ngroups);
  m.data.resize((size_t)rows * cols / 2);
  m.wsums.assign((size_t)rows * m.ngroups, 0);
  for (int r = 0; r < rows; ++r)
    quantize_q4_row(W + (size_t)r * cols, cols, m.ngroups, m.scales.data() + (size_t)r * m.ngroups,
                    m.zeros.data() + (size_t)r * m.ngroups, m.data.data() + (size_t)r * cols / 2,
                    m.wsums.data() + (size_t)r * m.ngroups);
  return m;
}

inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {  // subnormal: renormalize
      exp = 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3FFu;
      f = sign | ((exp + 112) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    f = sign | ((exp + 112) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

// Round an fp32 value to fp16 precision (value-preserving roundtrip).
// Matches ggml storing activation scales as fp16 (GGML_FP32_TO_FP16).
inline float fp32_round_fp16(float f) {
  uint16_t h;
#if defined(__F16C__)
  h = _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
  uint32_t u;
  std::memcpy(&u, &f, 4);
  uint32_t sign = (u >> 16) & 0x8000u;
  int exp = ((int)((u >> 23) & 0xFF)) - 112;
  uint32_t mant = u & 0x7FFFFFu;
  if (exp <= 0) {
    h = (uint16_t)sign;  // flush (activation scales never underflow)
  } else if (exp >= 31) {
    h = (uint16_t)(sign | 0x7BFFu);
  } else {
    uint32_t m = mant >> 13;  // round mantissa to 10 bits, nearest-even
    uint32_t rest = mant & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (m & 1u))) {
      if (++m == 0x400u) {
        m = 0;
        if (++exp >= 31) return fp16_to_fp32((uint16_t)(sign | 0x7BFFu));
      }
    }
    h = (uint16_t)(sign | ((uint32_t)exp << 10) | m);
  }
#endif
  return fp16_to_fp32(h);
}

inline float bf16_to_fp32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint16_t fp32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  // round-to-nearest-even
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}

}  // namespace molot
