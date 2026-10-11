#pragma once
// GGUF block repack into molot runtime layouts. Exact (bit-preserving) where
// the formats align, single-quant noise where they don't (documented below).
//
//   Q4_0 (per-32: fp16 d + 16B nibbles, val=(nib-8)*d) -> Q4Matrix GS=32:
//     EXACT. scale=d, zero=8, nibbles copied verbatim, wsums recomputed.
//   Q8_0 (per-32: fp16 d + 32xi8) -> Q8Matrix (per-row absmax):
//     requantized (double-quant noise ~2^-7 relative, negligible next to the
//     8-bit error itself). Used for the LM head; embeddings stay dequantized
//     F32 (gather needs one row at a time anyway).
//   F32 -> memcpy. BF16 -> widen. F16 -> widen.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "molot/gguf.h"
#include "molot/quant.h"

namespace molot::repack {

// (fp16_to_fp32 lives in quant.h; repack uses it for Q4_0/Q8_0 scales.)

// ---- K-quant dequantizers (exact ggml logic; see ggml-quants.c) ----

// GGUF dim convention: dims[0] = row length (K), dims[1] = row count (N).
inline void gguf_shape_2d(const gguf::Tensor& t, int& rows, int& cols) {
  if (t.dims.size() != 2) throw std::runtime_error("repack: expected 2-D tensor " + t.name);
  cols = (int)t.dims[0];
  rows = (int)t.dims[1];
}

// Q4_K super-block (256 elems, 144 B): fp16 d,dmin + 12B 6-bit scales/mins +
// 128B nibbles. Nibble layout per 64-elem chunk: byte l holds elems l (low)
// and l+32 (high) — same half-split family as Q4_0.
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

inline void dequant_q4K_row(const uint8_t* data, int cols, float* out) {
  for (int sb = 0; sb < cols; sb += 256) {
    float d = fp16_to_fp32((uint16_t)(data[0] | (data[1] << 8)));
    float mn = fp16_to_fp32((uint16_t)(data[2] | (data[3] << 8)));
    const uint8_t* scales = data + 4;
    const uint8_t* qs = data + 16;
    float* y = out + sb;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
      uint8_t sc, m;
      get_scale_min_k4(is, scales, &sc, &m);
      float d1 = d * sc, m1 = mn * m;
      get_scale_min_k4(is + 1, scales, &sc, &m);
      float d2 = d * sc, m2 = mn * m;
      for (int l = 0; l < 32; ++l) y[j + l] = d1 * (qs[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l) y[j + 32 + l] = d2 * (qs[l] >> 4) - m2;
      qs += 32;
      is += 2;
    }
    data += 144;
  }
}

inline void dequant_q5K_row(const uint8_t* data, int cols, float* out) {
  for (int sb = 0; sb < cols; sb += 256) {
    float d = fp16_to_fp32((uint16_t)(data[0] | (data[1] << 8)));
    float mn = fp16_to_fp32((uint16_t)(data[2] | (data[3] << 8)));
    const uint8_t* scales = data + 4;
    const uint8_t* qh = data + 16;
    const uint8_t* ql = data + 48;
    float* y = out + sb;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
      uint8_t sc, m;
      get_scale_min_k4(is, scales, &sc, &m);
      float d1 = d * sc, m1 = mn * m;
      get_scale_min_k4(is + 1, scales, &sc, &m);
      float d2 = d * sc, m2 = mn * m;
      for (int l = 0; l < 32; ++l)
        y[j + l] = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
      for (int l = 0; l < 32; ++l)
        y[j + 32 + l] = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
      ql += 32;
      is += 2;
      u1 <<= 2;
      u2 <<= 2;
    }
    data += 176;
  }
}

inline void dequant_q6K_row(const uint8_t* data, int cols, float* out) {
  for (int sb = 0; sb < cols; sb += 256) {
    float d = fp16_to_fp32((uint16_t)(data[208] | (data[209] << 8)));
    const uint8_t* ql = data;
    const uint8_t* qh = data + 128;
    const int8_t* sc = (const int8_t*)(data + 192);
    float* y = out + sb;
    for (int n = 0; n < 256; n += 128) {
      for (int l = 0; l < 32; ++l) {
        int is = l / 16;
        int8_t q1 = (int8_t)(((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4))) - 32;
        int8_t q2 = (int8_t)(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4))) - 32;
        int8_t q3 = (int8_t)(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4))) - 32;
        int8_t q4 = (int8_t)(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4))) - 32;
        y[l] = d * sc[is] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
      }

      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
    data += 210;
  }
}

inline void dequant_q2K_row(const uint8_t* data, int cols, float* out) {
  for (int sb = 0; sb < cols; sb += 256) {
    const uint8_t* sc = data, *qs = data + 16;
    float d = fp16_to_fp32((uint16_t)(data[80] | (data[81] << 8)));
    float dm = fp16_to_fp32((uint16_t)(data[82] | (data[83] << 8)));
    int is = 0;
    for (int n = 0; n < 256; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        float ds = d * (sc[is] & 0xf), ms = dm * (sc[is++] >> 4);
        for (int i = 0; i < 16; ++i) out[sb + n + j * 32 + i] =
            ds * ((qs[i] >> shift) & 3) - ms;
        ds = d * (sc[is] & 0xf); ms = dm * (sc[is++] >> 4);
        for (int i = 0; i < 16; ++i) out[sb + n + j * 32 + 16 + i] =
            ds * ((qs[i + 16] >> shift) & 3) - ms;
        shift += 2;
      }
      qs += 32;
    }
    data += 84;
  }
}

inline void dequant_q3K_row(const uint8_t* data, int cols, float* out) {
  for (int sb = 0; sb < cols; sb += 256) {
    const uint8_t* h = data, *qs = data + 32;
    uint8_t raw[12]; std::memcpy(raw, data + 96, 12);
    uint32_t aux[4] = {};
    std::memcpy(aux, raw, 12);
    const uint32_t km1 = 0x03030303, km2 = 0x0f0f0f0f;
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & km2) | (((tmp >> 4) & km1) << 4);
    aux[3] = ((aux[1] >> 4) & km2) | (((tmp >> 6) & km1) << 4);
    aux[0] = (aux[0] & km2) | (((tmp >> 0) & km1) << 4);
    aux[1] = (aux[1] & km2) | (((tmp >> 2) & km1) << 4);
    const int8_t* sc = (const int8_t*)aux;
    float d = fp16_to_fp32((uint16_t)(data[108] | (data[109] << 8)));
    int is = 0; uint8_t mask = 1;
    for (int n = 0; n < 256; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        float ds = d * (sc[is++] - 32);
        for (int i = 0; i < 16; ++i) out[sb + n + j * 32 + i] =
            ds * (((qs[i] >> shift) & 3) - ((h[i] & mask) ? 0 : 4));
        ds = d * (sc[is++] - 32);
        for (int i = 0; i < 16; ++i) out[sb + n + j * 32 + 16 + i] =
            ds * (((qs[i + 16] >> shift) & 3) - ((h[i + 16] & mask) ? 0 : 4));
        shift += 2; mask <<= 1;
      }
      qs += 32;
    }
    data += 110;
  }
}

inline void dequant_q8K_row(const uint8_t* data, int cols, float* out) {
  for (int sb = 0; sb < cols; sb += 256) {
    float d;
    std::memcpy(&d, data, sizeof(d));
    const int8_t* qs = (const int8_t*)(data + 4);
    for (int i = 0; i < 256; ++i) out[sb + i] = d * qs[i];
    data += 292;
  }
}

// Streaming K-quant -> internal Q4 repack: dequant one row (fp32 temp),
// requant per-32-minmax. Double-quant noise is inherent (gate judges); no
// full-matrix fp32 transient.
inline size_t kquant_block_bytes(uint32_t dtype) {
  switch (dtype) {
    case gguf::Q4_K: return 144;
    case gguf::Q5_K: return 176;
    case gguf::Q6_K: return 210;
    case gguf::Q2_K: return 84;
    case gguf::Q3_K: return 110;
    case gguf::Q8_K: return 292;
    default: throw std::runtime_error("repack: not a supported K-quant");
  }
}

inline void dequant_k_row(uint32_t dtype, const uint8_t* blk, int cols, float* out) {
  const uint8_t* d = blk;
  for (int sb = 0; sb < cols; sb += 256) {
    if (dtype == gguf::Q2_K)
      dequant_q2K_row(d, 256, out + sb);
    else if (dtype == gguf::Q3_K)
      dequant_q3K_row(d, 256, out + sb);
    else if (dtype == gguf::Q4_K)
      dequant_q4K_row(d, 256, out + sb);
    else if (dtype == gguf::Q5_K)
      dequant_q5K_row(d, 256, out + sb);
    else if (dtype == gguf::Q8_K)
      dequant_q8K_row(d, 256, out + sb);
    else
      dequant_q6K_row(d, 256, out + sb);
    d += kquant_block_bytes(dtype);
  }
}

inline Q4Matrix repack_kquant(const uint8_t* data, uint32_t dtype, int rows, int cols) {
  if (cols % 256 != 0) throw std::runtime_error("repack_kquant: cols % 256 != 0");
  Q4Matrix m;
  m.rows = rows;
  m.cols = cols;
  m.ngroups = cols / Q4Matrix::GS;
  m.scales.resize((size_t)rows * m.ngroups);
  m.zeros.resize((size_t)rows * m.ngroups);
  m.data.resize((size_t)rows * cols / 2);
  m.wsums.assign((size_t)rows * m.ngroups, 0);
  size_t stride = (size_t)(cols / 256) * kquant_block_bytes(dtype);
  std::vector<float> tmp(cols);
  for (int r = 0; r < rows; ++r) {
    dequant_k_row(dtype, data + (size_t)r * stride, cols, tmp.data());
    quantize_q4_row(tmp.data(), cols, m.ngroups, m.scales.data() + (size_t)r * m.ngroups,
                    m.zeros.data() + (size_t)r * m.ngroups,
                    m.data.data() + (size_t)r * cols / 2,
                    m.wsums.data() + (size_t)r * m.ngroups);
  }
  return m;
}

inline Q4Matrix repack_q4_0(const uint8_t* data, int rows, int cols) {
  if (cols % 32 != 0) throw std::runtime_error("repack_q4_0: cols % 32 != 0");
  Q4Matrix m;
  m.rows = rows;
  m.cols = cols;
  m.ngroups = cols / Q4Matrix::GS;
  m.scales.resize((size_t)rows * m.ngroups);
  m.zeros.assign((size_t)rows * m.ngroups, 8);
  m.data.resize((size_t)rows * cols / 2);
  m.wsums.assign((size_t)rows * m.ngroups, 0);
  // GGUF Q4_0 block: fp16 d (2B) + 16B qs. Nibble layout is SHUFFLED for the
  // SIMD dot (see ggml quantize_row_q4_0_ref / vec_dot_q4_0_q8_0): byte k
  // holds elements k (low) and k+16 (high), NOT 2k/2k+1. value=(nib-8)*d.
  // Repack converts to our sequential low-first layout (out[2k], out[2k+1]
  // hold elements 2k, 2k+1). Repack is offline; clarity wins.
  for (int r = 0; r < rows; ++r) {
    const uint8_t* row = data + (size_t)r * (cols / 32) * 18;
    for (int g = 0; g < m.ngroups; ++g) {
      const uint8_t* blk = row + (size_t)g * 18;
      uint16_t dh;
      std::memcpy(&dh, blk, 2);
      float d = fp16_to_fp32(dh);
      m.scales[(size_t)r * m.ngroups + g] = d;
      int ev[32];  // dequantized element values in order
      for (int k = 0; k < 16; ++k) {
        uint8_t b = blk[2 + k];
        ev[k] = (b & 0xF) - 8;
        ev[k + 16] = ((b >> 4) & 0xF) - 8;
      }
      int wsum = 0;
      for (int k = 0; k < 16; ++k) {
        int v0 = ev[2 * k], v1 = ev[2 * k + 1];
        m.data[(size_t)r * cols / 2 + g * 16 + k] =
            (uint8_t)((v0 + 8) | ((v1 + 8) << 4));
        wsum += v0 + v1;
      }
      m.wsums[(size_t)r * m.ngroups + g] = (int16_t)wsum;
    }
  }
  return m;
}

// Dequantize one Q4_0 row to float (embedding gather path). Same shuffled
// nibble layout as above: byte k -> elems k (low) and k+16 (high).
inline void dequant_q4_0_row(const uint8_t* data, int cols, float* out) {
  for (int g = 0; g < cols / 32; ++g) {
    const uint8_t* blk = data + (size_t)g * 18;
    uint16_t dh;
    std::memcpy(&dh, blk, 2);
    float d = fp16_to_fp32(dh);
    for (int k = 0; k < 16; ++k) {
      uint8_t b = blk[2 + k];
      out[g * 32 + k] = ((b & 0xF) - 8) * d;
      out[g * 32 + 16 + k] = (((b >> 4) & 0xF) - 8) * d;
    }
  }
}

// Dequantize one Q8_0 row to float. Q8_0 block: fp16 d + 32xi8.
inline void dequant_q8_0_row(const uint8_t* data, int cols, float* out) {
  for (int g = 0; g < cols / 32; ++g) {
    const uint8_t* blk = data + (size_t)g * 34;
    uint16_t dh;
    std::memcpy(&dh, blk, 2);
    float d = fp16_to_fp32(dh);
    for (int k = 0; k < 32; ++k) out[g * 32 + k] = (int8_t)blk[2 + k] * d;
  }
}

inline Q8Matrix repack_q8_0(const uint8_t* data, int rows, int cols) {
  std::vector<float> tmp((size_t)rows * cols);
  size_t stride = (size_t)(cols / 32) * 34;
  for (int r = 0; r < rows; ++r) dequant_q8_0_row(data + (size_t)r * stride, cols, tmp.data() + (size_t)r * cols);
  return quantize_q8(tmp.data(), rows, cols);
}

inline void copy_f32(const uint8_t* data, int n, float* out) {
  std::memcpy(out, data, (size_t)n * 4);
}
inline void widen_bf16(const uint8_t* data, int n, float* out) {
  for (int i = 0; i < n; ++i) {
    uint16_t b;
    std::memcpy(&b, data + (size_t)i * 2, 2);
    out[i] = bf16_to_fp32(b);
  }
}

}  // namespace molot::repack
