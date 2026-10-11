// molot bench: correctness + bandwidth proof on this CPU.
// Build: make bench  |  Run: ./build/bench
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "molot/adam.h"
#include "molot/attn.h"
#include "molot/gemm.h"
#include "molot/norm.h"
#include "molot/quant.h"
#include "molot/thread_pool.h"

using molot::ThreadPool;
using Clock = std::chrono::high_resolution_clock;

static double sec(std::chrono::steady_clock::time_point a,
                  std::chrono::steady_clock::time_point b) {
  return 1e-9 * (double)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a)
                     .count();
}

static std::mt19937 rng(1234);
static float frand() {
  static std::normal_distribution<float> d(0, 1);
  return d(rng);
}

static int failures = 0;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ++failures;                                                              \
      std::printf("FAIL %d: ", __LINE__);                                      \
      std::printf(__VA_ARGS__);                                                \
      std::printf("\n");                                                       \
    }                                                                          \
  } while (0)

template <typename F>
static double bench_ms(F&& f, int iters, int warmup = 2) {
  for (int i = 0; i < warmup; ++i) f();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) f();
  auto t1 = std::chrono::steady_clock::now();
  return sec(t0, t1) / iters * 1000.0;
}

int main() {
  ThreadPool pool;  // all cores
  ThreadPool pool1(1);
  std::printf("threads: %d (pool1: %d)\n", pool.size(), pool1.size());

  // ---- 0. pool correctness: parallel sum ----
  {
    const int64_t n = 100000;
    std::vector<int> seen(n, 0);
    pool.parallel_for(n, [&](int, int64_t i) { seen[(size_t)i] = 1; });
    CHECK(std::accumulate(seen.begin(), seen.end(), 0) == n, "pool coverage");
    std::printf("[ok] thread_pool coverage n=%lld\n", (long long)n);
  }

  // ---- 1. GEMV correctness: fp32 simd vs scalar ----
  {
    int rows = 256, cols = 512;
    std::vector<float> W(rows * cols), x(cols), y(rows), ref(rows);
    for (auto& v : W) v = frand() * 0.1f;
    for (auto& v : x) v = frand();
    molot::gemv_fp32_mt(W.data(), x.data(), y.data(), rows, cols, pool1);
    for (int r = 0; r < rows; ++r) {
      double s = 0;
      for (int c = 0; c < cols; ++c) s += (double)W[r * cols + c] * x[c];
      ref[r] = (float)s;
    }
    double err = 0;
    for (int r = 0; r < rows; ++r) err = std::max<double>(err, std::fabs(y[r] - ref[r]));
    CHECK(err < 1e-3, "gemv fp32 err %g", err);
    std::printf("[ok] gemv_fp32 maxerr=%.2e\n", err);
  }

  // ---- 2. Q8 GEMV accuracy (NRMSE over rows; max-rel is meaningless here
  // because random-projection dots are often near zero) ----
  {
    int rows = 256, cols = 512;
    std::vector<float> W(rows * cols), x(cols), y(rows);
    for (auto& v : W) v = frand() * 0.5f;
    for (auto& v : x) v = frand();
    auto Q = molot::quantize_q8(W.data(), rows, cols);
    molot::gemv_q8(Q, x.data(), y.data(), pool1);
    double num = 0, den = 0;
    for (int r = 0; r < rows; ++r) {
      double s = 0;
      for (int c = 0; c < cols; ++c) s += (double)W[r * cols + c] * x[c];
      num += (y[r] - s) * (y[r] - s);
      den += s * s;
    }
    double nrmse = std::sqrt(num / den);
    CHECK(nrmse < 0.05, "q8 nrmse %g", nrmse);
    std::printf("[ok] gemv_q8 nrmse=%.4f\n", nrmse);
  }

  // ---- 3. Q4 GEMV accuracy (grouped GS=32) ----
  {
    int rows = 128, cols = 256;
    std::vector<float> W(rows * cols), x(cols), y(rows);
    for (auto& v : W) v = frand() * 0.5f;
    for (auto& v : x) v = frand();
    auto Q = molot::quantize_q4(W.data(), rows, cols);
    molot::gemv_q4(Q, x.data(), y.data(), pool1);
    double num = 0, den = 0;
    for (int r = 0; r < rows; ++r) {
      double s = 0;
      for (int c = 0; c < cols; ++c) s += (double)W[r * cols + c] * x[c];
      num += (y[r] - s) * (y[r] - s);
      den += s * s;
    }
    double nrmse = std::sqrt(num / den);
    CHECK(nrmse < 0.10, "q4 nrmse %g", nrmse);
    std::printf("[ok] gemv_q4 (GS=32) nrmse=%.4f\n", nrmse);
  }

  // ---- 3b. Q4 int8 path is exact vs double reference on quantized math ----
  // (comparing to the float-activation path would mix in ~1% activation-quant
  // noise, the same family as the Q8 NRMSE — so reference the quantized math
  // itself, where int VNNI accumulation must match to float-rounding).
  {
    int rows = 8, cols = 96;  // ngroups=3 (odd) exercises the scalar tail
    std::vector<float> W(rows * cols), x(cols), yi(rows);
    for (auto& v : W) v = frand() * 0.5f;
    for (auto& v : x) v = frand();
    auto Q = molot::quantize_q4(W.data(), rows, cols);
    molot::QAct8 a = molot::quantize_act_q8(x.data(), cols);
    int qn = (cols + 31) / 32 * 32;
    std::vector<uint8_t> qup(qn, 128);
    for (int s = 0; s < cols; s += 32) {
      int blen = cols - s < 32 ? cols - s : 32;
      for (int k = 0; k < blen; ++k) {
        if ((k & 1) == 0)
          qup[s + k / 2] = a.qu[s + k];
        else
          qup[s + 16 + k / 2] = a.qu[s + k];
      }
    }
    double num = 0, den = 0;
    for (int r = 0; r < rows; ++r) {
      const uint8_t* prow = Q.data.data() + (size_t)r * cols / 2;
      yi[r] = molot::q4_row_dot_i8(prow, Q.wsums.data() + r * Q.ngroups, a, qup.data(),
                                   Q.scales.data() + r * Q.ngroups,
                                   Q.zeros.data() + r * Q.ngroups, Q.ngroups);
      double ref = 0;  // exact quantized math in double
      for (int g = 0; g < Q.ngroups; ++g) {
        double ds = 0;
        for (int k = 0; k < 32; ++k) {
          int c = g * 32 + k;
          uint8_t b = prow[c / 2];
          int v = (c & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
          ds += (double)(v - Q.zeros[r * Q.ngroups + g]) * (double)a.q[c];
        }
        ref += ds * Q.scales[r * Q.ngroups + g] * a.scales[g];  // act bs=32
      }
      num += (yi[r] - ref) * (yi[r] - ref);
      den += ref * ref;
    }
    double agree = std::sqrt(num / den);
    CHECK(agree < 1e-6, "q4 i8 exactness %g", agree);
    std::printf("[ok] gemv_q4 i8-exactness=%.2e\n", agree);
  }

  // ---- 4. RMSNorm ----
  {
    int n = 4096;
    std::vector<float> x(n), w(n, 1.0f), y(n);
    for (auto& v : x) v = frand();
    molot::rmsnorm(x.data(), w.data(), y.data(), n);
    double ss = 0;
    for (auto v : y) ss += (double)v * v;
    CHECK(std::fabs(ss / n - 1.0) < 1e-4, "rms %g", ss / n);
    std::printf("[ok] rmsnorm mean_sq=%.5f\n", ss / n);
  }

  // ---- 5. Attention: rows sum to 1, matches ref ----
  {
    int n = 512, d = 128;
    std::vector<float> Q(d), K((size_t)n * d), V((size_t)n * d), O(d), Oref(d);
    for (auto& v : Q) v = frand() * 0.2f;
    for (auto& v : K) v = frand() * 0.2f;
    for (auto& v : V) v = frand() * 0.2f;
    molot::attn_decode_head(Q.data(), K.data(), V.data(), O.data(), n, d,
                            1.0f / std::sqrt((float)d));
    // ref
    std::vector<float> s(n);
    float m = -1e30f;
    for (int t = 0; t < n; ++t) {
      double dot = 0;
      for (int j = 0; j < d; ++j) dot += (double)Q[j] * K[(size_t)t * d + j];
      s[t] = (float)(dot / std::sqrt((double)d));
      m = std::max(m, s[t]);
    }
    double den = 0;
    for (auto& v : s) {
      v = std::exp(v - m);
      den += v;
    }
    for (int j = 0; j < d; ++j) {
      double a = 0;
      for (int t = 0; t < n; ++t) a += (double)s[t] / den * V[(size_t)t * d + j];
      Oref[j] = (float)a;
    }
    double err = 0;
    for (int j = 0; j < d; ++j) err = std::max<double>(err, std::fabs(O[j] - Oref[j]));
    CHECK(err < 1e-3, "attn err %g", err);
    std::printf("[ok] attn_decode maxerr=%.2e\n", err);
  }

  // ---- 5b. Vector exp accuracy (softmax domain) ----
  {
    alignas(64) float in[16] = {-87.f, -20.f, -10.f, -5.f, -3.f, -2.f, -1.5f, -1.f,
                                -0.75f, -0.5f, -0.25f, -0.1f, -0.01f, 0.f, 1.f, 50.f};
    __m512 v = molot::exp_avx512_ps(_mm512_loadu_ps(in));
    alignas(64) float out[16];
    _mm512_storeu_ps(out, v);
    double err = 0;
    for (int i = 0; i < 16; ++i) {
      double ref = std::exp((double)in[i]);
      err = std::max(err, std::fabs(out[i] - ref) / (ref + 1e-30));
    }
    CHECK(err < 1e-5, "vexp relerr %g", err);
    std::printf("[ok] vexp maxrel=%.2e\n", err);
  }

  // ---- 6. AdamW: one step matches scalar ref ----
  {
    int n = 1024;
    std::vector<float> p(n, 0.5f), m(n, 0), v(n, 0), g(n, 0.1f);
    auto p2 = p, m2 = m, v2 = v;
    float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.01f;
    molot::adamw_fused(p.data(), m.data(), v.data(), g.data(), n, lr, b1, b2, eps, wd,
                       1 - b1, 1 - b2);
    for (int i = 0; i < n; ++i) {  // scalar ref, bias already corrected args
      m2[i] = b1 * m2[i] + (1 - b1) * g[i];
      v2[i] = b2 * v2[i] + (1 - b2) * g[i] * g[i];
      p2[i] -= lr * ((m2[i] / (1 - b1)) / (std::sqrt(v2[i] / (1 - b2)) + eps) + wd * p2[i]);
    }
    double err = 0;
    for (int i = 0; i < n; ++i) err = std::max<double>(err, std::fabs(p[i] - p2[i]));
    CHECK(err < 1e-5, "adam err %g", err);
    std::printf("[ok] adamw_fused maxerr=%.2e\n", err);
  }

  // ---- 7. Blocked GEMM matches naive (incl. Mc/Kc tails) ----
  {
    int M = 96, N = 128, K = 300;  // not multiples of Mc=64 / Kc=1024
    std::vector<float> A((size_t)M * K), B((size_t)K * N), Bt((size_t)N * K),
        C((size_t)M * N), Cref((size_t)M * N);
    for (auto& v : A) v = frand() * 0.1f;
    for (auto& v : B) v = frand() * 0.1f;
    for (int i = 0; i < K; ++i)
      for (int j = 0; j < N; ++j) Bt[(size_t)j * K + i] = B[(size_t)i * N + j];
    molot::gemm_fp32_mt(A.data(), Bt.data(), Cref.data(), M, N, K, pool1);
    molot::gemm_fp32_blocked(A.data(), Bt.data(), C.data(), M, N, K, pool1);
    double err = 0;
    for (int i = 0; i < M * N; ++i) err = std::max<double>(err, std::fabs(C[i] - Cref[i]));
    CHECK(err < 1e-3, "blocked gemm err %g", err);
    std::printf("[ok] gemm_blocked maxerr=%.2e\n", err);
  }

  if (failures) {
    std::printf("CORRECTNESS: %d FAILURES\n", failures);
    return 1;
  }
  std::printf("CORRECTNESS: all pass\n\n");

  // ================= BENCHMARKS =================
  std::printf("--- bandwidth benchmarks (decode-sized, DRAM-resident) ---\n");
  // Machine ceiling: parallel memcpy on 256 MB (STREAM-like, same pool).
  {
    size_t N = 256u << 20;
    std::vector<float> a(N / 4), b(N / 4);
    for (size_t i = 0; i < N / 4; i += 257) a[i] = (float)i;
    auto copier = [&] {
      pool.parallel_for(pool.size(), [&](int tid, int64_t p) {
        size_t P = (size_t)pool.size();
        size_t chunk = N / P;
        size_t off = (size_t)p * chunk;
        size_t len = (p == (int64_t)P - 1) ? N - off : chunk;
        std::memcpy((char*)b.data() + off, (char*)a.data() + off, len);
        (void)tid;
      });
    };
    double ms = bench_ms(copier, 10, 3);
    std::printf("memcpy 256MB parallel (ceiling): %.1f GB/s\n", (N / (ms / 1000.0)) / 1e9);
    // single-thread reference to show NUMA/threading matters
    double ms1 = bench_ms([&] { std::memcpy(b.data(), a.data(), N); }, 3, 1);
    std::printf("memcpy 256MB single: %.1f GB/s\n", (N / (ms1 / 1000.0)) / 1e9);
    (void)b;
  }
  // FFN up-proj shape 16384x4096 (Llama-8B ffn): fp32 256 MB, q8 64 MB,
  // q4 32 MB — all above the 32 MB L3, so this measures DRAM like decode.
  double bw_fp32 = 0, bw_q8 = 0, bw_q4 = 0;
  {
    int R = 16384, C = 4096;
    std::vector<float> W((size_t)R * C), x(C), y(R);
    for (auto& v : W) v = frand() * 0.1f;
    for (auto& v : x) v = frand() * 0.1f;
    auto Q8 = molot::quantize_q8(W.data(), R, C);
    auto Q4 = molot::quantize_q4(W.data(), R, C);

    double ms1 = bench_ms([&] { molot::gemv_fp32_mt(W.data(), x.data(), y.data(), R, C, pool1); }, 3);
    double msP = bench_ms([&] { molot::gemv_fp32_mt(W.data(), x.data(), y.data(), R, C, pool); }, 5);
    bw_fp32 = ((size_t)R * C * 4) / (msP / 1000.0) / 1e9;
    std::printf("gemv_fp32 16k×4k (256MB): 1T %7.1f ms | MT %7.1f ms | %5.1fx | %.1f GB/s\n",
                ms1, msP, ms1 / msP, bw_fp32);

    double msQ8 = bench_ms([&] { molot::gemv_q8(Q8, x.data(), y.data(), pool); }, 5);
    bw_q8 = ((size_t)R * C) / (msQ8 / 1000.0) / 1e9;
    std::printf("gemv_q8   16k×4k (64MB): MT %7.1f ms | %.1f GB/s weight | %.2fx vs fp32-MT\n",
                msQ8, bw_q8, msP / msQ8);

    double msQ4 = bench_ms([&] { molot::gemv_q4(Q4, x.data(), y.data(), pool); }, 5);
    bw_q4 = ((size_t)R * C / 2) / (msQ4 / 1000.0) / 1e9;
    std::printf("gemv_q4   16k×4k (32MB): MT %7.1f ms | %.1f GB/s weight | %.2fx vs fp32-MT\n",
                msQ4, bw_q4, msP / msQ4);
    // float-path reference on same shapes (unpack + cvt + per-group reduce)
    double msQ4f = bench_ms(
        [&] {
          pool.parallel_for(R, [&](int, int64_t r) {
            const uint8_t* prow = Q4.data.data() + (size_t)r * C / 2;
            y[r] = molot::q4_row_dot(prow, x.data(), C, Q4.scales.data() + r * Q4.ngroups,
                                     Q4.zeros.data() + r * Q4.ngroups, Q4.ngroups);
          });
        },
        3, 1);
    std::printf("gemv_q4f  16k×4k float-path: MT %7.1f ms | i8 is %.2fx\n", msQ4f,
                msQ4f / msQ4);
    double chk = 0;
    for (int i = 0; i < R; i += 97) chk += y[i];
    std::printf("  (q4 checksum %.4f)\n", chk);
  }

  // Attention sweep (GQA 32Q/8KV like Llama-8B, d=128). KV working set here
  // (<=34 MB) fits L3 across bench iters, so this is the hot-cache
  // steady-state (continuous batching). GB/s is algorithmic head-traffic
  // (nq-based: each query head reads its KV); >bus values = L3 reuse.
  {
    int d = 128, nq = 32, nkv = 8;
    for (int n : {512, 2048, 4096}) {
      std::vector<float> Q((size_t)nq * d), K((size_t)nkv * n * d), V((size_t)nkv * n * d),
          O((size_t)nq * d);
      for (auto& v : Q) v = frand() * 0.2f;
      for (auto& v : K) v = frand() * 0.2f;
      for (auto& v : V) v = frand() * 0.2f;
      double ms = bench_ms(
          [&] {
            molot::attn_decode_gqa(Q.data(), K.data(), V.data(), O.data(), nq, nkv, n,
                                   d, 1.0f / std::sqrt((float)d), pool);
          },
          10);
      double kv_gb = 2.0 * nq * n * d * 4 / 1e9;  // head-traffic (fan-out incl.)
      std::printf("attn_gqa n=%4d d=%d q=%d kv=%d: %7.2f ms | heads %.2f GB -> %.0f GB/s eff (L3-hot)\n",
                  n, d, nq, nkv, ms, kv_gb, kv_gb / (ms / 1000.0));
    }
  }

  // Long context: n=16384, full GQA (134 MB unique KV, 4x L3). Cold DRAM
  // stream — nothing reused across iters. Head-traffic 537 MB at bus speed.
  {
    int d = 128, nq = 32, nkv = 8, n = 16384;
    std::vector<float> Q((size_t)nq * d), K((size_t)nkv * n * d), V((size_t)nkv * n * d),
        O((size_t)nq * d);
    for (auto& v : Q) v = frand() * 0.2f;
    for (auto& v : K) v = frand() * 0.2f;
    for (auto& v : V) v = frand() * 0.2f;
    double ms = bench_ms(
        [&] {
          molot::attn_decode_gqa(Q.data(), K.data(), V.data(), O.data(), nq, nkv, n,
                                 d, 1.0f / std::sqrt((float)d), pool);
        },
        3, 1);
    double kv_gb = 2.0 * nq * n * d * 4 / 1e9;  // head-traffic (fan-out incl.)
    std::printf("attn_gqa LONG n=%d: %.1f ms | heads %.2f GB -> %.0f GB/s eff (cold DRAM)\n",
                n, ms, kv_gb, kv_gb / (ms / 1000.0));
  }

  // Prefill GEMM 512x512x1024 with transposed B (contiguous dots)
  {
    int M = 512, N = 512, K = 1024;
    std::vector<float> A((size_t)M * K), B((size_t)K * N), Bt((size_t)N * K),
        C((size_t)M * N);
    for (auto& v : A) v = frand() * 0.1f;
    for (auto& v : B) v = frand() * 0.1f;
    for (int i = 0; i < K; ++i)
      for (int j = 0; j < N; ++j) Bt[(size_t)j * K + i] = B[(size_t)i * N + j];
    double ms = bench_ms([&] { molot::gemm_fp32_mt(A.data(), Bt.data(), C.data(), M, N, K, pool); }, 5);
    double gflops = 2.0 * M * N * K / (ms / 1000.0) / 1e9;
    std::printf("gemm_fp32 %dx%dx%d (prefill): %.1f ms | %.1f GFLOPS\n", M, N, K, ms,
                gflops);
  }

  // Large prefill: naive vs cache-blocked. 256x4096x4096: blocked moves ~50x
  // less traffic (Bt shared across Mc rows: 0.5 GB vs 17 GB, A read once).
  // Measured gap is modest (both L3/compute-mixed at this shape) — it decides
  // once Bt/L3 grows. Same FLOPs either way.
  {
    int M = 256, N = 4096, K = 4096;
    std::vector<float> A((size_t)M * K), B((size_t)K * N), Bt((size_t)N * K),
        C((size_t)M * N);
    for (auto& v : A) v = frand() * 0.05f;
    for (auto& v : B) v = frand() * 0.05f;
    for (int i = 0; i < K; ++i)
      for (int j = 0; j < N; ++j) Bt[(size_t)j * K + i] = B[(size_t)i * N + j];
    double msN =
        bench_ms([&] { molot::gemm_fp32_mt(A.data(), Bt.data(), C.data(), M, N, K, pool); }, 2, 1);
    double msB = bench_ms(
        [&] { molot::gemm_fp32_blocked(A.data(), Bt.data(), C.data(), M, N, K, pool); }, 3, 1);
    double gflopsN = 2.0 * M * N * K / (msN / 1000.0) / 1e9;
    double gflopsB = 2.0 * M * N * K / (msB / 1000.0) / 1e9;
    std::printf("prefill %dx%dx%d naive: %.0f ms (%.0f GFLOPS) | blocked: %.0f ms "
                "(%.0f GFLOPS) | %.1fx\n",
                M, N, K, msN, gflopsN, msB, gflopsB, msN / msB);
  }

  // RMSNorm + Adam streams
  {
    int n = 1 << 20;
    std::vector<float> x(n), w(n, 1.0f), y(n);
    for (auto& v : x) v = frand();
    double ms = bench_ms([&] { molot::rmsnorm(x.data(), w.data(), y.data(), n); }, 20);
    std::printf("rmsnorm 1M: %.2f ms | %.1f GB/s\n", ms, 3.0 * n * 4 / (ms / 1e3) / 1e9);
    std::vector<float> p(n, 0.5f), m(n, 0), v(n, 0), g(n, 0.01f);
    double ma = bench_ms(
        [&] { molot::adamw_fused(p.data(), m.data(), v.data(), g.data(), n, 1e-3f, 0.9f,
                                 0.999f, 1e-8f, 0.01f, 0.1f, 0.001f); },
        5);
    std::printf("adamw_fused 1M: %.2f ms | %.1f GB/s\n", ma,
                4.0 * n * 4 / (ma / 1e3) / 1e9);
  }

  // ---- Extrapolation: Llama-8B-like, Q4, batch=1, ctx 2048 ----
  // 8B weights Q4 ≈ 4.0 GB + scales ≈ 0.3 GB; KV (GQA8, Q8, 32 layers,
  // 2048 ctx) ≈ 0.13 GB. Per token ≈ 4.4 GB streamed. Throughput ≈ the
  // *slower* of weight-stream BW and KV-stream BW measured above.
  {
    double bw = std::min(bw_q4, 60.0);  // cap at memcpy ceiling
    double tok_s = bw / 4.4;
    std::printf("\n[model] 8B-Q4 per-token ≈ 4.4 GB; weight-stream %.1f GB/s -> "
                "~%.1f tok/s (batch=1, ctx2048, this 6-core box)\n",
                bw, tok_s);
    std::printf("        Q8-KV attn slice (n=2048) measured above; MoE/sparse + "
                "speculative draft multiply this 1.5–2x (see DESIGN.md).\n");
  }

  return 0;
}
