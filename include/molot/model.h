#pragma once
// Arch-generic model load: GGUF metadata drives everything (no per-arch code).
// Optional tensors are tolerated (missing = feature absent). Verified against
// a reference implementation on gemma4 (dense, 42 blocks, shared KV tail,
// tied output, per-layer input embeddings, QK-norms, proportional rope).
// Shared-KV rule (matches reference): tail layers share first_nokv-2 (SWA)
// or first_nokv-1 (global).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "molot/gguf.h"
#include "molot/pack.h"
#include "molot/quant.h"
#include "molot/repack.h"

namespace molot::model {

struct HParams {
  std::string arch;
  int n_layer = 0, n_embd = 0, n_head = 0, n_kv = 0, n_ffn = 0;
  float eps = 1e-6f;
  float rope_base = 10000, rope_swa_base = 10000;
  int rope_dim = 0, rope_swa_dim = 0;
  float logit_cap = 0, logit_scale = 1.0f;
  int sw_window = 0;
  int vocab = 0, bos = 1, eos = 2;
  std::vector<char> is_swa;  // per layer (from sliding_window_pattern if present)
  // arch behavior flags (set by arch defaults in load; documented per arch)
  float post_eps = 0;     // 0 = use eps; glimmer uses 1e-8 for post norms
  int ffn_silu = 0;       // 0 = GELU (gemma4), 1 = SiLU (glimmer)
  int v_norm = 1;         // plain-RMS V before cache (gemma4); glimmer skips
  int emb_plain_norm = 0;  // 1 = RMSNorm(emb) init instead of sqrt scale
  int emb_scale = 1;       // legacy sqrt(E) embedding scale
  int qwen35 = 0;
  int rope_swa_only = 0;   // 1 = rope only on SWA layers (glimmer NoPE global)
  int attn_scale_one = 1;  // 1 = scale 1.0 (gemma4 QK-norms); 0 = 1/sqrt(qhd)
  int expert_count = 0, expert_used_count = 0;
};

struct Layer {
  bool has_kv = false;
  bool swa = false;
  int kv_src = -1;  // shared-KV source layer (A1)
  int qhd = 0, khd = 0;
  int rope_dim = 0;
  float rope_base = 10000;
  int window = 0;  // 0 = full prefix
  std::vector<float> freq;  // baked rope inv_freq (hd/2); global entries carry
                            // the file's proportional factors (angle=pos*inv/ff)
  // weights (repacked)
  Q4Matrix q, k, v, o, gate, up, down;
  std::vector<float> attn_norm, q_norm, k_norm, ffn_norm;
  // gemma4 extras (empty when absent)
  std::vector<float> post_attn_norm, post_ffw_norm, post_norm;
  Q4Matrix inp_gate, proj;
  // attention Q-gate (muse-glimmer attn_gate: same rows as q; sigmoid-gated
  // attention output before O). Distinct from gemma4's pe-path inp_gate.
  Q4Matrix attn_gate;
  Q4Matrix moe_gate_inp;
  std::vector<Q4Matrix> moe_gate_exps, moe_up_exps, moe_down_exps;
  int expert_count = 0, expert_used_count = 0;
  int is_ssm = 0;
  Q4Matrix ssm_qkv, ssm_z, ssm_out;
  std::vector<float> ssm_a, ssm_dt, ssm_alpha, ssm_beta, ssm_conv, ssm_norm;
  std::vector<float> moe_shared_router;
  Q4Matrix moe_shared_gate, moe_shared_up, moe_shared_down;
  int ssm_key_heads = 0, ssm_value_heads = 0, ssm_key_dim = 0, ssm_value_dim = 0;
  float out_scale = 1.0f;
};

struct Model {
  HParams hp;
  std::vector<Layer> layers;
  // globals
  std::vector<float> emb;  // F32 [vocab, embd] dequantized (gather path; F32 sources)
  std::vector<uint8_t> emb_raw;  // raw blocks when quantized (see emb_kind)
  int emb_kind = 0;              // 0 = use emb F32; else ggml dtype of emb_raw
  int emb_cols = 0;
  Q8Matrix head;           // tied or separate output projection
  // exact-mode sources (avoid double-quant noise in MOLOT_FP32=1 runs)
  std::vector<uint8_t> head_q80_raw;  // Q8_0 blocks when tied [vocab, embd]
  int head_q80_cols = 0;
  std::vector<float> plan_fp32;  // widened per_layer_model_proj [D, E]... stored [E][D]
  std::vector<float> out_norm;
  // per-layer input embeddings (gemma4)
  bool has_layer_emb = false;
  std::vector<uint8_t> layer_emb_raw;  // Q4_0 blocks [n_layer*256, vocab]
  int layer_emb_dim = 0;               // 256
  std::vector<float> layer_proj_norm;  // F32 [256]
  Q8Matrix layer_plan;                 // per_layer_model_proj requantized Q8
  std::vector<float> rope_freqs;       // global inv_freq table (d/2)
};

inline std::vector<float> load_f32(const gguf::File& f, const char* name) {
  const auto* t = f.find_tensor(name);
  if (!t) return {};
  if (t->dtype != gguf::F32) throw std::runtime_error("model: not F32: " + t->name);
  std::vector<float> v(t->ne());
  repack::copy_f32(f.tensor_data(*t), (int)t->ne(), v.data());
  return v;
}

inline Q4Matrix load_q4_0(const gguf::File& f, const char* name, bool& present) {
  const auto* t = f.find_tensor(name);
  present = t != nullptr;
  if (!t) return {};
  if (t->dtype != gguf::Q4_0) throw std::runtime_error("model: not Q4_0: " + t->name);
  int rows, cols;
  repack::gguf_shape_2d(*t, rows, cols);
  return repack::repack_q4_0(f.tensor_data(*t), rows, cols);
}

// Any supported 4-bit-family source -> internal Q4. Q4_0 repacks verbatim;
// K-quants (Q4_K/Q5_K/Q6_K) dequantize per row and requant (double-quant
// noise inherent; the quality gate judges). Throws on anything else.
inline Q4Matrix load_any_q4(const gguf::File& f, const char* name, bool& present) {
  const auto* t = f.find_tensor(name);
  present = t != nullptr;
  if (!t) return {};
  int rows, cols;
  repack::gguf_shape_2d(*t, rows, cols);
  if (t->dtype == gguf::Q4_0) return repack::repack_q4_0(f.tensor_data(*t), rows, cols);
  if (t->dtype == gguf::Q8_0) {
    std::vector<float> tmp((size_t)rows * cols);
    size_t stride = (size_t)(cols / 32) * 34;
    for (int r = 0; r < rows; ++r)
      repack::dequant_q8_0_row(f.tensor_data(*t) + (size_t)r * stride, cols,
                               tmp.data() + (size_t)r * cols);
    return quantize_q4(tmp.data(), rows, cols);
  }
  if (t->dtype == gguf::Q2_K || t->dtype == gguf::Q3_K || t->dtype == gguf::Q4_K ||
      t->dtype == gguf::Q5_K || t->dtype == gguf::Q6_K || t->dtype == gguf::Q8_K)
    return repack::repack_kquant(f.tensor_data(*t), t->dtype, rows, cols);
  throw std::runtime_error("model: unsupported GEMM dtype for " + t->name);
}

inline Q4Matrix load_moe_expert(const gguf::File& f, const char* name, int expert,
                                int& experts, bool& present) {
  const auto* t = f.find_tensor(name);
  present = t != nullptr;
  if (!t) return {};
  if (t->dims.size() != 3) throw std::runtime_error("model: MoE tensor is not 3-D: " + t->name);
  int rows = (int)t->dims[1], cols = (int)t->dims[0];
  experts = (int)t->dims[2];
  if (expert >= experts) throw std::runtime_error("model: bad MoE expert index");
  size_t block = (size_t)cols / gguf::type_info(t->dtype).block_elems *
                 gguf::type_info(t->dtype).block_bytes;
  size_t stride = (size_t)rows * block;
  const uint8_t* p = f.tensor_data(*t) + (size_t)expert * stride;
  if (t->dtype == gguf::F32) {
    std::vector<float> tmp((size_t)rows * cols);
    repack::copy_f32(p, rows * cols, tmp.data());
    return quantize_q4(tmp.data(), rows, cols);
  }
  if (t->dtype == gguf::Q4_0) return repack::repack_q4_0(p, rows, cols);
  return repack::repack_kquant(p, t->dtype, rows, cols);
}

inline Q4Matrix load_router(const gguf::File& f, const char* name, bool& present) {
  const auto* t = f.find_tensor(name);
  present = t != nullptr;
  if (!t) return {};
  int rows, cols;
  repack::gguf_shape_2d(*t, rows, cols);
  if (t->dtype == gguf::F32) {
    std::vector<float> v((size_t)rows * cols);
    repack::copy_f32(f.tensor_data(*t), rows * cols, v.data());
    return quantize_q4(v.data(), rows, cols);
  }

  return load_any_q4(f, name, present);
}

inline Q4Matrix slice_rows(const Q4Matrix& src, int first, int count) {
  Q4Matrix d;
  d.rows = count; d.cols = src.cols; d.ngroups = src.ngroups;
  size_t rb = (size_t)src.cols / 2, gs = src.ngroups;
  d.data.assign(src.data.begin() + (size_t)first * rb,
                src.data.begin() + (size_t)(first + count) * rb);
  d.scales.assign(src.scales.begin() + (size_t)first * gs,
                  src.scales.begin() + (size_t)(first + count) * gs);
  d.zeros.assign(src.zeros.begin() + (size_t)first * gs,
                 src.zeros.begin() + (size_t)(first + count) * gs);
  d.wsums.assign(src.wsums.begin() + (size_t)first * gs,
                 src.wsums.begin() + (size_t)(first + count) * gs);
  return d;
}

// Fast load from a .molot pack (no repack compute; blobs memcpy into place).
// Tolerates absent optional tensors exactly like load().
inline Model load_pack(const char* path) {
  pack::Reader rd(path);
  Model m;
  HParams& hp = m.hp;
  auto& ph = rd.meta.hp;
  hp.arch = ph.arch;
  hp.n_layer = ph.n_layer;
  hp.n_embd = ph.n_embd;
  hp.n_head = ph.n_head;
  hp.n_kv = ph.n_kv;
  hp.n_ffn = ph.n_ffn;
  hp.eps = ph.eps;
  hp.rope_base = ph.rope_base;
  hp.rope_swa_base = ph.rope_swa_base;
  hp.logit_cap = ph.logit_cap;
  hp.rope_dim = ph.rope_dim;
  hp.rope_swa_dim = ph.rope_swa_dim;
  hp.sw_window = ph.sw_window;
  hp.vocab = ph.vocab;
  hp.bos = ph.bos;
  hp.eos = ph.eos;
  auto get_q4 = [&](const std::string& name) {
    Q4Matrix q;
    const auto* t = rd.find(name);
    if (!t) return q;
    if (t->dtype != pack::Q4PACK) throw std::runtime_error("pack: not Q4PACK: " + name);
    q.rows = (int)t->rows;
    q.cols = (int)t->cols;
    q.ngroups = q.cols / Q4Matrix::GS;
    const uint8_t* d = rd.blob(*t);
    size_t nd = (size_t)q.rows * q.cols / 2, ns = (size_t)q.rows * q.ngroups;
    q.data.assign(d, d + nd);
    d += nd;
    q.scales.resize(ns);
    std::memcpy(q.scales.data(), d, ns * 4);
    d += ns * 4;
    q.zeros.assign(d, d + ns);
    d += ns;
    q.wsums.resize(ns);
    std::memcpy(q.wsums.data(), d, ns * 2);
    return q;
  };
  auto get_f32 = [&](const std::string& name) {
    std::vector<float> v;
    const auto* t = rd.find(name);
    if (!t) return v;
    if (t->dtype != pack::F32) throw std::runtime_error("pack: not F32: " + name);
    v.resize(t->nelems);
    std::memcpy(v.data(), rd.blob(*t), t->nbytes);
    return v;
  };
  m.layers.resize(hp.n_layer);
  for (int b = 0; b < hp.n_layer; ++b) {
    Layer& L = m.layers[b];
    auto& lr = rd.meta.layers[b];
    L.has_kv = lr.has_kv != 0;
    L.swa = lr.swa != 0;
    L.kv_src = lr.kv_src;
    L.qhd = lr.qhd;
    L.khd = lr.khd;
    L.window = lr.window;
    L.out_scale = lr.out_scale;
    std::string P = "blk." + std::to_string(b) + ".";
    L.q = get_q4(P + "attn_q.weight");
    L.k = get_q4(P + "attn_k.weight");
    L.v = get_q4(P + "attn_v.weight");
    L.o = get_q4(P + "attn_output.weight");
    L.gate = get_q4(P + "ffn_gate.weight");
    L.up = get_q4(P + "ffn_up.weight");
    L.down = get_q4(P + "ffn_down.weight");
    L.inp_gate = get_q4(P + "inp_gate.weight");
    L.proj = get_q4(P + "proj.weight");
    L.attn_norm = get_f32(P + "attn_norm.weight");
    L.q_norm = get_f32(P + "attn_q_norm.weight");
    L.k_norm = get_f32(P + "attn_k_norm.weight");
    L.ffn_norm = get_f32(P + "ffn_norm.weight");
    L.post_attn_norm = get_f32(P + "post_attention_norm.weight");
    L.post_ffw_norm = get_f32(P + "post_ffw_norm.weight");
    L.post_norm = get_f32(P + "post_norm.weight");
    L.freq = get_f32(P + "rope_freq.weight");
    if (L.q.rows == 0) throw std::runtime_error("pack: blk missing attn_q");
  }
  if (const auto* te = rd.find("token_embd.weight")) {
    int rows = (int)te->rows, cols = (int)te->cols;
    hp.vocab = rows;
    m.emb.resize((size_t)rows * cols);
    if (te->dtype == pack::Q8RAW) {
      const uint8_t* d = rd.blob(*te);
      for (int r = 0; r < rows; ++r)
        repack::dequant_q8_0_row(d + (size_t)r * (cols / 32) * 34, cols,
                                 m.emb.data() + (size_t)r * cols);
    } else if (te->dtype == pack::F32) {
      std::memcpy(m.emb.data(), rd.blob(*te), (size_t)rows * cols * 4);
    } else {
      throw std::runtime_error("pack: token_embd dtype unsupported");
    }
  } else {
    throw std::runtime_error("pack: no token_embd");
  }
  if (const auto* th = rd.find("head.weight")) {
    if (th->dtype != pack::Q8PACK) throw std::runtime_error("pack: head not Q8PACK");
    Q8Matrix& h = m.head;
    h.rows = (int)th->rows;
    h.cols = (int)th->cols;
    h.nblocks = (h.cols + 31) / 32;
    const uint8_t* d = rd.blob(*th);
    size_t nd = (size_t)h.rows * h.cols;
    h.data.assign(d, d + nd);
    d += nd;
    h.scales.resize(h.rows);
    std::memcpy(h.scales.data(), d, (size_t)h.rows * 4);
    d += (size_t)h.rows * 4;
    size_t ns = (size_t)h.rows * h.nblocks;
    h.bsums.resize(ns);
    std::memcpy(h.bsums.data(), d, ns * 4);
  }
  m.out_norm = get_f32("output_norm.weight");
  if (const auto* ple = rd.find("per_layer_token_embd.weight")) {
    const uint8_t* d = rd.blob(*ple);
    m.layer_emb_raw.assign(d, d + ple->nbytes);
  }
  m.has_layer_emb = ph.has_layer_emb != 0;
  m.layer_emb_dim = ph.layer_emb_dim;
  if (const auto* plp = rd.find("per_layer_model_proj.weight")) {
    int rows = (int)plp->rows, cols = (int)plp->cols;
    const uint8_t* d = rd.blob(*plp);
    std::vector<float> wide((size_t)rows * cols);
    if (plp->dtype == pack::B16RAW)
      repack::widen_bf16(d, rows * cols, wide.data());
    else if (plp->dtype == pack::F32)
      std::memcpy(wide.data(), d, (size_t)rows * cols * 4);
    else
      throw std::runtime_error("pack: per_layer_model_proj dtype unsupported");
    m.layer_plan = quantize_q8(wide.data(), rows, cols);
    m.plan_fp32 = std::move(wide);
  }
  m.layer_proj_norm = get_f32("per_layer_proj_norm.weight");
  return m;
}

inline Model load_gguf(const char* path);
inline Model load_pack(const char* path);

// Dispatcher: .molot packs (magic "MOLT") load fast, anything else loads GGUF.
inline Model load(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) throw std::runtime_error("model: cannot open file");
  char magic[4] = {};
  size_t n = fread(magic, 1, 4, f);
  fclose(f);
  if (n == 4 && !std::memcmp(magic, "MOLT", 4)) return load_pack(path);
  return load_gguf(path);
}

inline Model load_gguf(const char* path) {
  gguf::File f(path);
  Model m;
  HParams& hp = m.hp;
  hp.arch = f.get_s("general.architecture", "?");
  auto U = [&](const char* k, uint64_t d = 0) { return (int)f.get_u64(hp.arch + "." + k, d); };
  auto F = [&](const char* k, double d = 0) { return (float)f.get_f(hp.arch + "." + k, d); };
  hp.n_layer = U("block_count");
  hp.n_embd = U("embedding_length");
  hp.n_head = U("attention.head_count");
  hp.n_kv = U("attention.head_count_kv", hp.n_head);
  hp.n_ffn = U("feed_forward_length");
  hp.expert_count = U("expert_count");
  hp.expert_used_count = U("expert_used_count", hp.expert_count ? 2 : 0);
  hp.eps = F("attention.layer_norm_rms_epsilon", 1e-6);
  hp.rope_base = F("rope.freq_base", 10000);
  hp.rope_swa_base = F("rope.freq_base_swa", hp.rope_base);
  hp.rope_dim = U("rope.dimension_count");
  hp.rope_swa_dim = U("rope.dimension_count_swa", hp.rope_dim);
  hp.logit_cap = F("final_logit_softcapping", 0);
  hp.logit_scale = F("logit_scale", 1.0);
  hp.sw_window = U("attention.sliding_window", 0);
  hp.bos = (int)f.get_u64("tokenizer.ggml.bos_token_id", 1);
  hp.eos = (int)f.get_u64("tokenizer.ggml.eos_token_id", 2);
  if (hp.n_layer <= 0 || hp.n_embd <= 0) throw std::runtime_error("model: bad hparams");
  const bool qwen35 = hp.arch == "qwen35moe";
  if (qwen35) {
    hp.qwen35 = 1;
    hp.rope_dim = U("rope.dimension_count", 64);
    hp.n_ffn = U("expert_feed_forward_length", 0);
    hp.emb_scale = 0;
  }

  // Arch behavior defaults (explicit, documented; generic code reads flags).
  // gemma4 keeps every default. muse-glimmer: plain-normed emb init, post
  // eps 1e-8, SiLU FFN, no V norm, rope on SWA layers only (NoPE global),
  // 1/sqrt(head) attention scale (QK-norms carry a converted factor only).
  if (hp.arch == "muse-glimmer") {
    hp.post_eps = 1e-8f;
    hp.ffn_silu = 1;
    hp.v_norm = 0;
    hp.emb_plain_norm = 1;
    hp.rope_swa_only = 1;
    hp.attn_scale_one = 0;
  }
  if (hp.post_eps == 0) hp.post_eps = hp.eps;

  // sliding pattern (optional; default: all global)
  hp.is_swa.assign(hp.n_layer, 0);
  if (const auto* pat = f.find(hp.arch + ".attention.sliding_window_pattern")) {
    for (int i = 0; i < hp.n_layer && i < (int)pat->arr.size(); ++i)
      hp.is_swa[i] = pat->arr[i].u ? 1 : 0;
  }

  // global rope table: file table if present (F32 d/2), else analytic global
  if (const auto* rt = f.find_tensor("rope_freqs.weight")) {
    m.rope_freqs.resize(rt->ne());
    repack::copy_f32(f.tensor_data(*rt), (int)rt->ne(), m.rope_freqs.data());
  }

  m.layers.resize(hp.n_layer);
  for (int b = 0; b < hp.n_layer; ++b) {
    Layer& L = m.layers[b];
    std::string P = "blk." + std::to_string(b) + ".";
    bool p = false;
    L.q = load_any_q4(f, (P + "attn_q.weight").c_str(), p);
    if (!p && qwen35) {
      // Linear-attention blocks store QKV as one [key,key,value] projection.
      L.is_ssm = f.find_tensor((P + "ssm_a").c_str()) != nullptr;
      if (L.is_ssm) {
        L.ssm_qkv = load_any_q4(f, (P + "attn_qkv.weight").c_str(), p);
        L.ssm_z = load_any_q4(f, (P + "attn_gate.weight").c_str(), p);
        L.ssm_out = load_any_q4(f, (P + "ssm_out.weight").c_str(), p);
        L.ssm_a = load_f32(f, (P + "ssm_a").c_str());
        L.ssm_dt = load_f32(f, (P + "ssm_dt.bias").c_str());
        L.ssm_alpha = load_f32(f, (P + "ssm_alpha.weight").c_str());
        L.ssm_beta = load_f32(f, (P + "ssm_beta.weight").c_str());
        L.ssm_conv = load_f32(f, (P + "ssm_conv1d.weight").c_str());
        L.ssm_norm = load_f32(f, (P + "ssm_norm.weight").c_str());
        L.ssm_key_heads = U("ssm.group_count", 16);
        L.ssm_value_heads = U("ssm.time_step_rank", 32);
        L.ssm_key_dim = U("ssm.state_size", 128);
        L.ssm_value_dim = U("ssm.inner_size", 4096) / L.ssm_value_heads;
        if (L.ssm_qkv.data.empty() || L.ssm_z.data.empty() || L.ssm_out.data.empty())
          throw std::runtime_error("model: incomplete qwen35moe SSM block " + P);
      } else {
        // Full-attention blocks may use the regular names below.
        throw std::runtime_error("model: blk missing attn_q: " + P);
      }
    }
    if (!L.is_ssm && !p) throw std::runtime_error("model: blk missing attn_q");
    if (!L.is_ssm && qwen35 && L.q.rows == 2 * hp.n_head * U("attention.key_length")) {
      L.attn_gate = slice_rows(L.q, L.q.rows / 2, L.q.rows / 2);
      L.q = slice_rows(L.q, 0, L.q.rows / 2);
    }
    if (L.is_ssm) {
      L.attn_norm = load_f32(f, (P + "attn_norm.weight").c_str());
      L.post_attn_norm = load_f32(f, (P + "post_attention_norm.weight").c_str());
      L.moe_gate_inp = load_router(f, (P + "ffn_gate_inp.weight").c_str(), p);
      L.moe_shared_router = load_f32(f, (P + "ffn_gate_inp_shexp.weight").c_str());
      L.moe_shared_gate = load_any_q4(f, (P + "ffn_gate_shexp.weight").c_str(), p);
      L.moe_shared_up = load_any_q4(f, (P + "ffn_up_shexp.weight").c_str(), p);
      L.moe_shared_down = load_any_q4(f, (P + "ffn_down_shexp.weight").c_str(), p);
    }
    L.k = load_any_q4(f, (P + "attn_k.weight").c_str(), L.has_kv);
    if (L.has_kv) {
      L.v = load_any_q4(f, (P + "attn_v.weight").c_str(), p);
      if (!p) throw std::runtime_error("model: attn_k without attn_v");
    }
    L.o = load_any_q4(f, (P + "attn_output.weight").c_str(), p);
    L.gate = load_any_q4(f, (P + "ffn_gate.weight").c_str(), p);
    L.up = load_any_q4(f, (P + "ffn_up.weight").c_str(), p);
    L.down = load_any_q4(f, (P + "ffn_down.weight").c_str(), p);
    L.attn_norm = load_f32(f, (P + "attn_norm.weight").c_str());
    L.q_norm = load_f32(f, (P + "attn_q_norm.weight").c_str());
    L.k_norm = load_f32(f, (P + "attn_k_norm.weight").c_str());
    L.ffn_norm = load_f32(f, (P + "ffn_norm.weight").c_str());
    // extras (may be absent on other archs)
    L.post_attn_norm = load_f32(f, (P + "post_attention_norm.weight").c_str());
    L.post_ffw_norm = load_f32(f, (P + "post_ffw_norm.weight").c_str());
    L.post_norm = load_f32(f, (P + "post_norm.weight").c_str());
    L.inp_gate = load_any_q4(f, (P + "inp_gate.weight").c_str(), p);
    L.proj = load_any_q4(f, (P + "proj.weight").c_str(), p);
    if (L.attn_gate.data.empty())
      L.attn_gate = load_any_q4(f, (P + "attn_gate.weight").c_str(), p);
    bool has_moe = false, have_router = false;
    int ne = 0;
    Q4Matrix eg = load_moe_expert(f, (P + "ffn_gate_exps.weight").c_str(), 0, ne, has_moe);
    Q4Matrix eu = load_moe_expert(f, (P + "ffn_up_exps.weight").c_str(), 0, ne, p);
    Q4Matrix ed = load_moe_expert(f, (P + "ffn_down_exps.weight").c_str(), 0, ne, p);
    if (has_moe) {
      if (ne <= 0 || eu.data.empty() || ed.data.empty())
        throw std::runtime_error("model: incomplete MoE expert tensors at " + P);
      L.expert_count = ne;
      L.expert_used_count = hp.expert_used_count > 0 ? hp.expert_used_count : 2;
      L.moe_gate_inp = load_router(f, (P + "ffn_gate_inp.weight").c_str(), have_router);
      if (!have_router) throw std::runtime_error("model: MoE router missing at " + P);
      L.moe_gate_exps.resize(ne);
      L.moe_up_exps.resize(ne);
      L.moe_down_exps.resize(ne);
      L.moe_gate_exps[0] = std::move(eg);
      L.moe_up_exps[0] = std::move(eu);
      L.moe_down_exps[0] = std::move(ed);
      for (int e = 1; e < ne; ++e) {
        int n2 = 0; bool q = false;
        L.moe_gate_exps[e] = load_moe_expert(f, (P + "ffn_gate_exps.weight").c_str(), e, n2, q);
        L.moe_up_exps[e] = load_moe_expert(f, (P + "ffn_up_exps.weight").c_str(), e, n2, q);
        L.moe_down_exps[e] = load_moe_expert(f, (P + "ffn_down_exps.weight").c_str(), e, n2, q);
      }
      L.gate = {}; L.up = {}; L.down = {};
    }
    if (L.is_ssm) {
      // Qwen3.5 stores RMSNorm parameters as (1 + weight).
      for (float& x : L.attn_norm) x -= 1.0f;
      for (float& x : L.post_attn_norm) x -= 1.0f;
      // SSM blocks use the same routed MoE and have no dense FFN tensors.
      L.ffn_norm = L.post_attn_norm;
      L.expert_count = ne;
      L.expert_used_count = hp.expert_used_count;
    }
    if (qwen35 && !L.is_ssm) {
      for (float& x : L.attn_norm) x -= 1.0f;
      for (float& x : L.q_norm) x -= 1.0f;
      for (float& x : L.k_norm) x -= 1.0f;
      for (float& x : L.post_attn_norm) x -= 1.0f;
      L.ffn_norm = L.post_attn_norm;
    }
    if (auto sc = load_f32(f, (P + "layer_output_scale.weight").c_str()); !sc.empty())
      L.out_scale = sc[0];
    // dims
    L.qhd = L.is_ssm ? 0 : L.q.rows / hp.n_head;
    L.swa = hp.is_swa[b];
    if (L.swa) {
      L.rope_dim = hp.rope_swa_dim ? hp.rope_swa_dim : L.qhd;
      L.rope_base = hp.rope_swa_base;
      L.window = hp.sw_window;
    } else {
      L.rope_dim = hp.rope_dim ? hp.rope_dim : L.qhd;
      L.rope_base = hp.rope_base;
      L.window = 0;
    }
    if (!L.is_ssm && L.rope_dim > L.qhd) L.rope_dim = L.qhd;
    // baked rope frequencies: angle = pos * inv/ff (proportional rope).
    // SWA layers: standard analytic (no table). Global: analytic / table.
    // rope_swa_only (glimmer NoPE global): non-SWA layers get empty freq.
    {
      int hd2 = L.qhd / 2;
      if (hp.rope_swa_only && !L.swa) {
        L.freq.clear();
      } else {
        L.freq.resize(hd2);
        if (!L.swa && !m.rope_freqs.empty() && (int)m.rope_freqs.size() >= hd2) {
          for (int i = 0; i < hd2; ++i) {
            float inv = 1.0f / std::pow(L.rope_base, (2.0f * i) / L.qhd);
            L.freq[i] = inv / m.rope_freqs[i];
          }
        } else {
          float base = L.swa ? hp.rope_swa_base : L.rope_base;
          for (int i = 0; i < hd2; ++i) L.freq[i] = 1.0f / std::pow(base, (2.0f * i) / L.qhd);
        }
      }
    }
    if (L.has_kv) {
      int krows = L.k.rows;
      L.khd = krows / hp.n_kv;
      if (L.khd != L.qhd)
        throw std::runtime_error("model: q/k head dim mismatch at blk." + std::to_string(b) +
                                 " q=" + std::to_string(L.qhd) +
                                 " k=" + std::to_string(L.khd) +
                                 " rows=" + std::to_string(krows) +
                                 " nkv=" + std::to_string(hp.n_kv));
    }
  }

  // second pass: shared-KV sources (reference rule — see llama-model.cpp
  // layer_reuse_cb for GEMMA4): tail layers share first_nokv-2 (SWA) or
  // first_nokv-1 (global). Here: 22 (SWA, 256-dim) / 23 (global, 512-dim).
  {
    int first_nokv = hp.n_layer;
    for (int b = 0; b < hp.n_layer; ++b)
      if (!m.layers[b].is_ssm && !m.layers[b].has_kv) {
        first_nokv = b;
        break;
      }
    for (int b = first_nokv; b < hp.n_layer; ++b) {
      Layer& L = m.layers[b];
      int src = first_nokv - (L.swa ? 2 : 1);
      if (src < 0 || !m.layers[src].has_kv)
        throw std::runtime_error("model: no KV source for shared layer");
      L.kv_src = src;
      L.khd = m.layers[src].khd;
      if (L.khd != L.qhd)
        throw std::runtime_error("model: shared-KV dim mismatch at blk");
      std::fprintf(stderr, "[molot] blk.%d shares KV with blk.%d\n", b, src);
    }
  }

  // embeddings + head
  if (const auto* te = f.find_tensor("token_embd.weight")) {
    int rows, cols;
    repack::gguf_shape_2d(*te, rows, cols);
    hp.vocab = rows;
    m.emb_cols = cols;
    if (te->dtype == gguf::F32) {
      m.emb.resize((size_t)rows * cols);
      repack::copy_f32(f.tensor_data(*te), rows * cols, m.emb.data());
      m.emb_kind = 0;
    } else if (te->dtype == gguf::Q8_0 || te->dtype == gguf::Q2_K ||
               te->dtype == gguf::Q3_K || te->dtype == gguf::Q4_K ||
               te->dtype == gguf::Q5_K || te->dtype == gguf::Q6_K ||
               te->dtype == gguf::Q8_K) {
      // keep raw blocks; gather+dequant per token at runtime (saves GBs)
      const uint8_t* d = f.tensor_data(*te);
      size_t bpe = te->dtype == gguf::Q8_0 ? 34 : repack::kquant_block_bytes(te->dtype);
      int per = te->dtype == gguf::Q8_0 ? 32 : 256;
      m.emb_raw.assign(d, d + (size_t)rows * (cols / per) * bpe);
      m.emb_kind = (int)te->dtype;
    } else {
      throw std::runtime_error("model: token_embd dtype unsupported");
    }
    if (const auto* ow = f.find_tensor("output.weight")) {
      int hr, hc;
      repack::gguf_shape_2d(*ow, hr, hc);
      if (ow->dtype == gguf::Q8_0)
        m.head = repack::repack_q8_0(f.tensor_data(*ow), hr, hc);
      else if (ow->dtype == gguf::F32) {
        std::vector<float> tmp((size_t)hr * hc);
        repack::copy_f32(f.tensor_data(*ow), hr * hc, tmp.data());
        m.head = quantize_q8(tmp.data(), hr, hc);
      } else if (ow->dtype == gguf::Q2_K || ow->dtype == gguf::Q3_K ||
                 ow->dtype == gguf::Q4_K || ow->dtype == gguf::Q5_K ||
                 ow->dtype == gguf::Q6_K || ow->dtype == gguf::Q8_K) {
        // streaming requant (no full-matrix fp32 transient for 202k rows)
        Q8Matrix h;
        h.rows = hr;
        h.cols = hc;
        h.nblocks = (hc + 31) / 32;
        h.scales.resize(hr);
        h.data.resize((size_t)hr * hc);
        h.bsums.assign((size_t)hr * h.nblocks, 0);
        const uint8_t* d = f.tensor_data(*ow);
        size_t stride = (size_t)(hc / 256) * repack::kquant_block_bytes(ow->dtype);
        std::vector<float> tmp(hc);
        for (int r = 0; r < hr; ++r) {
          repack::dequant_k_row(ow->dtype, d + (size_t)r * stride, hc, tmp.data());
          quantize_q8_row(tmp.data(), hc, h.nblocks, &h.scales[r],
                          h.data.data() + (size_t)r * hc,
                          h.bsums.data() + (size_t)r * h.nblocks);
        }
        m.head = std::move(h);
      } else {
        throw std::runtime_error("model: output.weight dtype unsupported");
      }
    } else {
      // tied output: requantize embeddings (one-time) + keep raw Q8_0 blocks
      // for exact mode
      m.head = quantize_q8(m.emb.data(), rows, cols);
      if (te->dtype == gguf::Q8_0) {
        const uint8_t* d = f.tensor_data(*te);
        size_t nb = (size_t)rows * (cols / 32) * 34;
        m.head_q80_raw.assign(d, d + nb);
        m.head_q80_cols = cols;
      }
    }
  } else {
    throw std::runtime_error("model: no token_embd");
  }
  m.out_norm = load_f32(f, "output_norm.weight");
  if (qwen35)
    for (float& x : m.out_norm) x -= 1.0f;
  // per-layer input embeddings (optional)
  if (const auto* ple = f.find_tensor("per_layer_token_embd.weight")) {
    m.has_layer_emb = true;
    m.layer_emb_dim = (int)f.get_u64(hp.arch + ".embedding_length_per_layer_input", 256);
    if (ple->dtype == gguf::Q4_0) {
      // keep raw blocks; gather+dequant per token at runtime
      const uint8_t* d = f.tensor_data(*ple);
      // Q4_0 rows: dims [D, V] -> row length D
      int rows, cols;
      repack::gguf_shape_2d(*ple, rows, cols);
      m.layer_emb_raw.assign(d, d + (size_t)rows * (cols / 32) * 18);
    }
    if (const auto* plp = f.find_tensor("per_layer_model_proj.weight")) {
      int rows, cols;
      repack::gguf_shape_2d(*plp, rows, cols);
      const uint8_t* d = f.tensor_data(*plp);
      // widened copy for exact mode (fp32 GEMV, no requant noise)
      m.plan_fp32.resize((size_t)rows * cols);
      if (plp->dtype == gguf::BF16)
        repack::widen_bf16(d, rows * cols, m.plan_fp32.data());
      else if (plp->dtype == gguf::F32)
        repack::copy_f32(d, rows * cols, m.plan_fp32.data());
      else
        throw std::runtime_error("model: per_layer_model_proj dtype unsupported");
      // Q8 working copy for the fast path
      m.layer_plan = quantize_q8(m.plan_fp32.data(), rows, cols);
    }
    m.layer_proj_norm = load_f32(f, "per_layer_proj_norm.weight");
  }
  return m;
}

}  // namespace molot::model
