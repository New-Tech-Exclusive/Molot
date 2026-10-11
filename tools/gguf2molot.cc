// gguf2molot: convert a GGUF file to a .molot runtime pack.
// Usage: gguf2molot model.gguf [model.molot] [--verify]
// Default output replaces .gguf with .molot. --verify reads the pack back
// and memcmps every tensor against the in-memory model (deterministic
// repack => bit-exact).
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "molot/gguf.h"
#include "molot/model.h"
#include "molot/pack.h"

static void append_bytes(std::vector<uint8_t>& out, const void* p, size_t n) {
  const uint8_t* b = (const uint8_t*)p;
  out.insert(out.end(), b, b + n);
}
template <typename T>
static void append_vec(std::vector<uint8_t>& out, const std::vector<T>& v) {
  append_bytes(out, v.data(), v.size() * sizeof(T));
}
static void add_q4pack(molot::pack::Pack& pk, const std::string& name, const molot::Q4Matrix& m) {
  if (m.data.empty()) return;
  molot::pack::TensorRec t;
  t.name = name;
  t.dtype = molot::pack::Q4PACK;
  t.rows = m.rows;
  t.cols = m.cols;
  t.nelems = (uint64_t)m.rows * m.cols;
  std::vector<uint8_t> b;
  append_vec(b, m.data);
  append_vec(b, m.scales);
  append_vec(b, m.zeros);
  append_vec(b, m.wsums);
  t.nbytes = b.size();
  pk.tensors.push_back(std::move(t));
  pk.blobs.push_back(std::move(b));
}
static void add_q8pack(molot::pack::Pack& pk, const std::string& name, const molot::Q8Matrix& m) {
  if (m.data.empty()) return;
  molot::pack::TensorRec t;
  t.name = name;
  t.dtype = molot::pack::Q8PACK;
  t.rows = m.rows;
  t.cols = m.cols;
  t.nelems = (uint64_t)m.rows * m.cols;
  std::vector<uint8_t> b;
  append_vec(b, m.data);
  append_vec(b, m.scales);
  append_vec(b, m.bsums);
  t.nbytes = b.size();
  pk.tensors.push_back(std::move(t));
  pk.blobs.push_back(std::move(b));
}
static void add_f32(molot::pack::Pack& pk, const std::string& name, const std::vector<float>& v) {
  if (v.empty()) return;
  molot::pack::TensorRec t;
  t.name = name;
  t.dtype = molot::pack::F32;
  t.rows = 0;
  t.cols = (uint64_t)v.size();
  t.nelems = v.size();
  std::vector<uint8_t> b;
  append_vec(b, v);
  t.nbytes = b.size();
  pk.tensors.push_back(std::move(t));
  pk.blobs.push_back(std::move(b));
}
static void add_raw(molot::pack::Pack& pk, const std::string& name, uint32_t dtype,
                    const uint8_t* data, size_t nbytes, uint64_t rows, uint64_t cols) {
  if (!nbytes) return;
  molot::pack::TensorRec t;
  t.name = name;
  t.dtype = dtype;
  t.rows = rows;
  t.cols = cols;
  t.nelems = rows * cols;
  t.nbytes = nbytes;
  pk.tensors.push_back(std::move(t));
  pk.blobs.emplace_back(data, data + nbytes);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage: gguf2molot model.gguf [model.molot] [--verify]\n");
    return 2;
  }
  std::string in = argv[1], out;
  bool verify = false;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--verify"))
      verify = true;
    else
      out = argv[i];
  }
  if (out.empty()) {
    out = in;
    auto p = out.rfind(".gguf");
    out = (p == std::string::npos) ? out + ".molot" : out.substr(0, p) + ".molot";
  }
  printf("[gguf2molot] loading %s ...\n", in.c_str());
  molot::model::Model m = molot::model::load(in.c_str());
  molot::gguf::File gf(in.c_str());  // tokenizer + raw blobs + provenance
  molot::pack::Pack pk;
  pk.source_hash = molot::pack::fnv1a_file(in.c_str());
  pk.source_size = molot::pack::file_size(in.c_str());
  auto& hp = m.hp;
  snprintf(pk.hp.arch, sizeof pk.hp.arch, "%s", hp.arch.c_str());
  pk.hp.n_layer = hp.n_layer;
  pk.hp.n_embd = hp.n_embd;
  pk.hp.n_head = hp.n_head;
  pk.hp.n_kv = hp.n_kv;
  pk.hp.n_ffn = hp.n_ffn;
  pk.hp.eps = hp.eps;
  pk.hp.rope_base = hp.rope_base;
  pk.hp.rope_swa_base = hp.rope_swa_base;
  pk.hp.logit_cap = hp.logit_cap;
  pk.hp.rope_dim = hp.rope_dim;
  pk.hp.rope_swa_dim = hp.rope_swa_dim;
  pk.hp.sw_window = hp.sw_window;
  pk.hp.vocab = hp.vocab;
  pk.hp.bos = hp.bos;
  pk.hp.eos = hp.eos;
  pk.hp.has_layer_emb = m.has_layer_emb ? 1 : 0;
  pk.hp.layer_emb_dim = m.layer_emb_dim;
  for (auto& L : m.layers) {
    molot::pack::LayerRec r;
    r.has_kv = L.has_kv ? 1 : 0;
    r.swa = L.swa ? 1 : 0;
    r.kv_src = L.kv_src;
    r.qhd = L.qhd;
    r.khd = L.khd;
    r.window = L.window;
    r.out_scale = L.out_scale;
    pk.layers.push_back(r);
  }
  for (size_t b = 0; b < m.layers.size(); ++b) {
    auto& L = m.layers[b];
    std::string P = "blk." + std::to_string(b) + ".";
    add_q4pack(pk, P + "attn_q.weight", L.q);
    add_q4pack(pk, P + "attn_k.weight", L.k);
    add_q4pack(pk, P + "attn_v.weight", L.v);
    add_q4pack(pk, P + "attn_output.weight", L.o);
    add_q4pack(pk, P + "ffn_gate.weight", L.gate);
    add_q4pack(pk, P + "ffn_up.weight", L.up);
    add_q4pack(pk, P + "ffn_down.weight", L.down);
    add_q4pack(pk, P + "inp_gate.weight", L.inp_gate);
    add_q4pack(pk, P + "proj.weight", L.proj);
    add_f32(pk, P + "attn_norm.weight", L.attn_norm);
    add_f32(pk, P + "attn_q_norm.weight", L.q_norm);
    add_f32(pk, P + "attn_k_norm.weight", L.k_norm);
    add_f32(pk, P + "ffn_norm.weight", L.ffn_norm);
    add_f32(pk, P + "post_attention_norm.weight", L.post_attn_norm);
    add_f32(pk, P + "post_ffw_norm.weight", L.post_ffw_norm);
    add_f32(pk, P + "post_norm.weight", L.post_norm);
    add_f32(pk, P + "rope_freq.weight", L.freq);
  }
  // embeddings: Q8_0 raw when source is Q8_0 (gather path), else F32
  if (auto* te = gf.find_tensor("token_embd.weight")) {
    int rows, cols;
    molot::repack::gguf_shape_2d(*te, rows, cols);
    if (te->dtype == molot::gguf::Q8_0)
      add_raw(pk, "token_embd.weight", molot::pack::Q8RAW, gf.tensor_data(*te),
              (size_t)rows * (cols / 32) * 34, rows, cols);
    else
      add_f32(pk, "token_embd.weight", m.emb);
  }
  add_q8pack(pk, "head.weight", m.head);
  add_f32(pk, "output_norm.weight", m.out_norm);
  if (auto* ple = gf.find_tensor("per_layer_token_embd.weight")) {
    int rows, cols;
    molot::repack::gguf_shape_2d(*ple, rows, cols);
    add_raw(pk, "per_layer_token_embd.weight", molot::pack::Q4RAW, gf.tensor_data(*ple),
            (size_t)rows * (cols / 32) * 18, rows, cols);
  }
  if (auto* plp = gf.find_tensor("per_layer_model_proj.weight")) {
    int rows, cols;
    molot::repack::gguf_shape_2d(*plp, rows, cols);
    uint32_t dt = plp->dtype == molot::gguf::BF16 ? molot::pack::B16RAW : molot::pack::F32;
    size_t esz = plp->dtype == molot::gguf::BF16 ? 2 : 4;
    add_raw(pk, "per_layer_model_proj.weight", dt, gf.tensor_data(*plp),
            (size_t)rows * cols * esz, rows, cols);
  }
  add_f32(pk, "per_layer_proj_norm.weight", m.layer_proj_norm);
  if (auto* tp = gf.find("tokenizer.ggml.tokens"))
    for (auto& e : tp->arr) pk.tok_pieces.push_back(e.s);
  if (auto* mp = gf.find("tokenizer.ggml.merges"))
    for (auto& e : mp->arr) pk.tok_merges.push_back(e.s);
  pk.chat_template = gf.get_s("tokenizer.chat_template", "");
  pk.tok_model = gf.get_s("tokenizer.ggml.model", "");
  pk.tok_model = gf.get_s("tokenizer.ggml.model", "");
  printf("[gguf2molot] %zu tensors, writing %s ...\n", pk.tensors.size(), out.c_str());
  molot::pack::write_pack(out.c_str(), pk);
  printf("[gguf2molot] wrote %s (src hash %016llx)\n", out.c_str(),
         (unsigned long long)pk.source_hash);
  if (verify) {
    printf("[gguf2molot] verifying against live model ...\n");
    molot::pack::Reader rd(out.c_str());
    if (rd.meta.tensors.size() != pk.tensors.size())
      throw std::runtime_error("verify: tensor count mismatch");
    for (size_t i = 0; i < pk.tensors.size(); ++i) {
      auto& a = pk.tensors[i];
      auto& b = rd.meta.tensors[i];
      if (a.name != b.name || a.nbytes != b.nbytes || a.dtype != b.dtype)
        throw std::runtime_error("verify: table mismatch at " + a.name);
      if (memcmp(pk.blobs[i].data(), rd.blob(b), a.nbytes))
        throw std::runtime_error("verify: bytes mismatch at " + a.name);
    }
    if (rd.meta.tok_pieces != pk.tok_pieces || rd.meta.tok_merges != pk.tok_merges)
      throw std::runtime_error("verify: tokenizer mismatch");
    printf("[gguf2molot] verify OK (%zu tensors, %zu pieces, %zu merges)\n", pk.tensors.size(),
           pk.tok_pieces.size(), pk.tok_merges.size());
  }
  return 0;
}
