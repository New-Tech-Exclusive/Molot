// gguf-info: integration checklist for a GGUF file.
// Usage: gguf-info model.gguf
// Prints arch + hyperparams, dtype inventory, per-block tensor presence
// (catches shared/missing tensors), tied output, tokenizer presence.
#include <cstdio>
#include <map>

#include "molot/gguf.h"

using molot::gguf::File;

static void show_kv(const File& f, const std::string& arch, const char* key) {
  std::string k = arch + "." + key;
  const auto* v = f.find(k.c_str());
  if (!v) {
    std::printf("  %-28s (absent)\n", key);
    return;
  }
  if (v->type == molot::gguf::STR)
    std::printf("  %-28s %s\n", key, v->s.c_str());
  else if (v->type == molot::gguf::F32M || v->type == molot::gguf::F64M)
    std::printf("  %-28s %g\n", key, v->f);
  else
    std::printf("  %-28s %llu\n", key, (unsigned long long)v->u);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::printf("usage: gguf-info model.gguf\n");
    return 2;
  }
  File f(argv[1]);
  std::string arch = f.get_s("general.architecture", "?");
  std::printf("file: %s\nversion: %u  tensors: %zu  kv: %zu  alignment: %u\n", argv[1],
              f.version, f.tensors.size(), f.kv.size(), f.alignment);
  std::printf("arch: %s\n", arch.c_str());
  const char* keys[] = {"context_length",        "embedding_length", "block_count",
                        "feed_forward_length",   "attention.head_count",
                        "attention.head_count_kv", "attention.key_length",
                        "attention.value_length", "rope.dimension_count",
                        "rope.freq_base",        "attention.layer_norm_rms_epsilon",
                        "expert_count",          "expert_used_count"};
  for (auto k : keys) show_kv(f, arch, k);

  std::map<uint32_t, std::pair<int, uint64_t>> inv;
  for (auto& t : f.tensors) {
    try {
      inv[t.dtype].first++;
      inv[t.dtype].second += t.nbytes();
    } catch (...) {
      inv[t.dtype].first++;
    }
  }
  std::printf("dtypes:\n");
  for (auto& [dt, c] : inv) {
    const char* nm = "?";
    try {
      nm = molot::gguf::type_info(dt).name;
    } catch (...) {
    }
    std::printf("  id=%-3u %-8s n=%-5d bytes=%llu\n", dt, nm, c.first,
                (unsigned long long)c.second);
  }
  // per-block presence for first 3 blocks + global tensors
  auto has = [&](const std::string& n) { return f.find_tensor(n) != nullptr; };
  std::printf("globals: token_embd=%d output=%s output_norm=%d\n", has("token_embd.weight"),
              has("output.weight") ? "present"
                                   : "ABSENT (tied to token_embd?)",
              has("output_norm.weight"));
  for (int b = 0; b < 3; ++b) {
    char p[32];
    snprintf(p, sizeof p, "blk.%d.", b);
    std::string P = p;
    std::printf("blk.%d: attn_q=%d attn_k=%d attn_v=%d attn_o=%d q_norm=%d k_norm=%d "
                "ffn_gate=%d ffn_up=%d ffn_down=%d attn_norm=%d ffn_norm=%d\n",
                b, has(P + "attn_q.weight"), has(P + "attn_k.weight"),
                has(P + "attn_v.weight"), has(P + "attn_output.weight"),
                has(P + "attn_q_norm.weight"), has(P + "attn_k_norm.weight"),
                has(P + "ffn_gate.weight"), has(P + "ffn_up.weight"),
                has(P + "ffn_down.weight"), has(P + "attn_norm.weight"),
                has(P + "ffn_norm.weight"));
  }
  // full-block sweep: catches shared/absent tensors past block 2
  {
    int nblocks = 0, missing_k = 0, missing_v = 0;
    for (int b = 0; has("blk." + std::to_string(b) + ".attn_norm.weight"); ++b) {
      nblocks++;
      std::string P = "blk." + std::to_string(b) + ".";
      if (!has(P + "attn_k.weight")) missing_k++;
      if (!has(P + "attn_v.weight")) missing_v++;
    }
    std::printf("blocks: %d with attn_norm, missing attn_k in %d, missing attn_v in %d\n",
                nblocks, missing_k, missing_v);
  }
  // blk.0 shapes: pins down head_dim mapping (q/k/v rows vs head counts)
  {
    const char* names[] = {"token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
                           "blk.0.attn_v.weight", "blk.0.attn_output.weight",
                           "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight",
                           "blk.0.ffn_down.weight", "blk.0.attn_norm.weight",
                           "blk.0.ffn_norm.weight"};
    for (auto nn : names) {
      const auto* t = f.find_tensor(nn);
      if (!t) {
        std::printf("shape %-24s ABSENT\n", nn);
        continue;
      }
      std::printf("shape %-24s dtype=%-6s dims=[", nn,
                  molot::gguf::type_info(t->dtype).name);
      for (size_t i = 0; i < t->dims.size(); ++i)
        std::printf("%s%llu", i ? "," : "", (unsigned long long)t->dims[i]);
      std::printf("]\n");
    }
  }
  const auto* toks = f.find("tokenizer.ggml.tokens");
  std::printf("tokenizer.ggml.tokens: %s\n",
              toks ? (std::to_string(toks->arr.size()) + " entries").c_str() : "ABSENT");
  const auto* merges = f.find("tokenizer.ggml.merges");
  std::printf("tokenizer.ggml.merges: %s\n",
              merges ? (std::to_string(merges->arr.size()) + " entries").c_str() : "ABSENT");
  return 0;
}
