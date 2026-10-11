// molot-run: GGUF -> tokens. Usage:
//   molot-run model.gguf "prompt" [-n 32] [-t 0.0] [--topk 40] [--topp 0.9] [--ctx 4096]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "molot/bpe.h"
#include "molot/forward.h"
#include "molot/gguf.h"
#include "molot/model.h"
#include "molot/pack.h"
#include "molot/sample.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::printf("usage: molot-run model.gguf \"prompt\" [-n 32] [-t 0.0] [--topk 40] [--topp 0.9] [--ctx 4096]\n");
    return 2;
  }
  int n_new = 32, top_k = 40, ctx = 4096, debug = 0;
  float temp = 0.0f, top_p = 0.9f, rep_pen = 1.0f;
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "-n") && i + 1 < argc) n_new = atoi(argv[++i]);
    if (!strcmp(argv[i], "-t") && i + 1 < argc) temp = (float)atof(argv[++i]);
    if (!strcmp(argv[i], "--topk") && i + 1 < argc) top_k = atoi(argv[++i]);
    if (!strcmp(argv[i], "--topp") && i + 1 < argc) top_p = (float)atof(argv[++i]);
    if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
    if (!strcmp(argv[i], "--rep-pen") && i + 1 < argc) rep_pen = (float)atof(argv[++i]);
    if (!strcmp(argv[i], "--debug")) debug = 1;
  }
  auto t0 = std::chrono::steady_clock::now();
  molot::model::Model m;
  try {
    m = molot::model::load(argv[1]);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "molot-run: %s\n", e.what());
    return 1;
  }
  auto t1 = std::chrono::steady_clock::now();
  std::fprintf(stderr, "[molot] loaded %s arch=%s layers=%d embd=%d vocab=%d in %.1fs\n",
               argv[1], m.hp.arch.c_str(), m.hp.n_layer, m.hp.n_embd, m.hp.vocab,
               std::chrono::duration<double>(t1 - t0).count());
  molot::bpe::Tok tok;
  {
    FILE* tf = fopen(argv[1], "rb");
    char magic[4] = {};
    bool is_pack = tf && fread(magic, 1, 4, tf) == 4 && !std::memcmp(magic, "MOLT", 4);
    if (tf) fclose(tf);
    if (is_pack) {
      molot::pack::Reader rd(argv[1]);
      tok = molot::bpe::Tok::from_pack(rd.meta.tok_pieces, rd.meta.tok_merges, rd.meta.hp.bos,
                                       rd.meta.hp.eos, rd.meta.tok_model == "gpt2");
    } else {
      molot::gguf::File gf(argv[1]);  // tokenizer arrays (mmap shared, cheap)
      tok = molot::bpe::Tok::from_gguf(gf);
    }
  }
  std::vector<int> prompt = tok.encode(argv[2], true);
  std::fprintf(stderr, "[molot] prompt tokens: %d\n", (int)prompt.size());

  molot::fwd::Runner rnn(std::move(m), ctx);
  std::vector<float> logits(rnn.m.hp.vocab);
  std::mt19937 rng(42);
  int pos = 0;
  auto tp0 = std::chrono::steady_clock::now();
  rnn.forward_prefill(prompt.data(), (int)prompt.size(), pos, logits.data());
  pos += (int)prompt.size();
  auto tp1 = std::chrono::steady_clock::now();
  std::fprintf(stderr, "[molot] prefill %d tokens in %.1fs\n", (int)prompt.size(),
               std::chrono::duration<double>(tp1 - tp0).count());

  std::vector<int> out;
  if (const char* lp = std::getenv("MOLOT_DUMP_LOGITS")) {
    FILE* f = fopen(lp, "wb");
    fwrite(logits.data(), 4, logits.size(), f);
    fclose(f);
  }
  if (debug) {
    // top-8 logits after prefill + entropy estimate
    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 8, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    double mx = logits[idx[0]], den = 0;
    for (float l : logits) den += std::exp(l - mx);
    std::fprintf(stderr, "[debug] top-8 after prefill:\n");
    for (int i = 0; i < 8; ++i) {
      int id = idx[i];
      std::string pc = id < (int)tok.piece.size() ? tok.piece[id] : "?";
      std::fprintf(stderr, "  id=%-7d p=%.4f %s\n", id, std::exp(logits[id] - mx) / den,
                   pc.c_str());
    }
    std::fprintf(stderr, "[debug] prompt decode-check: '%s' ids=",
                 tok.decode(prompt).c_str());
    for (int id : prompt) std::fprintf(stderr, "%d ", id);
    std::fprintf(stderr, "\n");
  }
  auto tg0 = std::chrono::steady_clock::now();
  int turn_close = -1;
  for (size_t i = 0; i < tok.piece.size(); ++i)
    if (tok.piece[i] == "<turn|>") turn_close = (int)i;
  for (int i = 0; i < n_new; ++i) {
    std::vector<int> recent(out.size() > 64 ? out.end() - 64 : out.begin(), out.end());
    int id = molot::sample::sample_topp(logits.data(), (int)logits.size(), temp, top_k,
                                        top_p, rng, rep_pen, recent);
    if (id == tok.eos || (turn_close >= 0 && id == turn_close)) break;
    out.push_back(id);
    rnn.forward_token(id, pos++, logits.data());
  }
  auto tg1 = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(tg1 - tg0).count();
  std::printf("%s\n", tok.decode(out).c_str());
  std::fprintf(stderr, "[molot] generated %d tokens in %.1fs (%.1f tok/s)\n", (int)out.size(),
               secs, out.size() / (secs > 0 ? secs : 1));
  return 0;
}
