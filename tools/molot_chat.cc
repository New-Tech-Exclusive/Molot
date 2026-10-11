// molot-chat: interactive chat over the molot runtime (GGUF or .molot pack).
// Usage: molot-chat model.[gguf|molot] [-t 0.7] [--topk 40] [--topp 0.9]
//        [-n 256] [--ctx 4096] [--sys "..."]
// Commands: /quit /reset /temp X /n X. History is kept as tokens; each user
// turn is appended raw (no Jinja template rendering in v1 — the template
// string ships in .molot packs for a future renderer). Streaming output.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "molot/bpe.h"
#include "molot/forward.h"
#include "molot/gguf.h"
#include "molot/model.h"
#include "molot/pack.h"
#include "molot/sample.h"

static bool is_molot_pack(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  char magic[4] = {};
  size_t n = fread(magic, 1, 4, f);
  fclose(f);
  return n == 4 && !std::memcmp(magic, "MOLT", 4);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage: molot-chat model.[gguf|molot] [-t 0.7] [--topk 40] [--topp 0.9] [-n 256] [--ctx 4096] [--sys \"...\"]\n");
    return 2;
  }
  int n_new = 256, top_k = 40, ctx = 4096;
  float temp = 0.7f, top_p = 0.9f, rep_pen = 1.1f;
  std::string sys = "You are a helpful AI assistant.";  // default role grounding;
                                                         // override with --sys, disable with --sys ""
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "-n") && i + 1 < argc) n_new = atoi(argv[++i]);
    if (!strcmp(argv[i], "-t") && i + 1 < argc) temp = (float)atof(argv[++i]);
    if (!strcmp(argv[i], "--topk") && i + 1 < argc) top_k = atoi(argv[++i]);
    if (!strcmp(argv[i], "--topp") && i + 1 < argc) top_p = (float)atof(argv[++i]);
    if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
    if (!strcmp(argv[i], "--sys") && i + 1 < argc) sys = argv[++i];
    if (!strcmp(argv[i], "--rep-pen") && i + 1 < argc) rep_pen = (float)atof(argv[++i]);
  }
  auto t0 = std::chrono::steady_clock::now();
  molot::model::Model m = molot::model::load(argv[1]);
  molot::bpe::Tok tok;
  if (is_molot_pack(argv[1])) {
    molot::pack::Reader rd(argv[1]);
    tok = molot::bpe::Tok::from_pack(rd.meta.tok_pieces, rd.meta.tok_merges, rd.meta.hp.bos,
                                     rd.meta.hp.eos, rd.meta.tok_model == "gpt2");
  } else {
    molot::gguf::File gf(argv[1]);
    tok = molot::bpe::Tok::from_gguf(gf);
  }
  std::fprintf(stderr, "[molot] loaded %s arch=%s layers=%d vocab=%d in %.1fs\n", argv[1],
               m.hp.arch.c_str(), m.hp.n_layer, m.hp.vocab,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

  molot::fwd::Runner rnn(std::move(m), ctx);
  std::vector<float> logits(rnn.m.hp.vocab);
  std::mt19937 rng(42);
  std::vector<int> hist;
  hist.push_back(tok.bos);
  int pos = 0;
  auto feed = [&](const std::vector<int>& ids) {
    for (int id : ids) {
      rnn.forward_token(id, pos++, logits.data());
      hist.push_back(id);
    }
  };
  // Turn framing from the file's chat template (literal markers verified):
  //   <|turn>{role}\n{content}<turn|>\n ... generation ends with <|turn>model\n
  // Marker ids resolved from vocab (absent => raw turn concatenation).
  int turn_open = -1, turn_close = -1;
  for (size_t i = 0; i < tok.piece.size(); ++i) {
    if (tok.piece[i] == "<|turn>") turn_open = (int)i;
    if (tok.piece[i] == "<turn|>") turn_close = (int)i;
  }
  auto frame = [&](const std::string& role, const std::string& text) {
    std::vector<int> ids;
    if (turn_open < 0 || turn_close < 0) {
      auto r = tok.encode(role + ": " + text, false);  // fallback: raw
      ids.insert(ids.end(), r.begin(), r.end());
      return ids;
    }
    ids.push_back(turn_open);
    auto r = tok.encode(role + "\n", false);
    ids.insert(ids.end(), r.begin(), r.end());
    // trim content like `| trim`
    size_t a = text.find_first_not_of(" \t\n"), b = text.find_last_not_of(" \t\n");
    std::string t = (a == std::string::npos) ? "" : text.substr(a, b - a + 1);
    auto c = tok.encode(t, false);
    ids.insert(ids.end(), c.begin(), c.end());
    ids.push_back(turn_close);
    auto nl = tok.encode("\n", false);
    ids.insert(ids.end(), nl.begin(), nl.end());
    return ids;
  };
  auto frame_open_model = [&]() {
    if (turn_open < 0) return std::vector<int>{};
    std::vector<int> ids = {turn_open};
    auto r = tok.encode("model\n", false);
    ids.insert(ids.end(), r.begin(), r.end());
    return ids;
  };
  if (!sys.empty()) feed(frame("system", sys));

  std::string line;
  printf("molot-chat (system: %s; /quit /reset /temp X /n X)\n", sys.empty() ? "-" : sys.c_str());
  while (std::cout << "> " && std::getline(std::cin, line)) {
    if (line == "/quit" || line == "/q") break;
    if (line == "/reset") {
      hist = {tok.bos};
      pos = 0;
      for (auto& c : rnn.kv) c.n = 0;  // fresh KV, no reload
      printf("(reset)\n");
      continue;
    }
    if (line.rfind("/temp ", 0) == 0) {
      temp = (float)atof(line.c_str() + 6);
      printf("(temp %.2f)\n", temp);
      continue;
    }
    if (line.rfind("/n ", 0) == 0) {
      n_new = atoi(line.c_str() + 3);
      printf("(n %d)\n", n_new);
      continue;
    }
    if (line.empty()) continue;
    auto tp0 = std::chrono::steady_clock::now();
    std::vector<int> turn = frame("user", line);
    std::vector<int> open = frame_open_model();
    turn.insert(turn.end(), open.begin(), open.end());
    rnn.forward_prefill(turn.data(), (int)turn.size(), pos, logits.data());
    pos += (int)turn.size();
    hist.insert(hist.end(), turn.begin(), turn.end());
    auto tp1 = std::chrono::steady_clock::now();
    // generate with streaming suffix-diff printing (multibyte-safe)
    std::vector<int> out;
    std::string shown;
    auto tg0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_new; ++i) {
      std::vector<int> recent(out.size() > 64 ? out.end() - 64 : out.begin(), out.end());
      int id = temp > 0 ? molot::sample::sample_topp(logits.data(), (int)logits.size(), temp,
                                                     top_k, top_p, rng, rep_pen, recent)
                        : molot::sample::sample_topp(logits.data(), (int)logits.size(), 0,
                                                     top_k, top_p, rng, rep_pen, recent);
      if (id == rnn.m.hp.eos || (turn_close >= 0 && id == turn_close)) break;
      out.push_back(id);
      std::string full = tok.decode(out);
      if (full.size() > shown.size()) {
        std::cout << full.substr(shown.size()) << std::flush;
        shown = full;
      }
      rnn.forward_token(id, pos++, logits.data());
      hist.push_back(id);
    }
    auto tg1 = std::chrono::steady_clock::now();
    std::cout << "\n";
    double gs = std::chrono::duration<double>(tg1 - tg0).count();
    std::fprintf(stderr, "[prefill %.1fs | %d tokens in %.1fs (%.1f tok/s)]\n",
                 std::chrono::duration<double>(tp1 - tp0).count(), (int)out.size(), gs,
                 out.size() / (gs > 0 ? gs : 1));
    if (std::getenv("MOLOT_PROFILE"))
      std::fprintf(stderr, "[profile] gemv_q4 %.0fms gemv_q8 %.0fms attn %.0fms\n", rnn.t_g4,
                   rnn.t_g8, rnn.t_attn);
    // close the model turn in history so multi-turn state stays well-formed
    if (!out.empty()) feed(frame("model", tok.decode(out)));
  }
  return 0;
}
