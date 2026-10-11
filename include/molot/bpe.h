#pragma once
// Byte-level BPE from GGUF tokenizer arrays (tokens + merges).
// v1 scope: whole-string BPE with merge-rank priority over UTF-8 chars.
// No pre-tokenizer regex splitting (none is specified in this file), so
// exotic spacing may segment differently than the reference — decode is
// exact regardless (tokens map back to strings losslessly).
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "molot/gguf.h"

namespace molot::bpe {

struct Tok {
  std::vector<std::string> piece;  // id -> string
  std::unordered_map<std::string, int> merge_rank;
  int bos = 1, eos = 2;
  bool bytes_level = false;  // gpt2-style: pieces are byte-mapped unicode

  // Standard GPT-2 bytes_to_unicode table (byte -> UTF-8 char).
  static const std::string& byte_to_uni(int b) {
    static const std::vector<std::string> table = [] {
      std::vector<int> cs;
      for (int c = 33; c <= 126; ++c) cs.push_back(c);
      for (int c = 161; c <= 172; ++c) cs.push_back(c);
      for (int c = 174; c <= 255; ++c) cs.push_back(c);
      std::vector<int> bt(256, -1);
      for (int c : cs) bt[c] = c;
      int n = 0;
      for (int b2 = 0; b2 < 256; ++b2)
        if (bt[b2] < 0) bt[b2] = 256 + (n++);
      std::vector<std::string> t(256);
      for (int b2 = 0; b2 < 256; ++b2) {
        int c = bt[b2];
        char o[5] = {};
        if (c < 0x80) {
          o[0] = (char)c;
        } else if (c < 0x800) {
          o[0] = (char)(0xC0 | (c >> 6));
          o[1] = (char)(0x80 | (c & 0x3F));
        } else {
          o[0] = (char)(0xE0 | (c >> 12));
          o[1] = (char)(0x80 | ((c >> 6) & 0x3F));
          o[2] = (char)(0x80 | (c & 0x3F));
        }
        t[b2] = o;
      }
      return t;
    }();
    return table[b];
  }

  // Reverse map: unicode char (UTF-8) -> byte, or -1.
  static int uni_to_byte(const std::string& ch) {
    static const std::unordered_map<std::string, int> rev = [] {
      std::unordered_map<std::string, int> r;
      for (int b = 0; b < 256; ++b) r[byte_to_uni(b)] = b;
      return r;
    }();
    auto it = rev.find(ch);
    return it == rev.end() ? -1 : it->second;
  }

  static Tok from_gguf(const gguf::File& f) {
    Tok t;
    t.bos = (int)f.get_u64("tokenizer.ggml.bos_token_id", 1);
    t.eos = (int)f.get_u64("tokenizer.ggml.eos_token_id", 2);
    t.bytes_level = (f.get_s("tokenizer.ggml.model", "") == "gpt2");
    if (const auto* tp = f.find("tokenizer.ggml.tokens")) {
      t.piece.reserve(tp->arr.size());
      for (auto& e : tp->arr) t.piece.push_back(e.s);
    }
    if (const auto* mp = f.find("tokenizer.ggml.merges")) {
      for (size_t i = 0; i < mp->arr.size(); ++i) t.merge_rank[mp->arr[i].s] = (int)i;
    }
    return t;
  }

  static Tok from_pack(const std::vector<std::string>& pieces,
                       const std::vector<std::string>& merges, int bos, int eos,
                       bool bytes_level = false) {
    Tok t;
    t.bos = bos;
    t.eos = eos;
    t.bytes_level = bytes_level;
    t.piece = pieces;
    for (size_t i = 0; i < merges.size(); ++i) t.merge_rank[merges[i]] = (int)i;
    return t;
  }

  // split into UTF-8 chars (single bytes for invalid sequences)
  static std::vector<std::string> chars(const std::string& s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
      unsigned char c = s[i];
      size_t len = 1;
      if ((c & 0x80) == 0) len = 1;
      else if ((c & 0xE0) == 0xC0) len = 2;
      else if ((c & 0xF0) == 0xE0) len = 3;
      else if ((c & 0xF8) == 0xF0) len = 4;
      if (i + len > s.size()) len = s.size() - i;
      out.emplace_back(s.substr(i, len));
      i += len;
    }
    return out;
  }

  std::vector<int> encode(const std::string& s, bool add_bos) const {
    std::vector<int> ids;
    if (add_bos) ids.push_back(bos);
    if (merge_rank.empty()) return ids;  // no merges: caller must handle
    // map piece -> id for single chars; OOV char -> skip (v1)
    std::unordered_map<std::string, int> id;
    id.reserve(piece.size() * 2);
    for (size_t i = 0; i < piece.size(); ++i)
      if (!id.count(piece[i])) id[piece[i]] = (int)i;
    if (bytes_level) {
      // GPT-2 style: map UTF-8 bytes through bytes_to_unicode, then split
      // into chunks (leading spaces attach to the following word) and BPE
      // each chunk independently (pre-tokenizer approximation).
      std::string mapped;
      for (size_t i = 0; i < s.size(); ++i) mapped += byte_to_uni((unsigned char)s[i]);
      std::string sp = byte_to_uni(' ');
      std::vector<std::string> ch = chars(mapped);
      std::vector<std::string> parts;
      std::string cur;
      for (auto& c : ch) {
        if (c == sp && !cur.empty()) {
          parts.push_back(cur);
          cur.clear();
        }
        cur += c;
      }
      if (!cur.empty()) parts.push_back(cur);
      for (auto& part : parts) {
        std::vector<std::string> w = chars(part);
        bpe_merge(w);
        for (auto& p : w) {
          auto it = id.find(p);
          if (it != id.end()) ids.push_back(it->second);
        }
      }
      return ids;
    }
    // SentencePiece convention: space -> U+2581 (there is no plain-space piece)
    std::string t;
    for (size_t i = 0; i < s.size(); ++i) {
      if (s[i] == ' ')
        t += "\xE2\x96\x81";
      else
        t += s[i];
    }
    std::vector<std::string> w = chars(t);
    bpe_merge(w);
    for (auto& p : w) {
      auto it = id.find(p);
      if (it != id.end()) ids.push_back(it->second);
    }
    return ids;
  }

  void bpe_merge(std::vector<std::string>& w) const {
    while (w.size() >= 2) {
      int best = -1, best_rank = std::numeric_limits<int>::max();
      for (size_t i = 0; i + 1 < w.size(); ++i) {
        auto it = merge_rank.find(w[i] + " " + w[i + 1]);
        int r = it == merge_rank.end() ? std::numeric_limits<int>::max() : it->second;
        if (r < best_rank) {
          best_rank = r;
          best = (int)i;
        }
      }
      if (best < 0) break;
      w[best] += w[best + 1];
      w.erase(w.begin() + best + 1);
    }
  }

  std::string decode(const std::vector<int>& ids) const {
    std::string s;
    for (int id : ids)
      if (id >= 0 && id < (int)piece.size()) s += piece[id];
    if (bytes_level) {
      // reverse byte map, unicode chars -> raw bytes
      std::string o;
      for (auto& c : chars(s)) {
        int b = uni_to_byte(c);
        if (b >= 0)
          o += (char)b;
        else
          o += c;  // control/special pieces pass through
      }
      return o;
    }
    // ▁ back to space for display
    std::string o;
    for (size_t i = 0; i < s.size();) {
      if (i + 3 <= s.size() && s[i] == '\xE2' && s[i + 1] == '\x96' && s[i + 2] == '\x81') {
        o += ' ';
        i += 3;
      } else {
        o += s[i++];
      }
    }
    return o;
  }
};

}  // namespace molot::bpe
