#pragma once
// Sampler: greedy / temperature / top-k / top-p over softcapped logits.
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace molot::sample {

inline int greedy(const float* logits, int n) {
  return (int)(std::max_element(logits, logits + n) - logits);
}

inline int sample_topp(const float* logits, int n, float temp, int top_k, float top_p,
                       std::mt19937& rng, float rep_pen = 1.0f,
                       const std::vector<int>& recent = {}) {
  if (!(temp > 0)) {
    if (rep_pen == 1.0f || recent.empty()) return greedy(logits, n);
    // greedy with penalty: recompute argmax over penalized logits
    int best = 0;
    float bv = -1e30f;
    for (int i = 0; i < n; ++i) {
      float l = logits[i];
      for (int r : recent)
        if (r == i) {
          l = l > 0 ? l / rep_pen : l * rep_pen;
          break;
        }
      if (l > bv) {
        bv = l;
        best = i;
      }
    }
    return best;
  }
  std::vector<std::pair<float, int>> s;
  s.reserve(n);
  float mx = *std::max_element(logits, logits + n);
  for (int i = 0; i < n; ++i) {
    float l = logits[i] - mx;
    if (rep_pen != 1.0f) {
      for (int r : recent)
        if (r == i) {
          l = l > 0 ? l / rep_pen : l * rep_pen;
          break;
        }
    }
    s.emplace_back(l / temp, i);
  }
  if (top_k > 0 && top_k < n) {
    std::nth_element(s.begin(), s.begin() + top_k, s.end(),
                     [](auto& a, auto& b) { return a.first > b.first; });
    s.resize(top_k);
  }
  std::sort(s.begin(), s.end(), [](auto& a, auto& b) { return a.first > b.first; });
  std::vector<float> probs(s.size());
  double den = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    probs[i] = std::exp(s[i].first);
    den += probs[i];
  }
  double cum = 0, r = std::uniform_real_distribution<double>(0, 1)(rng);
  for (size_t i = 0; i < s.size(); ++i) {
    cum += probs[i] / den;
    if (r < cum || i + 1 == s.size()) return s[i].second;
    if (top_p < 1 && cum > top_p && i > 0) return s[i].second;
  }
  return s.back().second;
}

}  // namespace molot::sample
