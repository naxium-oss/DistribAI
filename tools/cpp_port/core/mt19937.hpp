// Torch-exact CPU float RNG.
//
// Empirically verified against torch 2.14 (this venv, 2026-09-23):
//   torch.manual_seed(S); torch.rand(N)
// is bit-identical to (MT19937 init_genrand(S) next_u32() & 0xFFFFFF) * 2^-24,
// and torch uniform_(from, to) for float is from + (to-from) * u where u is
// that masked-24 uniform. Weight init (kaiming_uniform(a=sqrt(5)) + bias
// uniform) consumes the same stream, weight-then-bias per Linear module.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace distribai {

class TorchRng {
 public:
  explicit TorchRng(uint64_t seed) { seed_(static_cast<uint32_t>(seed & 0xFFFFFFFFu)); }

  uint32_t next_u32() {
    if (idx_ >= 624) generate_();
    uint32_t y = mt_[idx_++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
  }

  // torch.rand float: (u32 & 0xFFFFFF) * 2^-24, in [0, 1)
  float rand_float() {
    return static_cast<float>(next_u32() & 0xFFFFFFu) * (1.0f / 16777216.0f);
  }

  // torch uniform_(from, to) for float tensors: from/to arrive as doubles,
  // are cast to float, then the affine transform is a FLOAT FMA:
  //   fmaf(to_f - from_f, u_f, from_f)
  // (fused multiply-add, no intermediate rounding). Verified empirically:
  // double-precision transforms diverge ~1 ULP on 208/1291 values;
  // float-FMA matches all 1291 init draws bit-for-bit (2026-09-23).
  float uniform(double from, double to) {
    const float lo = static_cast<float>(from);
    const float hi = static_cast<float>(to);
    return std::fma(hi - lo, rand_float(), lo);
  }

  void rand_n(float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = rand_float();
  }

  // Full-state capture/restore for checkpointing: the resumed generator must
  // draw exactly the u32s the crashed run would have drawn next.
  struct State {
    uint32_t mt[624];
    int32_t index;
  };

  State export_state() const {
    State s;
    std::memcpy(s.mt, mt_, sizeof(mt_));
    s.index = idx_;
    return s;
  }

  void import_state(const State& s) {
    std::memcpy(mt_, s.mt, sizeof(mt_));
    idx_ = s.index;
    if (idx_ > 624 || idx_ < 0) idx_ = 624;
  }

 private:
  void seed_(uint32_t s) {
    mt_[0] = s;
    for (int i = 1; i < 624; ++i) {
      mt_[i] = 1812433253u * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + static_cast<uint32_t>(i);
    }
    idx_ = 624;
  }

  void generate_() {
    for (int i = 0; i < 624; ++i) {
      const uint32_t y = (mt_[i] & 0x80000000u) | (mt_[(i + 1) % 624] & 0x7fffffffu);
      uint32_t v = mt_[(i + 397) % 624] ^ (y >> 1);
      if (y & 1u) v ^= 0x9908b0dfu;
      mt_[i] = v;
    }
    idx_ = 0;
  }

  uint32_t mt_[624];
  int idx_;
};

// torch nn.Linear default init, in construction order (weight then bias):
//   weight (out, fan_in) ~ kaiming_uniform(a=sqrt(5)) => uniform(-b, b), b = 1/sqrt(fan_in)
//   bias   (out)         ~ uniform(-b, b), same b
inline void torch_linear_init(TorchRng& rng, int fan_in, int out, float* w, float* b) {
  // kaiming_uniform(a=sqrt(5)) for Linear: bound = 1/sqrt(fan_in), kept as
  // double through the transform (float-cast bounds change the bytes).
  const double bound = 1.0 / std::sqrt(static_cast<double>(fan_in));
  const size_t wn = static_cast<size_t>(out) * static_cast<size_t>(fan_in);
  for (size_t i = 0; i < wn; ++i) w[i] = rng.uniform(-bound, bound);
  for (int i = 0; i < out; ++i) b[i] = rng.uniform(-bound, bound);
}

}  // namespace distribai
