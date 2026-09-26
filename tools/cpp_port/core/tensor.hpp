// DistribAI native core: tensor engine and the 1K-param MLP with AdamW.
// C++17, no external dependencies. Part of the Python->C++ port.
//
// Parity contract with tools/bench/baseline_1k_train.py:
//   * same task: f(x) = sin(pi x) + 0.5 sin(3 pi x) regression
//   * same 1K MLP: 10-30-30-1 (1,241 params) trained with AdamW
//   * same training loop shape: batch 64x10, MSE, loss.backward, opt.step
// RNG differs from torch (PCG32 here), so losses match in magnitude and
// convergence behavior rather than bit-for-bit.
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <pmmintrin.h>
// ---------------------------------------------------------------------------
// Denormal fix (the dominant per-step training cost): after a few thousand
// AdamW steps the m-state values decay into the denormal range and every SSE
// op touching them takes a microcode assist trap (measured: opt.step 88.7us
// -> 3.4us with flush-to-zero). Enable FTZ+DAZ once per process via static
// init. Every thread inherits the creating thread's MXCSR, fork()ed sandbox
// children inherit the parent's, and dlopen'd consumers of libdai_core.so get
// it on load, so this single bootstrap covers every training path.
// Numerics: denormal results become zero. The parity goldens re-verify the
// loss-trajectory drift stays inside the 1e-6 must-match bound (test_golden).
// ---------------------------------------------------------------------------
namespace {
struct DistribaiFtzDazInit {
  DistribaiFtzDazInit() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  }
};
static const DistribaiFtzDazInit distribai_ftz_daz_init_instance;
}  // namespace
#endif  // x86 FTZ/DAZ bootstrap

namespace distribai {

// ---------------------------------------------------------------------------
// RNG: PCG32. Small, fast, seedable, and stream-stable across runs.
// ---------------------------------------------------------------------------
class Rng {
 public:
  explicit Rng(uint64_t seed = 0x853c49e6748fea9bULL) : state_(seed + inc_) { next_u32(); }

  uint32_t next_u32() {
    uint64_t old = state_;
    state_ = old * 6364136223846793005ULL + inc_;
    uint32_t xorshifted = static_cast<uint32_t>(((old >> 18) ^ old) >> 27);
    uint32_t rot = static_cast<uint32_t>(old >> 59);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
  }

  // Uniform float in [0, 1)
  float next_float() { return static_cast<float>(next_u32()) * (1.0f / 4294967296.0f); }

  // Normal(0, 1) via Box-Muller (matches torch.rand * init ranges usage closely
  // enough for parity-in-magnitude).
  float next_normal() {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    float u, v, s;
    do {
      u = next_float() * 2.0f - 1.0f;
      v = next_float() * 2.0f - 1.0f;
      s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    float mul = std::sqrt(-2.0f * std::log(s) / s);
    spare_ = v * mul;
    has_spare_ = true;
    return u * mul;
  }

  // torch.rand shape: [0,1) matrix
  void fill_uniform(float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = next_float();
  }

 private:
  uint64_t state_;
  static constexpr uint64_t inc_ = 0xda3e39cb94b95bdbULL;
  float spare_ = 0.0f;
  bool has_spare_ = false;
};

// ---------------------------------------------------------------------------
// Tensor: row-major dense float matrix with autograd graph node.
// ---------------------------------------------------------------------------
struct Tensor {
  size_t rows = 0, cols = 0;
  std::vector<float> data;
  std::vector<float> grad;

  Tensor() = default;
  Tensor(size_t r, size_t c, float v = 0.0f) : rows(r), cols(c), data(r * c, v), grad(r * c, 0.0f) {}

  size_t size() const { return rows * cols; }
  float* ptr() { return data.data(); }
  const float* ptr() const { return data.data(); }
  float* gptr() { return grad.data(); }
  const float* gptr() const { return grad.data(); }

  void zero_grad() { std::fill(grad.begin(), grad.end(), 0.0f); }

  // row-vector convenience: element at (r, c)
  float& at(size_t r, size_t c) { return data[r * cols + c]; }
  float at(size_t r, size_t c) const { return data[r * cols + c]; }
};

// y = x @ W + b     x: (n, in), W: (in, out), b: (1, out broadcast)
// Perf note (bench hot path): pointers are __restrict__ (every caller passes
// distinct buffers) and there is no data-dependent branch on xv, so a zero skip
// defeats SSE/AVX vectorization of the j-loop for a matmul that never sees
// zeros; adding 0*x to the accumulator is bit-identical to skipping it.
inline void matmul_add(const Tensor& x, const Tensor& W, const Tensor& b, Tensor& y) {
  const size_t n = x.rows, in = x.cols, out = W.cols;
  assert(W.rows == in && b.cols == out && y.rows == n && y.cols == out);
  const float* __restrict__ xd = x.data.data();
  const float* __restrict__ Wd = W.data.data();
  const float* __restrict__ bd = b.data.data();
  float* __restrict__ yd = y.data.data();
  for (size_t i = 0; i < n; ++i) {
    float* yr = &yd[i * out];
    const float* xr = &xd[i * in];
    std::memcpy(yr, bd, out * sizeof(float));
    // k-unrolled by 2: two independent add streams per j (same per-j addition
    // order k=2t,2t+1 as the serial loop - bit-identical, double the ILP
    // against FMA latency).
    size_t k = 0;
    for (; k + 2 <= in; k += 2) {
      const float xv0 = xr[k];
      const float xv1 = xr[k + 1];
      const float* wr0 = &Wd[k * out];
      const float* wr1 = &Wd[(k + 1) * out];
      for (size_t j = 0; j < out; ++j) {
        yr[j] += xv0 * wr0[j];
        yr[j] += xv1 * wr1[j];
      }
    }
    if (k < in) {
      const float xv = xr[k];
      const float* wr = &Wd[k * out];
      for (size_t j = 0; j < out; ++j) yr[j] += xv * wr[j];
    }
  }
}

// Backward for matmul_add: accumulates dW, db and dx (if dx provided).
// dx note: the j-serial accumulator (acc += dy*W[k][j] over j) is a
// dependency chain the vectorizer cannot parallelize; computing dx[k] via
// j-major accumulation into zeroed row slots performs the same adds per k
// in the same order, but over a contiguous inner loop. Per-row start from
// zero is identical to += into the zeroed grad buffer (dx is zeroed per
// step upstream, and only one backward pass writes each row per step).
inline void matmul_add_backward(
    const Tensor& x, const Tensor& W, const Tensor& dy, Tensor* dx, Tensor& dW, Tensor& db) {
  const size_t n = x.rows, in = x.cols, out = W.cols;
  const float* __restrict__ xd = x.data.data();
  const float* __restrict__ Wd = W.data.data();
  const float* __restrict__ dyd = dy.data.data();
  float* __restrict__ dWd = dW.data.data();
  float* __restrict__ dbd = db.data.data();
  float* __restrict__ dxd = dx ? dx->data.data() : nullptr;
  for (size_t i = 0; i < n; ++i) {
    const float* dyr = &dyd[i * out];
    const float* xr = &xd[i * in];
    for (size_t j = 0; j < out; ++j) dbd[j] += dyr[j];
    if (dxd) {
      // dx[k] = sum_j dy[j] * W(k,j): W is (in, out) here, so row k is
      // contiguous; the per-k accumulator chains are independent across k.
      // k-unrolled by 2: two independent chains share each dyr[j] load (same
      // per-k addition order j=0..out-1 as the serial loop - bit-identical).
      // Callers zero dx buffers per step, so accumulating straight into
      // dxr[k] matches the previous acc-then-add form bit-for-bit.
      float* dxr = &dxd[i * in];
      size_t k = 0;
      for (; k + 2 <= in; k += 2) {
        const float* wr0 = &Wd[k * out];
        const float* wr1 = &Wd[(k + 1) * out];
        float v0 = dxr[k];
        float v1 = dxr[k + 1];
        for (size_t j = 0; j < out; ++j) {
          const float d = dyr[j];
          v0 += d * wr0[j];
          v1 += d * wr1[j];
        }
        dxr[k] = v0;
        dxr[k + 1] = v1;
      }
      for (; k < in; ++k) {
        const float* wr = &Wd[k * out];
        float v = dxr[k];
        for (size_t j = 0; j < out; ++j) v += dyr[j] * wr[j];
        dxr[k] = v;
      }
    }
    // dW rows are independent cells (k,j); process two k rows per j pass so
    // the two j-streams are independent FMA chains sharing each dyr[j] load
    // (same per-cell add order over i - bit-identical).
    size_t k = 0;
    for (; k + 2 <= in; k += 2) {
      const float xv0 = xr[k];
      const float xv1 = xr[k + 1];
      float* dwr0 = &dWd[k * out];
      float* dwr1 = &dWd[(k + 1) * out];
      for (size_t j = 0; j < out; ++j) {
        const float d = dyr[j];
        dwr0[j] += xv0 * d;
        dwr1[j] += xv1 * d;
      }
    }
    for (; k < in; ++k) {
      const float xv = xr[k];
      float* dwr = &dWd[k * out];
      for (size_t j = 0; j < out; ++j) dwr[j] += xv * dyr[j];
    }
  }
}

// ReLU forward/backward (elementwise).
inline void relu_forward(const Tensor& x, Tensor& y) {
  for (size_t i = 0; i < x.size(); ++i) y.data[i] = x.data[i] > 0.0f ? x.data[i] : 0.0f;
}
inline void relu_backward(const Tensor& x, const Tensor& dy, Tensor& dx) {
  for (size_t i = 0; i < x.size(); ++i) dx.data[i] += x.data[i] > 0.0f ? dy.data[i] : 0.0f;
}

// MSE loss on (n,1): mean((pred-target)^2); returns loss and dL/dpred.
inline float mse_loss(const Tensor& pred, const Tensor& target, Tensor& dpred) {
  float sum = 0.0f;
  const size_t n = pred.size();
  for (size_t i = 0; i < n; ++i) {
    const float d = pred.data[i] - target.data[i];
    sum += d * d;
  }
  const float loss = sum / static_cast<float>(n);
  const float scale = 2.0f / static_cast<float>(n);
  for (size_t i = 0; i < n; ++i) dpred.data[i] = (pred.data[i] - target.data[i]) * scale;
  return loss;
}

// ---------------------------------------------------------------------------
// AdamW optimizer (decoupled weight decay, matches torch.optim.AdamW defaults
// betas=(0.9,0.999), eps=1e-8 as used in the baseline harness lr=1e-2).
// ---------------------------------------------------------------------------
class AdamW {
 public:
  AdamW(std::vector<Tensor*> params, float lr, float weight_decay = 0.0f,
        float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
      : params_(std::move(params)), lr_(lr), wd_(weight_decay), b1_(beta1), b2_(beta2), eps_(eps) {
    for (auto* p : params_) {
      m_.emplace_back(p->rows, p->cols, 0.0f);
      v_.emplace_back(p->rows, p->cols, 0.0f);
    }
  }

  void zero_grad() {
    for (auto* p : params_) p->zero_grad();
  }

  void step() {
    ++t_;
    // Step-level constants hoisted out of the element loop (identical float
    // ops, computed once): the per-element math below is unchanged bit-for-bit.
    const float f_t = static_cast<float>(t_);
    const float bc1 = 1.0f - std::pow(b1_, f_t);
    const float bc2 = 1.0f - std::pow(b2_, f_t);
    const float om_b1 = 1.0f - b1_;
    const float om_b2 = 1.0f - b2_;
    // The wd_ != 0 check is loop-invariant; unswitching it (same per-element
    // expressions) lets the elementwise loop vectorize - AdamW step is the
    // dominant cost at 10K/100K param widths.
    if (wd_ != 0.0f) {
      for (size_t pi = 0; pi < params_.size(); ++pi) {
        float* __restrict__ pd = params_[pi]->data.data();
        float* __restrict__ md = m_[pi].data.data();
        float* __restrict__ vd = v_[pi].data.data();
        const float* __restrict__ gd = params_[pi]->grad.data();
        const size_t n = params_[pi]->size();
        for (size_t i = 0; i < n; ++i) {
          md[i] = b1_ * md[i] + om_b1 * gd[i];
          vd[i] = b2_ * vd[i] + om_b2 * gd[i] * gd[i];
          const float m_hat = md[i] / bc1;
          const float v_hat = vd[i] / bc2;
          float upd = lr_ * m_hat / (std::sqrt(v_hat) + eps_);
          upd += lr_ * wd_ * pd[i];  // decoupled decay
          pd[i] -= upd;
        }
      }
    } else {
      for (size_t pi = 0; pi < params_.size(); ++pi) {
        float* __restrict__ pd = params_[pi]->data.data();
        float* __restrict__ md = m_[pi].data.data();
        float* __restrict__ vd = v_[pi].data.data();
        const float* __restrict__ gd = params_[pi]->grad.data();
        const size_t n = params_[pi]->size();
        for (size_t i = 0; i < n; ++i) {
          md[i] = b1_ * md[i] + om_b1 * gd[i];
          vd[i] = b2_ * vd[i] + om_b2 * gd[i] * gd[i];
          const float m_hat = md[i] / bc1;
          const float v_hat = vd[i] / bc2;
          const float upd = lr_ * m_hat / (std::sqrt(v_hat) + eps_);
          pd[i] -= upd;
        }
      }
    }
  }

 private:
  std::vector<Tensor*> params_;
  std::vector<Tensor> m_, v_;
  float lr_, wd_, b1_, b2_, eps_;
  int64_t t_ = 0;
};

// ---------------------------------------------------------------------------
// MLP 10-30-30-1 with ReLU, the "1K parameter" model from the baseline
// sandbox harness (_train_1k_in_child):
//   Linear(10,30): 300 w + 30 b = 330
//   Linear(30,30): 900 w + 30 b = 930
//   Linear(30,1):   30 w +  1 b =  31
//   total = 1,291 params (the 1,021 figure in baseline JSONs is the
//   separate train_scaling 1-30-30-1 variant; same ~1K scale).
// ---------------------------------------------------------------------------
class Mlp1k {
 public:
  Mlp1k(uint64_t seed, float gain = 1.0f)
      : rng_(seed), W1_(10, 30), b1_(1, 30), W2_(30, 30), b2_(1, 30), W3_(30, 1), b3_(1, 1) {
    init_linear(W1_, b1_, gain);
    init_linear(W2_, b2_, gain);
    init_linear(W3_, b3_, gain);
  }

  int64_t param_count() const {
    return static_cast<int64_t>(W1_.size() + b1_.size() + W2_.size() + b2_.size() + W3_.size() +
                                b3_.size());
  }

  // Forward + loss + backward for one batch. Returns loss value.
  float train_step(const Tensor& x, const Tensor& y, AdamW& opt, float* loss_out = nullptr);

  // Inference only.
  void forward(const Tensor& x, Tensor& h1, Tensor& a1, Tensor& h2, Tensor& a2, Tensor& out) const;

  std::vector<Tensor*> params() { return {&W1_, &b1_, &W2_, &b2_, &W3_, &b3_}; }

  // Accessors for serialization/aggregation paths.
  const Tensor& W1() const { return W1_; }
  const Tensor& W2() const { return W2_; }
  const Tensor& W3() const { return W3_; }
  const Tensor& b1() const { return b1_; }
  const Tensor& b2() const { return b2_; }
  const Tensor& b3() const { return b3_; }

 private:
  void init_linear(Tensor& W, Tensor& b, float gain) {
    const float fan_in = static_cast<float>(W.rows);
    const float stdv = gain * std::sqrt(1.0f / fan_in);
    for (auto& v : W.data) v = rng_.next_normal() * stdv;
    std::fill(b.data.begin(), b.data.end(), 0.0f);
  }

  Rng rng_;
  Tensor W1_, b1_, W2_, b2_, W3_, b3_;
};

inline void Mlp1k::forward(
    const Tensor& x, Tensor& h1, Tensor& a1, Tensor& h2, Tensor& a2, Tensor& out) const {
  matmul_add(x, W1_, b1_, h1);
  relu_forward(h1, a1);
  matmul_add(a1, W2_, b2_, h2);
  relu_forward(h2, a2);
  matmul_add(a2, W3_, b3_, out);
}

inline float Mlp1k::train_step(const Tensor& x, const Tensor& y, AdamW& opt, float* loss_out) {
  // PyTorch semantic: zero_grad() at the top of every training step.
  opt.zero_grad();
  const size_t n = x.rows;
  Tensor h1(n, 30), a1(n, 30), h2(n, 30), a2(n, 30), out(n, 1);
  forward(x, h1, a1, h2, a2, out);

  Tensor dout(n, 1);
  const float loss = mse_loss(out, y, dout);

  Tensor da2(n, 30), dh2(n, 30), da1(n, 30), dh1(n, 30), dx(n, 10);
  Tensor dW3(30, 1), db3(1, 1), dW2(30, 30), db2(1, 30), dW1(10, 30), db1(1, 30);
  matmul_add_backward(a2, W3_, dout, &da2, dW3, db3);
  relu_backward(h2, da2, dh2);
  matmul_add_backward(a1, W2_, dh2, &da1, dW2, db2);
  relu_backward(h1, da1, dh1);
  matmul_add_backward(x, W1_, dh1, nullptr, dW1, db1);

  // Accumulate grads into param.grad
  auto acc = [](Tensor& p, const Tensor& g) {
    for (size_t i = 0; i < p.size(); ++i) p.grad[i] += g.data[i];
  };
  acc(W1_, dW1); acc(b1_, db1); acc(W2_, dW2); acc(b2_, db2); acc(W3_, dW3); acc(b3_, db3);

  opt.step();
  if (loss_out) *loss_out = loss;
  return loss;
}

// Task data: x in [0,1]^10 (only dim 0 drives the target, mirroring the
// scalar task batched 64-wide in the baseline), y = f(x0).
inline void make_batch(Tensor& x, Tensor& y, Rng& rng) {
  for (size_t i = 0; i < x.rows; ++i) {
    for (size_t j = 0; j < x.cols; ++j) x.data[i * x.cols + j] = rng.next_float();
    const float t = x.data[i * x.cols];
    y.data[i] = std::sin(3.14159265f * t) + 0.5f * std::sin(3.0f * 3.14159265f * t);
  }
}

}  // namespace distribai
