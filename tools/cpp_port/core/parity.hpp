// Bit-parity training core. Matches tools/bench/baseline_1k_train.py's
// sandbox child semantics contract-for-contract:
//   * data: per step, x = rand(64,10) then y = rand(64,1)  (random targets)
//   * model: nn.Sequential(Linear(10,30), ReLU, Linear(30,30), ReLU, Linear(30,1))
//     weights stored (out, in) like torch; init kaiming_uniform(a=sqrt5) weight
//     then bias, consuming one shared stream in module construction order
//   * opt: AdamW(lr=0.01) with torch defaults (wd=0.01, betas 0.9/0.999, eps 1e-8),
//     op order mirrors torch _single_tensor_adamw
//   * loss: MSE mean over 64
//   * grad export: flattened per-param in model.parameters() order
#pragma once

#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include "mt19937.hpp"
#include "tensor.hpp"  // Tensor storage (rows=first dim); grad zeroing
#include "workspace.hpp"

namespace distribai {

// ---------------------------------------------------------------------------
// STensor: non-owning view over scratch memory (no heap, no grad buffer).
// Same interface surface as Tensor (rows/cols/data/size) so the templated
// kernels below work on both; numeric behavior is identical because the
// kernels are the single source of math for either type.
// ---------------------------------------------------------------------------
struct STensor {
  size_t rows = 0, cols = 0;
  float* data = nullptr;

  STensor() = default;
  STensor(size_t r, size_t c, Scratch& s) : rows(r), cols(c) {
    data = static_cast<float*>(s.alloc(r * c * sizeof(float)));
    // fall back to heap when the scratch is exhausted (never in practice)
    if (!data) heap_.reset(new float[r * c]);
  }
  float* raw() { return heap_ ? heap_.get() : data; }

  size_t size() const { return rows * cols; }
  float* gptr() { return nullptr; }

  // heap fallback ownership (transparent to kernels)
  std::unique_ptr<float[]> heap_;
};

// torch AdamW single-tensor step, in torch's op order:
//   p *= (1 - lr*wd)
//   m = b1*m + (1-b1)*g                      (lerp form)
//   v = b2*v + (1-b2)*g*g
//   denom = sqrt(v)/sqrt(bc2) + eps
//   p -= (lr/bc1) * m / denom
class ParityAdamW {
 public:
  ParityAdamW(std::vector<Tensor*> params, float lr, float wd = 0.01f,
              float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
      : params_(std::move(params)), lr_(lr), wd_(wd), b1_(b1), b2_(b2), eps_(eps) {
    for (auto* p : params_) {
      m_.emplace_back(p->rows, p->cols, 0.0f);
      v_.emplace_back(p->rows, p->cols, 0.0f);
    }
  }

  void zero_grad() {
    for (auto* p : params_) p->zero_grad();
  }

  // Checkpoint support: m/v buffers in parameters() order, plus the AdamW
  // step counter as an explicit integer (never packed into the floats - a
  // packed counter misaligns the v half and produced NaNs once).
  void export_state(std::vector<float>& m, std::vector<float>& v) const {
    m.clear();
    v.clear();
    for (const auto& t : m_) m.insert(m.end(), t.data.begin(), t.data.end());
    for (const auto& t : v_) v.insert(v.end(), t.data.begin(), t.data.end());
  }

  int64_t export_steps() const { return t_; }

  void import_state(const std::vector<float>& m, const std::vector<float>& v, int64_t steps) {
    if (m.size() != total_elems() || v.size() != total_elems()) return;  // shape mismatch
    size_t off = 0;
    for (auto& t : m_) {
      std::memcpy(t.data.data(), m.data() + off, t.data.size() * 4);
      off += t.data.size();
    }
    off = 0;
    for (auto& t : v_) {
      std::memcpy(t.data.data(), v.data() + off, t.data.size() * 4);
      off += t.data.size();
    }
    t_ = steps;
  }

  void step() {
    ++t_;
    const float f_t = static_cast<float>(t_);
    const float bc1 = 1.0f - std::pow(b1_, f_t);
    const float bc2s = std::sqrt(1.0f - std::pow(b2_, f_t));
    const float one_minus_lr_wd = 1.0f - lr_ * wd_;
    const float om_b1 = 1.0f - b1_;
    const float om_b2 = 1.0f - b2_;
    const float lr_over_bc1 = lr_ / bc1;
    for (size_t pi = 0; pi < params_.size(); ++pi) {
      float* __restrict__ p = params_[pi]->data.data();
      float* __restrict__ m = m_[pi].data.data();
      float* __restrict__ v = v_[pi].data.data();
      const float* __restrict__ g = params_[pi]->grad.data();
      const size_t n = params_[pi]->size();
      for (size_t i = 0; i < n; ++i) {
        p[i] *= one_minus_lr_wd;
        m[i] = b1_ * m[i] + om_b1 * g[i];
        v[i] = b2_ * v[i] + om_b2 * g[i] * g[i];
        const float denom = std::sqrt(v[i]) / bc2s + eps_;
        p[i] -= lr_over_bc1 * m[i] / denom;
      }
    }
  }

 private:
  size_t total_elems() const {
    size_t n = 0;
    for (const auto& t : m_) n += t.data.size();
    return n;
  }

  std::vector<Tensor*> params_;
  std::vector<Tensor> m_, v_;
  float lr_, wd_, b1_, b2_, eps_;
  int64_t t_ = 0;
};

// MLP 10-30-30-1 with torch (out,in) weight layout.
// Tensor(rows, cols): W1=(30,10) b1=(1,30) W2=(30,30) b2=(1,30) W3=(1,30) b3=(1,1)
class ParityMlp {
 public:
  explicit ParityMlp(uint64_t seed)
      : rng_(seed),
        W1_(30, 10), b1_(1, 30), W2_(30, 30), b2_(1, 30), W3_(1, 30), b3_(1, 1) {
    torch_linear_init(rng_, 10, 30, W1_.data.data(), b1_.data.data());
    torch_linear_init(rng_, 30, 30, W2_.data.data(), b2_.data.data());
    torch_linear_init(rng_, 30, 1, W3_.data.data(), b3_.data.data());
  }

  int64_t param_count() const { return 1291; }

  // y = relu(relu(x W1^T + b1) W2^T + b2) W3^T + b3, W (out,in): y[n,j] = b[j] + sum_k x[n,k] W[j,k]
  void forward(const Tensor& x, Tensor& h1, Tensor& a1, Tensor& h2, Tensor& a2,
               Tensor& out) const {
    dense_relu(x, W1_, b1_, h1, a1);
    dense_relu(a1, W2_, b2_, h2, a2);
    dense(a2, W3_, b3_, out);
  }

  // One parity training step: rand data, zero_grad, forward, MSE backward, AdamW.
  float train_step(Tensor& x, Tensor& y, ParityAdamW& opt) {
    opt.zero_grad();
    make_batch_parity(x, y, rng_);
    return train_step_fixed_data(x, y, opt);
  }

  // Same, but data already drawn (used to replay golden batches).
  // Zeroes grads like every torch training step: a driver that calls this
  // directly must not inherit stale gradients from the previous step (they
  // silently accumulate and break checkpoint-resume equality).
  //
  // Fast path (O1): all per-step buffers come from the thread scratch
  // (bump allocator, reset once per step) and backward writes grads directly
  // into param.grad buffers - no temp dW tensors, no acc() copy pass.
  // Numerics are identical: same kernels, same op order, same zeroed start
  // (a += d*x with dW zero-initialized equals grad = d*x; golden tests pin it).
  float train_step_fixed_data(const Tensor& x, const Tensor& y, ParityAdamW& opt) {
    opt.zero_grad();
    Scratch& s = thread_scratch();
    s.reset();
    const size_t n = x.rows;

    // forward activations (scratch)
    STensor h1(n, 30, s), a1(n, 30, s), h2(n, 30, s), a2(n, 30, s), out(n, 1, s);
    dense_raw(x.data.data(), W1_.data.data(), b1_.data.data(), h1.raw(), n, 10, 30);
    relu_raw(h1.raw(), a1.raw(), h1.size());
    dense_raw(a1.raw(), W2_.data.data(), b2_.data.data(), h2.raw(), n, 30, 30);
    relu_raw(h2.raw(), a2.raw(), h2.size());
    dense_raw(a2.raw(), W3_.data.data(), b3_.data.data(), out.raw(), n, 30, 1);

    // loss + dL/dout (raw: mean((out-y)^2), dout = 2(out-y)/n; verbatim)
    float loss = 0.0f;
    STensor dout(n, 1, s);
    {
      float sum = 0.0f;
      const size_t cnt = out.size();
      for (size_t i = 0; i < cnt; ++i) {
        const float d = out.raw()[i] - y.data[i];
        sum += d * d;
      }
      loss = sum / static_cast<float>(cnt);
      const float scale = 2.0f / static_cast<float>(cnt);
      for (size_t i = 0; i < cnt; ++i) dout.raw()[i] = (out.raw()[i] - y.data[i]) * scale;
    }

    // backward: grads land directly in param.grad (zeroed above).
    // da2/da1 are ACCUMULATED into by dense_backward (dx path), so they must
    // start zeroed; dh1/dh2 are assignments via relu_raw_grad.
    STensor da2(n, 30, s), dh2(n, 30, s), da1(n, 30, s), dh1(n, 30, s);
    std::memset(da2.raw(), 0, da2.size() * 4);
    std::memset(da1.raw(), 0, da1.size() * 4);
    // Dependency order preserved (da2 -> dh2 -> da1 -> dh1 -> W1 grads);
    // dx rows use the assignment-form kernel, weight grads use the combined
    // kernel without the dx branch.
    dense_backward_raw(a2.raw(), W3_.data.data(), dout.raw(), nullptr,
                       W3_.grad.data(), b3_.grad.data(), n, 30, 1);
    for (size_t i = 0; i < n; ++i)
      dense_dx_row_raw(W3_.data.data(), dout.raw() + i * 1, da2.raw() + i * 30, 30, 1);
    relu_raw_grad(h2.raw(), da2.raw(), dh2.raw(), h2.size());
    dense_backward_raw(a1.raw(), W2_.data.data(), dh2.raw(), nullptr,
                       W2_.grad.data(), b2_.grad.data(), n, 30, 30);
    for (size_t i = 0; i < n; ++i)
      dense_dx_row_raw(W2_.data.data(), dh2.raw() + i * 30, da1.raw() + i * 30, 30, 30);
    relu_raw_grad(h1.raw(), da1.raw(), dh1.raw(), h1.size());
    dense_backward_raw(x.data.data(), W1_.data.data(), dh1.raw(), nullptr,
                       W1_.grad.data(), b1_.grad.data(), n, 10, 30);

    opt.step();
    return loss;
  }

  // Flat grads in torch model.parameters() order: W1, b1, W2, b2, W3, b3.
  std::vector<float> flat_grads() const {
    std::vector<float> out;
    out.reserve(1291);
    for (const Tensor* t : {&W1_, &b1_, &W2_, &b2_, &W3_, &b3_})
      out.insert(out.end(), t->grad.begin(), t->grad.end());
    return out;
  }

  // Flat weights in the same order (for weight-level diffs).
  std::vector<float> flat_weights() const {
    std::vector<float> out;
    out.reserve(1291);
    for (const Tensor* t : {&W1_, &b1_, &W2_, &b2_, &W3_, &b3_})
      out.insert(out.end(), t->data.begin(), t->data.end());
    return out;
  }

  // Checkpoint restore: load flat weights in parameters() order.
  void load_flat_weights(const std::vector<float>& w) {
    if (w.size() != 1291) return;
    size_t off = 0;
    for (Tensor* t : {&W1_, &b1_, &W2_, &b2_, &W3_, &b3_}) {
      std::memcpy(t->data.data(), w.data() + off, t->size() * 4);
      off += t->size();
    }
  }

  TorchRng& rng() { return rng_; }

  std::vector<Tensor*> params() { return {&W1_, &b1_, &W2_, &b2_, &W3_, &b3_}; }

  // Public draw helper for drivers/tests that manage the loop themselves.
  void make_batch_external(Tensor& x, Tensor& y, TorchRng& rng) {
    make_batch_parity(x, y, rng);
  }


  // Baseline child pod: one full run of `steps` steps, seed fixed at ctor.
  struct RunResult {
    double wall_s = 0;
    std::vector<float> losses;
    std::vector<float> grad_flat;
  };

  RunResult run(int steps) {
    RunResult r;
    ParityAdamW opt(params(), 0.01f);
    Tensor x(64, 10), y(64, 1);
    r.losses.reserve(steps);
    const double t0 = now_();
    for (int s = 0; s < steps; ++s) {
      r.losses.push_back(train_step(x, y, opt));
    }
    r.wall_s = now_() - t0;
    r.grad_flat = flat_grads();
    return r;
  }

 private:
  // ---- raw-pointer kernels: the single source of math. Op order is verbatim
  // from the golden-verified implementation; Tensor and scratch STensor both
  // feed pointers into these, so the workspace cannot change numerics.
  // y = x W^T + b  (x: n x in, W: out x in, b: out)
  static void dense_raw(const float* __restrict__ x, const float* __restrict__ W,
                        const float* __restrict__ b, float* __restrict__ y,
                        const size_t n, const size_t in, const size_t out) {
    for (size_t i = 0; i < n; ++i) {
      const float* xr = x + i * in;
      float* yr = y + i * out;
      for (size_t j = 0; j < out; ++j) {
        const float* wr = W + j * in;
        float acc = b[j];
        for (size_t k = 0; k < in; ++k) acc += xr[k] * wr[k];
        yr[j] = acc;
      }
    }
  }

  static void relu_raw(const float* __restrict__ x, float* __restrict__ y, const size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
  }

  // ReLU backward: dx = dy where x > 0 else 0. Elementwise, so this is a
  // direct assignment (no zero-init needed, unlike the accumulating +=).
  static void relu_raw_grad(const float* __restrict__ x, const float* __restrict__ dy,
                            float* __restrict__ dx, const size_t n) {
    for (size_t i = 0; i < n; ++i) dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
  }

  // dy known: accumulate dW (out x in), db (out), and dx (n x in) when given.
  static void dense_backward_raw(const float* __restrict__ x, const float* __restrict__ W,
                                 const float* __restrict__ dy, float* __restrict__ dx,
                                 float* __restrict__ dW, float* __restrict__ db,
                                 const size_t n, const size_t in, const size_t out) {
    for (size_t i = 0; i < n; ++i) {
      const float* dyr = dy + i * out;
      const float* xr = x + i * in;
      for (size_t j = 0; j < out; ++j) {
        const float d = dyr[j];
        db[j] += d;
        float* dwr = dW + j * in;
        for (size_t k = 0; k < in; ++k) dwr[k] += d * xr[k];
      }
      if (dx) {
        float* dxr = dx + i * in;
        for (size_t j = 0; j < out; ++j) {
          const float d = dyr[j];
          const float* wr = W + j * in;
          for (size_t k = 0; k < in; ++k) dxr[k] += d * wr[k];
        }
      }
    }
  }

  // dx only: dx[i,k] = sum_j dy[i,j] W[j,k] for one row; assignment form
  // (dx row is fully overwritten). Same math as the dx branch above with the
  // accumulation identity removed because the buffer is per-step scratch.
  // NOTE: caller must guarantee dy covers the row (contiguous out floats).
  static void dense_dx_row_raw(const float* __restrict__ W, const float* __restrict__ dy,
                               float* __restrict__ dx, const size_t in, const size_t out) {
    for (size_t k = 0; k < in; ++k) dx[k] = 0.0f;
    for (size_t j = 0; j < out; ++j) {
      const float d = dy[j];
      const float* wr = W + j * in;
      for (size_t k = 0; k < in; ++k) dx[k] += d * wr[k];
    }
  }

  // legacy wrappers (kept for external callers; same math via raw kernels)
  static void dense(const Tensor& x, const Tensor& W, const Tensor& b, Tensor& y) {
    dense_raw(x.data.data(), W.data.data(), b.data.data(), y.data.data(), x.rows, x.cols, W.rows);
  }
  static void dense_relu(const Tensor& x, const Tensor& W, const Tensor& b, Tensor& h,
                         Tensor& a) {
    dense(x, W, b, h);
    relu_raw(h.data.data(), a.data.data(), h.size());
  }
  static void dense_backward(const Tensor& x, const Tensor& W, const Tensor& dy, Tensor* dx,
                             Tensor& dW, Tensor& db) {
    dense_backward_raw(x.data.data(), W.data.data(), dy.data.data(),
                       dx ? dx->data.data() : nullptr, dW.data.data(), db.data.data(),
                       x.rows, x.cols, W.rows);
  }

  static double now_() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  // Baseline contract: fresh x = rand(64,10) THEN y = rand(64,1) per step,
  // same stream as init (one generator for the whole child run).
  void make_batch_parity(Tensor& x, Tensor& y, TorchRng& rng) {
    rng.rand_n(x.data.data(), x.size());
    rng.rand_n(y.data.data(), y.size());
  }

  TorchRng rng_;
  Tensor W1_, b1_, W2_, b2_, W3_, b3_;
};

}  // namespace distribai
