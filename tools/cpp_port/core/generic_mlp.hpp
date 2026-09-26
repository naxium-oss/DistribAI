// GenericMlp (improvement O4): parameterized-width 1-H-H-1 MLP mirroring the
// Python train_scaling variants (10K / 100K param models). Same math and
// float op order as the Tensor-path kernels in core/tensor.hpp (matmul_add /
// matmul_add_backward / relu / mse / AdamW step), but per-step activations and
// delta buffers come from the thread scratch (O1) and weight grads are written
// directly into param.grad, with no temp dW tensors and no acc() copy pass.
//
// Weight layout matches Mlp1k's (in, out) rows: W1=(1,H) W2=(H,H) W3=(H,1),
// biases (1, out). Numerics are bit-identical to the GenericMlp shape run
// through the Tensor path (build/prof/prof10 compiles both; losses match
// bit-for-bit at 1k/10k/100k, and the kernel-level bit-exactness argument is
// the same as ParityMlp's O1 note: a += d*x into a zeroed cell equals the
// temp-then-accumulate form, and every loop's per-cell addition order is
// unchanged).
#pragma once

#include <cmath>
#include <cstring>
#include <vector>

#include "tensor.hpp"
#include "workspace.hpp"

namespace distribai {

class GenericMlp {
 public:
  explicit GenericMlp(size_t hidden, uint64_t seed = 7)
      : H(hidden), rng_(seed), W1_(1, H), b1_(1, H), W2_(H, H), b2_(1, H), W3_(H, 1), b3_(1, 1) {
    init(W1_, b1_);
    init(W2_, b2_);
    init(W3_, b3_);
  }

  int64_t param_count() const {
    return static_cast<int64_t>(W1_.size() + b1_.size() + W2_.size() + b2_.size() + W3_.size() +
                                b3_.size());
  }

  std::vector<Tensor*> params() { return {&W1_, &b1_, &W2_, &b2_, &W3_, &b3_}; }

  // Same loop shape as Mlp1k::train_step / Python: zero_grad, forward, MSE,
  // backward, step. Batch here is (n, 1); callers use n=1 to mirror
  // train_scaling, but any n works.
  float train_step(const Tensor& x, const Tensor& y, AdamW& opt, float* loss_out = nullptr) {
    opt.zero_grad();
    Scratch& s = thread_scratch();
    s.reset();
    const size_t n = x.rows;

    // forward (scratch-backed)
    SBuf h1(n, H, s), a1(n, H, s), h2(n, H, s), a2(n, H, s), out(n, 1, s);
    fwd(x.data.data(), W1_.data.data(), b1_.data.data(), h1.p, a1.p, n, 1, H);
    fwd(a1.p, W2_.data.data(), b2_.data.data(), h2.p, a2.p, n, H, H);
    dense_only(a2.p, W3_.data.data(), b3_.data.data(), out.p, n, H, 1);

    // loss + dL/dout (same form as mse_loss)
    float loss = 0.0f;
    SBuf dout(n, 1, s);
    {
      float sum = 0.0f;
      for (size_t i = 0; i < n; ++i) {
        const float d = out.p[i] - y.data[i];
        sum += d * d;
      }
      loss = sum / static_cast<float>(n);
      const float scale = 2.0f / static_cast<float>(n);
      for (size_t i = 0; i < n; ++i) dout.p[i] = (out.p[i] - y.data[i]) * scale;
    }

    // backward: grads land directly in param.grad (zeroed above).
    // da2/da1 accumulate the dx contribution (dxr[k] += ...), so zero them.
    SBuf da2(n, H, s), dh2(n, H, s), da1(n, H, s), dh1(n, H, s);
    std::memset(da2.p, 0, n * H * 4);
    std::memset(da1.p, 0, n * H * 4);

    // W3 row: out=1
    bwd(a2.p, W3_.data.data(), dout.p, nullptr, W3_.grad.data(), b3_.grad.data(), n, H, 1);
    dx_rows(W3_.data.data(), dout.p, da2.p, n, H, 1);
    relu_grad(h2.p, da2.p, dh2.p, n * H);
    bwd(a1.p, W2_.data.data(), dh2.p, nullptr, W2_.grad.data(), b2_.grad.data(), n, H, H);
    dx_rows(W2_.data.data(), dh2.p, da1.p, n, H, H);
    relu_grad(h1.p, da1.p, dh1.p, n * H);
    bwd(x.data.data(), W1_.data.data(), dh1.p, nullptr, W1_.grad.data(), b1_.grad.data(), n, 1, H);

    opt.step();
    if (loss_out) *loss_out = loss;
    return loss;
  }

  const size_t H;

 private:
  // scratch-backed float buffer with heap fallback (same contract as STensor)
  struct SBuf {
    SBuf(size_t r, size_t c, Scratch& s) : rows(r), cols(c) {
      p = static_cast<float*>(s.alloc(r * c * sizeof(float)));
      if (!p) heap_.reset(new float[r * c]), p = heap_.get();
    }
    size_t rows, cols;
    float* p = nullptr;
    std::unique_ptr<float[]> heap_;
  };

  // These statics mirror the exact loop bodies of tensor.hpp's matmul_add /
  // matmul_add_backward (post-fix, incl. the k-unroll and the 2-row unrolls)
  // with raw pointers so scratch buffers can feed them.

  // y = x W + b, then relu -> (h pre-activation, a activation)
  static void fwd(const float* __restrict__ x, const float* __restrict__ W,
                  const float* __restrict__ b, float* __restrict__ h, float* __restrict__ a,
                  const size_t n, const size_t in, const size_t out) {
    for (size_t i = 0; i < n; ++i) {
      float* hr = &h[i * out];
      const float* xr = &x[i * in];
      std::memcpy(hr, b, out * sizeof(float));
      size_t k = 0;
      for (; k + 2 <= in; k += 2) {
        const float xv0 = xr[k], xv1 = xr[k + 1];
        const float* wr0 = &W[k * out];
        const float* wr1 = &W[(k + 1) * out];
        for (size_t j = 0; j < out; ++j) {
          hr[j] += xv0 * wr0[j];
          hr[j] += xv1 * wr1[j];
        }
      }
      if (k < in) {
        const float xv = xr[k];
        const float* wr = &W[k * out];
        for (size_t j = 0; j < out; ++j) hr[j] += xv * wr[j];
      }
      for (size_t j = 0; j < out; ++j) a[i * out + j] = hr[j] > 0.0f ? hr[j] : 0.0f;
    }
  }

  // y = x W + b without relu (last layer)
  static void dense_only(const float* __restrict__ x, const float* __restrict__ W,
                         const float* __restrict__ b, float* __restrict__ y, const size_t n,
                         const size_t in, const size_t out) {
    for (size_t i = 0; i < n; ++i) {
      float* yr = &y[i * out];
      const float* xr = &x[i * in];
      std::memcpy(yr, b, out * sizeof(float));
      size_t k = 0;
      for (; k + 2 <= in; k += 2) {
        const float xv0 = xr[k], xv1 = xr[k + 1];
        const float* wr0 = &W[k * out];
        const float* wr1 = &W[(k + 1) * out];
        for (size_t j = 0; j < out; ++j) {
          yr[j] += xv0 * wr0[j];
          yr[j] += xv1 * wr1[j];
        }
      }
      if (k < in) {
        const float xv = xr[k];
        const float* wr = &W[k * out];
        for (size_t j = 0; j < out; ++j) yr[j] += xv * wr[j];
      }
    }
  }

  // backward for one dense layer (no dx): dW(out-rows) += dy^T x, db += dy
  static void bwd(const float* __restrict__ x, const float* __restrict__ W,
                  const float* __restrict__ dy, float* __restrict__ dx, float* __restrict__ dW,
                  float* __restrict__ db, const size_t n, const size_t in, const size_t out) {
    (void)W;
    (void)dx;
    for (size_t i = 0; i < n; ++i) {
      const float* dyr = &dy[i * out];
      const float* xr = &x[i * in];
      for (size_t j = 0; j < out; ++j) db[j] += dyr[j];
      size_t k = 0;
      for (; k + 2 <= in; k += 2) {
        const float xv0 = xr[k], xv1 = xr[k + 1];
        float* dwr0 = &dW[k * out];
        float* dwr1 = &dW[(k + 1) * out];
        for (size_t j = 0; j < out; ++j) {
          const float d = dyr[j];
          dwr0[j] += xv0 * d;
          dwr1[j] += xv1 * d;
        }
      }
      for (; k < in; ++k) {
        const float xv = xr[k];
        float* dwr = &dW[k * out];
        for (size_t j = 0; j < out; ++j) dwr[j] += xv * dyr[j];
      }
    }
  }

  // dx rows for the hidden layers: dxr[k] += sum_j dy[j] W(k,j) per row
  // (matches matmul_add_backward's dx branch, in(out) layout).
  static void dx_rows(const float* __restrict__ W, const float* __restrict__ dy,
                      float* __restrict__ dx, const size_t n, const size_t in,
                      const size_t out) {
    for (size_t i = 0; i < n; ++i) {
      const float* dyr = &dy[i * out];
      float* dxr = &dx[i * in];
      size_t k = 0;
      for (; k + 2 <= in; k += 2) {
        const float* wr0 = &W[k * out];
        const float* wr1 = &W[(k + 1) * out];
        float v0 = dxr[k], v1 = dxr[k + 1];
        for (size_t j = 0; j < out; ++j) {
          const float d = dyr[j];
          v0 += d * wr0[j];
          v1 += d * wr1[j];
        }
        dxr[k] = v0;
        dxr[k + 1] = v1;
      }
      for (; k < in; ++k) {
        const float* wr = &W[k * out];
        float v = dxr[k];
        for (size_t j = 0; j < out; ++j) v += dyr[j] * wr[j];
        dxr[k] = v;
      }
    }
  }

  static void relu_grad(const float* __restrict__ h, const float* __restrict__ dy,
                        float* __restrict__ dx, const size_t n) {
    for (size_t i = 0; i < n; ++i) dx[i] = h[i] > 0.0f ? dy[i] : 0.0f;
  }

  void init(Tensor& W, Tensor& b) {
    const float fan_in = static_cast<float>(W.rows);
    const float stdv = std::sqrt(1.0f / fan_in);
    for (auto& v : W.data) v = rng_.next_normal() * stdv;
    std::fill(b.data.begin(), b.data.end(), 0.0f);
  }

  Rng rng_;
  Tensor W1_, b1_, W2_, b2_, W3_, b3_;
};

}  // namespace distribai
