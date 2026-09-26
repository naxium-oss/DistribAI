// Parity + correctness tests for the native core.
// Verifies the ported training path actually learns the task and matches
// the Python baseline's contract (loss magnitude, convergence, grad flow).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "../core/tensor.hpp"

using namespace distribai;

static int failures = 0;
#define CHECK(cond, msg)                                     \
  do {                                                       \
    if (!(cond)) {                                           \
      ++failures;                                            \
      std::printf("FAIL: %s (line %d)\n", msg, __LINE__);    \
    }                                                        \
  } while (0)

int main() {
  // ---- 1. Param count matches the Python sandbox model ----
  {
    Mlp1k m(42);
    const int64_t n = m.param_count();
    std::printf("param_count = %lld\n", static_cast<long long>(n));
    CHECK(n == 1291, "model has exactly 1,291 params (10-30-30-1)");
  }

  // ---- 2. Determinism: same seed -> identical loss trajectory ----
  {
    Rng rng_a(7), rng_b(7);
    Tensor xa(64, 10), ya(64, 1), xb(64, 10), yb(64, 1);
    make_batch(xa, ya, rng_a);
    make_batch(xb, yb, rng_b);
    bool same_data = true;
    for (size_t i = 0; i < xa.size(); ++i)
      if (xa.data[i] != xb.data[i]) { same_data = false; break; }
    CHECK(same_data, "same seed yields identical batches");

    Mlp1k ma(7), mb(7);
    AdamW oa(ma.params(), 0.01f), ob(mb.params(), 0.01f);
    float la = 0, lb = -1;
    for (int s = 0; s < 25; ++s) {
      la = ma.train_step(xa, ya, oa);
      lb = mb.train_step(xb, yb, ob);
    }
    CHECK(std::fabs(la - lb) < 1e-9f, "same seed -> identical loss trajectory");
    std::printf("determinism: step25 loss=%.6f\n", la);
  }

  // ---- 3. Convergence: loss drops substantially over 400 steps ----
  {
    Rng rng(123);
    Mlp1k m(123);
    AdamW opt(m.params(), 0.01f);
    Tensor x(64, 10), y(64, 1);
    make_batch(x, y, rng);
    float first = 0, last = 0;
    for (int s = 0; s < 400; ++s) {
      make_batch(x, y, rng);
      last = m.train_step(x, y, opt);
      if (s == 0) first = last;
    }
    std::printf("convergence: first=%.4f last=%.4f\n", first, last);
    CHECK(first > 0.05f, "initial loss in expected magnitude");
    CHECK(last < first * 0.5f, "loss drops >50% over 400 steps");
    CHECK(last < 0.09f, "final loss in the Python baseline band (<=0.09)");
  }

  // ---- 4. Gradient flow: all parameter grads nonzero ----
  {
    Rng rng(9);
    Mlp1k m(9);
    AdamW opt(m.params(), 0.01f);
    Tensor x(64, 10), y(64, 1);
    make_batch(x, y, rng);
    // train_step zeroes via opt.zero_grad? It does not; do a manual step:
    opt.zero_grad();
    m.train_step(x, y, opt);
    bool all_nonzero = true;
    for (auto* p : m.params()) {
      bool any = false;
      for (size_t i = 0; i < p->size(); ++i)
        if (p->grad[i] != 0.0f) { any = true; break; }
      if (!any) { all_nonzero = false; break; }
    }
    CHECK(all_nonzero, "every parameter tensor receives gradient");
  }

  // ---- 5. AdamW step actually updates weights ----
  {
    Rng rng(11);
    Mlp1k m(11);
    AdamW opt(m.params(), 0.01f);
    Tensor x(4, 10), y(4, 1);
    make_batch(x, y, rng);
    std::vector<float> before(m.W1().size());
    for (size_t i = 0; i < m.W1().size(); ++i) before[i] = m.W1().data[i];
    opt.zero_grad();
    m.train_step(x, y, opt);
    float max_delta = 0;
    for (size_t i = 0; i < m.W1().size(); ++i)
      max_delta = std::max(max_delta, std::fabs(before[i] - m.W1().data[i]));
    std::printf("adamw max_weight_delta=%.6f\n", max_delta);
    CHECK(max_delta > 1e-5f, "weights move after an AdamW step");
  }

  if (failures == 0) {
    std::printf("ALL PARITY TESTS PASSED\n");
    return 0;
  }
  std::printf("%d FAILURES\n", failures);
  return 1;
}
