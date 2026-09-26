// golden_cases (suite 1): deterministic runner for the golden-50 matrix.
// One JSON line on stdout:
//   {"case":N,"digest":"hex8","losses":[...],"grad_len":K,"grad_first3":[...],
//    "init_w1_first8":[...],"params":P,"final_loss":x,"loss_first":x}
// Modes:
//   --sandbox 0  compute in-process
//   --sandbox 1  compute inside one run_sandboxed child (fork+rlimits+ns)
//   --sandbox 2  two concurrent sandboxes, SAME config: digests must agree
// hidden=0 is the 1-param model (w * x, MSE) mirroring Python train_scaling.
// hidden>0 is GenericMlp (1-H-H-1) from core/generic_mlp.hpp.
// Digest: FNV-1a over the loss trajectory's raw f32 bits (bit-exactness gate).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "core/generic_mlp.hpp"
#include "sandbox/sandbox.hpp"

using namespace distribai;

// Must mirror golden_cases.py ADAM_GRID.
static void adam_settings(int idx, float& lr, float& wd) {
  switch (idx) {
    case 0: lr = 0.01f; wd = 0.01f; break;
    case 1: lr = 0.001f; wd = 0.0f; break;
    case 2: lr = 0.05f; wd = 0.05f; break;
    default: lr = 0.01f; wd = 0.0f; break;
  }
}

static uint32_t fnv1a(const float* data, size_t n) {
  uint32_t h = 2166136261u;
  const auto* p = reinterpret_cast<const uint8_t*>(data);
  for (size_t i = 0; i < n * sizeof(float); ++i) {
    h ^= p[i];
    h *= 16777619u;
  }
  return h;
}

struct CaseOut {
  std::vector<float> losses;
  uint32_t digest = 0;
  float final_loss = 0, loss_first = 0;
  std::vector<float> init_w1;    // first 8 weight values (1 for hidden=0)
  std::vector<float> grad_first; // first 3 grads of the first param tensor
  int64_t params = 0;
};

// The shared computation (identical code path in-process and in children).
static CaseOut compute_case(int hidden, uint64_t seed, int steps, int adam_idx) {
  float lr, wd;
  adam_settings(adam_idx, lr, wd);
  CaseOut out;
  auto batch1 = [](Tensor& x, Tensor& y, Rng& rng) {
    x.data[0] = rng.next_float();
    const float t = x.data[0];
    y.data[0] = std::sin(3.14159265f * t) + 0.5f * std::sin(3.0f * 3.14159265f * t);
  };

  if (hidden == 0) {
    // 1-param model: w starts at 1 (torch ones(1)), pred = w * x, MSE.
    Tensor w(1, 1);
    w.data[0] = 1.0f;
    AdamW opt({&w}, lr, wd);
    Rng rng(seed);
    Tensor x(1, 1), y(1, 1);
    out.params = 1;
    out.init_w1 = {w.data[0]};
    for (int s = 0; s < steps; ++s) {
      batch1(x, y, rng);
      opt.zero_grad();
      const float pred = w.data[0] * x.data[0];
      const float d = pred - y.data[0];
      const float loss = d * d;  // MSE mean over 1
      w.grad[0] = 2.0f * d * x.data[0];
      opt.step();
      out.losses.push_back(loss);
    }
    out.final_loss = out.losses.back();
    out.loss_first = out.losses.front();
    // first param tensor has exactly 1 grad; pad the 3-slot wire format
    out.grad_first = {w.grad[0], 0.0f, 0.0f};
  } else {
    GenericMlp model(static_cast<size_t>(hidden), seed);
    AdamW opt(model.params(), lr, wd);
    Rng rng(seed);
    Tensor x(1, 1), y(1, 1);
    out.params = model.param_count();
    const auto& w1 = model.params()[0]->data;
    for (int i = 0; i < 8 && i < static_cast<int>(w1.size()); ++i) out.init_w1.push_back(w1[i]);
    for (int s = 0; s < steps; ++s) {
      batch1(x, y, rng);
      const float loss = model.train_step(x, y, opt);
      out.losses.push_back(loss);
    }
    out.final_loss = out.losses.back();
    out.loss_first = out.losses.front();
    const auto& g = model.params()[0]->grad;
    for (int i = 0; i < 3 && i < static_cast<int>(g.size()); ++i) out.grad_first.push_back(g[i]);
  }
  out.digest = fnv1a(out.losses.data(), out.losses.size());
  return out;
}

// Blob layout after the POD: losses f32 * n | digest u32 | init_w1 f32 * k | grad f32 * 3.
static void write_case_payload(int wfd, const CaseOut& c) {
  TrainPod p{};
  p.status = 0;
  p.steps = c.losses.size();
  p.param_count = static_cast<uint64_t>(c.params);
  p.final_loss = c.final_loss;
  p.loss_first = c.loss_first;
  const size_t blob = c.losses.size() * 4 + 4 + c.init_w1.size() * 4 + c.grad_first.size() * 4;
  p.grad_blob_len = blob;
  (void)write(wfd, &p, sizeof(p));
  if (!c.losses.empty()) (void)write(wfd, c.losses.data(), c.losses.size() * 4);
  (void)write(wfd, &c.digest, 4);
  if (!c.init_w1.empty()) (void)write(wfd, c.init_w1.data(), c.init_w1.size() * 4);
  if (!c.grad_first.empty()) (void)write(wfd, c.grad_first.data(), c.grad_first.size() * 4);
}

int main(int argc, char** argv) {
  int case_id = 0, hidden = 30, steps = 100, adam = 0, sandbox = 1;
  uint64_t seed = 42;
  for (int i = 1; i < argc; ++i) {
    auto val = [&](int& v) { v = std::atoi(argv[++i]); };
    if (!std::strcmp(argv[i], "--case")) val(case_id);
    else if (!std::strcmp(argv[i], "--hidden")) val(hidden);
    else if (!std::strcmp(argv[i], "--steps")) val(steps);
    else if (!std::strcmp(argv[i], "--adam")) val(adam);
    else if (!std::strcmp(argv[i], "--sandbox")) val(sandbox);
    else if (!std::strcmp(argv[i], "--seed")) seed = std::strtoull(argv[++i], nullptr, 10);
  }

  // In-process mode: compute directly.
  if (sandbox == 0) {
    const CaseOut c = compute_case(hidden, seed, steps, adam);
    std::printf("{\"case\":%d,\"digest\":\"%08x\",\"losses\":[", case_id, c.digest);
    for (size_t i = 0; i < c.losses.size(); ++i)
      std::printf("%s%.9g", i ? "," : "", static_cast<double>(c.losses[i]));
    std::printf("],\"grad_len\":%lld,\"grad_first3\":[", static_cast<long long>(c.params));
    for (size_t i = 0; i < c.grad_first.size(); ++i)
      std::printf("%s%.9g", i ? "," : "", static_cast<double>(c.grad_first[i]));
    std::printf("],\"init_w1_first8\":[");
    for (size_t i = 0; i < c.init_w1.size(); ++i)
      std::printf("%s%.9g", i ? "," : "", static_cast<double>(c.init_w1[i]));
    std::printf("],\"params\":%lld,\"final_loss\":%.9g,\"loss_first\":%.9g}\n",
                static_cast<long long>(c.params), static_cast<double>(c.final_loss),
                static_cast<double>(c.loss_first));
    return 0;
  }

  // Sandbox mode(s): children compute, parent prints.
  auto run_one = [&](TrainPod& pod, std::vector<uint8_t>& blob) {
    SandboxLimits lim;
    return run_sandboxed(
        lim, lim.cpu_sec + 30,
        [&](int wfd) { write_case_payload(wfd, compute_case(hidden, seed, steps, adam)); },
        &pod, sizeof(pod), &blob);
  };

  auto unpack = [&](const TrainPod& pod, const std::vector<uint8_t>& blob, CaseOut& c) -> bool {
    size_t off = 0;
    auto take = [&](void* dst, size_t n) {
      if (off + n > blob.size()) return false;
      std::memcpy(dst, blob.data() + off, n);
      off += n;
      return true;
    };
    c.losses.resize(pod.steps);
    if (!take(c.losses.data(), pod.steps * 4)) return false;
    if (!take(&c.digest, 4)) return false;
    const size_t init_n = hidden == 0 ? 1 : (hidden < 8 ? hidden : 8);
    c.init_w1.resize(init_n);
    if (!take(c.init_w1.data(), init_n * 4)) return false;
    c.grad_first.resize(3);
    if (!take(c.grad_first.data(), 3 * 4)) return false;
    c.params = static_cast<int64_t>(pod.param_count);
    c.final_loss = pod.final_loss;
    c.loss_first = pod.loss_first;
    return true;
  };

  auto emit = [&](const CaseOut& c) {
    std::printf("{\"case\":%d,\"digest\":\"%08x\",\"losses\":[", case_id, c.digest);
    for (size_t i = 0; i < c.losses.size(); ++i)
      std::printf("%s%.9g", i ? "," : "", static_cast<double>(c.losses[i]));
    std::printf("],\"grad_len\":%lld,\"grad_first3\":[", static_cast<long long>(c.params));
    for (size_t i = 0; i < c.grad_first.size(); ++i)
      std::printf("%s%.9g", i ? "," : "", static_cast<double>(c.grad_first[i]));
    std::printf("],\"init_w1_first8\":[");
    for (size_t i = 0; i < c.init_w1.size(); ++i)
      std::printf("%s%.9g", i ? "," : "", static_cast<double>(c.init_w1[i]));
    std::printf("],\"params\":%lld,\"final_loss\":%.9g,\"loss_first\":%.9g}\n",
                static_cast<long long>(c.params), static_cast<double>(c.final_loss),
                static_cast<double>(c.loss_first));
  };

  TrainPod pa{}, pb{};
  std::vector<uint8_t> blob_a, blob_b;
  SandboxResult ra = run_one(pa, blob_a);
  if (!ra.ok) {
    std::fprintf(stderr, "sandbox child A failed: %s\n", ra.error.c_str());
    return 1;
  }
  CaseOut ca;
  if (!unpack(pa, blob_a, ca)) {
    std::fprintf(stderr, "payload unpack failed\n");
    return 1;
  }
  if (sandbox == 2) {
    SandboxResult rb;
    std::thread tb([&] { rb = run_one(pb, blob_b); });
    // rerun A concurrently: two SAME-config sandboxes at once
    TrainPod pa2{};
    std::vector<uint8_t> blob_a2;
    SandboxResult ra2;
    std::thread ta([&] { ra2 = run_one(pa2, blob_a2); });
    tb.join();
    ta.join();
    if (!rb.ok || !ra2.ok) {
      std::fprintf(stderr, "sandbox child failed in 2x mode\n");
      return 1;
    }
    CaseOut cb;
    if (!unpack(pb, blob_b, cb)) {
      std::fprintf(stderr, "payload unpack failed (B)\n");
      return 1;
    }
    if (cb.digest != ca.digest) {
      std::fprintf(stderr, "2x sandbox digest mismatch: %08x vs %08x\n", ca.digest, cb.digest);
      return 1;
    }
  }
  emit(ca);
  return 0;
}
