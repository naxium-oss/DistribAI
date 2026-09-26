// distribai_dualrun: C++ side of the dual-run diff machine.
// Trains the parity model (same contract as baseline_1k_train.py's sandbox
// child) and emits the FULL Python envelope as JSON with identical keys and
// semantics. With --sandbox, the whole training run executes inside the
// native C++ sandbox (fork + rlimits + own-system namespaces + POD pipe),
// mirroring how the Python reference runs inside its forked rlimit child.
//
// Usage: distribai_dualrun --seed 42 --steps 200 [--sandbox] [--mem-mb 4096]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include "../core/parity.hpp"
#include "../sandbox/sandbox.hpp"

using namespace distribai;

static double steady_now_() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static void print_envelope(const TrainPod& p) {
  auto r6 = [](double v) { return std::round(v * 1e6) / 1e6; };
  std::printf("{");
  std::printf("\"status\": \"%s\", ", p.status == 0 ? "ok" : "error");
  std::printf("\"n_params\": %lld, ", static_cast<long long>(p.param_count));
  std::printf("\"wall_s\": %.6f, ", p.wall_s);
  std::printf("\"steps_per_s\": %.1f, ", p.steps_per_s);
  std::printf("\"ms_per_step\": %.3f, ", p.ms_per_step);
  std::printf("\"final_loss\": %.6f, ", r6(p.final_loss));
  std::printf("\"first5_losses\": [");
  for (int i = 0; i < 5; ++i) std::printf("%.6f%s", r6(p.loss_first5[i]), i < 4 ? ", " : "");
  std::printf("], ");
  std::printf("\"grad_len\": %zu, ", static_cast<size_t>(p.grad_len));
  std::printf("\"grad_sum\": %.17g, ", p.grad_sum);
  std::printf("\"grad_first3\": [%.6f, %.6f, %.6f]", r6(p.grad_first3[0]), r6(p.grad_first3[1]),
              r6(p.grad_first3[2]));
  std::printf("}\n");
}

static void run_child_train(int wfd, uint64_t seed, int steps) {
  TrainPod p{};
  ParityMlp model(seed);
  ParityAdamW opt(model.params(), 0.01f);
  Tensor x(64, 10), y(64, 1);
  float losses[200];
  const bool clamped = steps <= 200;
  std::vector<float> loss_vec;
  loss_vec.reserve(steps);
  const double t0 = steady_now_();
  for (int s = 0; s < steps; ++s) loss_vec.push_back(model.train_step(x, y, opt));
  const double wall = steady_now_() - t0;

  const auto grads = model.flat_grads();
  double gsum = 0;
  for (float v : grads) gsum += v;

  p.status = 0;
  p.steps = static_cast<uint64_t>(steps);
  p.param_count = static_cast<uint64_t>(model.param_count());
  p.grad_len = grads.size();
  p.wall_s = wall;
  p.steps_per_s = steps / wall;
  p.ms_per_step = 1000.0 * wall / steps;
  p.final_loss = loss_vec.back();
  for (int i = 0; i < 5; ++i) p.loss_first5[i] = loss_vec[i];
  p.grad_sum = gsum;
  for (int i = 0; i < 3; ++i) p.grad_first3[i] = grads[i];
  (void)clamped;
  (void)losses;
  (void)write(wfd, &p, sizeof(p));
}

int main(int argc, char** argv) {
  uint64_t seed = 42;
  int steps = 200;
  bool sandbox = false;
  SandboxLimits lim;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&] { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--seed") seed = std::strtoull(nx(), nullptr, 10);
    else if (a == "--steps") steps = std::atoi(nx());
    else if (a == "--sandbox") sandbox = true;
    else if (a == "--mem-mb") lim.mem_mb = std::strtoull(nx(), nullptr, 10);
  }

  if (!sandbox) {
    // In-process run: build the pod locally, print from it (same code path
    // as the sandboxed child, so both modes are contract-identical).
    TrainPod p{};
    ParityMlp model(seed);
    ParityAdamW opt(model.params(), 0.01f);
    Tensor x(64, 10), y(64, 1);
    std::vector<float> loss_vec;
    loss_vec.reserve(steps);
    const double t0 = steady_now_();
    for (int s = 0; s < steps; ++s) loss_vec.push_back(model.train_step(x, y, opt));
    const double wall = steady_now_() - t0;
    const auto grads = model.flat_grads();
    double gsum = 0;
    for (float v : grads) gsum += v;
    p.status = 0;
    p.steps = static_cast<uint64_t>(steps);
    p.param_count = static_cast<uint64_t>(model.param_count());
    p.grad_len = grads.size();
    p.wall_s = wall;
    p.steps_per_s = steps / wall;
    p.ms_per_step = 1000.0 * wall / steps;
    p.final_loss = loss_vec.back();
    for (int i = 0; i < 5; ++i) p.loss_first5[i] = loss_vec[i];
    p.grad_sum = gsum;
    for (int i = 0; i < 3; ++i) p.grad_first3[i] = grads[i];
    print_envelope(p);
    return 0;
  }

  // Sandboxed run: child trains under rlimits, parent prints the envelope.
  TrainPod pod{};
  SandboxResult res = run_sandboxed(
      lim, lim.cpu_sec + 30, [&](int wfd) { run_child_train(wfd, seed, steps); }, &pod,
      sizeof(pod));
  if (!res.ok) {
    std::printf("{\"status\": \"error\", \"error\": \"%s\"}\n", res.error.c_str());
    return 1;
  }
  print_envelope(pod);
  return 0;
}
