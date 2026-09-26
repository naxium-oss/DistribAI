// distribai_train: train the 1K-param model, in-process or inside the
// native sandbox. Mirrors the Python harness contract:
//   --steps N --seed S --threads T [--mem-mb M] [--cpu-sec C]
//   [--sandbox]   run inside the native sandbox (own-system namespaces)
//   --json        emit a machine-readable result on stdout
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>

#include "../core/tensor.hpp"
#include "../sandbox/sandbox.hpp"
#include "../gpu/gpu_shim.hpp"

using namespace distribai;

struct TrainCfg {
  int steps = 1000;
  uint64_t seed = 42;
  int threads = 1;
  bool sandbox = false;
  bool json = false;
  SandboxLimits limits;
};

static TrainCfg parse(int argc, char** argv) {
  TrainCfg c;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--steps") c.steps = std::atoi(next());
    else if (a == "--seed") c.seed = std::strtoull(next(), nullptr, 10);
    else if (a == "--threads") c.threads = std::atoi(next());
    else if (a == "--sandbox") c.sandbox = true;
    else if (a == "--json") c.json = true;
    else if (a == "--mem-mb") c.limits.mem_mb = std::strtoull(next(), nullptr, 10);
    else if (a == "--cpu-sec") c.limits.cpu_sec = std::strtoull(next(), nullptr, 10);
  }
  return c;
}

// Full training run; returns a POD-safe result struct.
struct TrainOutcome {
  int status = 0;
  uint64_t param_count = 0;
  double wall_s = 0, steps_per_s = 0, ms_per_step = 0, final_loss = 0, first_loss = 0;
  char error[256] = {0};
};

static TrainOutcome run_training(int steps, uint64_t seed, int /*threads*/) {
  TrainOutcome out;
  out.param_count = static_cast<uint64_t>(Mlp1k(seed).param_count());
  Rng rng(seed);
  Mlp1k model(seed);
  AdamW opt(model.params(), 0.01f, 0.0f);
  Tensor x(64, 10), y(64, 1);
  make_batch(x, y, rng);  // initial batch
  float first = 0, last = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int s = 0; s < steps; ++s) {
    make_batch(x, y, rng);
    last = model.train_step(x, y, opt);
    if (s == 0) first = last;
  }
  const auto t1 = std::chrono::steady_clock::now();
  out.wall_s = std::chrono::duration<double>(t1 - t0).count();
  out.steps_per_s = steps / out.wall_s;
  out.ms_per_step = 1000.0 * out.wall_s / steps;
  out.final_loss = last;
  out.first_loss = first;
  return out;
}

int main(int argc, char** argv) {
  const TrainCfg cfg = parse(argc, argv);

  if (!cfg.sandbox) {
    const TrainOutcome o = run_training(cfg.steps, cfg.seed, cfg.threads);
    if (cfg.json) {
      std::printf(
          "{\"status\":%d,\"params\":%llu,\"wall_s\":%.4f,\"steps_per_s\":%.1f,"
          "\"ms_per_step\":%.3f,\"first_loss\":%.4f,\"final_loss\":%.4f,"
          "\"sandbox\":false,\"gpu\":%s}\n",
          o.status, static_cast<unsigned long long>(o.param_count), o.wall_s,
          o.steps_per_s, o.ms_per_step, o.first_loss, o.final_loss,
          gpu::cuda_available() ? "true" : "false");
    } else {
      std::printf(
          "params=%llu steps=%d wall=%.4fs steps/s=%.1f loss=%.4f\n",
          static_cast<unsigned long long>(o.param_count), cfg.steps, o.wall_s,
          o.steps_per_s, o.final_loss);
    }
    return o.status;
  }

  // ---- sandboxed run: child computes, parent reports ----
  const int steps = cfg.steps;
  const uint64_t seed = cfg.seed;
  const int threads = cfg.threads;

  TrainPod pod{};
  SandboxResult res = run_sandboxed(
      cfg.limits, static_cast<uint64_t>(cfg.limits.cpu_sec) + 30,
      [&](int wfd) {
        TrainPod p{};
        p.status = 0;
        p.steps = static_cast<uint64_t>(steps);
        const TrainOutcome o = run_training(steps, seed, threads);
        p.param_count = o.param_count;
        p.wall_s = o.wall_s;
        p.steps_per_s = o.steps_per_s;
        p.ms_per_step = o.ms_per_step;
        p.final_loss = o.final_loss;
        p.loss_first = o.first_loss;
        if (write(wfd, &p, sizeof(p)) != sizeof(p)) {
          p.status = 1;
          std::strncpy(p.error, "write failed", sizeof(p.error) - 1);
          (void)write(wfd, &p, sizeof(p));
        }
      },
      &pod, sizeof(pod));

  if (cfg.json) {
    std::printf(
        "{\"status\":%d,\"sandbox\":true,\"own_system\":true,\"timed_out\":%s,"
        "\"limit_killed\":%s,\"error\":\"%s\",\"spawn_join_s\":%.4f,"
        "\"wall_s\":%.4f,\"steps_per_s\":%.1f,\"ms_per_step\":%.3f,"
        "\"final_loss\":%.4f,\"first_loss\":%.4f,\"params\":%llu,"
        "\"rlimit_mem_mb\":%llu,\"rlimit_cpu_sec\":%llu,\"gpu\":%s}\n",
        res.ok ? 0 : 1, res.timed_out ? "true" : "false",
        res.limit_killed ? "true" : "false", res.error.c_str(), res.spawn_join_s,
        pod.wall_s, pod.steps_per_s, pod.ms_per_step, pod.final_loss,
        pod.loss_first, static_cast<unsigned long long>(pod.param_count),
        static_cast<unsigned long long>(cfg.limits.mem_mb),
        static_cast<unsigned long long>(cfg.limits.cpu_sec),
        gpu::cuda_available() ? "true" : "false");
    return res.ok ? 0 : 1;
  }
  std::printf(
      "sandbox ok=%d spawn=%.4fs steps/s=%.1f loss=%.4f err=%s\n",
      res.ok ? 1 : 0, res.spawn_join_s, pod.steps_per_s, pod.final_loss,
      res.error.c_str());
  return res.ok ? 0 : 1;
}
