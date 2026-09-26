// suite_run.cpp - "run" category: continuous train+sandbox stress under the
// mandatory limits, checking the operational invariants:
//   * FDs do not leak across many fork/spawn cycles
//   * threads do not accumulate
//   * RSS stays under a ceiling for the whole run
//   * sandbox spawn cost does not degrade over time (zombie/reap health)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>

#include "../framework.hpp"
#include "core/generic_mlp.hpp"
#include "sandbox/sandbox.hpp"

using namespace distribai;

static long scan_status_line(const char* key) {
  FILE* f = std::fopen("/proc/self/status", "r");
  if (!f) return -1;
  char line[256];
  long val = -1;
  while (std::fgets(line, sizeof(line), f)) {
    if (std::strncmp(line, key, std::strlen(key)) == 0) {
      val = std::strtol(line + std::strlen(key), nullptr, 10);
      break;
    }
  }
  std::fclose(f);
  return val;
}

static long count_fds() {
  DIR* d = opendir("/proc/self/fd");
  if (!d) return -1;
  long n = 0;
  while (readdir(d)) ++n;
  closedir(d);
  return n - 3;  // ., .., dirfd
}

// vmem RSS in kB.
static long rss_kb() { return scan_status_line("VmRSS:"); }
static long threads_now() { return scan_status_line("Threads:"); }

int main(int argc, char** argv) {
  int seconds = 45;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = std::atoi(argv[++i]);
  }

  suite::section("run: continuous train+sandbox stress");
  const long fd0 = count_fds();
  const long th0 = threads_now();
  long rss_peak = 0;
  std::vector<double> spawn_walls;

  TrainPod pod{};
  int train_iters = 0, sandbox_iters = 0, failures = 0;
  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
    // (a) in-process training burst
    {
      GenericMlp model(30, 42);
      AdamW opt(model.params(), 0.01f, 0.0f);
      Rng rng(42);
      Tensor x(1, 1), y(1, 1);
      for (int s = 0; s < 20; ++s) {
        x.data[0] = rng.next_float();
        y.data[0] = std::sin(3.14159265f * x.data[0]);
        if (!std::isfinite(model.train_step(x, y, opt))) ++failures;
      }
      ++train_iters;
    }
    // (b) sandboxed training pod
    {
      SandboxLimits lim;
      lim.mem_mb = 128;
      const auto s0 = std::chrono::steady_clock::now();
      const SandboxResult res = run_sandboxed(
          lim, lim.cpu_sec + 30,
          [](int wfd) {
            GenericMlp model(30, 43);
            AdamW opt(model.params(), 0.01f, 0.0f);
            Rng rng(43);
            Tensor x(1, 1), y(1, 1);
            float loss = 0;
            for (int s = 0; s < 20; ++s) {
              x.data[0] = rng.next_float();
              y.data[0] = std::sin(3.14159265f * x.data[0]);
              loss = model.train_step(x, y, opt);
            }
            TrainPod p{};
            p.status = 0;
            p.final_loss = loss;
            (void)write(wfd, &p, sizeof(p));
          },
          &pod, sizeof(pod));
      const double wall =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
      spawn_walls.push_back(wall);
      if (!res.ok || pod.status != 0) ++failures;
      ++sandbox_iters;
    }
    if (const long r = rss_kb(); r > rss_peak) rss_peak = r;
  }

  const long fd1 = count_fds();
  const long th1 = threads_now();

  std::printf("run: %d train bursts, %d sandbox spawns, %d failures\n", train_iters,
              sandbox_iters, failures);
  CHECK(train_iters > 3 && sandbox_iters > 3, "run: loop actually exercised both paths");
  CHECK(failures == 0, "run: no failed iterations");
  CHECK(fd0 >= 0 && fd1 >= 0 && fd1 - fd0 <= 10, "run: no fd leak across spawn cycles");
  CHECK(th0 > 0 && th1 - th0 <= 5, "run: no thread accumulation");
  CHECK(rss_peak > 0 && rss_peak < 512 * 1024, "run: RSS under 512 MiB ceiling");
  if (spawn_walls.size() >= 10) {
    double first5 = 0, last5 = 0;
    for (int i = 0; i < 5; ++i) first5 += spawn_walls[i];
    for (size_t i = spawn_walls.size() - 5; i < spawn_walls.size(); ++i) last5 += spawn_walls[i];
    first5 /= 5;
    last5 /= 5;
    CHECK(last5 < first5 * 4.0 + 0.05,
          "run: spawn cost does not degrade over the session (reaping healthy)");
  } else {
    CHECK(false, "run: enough spawns sampled for degradation check");
  }
  return suite::finish("run");
}
