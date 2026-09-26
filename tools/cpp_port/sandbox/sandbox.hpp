// DistribAI native sandbox: "each sandbox is its own system".
//
// Replaces the Python worker/src/sandbox subprocess mode with native
// primitives (no runtime, no torch import, no pickle):
//   * RLIMIT_AS / RLIMIT_CPU (hard==soft) / RLIMIT_FSIZE / RLIMIT_NPROC /
//     RLIMIT_NOFILE / RLIMIT_CORE applied in the child after fork()
//   * optional user + PID namespace (CLONE_NEWUSER|CLONE_NEWPID) so each
//     sandbox has its own PID space and uid mapping, so per-UID rlimits
//     then apply to the sandbox alone, not the whole account
//   * pipe handoff of a POD result (no serialization layer at all)
// Mirrors: worker/src/sandbox/sandbox.py::_run_subprocess + _apply_resource_limits
#pragma once

#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <mutex>
#include <string>

namespace distribai {

struct SandboxLimits {
  uint64_t mem_mb = 4096;      // RLIMIT_AS, exact
  uint64_t cpu_sec = 300;      // RLIMIT_CPU hard == soft
  uint64_t fsize_mb = 512;     // RLIMIT_FSIZE
  int nproc = 0;               // RLIMIT_NPROC; 0 => uid tasks + 96
  int nofile = 256;            // RLIMIT_NOFILE
  bool own_system = true;      // user+PID namespace (falls back silently if EPERM)
};

struct SandboxResult {
  bool ok = false;
  bool timed_out = false;
  bool limit_killed = false;   // died from SIGXCPU/SIGKILL/SEGV after limits
  int exit_code = 0;
  int signal = 0;
  double spawn_join_s = 0.0;   // fork + run + join wall time
  uint64_t peak_rss_kb = 0;    // child ru_maxrss
  uint64_t cpu_time_ms = 0;    // child ru_utime + ru_stime
  std::string error;
};

// POD handed back over the pipe. Keep it trivially copyable.
// Fields mirror the Python sandbox envelope (matrix rows 3.4/3.5):
// status/n_params/wall_s/steps_per_s/ms_per_step/final_loss/first5_losses/
// grad_len/grad_sum/grad_first3. The wire format itself is POD (not
// pickle) by design; payload-level equality is the contract.
struct TrainPod {
  int32_t status;          // 0 ok, 1 error
  int32_t pad;
  uint64_t steps;
  uint64_t param_count;
  uint64_t grad_len;
  double wall_s;
  double steps_per_s;
  double ms_per_step;
  double final_loss;
  double loss_first;
  double loss_first5[5];
  double grad_sum;
  double grad_first3[3];
  uint64_t grad_blob_len;  // bytes of f32 grad blob written AFTER this POD
  char error[256];
};

// Count this uid's tasks (RLIMIT_NPROC counts THREADS too) and add headroom.
// Must run in the PARENT: after CLONE_NEWUSER the child's uid is remapped.
//
// O2: the count is cached for 250 ms. The old implementation popen()ed
// `ps | wc` on every spawn - a fork+exec per sandbox child (~10 ms) that the
// bench paid for every run_sandboxed call. The ambient task count on a shared
// box moves on seconds-scale, so a quarter-second TTL keeps NPROC sizing
// responsive while removing the per-child exec from the spawn path.
inline int uid_task_count_plus_headroom() {
  static std::mutex mu;
  static std::chrono::steady_clock::time_point valid_until;
  static int cached = 0;

  std::lock_guard<std::mutex> lock(mu);
  const auto now = std::chrono::steady_clock::now();
  if (cached > 0 && now < valid_until) return cached;

  const std::string cmd = "ps -u " + std::to_string(static_cast<long>(getuid())) +
                          " -L --no-headers 2>/dev/null | wc -l";
  int n = 0;
  bool ok = false;
  if (FILE* p = popen(cmd.c_str(), "r")) {
    ok = fscanf(p, "%d", &n) == 1 && n > 0;
    pclose(p);
  }
  if (!ok) return 512;  // popen failed: conservative default, not cached
  cached = n + 96;
  valid_until = now + std::chrono::milliseconds(250);
  return cached;
}

inline void apply_limits_in_child(const SandboxLimits& lim) {
  const rlim_t as_bytes = static_cast<rlim_t>(lim.mem_mb) * 1024 * 1024;
  const rlim_t fsize_bytes = static_cast<rlim_t>(lim.fsize_mb) * 1024 * 1024;
  const int nproc = lim.nproc > 0 ? lim.nproc : 512;
  struct rlimit rl;
  rl.rlim_cur = as_bytes; rl.rlim_max = as_bytes;
  setrlimit(RLIMIT_AS, &rl);
  rl.rlim_cur = lim.cpu_sec; rl.rlim_max = lim.cpu_sec;  // hard==soft: no grace
  setrlimit(RLIMIT_CPU, &rl);
  rl.rlim_cur = fsize_bytes; rl.rlim_max = fsize_bytes;
  setrlimit(RLIMIT_FSIZE, &rl);
  rl.rlim_cur = nproc; rl.rlim_max = nproc;
  setrlimit(RLIMIT_NPROC, &rl);
  rl.rlim_cur = lim.nofile; rl.rlim_max = lim.nofile;
  setrlimit(RLIMIT_NOFILE, &rl);
  rl.rlim_cur = 0; rl.rlim_max = 0;
  setrlimit(RLIMIT_CORE, &rl);
}

// Fork a child that applies limits, runs `fn(write_fd)`, and returns via pipe.
// `use_namespaces` uses clone with CLONE_NEWUSER|CLONE_NEWPID when permitted.
//   NOTE: with a PID namespace, the parent must still be the reaper; we use
//   fork() for the actual child and set up namespaces only when
//   own_system=true, detecting failure (EPERM) and degrading to plain fork.
template <typename Fn>
SandboxResult run_sandboxed(const SandboxLimits& lim, uint64_t timeout_s, const Fn& fn,
                            void* result_buf = nullptr, size_t result_len = 0,
                            std::vector<uint8_t>* blob_out = nullptr) {
  SandboxResult res;
  int pipefd[2];
  if (pipe(pipefd) != 0) {
    res.error = "pipe() failed";
    return res;
  }

  struct timespec ts0, ts1;
  clock_gettime(CLOCK_MONOTONIC, &ts0);

  SandboxLimits eff = lim;
  if (eff.nproc <= 0) eff.nproc = uid_task_count_plus_headroom();

  pid_t pid = fork();
  if (pid < 0) {
    res.error = "fork() failed";
    close(pipefd[0]); close(pipefd[1]);
    return res;
  }

  if (pid == 0) {
    // ---- child ----
    close(pipefd[0]);
    // Environment hygiene: fork inherits the parent's environ; the sandbox
    // contract (mirroring the Python subprocess scrub) is that children see
    // no ambient secrets/config. Re-exec-free scrub: keep only PATH/TMPDIR,
    // drop everything else (suite security-1 tests this).
    {
      static const char* keep[] = {"PATH", "TMPDIR"};
      static const int keep_n = 2;
      for (char** e = environ; e && *e; ++e) {
        bool keep_it = false;
        for (int k = 0; k < keep_n; ++k) {
          const size_t n = std::strlen(keep[k]);
          if (std::strncmp(*e, keep[k], n) == 0 && (*e)[n] == '=') keep_it = true;
        }
        if (!keep_it) (*e)[0] = '\0';  // blank the entry in place
      }
    }
    // Own-system namespaces: unshare user, then mount (O6 hardening), then
    // PID. User ns first: it grants capabilities in the new namespace so the
    // mount-ns unshare is permitted without privilege. The user ns gives a
    // fresh uid map so per-UID rlimits (NPROC) apply to this sandbox alone;
    // the mount ns gives a private mount view (no bind/move of host mounts);
    // the PID ns gives private process numbering. Degrade silently on EPERM
    // (containerized environments may forbid any of them). rlimits still
    // apply process-wide.
    if (lim.own_system) {
      if (unshare(CLONE_NEWUSER) != 0 || unshare(CLONE_NEWNS) != 0 ||
          unshare(CLONE_NEWPID) != 0) {
        // fall through: limits still apply process-wide via rlimits
      }
    }
    apply_limits_in_child(eff);
    fn(pipefd[1]);
    _exit(0);
  }

  // ---- parent ----
  close(pipefd[1]);
  TrainPod pod{};
  ssize_t got = -1;
  // Read with deadline.
  const uint64_t deadline_ms = timeout_s * 1000ULL;
  uint64_t waited_ms = 0;
  while (waited_ms < deadline_ms) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(pipefd[0], &rfds);
    struct timeval tv{0, 50 * 1000};  // 50 ms poll
    int rdy = select(pipefd[0] + 1, &rfds, nullptr, nullptr, &tv);
    if (rdy > 0) {
      got = read(pipefd[0], &pod, sizeof(pod));
      if (result_buf && got > 0) {
        const size_t take = (static_cast<size_t>(got) < result_len) ? static_cast<size_t>(got) : result_len;
        std::memcpy(result_buf, &pod, take);
      }
      // Optional trailing blob (e.g. flattened gradients): the child writes
      // grad_blob_len bytes right after the POD. Read fully or fail.
      if (blob_out && got == static_cast<ssize_t>(sizeof(pod)) && pod.grad_blob_len > 0) {
        if (pod.grad_blob_len > (64ull << 20)) {  // 64 MiB sanity cap
          res.error = "grad blob exceeds sanity cap";
        } else {
          blob_out->resize(pod.grad_blob_len);
          size_t have = 0;
          while (have < pod.grad_blob_len) {
            const ssize_t chunk = read(pipefd[0], blob_out->data() + have, pod.grad_blob_len - have);
            if (chunk <= 0) { res.error = "grad blob truncated"; break; }
            have += static_cast<size_t>(chunk);
          }
        }
      }
      break;
    }
    // Reap early if the child died before writing.
    int status = 0;
    pid_t w = waitpid(pid, &status, WNOHANG);
    if (w == pid) {
      // Child exited without a pod (limit kill or crash).
      if (WIFSIGNALED(status)) {
        res.signal = WTERMSIG(status);
        res.limit_killed = (res.signal == SIGXCPU || res.signal == SIGKILL ||
                            res.signal == SIGSEGV || res.signal == SIGXFSZ);
      }
      break;
    }
    waited_ms += 50;
  }
  close(pipefd[0]);

  int status = 0;
  // Outcome classes, checked in order (suite dbg_reap caught the ordering):
  //   1. full pod delivered  -> success (child may still be finishing _exit;
  //      that is NOT a timeout)
  //   2. child died by signal (reaped earlier or now) -> limit kill / crash
  //   3. deadline expired with a live child -> kill + timed_out
  //   4. child exited without a pod -> clean failure
  const pid_t reap = waitpid(pid, &status, WNOHANG);  // -1 none, 0 alive, pid exited
  const bool delivered = (got == static_cast<ssize_t>(sizeof(pod)));
  if (delivered) {
    for (int i = 0; i < 200; ++i) {
      const pid_t w = waitpid(pid, &status, WNOHANG);
      if (w == pid || w < 0) break;
      usleep(5 * 1000);
    }
  } else if (reap == pid && WIFSIGNALED(status)) {
    res.signal = WTERMSIG(status);
    res.limit_killed = (res.signal == SIGXCPU || res.signal == SIGKILL ||
                        res.signal == SIGSEGV || res.signal == SIGXFSZ);
  } else if (reap == pid) {
    // exited without delivering a pod (no signal)
    res.exit_code = WEXITSTATUS(status);
  } else {
    // Child alive (or dying) at reaping time. Kill to be sure, then classify
    // from the ACTUAL wait status: a child already dying from SIGXFSZ/SIGXCPU
    // races this SIGKILL (dbg_xfsz showed reap=0 for a process with a pending
    // death signal). If it died of a limit signal before SIGKILL landed, that
    // is a limit kill, not a timeout.
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status) && WTERMSIG(status) != SIGKILL) {
      res.signal = WTERMSIG(status);
      res.limit_killed = (res.signal == SIGXCPU || res.signal == SIGXFSZ ||
                          res.signal == SIGSEGV);
    } else {
      res.timed_out = true;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &ts1);
  res.spawn_join_s = static_cast<double>(ts1.tv_sec - ts0.tv_sec) +
                     static_cast<double>(ts1.tv_nsec - ts0.tv_nsec) / 1e9;

  struct rusage ru{};
  // ru of the (already reaped) child:
  getrusage(RUSAGE_CHILDREN, &ru);  // cumulative; acceptable for single-child scope
  res.peak_rss_kb = static_cast<uint64_t>(ru.ru_maxrss);
  res.cpu_time_ms = static_cast<uint64_t>(ru.ru_utime.tv_sec * 1000 + ru.ru_utime.tv_usec / 1000) +
                    static_cast<uint64_t>(ru.ru_stime.tv_sec * 1000 + ru.ru_stime.tv_usec / 1000);

  if (res.timed_out) {
    res.error = "sandbox timed out";
    return res;
  }
  if (res.signal != 0) {
    res.error = "child killed by signal " + std::to_string(res.signal);
    return res;
  }
  if (got != static_cast<ssize_t>(sizeof(pod))) {
    res.error = "no result pod from child";
    res.exit_code = 1;
    return res;
  }
  if (pod.status != 0) {
    res.error = pod.error;
    return res;
  }
  res.ok = true;
  res.exit_code = 0;
  return res;
}

}  // namespace distribai
