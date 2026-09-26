// distribai_worker - runs grid work on this machine.
//
// The worker registers with an orchestrator, heartbeats, claims one replica at a
// time, materializes the job bundle it is handed, and runs the LibTorch trainer
// as a child process under the native rlimit contract (RLIMIT_AS, RLIMIT_CPU,
// RLIMIT_FSIZE, RLIMIT_NPROC, RLIMIT_NOFILE, RLIMIT_CORE through
// distribai::apply_limits_in_child). It then reads the GradReport envelope the
// trainer wrote and reports it back.
//
// Nothing here needs a shared filesystem: the bundle arrives in the claim
// response, so a Colab, Kaggle or VPS worker only needs the URL and an invite
// code.
//
//   distribai_worker --orchestrator http://127.0.0.1:50061 [--invite CODE]
//                    [--node-id NAME] [--work-dir DIR] [--trainer PATH]
//                    [--mem-mb N] [--cpu-sec N] [--once] [--max-jobs N]
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../sandbox/sandbox.hpp"
#include "http_client.hpp"
#include "protocol.hpp"

namespace {

using distribai::SandboxLimits;
using distribai::grid::Jw;
namespace grid = distribai::grid;
namespace json = distribai::json;

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop.store(true); }

bool make_dirs(const std::string& path) {
  std::string acc;
  for (size_t i = 0; i < path.size(); ++i) {
    acc += path[i];
    if (path[i] == '/') {
      if (acc.size() > 1) ::mkdir(acc.c_str(), 0755);
    }
  }
  ::mkdir(path.c_str(), 0755);
  struct stat st{};
  return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool is_exe(const std::string& path) {
  return !path.empty() && ::access(path.c_str(), X_OK) == 0;
}

int cpu_count() {
  const long n = ::sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? static_cast<int>(n) : 1;
}

std::string hostname() {
  char buf[256] = {0};
  if (::gethostname(buf, sizeof(buf) - 1) != 0) return "worker";
  return buf;
}

// Total memory in MiB from /proc/meminfo, or 0 when unreadable.
int64_t total_mem_mb() {
  std::ifstream in("/proc/meminfo");
  std::string key;
  int64_t kb = 0;
  std::string unit;
  while (in >> key >> kb >> unit) {
    if (key == "MemTotal:") return kb / 1024;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Child process with rlimits
// ---------------------------------------------------------------------------

struct RunResult {
  int exit_code = 0;
  bool timed_out = false;
  bool spawned = false;
  int signal = 0;
  double wall_s = 0;
  std::string error;
};

// Runs argv with the sandbox limit set applied in the child, stdout captured to
// out_path. The child gets its own process group so a timeout kills whatever it
// started as well.
RunResult run_limited(const std::vector<std::string>& argv, const SandboxLimits& lim,
                      uint64_t timeout_s, const std::string& out_path) {
  RunResult res;
  const int out_fd = ::open(out_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (out_fd < 0) {
    res.error = "cannot open " + out_path;
    return res;
  }
  const auto t0 = std::chrono::steady_clock::now();
  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(out_fd);
    res.error = std::string("fork() failed: ") + std::strerror(errno);
    return res;
  }
  if (pid == 0) {
    ::setpgid(0, 0);
    ::dup2(out_fd, STDOUT_FILENO);
    ::dup2(out_fd, STDERR_FILENO);
    ::close(out_fd);
    distribai::apply_limits_in_child(lim);
    std::vector<char*> cargv;
    cargv.reserve(argv.size() + 1);
    for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
    cargv.push_back(nullptr);
    ::execvp(cargv[0], cargv.data());
    std::fprintf(stderr, "cannot exec %s\n", cargv[0]);
    ::_exit(127);
  }
  ::close(out_fd);
  res.spawned = true;

  const uint64_t deadline_s = timeout_s;
  int status = 0;
  bool reaped = false;
  for (uint64_t waited = 0; waited <= deadline_s * 10; ++waited) {
    const pid_t w = ::waitpid(pid, &status, WNOHANG);
    if (w == pid) {
      reaped = true;
      break;
    }
    if (g_stop.load()) break;
    ::usleep(100 * 1000);
  }
  if (!reaped) {
    res.timed_out = true;
    ::kill(-pid, SIGKILL);
    ::kill(pid, SIGKILL);
    ::waitpid(pid, &status, 0);
  }
  res.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (WIFSIGNALED(status)) {
    res.signal = WTERMSIG(status);
    if (!res.timed_out) res.error = "trainer killed by signal " + std::to_string(res.signal);
  } else {
    res.exit_code = WEXITSTATUS(status);
  }
  if (res.timed_out) res.error = "trainer exceeded its CPU budget";
  return res;
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

struct Config {
  std::string orchestrator = "http://127.0.0.1:50061";
  std::string node_id;
  std::string invite;
  std::string work_dir = "runtime/grid/worker";
  std::string trainer = "build/cpp_port/distribai_torch_train";
  int64_t mem_mb = 8192;
  int64_t cpu_sec = 300;
  int64_t fsize_mb = 512;
  int nproc = 0;
  int nofile = 256;
  int64_t poll_s = 2;
  int64_t heartbeat_s = 10;
  int max_jobs = 0;  // 0 = run until stopped
  bool once = false;
  bool idle_exit = false;  // exit instead of polling when there is no work
  // Seconds to keep retrying the first registration while the coordinator is
  // unreachable. A refusal the coordinator sent is never retried.
  int64_t register_wait = 20;
};

class Worker {
 public:
  Worker(Config cfg, distribai::http::Url url) : cfg_(std::move(cfg)), url_(url) {}

  bool register_node(std::string& err) {
    transport_failed_ = false;
    if (cfg_.node_id.empty()) {
      cfg_.node_id = hostname() + "-" + std::to_string(::getpid());
    }
    const int64_t mem = total_mem_mb();
    const std::string hardware =
        Jw()
            .n("cpus", cpu_count())
            .n("mem_mb", static_cast<double>(mem))
            .s("os", "linux")
            .s("arch", arch())
            .str();
    const std::string bench =
        Jw()
            .n("effective_cpus", cpu_count())
            .n("mem_mb", static_cast<double>(mem))
            .s("trainer", cfg_.trainer)
            .b("torch", std::strstr(cfg_.trainer.c_str(), "torch") != nullptr)
            .str();
    const std::string body = Jw()
                                 .n("proto", grid::kProto)
                                 .s("node_id", cfg_.node_id)
                                 .s("role", "worker")
                                 .s("invite", cfg_.invite)
                                 .raw("hardware", hardware)
                                 .raw("benchmark", bench)
                                 .str();

    distribai::http::ClientResponse res;
    if (!distribai::http::request(url_, "POST", grid::kRegister, body,
                                  "application/json", res, err)) {
      transport_failed_ = true;
      return false;
    }
    transport_failed_ = false;
    if (res.status != 200) {
      err = "register refused (" + std::to_string(res.status) + "): " + res.body;
      return false;
    }
    json::Value welcome;
    if (!grid::parse_body(res.body, welcome, err)) return false;
    token_ = grid::str_field(welcome, "session_token");
    if (token_.empty()) {
      err = "register reply carried no session token";
      return false;
    }
    heartbeat_s_ = std::max<int64_t>(2, grid::i64_field(welcome, "heartbeat_s", 10));
    poll_s_ = std::max<int64_t>(1, grid::i64_field(welcome, "poll_s", 2));
    aggregate_ = grid::str_field(welcome, "aggregate", "trimmed_mean");
    std::printf("worker: registered as %s\n", cfg_.node_id.c_str());
    return true;
  }

  void register_signal_handlers() {
    struct sigaction sa{};
    sa.sa_handler = on_signal;
    ::sigaction(SIGINT, &sa, nullptr);
    ::sigaction(SIGTERM, &sa, nullptr);
  }

  // Registration retries while the coordinator is unreachable. A worker on a
  // notebook host or behind a fresh tunnel is often up a moment before the
  // coordinator answers, and exiting on the first refused connection throws
  // away the worker's whole session. A refusal the coordinator actually sent
  // (a bad invite code, a protocol mismatch) is final and is not retried.
  bool register_with_retry(std::string& err) {
    const int64_t budget_s = std::max<int64_t>(0, cfg_.register_wait);
    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(budget_s);
    int attempt = 0;
    for (;;) {
      if (register_node(err)) return true;
      if (!transport_failed_) return false;
      if (std::chrono::steady_clock::now() >= deadline) return false;
      ++attempt;
      const int wait = std::min(2, attempt);
      std::fprintf(stderr, "worker: %s; retrying in %ds\n", err.c_str(), wait);
      std::this_thread::sleep_for(std::chrono::seconds(wait));
      if (g_stop.load()) return false;
    }
  }

  int run() {
    register_signal_handlers();
    std::string err;
    if (!register_with_retry(err)) {
      std::fprintf(stderr, "worker: %s\n", err.c_str());
      return 1;
    }
    std::thread hb([this] { heartbeat_loop(); });

    int jobs_done = 0;
    int report_failures = 0;
    while (!g_stop.load()) {
      if (cfg_.max_jobs > 0 && jobs_done >= cfg_.max_jobs) break;
      json::Value task;
      bool got = false;
      if (!claim(task, got, err)) {
        std::fprintf(stderr, "worker: claim failed: %s\n", err.c_str());
        std::this_thread::sleep_for(std::chrono::seconds(5));
        continue;
      }
      if (!got) {
        if (cfg_.once || cfg_.idle_exit) {
          std::printf("worker: no work waiting\n");
          break;
        }
        sleep_checked(static_cast<int>(poll_s_));
        continue;
      }
      if (!run_task(task, err)) {
        // The work may still have been done; only the report failed. Say so
        // instead of counting it as a clean job.
        ++report_failures;
        std::fprintf(stderr, "worker: %d task(s) could not be reported\n", report_failures);
      }
      ++jobs_done;
      if (cfg_.once) break;
    }

    g_stop.store(true);
    hb.join();
    say_goodbye();
    std::printf("worker: stopped after %d job(s)\n", jobs_done);
    return 0;
  }

 private:
  static std::string arch() {
#if defined(__x86_64__)
    return "x86_64";
#elif defined(__aarch64__)
    return "aarch64";
#else
    return "unknown";
#endif
  }

  void sleep_checked(int seconds) {
    for (int i = 0; i < seconds * 10 && !g_stop.load(); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  void heartbeat_loop() {
    while (!g_stop.load()) {
      // Sleep in short slices so a stop request does not wait out the whole
      // heartbeat interval before the thread joins.
      const int slices = static_cast<int>(heartbeat_s_ * 10);
      for (int i = 0; i < slices && !g_stop.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      if (g_stop.load()) break;
      std::string current;
      {
        std::lock_guard<std::mutex> lock(mu_);
        current = current_task_;
      }
      const std::string body = Jw()
                                   .s("node_id", cfg_.node_id)
                                   .s("token", token_)
                                   .s("status", current.empty() ? "idle" : "busy")
                                   .s("current_task_id", current)
                                   .str();
      distribai::http::ClientResponse res;
      std::string err;
      distribai::http::request(url_, "POST", grid::kHeartbeat, body, "application/json", res,
                               err, 20);
    }
  }

  void say_goodbye() {
    if (token_.empty()) return;
    const std::string body =
        Jw().s("node_id", cfg_.node_id).s("token", token_).str();
    distribai::http::ClientResponse res;
    std::string err;
    distribai::http::request(url_, "POST", grid::kBye, body, "application/json", res, err, 10);
  }

  bool claim(json::Value& task, bool& got, std::string& err) {
    const std::string body =
        Jw().s("node_id", cfg_.node_id).s("token", token_).str();
    distribai::http::ClientResponse res;
    if (!distribai::http::request(url_, "POST", grid::kClaim, body, "application/json", res, err,
                                  60)) {
      return false;
    }
    if (res.status == 401) {
      err = "session expired";
      if (!register_node(err)) return false;
      return claim(task, got, err);
    }
    if (res.status != 200) {
      err = "claim refused (" + std::to_string(res.status) + "): " + res.body;
      return false;
    }
    if (!grid::parse_body(res.body, task, err)) return false;
    got = grid::str_field(task, "type") == grid::kTTask;
    return true;
  }

  bool run_task(const json::Value& task, std::string& err) {
    const std::string task_id = grid::str_field(task, "task_id");
    const std::string job_id = grid::str_field(task, "job_id");
    const int64_t steps = grid::i64_field(task, "steps", 1);
    const std::string spec = grid::str_field(task, "spec");
    {
      std::lock_guard<std::mutex> lock(mu_);
      current_task_ = task_id;
    }
    std::printf("worker: task %s (%s, %lld steps)\n", task_id.c_str(), job_id.c_str(),
                static_cast<long long>(steps));

    const std::string dir = cfg_.work_dir + "/" + task_id;
    if (!make_dirs(dir)) {
      return report_failure(task_id, "cannot create " + dir, err);
    }
    if (!distribai::http::write_file(dir + "/job.json", spec)) {
      return report_failure(task_id, "cannot write job.json", err);
    }
    if (const json::Value* files = task.get("files")) {
      for (const auto& f : files->as_array()) {
        if (!f.is_object()) continue;
        const std::string name = grid::str_field(f, "name");
        if (name.empty() || name.find('/') != std::string::npos) continue;
        std::vector<uint8_t> bytes;
        std::string derr;
        if (!distribai::b64::decode(grid::str_field(f, "b64"), bytes, derr)) {
          return report_failure(task_id, "bundle file " + name + ": " + derr, err);
        }
        const std::string blob(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        if (!distribai::http::write_file(dir + "/" + name, blob)) {
          return report_failure(task_id, "cannot write " + dir + "/" + name, err);
        }
      }
    }

    if (!is_exe(cfg_.trainer)) {
      return report_failure(task_id,
                            "trainer not found at " + cfg_.trainer +
                                " (build it with: make torch)",
                            err);
    }

    SandboxLimits lim;
    lim.mem_mb = static_cast<uint64_t>(cfg_.mem_mb);
    lim.cpu_sec = static_cast<uint64_t>(cfg_.cpu_sec);
    lim.fsize_mb = static_cast<uint64_t>(cfg_.fsize_mb);
    // RLIMIT_NPROC counts threads for the whole uid, so a fixed small number
    // breaks a trainer: LibTorch starts thread pools, and the limit is checked
    // against every task the uid already has. The default therefore follows the
    // ambient count plus headroom, the same rule the port's sandbox uses.
    lim.nproc = cfg_.nproc > 0 ? cfg_.nproc : distribai::uid_task_count_plus_headroom();
    lim.nofile = cfg_.nofile;

    const std::string envelope_path = dir + "/result.env";
    const std::string stdout_path = dir + "/trainer.log";
    const std::vector<std::string> argv{cfg_.trainer,
                                       "--spec",
                                       dir + "/job.json",
                                       "--json",
                                       "--envelope-out",
                                       envelope_path};
    const RunResult run =
        run_limited(argv, lim, static_cast<uint64_t>(cfg_.cpu_sec) + 60, stdout_path);
    if (!run.spawned) return report_failure(task_id, run.error, err);

    std::string stdout_text;
    distribai::http::read_file(stdout_path, stdout_text);
    json::Value metrics;
    bool have_metrics = false;
    {
      const size_t last_brace = stdout_text.find_last_of('{');
      if (last_brace != std::string::npos) {
        std::string perr;
        json::Value candidate;
        if (json::parse(stdout_text.substr(last_brace), candidate, perr) &&
            candidate.is_object()) {
          metrics = candidate;
          have_metrics = true;
        }
      }
    }
    const std::string status = have_metrics ? metrics.str("status", "error") : "error";
    const bool ok = run.exit_code == 0 && status == "ok" &&
                    distribai::http::is_readable(envelope_path);
    if (!ok) {
      std::string reason = run.error;
      if (reason.empty() && have_metrics) reason = metrics.str("error");
      if (reason.empty()) {
        reason = "trainer exited " + std::to_string(run.exit_code) + " (see " + stdout_path + ")";
      }
      return report_failure(task_id, reason, err);
    }

    std::string envelope;
    if (!distribai::http::read_file(envelope_path, envelope)) {
      return report_failure(task_id, "cannot read " + envelope_path, err);
    }

    std::string body =
        Jw()
            .s("node_id", cfg_.node_id)
            .s("token", token_)
            .s("task_id", task_id)
            .b("ok", true)
            .s("envelope_b64", distribai::b64::encode(envelope))
            .n("steps", metrics.num("steps", static_cast<double>(steps)))
            .n("final_loss", metrics.num("final_loss"))
            .n("wall_s", metrics.num("wall_s", run.wall_s))
            .n("steps_per_s", metrics.num("steps_per_s"))
            .s("checkpoint_id", metrics.str("checkpoint"))
            .s("engine", metrics.str("engine", "libtorch"))
            .str();
    std::string current_body = body;  // refreshed with the new token after a re-register
    // The replica is already trained, so a failed report wastes real work.
    // Retry the transport, and re-register once when the session was rotated
    // out from under this worker (two workers sharing a node id do that).
    distribai::http::ClientResponse res;
    bool reported = false;
    for (int attempt = 0; attempt < 3 && !reported; ++attempt) {
      if (!distribai::http::request(url_, "POST", grid::kResult, current_body,
                                    "application/json", res, err, 300)) {
        std::fprintf(stderr, "worker: reporting %s failed: %s\n", task_id.c_str(), err.c_str());
        std::this_thread::sleep_for(std::chrono::seconds(1));
        continue;
      }
      if (res.status == 401) {
        std::fprintf(stderr, "worker: session rotated; re-registering to report %s\n",
                     task_id.c_str());
        std::string rereg_err;
        if (!register_with_retry(rereg_err)) {
          err = rereg_err;
          break;
        }
        current_body = Jw()
                           .s("node_id", cfg_.node_id)
                           .s("token", token_)
                           .s("task_id", task_id)
                           .b("ok", true)
                           .s("envelope_b64", distribai::b64::encode(envelope))
                           .n("steps", metrics.num("steps", static_cast<double>(steps)))
                           .n("final_loss", metrics.num("final_loss"))
                           .n("wall_s", metrics.num("wall_s", run.wall_s))
                           .n("steps_per_s", metrics.num("steps_per_s"))
                           .s("checkpoint_id", metrics.str("checkpoint"))
                           .s("engine", metrics.str("engine", "libtorch"))
                           .str();
        continue;
      }
      if (res.status != 200) {
        err = "result refused (" + std::to_string(res.status) + "): " + res.body;
        break;
      }
      reported = true;
    }
    if (!reported) {
      std::fprintf(stderr, "worker: could not report %s (%s)\n", task_id.c_str(), err.c_str());
      std::lock_guard<std::mutex> lock(mu_);
      current_task_.clear();
      return false;
    }
    std::printf("worker: task %s done, loss=%.6f\n", task_id.c_str(), metrics.num("final_loss"));
    {
      std::lock_guard<std::mutex> lock(mu_);
      current_task_.clear();
    }
    return true;
  }

  bool report_failure(const std::string& task_id, const std::string& reason, std::string& err) {
    std::fprintf(stderr, "worker: task %s failed: %s\n", task_id.c_str(), reason.c_str());
    const std::string body = Jw()
                                 .s("node_id", cfg_.node_id)
                                 .s("token", token_)
                                 .s("task_id", task_id)
                                 .b("ok", false)
                                 .s("error", reason)
                                 .str();
    distribai::http::ClientResponse res;
    distribai::http::request(url_, "POST", grid::kResult, body, "application/json", res, err, 60);
    {
      std::lock_guard<std::mutex> lock(mu_);
      current_task_.clear();
    }
    return true;  // the failure is reported; the loop keeps going
  }

  Config cfg_;
  distribai::http::Url url_;
  std::string token_;
  std::string aggregate_ = "trimmed_mean";
  int64_t heartbeat_s_ = 10;
  int64_t poll_s_ = 2;
  std::mutex mu_;
  std::string current_task_;
  // Set by register_node so the caller can tell a refused registration from an
  // unreachable coordinator.
  bool transport_failed_ = false;
};

struct Args {
  Config cfg;
  bool ok = true;
  std::string error;
};

void print_usage() {
  std::printf(
      "distribai_worker: run grid work here\n"
      "  --orchestrator URL  http(s)://host:port or host:port (default http://127.0.0.1:50061)\n"
      "  --node-id NAME      node name shown in the dashboard (default hostname-pid)\n"
      "  --invite CODE       invite code, when the grid requires one\n"
      "  --work-dir DIR      where bundles are unpacked (default runtime/grid/worker)\n"
      "  --trainer PATH      trainer binary (default build/cpp_port/distribai_torch_train)\n"
      "  --mem-mb N          RLIMIT_AS for the trainer (default 8192)\n"
      "  --cpu-sec N         RLIMIT_CPU hard and soft (default 300)\n"
      "  --fsize-mb N        RLIMIT_FSIZE (default 512)\n"
      "  --nproc N           RLIMIT_NPROC; 0 (the default) means the uid's current task\n"
      "                      count plus headroom, which threaded trainers need\n"
      "  --register-wait S   seconds to keep retrying the first registration\n"
      "                      while the coordinator is unreachable (default 20, 0 = one try)\n"
      "  --once              claim at most one task and exit\n"
      "  --idle-exit         exit instead of polling when the queue is empty\n"
      "  --max-jobs N        stop after N tasks\n");
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
    if (flag == "--orchestrator" || flag == "--url") a.cfg.orchestrator = next();
    else if (flag == "--node-id") a.cfg.node_id = next();
    else if (flag == "--invite") a.cfg.invite = next();
    else if (flag == "--work-dir") a.cfg.work_dir = next();
    else if (flag == "--trainer") a.cfg.trainer = next();
    else if (flag == "--mem-mb") a.cfg.mem_mb = std::atoll(next().c_str());
    else if (flag == "--cpu-sec") a.cfg.cpu_sec = std::atoll(next().c_str());
    else if (flag == "--fsize-mb") a.cfg.fsize_mb = std::atoll(next().c_str());
    else if (flag == "--nproc") a.cfg.nproc = std::atoi(next().c_str());
    else if (flag == "--register-wait") a.cfg.register_wait = std::atoll(next().c_str());
    else if (flag == "--once") a.cfg.once = true;
    else if (flag == "--idle-exit") a.cfg.idle_exit = true;
    else if (flag == "--max-jobs") a.cfg.max_jobs = std::atoi(next().c_str());
    else if (flag == "--help" || flag == "-h") {
      print_usage();
      std::exit(0);
    } else {
      a.ok = false;
      a.error = "unknown flag: " + flag;
    }
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  // Line-buffered output: a worker's log is normally redirected to a file, and
  // a killed worker should still leave a record of what it was doing.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const Args args = parse_args(argc, argv);
  if (!args.ok) {
    std::fprintf(stderr, "distribai_worker: %s\n", args.error.c_str());
    return 2;
  }
  distribai::http::Url url;
  std::string err;
  if (!distribai::http::parse_url(args.cfg.orchestrator, url, err)) {
    std::fprintf(stderr, "distribai_worker: %s\n", err.c_str());
    return 2;
  }
  // The worker fills in a node id from the hostname when none was given.
  Worker worker(args.cfg, url);
  return worker.run();
}
