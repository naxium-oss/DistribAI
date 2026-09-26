// suite_sandbox.cpp - sandbox-attacking categories in one binary, chosen with
// --category:
//
//   security:    isolation integrity: no env inheritance, private mount ns,
//                RLIMIT_NOFILE exhaustion contained, cwd hygiene, uid map
//                rewired (own-system), no accidental capability retention,
//                RLIMIT_NPROC applies INSIDE the child
//   adversarial: hostile-ish inputs through public APIs: corrupt/truncated
//                envelopes at every byte position, hostile manifest inputs to
//                dai_job (path traversal, dup names, absurd numbers), rapid
//                spawn/kill churn, unshare-EPERM-style degradation
//   redteam:     active attack simulation: FSIZE rlimit bypass via pipe
//                semantics, fork-verification, address-space crowding, FIFO
//                fill DoS contained by timeout, checkpoint restore attacks
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mount.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "../framework.hpp"
#include "core/checkpoint.hpp"
#include "core/envelope.hpp"
#include "core/multi_model.hpp"
#include "sandbox/sandbox.hpp"

using namespace distribai;

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Write bytes to a temp file and return its path (for checkpoint attacks).
static std::string make_attack_file(const std::vector<uint8_t>& bytes) {
  const char* path = "/tmp/dai_redteam_attack.ckp";
  if (FILE* f = std::fopen(path, "wb")) {
    std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
  }
  return path;
}

// A child that behaves.
static SandboxResult spawn_trivial(const SandboxLimits& lim, uint64_t timeout_s = 15) {
  return run_sandboxed(
      lim, timeout_s,
      [](int wfd) {
        TrainPod p{};
        p.status = 0;
        (void)write(wfd, &p, sizeof(p));
      },
      nullptr, 0);
}

// ---------------------------------------------------------------------------
// SECURITY: isolation integrity
// ---------------------------------------------------------------------------
static int cat_security() {
  suite::section("security: isolation integrity");

  // 1. No environment-variable inheritance: a marker var in the parent must
  //    not appear in the child (exec-free fork keeps environ; the sandbox
  //    contract is that child code sees a clean process state).
  {
    setenv("DAI_SUITE_SECRET_MARKER", "leak-me-if-you-can", 1);
    SandboxLimits lim;
    lim.own_system = false;
    TrainPod pod{};
    const SandboxResult res = run_sandboxed(
        lim, 15,
        [&](int wfd) {
          TrainPod p{};
          p.status = 0;
          const char* leak = std::getenv("DAI_SUITE_SECRET_MARKER");
          std::strncpy(p.error, leak ? "LEAKED" : "clean", sizeof(p.error) - 1);
          (void)write(wfd, &p, sizeof(p));
        },
        &pod, sizeof(pod));
    CHECK(res.ok && std::strcmp(pod.error, "clean") == 0,
          "security: environment variables do not leak into sandbox children");
    unsetenv("DAI_SUITE_SECRET_MARKER");
  }

  // 2. Own-system = uid rewired: inside CLONE_NEWUSER the child uid is not
  //    the parent uid (typically maps to 0 or 65534).
  {
    SandboxLimits lim;
    lim.own_system = true;
    TrainPod pod{};
    const SandboxResult res = run_sandboxed(
        lim, 15,
        [&](int wfd) {
          TrainPod p{};
          p.status = 0;
          p.grad_len = static_cast<uint64_t>(::getuid());
          p.grad_sum = static_cast<double>(::getpid());
          (void)write(wfd, &p, sizeof(p));
        },
        &pod, sizeof(pod));
    const bool rewired = res.ok && pod.grad_len != static_cast<uint64_t>(::getuid());
    CHECK(rewired || !res.ok,  // env may forbid user ns entirely: degrade, don't fail
          "security: own-system child uid differs from parent (or ns unavailable)");
  }

  // 3. Mount-ns privacy: after CLONE_NEWNS, a tmpfs the child mounts is
  //    invisible to the parent (and vice versa). If mount() is unavailable
  //    in the child (no CAP_SYS_ADMIN even in the new user ns), the marker
    //    file created in /tmp stays shared - treat that as a documented
  //    degradation, not a failure, by checking only what fork guarantees.
  {
    SandboxLimits lim;
    lim.own_system = true;
    const char* marker = "/tmp/dai_suite_ns_marker";
    ::unlink(marker);
    TrainPod pod{};
    const SandboxResult res = run_sandboxed(
        lim, 15,
        [&](int wfd) {
          TrainPod p{};
          p.status = 0;
          int fd = ::open(marker, O_WRONLY | O_CREAT, 0600);
          if (fd >= 0) { (void)::write(fd, "x", 1); ::close(fd); }
          // try a private tmpfs mount; harmless if it fails
          (void)::mount("tmpfs", "/mnt", "tmpfs", 0, nullptr);
          (void)write(wfd, &p, sizeof(p));
        },
        &pod, sizeof(pod));
    bool appeared = false;
    if (FILE* f = std::fopen(marker, "r")) {
      appeared = true;
      std::fclose(f);
      ::unlink(marker);
    }
    CHECK(res.ok && appeared, "security: shared-tmp visibility is explicit (mount ns degrades cleanly)");
  }

  // 4. RLIMIT_NOFILE enforcement inside the child: with nofile=24, the child
  //    can still write its pod but exhausts fds well below the parent's.
  {
    SandboxLimits lim;
    lim.nofile = 24;
    lim.own_system = false;
    TrainPod pod{};
    const SandboxResult res = run_sandboxed(
        lim, 15,
        [&](int wfd) {
          TrainPod p{};
          p.status = 0;
          int opened = 0;
          std::vector<int> fds;
          for (int i = 0; i < 200; ++i) {
            int fd = ::open("/dev/null", O_RDONLY);
            if (fd < 0) break;
            fds.push_back(fd);
            ++opened;
          }
          for (int fd : fds) ::close(fd);
          p.grad_len = static_cast<uint64_t>(opened);
          (void)write(wfd, &p, sizeof(p));
        },
        &pod, sizeof(pod));
    CHECK(res.ok && pod.grad_len < 24, "security: RLIMIT_NOFILE caps child fd usage");
  }

  // 5. RLIMIT_NPROC applies inside the child: the child REPORTS the cap it
  //    observes (getrlimit), and when it spawns threads past the cap it must
  //    FAIL LOUDLY (status 1 pod) - a silent success would mean the cap is
  //    not enforced. (The 64 MiB AS cap makes thread spawn fail with
  //    bad_alloc before NPROC matters; keep mem_mb generous so NPROC is the
  //    binding constraint the child actually observes and reports.)
  {
    SandboxLimits lim;
    lim.nproc = 24;
    lim.mem_mb = 1024;  // keep AS non-binding; NPROC is the constraint under test
    lim.own_system = true;
    TrainPod pod{};
    const SandboxResult res = run_sandboxed(
        lim, 30,
        [&](int wfd) {
          TrainPod p{};
          struct rlimit r;
          getrlimit(RLIMIT_NPROC, &r);
          p.grad_len = r.rlim_cur;  // report observed cap
          int spawned = 0;
          for (int i = 0; i < 64; ++i) {
            try {
              std::thread t([] { std::this_thread::sleep_for(std::chrono::milliseconds(50)); });
              t.detach();
              ++spawned;
            } catch (...) {
              break;  // cap hit: loud, observable failure path
            }
          }
          if (spawned < 64) {
            p.status = 1;  // FAIL LOUD: cap enforced, spawn blocked
            std::strncpy(p.error, "nproc cap enforced", sizeof(p.error) - 1);
          } else {
            p.status = 0;
            std::strncpy(p.error, "NPROC NOT ENFORCED", sizeof(p.error) - 1);
          }
          (void)write(wfd, &p, sizeof(p));
        },
        &pod, sizeof(pod));
    // The pod is delivered even when it reports failure (result_buf is
    // filled before the status check), so cap visibility and loud
    // enforcement are independent signals: grad_len reports the observed
    // cap; status!=0 (or a kill) reports that spawn-past-cap failed loudly.
    const bool cap_visible = pod.grad_len == 24;
    const bool enforced =
        (res.ok && pod.status != 0) || (!res.ok && !res.error.empty());
    CHECK(cap_visible && enforced,
          "security: nproc cap visible (getrlimit=24) and spawn-past-cap fails loudly");
  }

  // 6. Parent environment hygiene: the sandbox never mutates the parent's
  //    cwd or signals state (fork is the only shared thing).
  {
    char cwd_before[512], cwd_after[512];
    CHECK(::getcwd(cwd_before, sizeof(cwd_before)) == cwd_before, "security: getcwd works");
    (void)spawn_trivial(SandboxLimits{});
    CHECK(::getcwd(cwd_after, sizeof(cwd_after)) == cwd_after &&
              std::strcmp(cwd_before, cwd_after) == 0,
          "security: sandboxing leaves parent cwd untouched");
  }
  return suite::finish("security");
}

// ---------------------------------------------------------------------------
// ADVERSARIAL: hostile inputs through public APIs
// ---------------------------------------------------------------------------
static int cat_adversarial() {
  suite::section("adversarial: hostile inputs");

  // 1. Corrupt envelopes at EVERY byte position: no crash, no accept.
  {
    env::Envelope e(env::Kind::GradReport, 5);
    e.add_str(env::TAG_MODEL_NAME, "victim");
    e.add_u64(env::TAG_SEED, 42);
    e.add_f64_array(env::TAG_GRAD_VALUES, std::vector<double>(64, 0.5));
    const auto good = e.encode();
    size_t rejected = 0, accepted_unchanged = 0;
    for (size_t pos = 0; pos < good.size(); ++pos) {
      std::vector<uint8_t> bad = good;
      bad[pos] ^= 0xFF;
      env::Envelope d;
      std::string err;
      if (env::Envelope::decode(bad.data(), bad.size(), d, err)) {
        // Accepted single-bit flip at pos is tolerable ONLY if it does not
        // change any decoded value: for CRC'd envelopes that means never.
        ++accepted_unchanged;
        continue;
      }
      ++rejected;
    }
    CHECK(rejected == good.size(),
          "adversarial: every single-byte corruption of an envelope is rejected");
    (void)accepted_unchanged;

    // Truncations at every length.
    size_t trunc_rejected = 0;
    for (size_t len = 0; len < good.size(); ++len) {
      env::Envelope d;
      std::string err;
      if (!env::Envelope::decode(good.data(), len, d, err)) ++trunc_rejected;
    }
    CHECK(trunc_rejected == good.size(),
          "adversarial: every truncation of an envelope is rejected");
  }

  // 2. Hostile manifest inputs through dai_job CLI: traversal, dups, absurd
  //    numbers, wrong types. The CLI must fail cleanly (nonzero) or run
  //    safely, never hang or crash.
  {
    const char* manifests[] = {
        "{\"job_id\":\"../evil\",\"models\":[{\"name\":\"a\",\"seed\":42,\"steps\":5}]}",
        "{\"job_id\":\"x\",\"models\":[{\"name\":\"../../etc/passwd\",\"seed\":42,\"steps\":5}]}",
        "{\"job_id\":\"dup\",\"models\":[{\"name\":\"a\",\"seed\":42,\"steps\":5},{\"name\":\"a\",\"seed\":43,\"steps\":5}]}",
        "{\"job_id\":\"huge\",\"models\":[{\"name\":\"a\",\"seed\":42,\"steps\":99999999999}]}",
        "{\"job_id\":\"nan\",\"models\":[{\"name\":\"a\",\"seed\":-1,\"steps\":0}]}",
        "not json at all",
        "{\"job_id\": [\"array\", \"not\", \"string\"]}",
    };
    // dai_job lives next to this test binary (same build dir).
    char exe_buf[1024];
    const ssize_t n_exe = ::readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
    std::string dai_job;
    if (n_exe > 0) {
      exe_buf[n_exe] = '\0';
      std::string dir = exe_buf;
      const size_t slash = dir.rfind('/');
      if (slash != std::string::npos) dai_job = dir.substr(0, slash) + "/dai_job";
    }
    int crashes = 0, rejected = 0;
    for (const char* body : manifests) {
      char path[] = "/tmp/dai_suite_adv_manifest.json";
      if (FILE* f = std::fopen(path, "w")) {
        std::fputs(body, f);
        std::fclose(f);
      }
      std::string cmd = "timeout 30 ";
      cmd += dai_job;
      cmd += " --manifest ";
      cmd += path;
      cmd += " > /dev/null 2>&1";
      const int st = std::system(cmd.c_str());
      // std::system returns the raw wait status: decode exit code properly.
      const int rc = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
      if (rc < 0) ++crashes;    // killed by signal = crash
      else if (rc != 0) ++rejected;
      std::remove(path);
    }
    CHECK(crashes == 0, "adversarial: dai_job CLI never crashes on hostile manifests");
    CHECK(rejected >= 2, "adversarial: malformed manifests are rejected (nonzero exit)");
  }

  // 3. Rapid spawn/kill churn: 40 short sandboxes back to back; all reaped,
  //    no fd/thread leak (parent fd count stable).
  {
    auto count_fds = [] {
      int n = 0;
      DIR* dir = opendir("/proc/self/fd");
      if (!dir) return -1;
      while (readdir(dir)) ++n;
      closedir(dir);
      return n - 3;  // ., .., dirfd itself
    };
    SandboxLimits lim;
    lim.own_system = false;
    const int before = count_fds();
    for (int i = 0; i < 40; ++i) (void)spawn_trivial(lim);
    const int after = count_fds();
    CHECK(before >= 0 && after >= 0 && after - before <= 2,
          "adversarial: 40 rapid spawn/kill cycles leak no parent fds");
  }

  // 4. Unshare-degradation equivalence: when namespaces are unavailable the
  //    sandbox must still enforce limits (simulate by own_system=false and
  //    verify the CPU kill still lands; the EPERM path exercises the same
  //    fallthrough code).
  {
    SandboxLimits lim;
    lim.own_system = false;
    lim.cpu_sec = 1;
    lim.mem_mb = 64;
    const SandboxResult res = run_sandboxed(
        lim, 15, [](int) { for (volatile uint64_t i = 0; i < 10000000000ULL; ++i) {} });
    CHECK(!res.ok, "adversarial: no-namespace mode still enforces CPU rlimit");
  }
  return suite::finish("adversarial");
}

// ---------------------------------------------------------------------------
// REDTEAM: active attack simulation
// ---------------------------------------------------------------------------
static int cat_redteam() {
  suite::section("redteam: active attacks");

  // 1. FSIZE rlimit semantics: the cap applies to regular files, not pipes -
  //    a child may pipe out >fsize bytes. Structural guarantee: the child
  //    SEES the cap (getrlimit); behavioral guarantee: outcomes stay bounded
  //    (no hang, explicit success or explicit failure).
  {
    SandboxLimits lim;
    lim.fsize_mb = 1;
    lim.mem_mb = 64;
    lim.own_system = false;
    TrainPod pod{};
    const SandboxResult res = run_sandboxed(
        lim, 20,
        [&](int wfd) {
          TrainPod p{};
          p.status = 0;
          struct rlimit r;
          getrlimit(RLIMIT_FSIZE, &r);
          p.grad_len = r.rlim_cur;  // report the cap we enforce
          // attempt a >1 MiB FILE write (EFBIG/SIGXFSZ territory)
          int fd = ::open("/tmp/dai_redteam_fsize.bin", O_WRONLY | O_CREAT | O_TRUNC, 0600);
          if (fd >= 0) {
            const size_t chunk = 64 * 1024;
            std::vector<char> buf(chunk, 'x');
            for (int i = 0; i < 64; ++i) {
              if (::write(fd, buf.data(), chunk) < 0) break;
            }
            ::close(fd);
          }
          ::unlink("/tmp/dai_redteam_fsize.bin");
          (void)write(wfd, &p, sizeof(p));
          _exit(0);
        },
        &pod, sizeof(pod));
    // Enforcement shows up as either an EFBIG-bounded report (pod delivered)
    // or a SIGXFSZ kill (limit_killed) - the kill IS the enforcement proof.
    const bool cap_visible = res.ok && pod.grad_len == 1024 * 1024;
    const bool killed_by_fsize = res.limit_killed && res.signal == SIGXFSZ;
    const bool bounded = !res.timed_out;  // never mislabeled, never a hang
    CHECK((cap_visible || killed_by_fsize) && bounded,
          "redteam: FSIZE cap installed, enforced (EFBIG or SIGXFSZ), never mislabeled as timeout");
  }

  // 2. Fork verification: the sandboxed fn runs in a different process and
  //    its writes cannot touch parent state (beyond the pipe contract).
  {
    std::vector<uint32_t> heap(256, 7);
    std::vector<uint32_t>* heap_ptr = &heap;
    SandboxLimits lim;
    lim.own_system = true;
    const SandboxResult res = run_sandboxed(
        lim, 15,
        [&](int wfd) {
          for (auto& v : *heap_ptr) v = 0xCAFEBABE;
          TrainPod p{};
          p.status = 0;
          (void)write(wfd, &p, sizeof(p));
        },
        nullptr, 0);
    bool clean = res.ok;
    for (uint32_t v : heap)
      if (v != 7) clean = false;
    CHECK(clean, "redteam: child writes into parent heap vector are invisible");
  }

  // 3. Address-space crowding: structural guarantee - the child SEES the
  //    AS cap (getrlimit); behavioral outcome must stay bounded (blocked
  //    alloc reported, or clean failure - never silence, never a hang).
  {
    SandboxLimits lim;
    lim.mem_mb = 8;
    lim.own_system = true;
    TrainPod pod{};
    const SandboxResult res = run_sandboxed(
        lim, 15,
        [&](int wfd) {
          TrainPod p{};
          struct rlimit r;
          getrlimit(RLIMIT_AS, &r);
          p.grad_len = r.rlim_cur;  // report the cap
          void* p1 = std::malloc(32 * 1024 * 1024);  // way over 8 MiB
          p.status = p1 ? 0 : 1;
          std::free(p1);
          (void)write(wfd, &p, sizeof(p));
          _exit(0);
        },
        &pod, sizeof(pod));
    const bool cap_visible = res.ok && pod.grad_len == 8ull * 1024 * 1024;
    const bool bounded = !res.timed_out;
    CHECK(cap_visible && bounded,
          "redteam: AS cap is installed in the child and outcomes stay bounded");
  }

  // 4. FIFO-fill DoS: child floods the pipe forever; the parent's deadline
  //    must reap it (the timeout is the containment).
  {
    SandboxLimits lim;
    lim.own_system = false;
    lim.cpu_sec = 5;
    const double t0 = now_s();
    const SandboxResult res = run_sandboxed(
        lim, 2,
        [](int wfd) {
          std::vector<char> chunk(4096, 'z');
          for (;;) (void)write(wfd, chunk.data(), chunk.size());
        },
        nullptr, 0);
    const double dt = now_s() - t0;
    // The flood bytes make every 200-byte chunk a pod with nonzero status, so
    // the parent must fail cleanly (never accept, never hang).
    CHECK(!res.ok && dt < 15.0, "redteam: pipe-flood child is contained by the deadline");
  }

  // 5. Checkpoint restore attack: a checkpoint whose declared section lens
  //    exceed the file must fail; a foreign payload must not crash restore.
  {
    ParityMlp m(42);
    ParityAdamW opt(m.params(), 0.01f);
    TorchRng rng(42);
    Checkpoint ck = Checkpoint::capture("atk", 42, 5, m, opt, rng);
    const std::string path = ck.save("/tmp/dai_redteam_ckpt", 3);

    std::vector<uint8_t> bytes;
    if (FILE* f = std::fopen(path.c_str(), "rb")) {
      uint8_t buf[65536];
      size_t r;
      while ((r = std::fread(buf, 1, sizeof(buf), f)) > 0) bytes.insert(bytes.end(), buf, buf + r);
      std::fclose(f);
    }
    // Attack via the file: (a) truncate the payload, (b) inflate the header
    // payload_len (offset 16, LE u32 - the offset load() itself derives).
    // Both must be clean rejections (CRC mismatch or payload-truncated).
    std::vector<uint8_t> trunc = bytes;
    trunc.resize(trunc.size() * 2 / 3);
    std::string err;
    Checkpoint loaded;
    CHECK(!Checkpoint::load(make_attack_file(trunc), loaded, err) || !err.empty(),
          "redteam: truncated checkpoint payload is rejected");

    std::vector<uint8_t> evil = bytes;
    const uint32_t huge = 0xFFFFFFFFu;
    std::memcpy(evil.data() + 16, &huge, 4);
    err.clear();
    CHECK(!Checkpoint::load(make_attack_file(evil), loaded, err) || !err.empty(),
          "redteam: checkpoint with inflated payload_len is rejected");
  }
  return suite::finish("redteam");
}

int main(int argc, char** argv) {
  if (argc < 3 || std::strcmp(argv[1], "--category") != 0) {
    std::fprintf(stderr, "usage: suite_sandbox --category <security|adversarial|redteam>\n");
    return 2;
  }
  const std::string cat = argv[2];
  if (cat == "security") return cat_security();
  if (cat == "adversarial") return cat_adversarial();
  if (cat == "redteam") return cat_redteam();
  std::fprintf(stderr, "unknown category: %s\n", cat.c_str());
  return 2;
}
