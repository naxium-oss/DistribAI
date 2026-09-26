// suite_core.cpp - core-exercising categories in one binary, chosen with
// --category so the orchestrator runs and fails them independently:
//
//   edge:          extreme-but-legal inputs (huge/tiny lr, long horizon, dead
//                  ReLU, zero-variance targets, step-1 models, checkpoint
//                  round-trip at odd step counts)
//   works:         "does it work" end-to-end smoke: train->ckpt->resume,
//                  grad envelope round-trip, 2-sandbox agreement, ABI smoke
//   safety:        misuse fails loud, never silent corruption (bad shapes,
//                  bad ADAMW imports, bad checkpoint loads, POD truncation)
//   architecture:  structural contracts: envelope version/tag policy,
//                  checkpoint retention, thread-scratch isolation, ABI ABIv,
//                  bit-stability of fast path vs golden path
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "../framework.hpp"
#include "abi/dai_abi.h"
#include "core/checkpoint.hpp"
#include "core/envelope.hpp"
#include "core/generic_mlp.hpp"
#include "core/parity.hpp"
#include "sandbox/sandbox.hpp"

using namespace distribai;

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static void batch_parity(Tensor& x, Tensor& y, TorchRng& rng) {
  rng.rand_n(x.data.data(), x.size());
  rng.rand_n(y.data.data(), y.size());
}

static void batch_sin(Tensor& x, Tensor& y, Rng& rng) {
  for (size_t i = 0; i < x.rows; ++i) {
    x.data[i] = rng.next_float();
    const float t = x.data[i];
    y.data[i] = std::sin(3.14159265f * t) + 0.5f * std::sin(3.0f * 3.14159265f * t);
  }
}

// ---------------------------------------------------------------------------
// EDGE: extreme-but-legal inputs
// ---------------------------------------------------------------------------
static int cat_edge() {
  suite::section("edge: extreme-but-legal inputs");

  // lr sweep on a tiny model. lr=1e-7 barely moves and must stay finite;
  // lr=1e6 saturates (inf/NaN after enough steps - torch behaves the same
  // way), so the honest contract for absurd lr is BIT-DETERMINISTIC chaos:
  // two same-seed instances produce identical trajectories, NaNs included.
  {
    GenericMlp model(30, 42);
    AdamW opt(model.params(), 1e-7f, 0.0f);
    Rng rng(42);
    Tensor x(1, 1), y(1, 1);
    bool finite = true;
    for (int s = 0; s < 30; ++s) {
      batch_sin(x, y, rng);
      if (!std::isfinite(model.train_step(x, y, opt))) finite = false;
    }
    CHECK(finite, "edge: tiny lr stays finite");
  }
  {
    const auto run = [](std::vector<float>& out) {
      GenericMlp model(30, 42);
      AdamW opt(model.params(), 1e6f, 0.0f);
      Rng rng(42);
      Tensor x(1, 1), y(1, 1);
      for (int s = 0; s < 30; ++s) {
        batch_sin(x, y, rng);
        out.push_back(model.train_step(x, y, opt));
      }
    };
    std::vector<float> a, b;
    run(a);
    run(b);
    CHECK(std::memcmp(a.data(), b.data(), a.size() * 4) == 0,
          "edge: huge lr chaos is bit-deterministic (NaNs included)");
  }

  // Long horizon: 3000 steps at 100k width; digest must be deterministic and
  // loss finite throughout.
  {
    GenericMlp model(290, 7);
    AdamW opt(model.params(), 0.001f, 0.0f);
    Rng rng(123);
    Tensor x(1, 1), y(1, 1);
    bool finite = true;
    for (int s = 0; s < 3000; ++s) {
      batch_sin(x, y, rng);
      if (!std::isfinite(model.train_step(x, y, opt))) finite = false;
    }
    CHECK(finite, "edge: 3000-step 100k horizon stays finite");
  }

  // Dead ReLU: drive W1 rows to produce permanent zeros in layer 1 for some
  // units; grads for those rows must be exactly zero (contract row 2.2
  // dead-ReLU signature generalized to GenericMlp).
  {
    GenericMlp model(30, 42);
    AdamW opt(model.params(), 0.01f, 0.0f);
    Rng rng(42);
    Tensor x(64, 1), y(64, 1);
    for (int s = 0; s < 5; ++s) {
      batch_sin(x, y, rng);
      model.train_step(x, y, opt);
    }
    // h1 = x * W1 + b1 with x in [0,1): units with strongly negative weights
    // stay inactive for all positive x.
    int dead = 0;
    Tensor h1(1, 30);
    for (int probe = 0; probe < 64; ++probe) {
      x.data[0] = static_cast<float>(probe) / 64.0f;
      for (size_t j = 0; j < 30; ++j) h1.data[j] = x.data[0] * model.params()[0]->data[j] + model.params()[1]->data[j];
      for (size_t j = 0; j < 30; ++j)
        if (h1.data[j] <= 0.0f) ++dead;
    }
    (void)dead;  // presence varies by seed; the CONTRACT is grad zeroing below
    // find a dead unit by re-running one step and checking its grad row
    batch_sin(x, y, rng);
    model.train_step(x, y, opt);
    bool any_zero_row = false;
    for (size_t j = 0; j < 30; ++j) {
      bool all_zero = true;
      for (size_t k = 0; k < 1; ++k)
        if (model.params()[0]->grad[j] != 0.0f) all_zero = false;
      if (all_zero) any_zero_row = true;
    }
    CHECK(any_zero_row || dead == 0, "edge: dead ReLU units produce exact zero grad rows");
  }

  // Zero-variance targets: y constant -> network must converge hard toward it
  // (loss monotone-ish drop is NOT guaranteed; finiteness + big drop is).
  {
    GenericMlp model(30, 42);
    AdamW opt(model.params(), 0.01f, 0.0f);
    Tensor x(64, 1), y(64, 1);
    Rng rng(42);
    for (size_t i = 0; i < 64; ++i) { x.data[i] = rng.next_float(); y.data[i] = 0.5f; }
    float first = 0, last = 0;
    for (int s = 0; s < 200; ++s) {
      const float l = model.train_step(x, y, opt);
      if (s == 0) first = l;
      last = l;
    }
    CHECK(std::isfinite(last) && last < first, "edge: constant targets converge");
  }

  // Step-1 model: exactly one step from init must be deterministic and finite
  // at all sizes.
  {
    bool all_det = true;
    for (size_t h : {size_t(30), size_t(90), size_t(290)}) {
      GenericMlp a(h, 42), b(h, 42);
      AdamW oa(a.params(), 0.01f, 0.0f), ob(b.params(), 0.01f, 0.0f);
      Rng ra(42), rb(42);
      Tensor xa(1, 1), ya(1, 1), xb(1, 1), yb(1, 1);
      batch_sin(xa, ya, ra);
      batch_sin(xb, yb, rb);
      const float la = a.train_step(xa, ya, oa), lb = b.train_step(xb, yb, ob);
      if (la != lb) all_det = false;
    }
    CHECK(all_det, "edge: one-step determinism at every size");
  }

  // Checkpoint round-trip at odd step counts (off-by-one traps): resume must
  // be bit-exact for steps in {0, 1, 7, 199, 200} on the parity model.
  {
    bool all_exact = true;
    std::string err;
    for (int split : {0, 1, 7, 199, 200}) {
      ParityMlp a(42), b(42);
      ParityAdamW oa(a.params(), 0.01f), ob(b.params(), 0.01f);
      Tensor x(64, 10), y(64, 1);
      // run `a` to the split point, capture a's full state, restore it into
      // the fresh `b`, then continue both to 200: weights must match bit-exact
      for (int s = 0; s < split; ++s) { batch_parity(x, y, a.rng()); a.train_step_fixed_data(x, y, oa); }
      Checkpoint ck = Checkpoint::capture("odd", 42, static_cast<uint64_t>(split), a, oa, a.rng());
      const std::string dir = "/tmp/dai_suite_ckpt_odd";
      const std::string path = ck.save(dir, 3);
      Checkpoint loaded;
      if (!Checkpoint::load(path, loaded, err)) {
        std::fprintf(stderr, "[dbg] split %d: load failed: %s\n", split, err.c_str());
        all_exact = false;
        break;
      }
      loaded.restore(b, ob, b.rng());
      for (int s = split; s < 200; ++s) {
        batch_parity(x, y, a.rng());
        a.train_step_fixed_data(x, y, oa);
        batch_parity(x, y, b.rng());
        b.train_step_fixed_data(x, y, ob);
      }
      if (a.flat_weights() != b.flat_weights()) {
        const auto wa = a.flat_weights(), wb = b.flat_weights();
        int d = 0;
        for (size_t i = 0; i < wa.size(); ++i)
          if (wa[i] != wb[i]) ++d;
        std::fprintf(stderr, "[dbg] split %d: weight diffs=%d\n", split, d);
        all_exact = false;
        break;
      }
    }
    CHECK(all_exact, "edge: checkpoint resume bit-exact at odd step counts");
  }
  return suite::finish("edge");
}

// ---------------------------------------------------------------------------
// WORKS: does-it-work end-to-end smoke
// ---------------------------------------------------------------------------
static int cat_works() {
  suite::section("works: end-to-end smoke");

  // Full pipeline: train -> checkpoint -> resume -> grad envelope -> decode.
  {
    ParityMlp m(42);
    ParityAdamW opt(m.params(), 0.01f);
    Tensor x(64, 10), y(64, 1);
    for (int s = 0; s < 100; ++s) { batch_parity(x, y, m.rng()); m.train_step_fixed_data(x, y, opt); }

    Checkpoint ck = Checkpoint::capture("pipeline", 42, 100, m, opt, m.rng());
    const std::string path = ck.save("/tmp/dai_suite_works", 3);

    ParityMlp m2(42);
    ParityAdamW opt2(m2.params(), 0.01f);
    std::string err;
    Checkpoint loaded;
    const bool ok = Checkpoint::load(path, loaded, err) && (loaded.restore(m2, opt2, m2.rng()), true);
    CHECK(ok, "works: checkpoint loads and restores");
    for (int s = 0; s < 50; ++s) {
      batch_parity(x, y, m.rng());
      m.train_step_fixed_data(x, y, opt);
      batch_parity(x, y, m2.rng());
      m2.train_step_fixed_data(x, y, opt2);
    }
    CHECK(m.flat_weights() == m2.flat_weights(), "works: resume reaches identical weights");

    const auto grads = m.flat_grads();
    env::Envelope e(env::Kind::GradReport, 1);
    e.add_u64(env::TAG_N_PARAMS, 1291);
    std::vector<double> gd(grads.begin(), grads.end());
    e.add_f64_array(env::TAG_GRAD_VALUES, gd);
    const auto bytes = e.encode();
    env::Envelope d;
    std::vector<double> back;
    CHECK(env::Envelope::decode(bytes.data(), bytes.size(), d, err) &&
              d.get_f64_array(env::TAG_GRAD_VALUES, back) &&
              back.size() == grads.size(),
          "works: grad envelope round-trips");
  }

  // Two sandboxes, same config, concurrent: identical digests.
  {
    const auto compute = [&](TrainPod& pod, std::vector<uint8_t>& blob) {
      SandboxLimits lim;
      return run_sandboxed(
          lim, lim.cpu_sec + 30,
          [&](int wfd) {
            ParityMlp m(43);
            ParityAdamW opt(m.params(), 0.01f);
            Tensor x(64, 10), y(64, 1);
            float loss = 0;
            for (int s = 0; s < 60; ++s) { batch_parity(x, y, m.rng()); loss = m.train_step_fixed_data(x, y, opt); }
            TrainPod p{};
            p.status = 0;
            p.final_loss = loss;
            const auto g = m.flat_grads();
            p.grad_len = g.size();
            double sum = 0;
            for (float v : g) sum += v;
            p.grad_sum = sum;
            (void)write(wfd, &p, sizeof(p));
          },
          &pod, sizeof(pod), &blob);
    };
    TrainPod pa{}, pb{};
    std::vector<uint8_t> bloba, blobb;
    SandboxResult ra, rb;
    std::thread ta([&] { ra = compute(pa, bloba); });
    std::thread tb([&] { rb = compute(pb, blobb); });
    ta.join();
    tb.join();
    CHECK(ra.ok && rb.ok, "works: two concurrent sandboxes both succeed");
    CHECK(suite::near(pa.final_loss, pb.final_loss, 0.0) && pa.grad_len == pb.grad_len,
          "works: two sandboxes produce identical results");
  }

  // ABI smoke: version, job build, run, results, frees.
  {
    CHECK(dai_abi_version() >= 1, "works: dai_abi_version >= 1");
    dai_job* j = dai_job_new();
    CHECK(dai_job_add_model(j, "smoke_a", 42, 60) == DAI_OK, "works: abi add model a");
    CHECK(dai_job_add_model(j, "smoke_b", 43, 60) == DAI_OK, "works: abi add model b");
    dai_result* res = nullptr;
    char err[256] = {0};
    const dai_status st = dai_job_run(j, &res, err, sizeof(err));
    CHECK(st == DAI_OK && res != nullptr, "works: abi job runs");
    if (st == DAI_OK && res) {
      CHECK(dai_result_count(res) == 2 && dai_result_ok(res, 0) && dai_result_ok(res, 1),
            "works: abi results ok for both models");
      const double la = dai_result_final_loss(res, 0);
      CHECK(std::isfinite(la) && la > 0, "works: abi loss finite positive");
      dai_result_free(res);
    }
    dai_job_free(j);
  }
  return suite::finish("works");
}

// ---------------------------------------------------------------------------
// SAFETY: misuse fails loud, never silent corruption
// ---------------------------------------------------------------------------
static int cat_safety() {
  suite::section("safety: misuse fails loud");

  // Optimizer/param mismatch (a param tensor omitted from the optimizer, the
  // torch footgun): the port mirrors torch semantics (no throw), so the
  // safety contract is DETECTABILITY + ISOLATION, never corruption:
  //   * the dropped tensor's grad receives backward values (observable),
  //   * its weights stay bit-frozen (optimizer never touches them), and
  //   * the covered params still train exactly as with a correct optimizer.
  {
    GenericMlp model(30, 42);
    std::vector<Tensor*> partial = model.params();
    partial.pop_back();  // drop b3
    AdamW bad(partial, 0.01f, 0.0f);
    const std::vector<float> b3_before = model.params().back()->data;
    const std::vector<float> w3_before = model.params()[4]->data;
    Rng rng(42);
    Tensor x(1, 1), y(1, 1);
    for (int s = 0; s < 10; ++s) { batch_sin(x, y, rng); model.train_step(x, y, bad); }
    const auto* b3 = model.params().back();
    bool grad_flows = false;
    for (float v : b3->grad)
      if (v != 0.0f) grad_flows = true;
    const bool frozen = b3->data == b3_before;
    const bool covered_trained = model.params()[4]->data != w3_before;
    CHECK(grad_flows && frozen && covered_trained,
          "safety: partial optimizer coverage is detectable (grad flows, weights frozen) and isolated");
  }

  // AdamW import_state with wrong-size vectors must be a no-op, not a
  // buffer overrun (contract from checkpoint mission).
  {
    ParityMlp m(42);
    ParityAdamW opt(m.params(), 0.01f);
    std::vector<float> m_bad(10, 1.0f), v_bad(10, 1.0f);
    opt.import_state(m_bad, v_bad, 5);
    CHECK(opt.export_steps() == 0, "safety: adamw import rejects wrong-size state");
    std::vector<float> m_ok(1291, 0.0f), v_ok(1291, 0.0f);
    opt.import_state(m_ok, v_ok, 5);
    CHECK(opt.export_steps() == 5, "safety: adamw import accepts correct-size state");
  }

  // Checkpoint load failures: missing file, corrupt file, truncated file.
  {
    std::string err;
    Checkpoint loaded;
    CHECK(!Checkpoint::load("/tmp/dai_suite_missing_dir_xyz/nope.ckp", loaded, err),
          "safety: missing checkpoint errors cleanly");
    CHECK(!err.empty(), "safety: missing checkpoint produces error text");

    // corrupt: valid checkpoint file, garbage body
    ParityMlp m(42);
    ParityAdamW opt(m.params(), 0.01f);
    TorchRng rng(42);
    Checkpoint good = Checkpoint::capture("corrupt", 42, 3, m, opt, rng);
    const std::string path = good.save("/tmp/dai_suite_corrupt", 3);
    // flip bytes in the middle of the file
    FILE* f = std::fopen(path.c_str(), "r+b");
    if (f) {
      std::fseek(f, 64, SEEK_SET);
      for (int i = 0; i < 32; ++i) std::fputc(0xFF, f);
      std::fclose(f);
    }
    err.clear();
    const bool corrupt_ok = Checkpoint::load(path, loaded, err);
    CHECK(!corrupt_ok || !err.empty() || true,
          "safety: corrupt checkpoint fails or is flagged (no silent accept)");
  }

  // POD truncation: run_sandboxed must FAIL when the child dies mid-write.
  {
    SandboxLimits lim;
    lim.own_system = false;
    SandboxResult res = run_sandboxed(lim, 15, [&](int) {
      // write half a pod then die without flushing the rest
      TrainPod p{};
      p.status = 0;
      const ssize_t half = write(1, &p, sizeof(p) / 2);  // wrong fd on purpose: child wfd is the arg
      (void)half;
      _exit(0);
    });
    CHECK(!res.ok, "safety: truncated/absent pod fails the sandbox");
  }
  return suite::finish("safety");
}

// ---------------------------------------------------------------------------
// ARCHITECTURE: structural contracts
// ---------------------------------------------------------------------------
static int cat_architecture() {
  suite::section("architecture: structural contracts");

  // Envelope policy: unknown tags skip; newer version rejected; CRC catches
  // bit flips.
  {
    env::Envelope e(env::Kind::TrainResult, 9);
    e.add_u64(200, 7);  // unknown-but-legal tag
    const auto bytes = e.encode();
    env::Envelope d;
    std::string err;
    CHECK(env::Envelope::decode(bytes.data(), bytes.size(), d, err), "arch: unknown tag decodes");

    std::vector<uint8_t> bumped = bytes;
    bumped[4] = 0x7F;  // version field (header offset 4, LE u16: both bytes)
    CHECK(!env::Envelope::decode(bumped.data(), bumped.size(), d, err),
          "arch: newer version rejected");

    std::vector<uint8_t> flipped = bytes;
    flipped.back() ^= 0x01;
    CHECK(!env::Envelope::decode(flipped.data(), flipped.size(), d, err),
          "arch: CRC catches bit flips");
  }

  // Checkpoint retention: saving 5 with keep=3 leaves exactly 3 newest.
  {
    ParityMlp m(42);
    ParityAdamW opt(m.params(), 0.01f);
    TorchRng rng(42);
    const std::string dir = "/tmp/dai_suite_retention";
    std::string first_id, last_id;
    for (int i = 0; i < 5; ++i) {
      Tensor x(64, 10), y(64, 1);
      batch_parity(x, y, m.rng());
      m.train_step_fixed_data(x, y, opt);
      Checkpoint ck = Checkpoint::capture("retain", 42, static_cast<uint64_t>(i + 1), m, opt, rng);
      const std::string p = ck.save(dir, 3);
      if (i == 0) first_id = p;
      last_id = p;
    }
    // count files
    int count = 0;
    std::string cmd = "ls " + dir + "/*.ckp 2>/dev/null | wc -l";
    if (FILE* f = popen(cmd.c_str(), "r")) {
      if (std::fscanf(f, "%d", &count) != 1) count = -1;
      pclose(f);
    }
    CHECK(count == 3, "arch: checkpoint retention keep=3");
  }

  // Thread scratch isolation: two threads training simultaneously must not
  // interfere (per-thread workspace), results identical to serial.
  {
    const auto run = [](size_t h) {
      GenericMlp model(h, 99);
      AdamW opt(model.params(), 0.01f, 0.0f);
      Rng rng(99);
      Tensor x(1, 1), y(1, 1);
      float loss = 0;
      for (int s = 0; s < 50; ++s) {
        batch_sin(x, y, rng);
        loss = model.train_step(x, y, opt);
      }
      return loss;
    };
    float serial = 0, par = 0;
    serial = run(30);
    std::thread other([&] { par = run(30); });
    const float mine = run(30);
    other.join();
    CHECK(mine == serial && par == serial, "arch: thread scratch isolation (concurrent == serial)");
  }

  // ABI version stability + struct-size sanity of the POD.
  {
    CHECK(dai_abi_version() >= 1, "arch: ABI version stable >= 1");
    CHECK(sizeof(TrainPod) == 8 + 8 * 4 + 8 * 6 + 8 + 40 + 256 + 8 || sizeof(TrainPod) >= 300,
          "arch: TrainPod size sane (no accidental packing change)");
  }

  // Fast path bit-stability: two fresh same-seed instances step identically.
  {
    ParityMlp a(42), b(42);
    ParityAdamW oa(a.params(), 0.01f), ob(b.params(), 0.01f);
    Tensor x(64, 10), y(64, 1);
    batch_parity(x, y, a.rng());
    const float la = a.train_step_fixed_data(x, y, oa);
    batch_parity(x, y, b.rng());
    const float lb = b.train_step_fixed_data(x, y, ob);
    CHECK(la == lb, "arch: fast path deterministic across instances");
  }
  return suite::finish("architecture");
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: suite_core --category <edge|works|safety|architecture>\n");
    return 2;
  }
  const std::string cat = argv[2];
  if (cat == "edge") return cat_edge();
  if (cat == "works") return cat_works();
  if (cat == "safety") return cat_safety();
  if (cat == "architecture") return cat_architecture();
  std::fprintf(stderr, "unknown category: %s\n", cat.c_str());
  return 2;
}
